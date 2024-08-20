import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from track_utils import sample_points_from_masks
from video_utils import create_video_from_images
from ultralytics import YOLO

def extract_frames(video_path, output_dir, target_fps=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    vidcap = cv2.VideoCapture(video_path)
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps // target_fps)
    
    success, image = vidcap.read()
    count = 0
    saved_count = 0
    
    while success:
        if count % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/{saved_count:05d}.jpg", image)
            saved_count += 1
        success, image = vidcap.read()
        count += 1
    
    vidcap.release()
    return output_dir

def initialize_models():
    """
    Initialize models and return them.
    """
    # Load models only when needed
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    # Grounding DINO model
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # YOLOv8 model
    yolo_model = YOLO("yolov8n.pt")
    
    return video_predictor, image_predictor, processor, grounding_model, yolo_model

def clear_gpu_memory(*models):
    """
    Move models to CPU and clear CUDA cache to free up memory.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            model.to('cpu')
    torch.cuda.empty_cache()

"""
Step 1: Environment settings and model initialization
"""
# Use Mixed Precision to reduce memory usage
scaler = torch.cuda.amp.GradScaler()

# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# `video_path` is the path to the input MP4 video file
video_path = "notebooks/videos/5FPS/fps_1.mp4"
video_dir = "notebooks/videos/frames"

# Extract frames from the video
extract_frames(video_path, video_dir)
print(f"Extracted frames from video: {video_path}")

# Scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

"""
Step 2: Initialize object tracking and detection
"""

tracked_objects = {}  # 保存するオブジェクトIDとその座標

"""
Step 3: Process each frame with memory-efficient techniques
"""
# Initialize models
video_predictor, image_predictor, processor, grounding_model, yolo_model = initialize_models()

# Initialize inference state
inference_state = video_predictor.init_state(video_path=video_dir)

for ann_frame_idx, frame_name in enumerate(frame_names):
    img_path = os.path.join(video_dir, frame_name)
    image = Image.open(img_path)

    # Resize the image to reduce memory usage
    image = image.resize((image.width // 4, image.height // 4))  # 解像度をさらに下げる

    # YOLOv8で人物を検出
    results = yolo_model(img_path)
    detected_persons = [
        result for result in results[0].boxes.data.cpu().numpy()
        if result[5] == 0  # personクラスID
    ]
    
    new_persons = []

    for person in detected_persons:
        box = person[:4]  # x1, y1, x2, y2
        iou_scores = []
        for obj_id, obj_box in tracked_objects.items():
            iou = calculate_iou(box, obj_box)
            iou_scores.append((iou, obj_id))
        
        if not iou_scores or max(iou_scores)[0] < 0.5:
            new_persons.append(box)
        else:
            best_match = max(iou_scores, key=lambda x: x[0])
            tracked_objects[best_match[1]] = box

    if new_persons:
        # Move Grounding DINO and SAM models to GPU
        grounding_model.to('cuda')
        image_predictor.model.to('cuda')  # ここでimage_predictorのモデルをGPUに移動
        
        # 新しい人物について、Grounding DINOとSAMを使用してIDを割り当てる
        inputs = processor(images=image, text="person.", return_tensors="pt").to("cuda")

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                dino_outputs = grounding_model(**inputs)

        dino_results = processor.post_process_grounded_object_detection(
            dino_outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        # SAMによるマスクの取得
        image_predictor.set_image(np.array(image.convert("RGB")))
        input_boxes = dino_results[0]["boxes"].cpu().numpy()
        masks, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, (label, points) in enumerate(zip(dino_results[0]["labels"], all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, _ = video_predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
            tracked_objects[object_id] = input_boxes[0]  # トラッキング対象として保存

    # Clear GPU memory after each frame processing
    clear_gpu_memory(grounding_model, image_predictor.model)

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
# Reload models only when needed for propagation
video_predictor.to('cuda')

video_segments = {}

for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Clear GPU memory after propagation
clear_gpu_memory(video_predictor)

"""
Step 5: Visualize the segment results across the video and save them
"""

save_dir = "./tracking_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def process_mask(mask):
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.ndim == 1:
        side_length = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(side_length, side_length)
    return mask

def get_xyxy_from_mask(mask):
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None
    return np.array([cols.min(), rows.min(), cols.max(), rows.max()])

for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
    
    object_ids = list(segments.keys())
    masks = [process_mask(segments[obj_id]) for obj_id in object_ids]
    
    xyxy_boxes = []
    valid_masks = []
    valid_object_ids = []
    
    for obj_id, mask in zip(object_ids, masks):
        xyxy = get_xyxy_from_mask(mask)
        if xyxy is not None:
            xyxy_boxes.append(xyxy)
            valid_masks.append(mask)
            valid_object_ids.append(obj_id)
    
    if not xyxy_boxes:
        print(f"No valid boxes for frame {frame_idx}, skipping...")
        continue
    
    xyxy_boxes = np.array(xyxy_boxes)
    valid_masks = np.array(valid_masks)

    detections = sv.Detections(
        xyxy=xyxy_boxes,
        mask=valid_masks,
        class_id=np.array(valid_object_ids, dtype=np.int32),
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    
    labels = [ID_TO_OBJECTS.get(i, "Unknown") for i in valid_object_ids]
    
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    # Clear CUDA cache after processing each frame
    torch.cuda.empty_cache()

"""
Step 6: Convert the annotated frames to video
"""

output_video_path = "./fps_1_20240807.mp4"
create_video_from_images(save_dir, output_video_path)
