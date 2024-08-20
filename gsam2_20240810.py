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

def iou(box1, box2):
    # Calculate intersection over union of two boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    return intersection / (box1_area + box2_area - intersection)

"""
Step 1: Environment settings and model initialization
"""
# use float16 for better memory efficiency
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Enable gradient checkpointing for more memory efficiency
#torch.utils.checkpoint.use_checkpoint(lambda f, *args, **kwargs: f(*args, **kwargs))

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# Use the tiny model for Grounding DINO
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Enable gradient checkpointing for the grounding model
grounding_model.gradient_checkpointing_enable()

# setup the input image and text prompt for SAM 2 and Grounding DINO
text = "person."

# `video_path` is the path to the input MP4 video file
video_path = "notebooks/videos/5FPS/fps_1.mp4"
video_dir = "notebooks/videos/frames"

# Extract frames from the video
extract_frames(video_path, video_dir)
print(f"Extracted frames from video: {video_path}")
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for each frame
"""

video_segments = {}  # video_segments contains the per-frame segmentation results
object_tracker = {}  # Track objects across frames

MAX_IMAGE_SIZE = 480  # Further reduced maximum dimension of the image

# Process frames in smaller batches
BATCH_SIZE = 2

for batch_start in range(0, len(frame_names), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(frame_names))
    batch_frames = frame_names[batch_start:batch_end]
    
    for ann_frame_idx, frame_name in enumerate(batch_frames, start=batch_start):
        if ann_frame_idx % 10 == 0:
            img_path = os.path.join(video_dir, frame_name)
            image = Image.open(img_path)
            
            # Resize image if it's too large
            if max(image.size) > MAX_IMAGE_SIZE:
                image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

            # run Grounding DINO on the image
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            
            # Use torch.cuda.amp.autocast() for mixed precision inference
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            # prompt SAM image predictor to get the mask for the object
            image_predictor.set_image(np.array(image.convert("RGB")))

            # process the detection results
            input_boxes = results[0]["boxes"].cpu().numpy()
            OBJECTS = results[0]["labels"]

            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            # convert the mask shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            """
            Step 3: Register each object's positive points to video predictor with separate add_new_points call
            """

            # sample the positive points from mask for each objects
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                
                # Check if this object overlaps with any existing tracked object
                new_box = input_boxes[object_id - 1]
                matched_id = None
                for tracked_id, tracked_box in object_tracker.items():
                    if iou(new_box, tracked_box) > 0.5:  # IOU threshold
                        matched_id = tracked_id
                        break
                
                if matched_id is not None:
                    # Update existing object
                    object_id = matched_id
                else:
                    # New object
                    object_id = max(object_tracker.keys(), default=0) + 1
                
                object_tracker[object_id] = new_box

                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )

            # Clear CUDA cache to avoid out of memory
            del inputs, outputs, results, masks, scores, logits
            torch.cuda.empty_cache()

    # Propagate for the batch
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

"""
Step 4: Visualize the segment results across the video and save them
"""

save_dir = "./tracking_results"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ID_TO_OBJECTS = {i: f"Person_{i}" for i in object_tracker.keys()}

def process_mask(mask):
    if mask.ndim == 3:
        mask = mask.squeeze()  # Remove singleton dimensions
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
    
    labels = [ID_TO_OBJECTS[i] for i in valid_object_ids]
    
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    # Clear CUDA cache to avoid out of memory
    del detections, annotated_frame, masks
    torch.cuda.empty_cache()

"""
Step 5: Convert the annotated frames to video
"""

output_video_path = "./fps_1_20240807.mp4"
create_video_from_images(save_dir, output_video_path)

print("Video processing completed. Output saved to:", output_video_path)