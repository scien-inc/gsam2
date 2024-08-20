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

"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "person."

# `video_path` is the path to the input MP4 video file
video_path = "notebooks/videos/5FPS/fps_1.mp4"
video_dir = "notebooks/videos/frames"

# Extract frames from the video
extract_frames(video_path, video_dir)
print(f"Extracted frames from video: {video_path}")
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)

ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
object_id_mapping = {}  # Mapping from frame index to detected object IDs


"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for each frame
"""

video_segments = {}  # video_segments contains the per-frame segmentation results

for ann_frame_idx in range(len(frame_names)):
    if ann_frame_idx % 10 == 0:
        # prompt grounding dino to get the box coordinates on specific frame
        img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
        image = Image.open(img_path)

        # run Grounding DINO on the image
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
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
        Step 3: Register each object's positive points to video predictor with seperate add_new_points call
        """

        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for obj_idx, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            obj_id = obj_idx + (ann_obj_id * 1000)  # Create a unique object ID based on the frame index
            object_id_mapping[(ann_frame_idx, obj_idx)] = obj_id
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        ann_obj_id += 1

        # Clear CUDA cache to avoid out of memory
        del inputs, outputs, results, masks, scores, logits
        torch.cuda.empty_cache()

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
batch_size = 10  # Adjust this based on your GPU memory capacity
total_frames = len(frame_names)

for start_idx in range(0, total_frames, batch_size):
    end_idx = min(start_idx + batch_size, total_frames)
    for out_frame_idx in range(start_idx, end_idx):
        current_out, pred_masks = video_predictor.propagate_in_video(inference_state)
        video_segments[out_frame_idx] = {
            out_obj_id: (pred_masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(current_out['out_obj_ids'])
        }

        # Clear CUDA cache to avoid out of memory
        del current_out, pred_masks
        torch.cuda.empty_cache()

"""
Step 5: Visualize the segment results across the video and save them
"""

save_dir = "./tracking_results"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Modify this line to include a default value for unknown object IDs
ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
DEFAULT_OBJECT_LABEL = "Unknown"

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
    
    # Modify this line to use .get() method with a default value
    labels = [ID_TO_OBJECTS.get(i, DEFAULT_OBJECT_LABEL) for i in valid_object_ids]
    
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    # Clear CUDA cache to avoid out of memory
    del detections, annotated_frame, masks
    torch.cuda.empty_cache()

"""
Step 6: Convert the annotated frames to video
"""

output_video_path = "./fps_1_20240807.mp4"
create_video_from_images(save_dir, output_video_path)
