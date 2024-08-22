import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import shutil
import torch.nn.functional as F



# Set the GPU device number (e.g., device_id = 0 for GPU 0)
device_id = 1
torch.cuda.set_device(device_id)

# delete ./outputs directory if it exists
if os.path.exists("./outputs"):
    shutil.rmtree("./outputs")

"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# setup the input text prompt for SAM 2 and Grounding DINO
text = "person."

# 'output_dir' is the directory to save the annotated frames
output_dir = "./outputs"
# create the output directory
CommonUtils.creat_dirs(output_dir)
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default device ID for webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize video writer to save output (optional)
output_video_path = "./outputs/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 15
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask"
objects_count = 0
step = 10  # step to process every nth frame

# Directory to store JPEG frames
jpeg_frame_dir = "./jpeg_frames"
if not os.path.exists(jpeg_frame_dir):
    os.makedirs(jpeg_frame_dir)

# Capture the first frame and save as JPEG to initialize inference state
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    exit()

# Save the first frame as JPEG
first_frame_path = os.path.join(jpeg_frame_dir, "00000.jpg")
cv2.imwrite(first_frame_path, frame)

# Initialize inference state with the saved JPEG frame
inference_state = video_predictor.init_state(video_path=jpeg_frame_dir)

frame_count = 0  # To keep track of frame indices

# Loop through webcam frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Save the frame as JPEG
    frame_path = os.path.join(jpeg_frame_dir, f"{frame_count:05d}.jpg")
    cv2.imwrite(frame_path, frame)
    
    # Convert the frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    # Prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    input_boxes = results[0]["boxes"]
    OBJECTS = results[0]["labels"]

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 2:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO)

    if mask_dict.promote_type == "mask":
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
    else:
        raise NotImplementedError("SAM 2 video predictor only supports mask prompts")

    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
    video_predictor.reset_state(inference_state)
    if len(mask_dict.labels) == 0:
        print("No object detected in the frame, skipping.")
        continue

    for object_id, object_info in mask_dict.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_count,  # Process the current frame
            obj_id=object_id,
            mask=object_info.mask,
        )
    
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=frame_count):
        frame_masks = MaskDictionaryModel()
        
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0)
            object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id))
            object_info.update_box()
            frame_masks.labels[out_obj_id] = object_info

        video_segments[out_frame_idx] = frame_masks
        sam2_masks = copy.deepcopy(frame_masks)

    # Save the tracking masks and json files
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            # Resize the mask to match the frame size before applying it
            resized_mask = F.interpolate(
                obj_info.mask.unsqueeze(0).unsqueeze(0).float(),
                size=(mask_img.shape[0], mask_img.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0).bool()

            mask_img[resized_mask] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        mask_name = f"mask_{frame_idx:05d}.npy"
        np.save(os.path.join(mask_data_dir, mask_name), mask_img)

        json_data = frame_masks_info.to_dict()
        json_data_path = os.path.join(json_data_dir, mask_name.replace(".npy", ".json"))
        with open(json_data_path, "w") as f:
            json.dump(json_data, f)


    # Display the resulting frame with masks (optional)
    cv2.imshow('Webcam', frame)

    # Write the frame to the output video
    out.write(frame)

    # Increment the frame count
    frame_count += 1

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

# Optionally, draw the final results and save the video
CommonUtils.draw_masks_and_box_with_supervision(output_dir, mask_data_dir, json_data_dir, result_dir)
