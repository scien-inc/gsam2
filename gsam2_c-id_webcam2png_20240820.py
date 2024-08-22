import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from PIL import Image
import collections
import argparse

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from sam2.build_sam import build_sam2_camera_predictor


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)


def load_model():

    # init grounding dino model from huggingface
    # model_id = "IDEA-Research/grounding-dino-tiny"
    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    grounding_processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # build sam2
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)

    return grounding_processor, grounding_model, predictor


def main():

    # load model
    grounding_processor, grounding_model, predictor = load_model()
    
    # load video
    # cap = cv2.VideoCapture(0) # camera
    cap = cv2.VideoCapture("notebooks/videos/case.mp4")
    
    # init
    query = "person."
    frame_list = [] # for visualization
    results = None
    if_init = False
    
    idx = 0
    # fps_cut = 2 # skip every fps_cut step to save time
    while True:
        idx += 1    
        print(idx)
        ret, frame = cap.read()
        if not ret:
            break
        # if idx % fps_cut == 0: continue

        if query and not if_init:
            width, height = frame.shape[:2][::-1]    
            predictor.load_first_frame(frame)

            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
            
            # box from groundingDINO
            inputs = grounding_processor(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), text=query, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)
            results = grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.6,
                text_threshold=0.6,
                target_sizes=[frame.shape[:2]]
            )
            
            # single box
            boxes = results[0]["boxes"]
            if boxes.shape[0] != 0:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=boxes,
                )
                if_init = True
            else:
                if_init = False
        
        elif if_init:
            out_obj_ids, out_mask_logits = predictor.track(frame)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                ) * 255

                all_mask = cv2.bitwise_or(all_mask, out_mask)

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

        frame_list.append(frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    # visualization 
    frame_list = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frame_list]
    # save as mp4
    imageio.mimsave("./result.mp4", frame_list, "MP4")
    # imageio.mimsave("./result.gif", frame_list, "GIF")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the system with a specified model.')
    args = parser.parse_args()
    
    main()
