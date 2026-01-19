import cv2
import numpy as np
import os
import sys
import traceback
import math

from mmpose.apis import MMPoseInferencer
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData

# 1. MMDetection now handles the Tracker
from mmdet.apis import init_detector, inference_mot
from mmdet.registry import VISUALIZERS

# 2. MMPose handles the Skeleton
from mmpose.apis import init_model as init_pose_model
from mmpose.apis import inference_topdown
from mmpose.utils import register_all_modules
import torch

# --------------------config-------------------- #

# IO
INPUT_VIDEO = './data/clip_1_shortened.mov'
OUTPUT_VIDEO = './data/output_26_3.mp4'
OUTPUT_CODEC = 'mp4v' # Codec for output video

# Models
#DET_MODEL = 'rtmdet-l' # Detection model
#POSE_MODEL = 'rtmpose-l_8xb512-700e_body8-halpe26-384x288' # Pose model
POSE_CONFIG = './configs/rtmpose_l_26.py'
POSE_CHECKPOINT = './weights/rtmpose_l_26.pth'
TRACK_CONFIG = './configs/ocsort.py'
DET_CHECKPOINT = './weights/rtmdet_l.pth'

TRACK_THRESH = 0.5

# --------------------end config-------------------- #

def setupInferencer():
    inferencer_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device='cuda:0')
    return inferencer_model

def setupTracker():
    tracker = init_detector(TRACK_CONFIG, DET_CHECKPOINT, device='cuda:0')
    return tracker

def openVideo():
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] File not found at: {INPUT_VIDEO}")
        sys.exit(1)

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print(f"[ERROR] File exists, but OpenCV cannot open it: {INPUT_VIDEO}")
        print("Possible causes: Wrong codec, corrupt file, or missing ffmpeg dependencies.")
        sys.exit(1)

    return cap

def main():
    # ---- Initialization ---- #
    print("Initializing...")

    register_all_modules()
    
    # Setup models
    inferencer_model = setupInferencer()
    tracker_model = setupTracker()

    # Video Setup
    cap = openVideo()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # ---- Pose Generation with Tracking and Smoothing ---- #
    print("Starting Pose Inference with Tracking and Smoothing...")

    frame_idx = 0
    while cap.isOpened():

        # Read frame
        ret, frame = cap.read()
        if not ret: break


        # Tracking
        track_results = inference_mot(tracker_model, frame, frame_id=frame_idx)
        pred_instances = track_results.pred_track_instances
        track_mask = pred_instances.scores > TRACK_THRESH # filter low confidence tracks

        if track_mask.any():
            track_ids = pred_instances.track_ids[track_mask]
            track_bboxes = pred_instances.bboxes[track_mask]

            pose_results = inference_topdown(inferencer_model, frame, track_bboxes)


    

        # ---- Visualization and Output ---- #
        pose_data_sample = PoseDataSample()
        pred_instances = InstanceData()
        
        num_kpts = NUM_KPTS # Default number of keypoints (adjust based on model)

        if len(final_people) > 0:
            # We must convert our list of dicts back into numpy arrays
            # MMPose expects shape (N, kpts, 2) for keypoints
            num_kpts = len(final_people[0]['keypoints'])
            pred_instances.keypoints = np.array([p['keypoints'] for p in final_people])
            
            # Extract scores (usually hidden inside keypoint_scores or bbox_score)
            # If keypoint_scores exists, use it, otherwise create default ones
            if 'keypoint_scores' in final_people[0]:
                pred_instances.keypoint_scores = np.array([p['keypoint_scores'] for p in final_people])
            else:
                # Fallback
                pred_instances.keypoint_scores = np.ones((len(final_people), num_kpts))

            # Bounding boxes (N, 4)
            # Note: Your bbox might be nested like [[x,y,x,y]], so we take [0]
            pred_instances.bboxes = np.array([p['bbox'][0] for p in final_people])
            
            # The Critical Part: Passing the Track ID so colors stay consistent
            pred_instances.instances_id = np.array([p['track_id'] for p in final_people])
            
        else:
            # If nobody is found, create empty arrays to prevent crashing
            pred_instances.keypoints = np.empty((0, num_kpts, 2))
            pred_instances.bboxes = np.empty((0, 4))
            pred_instances.keypoint_scores = np.empty((0, num_kpts))
            pred_instances.instances_id = np.empty((0))

        # 3. Store the instances in the data sample
        pose_data_sample.pred_instances = pred_instances

        # Draw data
        # We access the visualizer attached to the inner inferencer
        visualizer = getattr(inferencer, 'inferencer', inferencer).visualizer
        if visualizer is None:
            print('[ERROR] Visualizer not initialized; skipping frame drawing.')
            continue

        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=pose_data_sample,
            draw_gt=False,
            draw_pred=True,
            draw_bbox=True,
            draw_heatmap=False,
            show_kpt_idx=False,
            skeleton_style='mmpose',
            show=False # no pop up window, just draw to buffer
        )
        
        # 5. Get the resulting image
        vis_frame = visualizer.get_image()

        # [OPTIONAL] 6. Manually draw the ID number big and clear
        # Sometimes the built-in visualizer ID is too small. 
        # This draws a big number on the chest.
        for person in final_people:
            # Calculate chest center (approximate between shoulders)
            # Keypoints 5 and 6 are usually shoulders in COCO format
            kpts = np.array(person['keypoints'])
            if len(kpts) > 6:
                shoulder_center = (kpts[5] + kpts[6]) / 2
                x, y = int(shoulder_center[0]), int(shoulder_center[1])
                
                track_id = person['track_id']
                cv2.putText(vis_frame, f"ID: {track_id}", (x - 20, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write to video file
        out.write(vis_frame)


    # -------- Cleanup -------- #

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Analysis saved to {OUTPUT_VIDEO}")

    # -------- End Cleanup -------- #


if __name__ == '__main__':
    main()