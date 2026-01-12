import cv2
import numpy as np
import os
import sys
import traceback
import math

from mmpose.apis import MMPoseInferencer
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
from one_euro_filter import OneEuroFilter

# --------------------config-------------------- #
INPUT_VIDEO = './data/IMG_1057.mov'
OUTPUT_VIDEO = './data/output_tracked2.mp4'
OUTPUT_CODEC = 'mp4v' # Codec for output video

DET_MODEL = 'rtmdet-l' # Detection model
POSE_MODEL = 'rtmw-l_8xb320-270e_cocktail14-384x288' # Pose model
NUM_KPTS = 133 # Number of keypoints for the selected pose model

TRACK_THRESH = 0.25 # standard confidence to start a track
MATCH_THRESH = 0.8 # threshold for matching (higher = stricter)
FRAME_RATE = 30
DIST_THRESH = 50 # threshold for tracking box to original box

SMOOTH_STATIC = 0.04 # OneEuroFilter static smoothing parameter
SMOOTH_DYNAMIC = 1 # OneEuroFilter dynamic smoothingparameter
MOTION_SPEED_THRESH = 1  # pixels per frame threshold between static/dynamic
# Beta values for static vs dynamic smoothing
BETA_STATIC = 0.01
BETA_DYNAMIC = 0.1
# --------------------end config-------------------- #

def setupInferencer():
    inferencer = None

    if MMPoseInferencer is None:
        print(f"\n[CRITICAL ERROR] MMPose is not available in this environment.")
        print("This is due to a CUDA/PyTorch version mismatch:")
        print("  - System CUDA: 11.8")
        print("  - PyTorch CUDA: 12.1")
        print("\nTo fix this, either:")
        print("  1. Update system CUDA to 12.1, or")
        print("  2. Reinstall PyTorch for CUDA 11.8")
        sys.exit(1)

    try:
        inferencer = MMPoseInferencer(
            pose2d=POSE_MODEL,
            det_model=DET_MODEL,
            det_cat_ids=[0], # [0] is for detecting people
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to initialize MMPose.")
        print(f"Error Details: {e}")
        print("-" * 30)
        traceback.print_exc() # <--- PRINTS THE ACTUAL STACK TRACE
        print("-" * 30)
        sys.exit(1) # Force exit with error code 1

    return inferencer

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

def get_center(bbox):
    """Returns center (x, y) of a bounding box [x1, y1, x2, y2]"""
    return np.array([ (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2 ])

def compute_speed(prev_kpts: np.ndarray, curr_kpts: np.ndarray) -> float:
    """Compute average pixel displacement between previous and current keypoints.
    Shapes expected: (K, 2). Returns mean L2 displacement across visible keypoints.
    """
    if prev_kpts is None or curr_kpts is None:
        return 0.0
    if prev_kpts.shape != curr_kpts.shape or prev_kpts.size == 0:
        return 0.0
    disp = curr_kpts - prev_kpts
    dists = np.linalg.norm(disp, axis=1)
    # Robust average to avoid outliers
    return float(np.median(dists))

def main():
    # ---- Initialization ---- #
    print("Initializing...")
    
    # Setup MMPOSE
    inferencer = setupInferencer()

    # Setup Tracker and Smoother
    smoothing_filters = {}
    active_tracks = {}
    next_track_id = 0
    prev_keypoints = {}

    # Video Setup
    cap = openVideo()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # ---- Pose Generation with Tracking and Smoothing ---- #
    print("Starting Pose Inference with Tracking and Smoothing...")

    while cap.isOpened():

        # Read frame
        ret, frame = cap.read()
        if not ret: break

        # ---- Detection and inital pose generation  ---- #
        result_generator = inferencer(frame, return_vis=False)
        result = next(result_generator)
        
        # Extract bounding boxes for the tracker.
        raw_predictions = result['predictions'][0] # [{'bbox': [x1,y1,x2,y2], 'bbox_score': 0.9, ...}, ...]
        
        # ---- Tracking ---- #
        
        # We need to assign IDs to 'raw_predictions' so the smoother knows who is who.
        current_centers = []
        for p in raw_predictions:
            bbox = p['bbox'][0]
            current_centers.append(get_center(bbox))
            
        # Match current detections to active tracks by distance
        assigned_ids = [-1] * len(raw_predictions)
        used_tracks = set()
        
        # Greedy matching: Find closest track for each detection
        for i, center in enumerate(current_centers):
            best_id = -1
            min_dist = float('inf')
            
            for track_id, last_center in active_tracks.items():
                if track_id in used_tracks: continue
                
                dist = np.linalg.norm(center - last_center)
                if dist < DIST_THRESH and dist < min_dist:
                    min_dist = dist
                    best_id = track_id
            
            if best_id != -1:
                assigned_ids[i] = best_id
                used_tracks.add(best_id)
            else:
                # New person found
                assigned_ids[i] = next_track_id
                next_track_id += 1

        # Update active_tracks for next frame
        new_active_tracks = {}
        
        final_people = []

        for i, person in enumerate(raw_predictions):
            track_id = assigned_ids[i]
            
            # Save center for next frame logic
            new_active_tracks[track_id] = current_centers[i]
            
            # 3. Apply Smoothing
            if track_id not in smoothing_filters:
                # Initialize OneEuroFilter for this new ID
                smoothing_filters[track_id] = OneEuroFilter(min_cutoff=SMOOTH_STATIC, beta=BETA_DYNAMIC, freq=fps if fps > 0 else FRAME_RATE)
            
            raw_kpts = np.array(person['keypoints'])
            # Extract motion parameters based on speed
            speed = compute_speed(prev_keypoints.get(track_id), raw_kpts)
            if speed < MOTION_SPEED_THRESH:
                smoothing_filters[track_id].min_cutoff = SMOOTH_STATIC
                smoothing_filters[track_id].beta = BETA_STATIC
            else:
                smoothing_filters[track_id].min_cutoff = SMOOTH_DYNAMIC
                smoothing_filters[track_id].beta = BETA_DYNAMIC
            smoothed_kpts = smoothing_filters[track_id](raw_kpts)
            person['keypoints'] = smoothed_kpts.tolist()
            person['track_id'] = track_id
            # Store for next frame speed estimation
            prev_keypoints[track_id] = raw_kpts
            
            final_people.append(person)
            
        active_tracks = new_active_tracks

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