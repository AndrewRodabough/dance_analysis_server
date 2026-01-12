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
INPUT_VIDEO = './data/clip_1.mov'
OUTPUT_VIDEO = './data/output_tracked5.mp4'
OUTPUT_CODEC = 'mp4v' # Codec for output video

DET_MODEL = 'rtmdet-l' # Detection model
POSE_MODEL = 'rtmw-l_8xb320-270e_cocktail14-384x288' # Pose model
NUM_KPTS = 133 # Number of keypoints for the selected pose model

TRACK_THRESH = 0.25 # standard confidence to start a track
MATCH_THRESH = 0.8 # threshold for matching (higher = stricter)
FRAME_RATE = 30
DIST_THRESH = 50 # threshold for tracking box to original box

# False positive filtering
BBOX_CONF_THRESH = 0.5     # minimum bbox detection confidence
KPT_CONF_THRESH = 0.3      # minimum average keypoint confidence
MIN_KEYPOINTS_VISIBLE = 5  # minimum visible keypoints to consider valid person
MIN_TRACK_FRAMES = 3       # frames a track must persist before displaying
ASPECT_RATIO_MIN = 0.3     # min height/width ratio (filters horizontal objects)
ASPECT_RATIO_MAX = 4.0     # max height/width ratio (filters vertical objects)

# Smoothing parameters (continuous scale-aware blending)
SMOOTH_MIN = 0.8    # min_cutoff for slow/static motion
SMOOTH_MAX = 3.5    # min_cutoff for fast motion
BETA_MIN = 0.1      # beta for slow/static motion
BETA_MAX = 1.0      # beta for fast motion
# Speed scaling (normalized by bbox diagonal)
SPEED_SCALE_LOW = 0.002   # normalized speed below this → full static smoothing
SPEED_SCALE_HIGH = 0.015  # normalized speed above this → full dynamic smoothing
BBOX_SCALE_SMOOTHING = 0.15  # exponential smoothing for bbox size (0.1-0.2 recommended)
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

def compute_speed(prev_kpts: np.ndarray, curr_kpts: np.ndarray, bbox_scale: float = 1.0) -> float:
    """Compute average pixel displacement between previous and current keypoints,
    normalized by bbox_scale for scale invariance.
    Shapes expected: (K, 2). Returns normalized median L2 displacement.
    """
    if prev_kpts is None or curr_kpts is None:
        return 0.0
    if prev_kpts.shape != curr_kpts.shape or prev_kpts.size == 0:
        return 0.0
    disp = curr_kpts - prev_kpts
    dists = np.linalg.norm(disp, axis=1)
    # Robust average to avoid outliers, normalized by bbox scale
    raw_speed = float(np.median(dists))
    return raw_speed / max(bbox_scale, 1e-6)

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
    smoothed_bbox_scale = {}  # pose-independent scale per track
    track_frame_count = {}    # persistence counter for false positive filtering

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
        
        # ---- False Positive Filtering ---- #
        filtered_predictions = []
        for p in raw_predictions:
            # 1. Check bbox confidence
            bbox_score = p.get('bbox_score', [1.0])[0] if isinstance(p.get('bbox_score'), list) else p.get('bbox_score', 1.0)
            if bbox_score < BBOX_CONF_THRESH:
                continue
            
            # 2. Check keypoint confidence and visibility
            kpt_scores = p.get('keypoint_scores', [])
            if len(kpt_scores) > 0:
                avg_kpt_conf = np.mean(kpt_scores)
                visible_kpts = np.sum(np.array(kpt_scores) > 0.3)
                if avg_kpt_conf < KPT_CONF_THRESH or visible_kpts < MIN_KEYPOINTS_VISIBLE:
                    continue
            
            # 3. Check aspect ratio (human body proportions)
            x1, y1, x2, y2 = p['bbox'][0]
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            aspect_ratio = height / width
            if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                continue
            
            filtered_predictions.append(p)
        
        raw_predictions = filtered_predictions
        
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
            
            # Update track persistence counter
            if track_id not in track_frame_count:
                track_frame_count[track_id] = 1
            else:
                track_frame_count[track_id] += 1
            
            # Skip rendering if track hasn't persisted long enough (likely false positive)
            if track_frame_count[track_id] < MIN_TRACK_FRAMES:
                continue
            
            # 3. Apply Smoothing
            if track_id not in smoothing_filters:
                # Initialize OneEuroFilter for this new ID with starting parameters
                smoothing_filters[track_id] = OneEuroFilter(min_cutoff=SMOOTH_MIN, beta=BETA_MIN, freq=fps if fps > 0 else FRAME_RATE)
            
            raw_kpts = np.array(person['keypoints'])
            
            # Compute bbox diagonal
            x1, y1, x2, y2 = person['bbox'][0]
            bbox_diagonal = math.hypot(x2 - x1, y2 - y1)
            
            # Apply exponential smoothing to bbox scale to ignore pose-based size changes
            # (arms out/in shouldn't affect speed normalization)
            if track_id not in smoothed_bbox_scale:
                smoothed_bbox_scale[track_id] = bbox_diagonal
            else:
                smoothed_bbox_scale[track_id] = (
                    BBOX_SCALE_SMOOTHING * bbox_diagonal + 
                    (1 - BBOX_SCALE_SMOOTHING) * smoothed_bbox_scale[track_id]
                )
            
            # Use smoothed scale for speed normalization
            speed_norm = compute_speed(prev_keypoints.get(track_id), raw_kpts, smoothed_bbox_scale[track_id])
            
            # Continuous parameter blending based on normalized speed
            # Use smooth interpolation (clamp between 0 and 1)
            if speed_norm <= SPEED_SCALE_LOW:
                blend = 0.0
            elif speed_norm >= SPEED_SCALE_HIGH:
                blend = 1.0
            else:
                # Linear interpolation between low and high
                blend = (speed_norm - SPEED_SCALE_LOW) / (SPEED_SCALE_HIGH - SPEED_SCALE_LOW)
            
            # Apply blended parameters
            smoothing_filters[track_id].min_cutoff = SMOOTH_MIN + blend * (SMOOTH_MAX - SMOOTH_MIN)
            smoothing_filters[track_id].beta = BETA_MIN + blend * (BETA_MAX - BETA_MIN)
            
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