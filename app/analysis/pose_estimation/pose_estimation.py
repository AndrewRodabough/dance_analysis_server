import cv2
from rtmpose3d import RTMPose3D
from one_euro_filter import OneEuroFilter
import numpy as np
from app.utils.device_manager import get_device

def pose_estimation(filepath_in: str, apply_smoothing: bool = True):
    # Initialize model (auto-downloads checkpoints)
    device = get_device()
    model = RTMPose3D.from_pretrained('rbarac/rtmpose3d', device=device)

    # Open video file
    cap = cv2.VideoCapture(filepath_in)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {filepath_in}")
    
    # Get FPS for smoothing
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if apply_smoothing else 30
    
    all_keypoints_3d = []
    all_keypoints_2d = []
    all_scores = []
    frame_count = 0
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference on current frame
        results = model(frame, return_tensors='np')
        
        # Collect results
        all_keypoints_3d.append(results['keypoints_3d'])  # [N, 133, 3]
        all_keypoints_2d.append(results['keypoints_2d'])  # [N, 133, 2]
        all_scores.append(results['scores'])              # [N, 133]
        
        frame_count += 1
    
    cap.release()
    
    # Apply smoothing if requested
    if apply_smoothing:
        print("Applying smoothing filters...")
        all_keypoints_2d, all_keypoints_3d = _apply_smoothing(
            all_keypoints_2d, all_keypoints_3d, fps
        )
        print(f"Smoothing complete - processed {len(all_keypoints_2d)} frames")
    
    return all_keypoints_2d, all_keypoints_3d


def _apply_smoothing(keypoints_2d_list, keypoints_3d_list, fps):
    """Apply One Euro Filter smoothing to remove jitter and pops"""
    # Track filters per person (dict of person_id -> filters)
    filters_2d = {}  # person_id -> list of filters (one per keypoint)
    filters_3d = {}  # person_id -> list of filters (one per keypoint)
    
    smoothed_2d_list = []
    smoothed_3d_list = []
    
    for frame_idx, (kp2d, kp3d) in enumerate(zip(keypoints_2d_list, keypoints_3d_list)):
        num_people = kp2d.shape[0] if len(kp2d.shape) > 2 else 1
        
        frame_kp2d = []
        frame_kp3d = []
        
        for person_idx in range(num_people):
            person_id = person_idx  # Simple tracking by index
            
            if len(kp2d.shape) == 2:
                person_kp2d = kp2d.copy()
                person_kp3d = kp3d.copy()
            else:
                person_kp2d = kp2d[person_idx].copy()
                person_kp3d = kp3d[person_idx].copy()
            
            # Initialize filters for new person
            if person_id not in filters_2d:
                filters_2d[person_id] = [OneEuroFilter(freq=fps, min_cutoff=5.0, beta=0.05, d_cutoff=1.0) 
                                          for _ in range(person_kp2d.shape[0])]
                filters_3d[person_id] = [OneEuroFilter(freq=fps, min_cutoff=5.0, beta=0.05, d_cutoff=1.0) 
                                          for _ in range(person_kp3d.shape[0])]
            
            # Apply filter to each keypoint
            timestamp = frame_idx / fps
            
            # Smooth 2D keypoints
            for kp_idx in range(person_kp2d.shape[0]):
                person_kp2d[kp_idx] = filters_2d[person_id][kp_idx](person_kp2d[kp_idx], timestamp)
            
            # Smooth 3D keypoints  
            for kp_idx in range(person_kp3d.shape[0]):
                person_kp3d[kp_idx] = filters_3d[person_id][kp_idx](person_kp3d[kp_idx], timestamp)
            
            frame_kp2d.append(person_kp2d)
            frame_kp3d.append(person_kp3d)
        
        # Convert back to array format
        if num_people == 1:
            smoothed_2d_list.append(frame_kp2d[0])
            smoothed_3d_list.append(frame_kp3d[0])
        else:
            smoothed_2d_list.append(np.array(frame_kp2d))
            smoothed_3d_list.append(np.array(frame_kp3d))
    
    return smoothed_2d_list, smoothed_3d_list