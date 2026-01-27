import cv2
import mediapipe as mp

def pose_estimation(filepath_in: str, apply_smoothing: bool = True, smooth_3d: bool = True):
    """
    Extract 2D and 3D pose keypoints from a video using MediaPipe Pose.
    
    Args:
        filepath_in: Path to the input video file
        apply_smoothing: Not used (kept for backward compatibility)
        smooth_3d: Not used (kept for backward compatibility)
    
    Returns:
        keypoints_2d_list: List of 2D keypoints per frame [frame, keypoint, (x, y, visibility)]
        keypoints_3d_list: List of 3D keypoints per frame [frame, keypoint, (x, y, z, visibility)]
    """
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    
    # Initialize result lists
    keypoints_2d_list = []
    keypoints_3d_list = []
    
    # Setup the Pose model
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        # Open the video file
        cap = cv2.VideoCapture(filepath_in)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {filepath_in}")
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Processing video: {filepath_in} (FPS: {fps})")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print(f"Finished processing. Total frames: {frame_count}")
                break
            
            # Convert BGR to RGB for MediaPipe
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = pose.process(image_rgb)
            
            # Extract keypoints if pose is detected
            if results.pose_landmarks and results.pose_world_landmarks:
                # Get image dimensions for 2D coordinates
                h, w = image.shape[:2]
                
                # Extract 2D keypoints (pixel coordinates)
                keypoints_2d = []
                for landmark in results.pose_landmarks.landmark:
                    x = landmark.x * w  # Convert normalized to pixel coordinates
                    y = landmark.y * h
                    visibility = landmark.visibility
                    keypoints_2d.append([x, y, visibility])
                
                # Extract 3D keypoints (world coordinates in meters)
                keypoints_3d = []
                for landmark in results.pose_world_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    visibility = landmark.visibility
                    keypoints_3d.append([x, y, z, visibility])
                
                keypoints_2d_list.append(keypoints_2d)
                keypoints_3d_list.append(keypoints_3d)
            else:
                # No pose detected, append empty arrays
                keypoints_2d_list.append([])
                keypoints_3d_list.append([])
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
    
    print(f"Pose estimation complete. Extracted keypoints from {len(keypoints_2d_list)} frames.")
    return keypoints_2d_list, keypoints_3d_list