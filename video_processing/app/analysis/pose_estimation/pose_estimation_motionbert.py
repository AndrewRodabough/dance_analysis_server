from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import logging

from app.utils.device_manager import get_device
from OneEuroFilter import OneEuroFilter

logger = logging.getLogger(__name__)

# Path to video_processing/ directory (4 levels up from this file)
# File structure: video_processing/app/analysis/pose_estimation/pose_estimation_motionbert.py
ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS = ROOT / 'models'
CONFIGS = MODELS / 'configs'
CHECKPOINTS = MODELS / 'checkpoints'

# Model Paths
DET_CFG: Path = CONFIGS / 'rtmdet_l_8xb32-300e_coco.py'
DET_CKPT: Path = CHECKPOINTS / 'rtmdet_l_8xb32-300e_coco.pth'

POSE2D_CFG: Path = CONFIGS / 'rtmpose-l_8xb256-420e_coco-384x288.py'
POSE2D_CKPT: Path = CHECKPOINTS / 'rtmpose-l_simcc-body7_pt-body7_420e-384x288.pth'

LIFT3D_CFG: Path = CONFIGS / 'motionbert_dstformer-ft-243frm_8xb32-120e_h36m_modified.py'
LIFT3D_CKPT: Path = CHECKPOINTS / 'motionbert_ft_h36m.pth'

MOTIONBERT_SEQ_LEN = 243 # MotionBERT requires 243 frames of temporal context

def lift_feet_to_3d(keypoints_2d_wholebody: np.ndarray, 
                   keypoints_3d_h36m: np.ndarray,
                   image_width: int = 1920,
                   image_height: int = 1080) -> np.ndarray:
    """
    Lift 2D feet to 3D with GLOBAL COORDINATE CORRECTION.
    Fixes the 'facing away' and 'mirrored' issues by flipping body axes first.
    """
    # Handle single person (no batch dimension)
    if len(keypoints_2d_wholebody.shape) == 2:
        keypoints_2d_wholebody = keypoints_2d_wholebody[np.newaxis, ...]
    if len(keypoints_3d_h36m.shape) == 2:
        keypoints_3d_h36m = keypoints_3d_h36m[np.newaxis, ...]
    
    num_people = keypoints_3d_h36m.shape[0]
    keypoints_3d_extended = np.zeros((num_people, 23, 3))
    
    # === STEP 1: GLOBAL COORDINATE CORRECTION ===
    # The raw MotionBERT output is likely facing 'away' (Z+) and mirrored (X).
    # We fix the BODY first so the feet have a valid structure to attach to.
    
    body_kps = keypoints_3d_h36m.copy()
    
    # 1. FLIP LATERAL (Fix the Mirror Effect)
    # This moves the arm from Left to Right to match the video.
    body_kps[:, :, 0] *= -1 
    
    # 2. FLIP DEPTH (Fix "Facing Away")
    # This turns the dancer 180 degrees to face the camera.
    body_kps[:, :, 2] *= -1
    
    # Store corrected body
    keypoints_3d_extended[:, :17, :] = body_kps
    
    for person_idx in range(num_people):
        kp2d = keypoints_2d_wholebody[person_idx]
        kp3d = body_kps[person_idx] # Use the CORRECTED body
        
        # Get 3D ankle positions (H36M format)
        left_ankle_3d = kp3d[H36M_LEFT_ANKLE]   # Index 6
        right_ankle_3d = kp3d[H36M_RIGHT_ANKLE]  # Index 3
        
        # Get 2D ankle positions (COCO format)
        left_ankle_2d = kp2d[COCO_LEFT_ANKLE]   # Index 15
        right_ankle_2d = kp2d[COCO_RIGHT_ANKLE]  # Index 16
        
        # Get 2D feet positions
        left_feet_2d = np.array([
            kp2d[COCO_LEFT_BIG_TOE],    # 17
            kp2d[COCO_LEFT_SMALL_TOE],  # 18
            kp2d[COCO_LEFT_HEEL],       # 19
        ])
        right_feet_2d = np.array([
            kp2d[COCO_RIGHT_BIG_TOE],   # 20
            kp2d[COCO_RIGHT_SMALL_TOE], # 21
            kp2d[COCO_RIGHT_HEEL],      # 22
        ])
        
        # --- SCALE CALCULATION ---
        left_shin_3d = np.linalg.norm(kp3d[5] - kp3d[6])
        right_shin_3d = np.linalg.norm(kp3d[2] - kp3d[3])
        avg_shin_3d = (left_shin_3d + right_shin_3d) / 2
        
        left_knee_2d = kp2d[13]
        right_knee_2d = kp2d[14]
        left_shin_2d = np.linalg.norm(left_knee_2d - left_ankle_2d)
        right_shin_2d = np.linalg.norm(right_knee_2d - right_ankle_2d)
        avg_shin_2d = (left_shin_2d + right_shin_2d) / 2
        
        MIN_SCALE, MAX_SCALE = 0.001, 0.010
        if avg_shin_2d > 10:
            scale = avg_shin_3d / avg_shin_2d
            scale = np.clip(scale, MIN_SCALE, MAX_SCALE)
        else:
            scale = 0.003

        # --- ANKLE HEIGHT CORRECTION ---
        # 2D Y is down, 3D Y is up. If heel is below ankle in 2D (positive diff),
        # we need to move ankle DOWN in 3D (negative).
        left_heel_2d = kp2d[COCO_LEFT_HEEL]
        left_ankle_heel_offset_2d = left_heel_2d[1] - left_ankle_2d[1]
        right_heel_2d = kp2d[COCO_RIGHT_HEEL]
        right_ankle_heel_offset_2d = right_heel_2d[1] - right_ankle_2d[1]
        
        left_ankle_correction = np.clip(-left_ankle_heel_offset_2d * scale, -0.15, 0.0)
        right_ankle_correction = np.clip(-right_ankle_heel_offset_2d * scale, -0.15, 0.0)
        
        # Lift feet (Pass dummy vectors as we use Camera Space now)
        dummy_vec = np.array([0,0,1])
        
        for i, foot_2d in enumerate(left_feet_2d):
            offset_3d = _compute_foot_offset_3d(
                i, foot_2d, left_ankle_2d, scale, dummy_vec, dummy_vec, True
            )
            offset_3d[1] += left_ankle_correction
            keypoints_3d_extended[person_idx, 17 + i, :] = left_ankle_3d + offset_3d
        
        for i, foot_2d in enumerate(right_feet_2d):
            offset_3d = _compute_foot_offset_3d(
                i, foot_2d, right_ankle_2d, scale, dummy_vec, dummy_vec, False
            )
            offset_3d[1] += right_ankle_correction
            keypoints_3d_extended[person_idx, 20 + i, :] = right_ankle_3d + offset_3d
    
    return keypoints_3d_extended

def _compute_foot_offset_3d(foot_index: int, foot_2d: np.ndarray, 
                            ankle_2d: np.ndarray, scale: float,
                            body_forward: np.ndarray, body_right: np.ndarray,
                            is_left: bool) -> np.ndarray:
    """
    Compute 3D offset using CAMERA-SPACE projection.
    Aligned for corrected body coordinates (X-flip, Z-flip).
    """
    vec_2d = foot_2d - ankle_2d
    mag_2d = np.linalg.norm(vec_2d)
    
    # 1. Estimate Foot Length
    if foot_index == 2:  # Heel
        min_length, max_length = 0.03, 0.10
    else:  # Toes
        min_length, max_length = 0.10, 0.25
    foot_length_3d = np.clip(mag_2d * scale, min_length, max_length)
    
    # 2. Get 2D Direction
    if mag_2d > 1e-6:
        dir_2d = vec_2d / mag_2d
    else:
        # Default: Heel back/up, Toe forward/down
        if foot_index == 2: dir_2d = np.array([0.0, -1.0]) 
        else: dir_2d = np.array([0.0, 1.0]) 
            
    # === MAPPING 2D to 3D (The Fix) ===
    
    # Lateral Offset (X)
    # 2D Image Right -> 3D Plot Right. 
    # Since we flipped the body X, positive X is now consistent with image Right.
    x_offset = dir_2d[0] * foot_length_3d
    
    # Depth Offset (Z) - THE REVERSAL FIX
    # 2D Image Down (+Y) = Closer to Camera.
    # Previous code used "-dir_2d[1]".
    # We REMOVE the negative sign to flip the depth.
    # Now: Positive 2D Y (Down) -> Positive Z Offset (Forward/Closer).
    z_offset = dir_2d[1] * foot_length_3d 
    
    # Vertical Offset (Y)
    # 2D Image Down (+Y) = 3D Space Down (-Y).
    # We still need negative here to map "Pixel Down" to "Height Down".
    raw_vertical = -dir_2d[1] * foot_length_3d
    
    # Damping & Clamping
    # We assume most vertical pixel movement is depth (Z), not height (Y).
    y_offset = raw_vertical * 0.3 
    
    if foot_index == 2: # Heel
        # Heel cannot be above ankle
        y_offset = np.clip(y_offset, -0.15, -0.01) 
    else: # Toes
        y_offset = np.clip(y_offset, -0.25, 0.05)

    offset_3d = np.array([x_offset, y_offset, z_offset])
    return offset_3d

class PoseEstimationPipeline:
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self.pose3d_model_name = None  # Will be set during model initialization
        self._init_models()
        
    def _init_models(self):
        """Initialize detection and pose models using unified MMPoseInferencer approach"""
        logger.info(f"Initializing models on {self.device}...")
        from mmpose.apis import MMPoseInferencer
        
        # Use MMPose's built-in model names - automatically downloads configs and weights
        
        # Initialize 2D-only inferencer for frame-by-frame processing
        logger.info("Loading detector and 2D pose estimator (auto-download)...")
        try:
            self.inferencer_2d = MMPoseInferencer(
                det_model='rtmdet-l',  # Auto-download RTMDet-L
                det_cat_ids=[0],  # 0 = Person class in COCO
                pose2d='rtmpose-l',  # Auto-download RTMPose-L
                device=self.device,
            )
            logger.info("✓ 2D pose estimator loaded successfully")
        except Exception as e:
            raise ValueError(f"2D model loading failed: {e}")
        
        # Initialize full pipeline inferencer (2D + 3D) for video processing
        logger.info("Loading 3D pose estimator...")
        
        # Try available 3D pose models in order of preference
        pose3d_models = [
            # Try local MotionBERT first (most accurate)
            ({
                'config': str(LIFT3D_CFG),
                'checkpoint': str(LIFT3D_CKPT)
            }, 'MotionBERT (local)'),
            # Fallback to MMPose registry models
            ('human3d', 'Human3D'),
            ('video-pose-lift', 'VideoPose3D'),
        ]
        
        last_error = None
        for model_spec, display_name in pose3d_models:
            try:
                logger.info(f"  Trying {display_name}...")
                
                # Handle dict (local files) vs string (registry name)
                if isinstance(model_spec, dict):
                    # Check if local files exist
                    config_path = Path(model_spec['config'])
                    ckpt_path = Path(model_spec['checkpoint'])
                    
                    if not config_path.exists():
                        logger.debug(f"  Config not found: {config_path}")
                        raise FileNotFoundError(f"Config not found: {config_path}")
                    if not ckpt_path.exists():
                        logger.debug(f"  Checkpoint not found: {ckpt_path}")
                        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                    
                    # Use local files
                    self.inferencer = MMPoseInferencer(
                        det_model='rtmdet-l',
                        det_cat_ids=[0],
                        pose2d='rtmpose-l',
                        pose3d=model_spec,
                        device=self.device,
                    )
                else:
                    # Use registry name
                    self.inferencer = MMPoseInferencer(
                        det_model='rtmdet-l',
                        det_cat_ids=[0],
                        pose2d='rtmpose-l',
                        pose3d=model_spec,
                        device=self.device,
                    )
                
                logger.info(f"✓ 3D pose estimator loaded: {display_name}")
                self.pose3d_model_name = display_name
                break
            except Exception as e:
                logger.debug(f"  Failed to load '{display_name}': {e}")
                last_error = e
                continue
        else:
            # None of the models worked
            raise ValueError(
                f"Failed to load any 3D pose model. Last error: {last_error}"
            )
        
        logger.info("✓ All models loaded successfully!")
    
    def process_video(self, video_path: Path, apply_smoothing: bool = False) -> Tuple[List, List, List]:
        """
        Process a video file and return 2D and 3D keypoints.
        
        Args:
            video_path: Path to input video
            apply_smoothing: Whether to apply OneEuroFilter smoothing
            
        Returns:
            Tuple of (keypoints_2d_list, keypoints_3d_list, scores_list)
            - keypoints_2d: List of [N, 133, 2] arrays (wholebody 2D)
            - keypoints_3d: List of [N, 17, 3] arrays (body 3D)
            - scores: List of [N, 17] arrays (confidence scores for 3D)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {total_frames} frames @ {fps} FPS")
        
        # Collect all 2D poses first
        all_keypoints_2d = []
        all_keypoints_2d_17 = []  # Body-only for MotionBERT
        all_scores_2d = []
        all_bboxes = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 50 == 0:
                logger.info(f"  Processing frame {frame_idx}/{total_frames} (2D pose)...")
            
            # Run 2D pose estimation using unified inferencer
            result_generator = self.inferencer_2d(frame)
            results_2d = next(result_generator)
            
            # Extract predictions - handle different result structures
            predictions = None
            if 'predictions' in results_2d:
                preds = results_2d['predictions']
                if len(preds) > 0:
                    predictions = preds[0] if isinstance(preds[0], list) else preds
            
            if predictions and len(predictions) > 0:
                # Extract keypoints for all detected persons
                frame_kp2d = []
                frame_kp2d_17 = []
                frame_scores = []
                frame_bboxes = []
                
                for pred in predictions:
                    # Handle dict or object format
                    if isinstance(pred, dict):
                        kps = np.array(pred.get('keypoints', pred.get('keypoints_2d', [])))
                        scores = np.array(pred.get('keypoint_scores', np.ones(len(kps))))
                        bbox = np.array(pred.get('bbox', [0, 0, frame.shape[1], frame.shape[0]]))
                    else:
                        # Handle object with attributes
                        kps = np.array(getattr(pred, 'keypoints', getattr(pred, 'keypoints_2d', [])))
                        scores = np.array(getattr(pred, 'keypoint_scores', np.ones(len(kps))))
                        bbox = np.array(getattr(pred, 'bbox', [0, 0, frame.shape[1], frame.shape[0]]))
                    
                    # Ensure 2D shape
                    if len(kps.shape) == 1:
                        kps = kps.reshape(-1, 2)
                    
                    num_kps = kps.shape[0]
                    frame_kp2d.append(kps)
                    
                    # Extract first 17 body keypoints for 3D lifting
                    frame_kp2d_17.append(kps[:17] if num_kps >= 17 else np.pad(kps, ((0, 17-num_kps), (0, 0))))
                    frame_scores.append(scores)
                    frame_bboxes.append(bbox)
                
                all_keypoints_2d.append(np.array(frame_kp2d))
                all_keypoints_2d_17.append(np.array(frame_kp2d_17))
                all_scores_2d.append(np.array(frame_scores))
                all_bboxes.append(np.array(frame_bboxes))
            else:
                # No detection - use zeros
                all_keypoints_2d.append(np.zeros((1, 133, 2)))
                all_keypoints_2d_17.append(np.zeros((1, 17, 2)))
                all_scores_2d.append(np.zeros((1, 133)))
                all_bboxes.append(np.zeros((1, 4)))
            
            frame_idx += 1
        
        cap.release()
        print(f"[INFO] 2D pose estimation complete: {len(all_keypoints_2d)} frames")
        
        # Now lift 2D to 3D using MotionBERT
        print("[INFO] Lifting 2D poses to 3D with MotionBERT...")
        
        # Try using the 3D inferencer directly on the video first
        # This is more accurate as MotionBERT uses temporal context
        all_keypoints_3d, all_scores_3d = self._lift_to_3d_via_video(
            video_path, all_keypoints_2d_17, all_scores_2d, all_bboxes, fps
        )
        
        # TEMPORARILY DISABLED: Custom feet lifting
        # Feet lifting will be re-enabled after skeleton refactoring
        # For now, we return the 17 body keypoints without custom feet lifting
        # Normalize outputs for downstream skeleton loading (frames, joints, dims).
        def _normalize_keypoints_list(keypoints_list):
            if isinstance(keypoints_list, np.ndarray):
                data = keypoints_list
            else:
                if not keypoints_list:
                    return np.zeros((0, 0, 0), dtype=np.float32)

                # Collapse to first person and pad joints to a common length.
                first_person = []
                max_joints = 0
                dims = None

                for item in keypoints_list:
                    arr = np.asarray(item)
                    if arr.ndim == 3:
                        arr = arr[0]
                    if arr.ndim == 1:
                        if dims is None:
                            dims = 2
                        arr = arr.reshape(-1, dims)
                    if dims is None:
                        dims = arr.shape[1]
                    max_joints = max(max_joints, arr.shape[0])
                    first_person.append(arr)

                data = np.zeros((len(first_person), max_joints, dims), dtype=np.float32)
                for idx, arr in enumerate(first_person):
                    data[idx, :arr.shape[0], :arr.shape[1]] = arr

            # If a person dimension exists, keep the first person for now.
            if data.ndim == 4:
                data = data[:, 0, :, :]

            return data

        def _normalize_scores_list(scores_list):
            if isinstance(scores_list, np.ndarray):
                data = scores_list
            else:
                if not scores_list:
                    return np.zeros((0, 0), dtype=np.float32)

                first_person = []
                max_joints = 0

                for item in scores_list:
                    arr = np.asarray(item)
                    if arr.ndim == 2:
                        arr = arr[0]
                    max_joints = max(max_joints, arr.shape[0])
                    first_person.append(arr)

                data = np.zeros((len(first_person), max_joints), dtype=np.float32)
                for idx, arr in enumerate(first_person):
                    data[idx, :arr.shape[0]] = arr

            if data.ndim == 3:
                data = data[:, 0, :]

            return data

        all_keypoints_2d = _normalize_keypoints_list(all_keypoints_2d)
        all_keypoints_3d = _normalize_keypoints_list(all_keypoints_3d)
        all_scores_3d = _normalize_scores_list(all_scores_3d)

        print("[INFO] Custom feet lifting TEMPORARILY DISABLED (skeleton refactoring in progress)")
        if all_keypoints_3d.size > 0:
            print(f"[INFO] Returning 3D keypoints with {all_keypoints_3d.shape[-2]} points (17 body keypoints only)")
        else:
            print("[INFO] Returning 3D keypoints with 0 points (no frames)")
        
        # Skip the feet lifting code block
        
        # Apply smoothing if requested
        if apply_smoothing:
            print("[INFO] Applying OneEuroFilter smoothing...")
            all_keypoints_2d, all_keypoints_3d = self._apply_smoothing(
                all_keypoints_2d, all_keypoints_3d, fps
            )
            all_keypoints_2d = _normalize_keypoints_list(all_keypoints_2d)
            all_keypoints_3d = _normalize_keypoints_list(all_keypoints_3d)
            print(f"[INFO] Smoothing complete")
        
        return all_keypoints_2d, all_keypoints_3d, all_scores_3d
    
    def _lift_to_3d_via_video(self, video_path: str,
                              keypoints_2d_list: List[np.ndarray],
                              scores_list: List[np.ndarray],
                              bboxes_list: List[np.ndarray],
                              fps: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Lift 2D to 3D by running the pose3d inferencer directly on the video.
        Falls back to coordinate transformation if that fails.
        """
        num_frames = len(keypoints_2d_list)
        
        try:
            print(f"  Running 3D inferencer on video: {video_path}")
            
            all_keypoints_3d = []
            all_scores_3d = []
            
            # Process video through unified 3D inferencer
            # MotionBERT uses temporal context (243 frames) for accurate 3D lifting
            result_generator = self.inferencer(str(video_path))
            
            frame_idx = 0
            for result in result_generator:
                if frame_idx % 50 == 0:
                    print(f"    Frame {frame_idx}/{num_frames}")
                
                # Extract 3D keypoints from result
                if 'predictions' in result and len(result['predictions']) > 0:
                    preds = result['predictions']
                    # Handle nested structure
                    if isinstance(preds[0], list):
                        preds = preds[0]
                    
                    if len(preds) > 0:
                        pred = preds[0]
                        # Get 3D keypoints
                        if isinstance(pred, dict):
                            kp3d = np.array(pred.get('keypoints', np.zeros((17, 3))))
                        else:
                            kp3d = np.array(getattr(pred, 'keypoints', np.zeros((17, 3))))
                        
                        # Ensure correct shape
                        if len(kp3d.shape) == 1:
                            kp3d = kp3d.reshape(-1, 3)
                        
                        all_keypoints_3d.append(kp3d[np.newaxis, ...])
                        
                        # Use 2D scores for confidence
                        if frame_idx < len(scores_list):
                            scores = scores_list[frame_idx]
                            if len(scores.shape) > 1:
                                scores = scores[0]
                            scores = scores[:17] if len(scores) >= 17 else np.pad(scores, (0, 17-len(scores)))
                        else:
                            scores = np.ones(17)
                        all_scores_3d.append(scores[np.newaxis, ...])
                    else:
                        all_keypoints_3d.append(np.zeros((1, 17, 3)))
                        all_scores_3d.append(np.zeros((1, 17)))
                else:
                    all_keypoints_3d.append(np.zeros((1, 17, 3)))
                    all_scores_3d.append(np.zeros((1, 17)))
                
                frame_idx += 1
            
            print(f"[INFO] 3D inference complete: {len(all_keypoints_3d)} frames")
            return all_keypoints_3d, all_scores_3d
                
        except Exception as e:
            print(f"[WARNING] 3D video inferencer failed: {e}")
            print("[INFO] Falling back to 2D coordinate transformation")
            return self._lift_to_3d_fallback(keypoints_2d_list, scores_list, fps)
    
    def _lift_to_3d_fallback(self, keypoints_2d_list: List[np.ndarray],
                             scores_list: List[np.ndarray],
                             fps: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Fallback: Convert 2D COCO keypoints to pseudo-3D H36M format.
        This converts format and flips Y axis, no actual depth estimation.
        """
        num_frames = len(keypoints_2d_list)
        all_keypoints_3d = []
        all_scores_3d = []
        
        # Build sequence for first person (in COCO format from 2D)
        keypoints_2d_seq = []
        scores_seq = []
        
        for kp2d, scores in zip(keypoints_2d_list, scores_list):
            if len(kp2d.shape) == 3 and kp2d.shape[0] > 0:
                # Multiple people - take first, get COCO 17 body keypoints
                person_kp = kp2d[0][:17]  # First 17 are body in COCO wholebody
                keypoints_2d_seq.append(person_kp)
                s = scores[0] if len(scores.shape) > 1 else scores
                scores_seq.append(s[:17] if len(s) >= 17 else np.pad(s, (0, 17-len(s))))
            elif len(kp2d.shape) == 2:
                person_kp = kp2d[:17]
                keypoints_2d_seq.append(person_kp)
                s = scores[0] if len(scores.shape) > 1 else scores
                scores_seq.append(s[:17] if len(s) >= 17 else np.pad(s, (0, 17-len(s))))
            else:
                keypoints_2d_seq.append(np.zeros((17, 2)))
                scores_seq.append(np.zeros(17))
        
        for i, (coco_kp2d, scores) in enumerate(zip(keypoints_2d_seq, scores_seq)):
            # Convert COCO 2D to H36M format
            h36m_kp2d = coco_to_h36m(coco_kp2d)
            
            # Create 3D keypoints in H36M format
            kp3d = np.zeros((17, 3))
            
            # X stays as X (left-right)
            kp3d[:, 0] = h36m_kp2d[:, 0]
            
            # Y in 3D = -Y in 2D (flip vertical so Y points up)
            kp3d[:, 1] = -h36m_kp2d[:, 1]
            
            # Z = 0 (no depth)
            kp3d[:, 2] = 0
            
            # Normalize scale (pixels to meters)
            scale = 1.7 / 500.0
            
            # Center around pelvis (H36M index 0)
            pelvis = kp3d[0].copy()
            kp3d = kp3d - pelvis
            kp3d = kp3d * scale
            
            all_keypoints_3d.append(kp3d[np.newaxis, ...])
            all_scores_3d.append(scores[np.newaxis, ...])
            
            if i % 100 == 0:
                print(f"    Frame {i}/{num_frames}")
        
        print(f"[INFO] Fallback 3D conversion complete: {len(all_keypoints_3d)} frames")
        return all_keypoints_3d, all_scores_3d
    
    def _apply_smoothing(self, keypoints_2d_list: List[np.ndarray], 
                        keypoints_3d_list: List[np.ndarray], 
                        fps: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Apply One Euro Filter smoothing to remove jitter"""
        filters_2d = {}
        filters_3d = {}
        
        smoothed_2d_list = []
        smoothed_3d_list = []
        
        for frame_idx, (kp2d, kp3d) in enumerate(zip(keypoints_2d_list, keypoints_3d_list)):
            # Handle batch dimension
            if len(kp2d.shape) == 3:
                num_people = kp2d.shape[0]
            else:
                num_people = 1
                kp2d = kp2d[np.newaxis, ...]
            
            if len(kp3d.shape) == 2:
                kp3d = kp3d[np.newaxis, ...]
            
            frame_kp2d = []
            frame_kp3d = []
            
            for person_idx in range(num_people):
                person_id = person_idx
                person_kp2d = kp2d[person_idx].copy()
                
                # Handle 3D - may have fewer keypoints
                if person_idx < kp3d.shape[0]:
                    person_kp3d = kp3d[person_idx].copy()
                else:
                    person_kp3d = np.zeros((17, 3))
                
                # Initialize filters for new person
                if person_id not in filters_2d:
                    num_kp_2d = person_kp2d.shape[0]
                    num_kp_3d = person_kp3d.shape[0]
                    
                    filters_2d[person_id] = [
                        [OneEuroFilter(freq=fps, mincutoff=5.0, beta=0.0, dcutoff=1.0) 
                         for _ in range(2)]
                        for _ in range(num_kp_2d)
                    ]
                    filters_3d[person_id] = [
                        [OneEuroFilter(freq=fps, mincutoff=5.0, beta=0.0, dcutoff=1.0) 
                         for _ in range(3)]
                        for _ in range(num_kp_3d)
                    ]
                
                timestamp = frame_idx / fps
                
                # Smooth 2D
                for kp_idx in range(person_kp2d.shape[0]):
                    for dim_idx in range(person_kp2d.shape[1]):
                        person_kp2d[kp_idx, dim_idx] = filters_2d[person_id][kp_idx][dim_idx](
                            person_kp2d[kp_idx, dim_idx], timestamp
                        )
                
                # Smooth 3D
                for kp_idx in range(person_kp3d.shape[0]):
                    for dim_idx in range(person_kp3d.shape[1]):
                        person_kp3d[kp_idx, dim_idx] = filters_3d[person_id][kp_idx][dim_idx](
                            person_kp3d[kp_idx, dim_idx], timestamp
                        )
                
                frame_kp2d.append(person_kp2d)
                frame_kp3d.append(person_kp3d)
            
            # THE FIX: Always append as an array to preserve the 'person' dimension
            smoothed_2d_list.append(np.array(frame_kp2d))
            smoothed_3d_list.append(np.array(frame_kp3d))
        
        return smoothed_2d_list, smoothed_3d_list


def pose_estimation(filepath_in: Path, apply_smoothing: bool = False):
    """
    Main entry point for pose estimation.
    
    Uses RTMDet-L + RTMW-L + MotionBERT pipeline.
    
    Args:
        filepath_in: Path to input video
        apply_smoothing: Whether to apply temporal smoothing
        
    Returns:
        Tuple of (keypoints_2d, keypoints_3d, keypoints_3d_before, scores)
        - keypoints_2d: List of [N, 133, 2] wholebody 2D keypoints
        - keypoints_3d: List of [N, 17, 3] body 3D keypoints
        - keypoints_3d_before: Same as keypoints_3d (no IK correction applied)
        - scores: List of [N, 17] confidence scores
    """
    pipeline = PoseEstimationPipeline()
    keypoints_2d, keypoints_3d, scores = pipeline.process_video(
        filepath_in, apply_smoothing=apply_smoothing
    )
    
    # For compatibility with existing code, return before/after (same for now)
    keypoints_3d_before = [kp.copy() for kp in keypoints_3d]
    
    return keypoints_2d, keypoints_3d, scores