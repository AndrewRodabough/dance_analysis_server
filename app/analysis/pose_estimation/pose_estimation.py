import os
import sys
import math
import numpy as np
import cv2
import yaml
from pathlib import Path

from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample
from mmpose.apis import init_model as init_pose_model
from mmpose.apis import inference_topdown
from mmpose.utils import register_all_modules
import torch


class PoseEstimationPipeline:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.detection_model = init_detector(
            self.config['detection_tracking']['config'],
            self.config['detection_tracking']['checkpoint'],
            device='cuda:0')
        
        self.pose_model = init_pose_model(
            self.config['pose_estimation']['config'],
            self.config['pose_estimation']['checkpoint'],
            device='cuda:0')
        
        self.pose_history = {}  # track_id -> list of keypoints for temporal smoothing

    def pose_estimation(self, filepath_in: str):
        
        estimation_2d = []
        estimation_3d = []
        register_all_modules()

        # Open video
        cap = self._open_video(filepath_in)
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process video frames
        frame_idx = 0
        while cap.isOpened():

            # Read frame
            ret, frame = cap.read()
            if not ret: break

            # Estimate poses
            estimation_2d_frame, estimation_3d_frame = self._process_frame(frame, frame_idx, video_len)
            
            # Store results
            estimation_2d.append(estimation_2d_frame)
            estimation_3d.append(estimation_3d_frame)
            frame_idx += 1
        
        cap.release()
        return estimation_2d, estimation_3d
    
    def _open_video(self, filepath_in):
        if not os.path.exists(filepath_in):
            print(f"[ERROR] File not found at: {filepath_in}")
            sys.exit(1)

        cap = cv2.VideoCapture(filepath_in)

        if not cap.isOpened():
            print(f"[ERROR] File exists, but OpenCV cannot open it: {filepath_in}")
            print("Possible causes: Wrong codec, corrupt file, or missing ffmpeg dependencies.")
            sys.exit(1)
        
        return cap
    
    def _process_frame(self, frame, frame_idx, video_len):

        # Detection & Tracking        
        track_results = self._detect_track(frame, frame_idx, video_len)

        # 2d pose estimation
        estimation_2d = self._estimate(frame, track_results)
        
        # Smoothing 2d
        estimation_2d_smooth = self._smooth_poses(estimation_2d)

        # Lifting 2d->3d (placeholder)
        estimation_3d = None  # Placeholder for lifting function

        # Smoothing 3d (placeholder)
        estimation_3d_smooth = None  # Placeholder for smoothing function

        return estimation_2d_smooth, estimation_3d_smooth
    
    def _detect_track(self, frame, frame_idx, video_len):
        # Prepare the input for detection model
        # The model expects data in specific format with 'inputs' key
        h, w = frame.shape[:2]
        
        # Resize frame to model input size while keeping aspect ratio
        target_size = 640
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to square
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to tensor: HWC -> CHW, BGR stays BGR (model expects BGR)
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0).to('cuda:0')
        
        # Create data sample with image metadata
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': (target_size, target_size),
            'ori_shape': (h, w),
            'scale_factor': (scale, scale),
            'pad_shape': (target_size, target_size),
        })
        
        # Prepare batch data in the format expected by the model
        data = {
            'inputs': img_tensor,
            'data_samples': [data_sample]
        }
        
        # Run inference
        with torch.no_grad():
            # Use data_preprocessor to normalize
            data = self.detection_model.data_preprocessor(data, False)
            results = self.detection_model._run_forward(data, mode='predict')
        
        if results and len(results) > 0:
            result = results[0]
            pred_instances = result.pred_instances
            
            if hasattr(pred_instances, 'scores') and len(pred_instances.scores) > 0:
                # Check if bboxes are already in original space or scaled space
                bboxes = pred_instances.bboxes.clone()
                
                # Debug: bboxes data on first frame
                if (self.config.get('debug', {}) .get('console_logs', True)):
                    if frame_idx == 0:  # Debug first frame
                        print(f"Raw bboxes from model: {bboxes.cpu().numpy()}")
                        print(f"Scale factor: {scale}, Frame size: {w}x{h}")
                
                # Expand bboxes by configured scale factor to avoid clipping body parts
                bbox_w = bboxes[:, 2] - bboxes[:, 0]
                bbox_h = bboxes[:, 3] - bboxes[:, 1]
                expand_ratio = self.config['detection_tracking']['bbox_scale'] - 1
                bboxes[:, 0] -= bbox_w * expand_ratio
                bboxes[:, 1] -= bbox_h * expand_ratio
                bboxes[:, 2] += bbox_w * expand_ratio
                bboxes[:, 3] += bbox_h * expand_ratio
                
                # Clip bboxes to frame boundaries after expansion
                bboxes[:, [0, 2]] = torch.clamp(bboxes[:, [0, 2]], 0, w)
                bboxes[:, [1, 3]] = torch.clamp(bboxes[:, [1, 3]], 0, h)
                
                # Debug: clipped bboxes data on first frame
                if (self.config.get('debug', {}) .get('console_logs', True)):
                    if frame_idx == 0:  # Debug first frame
                        print(f"Bboxes after clipping: {bboxes.cpu().numpy()}")
                
                # Filter by confidence and only keep person class (label 0)
                conf_thresh = self.config['detection_tracking']['confidence_threshold']
                person_mask = (pred_instances.labels == 0) & (pred_instances.scores > conf_thresh)
                
                if person_mask.any():
                    tracks = InstanceData()
                    tracks.bboxes = bboxes[person_mask].cpu().numpy()
                    tracks.scores = pred_instances.scores[person_mask].cpu().numpy()
                    tracks.labels = pred_instances.labels[person_mask].cpu().numpy()
                    tracks.track_ids = np.arange(len(tracks.bboxes))
                    return tracks
        
        return None

    def _estimate(self, frame, track_results):
        if track_results is None:
            return []

        # Debug: check bboxes data on first frame
        if (self.config.get('debug', {}) .get('console_logs', True)):
            if hasattr(self, '_first_estimate'):
                pass
            else:
                self._first_estimate = True
                print(f"Frame shape: {frame.shape}")
                print(f"Number of detected people: {len(track_results.bboxes)}")
                print(f"Bboxes: {track_results.bboxes}")
                print(f"Bbox scores: {track_results.scores}")

        # 2D Pose Estimation
        pose_results = inference_topdown(self.pose_model, frame, track_results.bboxes)

        # Parse results
        poses_2d = []
        for i, res in enumerate(pose_results):
            # Handle both tensor and numpy array cases
            keypoints = res.pred_instances.keypoints
            scores = res.pred_instances.keypoint_scores
            
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            
            # Remove batch dimension if present
            if len(keypoints.shape) == 3:
                keypoints = keypoints[0]  # (1, 133, 2) -> (133, 2)
            if len(scores.shape) == 2:
                scores = scores[0]  # (1, 133) -> (133,)
            
            # Debug: check keypoints data on first frame
            if (self.config.get('debug', {}) .get('console_logs', True)):
                if not hasattr(self, '_first_pose_debug'):
                    self._first_pose_debug = True
                    print(f"Raw keypoints from model shape: {res.pred_instances.keypoints.shape}")
                    print(f"After processing keypoints shape: {keypoints.shape}")
                    print(f"Sample keypoints: {keypoints[:5]}")
                    print(f"Sample scores: {scores[:5]}")
                    
            poses_2d.append({
                'track_id': int(track_results.track_ids[i]),
                'keypoints': keypoints,
                'scores': scores,
                'bbox': track_results.bboxes[i] if i < len(track_results.bboxes) else None
            })

        return poses_2d
        
    def _smooth_poses(self, poses):
        if not self.config['smoothing']['enabled'] or not poses:
            return poses
        
        smoothed = []
        window_size = self.config['smoothing'].get('window_size', 5)
        
        for person in poses:
            track_id = person['track_id']
            
            # Store in history
            if track_id not in self.pose_history:
                self.pose_history[track_id] = []
            self.pose_history[track_id].append(person['keypoints'])
            
            # Keep only recent frames
            if len(self.pose_history[track_id]) > window_size:
                self.pose_history[track_id].pop(0)
            
            # Apply temporal smoothing (moving average)
            history = self.pose_history[track_id]
            smoothed_keypoints = np.mean(history, axis=0)
            
            smoothed.append({
                **person,
                'keypoints': smoothed_keypoints
            })
        
        return smoothed


def pose_estimation(filepath_in: str):

    pipeline = PoseEstimationPipeline()
    return pipeline.pose_estimation(filepath_in)