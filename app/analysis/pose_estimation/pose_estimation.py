import os
import sys
import math
import numpy as np
import cv2
import yaml
from pathlib import Path

from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
from mmdet.apis import init_detector, inference_mot
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

        self.detection_model = init_detector(self.config['detection_tracking']['config'],
                                            self.config['detection_tracking']['checkpoint'],
                                            device='cuda:0')
        self.pose_model = init_pose_model(self.config['pose_estimation']['config'],
                                          self.config['pose_estimation']['checkpoint'],
                                          device='cuda:0')
        self.pose_history = {}  # track_id -> list of keypoints for temporal smoothing

    def pose_estimation(self, filepath_in: str):
        
        estimation_2d = []
        estimation_3d = []
        register_all_modules()

        # Open video
        cap = self._open_video(filepath_in) 

        # Process video frames
        frame_idx = 0
        while cap.isOpened():

            # Read frame
            ret, frame = cap.read()
            if not ret: break

            estimation_2d_frame, estimation_3d_frame = self._process_frame(frame, frame_idx)
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
    


    def _process_frame(self, frame, frame_idx):

        # Detection & Tracking        
        track_results = self._detect_track(frame, frame_idx)

        # 2d pose estimation
        estimation_2d = self._estimate(frame, track_results)
        
        # Smoothing 2d
        estimation_2d_smooth = self._smooth_poses(estimation_2d)

        # Lifting 2d->3d (placeholder)
        estimation_3d = None  # Placeholder for lifting function

        return estimation_2d_smooth, estimation_3d
    

    def _detect_track(self, frame, frame_idx):
        # Bounding boxes & tracking
        track_results = inference_mot(self.detection_model, frame, frame_id=frame_idx)
        pred_instances = track_results.pred_track_instances
        track_mask = pred_instances.scores > self.config['detection_tracking']['track_thresh'] # filter low confidence tracks

        if track_mask.any():
            tracks = InstanceData()
            tracks.bboxes = pred_instances.bboxes[track_mask].cpu().numpy()
            tracks.scores = pred_instances.scores[track_mask].cpu().numpy()
            tracks.labels = pred_instances.labels[track_mask].cpu().numpy()
            tracks.track_ids = pred_instances.track_ids[track_mask].cpu().numpy()
            return tracks
        
        return None

    def _estimate(self, frame, track_results):
        if track_results is None:
            return []

        pose_results = inference_topdown(self.pose_model, frame, track_results.bboxes)

        poses_2d = []
        for i, res in enumerate(pose_results):
            poses_2d.append({
                'track_id': int(track_results.track_ids[i]),
                'keypoints': res.pred_instances.keypoints.cpu().numpy(),
                'scores': res.pred_instances.keypoint_scores.cpu().numpy()
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