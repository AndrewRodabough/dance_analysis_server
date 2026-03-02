"""Utilities for scaling 2D pose measurements based on distance from camera."""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Calibration constant: Assumed average human height in meters
# This is used to calculate the pixels-per-meter ratio from detected keypoints
ASSUMED_HUMAN_HEIGHT_M = 1.7

# Baseline pixels-per-meter: Represents the ground truth scale calibration
# from the original test video. Measured empirically from actual test video.
# The reference test video has person at known distance with detected height.
# When compute_pixel_scale() is called on the reference video, it returns this value.
# Set conservatively at ~180 px/m so that the reference video gets scaling ~0.9x
# (slightly more strict), and farther distances get progressively more lenient.
BASELINE_PIXELS_PER_METER = 180.0


def compute_person_height_pixels(
    skeleton_2d_data: np.ndarray,
    top_joint_idx: int = 0,  # COCO-17: 0=nose (head)
    bottom_joint_idx: int = 16  # COCO-17: 16=right_ankle
) -> Tuple[float, int]:
    """
    Estimate person height in pixels by measuring the distance from a top keypoint
    (head/nose) to a bottom keypoint (ankle) across all frames.
    
    Uses the median height over frames to handle transient poses and outliers.
    
    Args:
        skeleton_2d_data: Numpy array of shape (Frames, Joints, 2) with 2D coordinates
        top_joint_idx: Joint index for top of body (default: COCO-17 nose = 0)
        bottom_joint_idx: Joint index for bottom of body (default: COCO-17 right_ankle = 16)
        
    Returns:
        Tuple of (median_height_pixels, valid_frame_count)
        Returns (0.0, 0) if insufficient valid frames
    """
    num_frames = skeleton_2d_data.shape[0]
    heights = []
    
    for frame_idx in range(num_frames):
        top_point = skeleton_2d_data[frame_idx, top_joint_idx, :]  # Shape: (2,)
        bottom_point = skeleton_2d_data[frame_idx, bottom_joint_idx, :]  # Shape: (2,)
        
        # Skip if either point is invalid (all zeros or NaN)
        if np.any(np.isnan(top_point)) or np.any(np.isnan(bottom_point)):
            continue
        if np.allclose(top_point, 0) or np.allclose(bottom_point, 0):
            continue
        
        # Calculate vertical distance (Y-axis in image space)
        # Use absolute value in case bottom point has lower Y (different coordinate systems)
        height_px = np.abs(bottom_point[1] - top_point[1])
        
        # Filter out unreasonable heights (< 10 pixels or > image height)
        if height_px >= 10:  # Minimum sensible height
            heights.append(height_px)
    
    if not heights:
        logger.warning(
            f"Could not compute person height: no valid frames found. "
            f"Top joint {top_joint_idx}, bottom joint {bottom_joint_idx}"
        )
        return 0.0, 0
    
    # Use median to be robust to outliers and transient poses
    median_height = np.median(heights)
    
    logger.debug(
        f"Person height estimation: median={median_height:.1f}px from {len(heights)} frames "
        f"(min={np.min(heights):.1f}, max={np.max(heights):.1f})"
    )
    
    return float(median_height), len(heights)


def compute_pixel_scale(
    skeleton_2d_data: np.ndarray,
    assumed_height_m: float = ASSUMED_HUMAN_HEIGHT_M,
    top_joint_idx: int = 0,
    bottom_joint_idx: int = 16
) -> float:
    """
    Compute pixels-per-meter scale factor by estimating person height from keypoints.
    
    This allows the analysis to be invariant to the distance of the person from the camera.
    A person closer to the camera appears taller in pixels, but this is accounted for by
    the higher pixel-per-meter ratio.
    
    Args:
        skeleton_2d_data: Numpy array of shape (Frames, Joints, 2) with 2D keypoints
        assumed_height_m: Assumed human height in meters (default 1.7m)
        top_joint_idx: Index of top joint (default: COCO-17 nose = 0)
        bottom_joint_idx: Index of bottom joint (default: COCO-17 right_ankle = 16)
        
    Returns:
        Pixels-per-meter ratio. Can be used to scale distance measurements and velocities.
        Returns BASELINE_PIXELS_PER_METER if height cannot be computed.
    """
    height_px, valid_count = compute_person_height_pixels(
        skeleton_2d_data,
        top_joint_idx=top_joint_idx,
        bottom_joint_idx=bottom_joint_idx
    )
    
    if height_px <= 0 or valid_count < 1:
        logger.warning(
            f"Could not compute pixel scale from person height. "
            f"Defaulting to baseline {BASELINE_PIXELS_PER_METER:.1f} px/m"
        )
        return BASELINE_PIXELS_PER_METER
    
    # Convert person height in pixels to pixels-per-meter
    pixels_per_meter = height_px / assumed_height_m
    
    logger.info(
        f"Computed 2D scale: {pixels_per_meter:.1f} px/m "
        f"(height={height_px:.1f}px / {assumed_height_m}m, "
        f"baseline={BASELINE_PIXELS_PER_METER:.1f} px/m, "
        f"scaling_factor={pixels_per_meter/BASELINE_PIXELS_PER_METER:.2f}x)"
    )
    
    return pixels_per_meter


def compute_fps_normalization_factor(fps: float, baseline_fps: float = 60.0) -> float:
    """
    Compute FPS normalization factor to make velocities independent of frame rate.
    
    Velocities are computed as position differences between frames. A person moving at
    the same real-world speed will show different pixel velocities in 30fps vs 60fps video.
    This factor corrects for that.
    
    Args:
        fps: Frame rate of the current video
        baseline_fps: Baseline FPS used for threshold calibration (default 60)
        
    Returns:
        Normalization factor to multiply velocity by. 
        For 30fps video, returns 2.0 (doubled velocity). 
        For 120fps video, returns 0.5 (halved velocity).
    """
    if fps <= 0:
        logger.warning(f"Invalid FPS {fps}, using baseline {baseline_fps}")
        return 1.0
    
    factor = baseline_fps / fps
    
    if factor != 1.0:
        logger.debug(
            f"FPS normalization: {fps}fps -> factor {factor:.2f}x "
            f"(baseline {baseline_fps}fps)"
        )
    
    return factor
