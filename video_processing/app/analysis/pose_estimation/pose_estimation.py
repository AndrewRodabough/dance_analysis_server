"""
Pose estimation module - delegates to the active pipeline.

Current pipeline: RTMDet-L + RTMW-L + MotionBERT
- RTMDet-L: Person detection
- RTMW-L: 2D wholebody pose estimation (133 keypoints)
- MotionBERT: 2D→3D pose lifting (17 body keypoints)
- OneEuroFilter: Temporal smoothing
"""

# Choose which pipeline to use
USE_MOTIONBERT_PIPELINE = True

if USE_MOTIONBERT_PIPELINE:
    from app.analysis.pose_estimation.pose_estimation_motionbert import (
        pose_estimation,
        PoseEstimationPipeline,
        ENABLE_ANATOMICAL_CONSTRAINTS
    )
else:
    # Legacy RTMPose3D pipeline
    from app.analysis.pose_estimation.pose_estimation_legacy import (
        pose_estimation,
        ENABLE_ANATOMICAL_CONSTRAINTS
    )

__all__ = ['pose_estimation', 'ENABLE_ANATOMICAL_CONSTRAINTS']
