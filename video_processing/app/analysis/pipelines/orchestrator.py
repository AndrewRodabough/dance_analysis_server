"""Pipeline Orchestrator - Coordinates all three analysis stages."""

import json
import logging
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Optional

try:
    from logging_config import log_job_status as worker_log_job_status
except ImportError:  # Backend context without worker logging
    worker_log_job_status = None

from .feature_extraction_pipeline import run_feature_extraction_pipeline
from .pose_estimation_pipeline import run_pose_estimation_pipeline
from .report_generation_pipeline import run_report_generation_pipeline

logger = logging.getLogger(__name__)


def run_analysis_pipeline(
    local_keypoints_2d_path: Optional[Path] = None,
    local_keypoints_3d_path: Optional[Path] = None,
    local_video_path: Optional[Path] = None,
    visualization_video_path: Optional[Path] = None,
    keypoints_2d_output_path: Optional[Path] = None,
    keypoints_3d_output_path: Optional[Path] = None,
    update_status: Optional[Callable[[str, int], None]] = None,
    job_id: Optional[str] = None,
) -> Dict:
    """
    Orchestrate the complete video analysis pipeline.

    Chains three stages together:
    1. Pose Estimation Pipeline - Load keypoints and create VectorizedPoseData objects
    2. Feature Extraction Pipeline - Analyze pose and calculate metrics from skeleton objects
    3. Report Generation Pipeline - Create reports and upload results

    Automatically handles skeleton conversion using coco_17 (2D) and human_17 (3D) formats.

    Args:
        local_keypoints_2d_path: Path to 2D keypoints JSON
        local_keypoints_3d_path: Path to 3D keypoints JSON
        local_video_path: Optional path to video file for keypoint generation
        visualization_video_path: Optional path to visualization video file
        keypoints_2d_output_path: Optional path to save 2D keypoints JSON after processing
        keypoints_3d_output_path: Optional path to save 3D keypoints JSON after processing
        update_status: Optional callback function(status: str, progress: int) for progress tracking
        job_id: Optional job identifier for structured logging and telemetry correlation

    Returns:
        Dictionary with complete analysis results including:
            - status: 'success' or error state
            - stage1_result: Pose estimation result with VectorizedPoseData objects
            - stage2_result: Feature extraction result
            - stage3_result: Report generation result
    """

    stage_start_times: Dict[str, float] = {}
    active_stage: Optional[str] = None

    def get_stage_duration(stage: str) -> Optional[float]:
        start = stage_start_times.get(stage)
        if start is None:
            return None
        return round((perf_counter() - start) * 1000, 2)

    def normalized_stage_name(stage: str) -> str:
        return stage.replace("_", "-")

    def log_stage(stage: str, status: str, **extra: Any) -> None:
        if status == "started":
            stage_start_times[stage] = perf_counter()
        elif status in {"completed", "failed"}:
            start_time = stage_start_times.pop(stage, None)
            if start_time is not None and "duration_ms" not in extra:
                extra["duration_ms"] = round((perf_counter() - start_time) * 1000, 2)

        log_data = {
            "event_type": "analysis_stage",
            "stage": stage,
            "status": status,
        }
        if job_id:
            log_data["job_id"] = job_id
        log_data.update(extra)
        logger.info(f"{stage} stage {status}", extra=log_data)

    def emit_job_status(
        stage: str,
        status: str,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        if not job_id or worker_log_job_status is None:
            return
        worker_log_job_status(
            job_id,
            status=status,
            stage=normalized_stage_name(stage),
            progress=progress,
            error=error,
            **extra,
        )

    def record_stage(
        stage: str,
        log_status: str,
        *,
        job_status: Optional[str] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        duration = get_stage_duration(stage)
        log_extra: Dict[str, Any] = {}
        if duration is not None:
            log_extra["duration_ms"] = duration
        if error:
            log_extra["error"] = error
        log_stage(stage, log_status, **log_extra)

        emit_extra: Dict[str, Any] = {}
        if duration is not None:
            emit_extra["duration_ms"] = duration
        status_value = job_status or log_status
        emit_job_status(
            stage,
            status_value,
            progress=progress,
            error=error,
            **emit_extra,
        )

    log_stage("analysis_pipeline", "started")
    emit_job_status("analysis_pipeline", "processing", progress=0)

    with tempfile.TemporaryDirectory() as temp_dir:

        try:
            # ============================================================================
            # STAGE 1: Pose Estimation Pipeline
            # ============================================================================

            active_stage = "pose_estimation"
            log_stage("pose_estimation", "started")
            if update_status:
                update_status('pose_estimation', 21)
            emit_job_status("pose_estimation", "processing", progress=21)

            stage1_result = run_pose_estimation_pipeline(
                local_keypoints_2d_path=local_keypoints_2d_path,
                local_keypoints_3d_path=local_keypoints_3d_path,
                local_video_path=local_video_path
            )

            pose_data_2d = stage1_result['pose_data_2d']
            pose_data_3d = stage1_result['pose_data_3d']
            record_stage("pose_estimation", "completed", job_status="processing", progress=55)
            active_stage = None

            # Save keypoints if output paths provided
            if keypoints_2d_output_path:
                keypoints_2d_output_path = Path(keypoints_2d_output_path)
                keypoints_2d_output_path.parent.mkdir(parents=True, exist_ok=True)
                keypoints_2d_json = pose_data_2d.skeleton.data.tolist()
                with open(keypoints_2d_output_path, 'w') as f:
                    json.dump(keypoints_2d_json, f)
                logger.info(f"Saved 2D keypoints to {keypoints_2d_output_path}")

            if keypoints_3d_output_path:
                keypoints_3d_output_path = Path(keypoints_3d_output_path)
                keypoints_3d_output_path.parent.mkdir(parents=True, exist_ok=True)
                keypoints_3d_json = pose_data_3d.skeleton.data.tolist()
                with open(keypoints_3d_output_path, 'w') as f:
                    json.dump(keypoints_3d_json, f)
                logger.info(f"Saved 3D keypoints to {keypoints_3d_output_path}")

            # ============================================================================
            # STAGE 2: Feature Extraction Pipeline
            # ============================================================================
            active_stage = "feature_extraction"
            log_stage("feature_extraction", "started")
            if update_status:
                update_status('feature_extraction', 70)
            emit_job_status("feature_extraction", "processing", progress=70)

            stage2_result = run_feature_extraction_pipeline(
                pose_data_2d=pose_data_2d,
                pose_data_3d=pose_data_3d,
            )
            record_stage("feature_extraction", "completed", job_status="processing", progress=85)
            active_stage = None

            # ============================================================================
            # STAGE 3: Report Generation Pipeline
            # ============================================================================

            active_stage = "report_generation"
            log_stage("report_generation", "started")
            if update_status:
                update_status('report_generation', 90)
            emit_job_status("report_generation", "processing", progress=90)

            stage3_result = run_report_generation_pipeline(
                pose_data_2d,
                pose_data_3d,
                stage2_result,
                local_video_path=local_video_path,
                visualization_video_path=visualization_video_path,
            )
            record_stage("report_generation", "completed", job_status="processing", progress=98)
            active_stage = None

            # ============================================================================
            # Results
            # ============================================================================

            final_result = {
                'status': 'success',
                'stage1_result': stage1_result,
                'stage2_result': stage2_result,
                'stage3_result': stage3_result,
                }
            record_stage("analysis_pipeline", "completed", job_status="completed", progress=100)

            return final_result

        except Exception as e:
            error_message = str(e)
            if active_stage:
                record_stage(active_stage, "failed", job_status="failed", error=error_message)
                active_stage = None
            record_stage("analysis_pipeline", "failed", job_status="failed", error=error_message)
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
