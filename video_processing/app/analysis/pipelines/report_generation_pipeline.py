"""Report Generation Pipeline - Stage 3: Create reports, visualizations, and upload results."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from ..report_generation.video_generation import generate_side_by_side_video
from ..report_generation.report_generation import generate_report
from shared.skeletons.pose_data import VectorizedPoseData

logger = logging.getLogger(__name__)


def run_report_generation_pipeline(
    pose_2d: VectorizedPoseData,
    pose_3d: VectorizedPoseData,
    features: Dict,
    local_video_path: Optional[Path] = None,
    visualization_video_path: Optional[Path] = None,
) -> Dict:
    """
    Stage 3: Report Generation Pipeline
    
    Generates visualization videos, feedback reports, and uploads all results to S3.
    
    Args:
        pose_2d: VectorizedPoseData object from Stage 1
        pose_3d: VectorizedPoseData object from Stage 1
        features: Features dictionary from Stage 2
        local_video_path: Optional path to original video for visualization
        visualization_video_path: Optional path to save visualization video
        
    Returns:
        Dictionary containing:
            - report: Generated report structure
            - report_path: Path to saved report.json (if available)
    """
    logger.info(f"[STAGE 3] Report Generation Pipeline: Creating reports")
    
    try:
        features = features or {}

        # Step 1: Generate Visualization Video
        if local_video_path and local_video_path.exists() and visualization_video_path:
            logger.debug("Generating visualization video")
            
            try:
                generate_side_by_side_video(str(local_video_path), pose_2d, pose_3d, str(visualization_video_path))
                logger.info(f"✓ Visualization video created: {visualization_video_path}")
            except Exception as e:
                logger.warning(f"Could not generate visualization video: {e}")
        else:
            logger.debug("Skipping visualization: video path not provided")
        
        # Step 2: Generate Feedback Report
        logger.debug("Generating feedback report")
        report = generate_report(
            analysis_results=features,
            total_frames=pose_2d.num_frames,
            frame_rate=60,
            person_id=0,
        )

        report_path = None
        if visualization_video_path:
            output_dir = Path(visualization_video_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / "report.json"
            report_path.write_text(json.dumps(report, indent=4))
            logger.info(f"✓ Report saved: {report_path}")

            analysis_path = output_dir / "analysis_results.json"
            analysis_path.write_text(json.dumps(features, indent=4))
            logger.info(f"✓ Analysis results saved: {analysis_path}")

        logger.info("✓ Feedback report generated")

        return {
            "report": report,
            "report_path": str(report_path) if report_path else None,
        }
    
    except Exception as e:
        logger.error(f"[STAGE 3] ✗ Report generation failed: {e}", exc_info=True)
        raise
