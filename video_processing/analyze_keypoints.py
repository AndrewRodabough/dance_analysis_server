#!/usr/bin/env python3
"""
Offline Keypoint Analysis Script

Runs the analysis pipeline directly on pre-computed 2D and 3D keypoints,
skipping the pose estimation stage. Ideal for local analysis of keypoints
that were previously generated or obtained from another source.

Usage:
    python analyze_keypoints.py --keypoints-2d path/to/2d.json --keypoints-3d path/to/3d.json [--video path/to/video.mp4] [--output-dir /path/to/outputs]

Example:
    python analyze_keypoints.py \\
        --keypoints-2d outputs/job_id/keypoints_2d.json \\
        --keypoints-3d outputs/job_id/keypoints_3d.json \\
        --video my_dance_video.mp4 \\
        --output-dir ./analysis_results
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processing.app.analysis.pipelines.orchestrator import run_analysis_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_update_status_callback(job_id: str):
    """
    Create a console-based status update callback.
    
    Args:
        job_id: Job identifier for logging
        
    Returns:
        Callback function(status: str, progress: int) that logs to console
    """
    def update_status(status: str, progress: int):
        print(f"[{job_id}] {status}: {progress}%")
        logger.info(f"[{job_id}] {status}: {progress}%")
    
    return update_status


def validate_input_files(keypoints_2d_path: Path, keypoints_3d_path: Path, video_path: Optional[Path]) -> bool:
    """
    Validate that all input files exist.
    
    Args:
        keypoints_2d_path: Path to 2D keypoints JSON
        keypoints_3d_path: Path to 3D keypoints JSON
        video_path: Optional path to video file
        
    Returns:
        True if all provided files exist, False otherwise
    """
    files_to_check = [
        (keypoints_2d_path, "2D keypoints"),
        (keypoints_3d_path, "3D keypoints"),
    ]
    
    if video_path:
        files_to_check.append((video_path, "Video"))
    
    all_valid = True
    for file_path, description in files_to_check:
        if file_path and not file_path.exists():
            logger.error(f"✗ {description} file not found: {file_path}")
            all_valid = False
        else:
            logger.info(f"✓ {description} file found: {file_path}")
    
    return all_valid


def main():
    """Main entry point for offline keypoint analysis."""
    parser = argparse.ArgumentParser(
        description="Offline Keypoint Analysis - Run analysis pipeline on pre-computed keypoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with keypoints and optional video
  python analyze_keypoints.py \\
    --keypoints-2d keypoints_2d.json \\
    --keypoints-3d keypoints_3d.json \\
    --video input_video.mp4

  # Without video, save results to specific directory
  python analyze_keypoints.py \\
    --keypoints-2d outputs/job_id/keypoints_2d.json \\
    --keypoints-3d outputs/job_id/keypoints_3d.json \\
    --output-dir ./my_analysis_results
        """
    )
    
    parser.add_argument(
        '--keypoints-2d',
        type=Path,
        required=True,
        help='Path to 2D keypoints JSON file'
    )
    
    parser.add_argument(
        '--keypoints-3d',
        type=Path,
        required=True,
        help='Path to 3D keypoints JSON file'
    )
    
    parser.add_argument(
        '--video',
        type=Path,
        default=None,
        help='Optional path to video file for visualization overlay'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for results (default: auto-generated UUID-based directory in ./outputs)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    logger.info("Validating input files...")
    if not validate_input_files(args.keypoints_2d, args.keypoints_3d, args.video):
        logger.error("Input validation failed. Exiting.")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        job_id = str(uuid.uuid4())
        output_dir = Path("outputs") / job_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {output_dir.resolve()}")
    
    # Generate job ID for logging
    job_id = output_dir.name
    
    # Prepare visualization video path
    visualization_video_path = output_dir / "video_visualization.mp4"
    
    # Prepare keypoint output paths
    keypoints_2d_output_path = output_dir / "keypoints_2d.json"
    keypoints_3d_output_path = output_dir / "keypoints_3d.json"
    
    # Create status callback
    update_status = create_update_status_callback(job_id)
    
    try:
        logger.info(f"Starting analysis pipeline for job {job_id}...")
        update_status("initialization", 0)
        
        # Run the analysis pipeline
        result = run_analysis_pipeline(
            local_keypoints_2d_path=args.keypoints_2d,
            local_keypoints_3d_path=args.keypoints_3d,
            local_video_path=args.video,
            visualization_video_path=visualization_video_path,
            keypoints_2d_output_path=keypoints_2d_output_path,
            keypoints_3d_output_path=keypoints_3d_output_path,
            update_status=update_status,
        )
        
        update_status("complete", 100)
        
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Output directory: {output_dir.resolve()}")
        logger.info(f"Status: {result.get('status', 'unknown')}")
        logger.info("=" * 80)
        
        # Print result summary
        if result.get('status') == 'success':
            logger.info("✓ Pipeline executed successfully")
            
            # List generated files
            generated_files = list(output_dir.glob("*"))
            if generated_files:
                logger.info("Generated files:")
                for file in generated_files:
                    logger.info(f"  - {file.name}")
            
            return 0
        else:
            logger.error("✗ Pipeline failed")
            return 1
            
    except Exception as e:
        logger.error(f"✗ Analysis failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
