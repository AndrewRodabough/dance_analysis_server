"""Orchestrate the feedback generation pipeline."""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple

from rq import get_current_job

from .keypoint_loading import (
    load_keypoints_from_json,
    create_skeleton_objects,
    create_pose_data_objects,
    create_all_skeleton_objects,
)
from .visualization_generator import generate_visualization_video
from .feature_extraction.feature_extraction import extract_features
from .heuristics_judge import judge_heuristics
from .score_calculation import calculate_scores
from .report_generator import generate_report
from .results_uploader import upload_results

logger = logging.getLogger(__name__)


def run_feedback_pipeline(
    job_id: str,
    s3_bucket: str,
    s3_client,
    local_keypoints_2d_path: Path,
    local_keypoints_3d_path: Path,
    local_video_path: Path = None,
    person_idx: int = 0,
) -> Dict:
    """
    Orchestrate the complete feedback generation pipeline.
    
    Supports both single-person and multi-person analysis.
    For multi-person videos, call this function once per person (varying person_idx).
    
    Coordinates all analysis steps:
    1. Load keypoints and create skeletons
    2. Generate visualization video
    3. Extract features
    4. Judge heuristics
    5. Calculate scores
    6. Generate report
    7. Upload results to S3
    
    Args:
        job_id: Unique job identifier
        s3_bucket: S3 bucket name
        s3_client: Boto3 S3 client
        local_keypoints_2d_path: Path to 2D keypoints JSON
        local_keypoints_3d_path: Path to 3D keypoints JSON
        local_video_path: Optional path to video file
        person_idx: Index of person to analyze (default 0, first person)
        
    Returns:
        Dictionary with results summary and S3 paths
    """
    job = get_current_job()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            logger.info(f"Starting feedback pipeline for job {job_id}, person {person_idx}")
            
            # Step 1: Load Keypoints
            job.meta['status'] = 'Loading keypoints'
            job.meta['progress'] = 10
            job.save_meta()
            
            keypoints_2d, keypoints_3d, num_people = load_keypoints_from_json(
                local_keypoints_2d_path,
                local_keypoints_3d_path
            )
            
            logger.info(f"Loaded {num_people} people from keypoints")
            
            # Validate person index
            if person_idx >= num_people:
                raise ValueError(f"Person index {person_idx} out of range [0, {num_people})")
            
            # Step 2: Create skeleton objects for specified person
            job.meta['status'] = 'Loading skeleton configurations'
            job.meta['progress'] = 15
            job.save_meta()
            
            skeleton_2d, skeleton_3d = create_skeleton_objects(
                keypoints_2d,
                keypoints_3d,
                person_idx=person_idx
            )
            
            pose_data_2d, pose_data_3d = create_pose_data_objects(
                skeleton_2d,
                skeleton_3d
            )
            
            # Step 3: Generate Visualization Video
            job.meta['status'] = 'Generating visualization video'
            job.meta['progress'] = 20
            job.save_meta()
            
            viz_video_path = temp_path / "visualization.mp4"
            generate_visualization_video(local_video_path, viz_video_path)
            
            # Step 4: Extract Features
            job.meta['status'] = 'Extracting features'
            job.meta['progress'] = 30
            job.save_meta()
            
            features = extract_features(pose_data_3d)
            features_path = temp_path / "features.txt"
            
            import json
            with open(features_path, 'w') as f:
                json.dump(features, f, indent=2)
            
            # Step 5: Judge Heuristics
            job.meta['status'] = 'Judging heuristics'
            job.meta['progress'] = 60
            job.save_meta()
            
            judge = judge_heuristics(pose_data_2d, features)
            judge_path = temp_path / "judge.json"
            
            with open(judge_path, 'w') as f:
                json.dump(judge, f, indent=2)
            
            # Step 6: Calculate Scores
            job.meta['status'] = 'Calculating scores'
            job.meta['progress'] = 80
            job.save_meta()
            
            scores = calculate_scores(judge, features)
            scores_path = temp_path / "scores.json"
            
            with open(scores_path, 'w') as f:
                json.dump(scores, f, indent=2)
            
            # Step 7: Generate Report
            job.meta['status'] = 'Generating report'
            job.meta['progress'] = 90
            job.save_meta()
            
            feedback_text = generate_report(scores, judge)
            feedback_path = temp_path / "feedback.txt"
            
            with open(feedback_path, 'w') as f:
                f.write(feedback_text)
            
            # Step 8: Upload Results to S3
            job.meta['status'] = 'Uploading results to S3'
            job.meta['progress'] = 95
            job.save_meta()
            
            s3_results = upload_results(
                s3_client,
                f"{job_id}_person_{person_idx}",  # Append person index to job ID
                s3_bucket,
                keypoints_2d_path=local_keypoints_2d_path,
                keypoints_3d_path=local_keypoints_3d_path,
                feedback_path=feedback_path,
                scores_path=scores_path,
                features_path=features_path,
                judge_path=judge_path,
                viz_video_path=viz_video_path,
            )
            
            # Mark complete
            job.meta['status'] = 'Complete'
            job.meta['progress'] = 100
            job.save_meta()
            
            result = {
                'status': 'success',
                'person_idx': person_idx,
                'num_people': num_people,
                'num_frames': keypoints_2d.shape[0],
                's3_results': s3_results
            }
            
            logger.info(f"Feedback pipeline complete for job {job_id}, person {person_idx}")
            return result
            
        except Exception as e:
            logger.error(f"Error in feedback pipeline: {e}", exc_info=True)
            job.meta['status'] = 'Failed'
            job.meta['error'] = str(e)
            job.save_meta()
            raise


def process_all_people(
    job_id: str,
    s3_bucket: str,
    s3_client,
    local_keypoints_2d_path: Path,
    local_keypoints_3d_path: Path,
    local_video_path: Path = None,
) -> Dict:
    """
    Process all people in a multi-person video by running the pipeline for each person.
    
    Args:
        job_id: Base job identifier (will be appended with person index)
        s3_bucket: S3 bucket name
        s3_client: Boto3 S3 client
        local_keypoints_2d_path: Path to 2D keypoints JSON
        local_keypoints_3d_path: Path to 3D keypoints JSON
        local_video_path: Optional path to video file
        
    Returns:
        Dictionary with results for all people
    """
    # Load to get number of people
    keypoints_2d, keypoints_3d, num_people = load_keypoints_from_json(
        local_keypoints_2d_path,
        local_keypoints_3d_path
    )
    
    logger.info(f"Processing {num_people} people in job {job_id}")
    
    results = {
        'status': 'success',
        'job_id': job_id,
        'num_people': num_people,
        'people_results': {}
    }
    
    for person_idx in range(num_people):
        try:
            person_result = run_feedback_pipeline(
                job_id,
                s3_bucket,
                s3_client,
                local_keypoints_2d_path,
                local_keypoints_3d_path,
                local_video_path,
                person_idx=person_idx
            )
            results['people_results'][person_idx] = person_result
        except Exception as e:
            logger.error(f"Failed to process person {person_idx}: {e}")
            results['people_results'][person_idx] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results
