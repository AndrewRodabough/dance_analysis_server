"""Video analysis endpoints - upload, queue, and status tracking."""

from fastapi import APIRouter, File, UploadFile, HTTPException, status, Query
from typing import Dict, Any
import uuid
import os
import logging
from pathlib import Path

from redis import Redis
from rq import Queue
from rq.job import Job
import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

# S3 Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "dance-videos")
# Public endpoint for presigned URLs (accessible from outside Docker)
S3_PUBLIC_ENDPOINT = os.getenv("S3_PUBLIC_ENDPOINT", "http://localhost:9000")

# Initialize S3 client (for internal operations)
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Initialize S3 client for presigned URLs (with public endpoint)
s3_client_public = boto3.client(
    's3',
    endpoint_url=S3_PUBLIC_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Redis/RQ setup
redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
pose_queue = Queue('pose-estimation', connection=redis_conn)
analysis_queue = Queue('analysis', connection=redis_conn)


@router.post("/upload-url", summary="Request presigned upload URL", status_code=status.HTTP_200_OK)
async def request_upload_url(
    filename: str = Query(..., description="Name of the video file"),
    content_type: str = Query("video/mp4", description="MIME type of the video")
) -> Dict[str, Any]:
    """
    Request a presigned URL for direct upload to S3.
    
    This enables clients to upload videos directly to S3 without going through the API server,
    reducing server load and improving upload performance.
    
    **Steps:**
    1. Call this endpoint to get upload URL and job_id
    2. PUT the video file to the upload_url
    3. Call /analyze/confirm with the job_id to start processing
    
    **Args:**
    - filename: Name of the video file (e.g., "dance_video.mp4")
    - content_type: MIME type of the video (default: "video/mp4")
    
    **Returns:**
    - upload_url: Presigned URL to upload the video (valid for 15 minutes)
    - job_id: Unique identifier for tracking this analysis
    - s3_key: S3 object key where the file will be stored
    """
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov'}
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # S3 key structure: uploads/{job_id}/{filename}
    s3_key = f"uploads/{job_id}/{filename}"
    
    try:
        # Generate presigned URL for direct upload (valid for 15 minutes)
        # Use public endpoint so clients outside Docker can access it
        upload_url = s3_client_public.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': s3_key,
                'ContentType': content_type
            },
            ExpiresIn=900,  # 15 minutes
            HttpMethod='PUT'
        )
        
        logger.info(f"Generated presigned upload URL for job {job_id}: {s3_key}")
        
        return {
            "job_id": job_id,
            "upload_url": upload_url,
            "s3_key": s3_key,
            "expires_in": 900,
            "instructions": "PUT the video file to upload_url, then call /analyze/confirm with job_id"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate upload URL: {str(e)}"
        )


@router.post("/confirm", summary="Confirm upload and start analysis", status_code=status.HTTP_202_ACCEPTED)
async def confirm_upload_and_start_analysis(
    job_id: str = Query(..., description="Job ID from upload-url endpoint"),
    s3_key: str = Query(..., description="S3 key from upload-url endpoint")
) -> Dict[str, str]:
    """
    Confirm that a video has been uploaded to S3 and start the analysis pipeline.
    
    Call this endpoint after successfully uploading the video using the presigned URL
    from /analyze/upload-url.
    
    **Args:**
    - job_id: The job ID received from /analyze/upload-url
    - s3_key: The S3 key received from /analyze/upload-url
    
    **Returns:**
    - Job details and queuing status
    """
    try:
        # Verify the object exists in S3
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        except Exception as e:
            logger.error(f"S3 object not found: {s3_key}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video not found in S3. Please upload the file first using the presigned URL."
            )
        
        # Queue pose estimation job (GPU stage)
        job = pose_queue.enqueue(
            'tasks.extract_keypoints',
            args=[s3_key, job_id],
            kwargs={'options': {'apply_smoothing': True}},
            job_id=job_id,
            job_timeout='1h',
            result_ttl=86400  # Keep results for 24 hours
        )
        
        logger.info(f"Job {job_id} queued for pose estimation (GPU stage)")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "stage": "pose-estimation",
            "s3_key": s3_key,
            "message": "Video confirmed and queued for pose estimation"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start analysis: {str(e)}"
        )


@router.post("", summary="Submit video for analysis (legacy)", status_code=status.HTTP_202_ACCEPTED)
async def submit_video_analysis(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov)")
) -> Dict[str, str]:
    """
    Upload a dance video for pose estimation analysis.
    
    **DEPRECATED**: This endpoint uploads videos through the API server.
    For better performance, use the direct upload flow:
    1. POST /analyze/upload-url to get presigned URL
    2. PUT video to the presigned URL
    3. POST /analyze/confirm to start processing
    
    The video is uploaded to S3 and queued for processing.
    Returns a job_id to track analysis progress.
    
    **Supported formats**: MP4, AVI, MOV
    """
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # S3 key structure: uploads/{job_id}/{filename}
    s3_key = f"uploads/{job_id}/{file.filename}"
    
    try:
        # Upload video to S3
        file_content = await file.read()
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType=file.content_type or 'video/mp4'
        )
        
        logger.info(f"Video uploaded to S3: {s3_key} ({len(file_content)} bytes)")
        
        # Normal flow: Queue pose estimation job (GPU stage)
        job = pose_queue.enqueue(
            'tasks.extract_keypoints',
            args=[s3_key, job_id],
            kwargs={'options': {'apply_smoothing': True}},
            job_id=job_id,
            job_timeout='1h',
            result_ttl=86400  # Keep results for 24 hours
        )
        
        logger.info(f"Job {job_id} queued for pose estimation (GPU stage)")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "stage": "pose-estimation",
            "s3_key": s3_key,
            "message": "Video uploaded and queued for pose estimation"
        }
        
    except Exception as e:
        logger.error(f"Failed to upload video: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload video: {str(e)}"
        )


@router.get("/{job_id}/status", summary="Check analysis status")
async def get_analysis_status(job_id: str) -> Dict:
    """
    Check the processing status of a video analysis job.
    
    **Possible statuses**:
    - `queued`: Waiting for worker
    - `started`: Processing in progress
    - `finished`: Complete and ready
    - `failed`: Error occurred
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    response = {
        'job_id': job_id,
        'status': job.get_status(),
        'created_at': job.created_at.isoformat() if job.created_at else None,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'ended_at': job.ended_at.isoformat() if job.ended_at else None,
    }
    
    # Add progress info from job metadata
    if job.meta:
        response['progress'] = job.meta.get('progress', 0)
        response['message'] = job.meta.get('status', '')
        if 'error' in job.meta:
            response['error'] = job.meta['error']
    
    # Add result summary if complete
    if job.is_finished and job.result:
        response['result'] = {
            'num_frames': job.result.get('num_frames'),
            'status': job.result.get('status')
        }
    
    return response


@router.get("/{job_id}/result", summary="Get analysis results")
async def get_analysis_result(job_id: str) -> Dict:
    """
    Get the complete results of a finished video analysis.
    
    Returns pose estimation data and links to download visualization video.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if not job.is_finished:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job not complete. Current status: {job.get_status()}"
        )
    
    if job.is_failed:
        error_msg = str(job.exc_info) if job.exc_info else "Unknown error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job failed: {error_msg}"
        )
    
    result = job.result
    
    try:
        # Generate pre-signed URLs for downloading (valid for 1 hour)
        keypoints_2d_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': f"results/{job_id}/keypoints_2d.json"},
            ExpiresIn=3600
        )
        
        keypoints_3d_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': f"results/{job_id}/keypoints_3d.json"},
            ExpiresIn=3600
        )
        
        visualization_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': f"results/{job_id}/visualization.mp4"},
            ExpiresIn=3600
        )
        
        return {
            "status": "complete",
            "job_id": job_id,
            "num_frames": result.get('num_frames'),
            "downloads": {
                "keypoints_2d": keypoints_2d_url,
                "keypoints_3d": keypoints_3d_url,
                "visualization": visualization_url
            },
            "s3_results": result.get('s3_results')
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve results: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}"
        )


@router.delete("/{job_id}", summary="Delete job and results")
async def delete_job(job_id: str) -> Dict:
    """
    Delete a job and all associated files from S3.
    
    Useful for cleaning up completed or failed jobs.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        job.delete()
    except:
        pass  # Job might not exist in Redis
    
    # Delete files from S3
    try:
        # Delete all files in job folder
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"uploads/{job_id}/"
        )
        
        if 'Contents' in response:
            for obj in response['Contents']:
                s3_client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
        
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"results/{job_id}/"
        )
        
        if 'Contents' in response:
            for obj in response['Contents']:
                s3_client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
        
        logger.info(f"Deleted job {job_id} from S3")
    except Exception as e:
        logger.error(f"Failed to delete S3 files: {e}")
    
    return {
        "job_id": job_id,
        "status": "deleted",
        "message": "Job and associated files removed"
    }
