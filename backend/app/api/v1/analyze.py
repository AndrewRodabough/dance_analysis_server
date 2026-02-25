"""Video analysis endpoints - upload, queue, and status tracking."""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import boto3
from app.core.deps import get_current_active_user
from app.core.logging import get_logger, log_job_status, log_storage_operation
from app.database import get_db
from app.models.job import Job as DBJob
from app.models.job import JobStatus
from app.models.user import User
from app.schemas.job import JobCreate, JobStatusUpdate
from app.services.job_service import JobService
from botocore.client import Config
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

logger = get_logger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Local MinIO configuration (internal processing storage)
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "dance-videos")

# Cloudflare R2 configuration (public upload storage)
R2_ENDPOINT = os.getenv("R2_ENDPOINT", "https://example.r2.cloudflarestorage.com")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET", "dance-videos-r2")

# Initialize S3 client (for internal MinIO operations)
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Initialize R2 client for presigned upload URLs and reads
r2_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

@router.post("/upload-url", summary="Request presigned upload URL", status_code=status.HTTP_200_OK)
async def request_upload_url(
    filename: str = Query(..., description="Name of the video file"),
    content_type: str = Query("video/mp4", description="MIME type of the video"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Request a presigned URL for direct upload to Cloudflare R2.

    This enables clients to upload videos directly to R2 without going through the API server,
    reducing server load and improving upload performance. The server will later pull the
    video from R2 into local MinIO on confirm.

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

    # Create job in PostgreSQL
    job_data = JobCreate(filename=filename)
    db_job = JobService.create_job(db, current_user.id, job_data)
    job_id = db_job.job_id


    rand_uuid = uuid.uuid4().hex
    # Object key structure: uploads/{user id}{job_id}{16 char uuid}/{filename}
    object_key = f"uploads/{current_user.id}-{job_id}-{rand_uuid}/{filename}"

    # Tentatively store video path as this key (location changes from R2 -> MinIO on confirm)
    JobService.update_job_video_path(db, job_id, object_key)

    try:
        # Generate presigned URL for direct upload to Cloudflare R2 (valid for 15 minutes)
        upload_url = r2_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': R2_BUCKET,
                'Key': object_key
            },
            ExpiresIn=900,  # 15 minutes
            HttpMethod='PUT'
        )

        log_storage_operation(
            operation="presign",
            provider="r2",
            bucket=R2_BUCKET,
            key=object_key,
            job_id=job_id,
        )

        return {
            "job_id": job_id,
            "upload_url": upload_url,
            "s3_key": object_key,  # keep field name for backward compatibility
            "expires_in": 900,
            "instructions": "PUT the video file to upload_url, then call /analyze/confirm with job_id"
        }

    except Exception as e:
        log_storage_operation(
            operation="presign",
            provider="r2",
            bucket=R2_BUCKET,
            key=object_key,
            job_id=job_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate upload URL: {str(e)}"
        )


@router.post("/confirm", summary="Confirm upload and start analysis", status_code=status.HTTP_202_ACCEPTED)
async def confirm_upload_and_start_analysis(
    job_id: str = Query(..., description="Job ID from upload-url endpoint"),
    s3_key: str = Query(..., description="S3 key from upload-url endpoint"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Confirm that a video has been uploaded to Cloudflare R2 and start the analysis pipeline.

    Call this endpoint after successfully uploading the video using the presigned URL
    from /analyze/upload-url. This will pull the video from R2 into local MinIO
    before queuing the job.

    **Args:**
    - job_id: The job ID received from /analyze/upload-url
    - s3_key: The S3 key received from /analyze/upload-url

    **Returns:**
    - Job details and queuing status
    """
    # Verify job exists and belongs to user
    db_job = JobService.get_job_by_id(db, job_id, current_user.id)
    if not db_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or access denied"
        )

    try:
        object_key = s3_key

        # 1. Verify the object exists in R2
        try:
            r2_client.head_object(Bucket=R2_BUCKET, Key=object_key)
        except Exception:
            log_storage_operation(
                operation="download",
                provider="r2",
                bucket=R2_BUCKET,
                key=object_key,
                job_id=job_id,
                error="Object not found in R2",
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found in R2. Please upload the file first using the presigned URL."
            )

        # 2. Copy from R2 -> local MinIO for processing
        try:
            copy_start = time.perf_counter()
            r2_obj = r2_client.get_object(Bucket=R2_BUCKET, Key=object_key)
            body = r2_obj["Body"].read()

            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=object_key,
                Body=body,
                ContentType=r2_obj.get("ContentType", "video/mp4")
            )
            copy_duration = (time.perf_counter() - copy_start) * 1000
            log_storage_operation(
                operation="copy",
                provider="r2",
                bucket=R2_BUCKET,
                key=object_key,
                job_id=job_id,
                bytes_transferred=len(body),
                duration_ms=copy_duration,
                destination_provider="minio",
                destination_bucket=S3_BUCKET,
            )
        except Exception as e:
            log_storage_operation(
                operation="copy",
                provider="r2",
                bucket=R2_BUCKET,
                key=object_key,
                job_id=job_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to move video from R2 to processing storage"
            )

        # 3. Optionally delete from R2 to save storage; ignore failures
        try:
            r2_client.delete_object(Bucket=R2_BUCKET, Key=object_key)
        except Exception as e:
            logger.warning(f"Failed to delete video from R2 for job {job_id}: {e}")

        # 4. Update job to reference MinIO location (key stays the same)
        JobService.update_job_video_path(db, job_id, object_key)

        # Job is already in PostgreSQL with PENDING status; worker will pick it up according to the
        # job processing contract (DB is authoritative, object storage is artifact only).
        log_job_status(job_id, status="queued", stage="pending")

        return {
            "job_id": job_id,
            "status": "pending",
            "stage": "queued",
            "s3_key": object_key,
            "message": "Video confirmed, moved to processing storage, and queued for processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        log_job_status(job_id, status="failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start analysis: {str(e)}"
        )


@router.post("", summary="Submit video for analysis (legacy)", status_code=status.HTTP_202_ACCEPTED)
async def submit_video_analysis(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov)"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
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

    # Create job in PostgreSQL
    job_data = JobCreate(filename=file.filename)
    db_job = JobService.create_job(db, current_user.id, job_data)
    job_id = db_job.job_id

    # S3 key structure: uploads/{job_id}/{filename}
    s3_key = f"uploads/{job_id}/{file.filename}"

    try:
        # Upload video to S3
        file_content = await file.read()
        upload_start = time.perf_counter()
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType=file.content_type or 'video/mp4'
        )
        upload_duration = (time.perf_counter() - upload_start) * 1000

        # Update job with video path
        JobService.update_job_video_path(db, job_id, s3_key)

        log_storage_operation(
            operation="upload",
            provider="minio",
            bucket=S3_BUCKET,
            key=s3_key,
            job_id=job_id,
            bytes_transferred=len(file_content),
            duration_ms=upload_duration,
        )

        # Job is in PostgreSQL with PENDING status - worker will pick it up
        log_job_status(job_id, status="queued", stage="pending")

        return {
            "job_id": job_id,
            "status": "queued",
            "stage": "analysis",
            "s3_key": s3_key,
            "message": "Video uploaded and queued for analysis"
        }

    except Exception as e:
        log_storage_operation(
            operation="upload",
            provider="minio",
            bucket=S3_BUCKET,
            key=s3_key,
            job_id=job_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload video: {str(e)}"
        )


@router.get("/{job_id}/status", summary="Check analysis status")
async def get_analysis_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Check the processing status of a video analysis job.

    **Possible statuses**:
    - `pending`: Job created, waiting to be queued
    - `processing`: Being processed by worker
    - `completed`: Complete and ready
    - `failed`: Error occurred
    """
    # First check PostgreSQL for job ownership
    db_job = JobService.get_job_by_id(db, job_id, current_user.id)
    if not db_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    response = {
        'job_id': job_id,
        'status': db_job.status.value,
        'filename': db_job.filename,
        'created_at': db_job.created_at.isoformat() if db_job.created_at else None,
        'started_at': db_job.started_at.isoformat() if db_job.started_at else None,
        'completed_at': db_job.completed_at.isoformat() if db_job.completed_at else None,
    }

    # Add error message if failed
    if db_job.error_message:
        response['error'] = db_job.error_message

    return response


@router.get("/{job_id}/result", summary="Get analysis results")
async def get_analysis_result(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Get the complete results of a finished video analysis.

    Returns pose estimation data and links to download visualization video.
    """
    # Check PostgreSQL for job ownership
    db_job = JobService.get_job_by_id(db, job_id, current_user.id)
    if not db_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if db_job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job not complete. Current status: {db_job.status.value}"
        )

    try:
        # Generate pre-signed URLs for downloading (valid for 1 hour)
        result_keys = {
            "keypoints_2d": f"results/{job_id}/keypoints_2d.json",
            "keypoints_3d": f"results/{job_id}/keypoints_3d.json",
            "visualization": f"results/{job_id}/visualization.mp4",
        }
        downloads = {}
        for name, key in result_keys.items():
            downloads[name] = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': key},
                ExpiresIn=3600
            )
            log_storage_operation(
                operation="presign",
                provider="minio",
                bucket=S3_BUCKET,
                key=key,
                job_id=job_id,
            )

        return {
            "status": "complete",
            "job_id": job_id,
            "filename": db_job.filename,
            "downloads": downloads
        }

    except Exception as e:
        log_storage_operation(
            operation="presign",
            provider="minio",
            bucket=S3_BUCKET,
            key=f"results/{job_id}/",
            job_id=job_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}"
        )


@router.delete("/{job_id}", summary="Delete job and results")
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Delete a job and all associated files from S3.

    Useful for cleaning up completed or failed jobs.
    """
    # Delete from PostgreSQL (also verifies ownership)
    deleted = JobService.delete_job(db, job_id, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or access denied"
        )

    # Delete files from S3
    try:
        for prefix in [f"uploads/{job_id}/", f"results/{job_id}/"]:
            response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
                    log_storage_operation(
                        operation="delete",
                        provider="minio",
                        bucket=S3_BUCKET,
                        key=obj['Key'],
                        job_id=job_id,
                    )
    except Exception as e:
        log_storage_operation(
            operation="delete",
            provider="minio",
            bucket=S3_BUCKET,
            key=f"uploads/{job_id}/",
            job_id=job_id,
            error=str(e),
        )

    return {
        "job_id": job_id,
        "status": "deleted",
        "message": "Job and associated files removed"
    }