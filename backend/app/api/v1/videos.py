"""Video file serving endpoints."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
import os
import boto3
from botocore.client import Config

from app.core.logging import get_logger, log_storage_operation

logger = get_logger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])

# S3 Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "dance-videos")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)


def _presign_and_log(job_id: str, key: str, label: str) -> str:
    """Generate a presigned download URL and log the operation."""
    try:
        url = s3_client.generate_presigned_url(
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
        return url
    except Exception as e:
        log_storage_operation(
            operation="presign",
            provider="minio",
            bucket=S3_BUCKET,
            key=key,
            job_id=job_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} not found"
        )


@router.get("/{job_id}/visualization", summary="Download visualization video")
async def download_visualization(job_id: str):
    """Download the pose estimation visualization video"""
    url = _presign_and_log(job_id, f"results/{job_id}/visualization.mp4", "Visualization")
    return {"url": url}


@router.get("/{job_id}/keypoints2d", summary="Download 2D keypoints")
async def download_keypoints_2d(job_id: str):
    """Download 2D keypoints JSON file"""
    url = _presign_and_log(job_id, f"results/{job_id}/keypoints_2d.json", "Keypoints 2D")
    return {"url": url}


@router.get("/{job_id}/keypoints3d", summary="Download 3D keypoints")
async def download_keypoints_3d(job_id: str):
    """Download 3D keypoints JSON file"""
    url = _presign_and_log(job_id, f"results/{job_id}/keypoints_3d.json", "Keypoints 3D")
    return {"url": url}


@router.get("/{job_id}/scores", summary="Download confidence scores")
async def download_scores(job_id: str):
    """Download confidence scores JSON file"""
    url = _presign_and_log(job_id, f"results/{job_id}/scores.json", "Scores")
    return {"url": url}


@router.get("/{job_id}/feedback", summary="Download feedback text")
async def download_feedback(job_id: str):
    """Download feedback text file"""
    url = _presign_and_log(job_id, f"results/{job_id}/feedback.txt", "Feedback")
    return {"url": url}
