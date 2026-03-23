"""Job artifact download endpoints (owner-only, non-leaky).

This router exposes presigned URLs (and a JSON report passthrough) for artifacts
produced by an analysis job.

Privacy requirements:
- Access is restricted to the job owner.
- Non-owners must receive 404 (not 403) to avoid leaking job existence.
- Artifact existence should not leak beyond the job-ownership boundary.

Artifacts are expected to live under:
- results/{job_id}/visualization.mp4
- results/{job_id}/keypoints_2d.json
- results/{job_id}/keypoints_3d.json
- results/{job_id}/scores.json
- results/{job_id}/feedback.txt
- results/{job_id}/report.json
"""

from __future__ import annotations

import json
from typing import Any, Dict

import boto3
from botocore.client import Config
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.authorization import require_job_owner
from app.core.config import settings
from app.core.deps import get_current_active_user
from app.core.logging import get_logger, log_storage_operation
from app.database import get_db
from app.models.user import User

logger = get_logger(__name__)

router = APIRouter()


def _get_s3_client():
    """Create an S3-compatible client for pulling job artifacts.

    Job artifacts are currently stored in the same S3-compatible storage used by
    the existing results pipeline (MinIO in dev; could be R2/S3 in prod).
    """
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name=getattr(settings, "S3_REGION", None) or "us-east-1",
    )


def _presign_get_and_log(*, job_id: str, key: str, label: str) -> str:
    """Generate a presigned GET URL for an artifact and log the operation."""
    client = _get_s3_client()
    bucket = settings.S3_BUCKET

    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        log_storage_operation(
            operation="presign_get",
            provider="s3",
            bucket=bucket,
            key=key,
            job_id=job_id,
        )
        return url
    except Exception as e:
        # Non-leaky for artifact existence: within job-owner boundary, returning 404 is fine.
        log_storage_operation(
            operation="presign_get",
            provider="s3",
            bucket=bucket,
            key=key,
            job_id=job_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} not found",
        )


@router.get(
    "/jobs/{job_id}/artifacts/visualization",
    summary="Get presigned URL for visualization video",
)
def get_visualization_url(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Owner-only: presigned URL for the pose visualization mp4."""
    require_job_owner(db, job_id, current_user.id)
    key = f"results/{job_id}/visualization.mp4"
    return {"url": _presign_get_and_log(job_id=job_id, key=key, label="Visualization")}


@router.get(
    "/jobs/{job_id}/artifacts/keypoints2d",
    summary="Get presigned URL for 2D keypoints JSON",
)
def get_keypoints_2d_url(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Owner-only: presigned URL for 2D keypoints json."""
    require_job_owner(db, job_id, current_user.id)
    key = f"results/{job_id}/keypoints_2d.json"
    return {"url": _presign_get_and_log(job_id=job_id, key=key, label="Keypoints 2D")}


@router.get(
    "/jobs/{job_id}/artifacts/keypoints3d",
    summary="Get presigned URL for 3D keypoints JSON",
)
def get_keypoints_3d_url(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Owner-only: presigned URL for 3D keypoints json."""
    require_job_owner(db, job_id, current_user.id)
    key = f"results/{job_id}/keypoints_3d.json"
    return {"url": _presign_get_and_log(job_id=job_id, key=key, label="Keypoints 3D")}


@router.get(
    "/jobs/{job_id}/artifacts/scores",
    summary="Get presigned URL for confidence scores JSON",
)
def get_scores_url(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Owner-only: presigned URL for scores json."""
    require_job_owner(db, job_id, current_user.id)
    key = f"results/{job_id}/scores.json"
    return {"url": _presign_get_and_log(job_id=job_id, key=key, label="Scores")}


@router.get(
    "/jobs/{job_id}/artifacts/feedback",
    summary="Get presigned URL for feedback text",
)
def get_feedback_url(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Owner-only: presigned URL for feedback.txt."""
    require_job_owner(db, job_id, current_user.id)
    key = f"results/{job_id}/feedback.txt"
    return {"url": _presign_get_and_log(job_id=job_id, key=key, label="Feedback")}


@router.get(
    "/jobs/{job_id}/artifacts/report",
    summary="Get structured feedback report JSON",
    response_model=Dict[str, Any],
)
def get_report_json(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Owner-only: return parsed JSON report.

    Reads results/{job_id}/report.json and returns it as a JSON object.

    Non-leaky ownership:
    - require_job_owner returns 404 for non-owners.
    """
    require_job_owner(db, job_id, current_user.id)

    client = _get_s3_client()
    bucket = settings.S3_BUCKET
    key = f"results/{job_id}/report.json"

    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        body = obj.get("Body")
        if body is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Report object missing body",
            )
        raw_bytes = body.read()
    except client.exceptions.NoSuchKey:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )
    except HTTPException:
        raise
    except Exception as e:
        log_storage_operation(
            operation="get_object",
            provider="s3",
            bucket=bucket,
            key=key,
            job_id=job_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving report",
        )

    try:
        report = json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored report is invalid JSON",
        )

    return report
