"""Storage abstraction for presigned URL generation and object operations."""

import uuid
from typing import Any
from uuid import UUID

import boto3
from botocore.client import Config

from app.core.config import settings
from app.core.logging import get_logger, log_storage_operation

logger = get_logger(__name__)


def _get_r2_client():
    """Lazy R2 client (for routine video uploads and downloads)."""
    return boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT,
        aws_access_key_id=settings.R2_ACCESS_KEY,
        aws_secret_access_key=settings.R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _get_minio_client():
    """Lazy MinIO client (for internal storage / downloads)."""
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def generate_storage_key(user_id: Any, session_id: Any, filename: str) -> str:
    """Generate a deterministic but non-guessable storage key.

    Format: session-videos/{session_id}/{user_id}-{unique}{ext}
    The original filename is NOT leaked into the key; only the extension is kept.
    """
    import os
    ext = os.path.splitext(filename)[1].lower() or ".bin"
    unique = uuid.uuid4().hex[:16]
    return f"session-videos/{session_id}/{user_id}-{unique}{ext}"


def create_presigned_put_url(
    storage_key: str,
    content_type: str = "video/mp4",
    expires_in: int = 900,
) -> str:
    """Generate a presigned PUT URL for uploading to R2."""
    bucket = settings.R2_BUCKET
    client = _get_r2_client()

    url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": storage_key, "ContentType": content_type},
        ExpiresIn=expires_in,
        HttpMethod="PUT",
    )
    log_storage_operation(
        operation="presign_put",
        provider="r2",
        bucket=bucket,
        key=storage_key,
    )
    return url


def create_presigned_get_url(
    storage_key: str,
    expires_in: int = 3600,
) -> str:
    """Generate a presigned GET URL for downloading from R2."""
    bucket = settings.R2_BUCKET
    client = _get_r2_client()

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": storage_key},
        ExpiresIn=expires_in,
    )
    log_storage_operation(
        operation="presign_get",
        provider="r2",
        bucket=bucket,
        key=storage_key,
    )
    return url


def head_object(storage_key: str) -> bool:
    """Check if an object exists in R2. Returns True if it does."""
    client = _get_r2_client()
    try:
        client.head_object(Bucket=settings.R2_BUCKET, Key=storage_key)
        return True
    except Exception:
        return False
