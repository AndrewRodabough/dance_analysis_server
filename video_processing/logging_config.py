"""
Centralized JSON logging configuration for the video processing worker.

Mirrors the backend's logging format so Alloy/Loki can parse structured fields
(level, service, event_type) the same way for both containers.

Usage:
    from logging_config import setup_logging, get_logger, log_job_status, log_storage_operation

    setup_logging()
    logger = get_logger(__name__)
    logger.info("Something happened", extra={"job_id": "abc"})
"""

import logging
import sys
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger


SERVICE_NAME = "dance-video-worker"


class WorkerJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter matching the backend's log structure."""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = SERVICE_NAME


def setup_logging(level: str = None) -> None:
    """
    Configure JSON logging for the worker process.

    Args:
        level: Log level. Defaults to LOG_LEVEL env var or INFO.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")

    formatter = WorkerJsonFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root_logger.info("Logging configured", extra={"log_level": log_level})


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def log_job_status(
    job_id: str,
    status: str,
    stage: Optional[str] = None,
    progress: Optional[int] = None,
    error: Optional[str] = None,
    **extra: Any,
) -> None:
    """
    Log a job status update with consistent structure.

    Args:
        job_id: Unique job identifier
        status: Job status (processing, completed, failed, etc.)
        stage: Processing stage (downloading, pose-estimation, analysis, etc.)
        progress: Progress percentage (0-100)
        error: Error message if failed
        **extra: Additional fields to include
    """
    logger = logging.getLogger("job_status")

    log_data: Dict[str, Any] = {
        "event_type": "job_status",
        "job_id": job_id,
        "status": status,
    }

    if stage:
        log_data["stage"] = stage
    if progress is not None:
        log_data["progress"] = progress
    if error:
        log_data["error"] = error

    log_data.update(extra)

    level = logging.ERROR if status == "failed" else logging.INFO
    logger.log(level, f"Job {job_id}: {status}", extra=log_data)


def log_storage_operation(
    operation: str,
    provider: str,
    bucket: str,
    key: str,
    job_id: Optional[str] = None,
    bytes_transferred: Optional[int] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    **extra: Any,
) -> None:
    """
    Log a storage operation (upload, download, delete).

    Args:
        operation: Operation type (upload, download, delete)
        provider: Storage provider (minio)
        bucket: Bucket name
        key: Object key
        job_id: Associated job ID
        bytes_transferred: Number of bytes transferred
        duration_ms: Operation duration in milliseconds
        error: Error message if failed
        **extra: Additional fields to include
    """
    logger = logging.getLogger("storage")

    log_data: Dict[str, Any] = {
        "event_type": "storage_operation",
        "operation": operation,
        "provider": provider,
        "bucket": bucket,
        "key": key,
    }

    if job_id:
        log_data["job_id"] = job_id
    if bytes_transferred is not None:
        log_data["bytes"] = bytes_transferred
    if duration_ms is not None:
        log_data["duration_ms"] = round(duration_ms, 2)
    if error:
        log_data["error"] = error

    log_data.update(extra)

    level = logging.ERROR if error else logging.INFO
    msg = f"Storage {operation} {provider}://{bucket}/{key}"
    if error:
        msg += f" failed: {error}"
    logger.log(level, msg, extra=log_data)
