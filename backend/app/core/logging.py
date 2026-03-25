"""
Centralized logging configuration with JSON formatting for Grafana/Loki.

Usage:
    from app.core.logging import setup_logging, get_logger, log_job_status

    # At app startup
    setup_logging()

    # In modules
    logger = get_logger(__name__)
    logger.info("Something happened", extra={"user_id": "123"})

    # For job status updates
    log_job_status("job-id", status="started", stage="pose-estimation")
"""

import logging
import sys
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional
from pythonjsonlogger import jsonlogger


# Context variables for request-scoped data
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
request_path_var: ContextVar[Optional[str]] = ContextVar("request_path", default=None)
request_method_var: ContextVar[Optional[str]] = ContextVar("request_method", default=None)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter that includes request context and standard fields."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Standard fields
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = "dance-analysis-api"

        # Request context (if available)
        request_id = request_id_var.get()
        if request_id:
            log_record["request_id"] = request_id

        request_path = request_path_var.get()
        if request_path:
            log_record["path"] = request_path

        request_method = request_method_var.get()
        if request_method:
            log_record["method"] = request_method


def setup_logging(level: str = None) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO,
               or LOG_LEVEL env var if set.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")

    # Create JSON formatter
    formatter = CustomJsonFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add stdout handler with JSON formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Log startup
    root_logger.info(f"Logging configured", extra={"log_level": log_level})


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured logger instance
    """
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

    This creates queryable logs in Grafana for job monitoring.

    Args:
        job_id: Unique job identifier
        status: Job status (created, queued, processing, completed, failed)
        stage: Processing stage (pose-estimation, analysis, etc.)
        progress: Progress percentage (0-100)
        error: Error message if failed
        **extra: Additional fields to include
    """
    logger = logging.getLogger("job_status")

    log_data = {
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
    Log a storage operation (upload, download, copy, delete, presign).

    This creates queryable logs in Grafana for storage monitoring.

    Args:
        operation: Operation type (upload, download, copy, delete, presign)
        provider: Storage provider (r2, minio)
        bucket: Bucket name
        key: Object key
        job_id: Associated job ID if applicable
        bytes_transferred: Number of bytes transferred
        duration_ms: Operation duration in milliseconds
        error: Error message if failed
        **extra: Additional fields to include
    """
    logger = logging.getLogger("storage")

    log_data: dict[str, Any] = {
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


def log_auth_event(
    action: str,
    email: Optional[str] = None,
    user_id: Optional[Any] = None,
    success: bool = True,
    error: Optional[str] = None,
    **extra: Any,
) -> None:
    """
    Log an authentication event.

    Args:
        action: Auth action (register, login)
        email: User email (logged for failed attempts, omit for privacy if preferred)
        user_id: User ID if known
        success: Whether the action succeeded
        error: Error/failure reason
        **extra: Additional fields to include
    """
    logger = logging.getLogger("auth")

    log_data: dict[str, Any] = {
        "event_type": "auth",
        "action": action,
        "success": success,
    }

    if email:
        log_data["email"] = email
    if user_id is not None:
        log_data["user_id"] = user_id
    if error:
        log_data["error"] = error

    log_data.update(extra)

    level = logging.WARNING if not success else logging.INFO
    status_str = "success" if success else "failed"
    logger.log(level, f"Auth {action}: {status_str}", extra=log_data)
