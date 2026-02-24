"""
Request logging middleware for FastAPI.

Automatically logs all HTTP requests with:
- Method, path, status code, duration
- Unique request ID for tracing
- Client IP and user agent
"""

import time
import uuid
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import request_id_var, request_path_var, request_method_var


logger = logging.getLogger("api.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs all HTTP requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]

        # Set context variables for use in other loggers
        request_id_var.set(request_id)
        request_path_var.set(request.url.path)
        request_method_var.set(request.method)

        # Record start time
        start_time = time.perf_counter()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            raise
        finally:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Build log record
            log_data = {
                "event_type": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params) if request.query_params else None,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
            }

            if error:
                log_data["error"] = error

            # Log at appropriate level
            if status_code >= 500:
                logger.error(
                    f"{request.method} {request.url.path} {status_code}",
                    extra=log_data,
                )
            elif status_code >= 400:
                logger.warning(
                    f"{request.method} {request.url.path} {status_code}",
                    extra=log_data,
                )
            else:
                logger.info(
                    f"{request.method} {request.url.path} {status_code}",
                    extra=log_data,
                )

            # Reset context vars
            request_id_var.set(None)
            request_path_var.set(None)
            request_method_var.set(None)

        # Add request ID to response headers for client-side tracing
        response.headers["X-Request-ID"] = request_id
        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
