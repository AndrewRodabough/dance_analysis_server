from fastapi import FastAPI

from app.api.v1 import analyze, auth, health, invite, jobs, videos
from app.core.logging import setup_logging
from app.middleware.logging import RequestLoggingMiddleware


def create_app() -> FastAPI:
    """Application factory to build the FastAPI app."""

    # Configure logging before anything else
    setup_logging()

    app = FastAPI(
        title="Dance Analysis API",
        version="1.0.0",
        description="Video dance analysis with pose estimation via microservices"
    )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)


    # Include routers with prefixes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
    app.include_router(invite.router, prefix="/api/v1", tags=["invites"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
    app.include_router(videos.router, prefix="/api/v1", tags=["videos"])

    return app

app = create_app()
