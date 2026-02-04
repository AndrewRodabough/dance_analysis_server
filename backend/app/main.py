from pathlib import Path

from app.api.v1 import analyze, health, videos
from fastapi import FastAPI


def create_app() -> FastAPI:
    """Application factory to build the FastAPI app."""

    app = FastAPI(
        title="Dance Analysis API",
        version="1.0.0",
        description="Video dance analysis with pose estimation via microservices"
    )

    # Include routers with prefixes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
    app.include_router(videos.router, prefix="/api/v1", tags=["videos"])

    return app

app = create_app()
