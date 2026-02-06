import os

from app.api.v1 import analyze, health, videos
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """Application factory to build the FastAPI app."""

    app = FastAPI(
        title="Dance Analysis API",
        version="1.0.0",
        description="Video dance analysis with pose estimation via microservices"
    )

    # Get allowed origins from environment variable
    # Default includes common local development ports
    allowed_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8080"
    ).split(",")

    # Add CORS middleware for client applications
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers with prefixes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
    app.include_router(videos.router, prefix="/api/v1", tags=["videos"])

    return app


app = create_app()
