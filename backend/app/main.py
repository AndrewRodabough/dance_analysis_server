from fastapi import FastAPI

from app.api.v1 import (
    analyze,
    auth,
    group_invites,
    groups,
    health,
    jobs,
    notes,
    routine_videos,
    routines,
    videos,
)
# Note: legacy invite router removed; group_invites replaces it
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

    # Core routes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])

    # Groups, memberships, and invites
    app.include_router(groups.router, prefix="/api/v1/groups", tags=["groups"])
    app.include_router(group_invites.router, prefix="/api/v1", tags=["group-invites"])

    # Routines (group-scoped)
    app.include_router(
        routines.router,
        prefix="/api/v1/groups/{group_id}/routines",
        tags=["routines"],
    )

    # Routine videos (group-scoped)
    app.include_router(
        routine_videos.router,
        prefix="/api/v1/groups/{group_id}/routines/{routine_id}/videos",
        tags=["routine-videos"],
    )

    # Notes (routine + video scoped, paths defined in router)
    app.include_router(notes.router, prefix="/api/v1", tags=["notes"])

    # Jobs and job artifacts
    app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
    app.include_router(videos.router, prefix="/api/v1", tags=["job-artifacts"])

    return app

app = create_app()
