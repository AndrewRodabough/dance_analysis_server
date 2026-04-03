from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import (
    analyze,
    auth,
    dancer_slots,
    dances,
    group_invites,
    groups,
    health,
    job_artifacts,
    jobs,
    notes,
    routine_sessions,
    routine_videos,
    routines,
    slot_assignments,
)

# Note: legacy invite router removed; group_invites replaces it
from app.core.config import settings
from app.core.logging import setup_logging
from app.middleware.logging import RequestLoggingMiddleware


def create_app() -> FastAPI:
    """Application factory to build the FastAPI app."""

    # Configure logging before anything else
    setup_logging()

    app = FastAPI(
        title="Dance Analysis API",
        version="1.0.0",
        description="Video dance analysis with pose estimation via microservices",
    )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Core routes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])

    # Groups, memberships, and invites
    app.include_router(groups.router, prefix="/api/v1/groups", tags=["groups"])
    app.include_router(group_invites.router, prefix="/api/v1", tags=["group-invites"])

    # Dances
    app.include_router(dances.router, prefix="/api/v1/dances", tags=["dances"])

    # Routines (top-level, user-owned)
    app.include_router(
        routines.router,
        prefix="/api/v1/routines",
        tags=["routines"],
    )

    # Routine sessions, dancer slots (paths defined in routers)
    app.include_router(
        routine_sessions.router, prefix="/api/v1", tags=["routine-sessions"]
    )
    app.include_router(
        dancer_slots.router, prefix="/api/v1", tags=["dancer-slots"]
    )
    app.include_router(
        slot_assignments.router, prefix="/api/v1", tags=["slot-assignments"]
    )

    # Session videos
    app.include_router(
        routine_videos.router,
        prefix="/api/v1/sessions/{session_id}/videos",
        tags=["session-videos"],
    )

    # Notes (session + video scoped, paths defined in router)
    app.include_router(notes.router, prefix="/api/v1", tags=["notes"])

    # Jobs and job artifacts
    app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
    app.include_router(job_artifacts.router, prefix="/api/v1", tags=["job-artifacts"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])

    return app


app = create_app()
