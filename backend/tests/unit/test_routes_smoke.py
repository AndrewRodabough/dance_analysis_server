"""
Smoke tests for app initialization and route existence verification.

These tests ensure:
1. The app boots without errors
2. Key group/routine/video/note/invite endpoints exist
3. Job artifacts are under /jobs/{job_id}/artifacts (not /videos)
4. No orphaned routes from the old pairwise invite system
"""

import pytest
from fastapi.routing import APIRoute


def test_app_creates_without_error(app):
    """Verify the FastAPI app can be created and initialized."""
    assert app is not None
    assert app.title == "Dance Analysis API"


def test_group_endpoints_exist(app):
    """Verify group CRUD endpoints are registered."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    assert "/api/v1/groups" in routes
    assert "/api/v1/groups/{group_id}" in routes
    assert "/api/v1/groups/{group_id}/members" in routes
    assert "/api/v1/groups/{group_id}/members/{user_id}" in routes


def test_group_invite_endpoints_exist(app):
    """Verify group invite endpoints are registered."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    # Create invite (group-scoped)
    assert "/api/v1/groups/{group_id}/invites" in routes

    # Accept invite (token-based, global)
    assert "/api/v1/group-invites/accept" in routes


def test_routine_endpoints_exist(app):
    """Verify routine CRUD endpoints are registered."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    assert "/api/v1/groups/{group_id}/routines" in routes
    assert "/api/v1/groups/{group_id}/routines/{routine_id}" in routes


def test_routine_video_endpoints_exist(app):
    """Verify routine video library endpoints are registered."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    # Upload/register
    assert "/api/v1/groups/{group_id}/routines/{routine_id}/videos" in routes

    # Video detail, finalize, download, delete
    assert "/api/v1/groups/{group_id}/routines/{routine_id}/videos/{video_id}" in routes
    assert (
        "/api/v1/groups/{group_id}/routines/{routine_id}/videos/{video_id}/finalize"
        in routes
    )
    assert (
        "/api/v1/groups/{group_id}/routines/{routine_id}/videos/{video_id}/download"
        in routes
    )


def test_note_endpoints_exist(app):
    """Verify note endpoints (routine + video scoped) are registered."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    # Routine notes
    assert "/api/v1/groups/{group_id}/routines/{routine_id}/notes" in routes
    assert "/api/v1/groups/{group_id}/routines/{routine_id}/notes/{note_id}" in routes

    # Video notes
    assert (
        "/api/v1/groups/{group_id}/routines/{routine_id}/videos/{video_id}/notes"
        in routes
    )


def test_job_artifacts_under_jobs_namespace(app):
    """Verify job artifacts have been moved to /jobs/{job_id}/artifacts and are NOT under /videos."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    # New correct location: /api/v1/jobs/{job_id}/artifacts/...
    assert "/api/v1/jobs/{job_id}/artifacts/visualization" in routes
    assert "/api/v1/jobs/{job_id}/artifacts/keypoints2d" in routes
    assert "/api/v1/jobs/{job_id}/artifacts/keypoints3d" in routes
    assert "/api/v1/jobs/{job_id}/artifacts/scores" in routes
    assert "/api/v1/jobs/{job_id}/artifacts/feedback" in routes
    assert "/api/v1/jobs/{job_id}/artifacts/report" in routes


def test_legacy_videos_job_artifacts_routes_removed(app):
    """Verify old /api/v1/videos/{job_id}/... routes have been removed (clean break)."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    # These should NOT exist anymore
    assert "/api/v1/videos/{job_id}/visualization" not in routes
    assert "/api/v1/videos/{job_id}/keypoints2d" not in routes
    assert "/api/v1/videos/{job_id}/report" not in routes


def test_job_endpoints_exist(app):
    """Verify job CRUD endpoints are registered."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    assert "/api/v1/jobs" in routes
    assert "/api/v1/jobs/{job_id}" in routes


def test_job_routes_prefix_correct(app):
    """Verify all job artifact routes use /jobs prefix, not /videos."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    # All should start with /api/v1/jobs
    job_artifact_routes = [r for r in routes if "artifacts" in r]
    for route in job_artifact_routes:
        assert route.startswith("/api/v1/jobs"), (
            f"Route {route} should be under /jobs, not /videos"
        )


def test_no_legacy_pairwise_invites_router(app):
    """Verify the old pairwise invite endpoint (/api/v1/invites) does not exist."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    # Old pairwise system: /invites (without group_id)
    # New group system: /groups/{group_id}/invites
    assert "/api/v1/invites" not in routes, (
        "Legacy pairwise invites endpoint should be removed"
    )


def test_required_endpoints_summary(app):
    """Summary test: verify at least N critical endpoints exist."""
    routes = {r.path for r in app.router.routes if isinstance(r, APIRoute)}

    critical_endpoints = {
        "/api/v1/groups",
        "/api/v1/groups/{group_id}/invites",
        "/api/v1/group-invites/accept",
        "/api/v1/groups/{group_id}/routines",
        "/api/v1/groups/{group_id}/routines/{routine_id}/videos",
        "/api/v1/groups/{group_id}/routines/{routine_id}/notes",
        "/api/v1/jobs/{job_id}/artifacts/report",
    }

    missing = critical_endpoints - routes
    assert not missing, f"Missing critical endpoints: {missing}"
