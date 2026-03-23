"""
DEPRECATED: Legacy /videos job-artifacts router.

Clean break: job artifacts have been moved under the Jobs API.
This module is intentionally left as a stub to avoid accidental reintroduction
of legacy routes and to make it explicit that `/api/v1/videos/...` is reserved
for routine video library endpoints (group/routine scoped).

If you need job artifacts, use:
- `/api/v1/jobs/{job_id}/artifacts/...`

This file should not define any routes.
"""

from fastapi import APIRouter

# Empty router (no endpoints). Retained only to prevent import errors if
# referenced elsewhere; ideally remove all imports and delete this file later.
router = APIRouter()
