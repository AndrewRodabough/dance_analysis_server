"""Legacy invite endpoint — replaced by group invites.

This module is kept as a stub to prevent import errors during the transition.
All invite functionality has been moved to the group_invites module.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/invites", tags=["invites-legacy"])
