"""Health and status endpoints."""

from typing import Dict
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/", summary="Service status")
def read_root() -> Dict[str, str]:
    return {"message": "Ballroom AI Engine is Online"}


@router.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}
