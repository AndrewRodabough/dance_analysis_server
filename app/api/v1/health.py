"""Health and status endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/", summary="Service status")
def read_root() -> dict[str, str]:
    return {"message": "Ballroom AI Engine is Online"}


@router.get("/health", summary="Health check")
def health() -> dict[str, str]:
    return {"status": "ok"}
