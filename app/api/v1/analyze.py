"""Video upload endpoints."""

from pathlib import Path

from fastapi import APIRouter, File, UploadFile

router = APIRouter(tags=["analyze"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze", summary="Upload a video for analysis")
async def upload_video(file: UploadFile = File(...)) -> dict[str, str]:
    file_location = UPLOAD_DIR / file.filename
    file_location.write_bytes(await file.read())

    return {
        "info": f"Video '{file.filename}' saved at '{file_location}'",
        "status": "Ready for AlphaPose analysis",
    }
