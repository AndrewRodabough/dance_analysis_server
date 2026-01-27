from pathlib import Path
from typing import Dict
from app.analysis.analysis import analyze_video

from fastapi import APIRouter, File, UploadFile

router = APIRouter(tags=["analyze"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze", summary="Upload a video for analysis")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, str]:
    file_location = UPLOAD_DIR / file.filename
    file_location.write_bytes(await file.read())

    analyze_video(str(file_location))

    return {
        "status": "Analysis Complete",
    }
