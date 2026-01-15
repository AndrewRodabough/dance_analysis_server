from fastapi import FastAPI, UploadFile, File
import os

app = FastAPI()

# Create an uploads folder if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Ballroom AI Engine is Online"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    return {
        "info": f"Video '{file.filename}' saved at '{file_location}'",
        "status": "Ready for AlphaPose analysis"
    }
