FROM python:3.10-slim

# System-level deps for video I/O and OpenCV display fallbacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Work inside the server app
WORKDIR /workspace/server

# Install Python dependencies: FastAPI stack plus MMPose via OpenMMLab
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip openmim \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && mim install "mmengine>=0.8.4" \
    && mim install "mmcv>=2.0.1" \
    && mim install "mmdet>=3.3.0" \
    && mim install "mmpose>=1.3.0"

# Copy the rest of the application code
COPY . .

# Default command to serve the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
