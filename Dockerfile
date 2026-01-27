# base image
FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy

# Environment Variables
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1
# Prevents apt-get from asking for user input during install
ENV DEBIAN_FRONTEND=noninteractive

# System Dependencies (The "Heavy" Layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Work inside the server app
WORKDIR /workspace

# Install Python dependencies: FastAPI stack plus MMPose via OpenMMLab
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip --break-system-packages && \
    python3 -m pip install --no-cache-dir -r requirements.txt --break-system-packages


# Copy the rest of the application code
COPY . .

# using uvicorn for FastAPI listening on all interfaces (0.0.0.0)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
