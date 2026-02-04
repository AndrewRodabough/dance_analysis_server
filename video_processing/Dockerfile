# base image
FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy

# Environment Variables
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1
# Prevents apt-get from asking for user input during install
ENV DEBIAN_FRONTEND=noninteractive

# System Dependencies (The "Heavy" Layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Work inside the server app
WORKDIR /workspace

# Install Python dependencies: RQ + boto3 + all ML/vision libraries
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p temp outputs

# Default command: Start RQ worker listening to video-processing queue
CMD ["python3", "-m", "rq", "worker", "--url", "redis://redis:6379/0", "video-processing"]