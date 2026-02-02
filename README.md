# Backend For dance-analysis Project

## Quick Start

### Local Development (Mock Mode - No GPU Required)

For API development without Docker/GPU dependencies:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install minimal dependencies
pip install -r requirements-minimal.txt

# Run with mock analysis
./run_local.sh

# Or manually:
export USE_MOCK_ANALYSIS=true
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode (Docker - GPU Required)

For full pipeline with actual pose estimation:

#### With NVIDIA GPU
```bash
docker compose --profile nvidia up
```

#### With AMD GPU
```bash
docker compose --profile amd up
```

#### With Mac GPU (Metal Performance Shaders)
```bash
docker compose --profile mac up
```
**Requirements:**
- Docker Desktop for Mac with GPU support enabled (Settings → Resources → GPU)
- Apple Silicon Mac (M1/M2/M3 or newer)
- PyTorch 1.12+

#### CPU Only
```bash
docker compose --profile cpu up
```

## API Endpoints

- `GET /` - Service status
- `GET /health` - Health check
- `POST /analyze` - Upload video for analysis
- API docs: http://localhost:8000/docs

## Development Modes

**Mock Mode** (USE_MOCK_ANALYSIS=true):
- No GPU/MediaPipe dependencies
- Returns realistic fake pose data
- Fast iteration for API development
- Perfect for frontend integration testing

**Production Mode** (USE_MOCK_ANALYSIS=false):
- Requires Docker with GPU support
- Uses MediaPipe for actual pose estimation
- Generates real visualization videos

## Microservices Architecture with Profiles

The system is organized as independent microservices with Redis queue and MinIO S3 storage:

### Services

- **Backend API**: FastAPI server for video uploads and result retrieval
- **Redis**: Job queue for async video processing
- **MinIO**: S3-compatible object storage for videos and results
- **Video Worker**: GPU-accelerated pose estimation processor

### GPU Profiles

Choose one profile based on your hardware:

```bash
# CPU only (no GPU)
./manage.sh cpu start

# NVIDIA GPU with CUDA
./manage.sh nvidia start

# AMD GPU with ROCm
./manage.sh amd start

# Mac Apple Silicon with Metal
./manage.sh mac start
```

### manage.sh Helper Script

Convenient wrapper around Docker Compose commands with profile support:

```bash
# Start services with a profile
./manage.sh [profile] start

# View logs
./manage.sh [profile] logs [service]

# Check status
./manage.sh [profile] status

# Open shell in container
./manage.sh [profile] shell-backend
./manage.sh [profile] shell-worker

# Upload test video
./manage.sh [profile] test-upload video.mp4

# View dashboards
./manage.sh minio-console    # MinIO: http://localhost:9001
./manage.sh rq-dashboard     # RQ Jobs: http://localhost:9181

# Cleanup
./manage.sh [profile] clean
```

**Examples:**
```bash
./manage.sh start                        # Start with CPU worker (default)
./manage.sh nvidia start                 # Start with NVIDIA GPU
./manage.sh amd logs video-worker-amd    # View AMD worker logs
./manage.sh mac shell-worker             # Shell into Mac worker
./manage.sh nvidia test-upload video.mp4 # Upload with NVIDIA profile
```

### Storage Architecture

**Redis** (Job Queue):
- Stores pending video processing jobs
- Tracks job status and progress
- Port: `6379`

**MinIO** (S3-Compatible Storage):
- Stores uploaded videos in `uploads/{job_id}/`
- Stores results in `results/{job_id}/`
- Web Console: `http://localhost:9001` (minioadmin/minioadmin)
- API: `http://localhost:9000`

### Data Flow

```
1. User uploads video → Backend API
2. Backend saves to MinIO S3
3. Backend queues job in Redis
4. Worker picks job from Redis
5. Worker downloads video from MinIO
6. Worker processes (pose estimation)
7. Worker uploads results to MinIO
8. Worker updates Redis with status
9. User polls status, downloads results
```
