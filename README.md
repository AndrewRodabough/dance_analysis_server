# Backend For dance-analysis Project

## Quick Start

### Expose API to the Internet (Cloudflare Tunnel)

To make your API accessible from anywhere (for client applications):

```bash
# See detailed setup guide
cat docs/CLOUDFLARE_TUNNEL.md

# Quick setup:
# 1. Copy .env.example to .env
cp .env.example .env

# 2. Add your Cloudflare tunnel token to .env
# 3. Start with tunnel profile
docker compose --profile tunnel --profile cpu up -d
```

**Full Guide**: [docs/CLOUDFLARE_TUNNEL.md](docs/CLOUDFLARE_TUNNEL.md)

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

#### With Cloudflare Tunnel (Public API)

```bash
# Expose API to the internet securely
docker compose --profile cpu --profile tunnel up

# Or with GPU
docker compose --profile nvidia --profile tunnel up
```

See [docs/CLOUDFLARE_TUNNEL.md](docs/CLOUDFLARE_TUNNEL.md) for setup instructions.

## API Endpoints

### Video Upload

**Direct Upload (Recommended)** - See [docs/DIRECT_UPLOAD.md](docs/DIRECT_UPLOAD.md):

- `POST /api/v1/analyze/upload-url` - Request presigned S3 URL
- `PUT <presigned-url>` - Upload video directly to S3
- `POST /api/v1/analyze/confirm` - Confirm upload and start processing

**Legacy Upload** (Deprecated):

- `POST /api/v1/analyze` - Upload video through API server

### Status & Results

- `GET /api/v1/analyze/{job_id}/status` - Check processing status
- `GET /api/v1/analyze/{job_id}/result` - Get analysis results

### Health

- `GET /api/v1/health` - Health check

API docs: http://localhost:8000/docs

See [docs/DIRECT_UPLOAD.md](docs/DIRECT_UPLOAD.md) for detailed documentation on the direct upload flow.

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

**Direct Upload (Recommended):**

```
1. User requests presigned URL → Backend API
2. Backend generates presigned S3 URL (15 min expiry)
3. User uploads video directly → MinIO S3
4. User confirms upload → Backend API
5. Backend queues job in Redis
6. Worker picks job from Redis
7. Worker downloads video from MinIO
8. Worker processes (pose estimation)
9. Worker uploads results to MinIO
10. Worker updates Redis with status
11. User polls status, downloads results
```

**Legacy Upload (Deprecated):**

```
1. User uploads video → Backend API (loads into memory)
2. Backend saves to MinIO S3
3. Backend queues job in Redis
4. [continues same as above from step 6]
```

## Testing

Run unit tests for pose data structures and mappers:

```bash
# Using unittest (built-in)
python3 -m unittest discover tests

# Using pytest (recommended)
pip install pytest pytest-cov
pytest tests/ -v

# With coverage
pytest tests/ --cov=app.models --cov-report=html
```

See `tests/README.md` for detailed testing documentation.
