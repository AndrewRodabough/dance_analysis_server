# 🎬 Dance Analysis Server with MinIO - START HERE

## What You Have

A complete, production-ready containerized system for video dance analysis with:
- ✅ REST API backend (FastAPI)
- ✅ Async video processing (RQ + Redis)
- ✅ S3-compatible storage (MinIO)
- ✅ GPU support
- ✅ Web dashboards for monitoring
- ✅ Works on single machine OR across networks

## Quick Start (60 seconds)

### Option 1: Automated Setup
```bash
chmod +x quick-start.sh
./quick-start.sh
```

### Option 2: Manual Setup
```bash
docker-compose up -d
```

### Verify It Works
```bash
curl http://localhost:8000/api/v1/health
```

## What Just Started

| Service | URL | Purpose |
|---------|-----|---------|
| Backend API | http://localhost:8000 | Upload videos, get results |
| MinIO Console | http://localhost:9001 | Browse files (minioadmin/minioadmin) |
| RQ Dashboard | http://localhost:9181 | Monitor video processing |
| Redis | localhost:6379 | Job queue |

## First Steps

### 1. Create a Test Video
```bash
ffmpeg -f lavfi -i color=c=blue:s=640x480:d=5 \
        -f lavfi -i sine=f=1000:d=5 \
        -pix_fmt yuv420p test_video.mp4
```

### 2. Upload for Analysis
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@test_video.mp4"
```

Save the returned `job_id`.

### 3. Check Progress
```bash
curl http://localhost:8000/api/v1/analyze/{job_id}/status
```

### 4. Get Results
```bash
curl http://localhost:8000/api/v1/analyze/{job_id}/result
```

## Documentation

- **MINIO_SETUP.md** - Complete setup guide
- **TESTING.md** - Comprehensive testing instructions
- **SETUP.md** - Detailed configuration and troubleshooting
- **API Docs** - http://localhost:8000/docs (Swagger)

## Helper Commands

```bash
# Manage services
./manage.sh start          # Start all services
./manage.sh stop           # Stop all services
./manage.sh logs           # View all logs
./manage.sh logs backend   # View specific service

# Dashboards
./manage.sh minio-console  # MinIO info
./manage.sh rq-dashboard   # RQ info
./manage.sh redis-cli      # Redis console

# Testing
./manage.sh test-upload video.mp4  # Test video upload

# Containers
./manage.sh shell-backend  # Shell into backend
./manage.sh shell-worker   # Shell into worker
```

## Architecture

```
User Request
    ↓
Backend API (port 8000)
    ↓
Upload to MinIO S3 (port 9000)
    ↓
Queue Job in Redis (port 6379)
    ↓
Video Worker picks up job
    ↓
Download from S3 → Process → Upload Results
    ↓
User retrieves results via API
```

## Common Tasks

### Upload a Video
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@video.mp4"
```

### Check Job Status
```bash
curl http://localhost:8000/api/v1/analyze/{job_id}/status
```

### Download Results
```bash
curl http://localhost:8000/api/v1/analyze/{job_id}/result
```

### View Worker Logs
```bash
docker-compose logs -f video-worker
```

### View All Uploaded Files
```bash
# Open MinIO Console
open http://localhost:9001
# Login: minioadmin / minioadmin
```

### Monitor Job Queue
```bash
# Open RQ Dashboard
open http://localhost:9181
```

## System Requirements

- Docker and Docker Compose
- 8GB+ RAM
- 20GB+ disk space (for MinIO storage)
- Optional: NVIDIA GPU + Docker runtime (for GPU acceleration)

## Production Deployment

To deploy to AWS S3 instead of MinIO:

1. Update `.env`:
```env
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=your-aws-key
S3_SECRET_KEY=your-aws-secret
S3_BUCKET=your-bucket
```

2. Remove volume mounts from `docker-compose.yml`

3. Scale workers: `docker-compose up -d --scale video-worker=3`

## Troubleshooting

### Services Not Starting
```bash
docker-compose logs
```

### Worker Not Processing
```bash
docker-compose logs video-worker
docker-compose restart video-worker
```

### Storage Issues
```bash
# Clear old jobs
docker-compose down -v
docker-compose up -d
```

## File Structure

```
dance_analysis_server/
├── backend/                    # FastAPI server
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       └── api/v1/
│           ├── analyze.py      # Video upload/queue
│           ├── videos.py       # Result downloads
│           └── health.py
├── video_processing/           # RQ worker
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── tasks.py               # Video processing logic
│   └── app/                   # Existing analysis code
├── docker-compose.yml         # Service orchestration
├── quick-start.sh            # Automated setup
├── manage.sh                 # Helper commands
└── MINIO_SETUP.md           # Full documentation
```

## What Happens When You Upload

1. **Backend receives video** → Validates file type
2. **Uploads to MinIO S3** → Object storage
3. **Creates RQ job** → Adds to Redis queue
4. **Returns job_id** → For status tracking
5. **Worker picks up job** → From Redis queue
6. **Downloads video** → From MinIO S3
7. **Processes video** → Pose estimation
8. **Uploads results** → Back to MinIO S3
9. **User retrieves** → Via pre-signed URLs

## API Endpoints

```
POST   /api/v1/analyze              # Upload video
GET    /api/v1/analyze/{id}/status  # Check progress
GET    /api/v1/analyze/{id}/result  # Get results
DELETE /api/v1/analyze/{id}         # Delete job

GET    /api/v1/videos/{id}/visualization  # Download viz
GET    /api/v1/videos/{id}/keypoints_2d   # Download 2D
GET    /api/v1/videos/{id}/keypoints_3d   # Download 3D
GET    /api/v1/videos/{id}/original       # Download original

GET    /api/v1/health               # Health check
```

Full API docs: http://localhost:8000/docs

## Next Steps

1. ✅ Run `./quick-start.sh` to start
2. ✅ Read **TESTING.md** for comprehensive testing
3. ✅ Read **SETUP.md** for production deployment
4. ✅ Check **MINIO_SETUP.md** for detailed configuration

## Support

**Stuck?** Check these in order:

1. `docker-compose logs` - View all service logs
2. http://localhost:9181 - RQ Dashboard (job status)
3. http://localhost:9001 - MinIO Console (files)
4. `docker stats` - Check system resources
5. **SETUP.md** - Troubleshooting section

---

**Status**: ✅ Ready to run!

Run `./quick-start.sh` now to get started! 🚀

