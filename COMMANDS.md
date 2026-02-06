# Quick Command Reference

## Start Services

```bash
# Local development (CPU only, no tunnel)
docker compose --profile cpu up -d

# Local with NVIDIA GPU
docker compose --profile nvidia up -d

# Production with Cloudflare Tunnel (CPU)
docker compose --profile cpu --profile tunnel up -d

# Production with Cloudflare Tunnel (NVIDIA)
docker compose --profile nvidia --profile tunnel up -d

# Production with Cloudflare Tunnel (AMD)
docker compose --profile amd --profile tunnel up -d

# Production with Cloudflare Tunnel (Mac)
docker compose --profile mac --profile tunnel up -d
```

## Stop Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

## View Logs

```bash
# All services
docker compose logs

# Follow logs in real-time
docker compose logs -f

# Specific service
docker compose logs backend
docker compose logs cloudflared
docker compose logs video-worker-cpu

# Follow specific service
docker compose logs -f backend
```

## Check Status

```bash
# List all containers
docker compose ps

# Check container health
docker compose ps | grep healthy

# View resource usage
docker stats
```

## Test API

```bash
# Health check (local)
curl http://localhost:8000/api/v1/health

# Health check (public - replace with your domain)
curl https://api.yourdomain.com/api/v1/health

# Upload video (local)
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@video.mp4"

# Upload video (public)
curl -X POST https://api.yourdomain.com/api/v1/analyze \
  -F "file=@video.mp4"

# Check job status
curl http://localhost:8000/api/v1/analyze/{job_id}/status

# Get results
curl http://localhost:8000/api/v1/analyze/{job_id}/result
```

## Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart backend
docker compose restart cloudflared
docker compose restart video-worker-cpu
```

## Update and Rebuild

```bash
# Pull latest code
git pull

# Rebuild images
docker compose build

# Rebuild and restart
docker compose up -d --build
```

## Troubleshooting

```bash
# Check backend can reach Redis
docker compose exec backend ping redis

# Check backend can reach MinIO
docker compose exec backend curl http://minio:9000/minio/health/live

# Get shell in backend container
docker compose exec backend /bin/bash

# Get shell in worker container
docker compose exec video-worker-cpu /bin/bash

# View Docker networks
docker network ls

# Inspect dance network
docker network inspect dance_dance-network

# Check disk usage
docker system df

# Clean up unused resources
docker system prune

# Clean up everything (careful!)
docker system prune -a --volumes
```

## Cloudflare Tunnel

```bash
# View tunnel logs
docker compose logs cloudflared

# Follow tunnel logs
docker compose logs -f cloudflared

# Restart tunnel
docker compose restart cloudflared

# Check tunnel status in logs
docker compose logs cloudflared | grep "Registered"
```

## Monitoring Dashboards

```bash
# MinIO Console
open http://localhost:9001
# Login: minioadmin / minioadmin

# RQ Dashboard (job queue)
open http://localhost:9181

# API Documentation (local)
open http://localhost:8000/docs

# API Documentation (public)
open https://api.yourdomain.com/docs
```

## Environment Setup

```bash
# Create .env from example
cp .env.example .env

# Edit environment variables
nano .env

# View current environment (don't print secrets!)
cat .env
```

## Docker Compose Profiles

Available profiles:
- `cpu` - Use CPU for video processing
- `nvidia` - Use NVIDIA GPU
- `amd` - Use AMD GPU  
- `mac` - Use Apple Silicon GPU
- `tunnel` - Enable Cloudflare Tunnel
- `production` - Production mode (includes tunnel)

Combine profiles with `--profile`:
```bash
docker compose --profile nvidia --profile tunnel up -d
```

## Quick Diagnostics

```bash
# Everything in one command
echo "=== Services ===" && \
docker compose ps && \
echo -e "\n=== Tunnel Status ===" && \
docker compose logs cloudflared | tail -5 && \
echo -e "\n=== Backend Health ===" && \
curl -s http://localhost:8000/api/v1/health && \
echo -e "\n\n=== Redis ===" && \
docker compose exec redis redis-cli ping && \
echo -e "\n=== Disk Usage ===" && \
docker system df
```

## Backup & Restore

```bash
# Backup MinIO data
docker compose exec minio mc mirror /data /backup

# Export Redis data
docker compose exec redis redis-cli --rdb /data/dump.rdb

# Backup volumes
docker run --rm -v dance_minio-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/minio-backup.tar.gz /data
```

## Scaling Workers

```bash
# Scale to 3 workers
docker compose --profile cpu up -d --scale video-worker-cpu=3

# Check worker count
docker compose ps | grep video-worker
```
