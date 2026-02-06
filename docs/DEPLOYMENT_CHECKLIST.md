# Deployment Checklist

A comprehensive checklist for deploying the Dance Analysis Server to production.

## Pre-Deployment

### Environment Setup

- [ ] Copy `.env.example` to `.env`
- [ ] Set `CLOUDFLARE_TUNNEL_TOKEN` with your actual tunnel token
- [ ] Configure `CORS_ORIGINS` with your production domains
- [ ] Set `USE_MOCK_ANALYSIS=false` for production
- [ ] Review all environment variables in `.env`

### Cloudflare Tunnel Setup

- [ ] Create Cloudflare account (if needed)
- [ ] Add your domain to Cloudflare
- [ ] Create tunnel in Zero Trust dashboard
- [ ] Name tunnel appropriately (e.g., `dance-api-prod`)
- [ ] Copy tunnel token to `.env`
- [ ] Configure public hostname in Cloudflare:
  - [ ] Subdomain: `api`
  - [ ] Domain: your production domain
  - [ ] Type: `HTTP`
  - [ ] URL: `backend:8000` (NOT localhost!)
- [ ] Test DNS resolution for your domain

### Docker & Infrastructure

- [ ] Install Docker and Docker Compose
- [ ] Verify Docker daemon is running
- [ ] Choose appropriate GPU profile (cpu/nvidia/amd/mac)
- [ ] Test Docker can pull images
- [ ] Ensure sufficient disk space (20GB+ recommended)
- [ ] Ensure sufficient RAM (8GB+ recommended)

### GPU Setup (if applicable)

#### NVIDIA

- [ ] Install NVIDIA drivers
- [ ] Install nvidia-docker2
- [ ] Test: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`

#### AMD

- [ ] Install ROCm drivers
- [ ] Verify GPU access: `ls -la /dev/kfd /dev/dri`

#### Mac Apple Silicon

- [ ] Verify Apple Silicon chip (M1/M2/M3)
- [ ] Enable GPU support in Docker Desktop settings

## Deployment

### Build and Start Services

- [ ] Pull latest code from repository
- [ ] Review `docker-compose.yml` configuration
- [ ] Build images: `docker compose build`
- [ ] Start services with appropriate profiles:
  ```bash
  # Example: NVIDIA with tunnel
  docker compose --profile nvidia --profile tunnel up -d
  ```
- [ ] Verify all containers are running: `docker compose ps`
- [ ] Check container health: `docker compose ps | grep healthy`

### Service Verification

- [ ] **Backend API**:
  - [ ] Container running: `docker compose ps backend`
  - [ ] Health check passes: `curl http://localhost:8000/api/v1/health`
  - [ ] API docs accessible: `http://localhost:8000/docs`

- [ ] **Redis**:
  - [ ] Container running: `docker compose ps redis`
  - [ ] Health check passes: `docker compose exec redis redis-cli ping`

- [ ] **MinIO**:
  - [ ] Container running: `docker compose ps minio`
  - [ ] Console accessible: `http://localhost:9001`
  - [ ] Bucket created: Check in MinIO console
  - [ ] Login works (minioadmin/minioadmin)

- [ ] **Video Worker**:
  - [ ] Container running: `docker compose ps video-worker-*`
  - [ ] Listening to queue: Check logs for "Listening on"
  - [ ] GPU accessible (if applicable): Check logs for device detection

- [ ] **Cloudflare Tunnel**:
  - [ ] Container running: `docker compose ps cloudflared`
  - [ ] Connected: Logs show "Registered tunnel connection"
  - [ ] No errors in logs

### Network & Connectivity

- [ ] Verify Docker network exists: `docker network ls | grep dance-network`
- [ ] Test backend can reach Redis:
  ```bash
  docker compose exec backend ping redis
  ```
- [ ] Test backend can reach MinIO:
  ```bash
  docker compose exec backend curl http://minio:9000/minio/health/live
  ```
- [ ] Test worker can reach Redis:
  ```bash
  docker compose exec video-worker-* ping redis
  ```

### Public API Testing

- [ ] Domain resolves: `nslookup api.yourdomain.com`
- [ ] HTTPS works: `curl https://api.yourdomain.com/api/v1/health`
- [ ] Returns healthy status: `{"status":"healthy"}`
- [ ] API docs accessible: `https://api.yourdomain.com/docs`
- [ ] CORS headers present: Check browser developer tools

### End-to-End Testing

- [ ] Upload test video via public API:
  ```bash
  curl -X POST https://api.yourdomain.com/api/v1/analyze \
    -F "file=@test_video.mp4"
  ```
- [ ] Receive job_id in response
- [ ] Check job status shows "queued" or "processing"
- [ ] Worker picks up job (check logs)
- [ ] Job completes successfully
- [ ] Results accessible via API
- [ ] Can download visualization video

### Monitoring Setup

- [ ] RQ Dashboard accessible: `http://localhost:9181` (or via tunnel)
- [ ] MinIO Console accessible for storage monitoring
- [ ] Container logs accessible: `docker compose logs`
- [ ] Set up log rotation (if needed)
- [ ] Configure monitoring alerts (optional)

## Security Hardening

### Before Going Live

- [ ] **Change default credentials**:
  - [ ] MinIO: Change from minioadmin/minioadmin
  - [ ] Update S3_ACCESS_KEY and S3_SECRET_KEY in `.env`
  - [ ] Restart services after changing

- [ ] **CORS Configuration**:
  - [ ] Remove localhost URLs from production CORS_ORIGINS
  - [ ] Only include production domains
  - [ ] Verify CORS_ORIGINS in `.env` matches deployed domains

- [ ] **Cloudflare Security**:
  - [ ] Enable Cloudflare WAF (Web Application Firewall)
  - [ ] Set up rate limiting rules
  - [ ] Configure bot protection
  - [ ] Enable DDoS protection settings

- [ ] **Access Control**:
  - [ ] Set up Cloudflare Access for admin panels (MinIO, RQ Dashboard)
  - [ ] Configure authentication policies
  - [ ] Test access restrictions work

- [ ] **Network Security**:
  - [ ] Verify no unnecessary ports exposed
  - [ ] Only port 8000 should be accessible via tunnel
  - [ ] MinIO and Redis should NOT be publicly accessible

### Future Security (Post-Launch)

- [ ] Implement API key authentication
- [ ] Add request signing
- [ ] Set up API usage quotas
- [ ] Implement user authentication (if needed)
- [ ] Add audit logging
- [ ] Regular security updates for Docker images

## Performance Optimization

- [ ] **Docker Resources**:
  - [ ] Set memory limits in docker-compose.yml (if needed)
  - [ ] Set CPU limits (if needed)
  - [ ] Configure restart policies

- [ ] **Storage**:
  - [ ] Set up storage cleanup policies
  - [ ] Configure S3 lifecycle rules (if using AWS S3)
  - [ ] Monitor disk usage: `docker system df`

- [ ] **Scaling** (optional):
  - [ ] Scale workers: `docker compose up -d --scale video-worker-*=3`
  - [ ] Test load balancing
  - [ ] Monitor queue depth

## Backup & Recovery

- [ ] **Data Backup**:
  - [ ] Document backup strategy for MinIO/S3 data
  - [ ] Document Redis persistence settings
  - [ ] Test restore procedure

- [ ] **Configuration Backup**:
  - [ ] Commit `.env.example` to git (WITHOUT secrets!)
  - [ ] Document tunnel configuration
  - [ ] Save Cloudflare settings

- [ ] **Disaster Recovery**:
  - [ ] Document restart procedure
  - [ ] Test container restart: `docker compose restart`
  - [ ] Test full redeploy procedure

## Documentation

- [ ] Document production environment variables
- [ ] Document any customizations made
- [ ] Create runbook for common operations
- [ ] Document troubleshooting steps
- [ ] Share API documentation with team
- [ ] Create onboarding guide for new developers

## Monitoring & Maintenance

### Daily

- [ ] Check service health: `docker compose ps`
- [ ] Monitor disk space: `df -h`
- [ ] Review error logs: `docker compose logs --tail=100`

### Weekly

- [ ] Review Cloudflare analytics
- [ ] Check RQ Dashboard for failed jobs
- [ ] Monitor resource usage: `docker stats`
- [ ] Review MinIO storage usage

### Monthly

- [ ] Update Docker images: `docker compose pull`
- [ ] Clean up old data: `docker system prune`
- [ ] Review and rotate logs
- [ ] Test backup restoration
- [ ] Review security settings

## Rollback Plan

In case of issues:

1. [ ] Stop services: `docker compose down`
2. [ ] Checkout previous working version from git
3. [ ] Restore previous `.env` configuration
4. [ ] Rebuild and restart: `docker compose up -d --build`
5. [ ] Verify rollback successful
6. [ ] Document what went wrong

## Post-Deployment

- [ ] Monitor for first 24 hours
- [ ] Test with real users/clients
- [ ] Gather performance metrics
- [ ] Document any issues encountered
- [ ] Update team on deployment status
- [ ] Schedule follow-up review

## Production URLs

Document your deployed URLs:

- **API Endpoint**: `https://api.yourdomain.com`
- **API Documentation**: `https://api.yourdomain.com/docs`
- **MinIO Console**: `https://minio.yourdomain.com` (if exposed)
- **RQ Dashboard**: `https://jobs.yourdomain.com` (if exposed)
- **Cloudflare Dashboard**: `https://one.dash.cloudflare.com/`

## Support Contacts

- **Cloud Infrastructure**: [Team/Person]
- **API Development**: [Team/Person]
- **Video Processing**: [Team/Person]
- **Cloudflare Support**: https://support.cloudflare.com/

## Compliance & Legal

- [ ] Review data retention policies
- [ ] Ensure GDPR compliance (if applicable)
- [ ] Document data processing procedures
- [ ] Review terms of service for uploaded content
- [ ] Ensure privacy policy covers video uploads

---

## Quick Commands Reference

```bash
# Start production
docker compose --profile nvidia --profile tunnel up -d

# View logs
docker compose logs -f

# Restart a service
docker compose restart backend

# Stop all services
docker compose down

# Update and restart
git pull
docker compose up -d --build

# Check health
curl https://api.yourdomain.com/api/v1/health

# Monitor resources
docker stats

# Clean up
docker system prune -a
```

---

**Deployment Date**: ___________

**Deployed By**: ___________

**Production Version**: ___________

**Notes**: ___________
