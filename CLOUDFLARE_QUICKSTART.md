# Cloudflare Tunnel Quick Start

Get your API online in 5 minutes! 🚀

## Prerequisites

- Cloudflare account (free tier works!)
- Domain managed by Cloudflare
- Docker Compose running

## Step-by-Step Setup

### 1. Create Tunnel in Cloudflare

1. Go to https://one.dash.cloudflare.com/
2. Navigate to: **Zero Trust** → **Networks** → **Tunnels**
3. Click **"Create a tunnel"**
4. Select **"Cloudflared"**
5. Name it: `dance-api`
6. Click **"Save tunnel"**

### 2. Get Your Token

After creating the tunnel, you'll see installation instructions with a Docker command:

```bash
docker run cloudflare/cloudflared:latest tunnel --no-autoupdate run --token eyJhIjoiXXXXXXXX...
```

**Copy the token** (the part after `--token`)

### 3. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env and paste your token
nano .env
```

Add this line to `.env`:
```env
CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoiXXXXXXXX...your-token-here
```

Update CORS origins with your domain:
```env
CORS_ORIGINS=https://api.yourdomain.com,https://app.yourdomain.com,http://localhost:3000
```

### 4. Configure Public Hostname

Back in Cloudflare dashboard:

1. Click on your tunnel (`dance-api`)
2. Go to **"Public Hostname"** tab
3. Click **"Add a public hostname"**
4. Fill in:
   - **Subdomain**: `api`
   - **Domain**: `yourdomain.com` (select from dropdown)
   - **Type**: `HTTP`
   - **URL**: `backend:8000` ⚠️ Important: NOT localhost!
5. Click **"Save hostname"**

### 5. Start Services

```bash
# CPU only
docker compose --profile cpu --profile tunnel up -d

# Or with NVIDIA GPU
docker compose --profile nvidia --profile tunnel up -d

# Or with AMD GPU
docker compose --profile amd --profile tunnel up -d

# Or with Mac Apple Silicon
docker compose --profile mac --profile tunnel up -d
```

### 6. Test It!

```bash
# Check tunnel status
docker compose logs cloudflared

# Should see: "Registered tunnel connection"

# Test the API
curl https://api.yourdomain.com/api/v1/health

# Expected: {"status":"healthy"}
```

## What You Just Did

✅ Created a secure tunnel from your server to Cloudflare  
✅ Got a public HTTPS endpoint with free SSL  
✅ Protected your server behind Cloudflare's network  
✅ No firewall configuration needed  
✅ No port forwarding required  

## Your API is Now Live! 🎉

Access your API at: `https://api.yourdomain.com`

### Available Endpoints

```bash
# Upload video
curl -X POST https://api.yourdomain.com/api/v1/analyze \
  -F "file=@video.mp4"

# Check status
curl https://api.yourdomain.com/api/v1/analyze/{job_id}/status

# Get results
curl https://api.yourdomain.com/api/v1/analyze/{job_id}/result

# API documentation
open https://api.yourdomain.com/docs
```

### Client Integration

```javascript
const API_URL = 'https://api.yourdomain.com';

async function uploadVideo(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/api/v1/analyze`, {
    method: 'POST',
    body: formData,
  });
  
  const { job_id } = await response.json();
  return job_id;
}
```

## Troubleshooting

### Tunnel won't connect
```bash
# Check logs
docker compose logs cloudflared

# Verify token in .env
cat .env | grep CLOUDFLARE_TUNNEL_TOKEN

# Restart tunnel
docker compose restart cloudflared
```

### 502 Bad Gateway
```bash
# Make sure you used backend:8000, not localhost:8000
# Check backend is running
docker compose ps backend

# Test backend health
docker compose exec backend curl http://localhost:8000/api/v1/health
```

### CORS errors in browser
```bash
# Add your domain to .env
echo "CORS_ORIGINS=https://api.yourdomain.com,https://app.yourdomain.com" >> .env

# Restart backend
docker compose restart backend
```

## Next Steps

1. ✅ Test your API endpoint
2. ✅ Update your client app to use new domain
3. ✅ Read full guide: [docs/CLOUDFLARE_TUNNEL.md](docs/CLOUDFLARE_TUNNEL.md)
4. ✅ Add authentication (after testing)
5. ✅ Enable rate limiting in Cloudflare
6. ✅ Set up Cloudflare Access for admin panels

## Optional: Expose Admin Panels

### MinIO Console

In Cloudflare, add another public hostname:
- **Subdomain**: `minio`
- **Domain**: `yourdomain.com`
- **URL**: `minio:9001`

Access: `https://minio.yourdomain.com`

### RQ Dashboard

Add another public hostname:
- **Subdomain**: `jobs`
- **Domain**: `yourdomain.com`
- **URL**: `rq-dashboard:9181`

Access: `https://jobs.yourdomain.com`

⚠️ **Protect these with Cloudflare Access!**

## Monitoring

View tunnel status:
1. Go to: **Zero Trust** → **Networks** → **Tunnels**
2. Click your tunnel
3. View connection status and traffic

## Need Help?

- Full documentation: [docs/CLOUDFLARE_TUNNEL.md](docs/CLOUDFLARE_TUNNEL.md)
- Cloudflare Docs: https://developers.cloudflare.com/cloudflare-one/
- Check logs: `docker compose logs cloudflared`

---

**You're done!** Your API is now accessible worldwide with enterprise-grade security. 🌍
