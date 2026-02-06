# Cloudflare Tunnel Setup Guide

This guide walks you through setting up Cloudflare Tunnel to expose your Dance Analysis API to the internet securely.

## What is Cloudflare Tunnel?

Cloudflare Tunnel creates a secure, outbound-only connection from your server to Cloudflare's edge network. This means:

- ✅ No need to open firewall ports
- ✅ No need for a public IP address
- ✅ DDoS protection from Cloudflare
- ✅ Free SSL/TLS certificates
- ✅ Works behind NAT/firewalls
- ✅ Built-in load balancing and failover

## Prerequisites

- A domain name managed by Cloudflare (free tier works!)
- Docker and Docker Compose installed
- Your Dance Analysis Server running

## Quick Start (5 Minutes)

### Step 1: Create a Cloudflare Tunnel

1. **Go to Cloudflare Zero Trust Dashboard**
   - Visit: https://one.dash.cloudflare.com/
   - Sign in with your Cloudflare account

2. **Navigate to Tunnels**
   - Click: **Zero Trust** → **Networks** → **Tunnels**

3. **Create a New Tunnel**
   - Click **"Create a tunnel"**
   - Select **"Cloudflared"** as the connector type
   - Name your tunnel: `dance-api`
   - Click **"Save tunnel"**

### Step 2: Get Your Tunnel Token

After creating the tunnel, Cloudflare shows installation instructions. You'll see a Docker command like:

```bash
docker run cloudflare/cloudflared:latest tunnel --no-autoupdate run --token eyJhIjoiXXXXXXXX...
```

**Copy the token** (the long string after `--token`). It starts with `eyJhIjoi...`

### Step 3: Configure Your Environment

1. **Copy the example environment file**:
   ```bash
   cd dance_analysis_server
   cp .env.example .env
   ```

2. **Edit `.env`** and add your tunnel token:
   ```env
   CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoiXXXXXXXX...your-actual-token
   ```

3. **Update CORS origins** with your domain:
   ```env
   CORS_ORIGINS=https://api.yourdomain.com,https://app.yourdomain.com,http://localhost:3000
   ```

### Step 4: Configure Public Hostname in Cloudflare

Back in the Cloudflare dashboard:

1. **Click on your tunnel** (`dance-api`)

2. **Go to the "Public Hostname" tab**

3. **Add a public hostname**:
   - **Subdomain**: `api` (or whatever you prefer)
   - **Domain**: Select your domain from the dropdown
   - **Type**: `HTTP`
   - **URL**: `backend:8000`
   
   ⚠️ **Important**: Use `backend:8000` NOT `localhost:8000`
   
   This uses Docker's internal network name.

4. **Click "Save hostname"**

Your API will now be accessible at `https://api.yourdomain.com`

### Step 5: Start Your Services

```bash
# Start with CPU worker and tunnel
docker compose --profile cpu --profile tunnel up -d

# Or with NVIDIA GPU
docker compose --profile nvidia --profile tunnel up -d

# Or with AMD GPU
docker compose --profile amd --profile tunnel up -d

# Or with Mac Apple Silicon
docker compose --profile mac --profile tunnel up -d
```

### Step 6: Verify It Works

1. **Check tunnel status**:
   ```bash
   docker compose logs cloudflared
   ```
   
   You should see: `Registered tunnel connection`

2. **Test the API**:
   ```bash
   curl https://api.yourdomain.com/api/v1/health
   ```
   
   Expected response:
   ```json
   {"status": "healthy"}
   ```

3. **View API docs**:
   ```
   https://api.yourdomain.com/docs
   ```

## Testing from Your Client Application

Once your tunnel is running, you can make API calls from anywhere:

### JavaScript/TypeScript Example

```javascript
const API_URL = 'https://api.yourdomain.com';

// Upload a video
async function uploadVideo(videoFile) {
  const formData = new FormData();
  formData.append('file', videoFile);
  
  const response = await fetch(`${API_URL}/api/v1/analyze`, {
    method: 'POST',
    body: formData,
  });
  
  const data = await response.json();
  return data.job_id;
}

// Check processing status
async function checkStatus(jobId) {
  const response = await fetch(`${API_URL}/api/v1/analyze/${jobId}/status`);
  const data = await response.json();
  return data.status; // 'queued', 'processing', 'completed', or 'failed'
}

// Get results
async function getResults(jobId) {
  const response = await fetch(`${API_URL}/api/v1/analyze/${jobId}/result`);
  return response.json();
}
```

### Python Example

```python
import requests

API_URL = "https://api.yourdomain.com"

# Upload a video
def upload_video(video_path):
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/v1/analyze", files=files)
        return response.json()['job_id']

# Check status
def check_status(job_id):
    response = requests.get(f"{API_URL}/api/v1/analyze/{job_id}/status")
    return response.json()['status']

# Get results
def get_results(job_id):
    response = requests.get(f"{API_URL}/api/v1/analyze/{job_id}/result")
    return response.json()
```

### cURL Example

```bash
# Upload a video
curl -X POST https://api.yourdomain.com/api/v1/analyze \
  -F "file=@dance_video.mp4"

# Check status (use job_id from above)
curl https://api.yourdomain.com/api/v1/analyze/{job_id}/status

# Get results
curl https://api.yourdomain.com/api/v1/analyze/{job_id}/result
```

## Optional: Expose Additional Services

You can expose other services through the same tunnel:

### MinIO Console (Admin Access)

In Cloudflare dashboard, add another public hostname:
- **Subdomain**: `minio`
- **Domain**: `yourdomain.com`
- **Type**: `HTTP`
- **URL**: `minio:9001`

Access at: `https://minio.yourdomain.com`

### RQ Dashboard (Job Monitoring)

Add another public hostname:
- **Subdomain**: `jobs`
- **Domain**: `yourdomain.com`
- **Type**: `HTTP`
- **URL**: `rq-dashboard:9181`

Access at: `https://jobs.yourdomain.com`

⚠️ **Security Note**: These admin interfaces should be protected! See the Security section below.

## Monitoring Your Tunnel

### View Tunnel Status in Cloudflare

1. Go to: **Zero Trust** → **Networks** → **Tunnels**
2. Click on your tunnel name (`dance-api`)
3. View:
   - Connection status (Healthy/Unhealthy)
   - Traffic analytics
   - Connected instances
   - Request logs

### View Local Logs

```bash
# View cloudflared logs
docker compose logs cloudflared

# Follow logs in real-time
docker compose logs -f cloudflared

# Check if tunnel is connected
docker compose ps cloudflared
```

## Troubleshooting

### Tunnel Not Connecting

**Symptom**: `cloudflared` logs show connection errors

**Solutions**:
1. Verify your tunnel token is correct in `.env`
2. Check if tunnel exists in Cloudflare dashboard
3. Restart the cloudflared container:
   ```bash
   docker compose restart cloudflared
   ```

### 502 Bad Gateway

**Symptom**: API returns 502 error

**Solutions**:
1. Verify backend container is running:
   ```bash
   docker compose ps backend
   ```
2. Check you used `backend:8000` not `localhost:8000` in Cloudflare config
3. Verify backend is healthy:
   ```bash
   docker compose exec backend curl http://localhost:8000/api/v1/health
   ```

### CORS Errors

**Symptom**: Browser shows CORS errors

**Solutions**:
1. Add your domain to `CORS_ORIGINS` in `.env`:
   ```env
   CORS_ORIGINS=https://api.yourdomain.com,https://app.yourdomain.com
   ```
2. Restart backend:
   ```bash
   docker compose restart backend
   ```

### DNS Not Resolving

**Symptom**: Domain doesn't resolve

**Solutions**:
1. Wait a few minutes for DNS propagation
2. Clear DNS cache:
   ```bash
   # macOS
   sudo dscacheutil -flushcache
   
   # Linux
   sudo systemd-resolve --flush-caches
   ```
3. Verify DNS in Cloudflare dashboard:
   - Go to: **DNS** → **Records**
   - Look for CNAME record pointing to your tunnel

## Advanced Configuration

### Using a Config File (Alternative to Token)

If you prefer more control, you can use a config file instead of a token:

1. **Install cloudflared locally**:
   ```bash
   # macOS
   brew install cloudflared
   
   # Linux
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared-linux-amd64.deb
   ```

2. **Authenticate**:
   ```bash
   cloudflared tunnel login
   ```

3. **Create tunnel**:
   ```bash
   cloudflared tunnel create dance-api
   ```

4. **Create config** at `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: <your-tunnel-id>
   credentials-file: /etc/cloudflared/<tunnel-id>.json
   
   ingress:
     - hostname: api.yourdomain.com
       service: http://backend:8000
       originRequest:
         noTLSVerify: true
         connectTimeout: 30s
     
     - hostname: minio.yourdomain.com
       service: http://minio:9001
     
     - hostname: jobs.yourdomain.com
       service: http://rq-dashboard:9181
     
     - service: http_status:404
   ```

5. **Update docker-compose.yml**:
   ```yaml
   cloudflared:
     image: cloudflare/cloudflared:latest
     command: tunnel --no-autoupdate run dance-api
     volumes:
       - ~/.cloudflared:/etc/cloudflared:ro
     networks:
       - dance-network
   ```

### Multiple Instances for High Availability

Run multiple cloudflared instances for redundancy:

```bash
# Scale to 2 instances
docker compose --profile tunnel up -d --scale cloudflared=2
```

Both instances will connect to Cloudflare, providing automatic failover.

## Security Best Practices

### 1. Protect Admin Interfaces

Add Cloudflare Access policies for MinIO and RQ Dashboard:

1. Go to: **Zero Trust** → **Access** → **Applications**
2. Click **"Add an application"**
3. Choose **"Self-hosted"**
4. Configure:
   - **Name**: MinIO Console
   - **Domain**: `minio.yourdomain.com`
5. Add a policy:
   - **Name**: Admins Only
   - **Action**: Allow
   - **Include**: Emails ending in `@yourcompany.com`

Repeat for `jobs.yourdomain.com`

### 2. Rate Limiting

Enable rate limiting in Cloudflare:

1. Go to: **Security** → **WAF** → **Rate limiting rules**
2. Create a rule to limit API requests per IP

### 3. Web Application Firewall (WAF)

Enable Cloudflare WAF for additional protection:

1. Go to: **Security** → **WAF**
2. Enable **Managed rules**

### 4. Add Authentication (Coming Soon)

After testing, add API key authentication to your backend. See `docs/SECURITY.md` (to be created).

## Cost

**Cloudflare Tunnel**: Free! ✨

Included in Cloudflare's free tier:
- Unlimited tunnels
- Unlimited bandwidth
- DDoS protection
- SSL/TLS certificates
- Basic analytics

## Next Steps

1. ✅ Test your API endpoint
2. ✅ Update your client application to use the new domain
3. ✅ Configure Cloudflare Access for admin interfaces
4. ✅ Enable WAF and rate limiting
5. ✅ Add API authentication (see future `docs/SECURITY.md`)
6. ✅ Set up monitoring and alerts

## Additional Resources

- [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/)
- [Cloudflare Community Forum](https://community.cloudflare.com/)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. View logs: `docker compose logs cloudflared`
3. Verify tunnel status in Cloudflare dashboard
4. Check Cloudflare's status page: https://www.cloudflarestatus.com/

---

**Status**: Ready to deploy! 🚀

Your API is now accessible from anywhere in the world with enterprise-grade security and performance.
