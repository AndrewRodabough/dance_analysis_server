# Quick Start: Direct Upload API

## 🚀 Using the New Direct Upload

The fastest way to upload videos is now **direct to S3**:

```bash
# Use the updated script with --direct flag
python scripts/dev/upload_video.py path/to/video.mp4 --direct --wait
```

## 📋 What Changed?

### Before (Legacy - Still works but not recommended)
```python
# Video goes through API server (slower, uses more memory)
files = {"file": ("video.mp4", file_handle, "video/mp4")}
response = requests.post("/api/v1/analyze", files=files)
```

### After (Direct Upload - Recommended)
```python
# Step 1: Get presigned URL
response = requests.post("/api/v1/analyze/upload-url?filename=video.mp4")
data = response.json()

# Step 2: Upload directly to S3 (fast!)
requests.put(data["upload_url"], data=video_bytes, 
             headers={"Content-Type": "video/mp4"})

# Step 3: Confirm and start processing
requests.post(f"/api/v1/analyze/confirm?job_id={data['job_id']}&s3_key={data['s3_key']}")
```

## 🎯 Benefits

| Feature | Legacy Upload | Direct Upload |
|---------|--------------|---------------|
| Speed | Slower | **2-5x faster** |
| Server Load | High | **Minimal** |
| File Size Limit | API server limit | **S3 limit (5TB)** |
| Memory Usage | Entire file in RAM | **Streaming** |
| Scalability | Limited | **Unlimited** |

## 📚 Full Documentation

See [docs/DIRECT_UPLOAD.md](DIRECT_UPLOAD.md) for:
- Complete API reference
- Error handling
- JavaScript/TypeScript examples
- cURL examples
- Security considerations

## 🔧 Testing

```bash
# Test direct upload
python scripts/dev/upload_video.py test_video.mp4 --direct

# Test legacy upload (for comparison)
python scripts/dev/upload_video.py test_video.mp4

# Check status
python scripts/dev/upload_video.py --status <job_id>

# Get results
python scripts/dev/upload_video.py --results <job_id>
```

## ⚠️ Migration Notes

- The legacy `POST /api/v1/analyze` endpoint still works but is **deprecated**
- All new integrations should use the direct upload flow
- Presigned URLs expire after **15 minutes**
- The API validates file existence in S3 before processing
