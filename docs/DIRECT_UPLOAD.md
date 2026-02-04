# Direct Upload to S3

## Overview

The Dance Analysis API now supports **direct-to-bucket uploads** using presigned URLs. This method is more efficient than the legacy upload approach because:

- ✅ **Reduces server load** - Videos bypass the API server
- ✅ **Faster uploads** - Direct connection to S3/MinIO
- ✅ **Better scalability** - No file size limits from API server
- ✅ **Lower bandwidth costs** - No double transfer (client → server → S3)

## Architecture

### Legacy Upload Flow (Deprecated)
```
Client → API Server → S3 → Processing Queue
         (uploads entire file to memory)
```

### Direct Upload Flow (Recommended)
```
Client → API Server (get presigned URL)
Client → S3 (direct upload)
Client → API Server (confirm & trigger processing)
```

## API Endpoints

### 1. Request Presigned Upload URL

**Endpoint:** `POST /api/v1/analyze/upload-url`

**Parameters:**
- `filename` (required): Name of the video file (e.g., "dance_video.mp4")
- `content_type` (optional): MIME type (default: "video/mp4")

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "upload_url": "https://s3.amazonaws.com/bucket/uploads/...",
  "s3_key": "uploads/550e8400-e29b-41d4-a716-446655440000/dance_video.mp4",
  "expires_in": 900,
  "instructions": "PUT the video file to upload_url, then call /analyze/confirm with job_id"
}
```

**Note:** The presigned URL expires in 15 minutes (900 seconds).

### 2. Upload Video to S3

**Endpoint:** Use the `upload_url` from step 1

**Method:** `PUT`

**Headers:**
- `Content-Type: video/mp4` (or appropriate video MIME type)

**Body:** Raw video file bytes

**Example with curl:**
```bash
curl -X PUT \
  -H "Content-Type: video/mp4" \
  --data-binary @dance_video.mp4 \
  "PRESIGNED_URL_HERE"
```

### 3. Confirm Upload and Start Processing

**Endpoint:** `POST /api/v1/analyze/confirm`

**Parameters:**
- `job_id` (required): Job ID from step 1
- `s3_key` (required): S3 key from step 1

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "stage": "pose-estimation",
  "s3_key": "uploads/550e8400-e29b-41d4-a716-446655440000/dance_video.mp4",
  "message": "Video confirmed and queued for pose estimation"
}
```

## Client Implementation

### Python Example

The `scripts/dev/upload_video.py` script supports direct upload:

```bash
# Direct upload (recommended)
python scripts/dev/upload_video.py path/to/video.mp4 --direct

# Direct upload with wait for completion
python scripts/dev/upload_video.py path/to/video.mp4 --direct --wait
```

### JavaScript/TypeScript Example

```typescript
async function uploadVideoDirect(videoFile: File): Promise<string> {
  // Step 1: Request presigned URL
  const urlResponse = await fetch(
    `${API_BASE}/api/v1/analyze/upload-url?filename=${encodeURIComponent(videoFile.name)}&content_type=${encodeURIComponent(videoFile.type)}`,
    { method: 'POST' }
  );
  const { job_id, upload_url, s3_key } = await urlResponse.json();
  
  // Step 2: Upload directly to S3
  await fetch(upload_url, {
    method: 'PUT',
    body: videoFile,
    headers: {
      'Content-Type': videoFile.type
    }
  });
  
  // Step 3: Confirm upload and start processing
  const confirmResponse = await fetch(
    `${API_BASE}/api/v1/analyze/confirm?job_id=${job_id}&s3_key=${encodeURIComponent(s3_key)}`,
    { method: 'POST' }
  );
  const result = await confirmResponse.json();
  
  return job_id;
}
```

### cURL Example

```bash
# Step 1: Get presigned URL
RESPONSE=$(curl -X POST "http://localhost:8000/api/v1/analyze/upload-url?filename=dance.mp4&content_type=video/mp4")
JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
UPLOAD_URL=$(echo $RESPONSE | jq -r '.upload_url')
S3_KEY=$(echo $RESPONSE | jq -r '.s3_key')

# Step 2: Upload video
curl -X PUT \
  -H "Content-Type: video/mp4" \
  --data-binary @dance.mp4 \
  "$UPLOAD_URL"

# Step 3: Confirm and start processing
curl -X POST "http://localhost:8000/api/v1/analyze/confirm?job_id=$JOB_ID&s3_key=$S3_KEY"
```

## Error Handling

### Common Errors

**1. Presigned URL Expired**
- Error: Upload fails with 403 Forbidden
- Solution: Request a new presigned URL (they expire after 15 minutes)

**2. Upload Not Confirmed**
- Error: `404 Not Found` when calling `/confirm`
- Solution: Ensure the video was successfully uploaded to S3 in step 2

**3. Invalid File Type**
- Error: `400 Bad Request` - Unsupported file type
- Solution: Use supported formats: MP4, AVI, MOV

**4. S3 Connection Issues**
- Error: Upload to presigned URL times out or fails
- Solution: Check network connectivity and S3/MinIO service status

## Security Considerations

1. **Presigned URLs are temporary** - They expire after 15 minutes
2. **One-time use** - Each upload requires a new presigned URL
3. **Validation** - The API validates that the file exists in S3 before processing
4. **Authentication** - In production, add authentication to the `/upload-url` endpoint

## Legacy Upload Method

The legacy upload method (`POST /api/v1/analyze` with multipart form data) is still available but **deprecated**. It will continue to work for backwards compatibility but is not recommended for new implementations.

## Migration Guide

To migrate from legacy upload to direct upload:

**Before (Legacy):**
```python
with open("video.mp4", "rb") as f:
    files = {"file": ("video.mp4", f, "video/mp4")}
    response = requests.post(f"{API_BASE}/api/v1/analyze", files=files)
```

**After (Direct Upload):**
```python
# 1. Request upload URL
response = requests.post(
    f"{API_BASE}/api/v1/analyze/upload-url",
    params={"filename": "video.mp4", "content_type": "video/mp4"}
)
data = response.json()

# 2. Upload to S3
with open("video.mp4", "rb") as f:
    requests.put(data["upload_url"], data=f, headers={"Content-Type": "video/mp4"})

# 3. Confirm
response = requests.post(
    f"{API_BASE}/api/v1/analyze/confirm",
    params={"job_id": data["job_id"], "s3_key": data["s3_key"]}
)
```

## Performance Comparison

| Metric | Legacy Upload | Direct Upload |
|--------|--------------|---------------|
| Upload Speed | ~10 MB/s | ~50+ MB/s |
| Server Memory | Entire file loaded | Minimal |
| Network Hops | 2 (client→server→S3) | 1 (client→S3) |
| API Server CPU | High | Low |
| Scalability | Limited by server | Unlimited |

## Questions?

For issues or questions, please refer to the main [README.md](../README.md) or check the API documentation at `http://localhost:8000/docs` when the server is running.
