# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is the backend server for the dance analysis system. It's a FastAPI application that performs pose estimation on uploaded dance videos using MediaPipe and generates both 2D and 3D visualization videos. The server is designed to run in Docker with GPU support for efficient video processing.

## Development Setup

### Running with Docker (Recommended)

```bash
# Build and start the server with hot-reload enabled
docker-compose up --build

# Stop the server
docker-compose down
```

The server runs on `http://localhost:8000` with the following endpoints:
- `GET /` - Service status
- `GET /health` - Health check
- `POST /analyze` - Upload video for analysis

### Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server with hot-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

### API Testing

Use the provided test script to upload a video for analysis:

```bash
# From outside the container
python test_scripts/test_api_video_upload.py /path/to/video.mp4
```

The test script sends a POST request to `/analyze` endpoint. Analysis results are saved to the `test_scripts/` directory:
- `estimation_2d.json` - 2D pose keypoints for all frames
- `estimation_3d.json` - 3D pose keypoints for all frames
- `pose_visualization.avi` - Video with 2D skeleton overlay
- `pose_visualization_3d_sidebyside.avi` - Side-by-side 2D and 3D visualization

Note: There are no automated unit tests in this project yet.

## Architecture

### Application Structure

```
app/
├── main.py                    # FastAPI application factory, router registration
├── api/v1/                    # API endpoints (versioned)
│   ├── health.py              # Health check endpoints
│   └── analyze.py             # Video upload and analysis endpoint
└── analysis/                  # Core analysis logic
    ├── analysis.py            # Main analysis orchestrator
    ├── pose_estimation/       
    │   ├── pose_estimation.py # MediaPipe-based pose extraction
    │   └── config.yaml        # Model configurations (currently unused - MediaPipe is used instead)
    └── video_generation.py    # Generate visualization videos with skeleton overlays
```

### Key Components

**Video Analysis Pipeline** (in `app/analysis/analysis.py`):
1. Receives video filepath from API endpoint
2. Calls `pose_estimation()` to extract 2D and 3D keypoints for each frame
3. Saves keypoint data as JSON files
4. Generates visualization videos using `generate_visualization_videos()`

**Pose Estimation** (in `app/analysis/pose_estimation/pose_estimation.py`):
- Currently uses MediaPipe Pose (not the MMPose models referenced in config.yaml)
- Processes video frame-by-frame
- Returns 2D keypoints (pixel coordinates) and 3D keypoints (world coordinates in meters)
- MediaPipe outputs 33 keypoints per person per frame

**Video Generation** (in `app/analysis/video_generation.py`):
- Takes original video and keypoint data
- Draws skeleton connections based on COCO wholebody format
- Generates two output videos:
  - 2D overlay on original video
  - Side-by-side with 3D skeleton visualization

### Data Flow

1. Client uploads video via `/analyze` endpoint → saved to `uploads/` directory
2. `analyze_video()` orchestrates the pipeline
3. Pose estimation extracts keypoints from each frame
4. Results saved to `test_scripts/` directory (both JSON and video files)
5. API returns success response with file location

### Models Directory

The `models/` directory contains MMPose/MMDetection configuration files and is referenced in `config.yaml`, but the current implementation uses MediaPipe instead. This suggests a potential migration path or alternative implementation.

## Docker Configuration

- Base image: `ubuntu:latest`
- GPU support: Configured for NVIDIA GPUs via `docker-compose.yml`
- Shared memory: 8GB allocated for video processing
- Volume mount: Project directory mounted to `/workspace` in container
- Hot-reload: Enabled via uvicorn's `--reload` flag in docker-compose

## Important Notes

- Uploaded videos are stored in `uploads/` directory (created automatically)
- Analysis outputs are saved to `test_scripts/` directory
- The config.yaml references MMPose models but MediaPipe is actually used for pose estimation
- No authentication or file cleanup mechanisms are currently implemented
