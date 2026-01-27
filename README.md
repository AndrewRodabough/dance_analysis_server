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