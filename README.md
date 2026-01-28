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
