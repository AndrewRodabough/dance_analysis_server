#!/bin/bash
# Local development script - runs API with mock analysis (no GPU/Docker required)

export USE_MOCK_ANALYSIS=true
c
echo "Starting server in MOCK mode (no GPU required)..."
echo "API will be available at http://localhost:8000"
echo "Docs available at http://localhost:8000/docs"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
