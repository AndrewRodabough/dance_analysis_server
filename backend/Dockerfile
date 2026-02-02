# base image
FROM python:3.10-slim

# Environment Variables
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1
# Prevents apt-get from asking for user input during install
ENV DEBIAN_FRONTEND=noninteractive

# System Dependencies (The "Heavy" Layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Work inside the server app
WORKDIR /app

# Install Python dependencies: FastAPI + RQ + boto3
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# using uvicorn for FastAPI listening on all interfaces (0.0.0.0)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
