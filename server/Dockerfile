# Start from your heavy AI engine
FROM alphapose_ready

# Install the web server tools
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# Set the working directory
WORKDIR /workspace/server

# The rest of your setup (copying files etc.)
COPY . .
