#!/bin/bash

# Dance Analysis Server - Helper Commands

# Check for profile argument
PROFILE="${PROFILE:-cpu}"  # Default to cpu if not set
if [[ "$1" =~ ^(cpu|nvidia|amd|mac)$ ]]; then
    PROFILE="$1"
    COMPOSE_PROFILE="--profile $PROFILE"
    shift  # Remove profile from arguments
else
    COMPOSE_PROFILE="--profile $PROFILE"
fi

case "$1" in
    start)
        echo "🚀 Starting services with profile: $PROFILE"
        docker compose $COMPOSE_PROFILE up -d
        echo "✅ Services started"
        ;;
    stop)
        echo "🛑 Stopping services..."
        docker compose $COMPOSE_PROFILE down
        echo "✅ Services stopped"
        ;;
    restart)
        echo "🔄 Restarting services..."
        docker compose $COMPOSE_PROFILE restart
        echo "✅ Services restarted"
        ;;
    logs)
        docker compose $COMPOSE_PROFILE logs -f "${2:-}"
        ;;
    status)
        echo "📊 Service Status (Profile: $PROFILE):"
        docker compose $COMPOSE_PROFILE ps
        ;;
    clean)
        echo "🧹 Cleaning up volumes and containers..."
        docker compose $COMPOSE_PROFILE down -v
        echo "✅ Cleanup complete"
        ;;
    build)
        echo "🔨 Building images..."
        docker compose $COMPOSE_PROFILE build
        echo "✅ Build complete"
        ;;
    shell-backend)
        echo "📦 Opening shell in backend container..."
        docker compose exec backend bash
        ;;
    shell-worker)
        echo "📦 Opening shell in video-worker container..."
        WORKER_NAME="dance-video-worker-$PROFILE"
        docker compose exec "$WORKER_NAME" bash
        ;;
    minio-console)
        echo "🌐 MinIO Console:"
        echo "   URL: http://localhost:9001"
        echo "   Username: minioadmin"
        echo "   Password: minioadmin"
        ;;
    rq-dashboard)
        echo "📊 RQ Dashboard:"
        echo "   URL: http://localhost:9181"
        ;;
    redis-cli)
        docker compose exec redis redis-cli
        ;;
    test-upload)
        if [ -z "$2" ]; then
            echo "❌ Usage: $0 [profile] test-upload <video-file>"
            exit 1
        fi
        echo "📤 Uploading test video: $2"
        curl -X POST http://localhost:8000/api/v1/analyze -F "file=@$2"
        echo ""
        ;;
    *)
        echo "Dance Analysis Server - Helper Commands"
        echo ""
        echo "Usage: $0 [profile] {command} [options]"
        echo ""
        echo "Profiles (choose one):"
        echo "  cpu                Use CPU worker (default)"
        echo "  nvidia             Use NVIDIA GPU worker"
        echo "  amd                Use AMD GPU worker"
        echo "  mac                Use Mac Apple Silicon worker"
        echo ""
        echo "Commands:"
        echo "  start              Start all services"
        echo "  stop               Stop all services"
        echo "  restart            Restart all services"
        echo "  logs [service]     View service logs"
        echo "  status             Show service status"
        echo "  clean              Stop and remove all containers/volumes"
        echo "  build              Build Docker images"
        echo ""
        echo "Containers:"
        echo "  shell-backend      Open shell in backend container"
        echo "  shell-worker       Open shell in GPU worker container"
        echo ""
        echo "Dashboards:"
        echo "  minio-console      Show MinIO console URL"
        echo "  rq-dashboard       Show RQ dashboard URL"
        echo ""
        echo "Tools:"
        echo "  redis-cli          Open Redis CLI"
        echo "  test-upload        Upload a test video"
        echo ""
        echo "Examples:"
        echo "  $0 start                           # Start with CPU worker"
        echo "  $0 nvidia start                    # Start with NVIDIA GPU"
        echo "  $0 amd logs video-worker-amd       # View AMD worker logs"
        echo "  $0 mac test-upload video.mp4       # Upload with Mac worker"
        echo "  $0 nvidia shell-worker             # Shell into NVIDIA worker"
        echo ""
        ;;
esac
