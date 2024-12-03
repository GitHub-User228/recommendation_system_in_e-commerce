#!/bin/bash
set -e

log() {
    echo "$(date): $1" >&2
}

# Start the FastAPI application
log "Starting FastAPI application"
uvicorn app.app_simple:app --reload --host "${HOST}" --port "${APP_DOCKER_PORT}"
log "FastAPI application exited"