#!/bin/sh
set -e

echo "Starting FastAPI lightweight LLM service..."
exec uvicorn src.main:app --host 0.0.0.0 --port 8000
