#!/usr/bin/env bash
set -e

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
