# Serves the dashboard and API with gunicorn. Build from this folder:
#   docker build -t predmaint .
#   docker run --rm -p 8000:8000 predmaint        # then open http://localhost:8000

# --- build: resolve the locked runtime dependencies (CPU PyTorch) into /app/.venv ---
FROM python:3.15-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# --- runtime: just Python, the virtualenv and the trained model ---
FROM python:3.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PM_ARTIFACTS_DIR=/app/artifacts \
    PM_SUBSET=FD001
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY artifacts/FD001 ./artifacts/FD001
RUN useradd --system --uid 10001 --create-home --home-dir /home/app app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status != 200)"]

# One worker: the fleet replay and Socket.IO rooms live in process memory. Threads serve requests.
CMD ["gunicorn", "--workers", "1", "--threads", "32", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "predmaint.wsgi:app"]
