# Multi-stage build keeps the final image small: build deps in one layer,
# copy only what's needed into a slim runtime.
FROM python:3.11-slim AS base

# uv for fast, reproducible installs (same tool as local dev).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first so this layer caches unless deps change.
COPY pyproject.toml uv.lock ./

# Install runtime dependencies only (no dev group) into the system environment.
ENV UV_SYSTEM_PYTHON=1
RUN uv export --no-dev --no-emit-project --format requirements-txt > /tmp/req.txt \
    && uv pip install --system -r /tmp/req.txt

# Copy the application code and the trained artifacts the model needs at runtime.
COPY src/ ./src/
COPY artifacts/model_lgbm.txt artifacts/encoding_maps.json artifacts/scaler_stats.json ./artifacts/

# A non-root user is a basic container-security hygiene step.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Health check lets orchestrators know when the service is actually ready.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
