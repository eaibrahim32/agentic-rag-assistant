# Multi-stage: keeps the runtime image free of build toolchain.
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local

# Run as non-root.
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser docs/ ./docs/
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data
USER appuser

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
