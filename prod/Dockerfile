# ===========================================================================
# Stage 1 — Install dependencies
# ===========================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY prod/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /build/requirements.txt

# ===========================================================================
# Stage 2 — Runtime image
# ===========================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="ANI BIOME Engineering <eng@ani.ai>"
LABEL description="Questions Agent — Adaptive Questionnaire Engine (Production)"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Security: non-root user
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --home /app --create-home --shell /bin/bash appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy application code
COPY --chown=appuser:appuser . /app

# Switch to non-root
USER appuser

EXPOSE 8080

# Healthcheck — lightweight GET /health every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "questions_agent_platform.prod.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--log-level", "info"]
