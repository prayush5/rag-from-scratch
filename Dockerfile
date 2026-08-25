# Stage 1: build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 

# Added python3-dev and pkg-config for C extension compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade core build tools before installing requirements
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

#stage 2: runtime environment
FROM python:3.11-slim as runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/home/appuser/.local/bin:$PATH"

RUN useradd -m -u 1000 appuser

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]