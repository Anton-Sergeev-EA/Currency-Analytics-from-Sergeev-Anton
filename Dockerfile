# syntax=docker/dockerfile:1

# --- Build stage: compile wheels (needs gcc/g++/make for lightgbm/xgboost) ---
FROM python:3.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Building wheels in a throwaway stage keeps gcc/g++/make and pip's build
# cache out of the final image entirely - on a VDS with little disk this
# is the difference between the runtime image being ~400MB or ~1.2GB.
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --- Runtime stage: slim, no compilers, no build cache ---
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC \
    # Single worker by default: on a 4GB VDS, each extra uvicorn worker is
    # a full extra copy of pandas/numpy/lightgbm/xgboost in RAM (roughly
    # 150-300MB per worker once models are loaded). Override with
    # UVICORN_WORKERS=2 only if you've confirmed the box has headroom.
    UVICORN_WORKERS=1

RUN sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN groupadd -r analytics && useradd -r -g analytics analytics

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=analytics:analytics . .

RUN mkdir -p /app/data/models /app/logs && \
    chown -R analytics:analytics /app

USER analytics

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS} --loop uvloop"]
