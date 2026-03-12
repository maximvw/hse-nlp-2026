# ─────────────────────────────────────────────────────────────
# Stage 1: builder — compiles native extensions (pywhispercpp)
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# C++ toolchain required by pywhispercpp (whisper.cpp compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies into .venv (no extras needed at build time)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache


# ─────────────────────────────────────────────────────────────
# Stage 2: runtime — lean image, no compiler
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system packages:
#   ffmpeg        — audio processing
#   libgomp1      — OpenMP (required by PyTorch CPU kernels)
#   curl          — used to fetch yt-dlp binary
#   ca-certificates — HTTPS for OpenRouter API calls
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp as a static binary (apt version is always outdated)
RUN curl -sL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
        -o /usr/local/bin/yt-dlp \
    && chmod +x /usr/local/bin/yt-dlp

# Copy compiled .venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY . /app

# ── Environment ──────────────────────────────────────────────
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Point all ML model caches to a single mount point /cache
    # pywhispercpp respects XDG_CACHE_HOME → /cache/pywhispercpp/
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    # Streamlit
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Health check — Streamlit exposes a health endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
