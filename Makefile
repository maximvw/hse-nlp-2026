SHELL := /bin/bash
PYTHON := python3.11
UV := uv
OUTPUT_DIR := output

# Detect OS
OS := $(shell uname -s)

.PHONY: help install install-system install-deps download-models run run-ui run-fast setup check clean add-user \
        docker-build docker-up docker-down docker-logs docker-models docker-add-user docker-shell

help:
	@echo "nlp-hw — YouTube Video Analysis Chatbot"
	@echo ""
	@echo "Local development:"
	@echo "  make setup           Full setup (system deps + python deps + download models)"
	@echo "  make install         Install python dependencies only"
	@echo "  make install-system  Install system dependencies (ffmpeg, yt-dlp)"
	@echo "  make download-models Pre-download all ML models (~5 GB)"
	@echo "  make run             Run CLI chatbot"
	@echo "  make run-ui          Run Streamlit web UI  (PORT=8501 by default)"
	@echo "  make add-user U=name P=pass D='Display Name'  Add a user for web UI"
	@echo "  make check           Check all dependencies are installed"
	@echo "  make clean           Remove output files and python cache"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-models   Download ML models into Docker volume (~5 GB, run once)"
	@echo "  make docker-up       Start Streamlit app in background"
	@echo "  make docker-down     Stop containers"
	@echo "  make docker-logs     Follow container logs"
	@echo "  make docker-add-user U=name P=pass D='Display Name'  Add user inside container"
	@echo "  make docker-shell    Open bash shell inside running container"

# ── Full setup ────────────────────────────────────────────────────────────────

setup: check-env install-system install download-models
	@echo ""
	@echo "✓ Setup complete. Run: make run"

# ── System dependencies ───────────────────────────────────────────────────────

install-system:
ifeq ($(OS), Linux)
	@echo "→ Installing system packages (apt)..."
	sudo apt-get update -qq
	sudo apt-get install -y ffmpeg python3.11 python3.11-dev build-essential
	@# Install yt-dlp as binary (apt version is often outdated)
	sudo curl -sL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
	sudo chmod +x /usr/local/bin/yt-dlp
	@# Install uv if not present
	@which uv > /dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh && echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> ~/.bashrc)
else ifeq ($(OS), Darwin)
	@echo "→ Installing system packages (brew)..."
	brew install ffmpeg yt-dlp
	@which uv > /dev/null 2>&1 || brew install uv
else
	@echo "Unknown OS: $(OS). Install ffmpeg, yt-dlp, uv manually."
	exit 1
endif

# ── Python dependencies ───────────────────────────────────────────────────────

install:
	@echo "→ Installing python dependencies..."
	$(UV) sync
	@echo "✓ Python dependencies installed"

# ── Model pre-download ────────────────────────────────────────────────────────

download-models:
	@echo "→ Pre-downloading ML models (~5 GB, may take a while)..."
	@mkdir -p $(OUTPUT_DIR)
	$(UV) run python -c "\
import sys; \
print('  Downloading Silero VAD...'); \
import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True); \
print('  Downloading WavLM-base-SV...'); \
from transformers import AutoFeatureExtractor, WavLMForXVector; \
AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-sv'); \
WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv'); \
print('  Downloading sentence-transformers...'); \
from langchain_community.embeddings import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', model_kwargs={'device': 'cpu'}); \
print('  Downloading Whisper large-v3-turbo-q5_0 (~3.5 GB)...'); \
from pywhispercpp.model import Model; \
Model('large-v3-turbo-q5_0'); \
print('✓ All models downloaded')"

# ── Run ───────────────────────────────────────────────────────────────────────

run:
	@[ -f .env ] || (echo "Error: .env not found. Copy .env.example and fill in OPENROUTER_API_KEY" && exit 1)
	$(UV) run python main.py $(ARGS)

# Run with more workers (for multi-core servers)
run-fast:
	@[ -f .env ] || (echo "Error: .env not found." && exit 1)
	$(UV) run python main.py --auto-workers $(ARGS)

run-ui:
	@[ -f .env ] || (echo "Error: .env not found. Copy .env.example and fill in OPENROUTER_API_KEY" && exit 1)
	@[ -f data/users.yaml ] || (echo "No users found. Run: make add-user U=name P=pass D='Name'" && exit 1)
	$(UV) run streamlit run app.py --server.port $${PORT:-8501} $(ARGS)

add-user:
	@[ -n "$(U)" ] || (echo "Usage: make add-user U=username P=password D='Display Name'" && exit 1)
	@[ -n "$(P)" ] || (echo "Usage: make add-user U=username P=password D='Display Name'" && exit 1)
	$(UV) run python manage_users.py add "$(U)" "$(P)" "$(D)"

# ── Checks ────────────────────────────────────────────────────────────────────

check:
	@echo "→ Checking dependencies..."
	@which ffmpeg > /dev/null 2>&1 && echo "  ✓ ffmpeg" || echo "  ✗ ffmpeg NOT FOUND"
	@which yt-dlp > /dev/null 2>&1 && echo "  ✓ yt-dlp" || echo "  ✗ yt-dlp NOT FOUND"
	@which uv > /dev/null 2>&1 && echo "  ✓ uv" || echo "  ✗ uv NOT FOUND"
	@$(UV) run python --version 2>/dev/null | grep -q "3.11" && echo "  ✓ Python 3.11" || echo "  ✗ Python 3.11 NOT FOUND"
	@[ -f .env ] && echo "  ✓ .env" || echo "  ✗ .env NOT FOUND (copy from .env.example)"
	@$(UV) run python -c "import torch, pywhispercpp, langchain" 2>/dev/null && echo "  ✓ python packages" || echo "  ✗ python packages not installed (run: make install)"

check-env:
	@[ -f .env ] || (cp .env.example .env && echo "→ Created .env from .env.example. Fill in OPENROUTER_API_KEY before running." && exit 1)

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(OUTPUT_DIR)/*.wav $(OUTPUT_DIR)/*.txt $(OUTPUT_DIR)/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned output files and python cache"

clean-all: clean
	rm -rf .venv
	@echo "✓ Removed .venv (run 'make install' to reinstall)"

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

# Download all ML models into the named Docker volume (run once after build)
docker-models:
	@[ -f .env ] || (echo "Error: .env not found. Copy .env.example and fill in OPENROUTER_API_KEY" && exit 1)
	docker compose run --rm app python -c "\
import sys; \
print('  Downloading Silero VAD...'); \
import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True); \
print('  Downloading WavLM-base-SV...'); \
from transformers import AutoFeatureExtractor, WavLMForXVector; \
AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-sv'); \
WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv'); \
print('  Downloading sentence-transformers...'); \
from langchain_community.embeddings import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', model_kwargs={'device': 'cpu'}); \
print('  Downloading Whisper large-v3-turbo-q5_0 (~3.5 GB)...'); \
from pywhispercpp.model import Model; \
Model('large-v3-turbo-q5_0'); \
print('✓ All models downloaded')"

docker-up:
	@[ -f .env ] || (echo "Error: .env not found. Copy .env.example and fill in OPENROUTER_API_KEY" && exit 1)
	@[ -f data/users.yaml ] || (echo "No users found. Run: make docker-add-user U=name P=pass D='Name'" && exit 1)
	docker compose up -d
	@echo "✓ App running at http://localhost:$${PORT:-8501}"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f app

docker-add-user:
	@[ -n "$(U)" ] || (echo "Usage: make docker-add-user U=username P=password D='Display Name'" && exit 1)
	@[ -n "$(P)" ] || (echo "Usage: make docker-add-user U=username P=password D='Display Name'" && exit 1)
	docker compose run --rm app python manage_users.py add "$(U)" "$(P)" "$(D)"

docker-shell:
	docker compose exec app bash
