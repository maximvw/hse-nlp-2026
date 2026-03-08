# YouTube Video Summarizer (Russian)

Локальный суммаризатор видео с YouTube на русском языке. Оптимизирован для MacBook с Apple Silicon (M4).

## Pipeline

```
YouTube URL
 ↓
yt-dlp (скачать audio)
 ↓
ffmpeg (конвертация в 16kHz mono WAV)
 ↓
Silero VAD (сегментация речи)
 ↓
whisper.cpp (ASR → текст с таймстемпами)
 ↓
pyannote (диаризация спикеров)
 ↓
Выравнивание спикеров + слов → транскрипт
 ↓
LLM суммаризация (OpenRouter API)
  - merge summary для коротких видео (<= 30 мин)
  - hierarchical summary для длинных видео (> 30 мин)
```

## Требования

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (менеджер пакетов)
- ffmpeg
- yt-dlp

### Установка системных зависимостей (macOS)

```bash
brew install ffmpeg yt-dlp
```

### Установка проекта

```bash
uv sync
```

## Настройка

Скопируй `.env` и заполни ключи:

```bash
cp .env .env.local   # или отредактируй .env напрямую
```

Переменные окружения в `.env`:

| Переменная | Описание | Обязательна |
|---|---|---|
| `OPENROUTER_API_KEY` | API-ключ [OpenRouter](https://openrouter.ai/) для суммаризации | Да |
| `HF_TOKEN` | Токен [HuggingFace](https://huggingface.co/settings/tokens) для pyannote | Только с диаризацией |

### Pyannote (диаризация спикеров)

Для работы диаризации нужно:

1. Создать аккаунт на [huggingface.co](https://huggingface.co)
2. Принять лицензии моделей:
   - [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   
3. Создать токен в [настройках](https://huggingface.co/settings/tokens) и вписать в `HF_TOKEN`

Если диаризация не нужна, используй флаг `--no-diarize`.

## Запуск

```bash
# Полный pipeline (с диаризацией спикеров)
uv run python main.py "https://www.youtube.com/watch?v=erFQqG_Y7Dc"

# Без диаризации (не нужен HF_TOKEN)
uv run python main.py --no-diarize "https://www.youtube.com/watch?v=VIDEO_ID"

# С кастомными параметрами
uv run python main.py \
  --whisper-model large-v3-turbo-q5_0 \
  --llm-model nvidia/nemotron-nano-9b-v2:free \
  --threads 8 \
  --language ru \
  -o output \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Параметры CLI

| Параметр | По умолчанию | Описание |
|---|---|---|
| `url` | — | URL видео на YouTube |
| `-o`, `--output-dir` | `output` | Директория для результатов |
| `--whisper-model` | `large-v3-turbo-q5_0` | Модель whisper.cpp (квантизованная) |
| `--llm-model` | `nvidia/nemotron-nano-9b-v2:free` | Модель LLM через OpenRouter |
| `--language` | `ru` | Язык для ASR |
| `--threads` | `8` | Количество потоков для whisper.cpp |
| `--no-diarize` | `false` | Пропустить диаризацию спикеров |

## Результаты

После запуска в `output/` появятся:

- `transcript.txt` — полный транскрипт с таймстемпами (и спикерами, если включена диаризация)
- `summary.txt` — саммари видео
