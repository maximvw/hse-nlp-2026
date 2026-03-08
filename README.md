# YouTube Video Summarizer & Q&A (Russian)

Локальный суммаризатор видео и Q&A-агент по YouTube-видео на русском языке.

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
┌─────────────────────────┬──────────────────────────────┐
│ summary mode            │ qa mode                      │
│                         │                              │
│ LLM суммаризация        │ sentence-transformers        │
│ (merge / hierarchical)  │   ↓                          │
│                         │ FAISS index                  │
│                         │   ↓                          │
│                         │ вопрос → top-k чанков → LLM  │
│                         │   ↓                          │
│                         │ ответ с таймкодами           │
└─────────────────────────┴──────────────────────────────┘
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

Скопируй `.env.example` и заполни ключи:

```bash
cp .env.example .env
```

Переменные окружения в `.env`:

| Переменная | Описание | Обязательна |
|---|---|---|
| `OPENROUTER_API_KEY` | API-ключ [OpenRouter](https://openrouter.ai/) для суммаризации и Q&A | Да |
| `HF_TOKEN` | Токен [HuggingFace](https://huggingface.co/settings/tokens) для pyannote | Только с диаризацией |

### Pyannote (диаризация спикеров)

Для работы диаризации нужно:

1. Создать аккаунт на [huggingface.co](https://huggingface.co)
2. Принять лицензии моделей:
   - [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

3. Создать токен в [настройках](https://huggingface.co/settings/tokens) и вписать в `HF_TOKEN`

Если диаризация не нужна, используй флаг `--no-diarize` (только для summary mode).

## Запуск

```bash
# Саммари видео (с диаризацией)
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID" summary

# Саммари без диаризации (не нужен HF_TOKEN)
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID" summary --no-diarize

# Интерактивные вопросы по видео (Q&A, требует диаризацию)
uv run python main.py "https://www.youtube.com/watch?v=Pgu2xWfWBoo" qa -o output

# С кастомными параметрами
uv run python main.py \
  "https://www.youtube.com/watch?v=VIDEO_ID" summary \
  --whisper-model large-v3-turbo-q5_0 \
  --llm-model nvidia/nemotron-nano-9b-v2:free \
  --threads 8 \
  --language ru \
  -o output
```

### Параметры CLI

| Параметр | По умолчанию | Описание |
|---|---|---|
| `url` | — | URL видео на YouTube |
| `mode` | — | `summary` или `qa` |
| `-o`, `--output-dir` | `output` | Директория для результатов |
| `--whisper-model` | `large-v3-turbo-q5_0` | Модель whisper.cpp (квантизованная) |
| `--llm-model` | `nvidia/nemotron-nano-9b-v2:free` | Модель LLM через OpenRouter |
| `--embedding-model` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Модель эмбеддингов для RAG |
| `--language` | `ru` | Язык для ASR |
| `--threads` | `8` | Количество потоков для whisper.cpp |
| `--no-diarize` | `false` | Пропустить диаризацию (только summary mode) |

## Результаты

После запуска в `output/` появятся:

- `transcript.txt` — полный транскрипт с таймстемпами (и спикерами, если включена диаризация)
- `summary.txt` — саммари видео (только в summary mode)

В Q&A mode ответы выводятся интерактивно в терминал.
