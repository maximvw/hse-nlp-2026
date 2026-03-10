# Architecture

## Overview

Чат-бот для анализа YouTube-видео на русском языке. Весь inference (кроме LLM) выполняется локально на Apple Silicon. LLM-запросы — через OpenRouter API (OpenAI-совместимый интерфейс).

Пользователь общается с агентом в терминале. Агент получает YouTube-ссылку, запускает пайплайн обработки и отвечает на вопросы, самостоятельно выбирая подходящий инструмент.

## Tech Stack

| Компонент | Технология |
|---|---|
| Package manager | uv |
| Audio download | yt-dlp (CLI) |
| Audio preprocessing | ffmpeg (CLI) |
| VAD | Silero VAD (torch.hub) |
| ASR | whisper.cpp (pywhispercpp) |
| Speaker diarization | WavLM-SV (microsoft/wavlm-base-sv) + AgglomerativeClustering |
| LLM agent | LangChain `create_agent` + OpenRouter |
| Semantic search | sentence-transformers + FAISS (faiss-cpu) |
| CLI output | rich |
| Config | python-dotenv |

## Project Structure

```
nlp-hw/
  main.py                    # Entry point, argparse, запуск чат-бота
  pipeline/
    __init__.py
    download.py              # download_audio(), preprocess_audio()
    vad.py                   # run_vad(), group_segments()
    transcribe.py            # transcribe()
    diarize.py               # diarize(), align_transcript_with_speakers(), format_transcript()
    summarize.py             # summarize() — три стратегии: single pass / rolling merge / hierarchical
    index.py                 # TranscriptIndex, build_index() — структурированный доступ + FAISS
    chatbot.py               # VideoState, build_chatbot_tools(), run_chatbot()
  output/                    # Генерируемые файлы (gitignored)
    raw_audio.wav
    audio_16k.wav
    transcript.txt
```

## Agent Architecture

Агент построен на `langchain.agents.create_agent` (LangGraph под капотом). Он получает 6 инструментов, привязанных к мутабельному `VideoState` через замыкания.

```
Пользователь вводит текст
         ↓
   LLM-агент (tool calling)
         ↓ выбирает инструмент
  ┌──────┬────────┬──────────┬──────────────┬────────────────┬───────────────┐
  │      │        │          │              │                │               │
  ▼      ▼        ▼          ▼              ▼                ▼               ▼
process summarize get_       get_segments_  get_segments_   semantic_
_video  _video    transcript  by_speaker    by_time         search
                  _metadata
  │
  ▼ (заполняет VideoState)
  TranscriptIndex
  (segments + FAISS)
```

### VideoState

Мутабельный контейнер состояния сессии:

```python
@dataclass
class VideoState:
    index: TranscriptIndex | None = None   # None до обработки видео
    processed_url: str | None = None
```

До вызова `process_video` все инструменты работы с видео возвращают подсказку вместо ошибки.

### Инструменты агента

| Инструмент | Когда использовать |
|---|---|
| `process_video(url)` | Получена YouTube-ссылка |
| `summarize_video()` | «Сделай саммари», «краткое содержание» |
| `get_transcript_metadata()` | «Сколько спикеров?», «кто говорил дольше?» |
| `get_segments_by_speaker(id, start_min, end_min)` | «Что сказал спикер №2 на 10-й минуте?» |
| `get_segments_by_time(start_min, end_min)` | «Что было в первые 5 минут?» |
| `semantic_search(query)` | «Что обсуждалось про X?» |

### TranscriptIndex

Структурированный индекс поверх `list[DiarizedSegment]`:

```python
@dataclass
class TranscriptIndex:
    segments: list[DiarizedSegment]   # все сегменты в памяти (~50-100 КБ для часового видео)
    faiss: FAISS                       # для семантического поиска

    def get_metadata() -> dict
    def get_by_speaker(speaker, start_min, end_min) -> list[DiarizedSegment]
    def get_by_time(start_min, end_min) -> list[DiarizedSegment]
    def semantic_search(query, k=5) -> list[Document]
```

### История диалога

Полная история сообщений накапливается между вопросами и передаётся агенту при каждом запросе — агент помнит контекст разговора.

## Processing Pipeline

Запускается инструментом `process_video(url)`:

```
download_audio(url) → raw_audio.wav
        ↓
preprocess_audio() → audio_16k.wav   (ffmpeg: 16kHz mono)
        ↓
run_vad() → list[SpeechSegment]       (Silero VAD)
        ↓
group_segments() → chunks             (группировка до 30 сек для whisper)
        ↓
transcribe() → list[TranscriptSegment] (whisper.cpp, word-level timestamps)
        ↓
diarize() → list[(start, end, speaker)] (WavLM-SV embeddings + AgglomerativeClustering)
        ↓
align_transcript_with_speakers() → list[DiarizedSegment]
        ↓
build_index() → TranscriptIndex        (сохраняется в VideoState)
```

## Data Types

```
SpeechSegment        (start: float, end: float)
        ↓
TranscriptSegment    (start, end, text, words: list[Word])
        ↓
DiarizedSegment      (speaker: str, start, end, text)
        ↓
TranscriptIndex      (segments: list[DiarizedSegment], faiss: FAISS)
```

## Key Design Decisions

1. **Агент вместо RAG**: LLM сам выбирает инструмент под тип вопроса. Структурные запросы («сколько спикеров») идут в `get_metadata`, а не в семантический поиск. RAG остаётся для тематических вопросов.

2. **Мутабельный state через замыкания**: Инструменты агента — функции, замкнутые на `VideoState`. Агент не может напрямую изменять состояние — только через вызов `process_video`.

3. **TranscriptIndex в памяти**: Часовой транскрипт занимает ~50-100 КБ. Хранить в памяти дешевле, чем читать из файла при каждом запросе.

4. **WavLM-SV вместо pyannote**: Не требует HF-токен и принятия лицензий. VAD-сегменты используются повторно — embeddings считаются только для них, без дополнительной сегментации.

5. **Три стратегии суммаризации** по длине транскрипта:
   - **Single pass** (< ~10 мин, ≤5000 символов) — один LLM-запрос на весь текст.
   - **Rolling merge** (10–30 мин, 5000–15000 символов) — `summary_n = summarize(summary_{n-1} + chunk_n)`: модель накапливает контекст предыдущих частей.
   - **Hierarchical** (> ~30 мин, >15000 символов) — каждый чанк суммаризируется независимо, затем все частичные саммари объединяются финальным запросом.

## Environment Variables

| Variable | Used in | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | `pipeline/chatbot.py`, `pipeline/summarize.py` | API-ключ OpenRouter |

## External Dependencies (system)

- `ffmpeg` — должен быть в PATH
- `yt-dlp` — должен быть в PATH
