# Architecture

## Overview

CLI-приложение для автоматического создания саммари и ответов на вопросы по YouTube-видео на русском языке.
Весь inference (кроме LLM) выполняется локально на MacBook с Apple Silicon (M4).
LLM-запросы (суммаризация, Q&A) — через OpenRouter API (OpenAI-совместимый интерфейс).

Два режима работы:
- `summary` — создание саммари видео
- `qa` — интерактивные вопросы по видео (RAG)

## Tech Stack

| Компонент | Технология | Назначение |
|---|---|---|
| Package manager | uv | Управление зависимостями, запуск |
| Audio download | yt-dlp (CLI) | Скачивание аудиодорожки с YouTube |
| Audio preprocessing | ffmpeg (CLI) | Конвертация в 16kHz mono WAV |
| VAD | Silero VAD (torch.hub) | Детекция речевых сегментов |
| ASR | whisper.cpp (pywhispercpp) | Распознавание речи, таймстемпы |
| Speaker diarization | pyannote-audio | Определение спикеров |
| Summarization LLM | LangChain + OpenRouter | Генерация саммари |
| RAG embeddings | sentence-transformers (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Локальные эмбеддинги для поиска по транскрипту |
| RAG vector store | FAISS (faiss-cpu) | In-memory индекс для similarity search |
| Q&A LLM | LangChain + OpenRouter | Ответы на вопросы с контекстом из RAG |
| CLI output | rich | Форматированный вывод в терминал |
| Config | python-dotenv | Загрузка переменных из .env |

## Project Structure

```
nlp-hw/
  main.py                    # Entry point, CLI (argparse), оркестрация pipeline
  pyproject.toml             # Зависимости, метаданные проекта
  .env                       # API-ключи (OPENROUTER_API_KEY, HF_TOKEN)
  pipeline/
    __init__.py
    download.py              # download_audio(), preprocess_audio()
    vad.py                   # run_vad(), group_segments()
    transcribe.py            # transcribe()
    diarize.py               # diarize(), align_transcript_with_speakers(), format_transcript()
    summarize.py             # summarize()
    rag.py                   # build_index(), ask()
  output/                    # Генерируемые файлы (gitignored)
    raw_audio.wav            # Оригинальный аудио с YouTube
    audio_16k.wav            # Предобработанный аудио (16kHz mono)
    transcript.txt           # Итоговый транскрипт
    summary.txt              # Итоговое саммари
```

## Pipeline Flow

### Common pipeline (steps 1-5, shared by both modes)

```
main.py::_process_video(args)
  |
  |-- 1. download_audio(url, output_dir) -> raw_audio.wav
  |       Вызывает yt-dlp через subprocess.
  |       Извлекает аудио из видео, сохраняет как WAV.
  |
  |-- 2. preprocess_audio(raw_audio, output_dir) -> audio_16k.wav
  |       Вызывает ffmpeg через subprocess.
  |       Конвертирует в 16kHz, mono — формат для Silero VAD и whisper.cpp.
  |
  |-- 3. run_vad(audio_16k) -> list[SpeechSegment]
  |       Загружает Silero VAD из torch.hub.
  |       Возвращает список сегментов речи (start, end в секундах).
  |       Фильтрует короткие сегменты (< 0.3s), мёржит близкие (gap < 0.5s).
  |
  |-- 3b. group_segments(segments) -> list[SpeechSegment]
  |       Группирует VAD-сегменты в чанки до 30 секунд для whisper.
  |       Нужно чтобы whisper не получал слишком длинные куски.
  |
  |-- 4. transcribe(audio_16k, chunks) -> list[TranscriptSegment]
  |       Загружает whisper.cpp модель через pywhispercpp.
  |       Для каждого чанка: вырезает кусок аудио (ffmpeg) -> транскрибирует.
  |       Корректирует таймстемпы каждого сегмента на offset чанка.
  |       Возвращает сегменты с текстом и word-level таймстемпами.
  |
  |-- 5a. diarize(audio_16k) -> list[(start, end, speaker)]  [optional in summary mode]
  |       Загружает pyannote/speaker-diarization-3.1 (требует HF_TOKEN).
  |       Использует MPS (Metal GPU) если доступен, иначе CPU.
  |       Возвращает DiarizeOutput -> извлекает speaker_diarization Annotation.
  |
  |-- 5b. align_transcript_with_speakers(transcript, speaker_turns)
          Для каждого ASR-сегмента берёт mid-point по времени.
          Находит активного спикера в этот момент (или ближайшего).
          Мёржит соседние сегменты одного спикера.
          Returns: (transcript_text, diarized_segments)
```

### Summary mode (`python main.py URL summary`)

```
_run_summary(transcript_text, args)
  |
  |-- summarize(transcript_text) -> str
        Использует LangChain ChatOpenAI с OpenRouter API.
        Две стратегии:
          - Merge (single pass): для транскриптов <= 15000 символов (~30 мин).
            Весь текст отправляется одним запросом.
          - Hierarchical: для длинных транскриптов.
            Текст разбивается на чанки по ~15000 символов (по границам строк).
            Каждый чанк суммаризируется отдельно.
            Затем все частичные саммари объединяются финальным запросом.
        Результат сохраняется в output/summary.txt.
```

### Q&A mode (`python main.py URL qa`)

```
_run_qa(diarized_segments, args)
  |
  |-- build_index(diarized_segments) -> FAISS
  |     1. Группирует DiarizedSegment'ы в чанки по ~1000 символов.
  |        Каждый чанк = Document с метаданными (start, end, speakers).
  |     2. Генерирует эмбеддинги через sentence-transformers
  |        (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, inference на MPS).
  |     3. Строит FAISS in-memory индекс.
  |
  |-- Interactive loop:
        User question
          -> index.similarity_search(question, k=5)   # top-5 чанков
          -> LLM prompt с контекстом из найденных чанков
          -> answer (с таймкодами и спикерами)
```

## Data Flow Types

```
SpeechSegment          # (start: float, end: float) — VAD-сегмент в секундах
  |
  v
TranscriptSegment      # (start, end, text, words: list[Word]) — ASR-результат
  |                      Word = (text, start, end)
  v
DiarizedSegment        # (speaker, start, end, text) — после alignment с pyannote
  |
  +--> str (transcript_text)    "[MM:SS] SPEAKER_XX: текст"
  |      |
  |      v
  |    str (summary)            Саммари от LLM (summary mode)
  |
  +--> Document (page_content + metadata)    Чанк для RAG (qa mode)
         metadata: {start, end, speakers}
         |
         v
       FAISS index -> similarity_search -> LLM -> str (answer)
```

## Environment Variables

| Variable | Used in | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | `pipeline/summarize.py`, `pipeline/rag.py` | API-ключ для OpenRouter (суммаризация и Q&A) |
| `HF_TOKEN` | `pipeline/diarize.py` | HuggingFace токен (pyannote модели) |

Загружаются из `.env` через `python-dotenv` в `main.py` при старте.

## CLI Arguments

Определены в `main.py::main()` через argparse:

- `url` (positional) — YouTube URL
- `mode` (positional) — `summary` или `qa`
- `-o / --output-dir` — директория вывода (default: `output`)
- `--whisper-model` — имя модели whisper.cpp (default: `large-v3-turbo-q5_0`)
- `--llm-model` — модель LLM через OpenRouter (default: `nvidia/nemotron-nano-9b-v2:free`)
- `--embedding-model` — модель эмбеддингов для RAG (default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- `--language` — язык ASR (default: `ru`)
- `--threads` — потоки для whisper.cpp (default: `8`)
- `--no-diarize` — пропустить диаризацию (только для summary mode; qa mode требует диаризацию)

## Key Design Decisions

1. **VAD перед ASR**: Silero VAD фильтрует тишину и шум, чтобы whisper не галлюцинировал на пустых участках. Сегменты группируются в чанки до 30 сек.

2. **Чанковая транскрипция**: Каждый VAD-чанк вырезается ffmpeg во временный файл и транскрибируется отдельно. Таймстемпы пересчитываются обратно к полному аудио через offset.

3. **Alignment по mid-point**: Спикер для каждого ASR-сегмента определяется по середине временного интервала сегмента. Если точного совпадения нет — берётся ближайший speaker turn.

4. **Две стратегии суммаризации**: Merge для коротких видео (один LLM-запрос), hierarchical для длинных (чанки -> частичные саммари -> финальное объединение). Порог — 15000 символов (~30 мин речи).

5. **Опциональная диаризация**: pyannote требует HF-токен и принятие лицензий. Флаг `--no-diarize` позволяет пропустить этот шаг — транскрипт будет без указания спикеров. В Q&A mode диаризация обязательна (нужны спикеры для контекста).

6. **pyannote на MPS**: На Apple Silicon pyannote автоматически использует Metal GPU (MPS device) для ускорения.

7. **RAG с локальными эмбеддингами**: Эмбеддинги генерируются локально через sentence-transformers (multilingual-e5-base), не требуют API. Индексируются в FAISS in-memory. Для одного видео этого достаточно — внешняя БД не нужна.

8. **Чанки RAG из diarized сегментов**: Используются уже размеченные по спикерам сегменты, группируются до ~1000 символов. Метаданные (speaker, timestamps) сохраняются и доступны LLM для точных ответов ("кто сказал", "когда упоминается").

9. **Shared pipeline**: Шаги 1-5 (download -> ASR -> diarize) общие для обоих режимов. Режим определяет только финальный шаг — суммаризация или построение RAG-индекса + Q&A loop.

## External Dependencies (system)

- `ffmpeg` — должен быть в PATH
- `yt-dlp` — должен быть в PATH
