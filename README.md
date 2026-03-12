# YouTube Video Assistant (Russian)

Чат-бот для анализа YouTube-видео на русском языке. Обрабатывает видео локально (транскрипция + диаризация спикеров), после чего отвечает на вопросы через LLM-агента с инструментами.

Доступны два режима: веб-интерфейс (Streamlit) и CLI.

## Что умеет

- Транскрибирует видео с определением спикеров
- Отвечает на структурные вопросы: «сколько спикеров?», «кто говорил дольше всех?»
- Находит реплики конкретного спикера в заданный момент времени
- Делает саммари видео
- Выполняет семантический поиск по содержанию
- Показывает метаданные видео (название, канал, главы, описание)

## Требования

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- ffmpeg
- yt-dlp

### Установка системных зависимостей

**macOS:**
```bash
brew install ffmpeg yt-dlp
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install -y ffmpeg
sudo curl -sL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -o /usr/local/bin/yt-dlp && sudo chmod +x /usr/local/bin/yt-dlp
```

### Установка проекта

```bash
uv sync
```

### Скачать ML-модели (~5 GB)

```bash
make download-models
```

## Настройка

```bash
cp .env.example .env
```

| Переменная | Описание | Обязательна |
|---|---|---|
| `OPENROUTER_API_KEY` | API-ключ [OpenRouter](https://openrouter.ai/) | Да |
| `HF_TOKEN` | HuggingFace токен (для приватных моделей) | Нет |

## Запуск

### Веб-интерфейс (Streamlit)

```bash
# Создать пользователя
make add-user U=admin P=yourpassword D="Admin"

# Запустить
make run-ui
# → http://localhost:8501
```

### CLI

```bash
make run
```

После запуска бот ждёт ссылку на YouTube-видео в диалоге. Видео обрабатывается автоматически, затем можно задавать вопросы.

### Примеры вопросов

```
Сколько спикеров в видео?
Что сказал спикер №2 примерно на 10-й минуте?
Что происходило в первые 5 минут?
Сделай саммари видео
Что обсуждалось про машинное обучение?
Кто говорил дольше всех?
О чём это видео? Что за канал?
```

### Параметры CLI

| Параметр | По умолчанию | Описание |
|---|---|---|
| `-o`, `--output-dir` | `output` | Директория для промежуточных файлов |
| `--whisper-model` | `large-v3-turbo-q5_0` | Модель whisper.cpp |
| `--llm-model` | `nvidia/nemotron-nano-9b-v2:free` | Модель LLM через OpenRouter |
| `--embedding-model` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Модель эмбеддингов |
| `--language` | `ru` | Язык ASR |
| `--threads` | `4` | Потоков CPU на воркер Whisper |
| `--workers` | `cpu_count // 4` | Параллельных воркеров Whisper |
| `--auto-workers` | — | Автоматически использовать все ядра CPU |

## Деплой на сервере

Смотри [DEPLOY.md](DEPLOY.md).

## Результаты

После обработки видео в `--output-dir/{video_id}/` сохраняются:
- `transcript.txt` — полный транскрипт с таймкодами и именами спикеров
- `metadata.json` — метаданные YouTube-видео
- `index/` — FAISS-индекс для семантического поиска
