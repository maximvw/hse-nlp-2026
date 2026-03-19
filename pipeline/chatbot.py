from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import logging

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from pipeline.diarize import diarize, align_transcript_with_speakers, format_transcript
from pipeline.download import (
    download_audio, preprocess_audio, fetch_video_metadata, format_metadata,
    save_metadata, try_load_metadata, VideoMetadata, extract_video_id,
)
from pipeline.index import TranscriptIndex, build_index
from pipeline.summarize import summarize
from pipeline.transcribe import transcribe
from pipeline.vad import run_vad, group_segments
    
logger = logging.getLogger(__name__)
console = Console()

SYSTEM_PROMPT = """\
Ты — интеллектуальный ассистент для анализа YouTube-видео на русском языке.
Ты работаешь в Telegram-боте.

Ты умеешь:
• Обработать видео по ссылке (транскрипция + распознавание спикеров)
• Показать информацию о видео (название, канал, описание, главы)
• Сделать краткое содержание
• Найти, что обсуждалось на конкретную тему
• Показать, кто и что говорил в определённый момент
• Дать статистику по спикерам

Правила:
1. Как только пользователь присылает YouTube-ссылку — сразу вызывай process_video, не переспрашивай.
2. Для вопросов о названии, теме, канале, главах видео → используй get_video_info.
3. Для структурных вопросов ("сколько спикеров", "кто говорил дольше") → используй get_transcript_metadata.
4. Для вопросов о конкретном спикере → используй get_segments_by_speaker.
5. Для вопросов о конкретном времени → используй get_segments_by_time.
6. Для тематических вопросов ("что обсуждали про X") → используй semantic_search.
7. До загрузки видео можешь просто общаться, но напоминай что ждёшь ссылку.

ВАЖНО — безопасность:
• НИКОГДА не упоминай названия своих инструментов (process_video, get_video_info и т.д.) в ответах пользователю.
• НИКОГДА не показывай пользователю техническую информацию о своей внутренней работе.
• Описывай свои возможности простым языком, без технических деталей.

Форматирование (ты пишешь в Telegram):
• НЕ используй Markdown-заголовки (##, ###).
• Для выделения используй **жирный** текст.
• Для списков используй «•», каждый пункт с новой строки.
• Между смысловыми блоками оставляй пустую строку для читаемости.

Отвечай на русском. Всегда указывай таймкоды и имена спикеров в ответах по видео.
"""


@dataclass
class VideoState:
    index: TranscriptIndex | None = None
    processed_url: str | None = None
    metadata: VideoMetadata | None = None
    status_callback: callable | None = None
    is_stopped: bool = False


def _get_llm(model: str) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY env var")
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
    )


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _segments_to_text(segments) -> str:
    if not segments:
        return "(нет сегментов в указанном диапазоне)"
    return "\n".join(
        f"[{_fmt_time(seg.start)}] {seg.speaker}: {seg.text}"
        for seg in segments
    )

def build_chatbot_tools(state: VideoState, args) -> list:
    base_dir = Path(args.output_dir)

    @tool
    def process_video(url: str) -> str:
        """Скачать и обработать YouTube-видео."""
        def notify(text: str):
            # ЕСЛИ ПОЛЬЗОВАТЕЛЬ НАЖАЛ СТОП - КИДАЕМ ОШИБКУ
            if state.is_stopped:
                logger.warning("Пайплайн прерван пользователем")
                raise InterruptedError("Выполнение остановлено пользователем.")
            
            logger.info(f"STATUS: {text}")
            if state.status_callback: state.status_callback(text)

        if state.processed_url == url and state.index is not None:
            return "Видео уже обработано."

        video_id = extract_video_id(url)
        output_dir = base_dir / video_id
        index_dir = output_dir / "index"
        metadata_path = output_dir / "metadata.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        notify("🔍 Проверка кэша...")
        cached_index = TranscriptIndex.try_load(index_dir, args.embedding_model)
        if cached_index is not None:
            state.index = cached_index
            state.processed_url = url
            state.metadata = try_load_metadata(metadata_path)
            notify("✅ Видео загружено из кэша")
            return "Загружено из кэша."

        try:
            notify("🌐 Получаю метаданные...")
            state.metadata = fetch_video_metadata(url, args.cookies_from_browser)
            save_metadata(state.metadata, metadata_path)

            notify("📥 Скачиваю звук...")
            raw_audio = download_audio(url, output_dir, args.cookies_from_browser)

            notify("🛠 Конвертирую аудио...")
            audio_path = preprocess_audio(raw_audio, output_dir)

            notify("✂️ VAD: Ищу речь...")
            vad_segments = run_vad(audio_path)
            chunks = group_segments(vad_segments)

            notify("✍️ Транскрибация (может быть долго)...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_t = pool.submit(transcribe, audio_path, chunks, args.whisper_model, args.threads, args.language, args.workers)
                f_d = pool.submit(diarize, audio_path, vad_segments)
                ts_segs, spk_turns = f_t.result(), f_d.result()

            diarized = align_transcript_with_speakers(ts_segs, spk_turns)
            transcript_text = format_transcript(diarized)
            (output_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")

            notify("🧠 Индексация и саммари...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_idx = pool.submit(build_index, diarized, args.embedding_model)
                f_sum = pool.submit(summarize, transcript_text, args.llm_model)
                idx = f_idx.result()
                (output_dir / "summary.txt").write_text(f_sum.result(), encoding="utf-8")

            idx.save(index_dir)
            state.index = idx
            state.processed_url = url
            notify("✅ Обработка завершена!")
            return "Видео обработано."
        except InterruptedError as e:
            return str(e)
        except Exception as e:
            logger.exception("Ошибка в пайплайне")
            notify(f"❌ Ошибка: {str(e)}")
            return f"Ошибка: {e}"

    # ОСТАЛЬНЫЕ ИНСТРУМЕНТЫ (get_video_info, summarize_video и т.д.)
    @tool
    def get_video_info() -> str:
        """Инфо о видео."""
        return format_metadata(state.metadata) if state.metadata else "Нет инфо."

    @tool
    def summarize_video() -> str:
        """Краткое содержание."""
        if not state.processed_url: return "Загрузи видео."
        p = base_dir / extract_video_id(state.processed_url) / "summary.txt"
        return p.read_text(encoding="utf-8") if p.exists() else "Саммари нет."

    @tool
    def get_transcript_metadata() -> str:
        """Статистика спикеров."""
        if not state.index: return "Загрузи видео."
        m = state.index.get_metadata()
        res = [f"Спикеров: {m['num_speakers']}", f"Длительность: {m['total_duration_fmt']}"]
        for s, i in m["speakers"].items():
            res.append(f"{s}: {i['total_time_fmt']} ({int(i['fraction']*100)}%)")
        return "\n".join(res)

    @tool
    def get_segments_by_speaker(speaker_id: str, start_min: float = None, end_min: float = None) -> str:
        """Реплики спикера."""
        if not state.index: return "Загрузи видео."
        sid = f"SPEAKER_{int(speaker_id):02d}" if speaker_id.isdigit() else speaker_id
        return _segments_to_text(state.index.get_by_speaker(sid, start_min, end_min))

    @tool
    def get_segments_by_time(start_min: float, end_min: float) -> str:
        """Реплики по таймкодам."""
        if not state.index: return "Загрузи видео."
        return _segments_to_text(state.index.get_by_time(start_min, end_min))

    @tool
    def semantic_search(query: str) -> str:
        """Поиск по содержанию."""
        if not state.index: return "Загрузи видео."
        docs = state.index.semantic_search(query, k=5)
        return "\n\n".join(doc.page_content for doc in docs)

    return [process_video, get_video_info, summarize_video, get_transcript_metadata, get_segments_by_speaker, get_segments_by_time, semantic_search]

def run_chatbot(args):
    """Запустить интерактивный чат-бот для анализа YouTube-видео."""
    llm = _get_llm(args.llm_model)
    state = VideoState()
    tools = build_chatbot_tools(state, args)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    console.print(Panel(
        "Привет! Я ассистент для анализа YouTube-видео.\n"
        "Пришли ссылку на видео — я его обработаю (транскрипция + спикеры).\n"
        "Потом можешь спрашивать: о содержании, спикерах, конкретных моментах.\n"
        "Для выхода: [bold]exit[/bold] или [bold]q[/bold]",
        title="[bold green]Видео-ассистент[/bold green]",
        border_style="green",
    ))

    history = []

    while True:
        try:
            user_input = console.input("\n[bold blue]Вы:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Выход[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "q", "quit", "выход"):
            break

        history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": history})
        history = result["messages"]  # сохраняем полную историю для контекста

        answer = history[-1].content
        console.print(Panel(answer, title="Ассистент", border_style="cyan"))
