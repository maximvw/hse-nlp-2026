from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from pipeline.diarize import diarize, align_transcript_with_speakers, format_transcript
from pipeline.download import download_audio, preprocess_audio, fetch_video_metadata, format_metadata, VideoMetadata, extract_video_id
from pipeline.index import TranscriptIndex, build_index
from pipeline.summarize import summarize
from pipeline.transcribe import transcribe
from pipeline.vad import run_vad, group_segments
    
console = Console()

SYSTEM_PROMPT = """\
Ты — интеллектуальный ассистент для анализа YouTube-видео на русском языке.

Твои возможности:
- process_video(url) — скачать и обработать видео по ссылке (транскрипция + диаризация спикеров)
- get_video_info() — название, описание, канал, теги, главы видео (метаданные из YouTube)
- summarize_video() — сделать краткое содержание обработанного видео
- get_transcript_metadata() — количество спикеров, длительность, статистика по времени речи
- get_segments_by_speaker(speaker_id, start_min, end_min) — реплики конкретного спикера
- get_segments_by_time(start_min, end_min) — всё что происходило в заданный временной промежуток
- semantic_search(query) — тематический поиск по содержанию

Правила:
1. Как только пользователь присылает YouTube-ссылку — сразу вызывай process_video, не переспрашивай.
2. Для вопросов о названии, теме, канале, главах видео → get_video_info.
3. Для структурных вопросов ("сколько спикеров", "кто говорил дольше") → get_transcript_metadata.
4. Для вопросов о конкретном спикере → get_segments_by_speaker.
5. Для вопросов о конкретном времени → get_segments_by_time.
6. Для тематических вопросов ("что обсуждали про X") → semantic_search.
7. До загрузки видео можешь просто общаться, но напоминай что ждёшь ссылку.

Отвечай на русском. Всегда указывай таймкоды и имена спикеров в ответах по видео.
"""


@dataclass
class VideoState:
    index: TranscriptIndex | None = None
    processed_url: str | None = None
    metadata: VideoMetadata | None = None


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
        """Скачать и обработать YouTube-видео: транскрипция + диаризация спикеров.
        Вызывай сразу как только пользователь пришлёт ссылку на YouTube-видео.

        Args:
            url: Ссылка на YouTube-видео.
        """
        video_id = extract_video_id(url)
        output_dir = base_dir / video_id
        console.print(f"\n[bold]Processing:[/bold] {url}  [dim](output: {output_dir})[/dim]")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0 = time.perf_counter()

            def _step(name: str, t_prev: float) -> float:
                t = time.perf_counter()
                console.print(f"  [dim]{name}: {t - t_prev:.1f}s[/dim]")
                return t

            t = t0
            try:
                state.metadata = fetch_video_metadata(url)
            except Exception as e:
                console.print(f"  [yellow]Metadata fetch failed: {e}[/yellow]")
                state.metadata = None
            t = _step("metadata", t)

            raw_audio = download_audio(url, output_dir)
            t = _step("download", t)

            audio_path = preprocess_audio(raw_audio, output_dir)
            t = _step("preprocess", t)

            vad_segments = run_vad(audio_path)
            t = _step("VAD", t)

            chunks = group_segments(vad_segments)

            # Run transcription and diarization concurrently — they only need audio_path
            console.print("  [dim]Running transcribe + diarize in parallel...[/dim]")
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_transcript = pool.submit(
                    transcribe, audio_path, chunks,
                    args.whisper_model, args.threads, args.language, args.workers,
                )
                f_diarize = pool.submit(diarize, audio_path, vad_segments)
                transcript_segments = f_transcript.result()
                speaker_turns = f_diarize.result()
            t = _step("transcribe+diarize", t)

            diarized = align_transcript_with_speakers(transcript_segments, speaker_turns)
            transcript_text = format_transcript(diarized)

            transcript_path = output_dir / "transcript.txt"
            transcript_path.write_text(transcript_text, encoding="utf-8")

            idx = build_index(diarized, embedding_model=args.embedding_model)
            t = _step("index", t)

            state.index = idx
            state.processed_url = url

            elapsed = time.perf_counter() - t0
            n_speakers = len({seg.speaker for seg in diarized})
            duration = _fmt_time(diarized[-1].end if diarized else 0)

            console.print(f"[bold green]Готово за {elapsed:.1f}s[/bold green]")
            return (
                f"Видео обработано за {elapsed:.0f}с ({elapsed/60:.1f} мин).\n"
                f"Длительность: {duration}, спикеров: {n_speakers}.\n"
                f"Транскрипция сохранена: {transcript_path}.\n"
                f"Теперь можешь задавать вопросы по видео."
            )
        except Exception as e:
            return f"Ошибка при обработке видео: {e}"

    @tool
    def get_video_info() -> str:
        """Возвращает метаданные видео из YouTube: название, канал, дату публикации,
        просмотры, теги, главы и описание.
        Используй для вопросов: 'о чём это видео?', 'что за канал?', 'какие темы в видео?'"""
        if state.metadata is None:
            return "Видео ещё не загружено. Пришли ссылку на YouTube-видео."
        return format_metadata(state.metadata)

    @tool
    def summarize_video() -> str:
        """Сделать краткое содержание (саммари) обработанного видео.
        Выделяет основные темы и ключевые тезисы."""
        if state.index is None:
            return "Видео ещё не загружено. Пришли ссылку на YouTube-видео."
        transcript_text = format_transcript(state.index.segments)
        return summarize(transcript_text, model=args.llm_model)

    @tool
    def get_transcript_metadata() -> str:
        """Возвращает общую информацию о видео: количество спикеров, длительность,
        время речи каждого спикера и долю эфирного времени.
        Используй для вопросов: 'сколько спикеров?', 'кто говорил дольше всех?'"""
        if state.index is None:
            return "Видео ещё не загружено. Пришли ссылку на YouTube-видео."
        meta = state.index.get_metadata()
        lines = [
            f"Длительность видео: {meta['total_duration_fmt']}",
            f"Количество спикеров: {meta['num_speakers']}",
            "",
            "Статистика по спикерам:",
        ]
        for speaker, info in meta["speakers"].items():
            lines.append(
                f"  {speaker}: {info['total_time_fmt']} "
                f"({int(info['fraction'] * 100)}% эфирного времени)"
            )
        return "\n".join(lines)

    @tool
    def get_segments_by_speaker(
        speaker_id: str,
        start_min: float | None = None,
        end_min: float | None = None,
    ) -> str:
        """Возвращает реплики конкретного спикера из транскрипции.
        Используй для: 'что сказал спикер №2?', 'что говорил SPEAKER_01 на 10-й минуте?'

        Args:
            speaker_id: ID спикера. Форматы: 'SPEAKER_00', 'SPEAKER_01', или просто '0', '1', '2'.
            start_min: Начало временного диапазона в минутах (опционально).
            end_min: Конец временного диапазона в минутах (опционально).
        """
        if state.index is None:
            return "Видео ещё не загружено. Пришли ссылку на YouTube-видео."
        sid = speaker_id.strip()
        if sid.isdigit():
            sid = f"SPEAKER_{int(sid):02d}"
        segments = state.index.get_by_speaker(sid, start_min, end_min)
        header = f"Реплики {sid}"
        if start_min is not None or end_min is not None:
            t_from = f"{start_min:.1f}" if start_min is not None else "начала"
            t_to = f"{end_min:.1f}" if end_min is not None else "конца"
            header += f" (с {t_from} по {t_to} мин)"
        return f"{header}:\n{_segments_to_text(segments)}"

    @tool
    def get_segments_by_time(start_min: float, end_min: float) -> str:
        """Возвращает все реплики в заданном временном диапазоне.
        Используй для: 'что было на 5-10 минуте?', 'что происходило в начале видео?'

        Args:
            start_min: Начало диапазона в минутах.
            end_min: Конец диапазона в минутах.
        """
        if state.index is None:
            return "Видео ещё не загружено. Пришли ссылку на YouTube-видео."
        segments = state.index.get_by_time(start_min, end_min)
        return (
            f"Транскрипция с {start_min:.1f} по {end_min:.1f} мин:\n"
            + _segments_to_text(segments)
        )

    @tool
    def semantic_search(query: str) -> str:
        """Семантический поиск по содержанию транскрипции.
        Используй для тематических вопросов: 'что обсуждалось про X?', 'когда упоминалось Y?'

        Args:
            query: Поисковый запрос на русском языке.
        """
        if state.index is None:
            return "Видео ещё не загружено. Пришли ссылку на YouTube-видео."
        docs = state.index.semantic_search(query, k=5)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    return [
        process_video,
        get_video_info,
        summarize_video,
        get_transcript_metadata,
        get_segments_by_speaker,
        get_segments_by_time,
        semantic_search,
    ]


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
