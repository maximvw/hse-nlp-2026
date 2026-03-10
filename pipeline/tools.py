from __future__ import annotations

from langchain_core.tools import tool

from pipeline.index import TranscriptIndex
from pipeline.diarize import DiarizedSegment


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _segments_to_text(segments: list[DiarizedSegment]) -> str:
    if not segments:
        return "(нет сегментов в указанном диапазоне)"
    return "\n".join(
        f"[{_fmt_time(seg.start)}] {seg.speaker}: {seg.text}"
        for seg in segments
    )


def build_tools(index: TranscriptIndex) -> list:
    """Создать список LangChain-инструментов, привязанных к индексу транскрипции."""

    @tool
    def get_transcript_metadata() -> str:
        """Возвращает общую информацию о видео: количество спикеров, общую длительность,
        время речи каждого спикера и их долю эфирного времени.
        Используй для вопросов: 'сколько спикеров?', 'кто говорил дольше всех?'."""
        meta = index.get_metadata()
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
        Используй для вопросов: 'что сказал спикер №2?', 'что говорил SPEAKER_01 на 10-й минуте?'

        Args:
            speaker_id: ID спикера. Форматы: 'SPEAKER_00', 'SPEAKER_01', или просто '0', '1', '2'.
            start_min: Начало временного диапазона в минутах (опционально).
            end_min: Конец временного диапазона в минутах (опционально).
        """
        sid = speaker_id.strip()
        if sid.isdigit():
            sid = f"SPEAKER_{int(sid):02d}"

        segments = index.get_by_speaker(sid, start_min, end_min)

        header = f"Реплики {sid}"
        if start_min is not None or end_min is not None:
            t_from = f"{start_min:.1f}" if start_min is not None else "начала"
            t_to = f"{end_min:.1f}" if end_min is not None else "конца"
            header += f" (с {t_from} по {t_to} мин)"

        return f"{header}:\n{_segments_to_text(segments)}"

    @tool
    def get_segments_by_time(start_min: float, end_min: float) -> str:
        """Возвращает все реплики всех спикеров в заданном временном диапазоне.
        Используй для вопросов: 'что происходило на 5-10 минуте?', 'что было в начале видео?'

        Args:
            start_min: Начало диапазона в минутах.
            end_min: Конец диапазона в минутах.
        """
        segments = index.get_by_time(start_min, end_min)
        return (
            f"Транскрипция с {start_min:.1f} по {end_min:.1f} мин:\n"
            + _segments_to_text(segments)
        )

    @tool
    def semantic_search(query: str) -> str:
        """Семантический поиск по содержанию транскрипции.
        Используй для тематических вопросов: 'что обсуждалось про X?', 'когда упоминалось Y?',
        'о чём говорили в целом?'

        Args:
            query: Поисковый запрос на русском языке.
        """
        docs = index.semantic_search(query, k=5)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    return [
        get_transcript_metadata,
        get_segments_by_speaker,
        get_segments_by_time,
        semantic_search,
    ]
