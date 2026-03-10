from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rich.console import Console

from pipeline.diarize import DiarizedSegment

console = Console()


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class TranscriptIndex:
    segments: list[DiarizedSegment]
    faiss: FAISS

    def get_metadata(self) -> dict:
        """Общая статистика: кол-во спикеров, длительность, время каждого спикера."""
        speaker_time: dict[str, float] = {}
        for seg in self.segments:
            speaker_time[seg.speaker] = speaker_time.get(seg.speaker, 0.0) + (seg.end - seg.start)

        total_duration = self.segments[-1].end if self.segments else 0.0

        return {
            "total_duration_sec": total_duration,
            "total_duration_fmt": _fmt_time(total_duration),
            "num_speakers": len(speaker_time),
            "speakers": {
                sp: {
                    "total_time_sec": round(t, 1),
                    "total_time_fmt": _fmt_time(t),
                    "fraction": round(t / total_duration, 2) if total_duration > 0 else 0.0,
                }
                for sp, t in sorted(speaker_time.items())
            },
        }

    def get_by_speaker(
        self,
        speaker: str,
        start_min: float | None = None,
        end_min: float | None = None,
    ) -> list[DiarizedSegment]:
        """Реплики конкретного спикера, опционально в диапазоне времени (в минутах)."""
        result = []
        for seg in self.segments:
            if seg.speaker.upper() != speaker.upper():
                continue
            if start_min is not None and seg.end < start_min * 60:
                continue
            if end_min is not None and seg.start > end_min * 60:
                continue
            result.append(seg)
        return result

    def get_by_time(self, start_min: float, end_min: float) -> list[DiarizedSegment]:
        """Все реплики в заданном временном диапазоне (в минутах)."""
        start_sec = start_min * 60
        end_sec = end_min * 60
        return [
            seg for seg in self.segments
            if seg.start < end_sec and seg.end > start_sec
        ]

    def semantic_search(self, query: str, k: int = 5) -> list[Document]:
        """Семантический поиск по тексту транскрипции."""
        return self.faiss.similarity_search(query, k=k)


def build_index(
    segments: list[DiarizedSegment],
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_max_chars: int = 1000,
) -> TranscriptIndex:
    """Построить TranscriptIndex из диаризованных сегментов."""
    console.print("[bold]Building vector index...[/bold]")
    console.print(f"  Embedding model: {embedding_model}")

    chunks = _make_chunks(segments, chunk_max_chars)
    console.print(f"  Created [green]{len(chunks)}[/green] chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
    )

    faiss_index = FAISS.from_documents(chunks, embeddings)
    console.print("  Index built")

    return TranscriptIndex(segments=segments, faiss=faiss_index)


def _make_chunks(segments: list[DiarizedSegment], chunk_max_chars: int) -> list[Document]:
    documents: list[Document] = []
    current_lines: list[str] = []
    current_len = 0
    chunk_start = segments[0].start if segments else 0.0
    chunk_speakers: set[str] = set()

    for seg in segments:
        line = f"[{_fmt_time(seg.start)}] {seg.speaker}: {seg.text}"
        line_len = len(line)

        if current_len + line_len > chunk_max_chars and current_lines:
            documents.append(Document(
                page_content="\n".join(current_lines),
                metadata={
                    "start": chunk_start,
                    "end": seg.start,
                    "speakers": ", ".join(sorted(chunk_speakers)),
                },
            ))
            current_lines = []
            current_len = 0
            chunk_start = seg.start
            chunk_speakers = set()

        current_lines.append(line)
        current_len += line_len + 1
        chunk_speakers.add(seg.speaker)

    if current_lines:
        last_end = segments[-1].end if segments else chunk_start
        documents.append(Document(
            page_content="\n".join(current_lines),
            metadata={
                "start": chunk_start,
                "end": last_end,
                "speakers": ", ".join(sorted(chunk_speakers)),
            },
        ))

    return documents
