from __future__ import annotations

import os
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from pipeline.vad import SpeechSegment

console = Console()


@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def _extract_chunk(audio_path: Path, start: float, end: float, out_path: Path) -> None:
    """Cut a chunk from the audio file using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(start),
            "-to", str(end),
            "-ar", "16000", "-ac", "1",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


# ---------- multiprocessing worker (module-level for pickling) ----------

_worker_model = None
_worker_language: str = "ru"


def _worker_init(model_name: str, n_threads: int, language: str) -> None:
    global _worker_model, _worker_language
    from pywhispercpp.model import Model
    _worker_model = Model(model_name, n_threads=n_threads)
    _worker_language = language


def _worker_transcribe(args: tuple) -> list[dict]:
    """Transcribe one chunk. Returns plain dicts (picklable)."""
    chunk_path_str, chunk_start = args
    result = _worker_model.transcribe(
        chunk_path_str,
        language=_worker_language,
        token_timestamps=True,
    )
    segments = []
    for seg in result:
        offset = chunk_start
        words = []
        if hasattr(seg, "words") and seg.words:
            words = [
                {
                    "text": w.text.strip(),
                    "start": w.t0 / 100.0 + offset,
                    "end": w.t1 / 100.0 + offset,
                }
                for w in seg.words
                if w.text.strip()
            ]
        segments.append({
            "start": seg.t0 / 100.0 + offset,
            "end": seg.t1 / 100.0 + offset,
            "text": seg.text.strip(),
            "words": words,
        })
    return segments


# -----------------------------------------------------------------------


def transcribe(
    audio_path: Path,
    chunks: list[SpeechSegment],
    model_name: str = "large-v3-turbo-q5_0",
    n_threads: int = 8,
    language: str = "ru",
    n_workers: int = 1,
) -> list[TranscriptSegment]:
    """Transcribe audio chunks using whisper.cpp via pywhispercpp.

    Args:
        audio_path: Path to full 16kHz mono WAV.
        chunks: VAD-grouped speech chunks.
        model_name: whisper.cpp model (auto-downloaded by pywhispercpp).
        n_threads: Total CPU threads. Split evenly across workers.
        language: Language code.
        n_workers: Number of parallel worker processes (default 1 = sequential).
    """
    n_workers = max(1, n_workers)
    threads_per_worker = max(1, n_threads // n_workers)

    console.print(
        f"[bold]Transcribing {len(chunks)} chunks[/bold] "
        f"({n_workers} worker{'s' if n_workers > 1 else ''}, "
        f"{threads_per_worker} threads each, model: {model_name})"
    )

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Extract all chunks upfront (fast, sequential ffmpeg calls)
        chunk_paths = []
        for i, chunk in enumerate(chunks):
            p = tmpdir / f"chunk_{i:04d}.wav"
            _extract_chunk(audio_path, chunk.start, chunk.end, p)
            chunk_paths.append(p)

        worker_args = [(str(p), chunk.start) for p, chunk in zip(chunk_paths, chunks)]

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(model_name, threads_per_worker, language),
        ) as pool:
            results = list(pool.map(_worker_transcribe, worker_args))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Flatten in original chunk order and convert to dataclasses
    segments: list[TranscriptSegment] = []
    for i, chunk_segs in enumerate(results):
        for seg in chunk_segs:
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                words=[Word(**w) for w in seg["words"]],
            ))
        console.print(f"  Chunk {i + 1}/{len(chunks)} done")

    console.print(f"  Total segments: [green]{len(segments)}[/green]")
    return segments
