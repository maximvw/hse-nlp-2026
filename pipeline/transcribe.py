from __future__ import annotations

import subprocess
import tempfile
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


def transcribe(
    audio_path: Path,
    chunks: list[SpeechSegment],
    model_name: str = "large-v3-turbo-q5_0",
    n_threads: int = 8,
    language: str = "ru",
) -> list[TranscriptSegment]:
    """Transcribe audio chunks using whisper.cpp via pywhispercpp.

    Args:
        audio_path: Path to full 16kHz mono WAV.
        chunks: VAD-grouped speech chunks.
        model_name: whisper.cpp model (auto-downloaded by pywhispercpp).
        n_threads: Number of CPU threads.
        language: Language code.
    """
    from pywhispercpp.model import Model

    console.print(f"[bold]Loading whisper.cpp model:[/bold] {model_name}")
    model = Model(model_name, n_threads=n_threads)

    segments: list[TranscriptSegment] = []

    console.print(f"[bold]Transcribing {len(chunks)} chunks...[/bold]")
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, chunk in enumerate(chunks):
            chunk_path = Path(tmpdir) / f"chunk_{i:04d}.wav"
            _extract_chunk(audio_path, chunk.start, chunk.end, chunk_path)

            result = model.transcribe(
                str(chunk_path),
                language=language,
                token_timestamps=True,
            )

            for seg in result:
                # Adjust timestamps back to full-audio time
                offset = chunk.start
                words = []
                if hasattr(seg, "words") and seg.words:
                    words = [
                        Word(
                            text=w.text.strip(),
                            start=w.t0 / 100.0 + offset,
                            end=w.t1 / 100.0 + offset,
                        )
                        for w in seg.words
                        if w.text.strip()
                    ]

                segments.append(
                    TranscriptSegment(
                        start=seg.t0 / 100.0 + offset,
                        end=seg.t1 / 100.0 + offset,
                        text=seg.text.strip(),
                        words=words,
                    )
                )

            console.print(f"  Chunk {i + 1}/{len(chunks)} done")

    console.print(f"  Total segments: [green]{len(segments)}[/green]")
    return segments
