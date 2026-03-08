from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console

console = Console()

SAMPLE_RATE = 16000


@dataclass
class SpeechSegment:
    start: float  # seconds
    end: float    # seconds


def run_vad(audio_path: Path, merge_gap: float = 0.5, min_duration: float = 0.3) -> list[SpeechSegment]:
    """Run Silero VAD and return speech segments.

    Args:
        audio_path: Path to 16kHz mono WAV.
        merge_gap: Merge segments closer than this (seconds).
        min_duration: Drop segments shorter than this (seconds).
    """
    console.print("[bold]Running VAD segmentation (Silero)...[/bold]")

    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    get_speech_timestamps, _, read_audio, _, _ = utils

    wav = read_audio(str(audio_path), sampling_rate=SAMPLE_RATE)

    raw_timestamps = get_speech_timestamps(
        wav, model, sampling_rate=SAMPLE_RATE, return_seconds=True
    )

    # Convert to SpeechSegment and merge close segments
    segments: list[SpeechSegment] = []
    for ts in raw_timestamps:
        seg = SpeechSegment(start=ts["start"], end=ts["end"])
        if seg.end - seg.start < min_duration:
            continue
        if segments and seg.start - segments[-1].end < merge_gap:
            segments[-1].end = seg.end
        else:
            segments.append(seg)

    console.print(f"  Found [green]{len(segments)}[/green] speech segments")
    return segments


def group_segments(segments: list[SpeechSegment], max_chunk_sec: float = 30.0) -> list[SpeechSegment]:
    """Group adjacent VAD segments into larger chunks for ASR (max ~30s each)."""
    if not segments:
        return []

    chunks: list[SpeechSegment] = []
    current = SpeechSegment(start=segments[0].start, end=segments[0].end)

    for seg in segments[1:]:
        if seg.end - current.start <= max_chunk_sec:
            current.end = seg.end
        else:
            chunks.append(current)
            current = SpeechSegment(start=seg.start, end=seg.end)
    chunks.append(current)

    console.print(f"  Grouped into [green]{len(chunks)}[/green] ASR chunks")
    return chunks
