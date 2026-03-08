from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from pipeline.transcribe import TranscriptSegment, Word

console = Console()


@dataclass
class DiarizedSegment:
    speaker: str
    start: float
    end: float
    text: str


def diarize(audio_path: Path) -> list[tuple[float, float, str]]:
    """Run pyannote speaker diarization.

    Requires HF_TOKEN env var and accepted model licenses:
    - https://huggingface.co/pyannote/speaker-diarization-3.1
    - https://huggingface.co/pyannote/segmentation-3.0

    Returns list of (start_sec, end_sec, speaker_label).
    """
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Set HF_TOKEN env var with your HuggingFace token. "
            "Also accept model licenses at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0"
        )

    console.print("[bold]Running speaker diarization (pyannote)...[/bold]")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline.to(device)

    output = pipeline(str(audio_path))

    # pyannote >= 3.3 returns DiarizeOutput; extract the Annotation from it
    annotation = getattr(output, "speaker_diarization", output)

    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))

    console.print(f"  Found [green]{len(turns)}[/green] speaker turns")
    return turns


def align_transcript_with_speakers(
    transcript: list[TranscriptSegment],
    speaker_turns: list[tuple[float, float, str]],
) -> list[DiarizedSegment]:
    """Align ASR transcript segments with speaker diarization by timestamp overlap."""
    result: list[DiarizedSegment] = []

    for seg in transcript:
        mid = (seg.start + seg.end) / 2.0
        speaker = _find_speaker_at(mid, speaker_turns)

        # Try to merge with previous segment if same speaker
        if result and result[-1].speaker == speaker:
            result[-1].end = seg.end
            result[-1].text += " " + seg.text
        else:
            result.append(DiarizedSegment(
                speaker=speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text,
            ))

    return result


def _find_speaker_at(time: float, turns: list[tuple[float, float, str]]) -> str:
    """Find the speaker active at a given time point."""
    best_speaker = "UNKNOWN"
    best_overlap = -1.0

    for start, end, speaker in turns:
        if start <= time <= end:
            return speaker
        # Fallback: nearest turn
        dist = min(abs(time - start), abs(time - end))
        if best_overlap < 0 or dist < best_overlap:
            best_overlap = dist
            best_speaker = speaker

    return best_speaker


def format_transcript(segments: list[DiarizedSegment]) -> str:
    """Format diarized transcript as readable text."""
    lines = []
    for seg in segments:
        ts = _fmt_time(seg.start)
        lines.append(f"[{ts}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
