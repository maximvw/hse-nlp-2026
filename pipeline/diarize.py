from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from rich.console import Console
from sklearn.cluster import AgglomerativeClustering

from pipeline.transcribe import TranscriptSegment
from pipeline.vad import SpeechSegment

console = Console()

SAMPLE_RATE = 16000
SPEAKER_MODEL = "microsoft/wavlm-base-sv"


def _load_speaker_model():
    """Load WavLM speaker verification model from HuggingFace."""
    from transformers import AutoFeatureExtractor, WavLMForXVector

    console.print(f"  Loading speaker model [dim]{SPEAKER_MODEL}[/dim]...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL)
    model = WavLMForXVector.from_pretrained(SPEAKER_MODEL)
    model.eval()
    return feature_extractor, model


def _embed_segment(
    audio_np: np.ndarray,
    feature_extractor,
    model: torch.nn.Module,
) -> np.ndarray | None:
    """Compute L2-normalized speaker embedding for one audio chunk (numpy, 16kHz)."""
    if len(audio_np) < int(SAMPLE_RATE * 0.2):
        return None

    inputs = feature_extractor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        embedding = model(**inputs).embeddings  # (1, hidden_size)

    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding.squeeze(0).numpy()


@dataclass
class DiarizedSegment:
    speaker: str
    start: float
    end: float
    text: str


def diarize(
    audio_path: Path,
    vad_segments: list[SpeechSegment],
    num_speakers: int | None = None,
    threshold: float = 0.4,
) -> list[tuple[float, float, str]]:
    """Speaker diarization via WavLM-SV embeddings + agglomerative clustering.

    Uses VAD segments already computed by Silero to avoid redundant segmentation.
    Each VAD segment is embedded with microsoft/wavlm-base-sv (x-vector head),
    then segments are clustered by cosine distance.

    Args:
        audio_path: Path to 16kHz mono WAV.
        vad_segments: Speech segments from Silero VAD.
        num_speakers: If known, fix the number of speakers. Otherwise auto-detect.
        threshold: Cosine distance threshold for clustering when num_speakers is None.
                   Lower = more speakers detected. Typical range: 0.3–0.6.

    Returns:
        List of (start_sec, end_sec, speaker_label).
    """
    if not vad_segments:
        return []

    console.print("[bold]Running speaker diarization (WavLM-SV + clustering)...[/bold]")

    feature_extractor, model = _load_speaker_model()

    # Load full audio once, convert to numpy for feature extractor
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_np = waveform.squeeze(0).numpy()  # (samples,)

    # Extract embedding per VAD segment
    embeddings: list[np.ndarray | None] = []
    for seg in vad_segments:
        start_sample = int(seg.start * SAMPLE_RATE)
        end_sample = int(seg.end * SAMPLE_RATE)
        chunk = audio_np[start_sample:end_sample]
        embeddings.append(_embed_segment(chunk, feature_extractor, model))

    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    valid_embeddings = np.array([embeddings[i] for i in valid_indices])

    if len(valid_embeddings) == 0:
        return []

    # Cluster
    if len(valid_embeddings) == 1:
        labels = np.array([0])
    else:
        clustering_kwargs: dict = {"metric": "cosine", "linkage": "average"}
        if num_speakers is not None:
            clustering_kwargs["n_clusters"] = num_speakers
        else:
            clustering_kwargs["n_clusters"] = None
            clustering_kwargs["distance_threshold"] = threshold

        labels = AgglomerativeClustering(**clustering_kwargs).fit_predict(valid_embeddings)

    # Build speaker turns, assign "nearest" label to skipped (too-short) segments
    turns: list[tuple[float, float, str]] = []
    label_iter = iter(zip(valid_indices, labels))
    next_valid_idx, next_label = next(label_iter, (None, None))

    for i, seg in enumerate(vad_segments):
        if i == next_valid_idx:
            speaker = f"SPEAKER_{next_label:02d}"
            next_valid_idx, next_label = next(label_iter, (None, None))
        else:
            speaker = turns[-1][2] if turns else "SPEAKER_00"
        turns.append((seg.start, seg.end, speaker))

    n_speakers = len(set(labels))
    console.print(
        f"  Found [green]{len(turns)}[/green] speaker turns, "
        f"[green]{n_speakers}[/green] speakers"
    )
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
    best_dist = float("inf")

    for start, end, speaker in turns:
        if start <= time <= end:
            return speaker
        dist = min(abs(time - start), abs(time - end))
        if dist < best_dist:
            best_dist = dist
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
