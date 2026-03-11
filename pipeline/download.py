import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from rich.console import Console

console = Console()

DESCRIPTION_MAX_CHARS = 2000


@dataclass
class VideoMetadata:
    url: str
    title: str = ""
    description: str = ""
    channel: str = ""
    upload_date: str = ""
    view_count: int | None = None
    tags: list[str] = field(default_factory=list)
    chapters: list[dict] = field(default_factory=list)  # [{"title": ..., "start_time": ...}]


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/").split("?")[0]
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    # shorts, embed, etc.
    parts = [p for p in parsed.path.split("/") if p]
    return parts[-1] if parts else "video"


def fetch_video_metadata(url: str) -> VideoMetadata:
    """Fetch video metadata from YouTube using yt-dlp (no download)."""
    console.print(f"[bold]Fetching metadata for:[/bold] {url}")
    result = subprocess.run(
        ["yt-dlp", "-j", "--no-playlist", url],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)

    description = data.get("description") or ""
    if len(description) > DESCRIPTION_MAX_CHARS:
        description = description[:DESCRIPTION_MAX_CHARS] + "…"

    chapters = [
        {"title": ch.get("title", ""), "start_time": ch.get("start_time", 0)}
        for ch in (data.get("chapters") or [])
    ]

    return VideoMetadata(
        url=url,
        title=data.get("title", ""),
        description=description,
        channel=data.get("channel") or data.get("uploader", ""),
        upload_date=data.get("upload_date", ""),
        view_count=data.get("view_count"),
        tags=data.get("tags") or [],
        chapters=chapters,
    )


def format_metadata(meta: VideoMetadata) -> str:
    """Format video metadata as a human-readable string."""
    lines = [
        f"Название: {meta.title}",
        f"Канал: {meta.channel}",
    ]
    if meta.upload_date:
        d = meta.upload_date
        lines.append(f"Дата публикации: {d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else f"Дата: {d}")
    if meta.view_count is not None:
        lines.append(f"Просмотры: {meta.view_count:,}")
    if meta.tags:
        lines.append(f"Теги: {', '.join(meta.tags[:10])}")
    if meta.chapters:
        lines.append("\nГлавы:")
        for ch in meta.chapters:
            m, s = divmod(int(ch["start_time"]), 60)
            h, m = divmod(m, 60)
            ts = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
            lines.append(f"  [{ts}] {ch['title']}")
    if meta.description:
        lines.append(f"\nОписание:\n{meta.description}")
    return "\n".join(lines)


def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio from YouTube video using yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "raw_audio.%(ext)s")

    console.print(f"[bold]Downloading audio from:[/bold] {url}")
    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--force-overwrites",
            "-o", output_template,
            "--no-playlist",
            url,
        ],
        check=True,
    )

    raw_path = output_dir / "raw_audio.wav"
    if not raw_path.exists():
        raise FileNotFoundError(f"Downloaded audio not found at {raw_path}")

    return raw_path


def preprocess_audio(input_path: Path, output_dir: Path) -> Path:
    """Convert audio to 16kHz mono WAV for ASR/VAD."""
    output_path = output_dir / "audio_16k.wav"

    console.print("[bold]Preprocessing audio:[/bold] 16kHz mono WAV")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    return output_path
