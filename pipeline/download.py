import subprocess
from pathlib import Path

from rich.console import Console

console = Console()


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
