import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

from pipeline.download import download_audio, preprocess_audio
from pipeline.vad import run_vad, group_segments
from pipeline.transcribe import transcribe
from pipeline.diarize import diarize, align_transcript_with_speakers, format_transcript
from pipeline.summarize import summarize

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="YouTube video summarizer (Russian, local inference)"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory for intermediate and output files (default: output)",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3-turbo-q5_0",
        help="whisper.cpp model name (default: large-v3-turbo-q5_0)",
    )
    parser.add_argument(
        "--llm-model",
        default="nvidia/nemotron-nano-9b-v2:free",
        help="LLM model for summarization via OpenRouter",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Language for ASR (default: ru)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads for whisper.cpp (default: 8)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization (no HF_TOKEN needed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download audio
    raw_audio = download_audio(args.url, output_dir)

    # 2. Preprocess: 16kHz mono WAV
    audio_path = preprocess_audio(raw_audio, output_dir)

    # 3. VAD segmentation
    vad_segments = run_vad(audio_path)
    chunks = group_segments(vad_segments)

    # 4. ASR transcription
    transcript_segments = transcribe(
        audio_path,
        chunks,
        model_name=args.whisper_model,
        n_threads=args.threads,
        language=args.language,
    )

    # 5. Speaker diarization + alignment (optional)
    if args.no_diarize:
        transcript_text = "\n".join(
            f"[{_fmt_time(seg.start)}] {seg.text}" for seg in transcript_segments
        )
    else:
        speaker_turns = diarize(audio_path)
        diarized = align_transcript_with_speakers(transcript_segments, speaker_turns)
        transcript_text = format_transcript(diarized)

    # Save transcript
    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")
    console.print(f"\nTranscript saved to [bold]{transcript_path}[/bold]")

    # 6. Summarization
    summary = summarize(transcript_text, model=args.llm_model)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    console.print(Panel(summary, title="Summary", border_style="green"))
    console.print(f"Summary saved to [bold]{summary_path}[/bold]")


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
