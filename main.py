import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

from pipeline.download import download_audio, preprocess_audio
from pipeline.vad import run_vad, group_segments
from pipeline.transcribe import transcribe
from pipeline.diarize import (
    DiarizedSegment,
    diarize,
    align_transcript_with_speakers,
    format_transcript,
)
from pipeline.summarize import summarize
from pipeline.rag import build_index, ask

console = Console()


def _process_video(args) -> tuple[str, list[DiarizedSegment] | None]:
    """Run the full pipeline: download -> preprocess -> VAD -> ASR -> diarize.
    Returns (transcript_text, diarized_segments_or_None).
    """
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

    # 5. Speaker diarization + alignment
    diarized_segments = None
    if args.no_diarize:
        transcript_text = "\n".join(
            f"[{_fmt_time(seg.start)}] {seg.text}" for seg in transcript_segments
        )
    else:
        speaker_turns = diarize(audio_path, vad_segments)
        diarized_segments = align_transcript_with_speakers(
            transcript_segments, speaker_turns
        )
        transcript_text = format_transcript(diarized_segments)

    # Save transcript
    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")
    console.print(f"\nTranscript saved to [bold]{transcript_path}[/bold]")

    return transcript_text, diarized_segments


def _run_summary(transcript_text: str, args):
    """Summarize and save."""
    output_dir = Path(args.output_dir)
    summary = summarize(transcript_text, model=args.llm_model)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    console.print(Panel(summary, title="Summary", border_style="green"))
    console.print(f"Summary saved to [bold]{summary_path}[/bold]")


def _run_qa(diarized_segments: list[DiarizedSegment], args):
    """Build RAG index and enter interactive Q&A loop."""
    index = build_index(diarized_segments, embedding_model=args.embedding_model)

    console.print(
        "\n[bold green]Q&A mode ready.[/bold green] "
        "Задавай вопросы по видео. Для выхода введи 'exit' или 'q'.\n"
    )

    while True:
        try:
            question = console.input("[bold blue]Вопрос:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Выход[/dim]")
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in ("exit", "q", "quit", "выход"):
            break

        answer = ask(question, index, model=args.llm_model)
        console.print(Panel(answer, title="Ответ", border_style="cyan"))
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description="YouTube video summarizer & Q&A (Russian, local inference)"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "mode",
        choices=["summary", "qa"],
        help="Mode: 'summary' for summarization, 'qa' for interactive Q&A",
    )
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
        help="LLM model for summarization / Q&A via OpenRouter",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="HuggingFace embedding model for RAG (default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)",
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

    if args.mode == "qa" and args.no_diarize:
        console.print(
            "[bold red]Error:[/bold red] Q&A mode requires diarization "
            "(speaker info needed for context). Remove --no-diarize."
        )
        raise SystemExit(1)

    start = time.perf_counter()

    transcript_text, diarized_segments = _process_video(args)

    end = time.perf_counter()
    
    print(f"process video time: {(end - start)/60:.2f} minutes")

    if args.mode == "summary":
        _run_summary(transcript_text, args)
    elif args.mode == "qa":
        _run_qa(diarized_segments, args)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
