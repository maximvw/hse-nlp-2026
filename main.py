import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from pipeline.chatbot import run_chatbot

_THREADS_PER_WORKER = 4  # sweet spot for whisper.cpp encoder


def main():
    parser = argparse.ArgumentParser(
        description="YouTube video assistant: transcription, diarization, Q&A, summary"
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
        default="nvidia/nemotron-3-nano-30b-a3b:free",
        help="LLM model via OpenRouter — must support tool calling (default: nvidia/nemotron-nano-9b-v2:free)",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="HuggingFace embedding model for semantic search",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Language for ASR (default: ru)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=f"Threads per whisper worker (default: {_THREADS_PER_WORKER}). "
             "Ignored when --auto-workers is set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel whisper worker processes (default: 1). "
             "Ignored when --auto-workers is set.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="Browser to extract cookies from for yt-dlp (e.g. chrome, firefox, safari)",
    )
    parser.add_argument(
        "--auto-workers",
        action="store_true",
        help=f"Use all CPU cores optimally: workers = cpu_count // {_THREADS_PER_WORKER}, "
             f"{_THREADS_PER_WORKER} threads each. Overrides --threads and --workers.",
    )
    args = parser.parse_args()

    cpu = os.cpu_count() or 4
    if args.auto_workers:
        args.workers = max(1, cpu // _THREADS_PER_WORKER)
        args.threads = _THREADS_PER_WORKER
        print(
            f"[auto-workers] {cpu} CPU cores → "
            f"{args.workers} workers × {args.threads} threads"
        )
    else:
        args.threads = args.threads or _THREADS_PER_WORKER
        args.workers = args.workers or 1

    run_chatbot(args)


if __name__ == "__main__":
    main()
