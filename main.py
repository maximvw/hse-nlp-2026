import argparse

from dotenv import load_dotenv

load_dotenv()

from pipeline.chatbot import run_chatbot


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
        default="nvidia/nemotron-nano-9b-v2:free",
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
        default=8,
        help="Number of threads for whisper.cpp (default: 8)",
    )
    args = parser.parse_args()
    run_chatbot(args)


if __name__ == "__main__":
    main()
