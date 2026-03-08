from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console

console = Console()

SYSTEM_PROMPT = (
    "Ты — ассистент для создания кратких содержательных саммари видео на русском языке. "
    "Пиши чётко, структурированно, сохраняя ключевые тезисы и факты. "
    "Используй маркированные списки для основных пунктов."
)

SUMMARIZE_PROMPT = """\
Ниже приведена транскрипция видео. Напиши подробное, но лаконичное саммари на русском языке.
Выдели основные темы и ключевые тезисы.

Транскрипция:
{transcript}
"""

CHUNK_SUMMARIZE_PROMPT = """\
Ниже приведён фрагмент транскрипции видео (часть {chunk_num} из {total_chunks}).
Напиши саммари этого фрагмента на русском языке, выделяя основные тезисы.

Фрагмент транскрипции:
{transcript}
"""

MERGE_PROMPT = """\
Ниже приведены саммари отдельных частей одного видео. Объедини их в единое связное саммари.
Убери повторы, сохрани ключевые тезисы и логическую структуру.

Саммари частей:
{summaries}
"""

# ~30 min of speech at ~150 words/min = ~4500 words
MERGE_THRESHOLD_CHARS = 15_000


def _get_llm(model: str) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY env var")

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
    )


def summarize(
    transcript: str,
    model: str = "nvidia/nemotron-nano-9b-v2:free",
) -> str:
    """Summarize transcript using LLM via OpenRouter.

    Uses merge strategy for short transcripts (<= ~30 min)
    and hierarchical strategy for longer ones.
    """
    if len(transcript) <= MERGE_THRESHOLD_CHARS:
        return _summarize_merge(transcript, model)
    return _summarize_hierarchical(transcript, model)


def _summarize_merge(transcript: str, model: str) -> str:
    """Single-pass summarization for short transcripts."""
    console.print("[bold]Summarizing (single pass)...[/bold]")

    llm = _get_llm(model)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=SUMMARIZE_PROMPT.format(transcript=transcript)),
    ])

    return response.content


def _summarize_hierarchical(transcript: str, model: str) -> str:
    """Hierarchical summarization: split into chunks, summarize each, then merge."""
    chunks = _split_transcript(transcript)
    console.print(
        f"[bold]Hierarchical summarization:[/bold] {len(chunks)} chunks"
    )

    llm = _get_llm(model)
    chunk_summaries: list[str] = []

    for i, chunk in enumerate(chunks):
        console.print(f"  Summarizing chunk {i + 1}/{len(chunks)}...")

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=CHUNK_SUMMARIZE_PROMPT.format(
                    chunk_num=i + 1,
                    total_chunks=len(chunks),
                    transcript=chunk,
                ),
            ),
        ])
        chunk_summaries.append(response.content)

    # Merge chunk summaries
    console.print("  Merging chunk summaries...")
    all_summaries = "\n\n---\n\n".join(
        f"Часть {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=MERGE_PROMPT.format(summaries=all_summaries)),
    ])

    return response.content


def _split_transcript(transcript: str, chunk_size: int = MERGE_THRESHOLD_CHARS) -> list[str]:
    """Split transcript into roughly equal chunks at line boundaries."""
    lines = transcript.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > chunk_size and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks
