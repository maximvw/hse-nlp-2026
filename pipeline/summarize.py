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

ROLLING_MERGE_FIRST_PROMPT = """\
Ниже приведён первый фрагмент транскрипции видео.
Напиши саммари этого фрагмента на русском языке, выделяя основные тезисы.

Фрагмент транскрипции:
{chunk}
"""

ROLLING_MERGE_NEXT_PROMPT = """\
Ниже приведены текущее накопленное саммари и следующий фрагмент транскрипции.
Обнови саммари, включив ключевые тезисы нового фрагмента. Сохрани связность и убери повторы.

Текущее саммари:
{current_summary}

Следующий фрагмент транскрипции:
{chunk}
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

# ~10 min of speech at ~150 wpm × ~6 chars/word + timestamps ≈ 5000 chars
SINGLE_PASS_THRESHOLD = 5_000
# ~30 min ≈ 15000 chars
HIERARCHICAL_THRESHOLD = 15_000
CHUNK_SIZE = 5_000


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

    Strategy selection by transcript length:
    - <= ~10 min: single pass (one LLM call)
    - 10–30 min:  rolling merge (summary_n = summarize(summary_{n-1} + chunk_n))
    - >  30 min:  hierarchical (summarize each chunk, then merge all)
    """
    if len(transcript) <= SINGLE_PASS_THRESHOLD:
        return _summarize_single_pass(transcript, model)
    if len(transcript) <= HIERARCHICAL_THRESHOLD:
        return _summarize_rolling_merge(transcript, model)
    return _summarize_hierarchical(transcript, model)


def _summarize_single_pass(transcript: str, model: str) -> str:
    """Single LLM call for short transcripts (< ~10 min)."""
    console.print("[bold]Summarizing (single pass)...[/bold]")

    llm = _get_llm(model)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=SUMMARIZE_PROMPT.format(transcript=transcript)),
    ])
    return response.content


def _summarize_rolling_merge(transcript: str, model: str) -> str:
    """Rolling merge for medium transcripts (10–30 min).

    summary_1 = summarize(chunk_1)
    summary_2 = summarize(summary_1 + chunk_2)
    summary_3 = summarize(summary_2 + chunk_3)
    ...
    """
    chunks = _split_transcript(transcript, CHUNK_SIZE)
    console.print(
        f"[bold]Summarizing (rolling merge):[/bold] {len(chunks)} chunks"
    )

    llm = _get_llm(model)

    console.print("  Chunk 1/{}...".format(len(chunks)))
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=ROLLING_MERGE_FIRST_PROMPT.format(chunk=chunks[0])),
    ])
    current_summary = response.content

    for i, chunk in enumerate(chunks[1:], start=2):
        console.print(f"  Chunk {i}/{len(chunks)}...")
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=ROLLING_MERGE_NEXT_PROMPT.format(
                current_summary=current_summary,
                chunk=chunk,
            )),
        ])
        current_summary = response.content

    return current_summary


def _summarize_hierarchical(transcript: str, model: str) -> str:
    """Hierarchical summarization for long transcripts (> ~30 min).

    Summarize each chunk independently, then merge all summaries in one call.
    """
    chunks = _split_transcript(transcript, HIERARCHICAL_THRESHOLD)
    console.print(
        f"[bold]Summarizing (hierarchical):[/bold] {len(chunks)} chunks"
    )

    llm = _get_llm(model)
    chunk_summaries: list[str] = []

    for i, chunk in enumerate(chunks):
        console.print(f"  Summarizing chunk {i + 1}/{len(chunks)}...")
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=CHUNK_SUMMARIZE_PROMPT.format(
                chunk_num=i + 1,
                total_chunks=len(chunks),
                transcript=chunk,
            )),
        ])
        chunk_summaries.append(response.content)

    console.print("  Merging chunk summaries...")
    all_summaries = "\n\n---\n\n".join(
        f"Часть {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=MERGE_PROMPT.format(summaries=all_summaries)),
    ])
    return response.content


def _split_transcript(transcript: str, chunk_size: int) -> list[str]:
    """Split transcript into chunks of ~chunk_size chars at line boundaries."""
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
