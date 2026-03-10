from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console

from pipeline.diarize import DiarizedSegment

console = Console()

SYSTEM_PROMPT = (
    "Ты — ассистент для ответов на вопросы по видео на русском языке. "
    "Отвечай точно и по делу, опираясь только на предоставленный контекст из транскрипции. "
    "Если в контексте нет информации для ответа — честно скажи об этом. "
    "Указывай таймкоды и спикеров, когда это уместно."
)

QA_PROMPT = """\
Контекст из транскрипции видео:

{context}

---

Вопрос: {question}
"""


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


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_index(
    segments: list[DiarizedSegment],
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_max_chars: int = 1000,
) -> FAISS:
    """Build a FAISS vector index from diarized transcript segments.

    Groups adjacent segments into chunks of ~chunk_max_chars,
    preserving speaker and timestamp metadata.
    """
    console.print(f"[bold]Building vector index...[/bold]")
    console.print(f"  Embedding model: {embedding_model}")

    chunks = _make_chunks(segments, chunk_max_chars)
    console.print(f"  Created [green]{len(chunks)}[/green] chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        # model_kwargs={"device": "mps"},
        model_kwargs={"device": "cpu"},
    )

    index = FAISS.from_documents(chunks, embeddings)
    console.print("  Index built")
    return index


def _make_chunks(
    segments: list[DiarizedSegment],
    chunk_max_chars: int,
) -> list[Document]:
    """Group diarized segments into chunks for embedding."""
    documents: list[Document] = []
    current_lines: list[str] = []
    current_len = 0
    chunk_start = segments[0].start if segments else 0.0
    chunk_speakers: set[str] = set()

    for seg in segments:
        line = f"[{_fmt_time(seg.start)}] {seg.speaker}: {seg.text}"
        line_len = len(line)

        if current_len + line_len > chunk_max_chars and current_lines:
            documents.append(Document(
                page_content="\n".join(current_lines),
                metadata={
                    "start": chunk_start,
                    "end": seg.start,
                    "speakers": ", ".join(sorted(chunk_speakers)),
                },
            ))
            current_lines = []
            current_len = 0
            chunk_start = seg.start
            chunk_speakers = set()

        current_lines.append(line)
        current_len += line_len + 1
        chunk_speakers.add(seg.speaker)

    if current_lines:
        last_end = segments[-1].end if segments else chunk_start
        documents.append(Document(
            page_content="\n".join(current_lines),
            metadata={
                "start": chunk_start,
                "end": last_end,
                "speakers": ", ".join(sorted(chunk_speakers)),
            },
        ))

    return documents


def ask(
    question: str,
    index: FAISS,
    model: str = "nvidia/nemotron-nano-9b-v2:free",
    top_k: int = 5,
) -> str:
    """Answer a question using RAG over the transcript."""
    docs = index.similarity_search(question, k=top_k)

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = _get_llm(model)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=QA_PROMPT.format(context=context, question=question)),
    ])

    return response.content
