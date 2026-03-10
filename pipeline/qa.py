from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from rich.console import Console

from pipeline.index import TranscriptIndex
from pipeline.tools import build_tools

console = Console()

SYSTEM_PROMPT = """\
Ты — интеллектуальный ассистент для анализа видеотранскрипций на русском языке.

У тебя есть доступ к четырём инструментам:
- get_transcript_metadata — количество спикеров, длительность, статистика по времени речи
- get_segments_by_speaker — реплики конкретного спикера (можно с фильтром по времени)
- get_segments_by_time — все реплики в заданном временном промежутке
- semantic_search — поиск по смыслу/теме

Стратегия выбора инструмента:
1. Вопрос о количестве спикеров / кто говорил дольше → get_transcript_metadata
2. Вопрос о конкретном спикере ("что сказал спикер №2") → get_segments_by_speaker
3. Вопрос о конкретном времени ("что было на 10-й минуте") → get_segments_by_time
4. Тематический вопрос ("что обсуждали про X") → semantic_search
5. Сложные вопросы → используй несколько инструментов последовательно

Отвечай на русском языке. Всегда указывай таймкоды и имена спикеров в ответе.
"""


def _get_llm(model: str) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY env var")

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
    )


def build_qa_agent(index: TranscriptIndex, model: str):
    """Создать агента для Q&A по транскрипции."""
    tools = build_tools(index)
    llm = _get_llm(model)

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def ask(question: str, agent) -> str:
    """Задать вопрос агенту и получить ответ."""
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
