from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from auth import verify_password
from pipeline.chatbot import _get_llm, create_chatbot
from pipeline.download import VideoMetadata, extract_video_id
from pipeline.index import TranscriptIndex

# ── Constants ─────────────────────────────────────────────────────────────────

CHATS_DIR = Path("data/chats")

_TOOL_LABELS: dict[str, str] = {
    "process_video": "Скачиваю и обрабатываю видео (может занять несколько минут)...",
    "summarize_video": "Создаю саммари...",
    "semantic_search": "Семантический поиск...",
    "get_video_info": "Получаю метаданные видео...",
    "get_transcript_metadata": "Анализирую транскрипцию...",
    "get_segments_by_speaker": "Ищу реплики спикера...",
    "get_segments_by_time": "Ищу по временному диапазону...",
}


@st.cache_resource
def _get_cached_llm(model: str):
    """Single shared LLM client for all chat sessions."""
    return _get_llm(model)


def _make_args() -> argparse.Namespace:
    cpu = os.cpu_count() or 4
    return argparse.Namespace(
        output_dir=os.environ.get("OUTPUT_DIR", "output"),
        whisper_model=os.environ.get("WHISPER_MODEL", "large-v3-turbo-q5_0"),
        llm_model=os.environ.get("LLM_MODEL", "nvidia/nemotron-nano-9b-v2:free"),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        language=os.environ.get("LANGUAGE", "ru"),
        threads=int(os.environ.get("WHISPER_THREADS", "4")),
        workers=int(os.environ.get("WHISPER_WORKERS", str(max(1, cpu // 4)))),
    )


# ── Chat store ────────────────────────────────────────────────────────────────

def _user_dir(username: str) -> Path:
    p = CHATS_DIR / username
    p.mkdir(parents=True, exist_ok=True)
    return p


@st.cache_data
def list_chats(username: str) -> list[dict]:
    """Return chats sorted newest-first: [{chat_id, title, video_url}]."""
    d = CHATS_DIR / username
    if not d.exists():
        return []
    result = []
    for p in sorted(d.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            result.append({"chat_id": p.stem, "title": data.get("title", p.stem), "video_url": data.get("video_url")})
        except Exception:
            pass
    return result


@st.cache_data
def load_chat(username: str, chat_id: str) -> dict | None:
    p = _user_dir(username) / f"{chat_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_chat(username: str, chat_id: str, data: dict) -> None:
    (_user_dir(username) / f"{chat_id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    load_chat.clear()
    list_chats.clear()


def delete_chat(username: str, chat_id: str) -> None:
    p = _user_dir(username) / f"{chat_id}.json"
    if p.exists():
        p.unlink()
    list_chats.clear()


# ── Agent helpers ─────────────────────────────────────────────────────────────

def _restore_lc_history(stored_messages: list[dict]) -> list:
    """Reconstruct LangChain history from persisted user/assistant messages."""
    history = []
    for m in stored_messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))
    return history


def _try_restore_index(state, video_url: str | None, args: argparse.Namespace) -> None:
    """Load saved TranscriptIndex and metadata from disk into state if available."""
    if not video_url or state.index is not None:
        return
    try:
        video_id = extract_video_id(video_url)
        video_dir = Path(args.output_dir) / video_id
        state.index = TranscriptIndex.load(video_dir / "index", args.embedding_model)
        state.processed_url = video_url
        meta_file = video_dir / "metadata.json"
        if meta_file.exists():
            state.metadata = VideoMetadata(**json.loads(meta_file.read_text(encoding="utf-8")))
    except Exception:
        pass  # index not on disk yet — user will need to re-send the URL


def _get_or_create_session(chat_id: str, stored_messages: list[dict], video_url: str | None) -> dict:
    """Lazily create agent session, restoring history and index from disk."""
    sessions: dict = st.session_state.setdefault("sessions", {})
    if chat_id not in sessions:
        args = _make_args()
        llm = _get_cached_llm(args.llm_model)
        agent, state = create_chatbot(args, llm=llm)
        _try_restore_index(state, video_url, args)
        sessions[chat_id] = {
            "agent": agent,
            "state": state,
            "lc_history": _restore_lc_history(stored_messages),
        }
    return sessions[chat_id]


def _stream_agent(session: dict) -> tuple[str, list]:
    """
    Run agent.stream(), show tool status, collect new messages.
    Returns (final_answer, new_messages).
    """
    agent = session["agent"]
    lc_history = session["lc_history"]

    new_messages: list = []
    final_answer = ""
    active_tool: str | None = None
    status_placeholder = st.empty()

    for step in agent.stream({"messages": lc_history}):
        # Iterate over node outputs without assuming specific node names
        for node_output in step.values():
            if not isinstance(node_output, dict):
                continue
            for msg in node_output.get("messages", []):
                new_messages.append(msg)
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    tool_name = tool_calls[0]["name"]
                    active_tool = tool_name
                    label = _TOOL_LABELS.get(tool_name, f"{tool_name}...")
                    status_placeholder.info(f"⚙️ {label}")
                elif getattr(msg, "content", ""):
                    # Only treat as final answer if it's not a tool call message
                    if not tool_calls:
                        final_answer = msg.content
                        status_placeholder.empty()

    status_placeholder.empty()
    return final_answer or "(нет ответа)", new_messages


# ── Pages ─────────────────────────────────────────────────────────────────────

def _login_page() -> None:
    st.title("Видео-ассистент")
    with st.form("login_form"):
        username = st.text_input("Имя пользователя")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти", use_container_width=True)

    if submitted:
        user = verify_password(username, password)
        if user:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Неверное имя пользователя или пароль")


def _sidebar(username: str, display_name: str) -> None:
    with st.sidebar:
        st.markdown(f"**{display_name}**")
        if st.button("Выйти", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()

        if st.button("＋ Новый чат", use_container_width=True, type="primary"):
            chat_id = uuid.uuid4().hex[:12]
            save_chat(username, chat_id, {"title": "Новый чат", "video_url": None, "messages": []})
            st.session_state["active_chat_id"] = chat_id
            st.rerun()

        st.markdown("**Чаты**")
        for chat in list_chats(username):
            cid = chat["chat_id"]
            title = chat["title"]
            if len(title) > 30:
                title = title[:27] + "..."
            is_active = cid == st.session_state.get("active_chat_id")

            col_t, col_d = st.columns([5, 1])
            with col_t:
                if st.button(
                    title,
                    key=f"chat_{cid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["active_chat_id"] = cid
                    st.rerun()
            with col_d:
                if st.button("✕", key=f"del_{cid}", use_container_width=True):
                    delete_chat(username, cid)
                    st.session_state.get("sessions", {}).pop(cid, None)
                    if st.session_state.get("active_chat_id") == cid:
                        st.session_state.pop("active_chat_id", None)
                    st.rerun()


def _chat_page(username: str) -> None:
    chat_id: str | None = st.session_state.get("active_chat_id")

    if not chat_id:
        st.markdown("### Выберите чат или создайте новый")
        return

    chat_data = load_chat(username, chat_id)
    if chat_data is None:
        st.warning("Чат не найден.")
        st.session_state.pop("active_chat_id", None)
        st.rerun()
        return

    stored_messages: list[dict] = chat_data.get("messages", [])

    # Render history
    for msg in stored_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Задайте вопрос или вставьте ссылку на YouTube...")
    if not user_input:
        return

    session = _get_or_create_session(chat_id, stored_messages, chat_data.get("video_url"))

    with st.chat_message("user"):
        st.markdown(user_input)

    session["lc_history"].append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        final_answer, new_messages = _stream_agent(session)
        st.markdown(final_answer)

    session["lc_history"].extend(new_messages)

    stored_messages.append({"role": "user", "content": user_input})
    stored_messages.append({"role": "assistant", "content": final_answer})

    # Auto-title from first message
    title = chat_data.get("title", "Новый чат")
    if title == "Новый чат":
        title = user_input[:50] + ("..." if len(user_input) > 50 else "")

    save_chat(username, chat_id, {
        "title": title,
        "video_url": session["state"].processed_url or chat_data.get("video_url"),
        "messages": stored_messages,
    })

    st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Видео-ассистент", page_icon="🎬", layout="wide")

    user = st.session_state.get("user")
    if not user:
        _login_page()
        return

    _sidebar(user["username"], user["display_name"])
    _chat_page(user["username"])


main()
