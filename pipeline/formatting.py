"""Markdown → Telegram HTML converter."""

from __future__ import annotations

import re
import html


def md_to_tg_html(text: str) -> str:
    """Convert common Markdown to Telegram-supported HTML.

    Supported: **bold**, *italic*, `code`, ```code blocks```,
    headers (##), bullet lists (- / *).
    """
    # Escape HTML entities first (but preserve any existing tags we'll add)
    text = html.escape(text)

    # Code blocks: ```...``` → <pre>...</pre>
    text = re.sub(
        r"```(?:\w*)\n?(.*?)```",
        lambda m: f"<pre>{m.group(1).strip()}</pre>",
        text,
        flags=re.DOTALL,
    )

    # Inline code: `...` → <code>...</code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # Italic: *text* or _text_ (but not inside words with underscores)
    text = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", text)

    # Headers: ## text → bold text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Bullet lists: "- item" or "* item" → "  • item"
    text = re.sub(r"^[\-\*]\s+", "  • ", text, flags=re.MULTILINE)

    # Numbered lists: clean up (keep as-is, they look fine)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
