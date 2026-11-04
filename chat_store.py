"""SQLite-backed storage and lightweight cross-chat retrieval."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("CHAT_DB_PATH", "data/chats.sqlite3")).expanduser()
_WORDS = re.compile(r"[a-zа-яё0-9]{3,}", re.IGNORECASE)
_STOP_WORDS = {
    "что", "как", "это", "для", "или", "его", "она", "они", "мне", "меня",
    "вот", "там", "тут", "есть", "быть", "был", "будет", "чтобы", "этот",
    "эта", "эти", "про", "при", "когда", "если", "the", "and", "for", "with",
    "that", "this", "from", "you", "your", "are", "was", "were", "как-то",
}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            messages TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL
        )"""
    )
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    cleaned = []
    for message in messages or []:
        if (
            isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ):
            cleaned.append({"role": message["role"], "content": message["content"]})
    return cleaned


def _title_for(messages: list[dict[str, str]], fallback: str) -> str:
    first = next((item["content"].strip() for item in messages if item["role"] == "user"), "")
    if not first:
        return fallback
    first_line = " ".join(first.split())
    return first_line[:57].rstrip() + ("…" if len(first_line) > 57 else "")


def _summary_for(messages: list[dict[str, str]]) -> str:
    prompts = [" ".join(m["content"].split()) for m in messages if m["role"] == "user"]
    if not prompts:
        return ""
    excerpts = prompts[:1] + prompts[-3:]
    return " | ".join(dict.fromkeys(item[:240] for item in excerpts))[:900]


def create_chat() -> str:
    chat_id = uuid.uuid4().hex
    with _connect() as db:
        db.execute(
            "INSERT INTO chats(id, title, summary, messages, updated_at) VALUES (?, ?, '', '[]', ?)",
            (chat_id, "Новый чат", _now()),
        )
    return chat_id


def latest_or_create_chat() -> str:
    with _connect() as db:
        row = db.execute("SELECT id FROM chats ORDER BY updated_at DESC LIMIT 1").fetchone()
        if row:
            return str(row["id"])
    return create_chat()


def get_chat(chat_id: str) -> list[dict[str, str]]:
    with _connect() as db:
        row = db.execute("SELECT messages FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not row:
        return []
    try:
        return _clean_messages(json.loads(row["messages"]))
    except (TypeError, json.JSONDecodeError):
        return []


def save_chat(chat_id: str, messages: list[dict[str, Any]] | None) -> None:
    cleaned = _clean_messages(messages)
    with _connect() as db:
        current = db.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
        fallback = str(current["title"]) if current else "Новый чат"
        db.execute(
            """INSERT INTO chats(id, title, summary, messages, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                   summary=excluded.summary, messages=excluded.messages,
                   updated_at=excluded.updated_at""",
            (chat_id, _title_for(cleaned, fallback), _summary_for(cleaned),
             json.dumps(cleaned, ensure_ascii=False), _now()),
        )


def list_chats(query: str = "") -> list[tuple[str, str]]:
    with _connect() as db:
        rows = db.execute("SELECT id, title, summary FROM chats ORDER BY updated_at DESC").fetchall()
    needle = query.strip().casefold()
    choices = []
    for row in rows:
        if needle and needle not in f"{row['title']} {row['summary']}".casefold():
            continue
        choices.append((str(row["title"]), str(row["id"])))
    return choices


def delete_chat(chat_id: str) -> None:
    with _connect() as db:
        db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def related_context(chat_id: str, prompt: str, limit: int = 2) -> str:
    query_terms = {word.casefold() for word in _WORDS.findall(prompt)} - _STOP_WORDS
    if not query_terms:
        return ""
    with _connect() as db:
        rows = db.execute(
            "SELECT id, title, summary, messages FROM chats WHERE id != ? ORDER BY updated_at DESC",
            (chat_id,),
        ).fetchall()

    ranked: list[tuple[int, str, str]] = []
    for row in rows:
        title = str(row["title"])
        summary = str(row["summary"])
        try:
            messages = _clean_messages(json.loads(row["messages"]))
        except (TypeError, json.JSONDecodeError):
            messages = []
        searchable = f"{title} {summary} " + " ".join(m["content"] for m in messages)
        terms = {word.casefold() for word in _WORDS.findall(searchable)} - _STOP_WORDS
        overlap = query_terms & terms
        if overlap:
            ranked.append((len(overlap), title, summary or " ".join(m["content"] for m in messages[:4])))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return "\n".join(
        f"- Чат «{title}»: {excerpt[:500]}" for _, title, excerpt in ranked[:limit]
    )
