from __future__ import annotations

import os
from pathlib import Path

import bcrypt
import yaml

USERS_FILE = Path(os.environ.get("USERS_FILE", "data/users.yaml"))


def _load() -> dict:
    if not USERS_FILE.exists():
        return {}
    with USERS_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE.open("w", encoding="utf-8") as f:
        yaml.dump(users, f, allow_unicode=True)


def verify_password(username: str, password: str) -> dict | None:
    """Verify credentials. Returns {username, display_name} or None."""
    users = _load()
    user = users.get(username)
    if not user:
        return None
    if bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return {"username": username, "display_name": user.get("display_name", username)}
    return None


def add_user(username: str, password: str, display_name: str = "") -> None:
    """Add or overwrite a user entry."""
    if not username.strip():
        raise ValueError("Username cannot be empty")
    users = _load()
    users[username] = {
        "password_hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
        "display_name": display_name or username,
    }
    _save(users)
