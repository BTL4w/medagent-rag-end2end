"""Load `.env` from project root (override path with DOTENV_PATH, e.g. for tests)."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def load_app_env() -> None:
    path = (os.getenv("DOTENV_PATH") or ".env").strip()
    if path:
        load_dotenv(path)
