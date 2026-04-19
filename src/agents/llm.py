from __future__ import annotations

import os
from typing import Any, List, Optional

from dotenv import load_dotenv

from langchain_core import messages
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


class OptionalLLM:
    """
    Small wrapper around `langchain_openai.ChatOpenAI`.

    Returns `None` if no API key is configured so callers can fallback safely.

    This intentionally uses *native* LangChain message objects to make it easy
    to later add structured output, tool calling, streaming, retry, callbacks,
    and tracing.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: Optional[int] = None,
    ) -> None:
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()
        env_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        effective_base_url = base_url or env_base_url
        self.base_url = effective_base_url.rstrip("/") if effective_base_url else None
        self.timeout = timeout
        env_max_retries = os.getenv("OPENAI_MAX_RETRIES")
        self.max_retries = max_retries if max_retries is not None else (int(env_max_retries) if env_max_retries else None)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def get_chat_model(self, *, temperature: float = 0.1) -> Optional[ChatOpenAI]:
        """Return a ready-to-use ChatOpenAI client (or None if disabled)."""
        if not self.enabled:
            return None

        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            temperature=temperature,
        )

    def chat_messages(
        self,
        chat_messages: List[BaseMessage],
        *,
        temperature: float = 0.1,
        **invoke_kwargs: Any,
    ) -> Optional[str]:
        """
        Chat using native LangChain BaseMessage objects.

        `invoke_kwargs` is forwarded to `ChatOpenAI.invoke(...)`.
        """
        if not self.enabled:
            return None

        llm = self.get_chat_model(temperature=temperature)
        if llm is None:
            return None

        resp = llm.invoke(chat_messages, **invoke_kwargs)
        content = getattr(resp, "content", None)
        return content.strip() if isinstance(content, str) else None

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Optional[str]:
        """Backward-compatible helper using system + user prompt strings."""
        system_msg = messages.SystemMessage(content=system_prompt)
        user_msg = messages.HumanMessage(content=user_prompt)
        return self.chat_messages([system_msg, user_msg], temperature=temperature)
