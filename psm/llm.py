"""
LLM wrapper around Ollama.
Exposes a simple generate() interface used by the LangGraph nodes.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterator

import ollama

from config import (
    MODEL_MAX_TOKENS,
    MODEL_N_CTX,
    MODEL_OLLAMA_NAME,
    MODEL_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Thin wrapper around the Ollama chat API.

    Parameters
    ----------
    model : str
        Ollama model name (e.g. "psm-qwen35-2b").
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum new tokens per generation.
    """

    def __init__(
        self,
        model: str = MODEL_OLLAMA_NAME,
        temperature: float = MODEL_TEMPERATURE,
        max_tokens: int = MODEL_MAX_TOKENS,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = ollama.Client()
        self._available = self._check()

    def _check(self) -> bool:
        try:
            names = [m.model for m in self._client.list().models]
            if self.model not in names:
                logger.error("Ollama model '%s' not found. Available: %s", self.model, names)
                return False
            logger.info("Ollama model '%s' ready", self.model)
            return True
        except Exception as exc:
            logger.error("Cannot reach Ollama server: %s", exc)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._available

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        if not self._available:
            raise RuntimeError("Ollama LLM not available. Is Ollama running?")

        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        options = {
            "temperature": temperature or self.temperature,
            "num_predict": max_tokens or self.max_tokens,
        }

        if stream:
            def _iter() -> Iterator[str]:
                for chunk in self._client.chat(
                    model=self.model,
                    messages=full_messages,
                    options=options,
                    stream=True,
                ):
                    yield chunk.message.content
            return _iter()
        else:
            response = self._client.chat(
                model=self.model,
                messages=full_messages,
                options=options,
                stream=False,
            )
            return response.message.content

    def count_tokens(self, text: str) -> int:
        return len(text.split()) * 4 // 3


@lru_cache(maxsize=1)
def get_llm() -> OllamaLLM:
    """Singleton accessor for the LLM instance."""
    return OllamaLLM()
