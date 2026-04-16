"""LLM backend adapters for explanation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_MODEL_LABEL = "Llama3.1-8b"
DEFAULT_HTTP_USER_AGENT = "xai-playground-openai-compatible/0.1"


class BackendError(RuntimeError):
    """Raised when a backend cannot complete a generation request."""


@dataclass(frozen=True)
class GenerationParams:
    model: str = DEFAULT_GROQ_MODEL
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 600
    timeout_seconds: int = 120

    def request_body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        reasoning_model = uses_openai_reasoning_model(self.model)
        body = {
            "model": self.model,
            "messages": normalize_messages_for_model(self.model, messages),
        }
        if reasoning_model:
            body["max_completion_tokens"] = self.max_tokens
        else:
            body["temperature"] = self.temperature
            body["top_p"] = self.top_p
            body["max_tokens"] = self.max_tokens
        return body


@dataclass(frozen=True)
class GenerationResult:
    text: str
    elapsed_seconds: float
    response_metadata: dict[str, Any] = field(default_factory=dict)


class ChatBackend(Protocol):
    name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
    ) -> GenerationResult:
        """Generate text from chat messages."""


@dataclass
class OpenAICompatibleChatBackend:
    """Minimal OpenAI-compatible chat-completions backend.

    This covers Groq, OpenAI-compatible hosted APIs, and local servers that expose
    `/chat/completions` with the OpenAI request/response shape.
    """

    name: str
    base_url: str
    api_key_env: str | None = None
    require_api_key: bool = True

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def generate(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
    ) -> GenerationResult:
        api_key = os.environ.get(self.api_key_env or "") if self.api_key_env else None
        if self.require_api_key and not api_key:
            raise BackendError(
                f"{self.name} requires an API key in {self.api_key_env}."
            )

        payload = json.dumps(params.request_body(messages)).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": DEFAULT_HTTP_USER_AGENT,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = Request(
            self.chat_completions_url,
            data=payload,
            headers=headers,
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urlopen(request, timeout=params.timeout_seconds) as response:
                response_body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            error_headers = _interesting_error_headers(exc)
            header_suffix = f" headers={json.dumps(error_headers)}" if error_headers else ""
            raise BackendError(
                f"{self.name} returned HTTP {exc.code}{header_suffix}: {error_body}"
            ) from exc
        except URLError as exc:
            raise BackendError(f"{self.name} request failed: {exc.reason}") from exc
        elapsed = time.perf_counter() - start

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise BackendError(
                f"{self.name} returned non-JSON response: {response_body[:500]}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise BackendError(f"{self.name} response did not include choices.")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        text = message.get("content") or first_choice.get("text") or ""

        metadata = {
            "id": data.get("id"),
            "created": data.get("created"),
            "model": data.get("model"),
            "usage": data.get("usage"),
            "finish_reason": first_choice.get("finish_reason"),
            "choice_index": first_choice.get("index"),
        }

        return GenerationResult(
            text=text,
            elapsed_seconds=elapsed,
            response_metadata=metadata,
        )


def uses_openai_reasoning_model(model: str) -> bool:
    return bool(re.match(r"^o\d", model))


def normalize_messages_for_model(
    model: str,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not uses_openai_reasoning_model(model):
        return messages

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            normalized.append({**message, "role": "developer"})
        else:
            normalized.append(message)
    return normalized


def _interesting_error_headers(exc: HTTPError) -> dict[str, str]:
    if exc.headers is None:
        return {}
    interesting = {}
    for key in ("cf-ray", "server", "content-type", "content-length"):
        value = exc.headers.get(key)
        if value is not None:
            interesting[key] = value
    return interesting


def build_backend(
    backend_name: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
    allow_missing_api_key: bool = False,
) -> OpenAICompatibleChatBackend:
    if backend_name == "groq":
        return OpenAICompatibleChatBackend(
            name="groq",
            base_url=base_url or DEFAULT_GROQ_BASE_URL,
            api_key_env=api_key_env or "GROQ_API_KEY",
            require_api_key=True,
        )
    if backend_name == "openai-compatible":
        return OpenAICompatibleChatBackend(
            name="openai-compatible",
            base_url=base_url or "https://api.openai.com/v1",
            api_key_env=api_key_env or "OPENAI_API_KEY",
            require_api_key=not allow_missing_api_key,
        )
    raise ValueError(f"Unknown backend: {backend_name}")
