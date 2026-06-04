"""
llm_client.py
-------------
Shared HTTP client for OpenRouter.

All modern-LLM calls in the pipeline (SimulatorAgent, JudgeAgent) route
through this module. OpenRouter exposes an OpenAI-compatible API, so only
``httpx`` and one environment variable are required.

Required environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  OPEN_ROUTER_API_KEY  — your OpenRouter API key

Available models
~~~~~~~~~~~~~~~~
Pass any OpenRouter model identifier as the ``model`` argument.
Recommended defaults:

  anthropic/claude-3.5-haiku          (fast, high quality — pipeline default)
  openai/gpt-4o-mini                  (cheaper, lower latency)
  meta-llama/llama-3.3-70b-instruct   (open-source alternative)

Full model catalogue: https://openrouter.ai/models
"""

from __future__ import annotations

import os

import httpx

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-3.5-haiku"

# Shared httpx client — reused across calls to amortise connection overhead
_http_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(timeout=120.0)
    return _http_client


def call_llm(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """
    Call an OpenRouter model and return the response text.

    Parameters
    ----------
    model:
        OpenRouter model identifier, e.g. ``"anthropic/claude-3.5-haiku"``.
    prompt:
        The user-role message content.
    system:
        Optional system-role message prepended before the user message.
    max_tokens:
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The model's response text.

    Raises
    ------
    KeyError
        If ``OPEN_ROUTER_API_KEY`` is not set.
    httpx.HTTPStatusError
        If the API returns a non-2xx status.
    """
    api_key = os.environ["OPEN_ROUTER_API_KEY"]

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_client().post(
        OPENROUTER_API_BASE,
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/cs153-project",
        },
        json={"model": model, "messages": messages, "max_tokens": max_tokens},
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
