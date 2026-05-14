"""
llm_client.py
-------------
Shared HTTP client for Cloudflare Workers AI.

All modern-LLM calls in the pipeline (SimulatorAgent, JudgeAgent) route
through this module. Using Cloudflare Workers AI means no dependency on
vendor-specific SDKs — only ``httpx`` (already required for the scientist
client) and two environment variables.

Required environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  CLOUDFLARE_ACCOUNT_ID  — your Cloudflare account ID
  CLOUDFLARE_API_TOKEN   — an API token with "Workers AI" read/run permissions

Available models
~~~~~~~~~~~~~~~~
Pass any Cloudflare Workers AI model identifier as the ``model`` argument.
Recommended defaults:

  @cf/meta/llama-3.3-70b-instruct-fp8-fast   (fast, high quality)
  @cf/meta/llama-3.1-8b-instruct-fast         (cheaper, lower latency)
  @cf/mistral/mistral-7b-instruct-v0.2        (alternative)

Full model catalogue: https://developers.cloudflare.com/workers-ai/models/
"""

from __future__ import annotations

import os

import httpx

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

# Shared httpx client — reused across calls to amortise connection overhead
_http_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(timeout=60.0)
    return _http_client


def call_llm(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """
    Call a Cloudflare Workers AI model and return the response text.

    Parameters
    ----------
    model:
        Cloudflare Workers AI model identifier, e.g.
        ``"@cf/meta/llama-3.3-70b-instruct-fp8-fast"``.
    prompt:
        The user-role message content.
    system:
        Optional system-role message prepended before the user message.
        Use this to pass static role instructions separately from the
        dynamic per-call content.
    max_tokens:
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The model's response text.

    Raises
    ------
    KeyError
        If ``CLOUDFLARE_ACCOUNT_ID`` or ``CLOUDFLARE_API_TOKEN`` are not set.
    httpx.HTTPStatusError
        If the API returns a non-2xx status.
    RuntimeError
        If the API returns ``success: false``.
    """
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    api_token = os.environ["CLOUDFLARE_API_TOKEN"]

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    url = f"{CLOUDFLARE_API_BASE}/{account_id}/ai/run/{model}"
    response = _get_client().post(
        url,
        headers={"Authorization": f"Bearer {api_token}"},
        json={"messages": messages, "max_tokens": max_tokens},
    )
    response.raise_for_status()

    data: dict = response.json()
    if not data.get("success"):
        errors = data.get("errors", [])
        raise RuntimeError(f"Cloudflare Workers AI error: {errors}")

    return data["result"]["response"]
