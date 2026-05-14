"""
ScientistAgent
--------------
Wraps the GPT-1900 model hosted on a DigitalOcean GPU droplet.

The scientist is the core agent in the re-discovery pipeline. It receives an
opening research question and a description of available tools, then drives
the experiment loop by generating either:

  - propose_experiment(description, parameters) — requests data from the simulator
  - conclude(hypothesis)                         — ends the loop with a final claim

Context management
~~~~~~~~~~~~~~~~~~
GPT-1900 has a 2048-token context window. To stay within this limit across
multiple experiment turns, older experiment result entries are compressed to a
single summary line once the running token count exceeds COMPRESS_THRESHOLD.
The compression itself is done locally (no LLM call) to avoid introducing
post-1900 framing from a modern model.

See CONCERNS in README.md:
  #2  Scientist Contamination
  #3  Context Window Pressure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

COMPRESS_THRESHOLD = 1500  # tokens; compress history below this headroom
COMPRESS_KEEP_TAIL = 2     # always keep the last N turns uncompressed
COMPRESS_MAX_CHARS = 200   # truncate compressed user turns to this many chars


@dataclass
class Turn:
    """A single turn in the scientist's conversation history."""

    role: str  # "user" | "assistant"
    content: str
    compressed: bool = False


@dataclass
class ScientistAgent:
    """
    Client for the GPT-1900 HTTP generation endpoint.

    Parameters
    ----------
    api_url:
        Base URL of the DigitalOcean-hosted nanochat server,
        e.g. ``http://<droplet-ip>:8000``.
    temperature:
        Sampling temperature. Hla's physics eval uses 0.7.
    top_k:
        Top-k sampling. Hla's physics eval uses 50.
    max_new_tokens:
        Maximum tokens to generate per turn.
    """

    api_url: str
    temperature: float = 0.7
    top_k: int = 50
    max_new_tokens: int = 512
    prompt_path: str = "prompts/scientist.txt"
    _history: list[Turn] = field(default_factory=list, init=False, repr=False)
    _system_prompt: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        with open(self.prompt_path) as f:
            self._system_prompt = f.read()

    def reset(self) -> None:
        """Clear conversation history for a new eval entry."""
        self._history = []

    def add_user_message(self, content: str) -> None:
        """Append a user-role message (question or tool result) to history."""
        self._history.append(Turn(role="user", content=content))

    def generate(self) -> str:
        """
        Run a forward pass and return the assistant's raw text response.

        Automatically compresses old experiment results if the history is
        approaching the context window limit.

        Returns
        -------
        str
            Raw generation from the model. The caller (ToolRouter) is responsible
            for parsing tool calls from this string.

        Raises
        ------
        httpx.HTTPStatusError
            If the generation endpoint returns a non-2xx response.
        """
        self._maybe_compress_history()
        prompt = self._build_prompt()
        response = httpx.post(
            f"{self.api_url}/generate",
            json={
                "prompt": prompt,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "max_new_tokens": self.max_new_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        text: str = response.json()["text"]
        self._history.append(Turn(role="assistant", content=text))
        return text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        """
        Concatenate history into a single prompt string for the causal LM.

        The first user message (the research question) is embedded into the
        scientist system prompt template via the {question} placeholder.
        Subsequent turns are formatted as alternating USER / ASSISTANT blocks.
        The prompt always ends with a bare "ASSISTANT:" to prime generation.
        """
        if not self._history:
            return ""

        # Render the system prompt with the opening question
        question = self._history[0].content
        rendered = self._system_prompt.format(question=question)

        parts = [rendered]
        for turn in self._history[1:]:
            prefix = "USER" if turn.role == "user" else "ASSISTANT"
            parts.append(f"\n\n{prefix}: {turn.content}")

        parts.append("\n\nASSISTANT:")
        return "".join(parts)

    def _maybe_compress_history(self) -> None:
        """
        Compress old turns to truncated summaries if context is too long.

        Skips the first turn (the question, embedded in the system prompt)
        and the last COMPRESS_KEEP_TAIL turns. Only compresses user turns
        (experiment reports) since they are typically the longest. Compression
        is character-based to avoid any LLM call.
        """
        estimated = self._estimate_tokens("".join(t.content for t in self._history))
        if estimated <= COMPRESS_THRESHOLD:
            return

        protected_tail = len(self._history) - COMPRESS_KEEP_TAIL
        for i, turn in enumerate(self._history):
            if i == 0 or i >= protected_tail or turn.compressed:
                continue
            if turn.role == "user" and len(turn.content) > COMPRESS_MAX_CHARS:
                turn.content = turn.content[:COMPRESS_MAX_CHARS].rstrip() + " [... truncated]"
                turn.compressed = True

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: 1 token ≈ 4 characters."""
        return len(text) // 4
