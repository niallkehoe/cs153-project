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

# Repetition detection: if any N-gram of this size appears this many times
# in a single generation, the output is a degeneration loop — truncate it.
REPEAT_NGRAM = 4
REPEAT_MAX   = 4


@dataclass
class Turn:
    """A single turn in the scientist's conversation history."""

    role: str  # "user" | "assistant"
    content: str
    compressed: bool = False


_SPECIAL_TOKEN_RE = __import__('re').compile(r'<\|[^|]+\|>')


def _strip_special_tokens(text: str) -> str:
    """Remove model chat special tokens (e.g. <|user_start|>) from generated text."""
    return _SPECIAL_TOKEN_RE.sub('', text).strip()


def _truncate_repetition(text: str, ngram: int = REPEAT_NGRAM, max_reps: int = REPEAT_MAX) -> str:
    """
    Return ``text`` truncated just before a repeating N-gram loop begins.

    Scans the text for the first N-gram that appears ``max_reps`` times.
    When found, returns everything up to (but not including) the point where
    the ``max_reps``-th repetition starts. If no loop is detected, returns
    the original text unchanged.
    """
    words = text.split()
    # #region agent log H1 - log early exit condition
    import json as _json, time as _time
    _dbg = {"sessionId":"4c613c","hypothesisId":"H1","location":"scientist.py:_truncate_repetition","message":"entry","data":{"word_count":len(words),"min_needed":ngram*max_reps,"will_early_exit":len(words)<ngram*max_reps},"timestamp":int(_time.time()*1000)}
    with open("/Users/niall/Dev/cs153-project/.cursor/debug-4c613c.log","a") as _f: _f.write(_json.dumps(_dbg)+"\n")
    # #endregion
    if len(words) < ngram * max_reps:
        return text

    seen: dict[tuple, list[int]] = {}
    for i in range(len(words) - ngram + 1):
        gram = tuple(words[i : i + ngram])
        positions = seen.setdefault(gram, [])
        positions.append(i)
        if len(positions) >= max_reps:
            # Truncate at the start of the (max_reps)th occurrence
            cut = positions[max_reps - 1]
            return " ".join(words[:cut]).strip()

    return text


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
    temperature: float = 0.6
    top_k: int = 20
    max_new_tokens: int = 256
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

        # #region agent log H6 - confirm new chat template format is in prompt
        import json as _json, time as _time
        _dbg = {"sessionId":"4c613c","hypothesisId":"H6","location":"scientist.py:generate","message":"pre-generate","data":{"prompt_chars":len(prompt),"has_user_start":"<|user_start|>" in prompt,"has_assistant_start":"<|assistant_start|>" in prompt,"prompt_tail":prompt[-300:]},"timestamp":int(_time.time()*1000)}
        with open("/Users/niall/Dev/cs153-project/.cursor/debug-4c613c.log","a") as _f: _f.write(_json.dumps(_dbg)+"\n")
        # #endregion

        response = httpx.post(
            f"{self.api_url}/generate",
            json={
                "prompt": prompt,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "max_new_tokens": self.max_new_tokens,
                "stop_sequences": ["\nEND", " END"],
            },
            timeout=120,
        )
        response.raise_for_status()
        text: str = response.json()["text"]
        text = _strip_special_tokens(text)
        text = _truncate_repetition(text)

        # #region agent log H7 - log cleaned response stored in history
        import json as _json, time as _time
        _dbg2 = {"sessionId":"4c613c","hypothesisId":"H7","location":"scientist.py:generate","message":"post-generate-cleaned","data":{"cleaned_text":text[:400],"word_count":len(text.split())},"timestamp":int(_time.time()*1000)}
        with open("/Users/niall/Dev/cs153-project/.cursor/debug-4c613c.log","a") as _f: _f.write(_json.dumps(_dbg2)+"\n")
        # #endregion

        self._history.append(Turn(role="assistant", content=text))
        return text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        """
        Build a prompt string using GPT-1900-instruct's chat special tokens.

        The model was fine-tuned with a per-round chat template; using plain
        "USER:" / "ASSISTANT:" markers causes it to ignore all instructions.

        Observed format (from model-generated continuations):
          <|bos|>                           ← prepended by deploy/server.py
          <|user_start|>TURN<|user_end|>
          <|assistant_start|>RESPONSE<|assistant_end|>
          <|bos|>                           ← between rounds
          <|user_start|>NEXT<|user_end|>
          <|assistant_start|>               ← generation primed here

        The BOS token is injected by the server before encoding, so we start
        the prompt string with <|user_start|>.
        """
        if not self._history:
            return ""

        # First history entry is always the question; embed it in the system prompt.
        question = self._history[0].content
        rendered = self._system_prompt.format(question=question)

        prompt = f"<|user_start|>{rendered}<|user_end|><|assistant_start|>"

        for turn in self._history[1:]:
            if turn.role == "assistant":
                prompt += f"{turn.content}<|assistant_end|>"
            else:
                # User turn (tool result or reprompt): start a new round with <|bos|>
                prompt += f"<|bos|><|user_start|>{turn.content}<|user_end|><|assistant_start|>"

        return prompt

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
