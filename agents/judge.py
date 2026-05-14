"""
JudgeAgent
----------
Wraps a modern LLM to score the scientist's final hypothesis.

The judge is invoked once per eval entry, after the scientist calls
``conclude(hypothesis)`` or the maximum turn count is reached. It does
NOT participate in the experiment loop — it only sees:

  - The opening research question
  - The scientist's final hypothesis text
  - The ground truth from the eval set

Scoring rubric (0–4 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each dimension is scored 0 or 1:

  1. Correct phenomenon identified
       Did the scientist identify *what* is happening, even without naming it?
       (e.g. "bodies of different weight fall at the same rate" ✓)

  2. Correct qualitative mechanism
       Did the scientist give a causally correct explanation?
       (e.g. "the Earth exerts an equal pull per unit of matter" ✓)

  3. Correct quantitative form (where applicable)
       Did the scientist arrive at a correct functional relationship?
       (e.g. "distance grows as the square of elapsed time" ✓)

  4. Appropriate confidence hedging
       Did the scientist express calibrated uncertainty?
       Neither overconfident ("this is certainly true for all bodies in nature")
       nor vacuous ("it is impossible to say").

See CONCERNS in README.md:
  #4  Binary Judge Scoring Is Insufficient
  #5  Judge Reward Hacking
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agents.llm_client import call_llm


@dataclass
class JudgeScore:
    """Structured output from the judge for a single eval entry."""

    phenomenon_identified: bool
    qualitative_mechanism: bool
    quantitative_form: bool
    confidence_calibrated: bool
    reasoning: str
    """Free-text explanation from the judge for each dimension."""

    @property
    def total(self) -> int:
        """Total score out of 4."""
        return sum([
            self.phenomenon_identified,
            self.qualitative_mechanism,
            self.quantitative_form,
            self.confidence_calibrated,
        ])


@dataclass
class JudgeAgent:
    """
    Modern LLM wrapper that evaluates the scientist's final hypothesis.

    Parameters
    ----------
    model:
        LLM identifier string, e.g. ``"claude-sonnet-4-5"`` or ``"gpt-4o"``.
    prompt_path:
        Path to the judge system prompt template.
    """

    model: str
    prompt_path: str = "prompts/judge.txt"
    _system_prompt: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        with open(self.prompt_path) as f:
            self._system_prompt = f.read()

    def score(
        self,
        question: str,
        hypothesis: str,
        ground_truth: str,
        key_concepts: list[str],
        num_experiments: int,
        max_turns: int,
    ) -> JudgeScore:
        """
        Evaluate the scientist's hypothesis against the ground truth.

        Parameters
        ----------
        question:
            The original research question posed to the scientist.
        hypothesis:
            The scientist's final conclusion text from ``conclude()``.
        ground_truth:
            Accepted scientific explanation from the eval set.
        key_concepts:
            List of concepts that must appear (even if not by modern name)
            for full credit, e.g. ``["constant acceleration", "mass-independent"]``.
        num_experiments:
            Number of experiments the scientist requested.
        max_turns:
            Maximum turns allowed for this eval entry. Used to penalise
            conclusions reached only after excessive experimentation.

        Returns
        -------
        JudgeScore
            Structured partial-credit score with reasoning.
        """
        prompt = self._build_scoring_prompt(
            question=question,
            hypothesis=hypothesis,
            ground_truth=ground_truth,
            key_concepts=key_concepts,
            num_experiments=num_experiments,
            max_turns=max_turns,
        )
        raw = call_llm(self.model, prompt, max_tokens=1024)
        return self._parse_score(raw)

    def _build_scoring_prompt(
        self,
        question: str,
        hypothesis: str,
        ground_truth: str,
        key_concepts: list[str],
        num_experiments: int = 0,
        max_turns: int = 10,
    ) -> str:
        """Render the judge prompt template with the eval-specific values."""
        efficiency_note = (
            f"\nNote: the scientist used {num_experiments} of {max_turns} allowed experiments. "
            "Factor this into the confidence_calibrated score — a scientist who took the full "
            "allotment of turns and still expresses high confidence should be penalised."
            if num_experiments >= max_turns
            else ""
        )
        return self._system_prompt.format(
            question=question,
            hypothesis=hypothesis,
            ground_truth=ground_truth,
            key_concepts=", ".join(key_concepts),
        ) + efficiency_note

    def _parse_score(self, raw: str) -> JudgeScore:
        """
        Extract the JSON scoring object from the judge's raw response.

        Falls back gracefully if the response is not valid JSON — marks all
        dimensions as False and records the raw text as the reasoning so the
        run is not silently lost.
        """
        # Extract the first JSON object from the response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                reasoning_parts = [
                    f"phenomenon: {data.get('phenomenon_reasoning', '')}",
                    f"mechanism: {data.get('mechanism_reasoning', '')}",
                    f"quantitative: {data.get('quantitative_reasoning', '')}",
                    f"confidence: {data.get('confidence_reasoning', '')}",
                ]
                if data.get("reward_hacking_flag"):
                    reasoning_parts.append(
                        f"REWARD_HACKING: {data.get('reward_hacking_note', '')}"
                    )
                return JudgeScore(
                    phenomenon_identified=bool(data.get("phenomenon_identified", 0)),
                    qualitative_mechanism=bool(data.get("qualitative_mechanism", 0)),
                    quantitative_form=bool(data.get("quantitative_form", 0)),
                    confidence_calibrated=bool(data.get("confidence_calibrated", 0)),
                    reasoning=" | ".join(reasoning_parts),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: could not parse JSON
        return JudgeScore(
            phenomenon_identified=False,
            qualitative_mechanism=False,
            quantitative_form=False,
            confidence_calibrated=False,
            reasoning=f"JSON parse failed. Raw response: {raw[:500]}",
        )
