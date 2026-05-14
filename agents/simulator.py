"""
SimulatorAgent
--------------
Wraps a modern LLM (Claude or GPT-4o) to act as a physics oracle.

The simulator receives a proposed experiment from the scientist and returns
the results that a well-equipped 19th-century laboratory would observe. It
has full modern physics knowledge but is tightly constrained by the system
prompt (prompts/simulator.txt) to:

  - Return ONLY raw observational data: numbers, descriptions, instrument
    readings. Never theoretical interpretation.
  - Use ONLY vocabulary that would be available to a scientist before 1900.
    (e.g., "the bodies arrive simultaneously" not "momentum is conserved")
  - Inject Gaussian measurement noise scaled by ``noise_level``.
  - Occasionally return a plausible experiment failure (broken instrument,
    confounding variable) with probability ``failure_rate``.

See CONCERNS in README.md:
  #1  Simulator Leakage      — the primary risk for this module
  #7  Noise and Failure Mode Realism
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agents.llm_client import call_llm

DEFAULT_NOISE_LEVEL = 0.05   # 5% relative Gaussian noise on numeric results
DEFAULT_FAILURE_RATE = 0.10  # 10% chance of returning a plausible failure


@dataclass
class ExperimentResult:
    """Structured result returned by the simulator for a single experiment."""

    success: bool
    observations: str
    """
    Human-readable observational data in period-appropriate language.
    If success=False, this describes what went wrong in the laboratory.
    """
    raw_values: dict[str, float] = field(default_factory=dict)
    """
    Optional numeric measurements (before noise injection).
    Stored for result validation and analysis; not sent to the scientist.
    """
    noise_level: float = DEFAULT_NOISE_LEVEL


@dataclass
class SimulatorAgent:
    """
    Modern LLM wrapper that simulates experiment results.

    Parameters
    ----------
    model:
        LLM identifier string, e.g. ``"claude-sonnet-4-5"`` or ``"gpt-4o"``.
    noise_level:
        Relative standard deviation of Gaussian noise added to numeric results.
        Set to 0.0 for deterministic tests.
    failure_rate:
        Probability [0, 1] that the simulator returns a plausible laboratory
        failure instead of valid results. Mimics real experimental confounders.
    prompt_path:
        Path to the simulator system prompt template.
    """

    model: str
    noise_level: float = DEFAULT_NOISE_LEVEL
    failure_rate: float = DEFAULT_FAILURE_RATE
    prompt_path: str = "prompts/simulator.txt"
    _system_prompt: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        with open(self.prompt_path) as f:
            self._system_prompt = f.read()

    def run_experiment(self, description: str, parameters: dict[str, Any]) -> ExperimentResult:
        """
        Simulate the proposed experiment and return observational results.

        Parameters
        ----------
        description:
            Free-text description of the experiment as written by the scientist.
        parameters:
            Key-value pairs of experimental parameters (e.g. mass, height, temperature).

        Returns
        -------
        ExperimentResult
            Raw observational data in period-appropriate language, with noise
            injected. May be a failure result depending on ``failure_rate``.
        """
        if random.random() < self.failure_rate:
            return self._simulate_failure(description, parameters)
        return self._simulate_success(description, parameters)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_success(self, description: str, parameters: dict[str, Any]) -> ExperimentResult:
        """Call the modern LLM with the simulator prompt and return observations."""
        params_text = (
            "\n".join(f"  {k}: {v}" for k, v in parameters.items())
            if parameters
            else "  (no specific parameters given)"
        )
        prompt = self._system_prompt.format(
            description=description,
            parameters=params_text,
        )
        observations = call_llm(self.model, prompt)
        return ExperimentResult(
            success=True,
            observations=observations,
            noise_level=self.noise_level,
        )

    def _simulate_failure(self, description: str, parameters: dict[str, Any]) -> ExperimentResult:
        """
        Ask the LLM to generate a plausible 19th-century laboratory failure.

        The failure prompt instructs the model to produce a realistic obstacle
        (instrument limitation, confounding variable, etc.) rather than a
        generic error, so the scientist must reason about what went wrong.
        """
        params_text = (
            "\n".join(f"  {k}: {v}" for k, v in parameters.items())
            if parameters
            else "  (no specific parameters given)"
        )
        failure_prompt = (
            self._system_prompt.format(
                description=description,
                parameters=params_text,
            )
            + "\n\nNOTE: Something went wrong during this experiment. "
            "Describe a realistic, physically plausible laboratory difficulty "
            "that prevented a clean result — for example, an instrument "
            "malfunctioning, an uncontrolled draught disturbing the apparatus, "
            "or a confounding effect that could not be eliminated. "
            "Do NOT reveal what the correct result would have been."
        )
        observations = call_llm(self.model, failure_prompt)
        return ExperimentResult(
            success=False,
            observations=observations,
            noise_level=self.noise_level,
        )

    def _inject_noise(self, value: float) -> float:
        """Apply Gaussian noise: value * N(1, noise_level)."""
        if self.noise_level == 0.0:
            return value
        return value * random.gauss(1.0, self.noise_level)
