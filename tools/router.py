"""
router.py
---------
Parses scientist tool calls from raw model output and dispatches them.

The ToolRouter sits between the ScientistAgent and the SimulatorAgent. On each
scientist turn it:

  1. Parses the raw generation text for a TOOL: ... END block
  2. Routes the call:
       - propose_experiment → SimulatorAgent.run_experiment()
                            → formats result as a user-turn message
                            → returns (LoopAction.CONTINUE, result_message)
       - conclude           → captures hypothesis + confidence
                            → returns (LoopAction.STOP, ConcludeArgs)
  3. If no valid tool call is found, emits a re-prompt message asking the
     scientist to use one of the available tools.
  4. Enforces max_turns: if the turn limit is exceeded with no conclude call,
     returns (LoopAction.STOP, last_generation) so the judge can still score.

See CONCERNS in README.md:
  #3  Context Window Pressure — the router is responsible for turn counting
  #8  Max Turns and Termination
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from agents.simulator import SimulatorAgent
from tools.schema import ConcludeArgs


class LoopAction(Enum):
    CONTINUE = auto()  # experiment proposed; loop should continue
    STOP = auto()      # conclude called or max turns reached; loop should end
    REPROMPT = auto()  # no valid tool call found; loop should re-prompt


@dataclass
class RouterResult:
    """Return value from a single router dispatch."""

    action: LoopAction
    payload: str | ConcludeArgs
    """
    CONTINUE  → payload is the formatted result string to send back to the scientist
    STOP      → payload is ConcludeArgs (if conclude was called) or raw generation (timeout)
    REPROMPT  → payload is a re-prompt message string
    """


class ToolRouter:
    """
    Parses and dispatches scientist tool calls.

    Parameters
    ----------
    simulator:
        The SimulatorAgent instance to forward experiment proposals to.
    max_turns:
        Maximum number of propose_experiment calls allowed before the loop
        is force-terminated. Set per eval entry in the eval set.
    """

    TOOL_BLOCK_PATTERN = re.compile(
        r"TOOL:\s*(?P<name>\w+)\s*\n(?P<body>.*?)END",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, simulator: SimulatorAgent, max_turns: int = 10) -> None:
        self.simulator = simulator
        self.max_turns = max_turns
        self._turn_count = 0

    def reset(self) -> None:
        """Reset turn counter for a new eval entry."""
        self._turn_count = 0

    def dispatch(self, raw_generation: str) -> RouterResult:
        """
        Parse the scientist's generation and dispatch the tool call.

        Parameters
        ----------
        raw_generation:
            Full text output from the ScientistAgent for a single turn.

        Returns
        -------
        RouterResult
            Action and payload for the orchestrator to act on.
        """
        if self._turn_count >= self.max_turns:
            return RouterResult(action=LoopAction.STOP, payload=raw_generation)

        match = self.TOOL_BLOCK_PATTERN.search(raw_generation)
        if match is None:
            return RouterResult(
                action=LoopAction.REPROMPT,
                payload=(
                    "Your response did not contain a valid tool call. "
                    "Please use TOOL: propose_experiment or TOOL: conclude."
                ),
            )

        tool_name = match.group("name").lower()
        body = match.group("body")

        if tool_name == "propose_experiment":
            return self._handle_propose(body)
        if tool_name == "conclude":
            return self._handle_conclude(body)

        return RouterResult(
            action=LoopAction.REPROMPT,
            payload=f"Unknown tool '{tool_name}'. Available tools: propose_experiment, conclude.",
        )

    # ------------------------------------------------------------------
    # Private dispatch helpers
    # ------------------------------------------------------------------

    def _handle_propose(self, body: str) -> RouterResult:
        """Parse propose_experiment args, call the simulator, format the result."""
        self._turn_count += 1
        parsed = self._parse_body(body)
        description = parsed.get("description", body.strip())
        parameters = parsed.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}

        result = self.simulator.run_experiment(description, parameters)
        message = self._format_experiment_result(result)
        return RouterResult(action=LoopAction.CONTINUE, payload=message)

    def _handle_conclude(self, body: str) -> RouterResult:
        """Parse conclude args and return a STOP action."""
        from tools.schema import ConfidenceLevel
        parsed = self._parse_body(body)
        hypothesis = parsed.get("hypothesis", body.strip())
        confidence_str = parsed.get("confidence", "medium").strip().lower()
        try:
            confidence = ConfidenceLevel(confidence_str)
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM
        args = ConcludeArgs(hypothesis=hypothesis, confidence=confidence)
        return RouterResult(action=LoopAction.STOP, payload=args)

    def _parse_body(self, body: str) -> dict[str, Any]:
        """
        Parse a plain-text TOOL body into a key-value dict.

        Handles both flat keys (KEY: value) and nested PARAMETERS blocks.
        A PARAMETERS block is detected by a line matching ``PARAMETERS:``
        followed by indented ``key: value`` lines.
        """
        result: dict[str, Any] = {}
        params: dict[str, str] = {}
        in_params = False
        current_key: str | None = None
        current_value_lines: list[str] = []

        for line in body.splitlines():
            # Entering PARAMETERS block
            if re.match(r"^\s*PARAMETERS\s*:\s*$", line, re.IGNORECASE):
                if current_key:
                    result[current_key] = " ".join(current_value_lines).strip()
                    current_key = None
                    current_value_lines = []
                in_params = True
                continue

            if in_params:
                # Indented key: value pair
                param_match = re.match(r"^\s+([A-Za-z][\w\s]*):\s*(.*)$", line)
                if param_match:
                    key = param_match.group(1).strip().lower().replace(" ", "_")
                    params[key] = param_match.group(2).strip()
                elif re.match(r"^\S", line):
                    # Un-indented line ends the PARAMETERS block
                    in_params = False
                continue

            # Top-level KEY: value line
            kv_match = re.match(r"^([A-Z][A-Z_]*):\s*(.*)$", line, re.IGNORECASE)
            if kv_match:
                if current_key:
                    result[current_key] = " ".join(current_value_lines).strip()
                current_key = kv_match.group(1).lower()
                current_value_lines = [kv_match.group(2)]
            elif current_key:
                current_value_lines.append(line.strip())

        if current_key:
            result[current_key] = " ".join(current_value_lines).strip()
        if params:
            result["parameters"] = params

        return result

    def _format_experiment_result(self, result: Any) -> str:
        """
        Convert an ExperimentResult into a user-turn message for the scientist.

        The message is written in period-appropriate language and must not
        contain any post-1900 theoretical interpretation.
        """
        label = f"Report of Experiment {self._turn_count}"
        if result.success:
            return f"{label}:\n\n{result.observations}"
        return (
            f"{label} — Difficulty encountered in the laboratory:\n\n"
            f"{result.observations}\n\n"
            "You may wish to modify the experimental conditions and try again."
        )
