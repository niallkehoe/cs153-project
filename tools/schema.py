"""
schema.py
---------
Pydantic models and JSON schema descriptors for the two scientist-facing tools.

Tools are described to the scientist in the system prompt (prompts/scientist.txt)
and are parsed from the model's raw text output by the ToolRouter.

Tool call format expected from the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The scientist is prompted to emit tool calls in a structured plain-text format
(not JSON, to match the Victorian prose style of the model):

  TOOL: propose_experiment
  DESCRIPTION: <free text description of the experiment>
  PARAMETERS:
    <key>: <value>
    <key>: <value>
  END

  TOOL: conclude
  HYPOTHESIS: <free text conclusion>
  CONFIDENCE: <high|medium|low>
  END

This plain-text format is used instead of JSON because GPT-1900 was not
trained on structured data and is unlikely to produce valid JSON reliably.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProposeExperimentArgs(BaseModel):
    """Arguments for the propose_experiment tool call."""

    description: str = Field(
        ...,
        description="Free-text description of the proposed experiment in the scientist's own words.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key-value pairs of experimental variables. "
            "Values should be numeric where possible (e.g. mass_kg=1.0)."
        ),
    )


class ConcludeArgs(BaseModel):
    """Arguments for the conclude tool call."""

    hypothesis: str = Field(
        ...,
        description="The scientist's final conclusion about the phenomenon under investigation.",
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="The scientist's self-reported confidence in the conclusion.",
    )


# JSON schema descriptors passed to the scientist in the system prompt
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "propose_experiment",
        "description": (
            "Propose an experiment to carry out and receive observational results. "
            "Use this to gather data before forming a conclusion."
        ),
        "parameters": ProposeExperimentArgs.model_json_schema(),
    },
    {
        "name": "conclude",
        "description": (
            "Record your final conclusion about the phenomenon. "
            "Call this when you have sufficient evidence to form a hypothesis."
        ),
        "parameters": ConcludeArgs.model_json_schema(),
    },
]
