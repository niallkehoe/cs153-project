"""
tools/
------
Tool schema definitions and call router for the experiment loop.

- schema  : Pydantic models and JSON schemas for propose_experiment and conclude
- router  : Dispatches tool calls from the scientist to the simulator or terminates the loop
"""

from tools.schema import ProposeExperimentArgs, ConcludeArgs, TOOL_SCHEMAS
from tools.router import ToolRouter

__all__ = ["ProposeExperimentArgs", "ConcludeArgs", "TOOL_SCHEMAS", "ToolRouter"]
