"""
agents/
-------
LLM agent wrappers for the three roles in the re-discovery pipeline.

- ScientistAgent  : wraps GPT-1900, generates tool calls and tracks conversation context
- SimulatorAgent  : wraps a modern LLM, simulates experiment results with noise
- JudgeAgent      : wraps a modern LLM, scores final hypotheses against ground truth
"""

from agents.scientist import ScientistAgent
from agents.simulator import SimulatorAgent
from agents.judge import JudgeAgent

__all__ = ["ScientistAgent", "SimulatorAgent", "JudgeAgent"]
