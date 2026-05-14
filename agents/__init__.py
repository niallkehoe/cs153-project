"""
agents/
-------
LLM agent wrappers for the three roles in the re-discovery pipeline.

- ScientistAgent  : wraps GPT-1900, generates tool calls and tracks conversation context
- SimulatorAgent  : wraps a modern LLM via Cloudflare Workers AI, simulates experiment results
- JudgeAgent      : wraps a modern LLM via Cloudflare Workers AI, scores final hypotheses
- llm_client      : shared HTTP client for Cloudflare Workers AI (used by Simulator and Judge)
"""

from agents.scientist import ScientistAgent
from agents.simulator import SimulatorAgent
from agents.judge import JudgeAgent

__all__ = ["ScientistAgent", "SimulatorAgent", "JudgeAgent"]
