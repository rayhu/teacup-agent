"""teacup_agent — a minimal AI agent that has all the organs.

    Agent = Model + State + Tools + Control Loop + Memory/Evals

Where each part lives:
    Model        -> model.py   (swappable: real OpenAI / offline scripted model)
    State        -> state.py   (goal, messages, steps, budget, status machine)
    Tools        -> tools.py   (function + JSON Schema + safe execution)
    Control Loop -> loop.py    (LLM -> tool call -> tool result -> LLM)
    Memory       -> memory.py  (short term = messages; long term = memory.json)
    Evals        -> evals.py   (offline assertions with a scripted model, no API key)
"""

from teacup_agent.loop import run
from teacup_agent.memory import Memory
from teacup_agent.state import AgentState

__all__ = ["run", "Memory", "AgentState"]
__version__ = "0.1.0"
