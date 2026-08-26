"""mini_agent —— 一个最小但五脏俱全的 AI Agent。

    Agent = Model + State + Tools + Control Loop + Memory/Evals

对应关系：
    Model        -> model.py   （可换：真实 OpenAI / 离线脚本模型）
    State        -> state.py   （目标、消息、步数、预算、状态机）
    Tools        -> tools.py   （函数 + JSON Schema + 安全执行）
    Control Loop -> loop.py    （LLM → tool call → tool result → LLM）
    Memory       -> memory.py  （短期 = messages；长期 = memory.json）
    Evals        -> evals.py   （用脚本模型跑离线断言，无需 API key）
"""

from mini_agent.loop import run
from mini_agent.memory import Memory
from mini_agent.state import AgentState

__all__ = ["run", "Memory", "AgentState"]
__version__ = "0.1.0"
