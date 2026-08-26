"""Memory —— 分两层，都刻意做得很小。

短期记忆：就是 AgentState.messages 本身，任务结束即消失。
长期记忆：一个 JSON 文件，跨会话保留；开局注入 system prompt，
          模型可以通过 `remember` 工具往里写。

真实项目里这一层会换成向量库 / 数据库，接口保持 remember() + recall() 即可。
"""

from __future__ import annotations

import json
import pathlib


class Memory:
    def __init__(self, path: str | pathlib.Path = "memory.json", limit: int = 20):
        self.path = pathlib.Path(path)
        self.limit = limit
        self.facts: list[str] = []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.facts = [str(x) for x in data.get("facts", [])]
            except (json.JSONDecodeError, OSError):
                self.facts = []  # 记忆损坏不该让 Agent 崩掉

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"facts": self.facts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def remember(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.facts:
            self.facts.append(fact)
            self.facts = self.facts[-self.limit :]  # 简单的淘汰策略：只留最近 N 条
            self.save()

    def recall(self) -> str:
        """取出可以拼进 system prompt 的记忆块。"""
        if not self.facts:
            return ""
        lines = "\n".join(f"- {f}" for f in self.facts)
        return f"你还记得以下事实（来自过去的会话）：\n{lines}"


class NullMemory(Memory):
    """不落盘的记忆，评测和单测用。"""

    def __init__(self) -> None:
        self.path = pathlib.Path("/dev/null")
        self.limit = 20
        self.facts = []

    def save(self) -> None:  # noqa: D102
        pass

    def load(self) -> None:  # noqa: D102
        pass
