"""落盘与恢复 —— 让一次运行在结束后还留得下东西。

之前每次跑完，上下文、留痕、花费全随进程消失：出了问题只能靠终端回滚里那 120 个字符猜。
（真出现过：模型引用了几条链接，我们没法判断是检索来的还是它编的。）

存下来之后能做三件事：
1. 复盘：完整的 messages 和 trace 都在，想看哪一步看哪一步；
2. 恢复：崩了、Ctrl-C 了、超时了，从上一步接着跑，不重复已完成的工具调用；
3. 评测：trajectory eval（roadmap #7）要的就是这份轨迹。

刻意做成**每步一存**而不是跑完再存 —— 跑完才存的话，最需要它的那次崩溃恰好什么都没留下。
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

from mini_agent.state import AgentState, ToolTrace

FILENAME = "state.json"


def save(state: AgentState, run_dir: str | pathlib.Path) -> pathlib.Path:
    """把整个状态写进 run_dir/state.json（先写临时文件再改名，避免写到一半崩掉留下残档）。"""
    directory = pathlib.Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(dataclasses.asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load(path: str | pathlib.Path) -> AgentState:
    """从 state.json（或它所在的目录）读回状态。"""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / FILENAME
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    trace = [ToolTrace(**t) for t in data.pop("trace", [])]
    return AgentState(**data, trace=trace)
