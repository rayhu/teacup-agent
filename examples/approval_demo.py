"""确认门的手动演示：让模型去发一封邮件，看三种策略下分别发生什么。

用脚本模型，所以**不需要 API key、不花钱**，随便跑几遍都行：

    uv run python examples/approval_demo.py
"""

import json
import os
import pathlib
import tempfile

os.environ["MINI_AGENT_SEARCH"] = "offline"

from mini_agent import loop  # noqa: E402
from mini_agent.memory import NullMemory  # noqa: E402
from mini_agent.model import ScriptedModel, assistant_calls, assistant_says  # noqa: E402

EMAIL = {"to": "boss@example.com", "subject": "周报", "body": "本周进展……"}


def script():
    """假模型的剧本：第一轮想发邮件，第二轮收尾。"""
    return [
        assistant_calls([("send_email", EMAIL)]),
        assistant_says("处理完毕。"),
    ]


POLICIES = {
    "默认（无人值守）": None,                      # 不传 approve = deny_all
    "人点了同意": lambda call, spec: True,
    "人点了拒绝": lambda call, spec: False,
    "按收件人白名单": lambda call, spec: json.loads(call.arguments)["to"].endswith("@example.com"),
}

os.chdir(tempfile.mkdtemp())  # send_email 会写当前目录，别弄脏项目
outbox = pathlib.Path("outbox.jsonl")

print(f"{'策略':22} {'执行':6} {'原因':10} 收件箱里的信")
print("-" * 62)
for name, approve in POLICIES.items():
    outbox.unlink(missing_ok=True)
    kwargs = {} if approve is None else {"approve": approve}
    state = loop.run("把周报发给老板", ScriptedModel(script()), memory=NullMemory(), **kwargs)

    trace = state.trace[0]
    sent = len(outbox.read_text().splitlines()) if outbox.exists() else 0
    print(f"{name:22} {str(trace.executed):6} {trace.skip_reason or '-':10} {sent} 封")

print("\n被拒绝时模型收到的结果消息：")
print(" ", loop.DENIED)
