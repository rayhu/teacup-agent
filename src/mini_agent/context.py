"""上下文工程 —— 让 agent 能跑长任务的那一层。

`state.messages` 天然是个只增不减的 list。跑 20-30 轮之后：撑爆 context window、
每轮全价重发、模型注意力被无关的旧工具输出稀释。三件事一起变坏。

这里做两件互补的事，**外置比压缩更有效，所以先做外置**：

1. **外置（externalize）**：大块工具结果写进 runs/ 目录，上下文里只留摘要 + 文件路径。
   模型需要细节时用 read_file 取回。省下来的 token 是实打实的，而且信息一点没丢。
2. **压缩（compact）**：上下文仍然超阈值时，把较早的一段交给模型摘要，
   用一条摘要消息替换掉原来的几十条。信息有损，所以是第二道手段。

压缩有一个**必须守住的约束**：不能把一个工具调用和它的结果拆散。
少了结果的 function_call 会让下一轮请求直接 400。所以只在「没有悬空调用」的位置下刀。
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from mini_agent.model import Model

# --- token 估算 -------------------------------------------------------------

_CJK = re.compile(r"[㐀-鿿豈-﫿]")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：英文约 4 字符/token，中日韩约 1.5 字符/token。

    只用来做「要不要压缩」的判断，不用来计费 —— 计费用模型返回的真实 usage。
    """
    cjk = len(_CJK.findall(text))
    return int(cjk / 1.5 + (len(text) - cjk) / 4) + 1


def messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(str(m)) for m in messages)


# --- 外置：大结果写文件，上下文只留摘要 + 路径 --------------------------------

EXCERPT = 600  # 留在上下文里的字符数


def externalize(result: str, run_dir: pathlib.Path, step: int, index: int, name: str) -> str:
    """把超长的工具结果写盘，返回「摘要 + 路径」的精简版。

    模型想看细节就用 read_file 读回来 —— 这比任何摘要算法都保真，
    代价只是多一次工具调用，而且**只在真的需要时**才付。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"step{step:02d}_{index}_{name}.txt"
    path.write_text(result, encoding="utf-8")
    try:
        rel = path.relative_to(pathlib.Path.cwd())
    except ValueError:
        rel = path
    return (
        f"{result[:EXCERPT]}\n\n"
        f"[内容较长（{len(result)} 字符），已存到 {rel}，"
        f"上面只是开头部分。需要完整内容就用 read_file 读这个路径。]"
    )


# --- 压缩：找安全切点，把早期上下文换成一条摘要 -------------------------------


def safe_cut_points(messages: list[dict[str, Any]]) -> list[int]:
    """所有「没有悬空工具调用」的位置 —— 只有在这些点下刀才不会破坏消息协议。

    扫描逻辑和 evals 里的顺序不变量是同一套：宣告的 id 进集合，回填的 id 出集合，
    集合空的时刻就是安全点。两种 API 形状都认。
    """
    open_ids: set[str] = set()
    points = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                open_ids.add(tc["id"])
        elif msg.get("role") == "tool":
            open_ids.discard(msg.get("tool_call_id"))
        elif msg.get("type") == "function_call":
            open_ids.add(msg["call_id"])
        elif msg.get("type") == "function_call_output":
            open_ids.discard(msg.get("call_id"))
        if not open_ids:
            points.append(i + 1)  # 切在这一条之后是安全的
    return points


SUMMARIZER = """你在压缩一个 AI agent 的工作上下文。

把下面这段历史浓缩成一份交接备忘，供它自己继续工作用。必须保留：
- 已经查证到的事实与结论，连同来源链接
- 试过什么、失败了什么（避免它重复踩坑）
- 还没解决的问题

删掉：寒暄、重复的中间过程、已经被更新结论取代的旧说法。
直接输出备忘正文，不要写「好的，这是摘要」之类的开场白。"""


def render(messages: list[dict[str, Any]]) -> str:
    """把一段消息摊平成给摘要器看的纯文本。"""
    lines = []
    for m in messages:
        kind = m.get("type") or m.get("role")
        body = m.get("content") or m.get("output") or m.get("arguments") or ""
        if isinstance(body, list):  # Responses 的富内容
            body = " ".join(str(x.get("text", "")) for x in body if isinstance(x, dict))
        lines.append(f"[{kind}] {str(body)[:1500]}")
    return "\n".join(lines)


def compact(
    state,
    model: Model,
    limit: int,
    keep_recent: int = 8,
) -> int:
    """上下文超限时把早期部分压成一条摘要，返回省下的估算 token 数（0 = 没压）。

    保留：system（前缀不能动，否则 prompt caching 失效）、原始目标、最近 keep_recent 条。
    """
    messages = state.messages
    if messages_tokens(messages) <= limit:
        return 0

    head = 2  # system + 原始目标，永远留着
    candidates = [p for p in safe_cut_points(messages) if head < p <= len(messages) - keep_recent]
    if not candidates:
        return 0  # 找不到安全切点就别硬切 —— 拆散调用和结果的代价是下一轮直接 400
    cut = candidates[-1]

    before = messages_tokens(messages)
    summary = model.complete(
        [
            {"role": "system", "content": SUMMARIZER},
            {"role": "user", "content": render(messages[head:cut])},
        ],
        [],
    )
    state.charge(summary.cost)
    if not summary.text:
        return 0

    state.messages = messages[:head] + [
        {
            "role": "system",
            "content": f"[上下文摘要] 早期的 {cut - head} 条记录已压缩如下：\n{summary.text}",
        }
    ] + messages[cut:]
    state.compactions += 1
    return before - messages_tokens(state.messages)
