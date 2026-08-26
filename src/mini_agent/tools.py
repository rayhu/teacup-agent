"""Tools —— Agent 能对世界做的动作。

每个工具 = 一个 Python 函数 + 一份 JSON Schema（描述给模型看）。
重点是把「注册 → 描述 → 调用 → 回填」这条链路走通；
search_web 已经是真实联网检索，其余几个仍是最小实现。
"""

from __future__ import annotations

import ast
import json
import operator
import os
import pathlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., str]


REGISTRY: dict[str, Tool] = {}


def tool(description: str, parameters: dict[str, Any]):
    """装饰器：把一个普通函数登记成模型可调用的工具。"""

    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        REGISTRY[fn.__name__] = Tool(fn.__name__, description, parameters, fn)
        return fn

    return deco


def specs() -> list[dict[str, Any]]:
    """导出成 OpenAI Chat Completions 的 tools 参数格式。"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in REGISTRY.values()
    ]


def execute(name: str, arguments: str) -> str:
    """执行一次工具调用。

    关键点：任何失败都**不抛异常**，而是把错误信息当作工具结果返回给模型。
    模型看到错误后可以自己改参数重试 —— 这正是 Agent 循环的价值所在。
    """
    tool_obj = REGISTRY.get(name)
    if tool_obj is None:
        return f"ERROR: 未知工具 {name!r}，可用工具：{', '.join(REGISTRY)}"

    try:
        # 注意：模型返回的 arguments 是 JSON **字符串**，必须先反序列化。
        # （原笔记里 `fn(**item.arguments)` 就是栽在这一步。）
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return f"ERROR: 参数不是合法 JSON（{e}）。收到的是：{arguments!r}"

    if not isinstance(kwargs, dict):
        return f"ERROR: 参数必须是 JSON 对象，收到 {type(kwargs).__name__}"

    try:
        return str(tool_obj.fn(**kwargs))
    except TypeError as e:
        return f"ERROR: 参数不匹配（{e}）。期望的 schema：{json.dumps(tool_obj.parameters, ensure_ascii=False)}"
    except Exception as e:  # 工具自身出错，同样回传给模型
        return f"ERROR: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# 具体工具：search_web 已接真实检索，其余仍是最小实现，可逐个替换成真货
# --------------------------------------------------------------------------

# 离线语料：没装 ddgs / 没网 / 显式要求离线时的兜底，同时保证评测确定性。
_CORPUS = {
    "nvidia gpu strategy": (
        "NVIDIA 的 GPU 策略围绕三层展开：(1) 数据中心 —— Blackwell/Rubin 架构按机柜级"
        "（NVL72）整体交付，把 GPU、CPU、NVLink 交换机打包成一个计算单元；"
        "(2) 软件护城河 —— CUDA + NIM/TensorRT-LLM 推理栈，让迁移成本极高；"
        "(3) 网络 —— 收购 Mellanox 后用 NVLink/InfiniBand 绑定整机方案。"
    ),
    "cuda": "CUDA 是 NVIDIA 的并行计算平台，也是其最深的护城河：生态迁移成本远高于硬件本身。",
    "agent": (
        "Agent = Model + State + Tools + Control Loop + Memory/Evals。"
        "本质循环是：LLM → tool call → tool result → LLM。"
    ),
}


def _search_corpus(query: str) -> str:
    """离线兜底检索。

    匹配用 all() 而不是 any()：早期版本用 any()，导致 "OpenAI strategy" 里的
    "strategy" 命中了 "nvidia gpu strategy" 这条，把 NVIDIA 的资料喂给了 OpenAI 的问题。
    宁可返回「没找到」也不能返回错的东西 —— 模型会照单全收。
    """
    words = set(re.findall(r"[\w一-鿿]+", query.lower()))
    hits = [text for key, text in _CORPUS.items() if set(key.split()) <= words]
    if not hits:
        return (
            f"没有找到与 {query!r} 相关的结果。"
            f"（当前是**离线**语料库，只收录了：{', '.join(_CORPUS)}）"
        )
    return "\n\n".join(hits[:3])


# 检索后端的限流保护。实测：连续快速发 4-5 个查询就会被 DuckDuckGo 掐断，
# 而 agent 恰好最爱一轮甩好几个查询。这里做两件事：请求间隔 + 失败退避重试。
# 实测（三个并行检索）：间隔 1.5s 要 8.3s，间隔 0 只要 4.7s —— 限流间隔才是瓶颈，不是网络。
# 折中到 0.5s：6 个检索连发零失败，耗时 8.9s；万一真被限流，还有退避重试兜底。
_MIN_INTERVAL = 0.5  # 秒，两次真实检索之间的最小间隔
_RETRIES = 3
_last_search_at = 0.0
_throttle_lock = threading.Lock()  # 工具会被并行执行，这个全局量必须加锁


def _throttle() -> None:
    """给真实检索之间垫上最小间隔。

    锁保证「等待 + 记账」是原子的：并行的三个检索会被排开成 0s / 1.5s / 3.0s 起跑，
    而不是同时冲出去撞限流。注意起跑被排开不等于失去并行 —— 各自的网络往返仍然重叠。
    """
    global _last_search_at
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_search_at)
        if wait > 0:
            time.sleep(wait)
        _last_search_at = time.monotonic()


def _search_web_backend(query: str, max_results: int) -> str:
    """真实检索后端：DuckDuckGo，不需要 API key。

    失败会退避重试（1s / 2s / 4s）。检索失败和「查无此事」是**完全不同**的两件事，
    绝不能让前者伪装成后者 —— 那会让模型得出「世界上没有这件事」的结论。
    """
    from ddgs import DDGS  # 延迟导入，离线路径不依赖它

    last_error: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            _throttle()
            results = DDGS().text(query, max_results=max_results)
            break
        except Exception as e:  # 多半是限流，退避后再试
            last_error = e
            if attempt < _RETRIES - 1:
                time.sleep(2**attempt)
    else:
        raise RuntimeError(f"检索重试 {_RETRIES} 次仍失败：{last_error}") from last_error

    if not results:
        return f"没有搜到与 {query!r} 相关的网页结果。"
    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        body = " ".join((r.get("body") or "").split())[:300]
        lines.append(f"{i}. {title}\n   {url}\n   {body}")
    return "\n".join(lines)


@tool(
    description=(
        "联网搜索，返回带标题、链接和摘要的结果列表。"
        "引用结论时请附上返回的链接；同一问题最多搜 2-3 次不同关键词即可。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {
                "type": "integer",
                "description": "返回条数，1-10，默认 5",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    },
)
def search_web(query: str, max_results: int = 5) -> str:
    """三种模式由环境变量 MINI_AGENT_SEARCH 控制：

    auto（默认）：能联网就联网，失败自动退回离线语料并注明原因
    web        ：只用真实检索，失败直接报错（避免模型误以为「世界上没有」）
    offline    ：只用本地语料，不发任何网络请求（评测/单测用）
    """
    mode = os.getenv("MINI_AGENT_SEARCH", "auto").lower()
    max_results = max(1, min(int(max_results), 10))

    if mode == "offline":
        return _search_corpus(query)

    try:
        return _search_web_backend(query, max_results)
    except ImportError:
        if mode == "web":
            return "ERROR: 未安装 ddgs，无法联网检索。请执行 uv sync，或设 MINI_AGENT_SEARCH=offline。"
        return f"[联网检索不可用（未装 ddgs），以下为离线语料结果]\n{_search_corpus(query)}"
    except Exception as e:
        fallback = _search_corpus(query)
        # 只有本地语料**真的命中**时才降级；否则必须明确报错，
        # 不能把「检索坏了」伪装成「没有找到」——模型会把后者当成「不存在」。
        if mode == "web" or fallback.startswith("没有找到"):
            return (
                f"ERROR: 检索失败（{type(e).__name__}: {e}）。"
                "这**不代表**该信息不存在，只说明检索通道暂时不可用；"
                "可稍后重试、换关键词，或基于已有结果作答并标注该项未能核实。"
            )
        return f"[联网检索失败：{type(e).__name__}，以下为离线语料结果]\n{fallback}"


# 安全的算术求值：只允许字面量和四则运算，不用 eval()。
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"不支持的表达式片段：{ast.dump(node)}")


@tool(
    description="计算一个算术表达式，例如 '(1200 * 0.85) / 3'。只支持 + - * / ** %。",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "算术表达式"}},
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    return str(_eval_node(ast.parse(expression, mode="eval").body))


@tool(
    description="读取当前项目目录下的一个文本文件（最多返回 2000 字符）。",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对当前目录的路径"}},
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    root = pathlib.Path.cwd().resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)):  # 简单的越界保护
        return "ERROR: 只允许读取当前项目目录内的文件"
    if not target.is_file():
        return f"ERROR: 文件不存在：{path}"
    return target.read_text(encoding="utf-8", errors="replace")[:2000]


# 长期记忆的写入口。为了保持简单，这里用模块级绑定，由 loop.run() 注入。
_memory = None


def bind_memory(memory) -> None:
    global _memory
    _memory = memory


@tool(
    description="把一条值得跨会话保留的事实写进长期记忆（例如用户偏好、稳定结论）。",
    parameters={
        "type": "object",
        "properties": {"fact": {"type": "string", "description": "一句话事实"}},
        "required": ["fact"],
    },
)
def remember(fact: str) -> str:
    if _memory is None:
        return "ERROR: 当前没有可用的长期记忆"
    _memory.remember(fact)
    return f"已记住：{fact}"
