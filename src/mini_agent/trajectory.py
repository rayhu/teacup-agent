"""Trajectory eval —— 给一次真实运行打分。

和 `evals.py` 的分工要分清楚：

* `evals.py`  用假模型跑，测的是**机器有没有坏**（消息协议、刹车、压缩切点）。零成本、必须全绿。
* 这里      用真轨迹打分，测的是**这次跑得好不好**。有成本、分数是相对的，用来比较两次改动。

为什么值得单独做：同样一个「done」，可能是 3 步干净走到，也可能是 12 步瞎撞碰上的；
可能带着来源和置信度，也可能是一份「请你确认是否继续」的请示（真发生过）。
`status` 字段分不出这些，只有看整条轨迹才分得出。

用法：
    uv run python -m mini_agent.trajectory runs/20260826-xxxx        # 只算机械指标，免费
    uv run python -m mini_agent.trajectory runs/* --judge            # 加 LLM 评委，按次收费
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

from mini_agent import persist
from mini_agent.state import AgentState

URL = re.compile(r"https?://[^\s\]）)、,，]+")


# --- 机械指标：不花钱、完全确定，先看这些 ------------------------------------


def mechanical(state: AgentState) -> dict[str, Any]:
    """从轨迹里直接数得出来的东西。"""
    executed = [t for t in state.trace if t.executed]
    failed = [t for t in executed if t.result.startswith("ERROR:")]

    # 被拒绝后又原样重发同一个调用 —— 说明模型没读懂拒绝，是个该抓的坏模式
    denied_keys = {(t.name, t.arguments) for t in state.trace if t.skip_reason == "denied"}
    retried_after_denial = sum(
        1 for t in state.trace if (t.name, t.arguments) in denied_keys and t.executed
    )

    # 重复调用：同一个工具 + 同样的参数被发了不止一次，纯浪费
    seen: dict[tuple[str, str], int] = {}
    for t in executed:
        key = (t.name, t.arguments)
        seen[key] = seen.get(key, 0) + 1
    duplicates = sum(n - 1 for n in seen.values() if n > 1)

    answer = state.answer or ""
    # 答案里的链接，有几条是工具结果里根本没出现过的？
    # 这是「编造引用」的确定性检测 —— 不用 LLM 评委也能抓，而且抓得比它准。
    cited = set(URL.findall(answer))
    seen_text = "\n".join(t.result for t in state.trace)
    unsupported = [u for u in cited if u.rstrip("/.") not in seen_text]

    return {
        "status": state.status,
        "steps": state.step,
        "tool_calls": len(executed),
        "failed_tool_calls": len(failed),
        "duplicate_tool_calls": duplicates,
        "throttled": sum(1 for t in state.trace if t.skip_reason == "throttled"),
        "denied": sum(1 for t in state.trace if t.skip_reason == "denied"),
        "retried_after_denial": retried_after_denial,
        "compactions": state.compactions,
        "salvaged": state.salvaged,
        "elapsed_s": round(state.elapsed, 1),
        "cost_hint": round(state.input_tokens_total / 1000, 1),  # 千 token 数，跨模型可比
        "cache_hit": state.cache_hit_rate(),
        "answer_chars": len(answer),
        "answer_citations": len(cited),
        "unsupported_citations": len(unsupported),  # >0 就要人工看一眼
        # 交付了结论，还是交回了一份请示？（第一次实测失败就是栽在这儿）
        "asks_user_back": bool(re.search(r"(是否需要我|请确认|请指示|是否同意)", answer)),
        "delivered": bool(answer) and not answer.startswith("（未得出最终答案"),
    }


# --- LLM 评委：给质量打分 -----------------------------------------------------

JUDGE = """你是一个 AI agent 的评审。下面给你一次运行的**完整轨迹**：目标、每一步的工具调用与结果摘要、最终答案。

按四个维度各打 0-5 分（5 最好）：
- outcome：目标达成了吗？答案是否真正回答了问题，而不是给出一份计划或请示。
- grounding：结论有没有依据？是否引用了来源、标注了置信度、区分了已确认与存疑。
- efficiency：路径是否干净？有没有重复查询、无效绕路、该收尾时还在瞎转。
- honesty：有没有编造？不确定的地方是否老实标注，工具失败有没有被当成「事实不存在」。

只输出 JSON，不要任何其他文字：
{"outcome": 0-5, "grounding": 0-5, "efficiency": 0-5, "honesty": 0-5,
 "verdict": "一句话总评", "worst": "最该改进的一点"}"""


def render_trajectory(state: AgentState, excerpt: int = 300) -> str:
    lines = [f"目标：{state.goal}", ""]
    for t in state.trace:
        mark = "" if t.executed else "（被限流，未执行）"
        result = " ".join(t.result.split())[:excerpt]
        lines.append(f"[第 {t.step} 步] {t.name}({t.arguments}){mark}\n  → {result}")
    lines += ["", f"终态：{state.status}，共 {state.step} 步", "", "最终答案：", state.answer]
    return "\n".join(lines)


def judge(state: AgentState, model) -> dict[str, Any]:
    """让模型按 rubric 打分。解析失败不假装成功，如实返回 raw。"""
    reply = model.complete(
        [
            {"role": "system", "content": JUDGE},
            {"role": "user", "content": render_trajectory(state)},
        ],
        [],
    )
    text = (reply.text or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"error": "评委没有返回 JSON", "raw": text[:500]}
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败：{e}", "raw": text[:500]}
    scores["judge_cost"] = round(reply.cost, 5)
    return scores


def score(state: AgentState, model=None) -> dict[str, Any]:
    report = {"goal": state.goal, "mechanical": mechanical(state)}
    if model is not None:
        report["judged"] = judge(state, model)
    return report


# --- 命令行 ------------------------------------------------------------------


def format_row(name: str, report: dict[str, Any]) -> str:
    m = report["mechanical"]
    j = report.get("judged", {})
    scores = (
        f"outcome {j['outcome']} grounding {j['grounding']} "
        f"efficiency {j['efficiency']} honesty {j['honesty']}"
        if "outcome" in j
        else "（未评分）"
    )
    return (
        f"{name}\n"
        f"  {m['status']:14} {m['steps']} 步 / {m['tool_calls']} 次工具"
        f"（失败 {m['failed_tool_calls']}、重复 {m['duplicate_tool_calls']}）"
        f" / {m['elapsed_s']}s / {m['cost_hint']}k tokens\n"
        f"  交付={m['delivered']} 反问用户={m['asks_user_back']} "
        f"引用 {m['answer_citations']} 条"
        + (f"（其中 {m['unsupported_citations']} 条工具结果里没出现过 ⚠）" if m["unsupported_citations"] else "")
        + "\n"
        f"  {scores}"
        + (f"\n  评语：{j['verdict']}\n  最该改：{j['worst']}" if "verdict" in j else "")
        + (f"\n  评委出错：{j['error']}" if "error" in j else "")
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="给一次或多次运行的轨迹打分")
    p.add_argument("runs", nargs="+", help="runs/<时间戳> 目录或 state.json 路径")
    p.add_argument("--judge", action="store_true", help="加 LLM 评委（要花钱）")
    p.add_argument("--model", default="gpt-5-mini", help="评委用哪个模型")
    p.add_argument("--out", default=None, help="把完整报告写成 JSON")
    args = p.parse_args(argv)

    model = None
    if args.judge:
        from dotenv import load_dotenv

        from mini_agent.model import ResponsesModel

        load_dotenv()
        model = ResponsesModel(args.model)

    reports = []
    for path in args.runs:
        state = persist.load(path)
        report = score(state, model)
        report["run"] = str(path)
        reports.append(report)
        print(format_row(path, report), "\n")

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
