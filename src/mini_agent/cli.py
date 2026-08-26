"""命令行入口：uv run mini-agent "你的目标"

默认离线（脚本模型，不花钱、不需要 key）；加 --live 走真实 OpenAI。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv

from mini_agent import loop, model as model_mod
from mini_agent.memory import Memory

DEFAULT_GOAL = "查一下 NVIDIA 的 GPU 策略，并算一下 1200 * 0.85 / 3"


def _offline_model() -> model_mod.ScriptedModel:
    """离线演示剧本：一轮并行调两个工具 → 一轮给最终答案。"""
    return model_mod.ScriptedModel(
        [
            model_mod.assistant_calls(
                [
                    ("search_web", {"query": "nvidia gpu strategy"}),
                    ("calculate", {"expression": "1200 * 0.85 / 3"}),
                ]
            ),
            model_mod.assistant_says(
                "NVIDIA 的策略是「数据中心整机 + CUDA 软件护城河 + 自有网络」三层绑定；"
                "另外 1200 * 0.85 / 3 = 340.0。（本条来自离线剧本模型）"
            ),
        ]
    )


def _printer(quiet: bool):
    def on_event(event: str, data: dict[str, Any]) -> None:
        if quiet:
            return
        if event == "tool_call":
            print(f"  [step {data['step']}] → {data['name']}({data['arguments']})")
        elif event == "tool_result":
            preview = data["result"].replace("\n", " ")
            print(f"  [step {data['step']}] ← {preview[:120]}")
        elif event == "throttled":
            print(
                f"  [step {data['step']}] ⚠ 模型一次要了 {data['requested']} 个工具调用，"
                f"只执行前 {data['cap']} 个，其余退回下一轮"
            )
        elif event == "salvaged":
            print("  ♻ 资源耗尽，已强制收尾：下面的答案基于已获得的信息，未再检索")
        elif event == "stopped":
            print(f"  ⏹ 停止：{data['reason']}")
        elif event == "error":
            print(f"  ✗ {data['message']}")

    return on_event


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # 读取 .env 里的 OPENAI_API_KEY

    p = argparse.ArgumentParser(description="一个最小的 AI Agent")
    p.add_argument("goal", nargs="?", default=DEFAULT_GOAL, help="要完成的目标")
    p.add_argument("--live", action="store_true", help="调用真实 OpenAI 模型（需要 API key）")
    p.add_argument("--model", default="gpt-5", help="--live 时使用的模型名")
    p.add_argument(
        "--api",
        choices=["responses", "chat"],
        default="responses",
        help="--live 时用哪个 OpenAI 接口。responses（默认）能跨工具调用保住推理状态；chat 是旧路径",
    )
    p.add_argument("--max-steps", type=int, default=8, help="最大循环轮数")
    p.add_argument(
        "--max-tool-calls",
        type=int,
        default=3,
        help="每轮最多执行几个工具调用，超出的会被拒绝并要求下一轮重发；0 = 不限",
    )
    p.add_argument("--budget", type=float, default=0.05, help="预算上限（美元）")
    p.add_argument(
        "--search",
        choices=["auto", "web", "offline"],
        default=None,
        help="检索模式；默认 --live 时用 auto（真联网），离线 demo 用 offline（零网络请求）",
    )
    p.add_argument("--memory", default="memory.json", help="长期记忆文件路径")
    p.add_argument("-q", "--quiet", action="store_true", help="只打印最终答案")
    args = p.parse_args(argv)

    # 检索模式与模型解耦：离线 demo 默认也走离线检索，保证「不联网、秒出结果」
    os.environ["MINI_AGENT_SEARCH"] = args.search or ("auto" if args.live else "offline")

    if not args.live:
        the_model = _offline_model()
    elif args.api == "responses":
        the_model = model_mod.ResponsesModel(args.model)
    else:
        the_model = model_mod.OpenAIModel(args.model)
    if not args.quiet:
        mode = f"live:{args.model}/{args.api}" if args.live else "offline:scripted"
        print(f"模式 {mode} ｜ 检索 {os.environ['MINI_AGENT_SEARCH']} ｜ 目标：{args.goal}\n")

    state = loop.run(
        goal=args.goal,
        model=the_model,
        memory=Memory(args.memory),
        max_steps=args.max_steps,
        budget=args.budget,
        max_tool_calls_per_step=args.max_tool_calls,
        on_event=_printer(args.quiet),
    )

    print(f"\n答案：{state.answer}")
    if not args.quiet:
        print(f"状态：{json.dumps(state.snapshot(), ensure_ascii=False)}")
    return 0 if state.status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
