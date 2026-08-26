"""命令行入口：uv run mini-agent "你的目标"

默认离线（脚本模型，不花钱、不需要 key）；加 --live 走真实 OpenAI。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any

from dotenv import load_dotenv

from mini_agent import loop, model as model_mod, persist
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
        elif event == "compacted":
            print(f"  🗜 上下文压缩：省下约 {data['saved_tokens']} tokens，现约 {data['now']}")
        elif event == "externalized":
            print(f"  📄 {data['name']} 返回 {data['chars']} 字符，已外置到文件，上下文只留摘要")
        elif event == "retry":
            print(f"  ↻ 模型调用失败（{data['error']}），{data['delay']}s 后重试第 {data['attempt']} 次")
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
        "--deadline",
        type=float,
        default=600.0,
        help="墙上时间上限（秒），默认 600（10 分钟）；填 0 表示不限。"
        "检索限流/重试/网络慢只烧时间不烧钱，靠这道刹车拦",
    )
    p.add_argument(
        "--search",
        choices=["auto", "web", "offline"],
        default=None,
        help="检索模式；默认 --live 时用 auto（真联网），离线 demo 用 offline（零网络请求）",
    )
    p.add_argument(
        "--tool-timeout",
        type=float,
        default=30.0,
        help="单个工具调用的超时（秒），默认 30。超时会给模型一条 ERROR 结果，循环继续",
    )
    p.add_argument(
        "--context-limit",
        type=int,
        default=30_000,
        help="上下文超过这么多 token 就压缩早期历史，默认 30000",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="本次运行的落盘目录（状态快照 + 外置的大块工具结果），"
        "默认 runs/<时间戳>；填 off 关闭",
    )
    p.add_argument("--memory", default="memory.json", help="长期记忆文件路径")
    p.add_argument(
        "--resume",
        default=None,
        help="从某个 runs/<时间戳>/state.json（或它所在目录）接着跑",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="只打印最终答案")
    args = p.parse_args(argv)

    # 检索模式与模型解耦：离线 demo 默认也走离线检索，保证「不联网、秒出结果」
    os.environ["MINI_AGENT_SEARCH"] = args.search or ("auto" if args.live else "offline")

    resumed = persist.load(args.resume) if args.resume else None
    if resumed is not None:
        # 恢复时，命令行给的上限是「**再给这么多**」：已经用掉的步数/时间都记在状态里，
        # 直接沿用会导致一恢复就立刻又触顶。
        resumed.max_steps = resumed.step + args.max_steps
        resumed.remaining_budget += args.budget
        resumed.time_budget = (resumed.elapsed + args.deadline) if args.deadline > 0 else None
        resumed.salvaged = False  # 上一次的强制收尾不该算在这一次头上
    if args.run_dir == "off":
        run_dir = None
    elif args.run_dir:
        run_dir = args.run_dir
    elif args.resume:  # 恢复时沿用原来的目录，别把一次运行拆成两处
        run_dir = pathlib.Path(args.resume).parent if args.resume.endswith(".json") else args.resume
    else:
        run_dir = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")

    if not args.live:
        the_model = _offline_model()
    elif args.api == "responses":
        the_model = model_mod.ResponsesModel(args.model)
    else:
        the_model = model_mod.OpenAIModel(args.model)
    if not args.quiet and resumed is not None:
        print(
            f"从 {args.resume} 恢复：已跑 {resumed.step} 轮、"
            f"{resumed.elapsed:.0f} 秒，再给 {args.max_steps} 轮\n"
        )
    if not args.quiet:
        mode = f"live:{args.model}/{args.api}" if args.live else "offline:scripted"
        print(f"模式 {mode} ｜ 检索 {os.environ['MINI_AGENT_SEARCH']} ｜ 目标：{args.goal}\n")

    state = loop.run(
        goal=args.goal,
        model=the_model,
        memory=Memory(args.memory),
        max_steps=args.max_steps,
        budget=args.budget,
        time_budget=args.deadline if args.deadline > 0 else None,  # 0 = 不限
        tool_timeout=args.tool_timeout,
        context_limit=args.context_limit,
        run_dir=run_dir,
        resume=resumed,
        max_tool_calls_per_step=args.max_tool_calls,
        on_event=_printer(args.quiet),
    )

    print(f"\n答案：{state.answer}")
    if run_dir is not None and not args.quiet:
        print(f"轨迹已存到 {pathlib.Path(run_dir) / persist.FILENAME}（可用 --resume 接着跑）")
    if not args.quiet:
        print(f"状态：{json.dumps(state.snapshot(), ensure_ascii=False)}")
    return 0 if state.status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
