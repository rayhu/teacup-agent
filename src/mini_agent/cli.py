"""Command-line entry point: uv run mini-agent "your goal"

Offline by default (scripted model: no cost, no API key). Add --live for real
OpenAI calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
import time
from typing import Any

from dotenv import load_dotenv

from mini_agent import loop, model as model_mod, persist
from mini_agent.memory import Memory

DEFAULT_GOAL = "Look up NVIDIA's GPU strategy, and compute 1200 * 0.85 / 3"


def _offline_model() -> model_mod.ScriptedModel:
    """Offline demo script: one turn with two parallel tool calls, then the answer."""
    return model_mod.ScriptedModel(
        [
            model_mod.assistant_calls(
                [
                    ("search_web", {"query": "nvidia gpu strategy"}),
                    ("calculate", {"expression": "1200 * 0.85 / 3"}),
                ]
            ),
            model_mod.assistant_says(
                "NVIDIA's strategy binds three layers together: full-rack data-center "
                "systems, the CUDA software moat, and its own networking. Also, "
                "1200 * 0.85 / 3 = 340.0. (This answer comes from the offline "
                "scripted model.)"
            ),
        ]
    )


def _make_approver(policy: str, quiet: bool):
    """Approval policy for side-effecting tools.

    Default is auto: ask a human when there is a terminal, deny when there is not
    (CI, background jobs). "Nobody is watching, so allow it" is the most dangerous
    default there is.
    """

    def approve(call, spec) -> bool:
        if policy == "deny":
            return False
        if policy == "allow":  # explicitly requested yolo mode
            if not quiet:
                print(f"  [unlocked] --approve allow: running {call.name} without asking")
            return True
        if not sys.stdin.isatty():  # auto, but there is nobody to ask
            return False
        print(f"\n  WARNING: approval needed for a side-effecting operation: {call.name}")
        print(f"    arguments: {call.arguments}")
        print(f"    what it does: {spec.description[:120]}")
        try:
            answer = input("    Allow it to run? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    return approve


def _printer(quiet: bool):
    def on_event(event: str, data: dict[str, Any]) -> None:
        if quiet:
            return
        if event == "tool_call":
            print(f"  [step {data['step']}] -> {data['name']}({data['arguments']})")
        elif event == "tool_result":
            preview = data["result"].replace("\n", " ")
            print(f"  [step {data['step']}] <- {preview[:120]}")
        elif event == "throttled":
            print(
                f"  [step {data['step']}] throttled: the model asked for "
                f"{data['requested']} tool calls; running the first {data['cap']}, "
                "the rest go back to the next turn"
            )
        elif event == "planned":
            print("  [checklist] " + " | ".join(f"{i}. {t}" for i, t in enumerate(data["items"], 1)))
        elif event == "completion_check":
            print(
                "  [completion check] stopped early with open items: "
                + "; ".join(data["pending"])
            )
        elif event == "denied":
            print(f"  [denied] {data['name']} needs human approval and did not get it")
        elif event == "approved":
            print(f"  [approved] {data['name']}")
        elif event == "compacted":
            print(
                f"  [compacted] context: saved ~{data['saved_tokens']} tokens, "
                f"now ~{data['now']}"
            )
        elif event == "externalized":
            print(
                f"  [externalized] {data['name']} returned {data['chars']} characters; "
                "saved to a file, only an excerpt stays in context"
            )
        elif event == "retry":
            print(
                f"  [retry] model call failed ({data['error']}); "
                f"attempt {data['attempt']} in {data['delay']}s"
            )
        elif event == "salvaged":
            print(
                "  [wrap-up] resources exhausted: the answer below comes from what "
                "was already gathered, with no further searching"
            )
        elif event == "stopped":
            print(f"  [stopped] {data['reason']}")
        elif event == "error":
            print(f"  [error] {data['message']}")

    return on_event


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # pick up OPENAI_API_KEY from .env

    p = argparse.ArgumentParser(description="A minimal AI agent")
    p.add_argument("goal", nargs="?", default=DEFAULT_GOAL, help="the goal to accomplish")
    p.add_argument(
        "--live", action="store_true", help="call the real OpenAI model (needs an API key)"
    )
    p.add_argument("--model", default="gpt-5", help="model name used with --live")
    p.add_argument(
        "--api",
        choices=["responses", "chat"],
        default="responses",
        help="which OpenAI API to use with --live. responses (default) preserves "
        "reasoning state across tool calls; chat is the older path",
    )
    p.add_argument("--max-steps", type=int, default=8, help="maximum loop turns")
    p.add_argument(
        "--max-tool-calls",
        type=int,
        default=3,
        help="tool calls executed per turn; the rest are refused and must be re-sent "
        "next turn. 0 = unlimited",
    )
    p.add_argument("--budget", type=float, default=0.05, help="spending ceiling in USD")
    p.add_argument(
        "--deadline",
        type=float,
        default=600.0,
        help="wall-clock ceiling in seconds, default 600 (10 minutes); 0 = unlimited. "
        "Search throttling, retries and slow networks burn time without burning "
        "money, and only this brake catches them",
    )
    p.add_argument(
        "--search",
        choices=["auto", "web", "offline"],
        default=None,
        help="search mode; defaults to auto (real network) with --live and offline "
        "(zero network calls) for the offline demo",
    )
    p.add_argument(
        "--tool-timeout",
        type=float,
        default=30.0,
        help="timeout per tool call in seconds, default 30. On timeout the model "
        "gets an ERROR result and the loop continues",
    )
    p.add_argument(
        "--context-limit",
        type=int,
        default=30_000,
        help="compact the early history once the context exceeds this many tokens, "
        "default 30000",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="directory for this run (state snapshots + externalized tool results), "
        "default runs/<timestamp>; pass off to disable",
    )
    p.add_argument("--memory", default="memory.json", help="long-term memory file")
    p.add_argument(
        "--approve",
        choices=["auto", "deny", "allow"],
        default="auto",
        help="how to handle side-effecting tools such as send_email: auto (default: "
        "ask you when there is a terminal, deny when there is not) / deny / allow",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="continue from a runs/<timestamp>/state.json (or the directory holding it)",
    )
    p.add_argument(
        "--no-plan",
        action="store_true",
        help="skip the upfront checklist. By default the goal is decomposed into "
        "action items (one extra model call) so a multi-part request cannot be "
        "half-finished silently",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="print only the final answer")
    args = p.parse_args(argv)

    # Search mode is decoupled from model mode: the offline demo also searches
    # offline, so it stays network-free and instant.
    os.environ["MINI_AGENT_SEARCH"] = args.search or ("auto" if args.live else "offline")

    resumed = persist.load(args.resume) if args.resume else None
    if resumed is not None:
        # On resume the command-line ceilings mean "**give it this much more**":
        # steps and time already spent live in the state, so reusing them verbatim
        # would hit the ceiling again immediately.
        resumed.max_steps = resumed.step + args.max_steps
        resumed.remaining_budget += args.budget
        resumed.time_budget = (resumed.elapsed + args.deadline) if args.deadline > 0 else None
        resumed.salvaged = False  # last run's wrap-up should not count for this one
    if args.run_dir == "off":
        run_dir = None
    elif args.run_dir:
        run_dir = args.run_dir
    elif args.resume:  # keep one run in one directory
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
            f"Resuming from {args.resume}: {resumed.step} turns and "
            f"{resumed.elapsed:.0f}s already spent, granting {args.max_steps} more turns\n"
        )
    if not args.quiet:
        mode = f"live:{args.model}/{args.api}" if args.live else "offline:scripted"
        print(f"mode {mode} | search {os.environ['MINI_AGENT_SEARCH']} | goal: {args.goal}\n")

    state = loop.run(
        goal=args.goal,
        model=the_model,
        memory=Memory(args.memory),
        max_steps=args.max_steps,
        budget=args.budget,
        time_budget=args.deadline if args.deadline > 0 else None,  # 0 = unlimited
        tool_timeout=args.tool_timeout,
        context_limit=args.context_limit,
        run_dir=run_dir,
        resume=resumed,
        approve=_make_approver(args.approve, args.quiet),
        max_tool_calls_per_step=args.max_tool_calls,
        plan=not args.no_plan and args.live,  # the scripted demo has nothing to plan
        on_event=_printer(args.quiet),
    )

    print(f"\nAnswer: {state.answer}")
    if run_dir is not None and not args.quiet:
        print(
            f"Trajectory saved to {pathlib.Path(run_dir) / persist.FILENAME} "
            "(resume it with --resume)"
        )
    if not args.quiet:
        print(f"State: {json.dumps(state.snapshot(), ensure_ascii=False)}")
    return 0 if state.status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
