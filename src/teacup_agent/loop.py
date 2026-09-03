"""Control Loop — the ~80 lines that wire Model / State / Tools / Memory together.

In one line:

    LLM -> tool call -> tool result -> LLM -> ...

Three traps, all marked in the code below:
1. The assistant message carrying tool_calls must be appended **before** the tool
   results;
2. One turn may contain **several** tool_calls, and every id needs its own
   role=tool message;
3. Termination is not a mysterious check_completion(): it is "the model stopped
   asking for tools" plus the step / budget / time ceilings.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable

from teacup_agent import context as ctx
from teacup_agent import hooks as hooks_mod
from teacup_agent import persist
from teacup_agent import plan as plan_mod
from teacup_agent import reflect as reflect_mod
from teacup_agent import skills as skills_mod
from teacup_agent import subagent as subagent_mod
from teacup_agent import tools as tools_mod
from teacup_agent.memory import Memory, NullMemory
from teacup_agent.model import Model, ToolCall, chat_tool_result
from teacup_agent.state import AgentState, ToolTrace

SYSTEM_PROMPT = """You are an autonomous, tool-using agent running **unattended**.

Today is {today}.

About recency (important):
- Your training data ends before today; the world kept moving after that.
- When a search result conflicts with your memory, **trust the search result**. Do
  not dismiss something as false because "that number is far larger than I
  remember" — that usually only means your memory is out of date.
- The right response is source grading plus cross-checking, not blanket rejection:
  primary sources (company sites, filings, regulators) > major media > secondary
  aggregators and SEO pages. Several independent sources agreeing is enough to
  accept and cite; only low-quality support means you mark it "unverified".
- You still may not invent specific numbers. If you did not find it, say so.

About how you search (equally important):
- Anchor time expressions in your queries to **today**, not to the years you
  remember. "The last six months" means the six months before today.
- Do not build queries on specific facts from memory (a funding amount, a model
  version number) — those are usually stale and drag you back to old news. Start
  broad and recent, then drill down.

Hard constraints of unattended mode:
- **Nobody will answer your questions.** Do not ask for permission, do not ask
  "should I continue", do not hand a to-do list back to the user. If you think a
  step is worth doing, do it.
- Each turn starts with your remaining steps, budget and time. If **any one** of
  them is running out, wrap up now — do not fixate on whichever is most generous.
- Never come back empty-handed: even with partial information, give "the best
  conclusion available now + confidence + what remains unverified" rather than a
  plan of action.

How to work:
- Call tools when you need external information; never make it up. You may request
  several tool calls in one turn, but at most {max_tool_calls} run per turn and the
  rest are rejected and must be re-sent next turn — so pick the important ones.
- A tool result starting with ERROR: means the call was wrong. Fix the arguments
  and retry instead of giving up.
- A few tools have external, irreversible side effects (sending email, for example)
  and need human approval before running. **Attempt the call anyway.** The approval
  prompt is exactly where the user grants or refuses permission, so:
  1. call the tool when the task asks for it;
  2. only if the call comes back denied should you take another route, or state in
     your final answer that this step is left to the user;
  3. never re-send an identical call that was denied.
  Do not ask for authorization in your answer instead of calling the tool. Nobody
  reads the answer before the run ends, and that is not how approval works here.
- No tool calls = you consider the task complete. That is the loop's stop signal;
  do not use it to ask a question.
- When you learn something worth keeping across sessions (user preferences, stable
  conclusions), write it down with the remember tool.
- Each turn shows a checklist of what the request asked for. Work through every item,
  including any side-effecting one, and mark each finished item with update_todo. An
  item you cannot complete is marked blocked with a reason — never left silently
  undone.

Keep answers concise, and label key conclusions with their source and confidence."""


def status_note(state: AgentState) -> dict[str, Any]:
    """Tell the model where it stands at the start of each turn.

    Why this is not part of the system prompt: putting it there would change the
    context prefix every turn and void prompt caching (roadmap #2). Appended at the
    end, it leaves the prefix untouched.
    """
    left = state.max_steps - state.step  # turns remaining after this one
    time_left = state.time_left()
    # Whichever brake is tightest gets the spotlight. In one real run the model was
    # misled by "91% of the budget left" and ignored that its steps were nearly
    # gone. With several brakes running, surface the **tightest** one.
    tight_on_time = time_left is not None and time_left <= max(15.0, (state.time_budget or 0) * 0.25)

    if left <= 0:
        urgency = (
            "WARNING: **this is the FINAL turn** and the tools have been taken away. "
            "Give your final conclusion from what you already have — confirmed "
            "findings, confidence, and what you could not verify. Do not come back "
            "empty-handed."
        )
    elif left <= 2 or tight_on_time:
        why = f"only {left} turn(s) left" if left <= 2 else f"only {time_left:.0f}s left"
        urgency = f"WARNING: {why}. Start wrapping up: at most one more lookup, then conclude."
    else:
        urgency = "Keep verifying while resources allow; conclude as soon as they run low."

    resources = f"turn {state.step}/{state.max_steps}; budget ${state.remaining_budget:.4f} left"
    if time_left is not None:
        resources += f"; {time_left:.0f}s left"

    content = f"[run status] {resources}. {urgency}"
    if checklist := plan_mod.render(state.todo):
        # The checklist rides along with the resource line for the same reason the
        # resource line exists at all: the model cannot act on what it is not told.
        content += f"\n{checklist}"
    return {"role": "system", "content": content}


COMPLETION_CHECK = """[completion check] You stopped calling tools, but the checklist
still has open items:

{pending}

Either do them now — including any step that needs approval, where attempting the call
is how the user is asked — or, if an item genuinely cannot be done, call update_todo
with status='blocked' and a reason, then give your final answer. Do not simply answer
around the missing item."""


def finalize(state: AgentState, model: Model, emit: Callable[..., None]) -> None:
    """Forced wrap-up turn: when resources run out, ask once more with no tools and
    squeeze a conclusion out of what is already there.

    The lesson from a real run: the model spent all 8 turns searching and left not
    one line of conclusion — ten searches paid for and nothing to show. A brake must
    not only stop the car, it must also unload it.
    """
    state.messages.append(
        {
            "role": "system",
            "content": (
                f"[forced wrap-up] Resources are exhausted ({state.status}) and tools "
                "are no longer available. Give your final conclusion from the "
                "information you already have: confirmed findings + confidence + "
                "what remains unverified."
                + (
                    "\nThese checklist items were never completed; say so plainly and "
                    "tell the user what is left for them to do: "
                    + "; ".join(t.text for t in plan_mod.pending(state.todo))
                    if plan_mod.pending(state.todo)
                    else ""
                )
            ),
        }
    )
    try:
        reply = model.complete(state.messages, [])  # empty tool list = talking only
    except Exception as e:
        emit("error", message=f"forced wrap-up failed: {type(e).__name__}: {e}")
        return
    state.charge(reply.cost)
    state.messages.extend(reply.items)

    # The wrap-up turn is given no tools, but a model can still emit tool calls.
    # **Every announced id must have a result**, or the message protocol breaks and
    # the next request (including a later --resume) fails with a 400.
    for call in reply.tool_calls:
        state.messages.append(
            result_item(
                model,
                call,
                "ERROR: the run has entered forced wrap-up; tools are no longer "
                "available. Give your conclusion directly.",
            )
        )

    if not reply.text:  # nothing salvaged — do not claim otherwise
        return
    state.answer = reply.text
    state.salvaged = True
    emit("salvaged", text=reply.text)


def result_item(model: Model, call: ToolCall, result: str) -> dict[str, Any]:
    """Ask the backend for the tool-result shape; fall back to the Chat shape."""
    fn = getattr(model, "tool_result_item", None)
    return fn(call, result) if fn else chat_tool_result(call, result)


def is_retryable(error: Exception) -> bool:
    """Rate limits (429), server errors (5xx) and network errors without a status
    code are worth retrying.

    A 4xx (bad arguments, bad credentials) returns the same answer however many
    times you ask; retrying only burns time.
    """
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status is None:
        return True
    return status == 429 or status >= 500


def complete_with_retries(
    model: Model,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    emit: Callable[..., None],
    attempts: int = 3,
    sleep: Callable[[float], None] | None = None,
):
    """Backoff retries around the model call.

    A retry is **not a step**: steps measure how many decisions the model made, and
    one rate limit should not eat into the agent's thinking allowance.
    """
    for attempt in range(attempts):
        try:
            return model.complete(messages, tools)
        except Exception as e:
            if attempt == attempts - 1 or not is_retryable(e):
                raise
            delay = 2**attempt
            emit("retry", error=f"{type(e).__name__}: {e}", attempt=attempt + 1, delay=delay)
            (sleep or time.sleep)(delay)  # resolved at call time so tests can patch it


DENIED = (
    "ERROR: this operation requires human approval, which was not granted, so it "
    "**was NOT executed**. Do not re-send the same call; take an approach that needs "
    "no approval, or state in your final answer that the user has to do this step."
)


def deny_all(call: ToolCall, tool: Any) -> bool:
    """Default policy: unattended runs deny anything that needs approval.

    "Nobody is watching, so allow it" is the most dangerous default there is — the
    run that goes wrong is precisely the one nobody was watching.
    """
    return False


EXTERNALIZE_OVER = 2000  # tool results longer than this go to disk, excerpt inline


def execute_calls(
    state: AgentState,
    calls: list[ToolCall],
    model: Model,
    emit: Callable[..., None],
    tool_timeout: float,
    run_dir: pathlib.Path | None = None,
    approve: Callable[[ToolCall, Any], bool] = deny_all,
) -> None:
    """Run all of a turn's tool calls in parallel, then feed results back **in the
    original order**.

    Two invariants that must hold (and that parallelism breaks most easily):
    1. results are appended in tool_calls order, and every id gets exactly one;
    2. a call the throttle rejected still needs a result message.

    On timeouts: Python cannot kill a stuck thread. After a timeout we hand the model
    an ERROR result and move on, leaving that thread to finish on its own. That is a
    trade-off, not an oversight — real isolation would need a subprocess.
    """
    cap = state.max_tool_calls_per_step
    if cap > 0 and len(calls) > cap:
        emit("throttled", requested=len(calls), cap=cap, step=state.step)

    throttled_msg = (
        f"ERROR: this turn's tool-call limit ({cap}) is reached, so this call was not "
        "executed. Read the results you already have; if it is still needed, send it "
        "again next turn (ideally merged into fewer queries)."
    )
    results: dict[int, str] = {}
    reasons: dict[int, str] = {}
    to_run: list[tuple[int, ToolCall]] = []
    for i, call in enumerate(calls):
        if cap > 0 and i >= cap:
            results[i], reasons[i] = throttled_msg, "throttled"
            continue
        # A project hook is checked first — cheaper than the approval gate (a local
        # Python call, not a possibly-interactive prompt) and parallel to it, not a
        # replacement: a hook can block a call the approval gate would have allowed,
        # and vice versa. It needs parsed arguments, unlike the raw JSON string every
        # other path here works with; a bad parse falls back to {} rather than
        # duplicating tools.execute()'s own error message.
        try:
            parsed_args = json.loads(call.arguments)
        except (json.JSONDecodeError, TypeError):
            parsed_args = {}
        try:
            veto = hooks_mod.veto(call.name, parsed_args)
        except Exception as e:  # a broken hook must not sink the run
            veto = None
            emit("hook_error", name=call.name, point="before_tool_call", error=f"{type(e).__name__}: {e}")
        if veto is not None:
            results[i], reasons[i] = veto, "hook_vetoed"
            emit("hook_vetoed", name=call.name, step=state.step)
            continue
        # The approval check must happen **before** the thread pool and serially —
        # it either asks a human or denies outright.
        spec = tools_mod.REGISTRY.get(call.name)
        if spec is not None and spec.requires_approval:
            emit("approval_required", name=call.name, arguments=call.arguments, step=state.step)
            if not approve(call, spec):
                results[i], reasons[i] = DENIED, "denied"
                emit("denied", name=call.name, step=state.step)
                continue
            emit("approved", name=call.name, step=state.step)
        to_run.append((i, call))

    if to_run:
        # The time brake reaches tools too: if less time remains than the per-call
        # timeout, the remaining time wins. A tool may also declare its own limit —
        # a subagent legitimately runs longer than a page fetch.
        left = state.time_left()
        with ThreadPoolExecutor(max_workers=len(to_run)) as pool:
            futures = {}
            limits = {}
            for i, call in to_run:
                spec = tools_mod.REGISTRY.get(call.name)
                limit = (spec.timeout if spec and spec.timeout else tool_timeout)
                if left is not None:
                    limit = min(limit, max(1.0, left))
                emit("tool_call", name=call.name, arguments=call.arguments, step=state.step)
                future = pool.submit(tools_mod.execute, call.name, call.arguments)
                futures[future] = i
                # An absolute deadline, so waiting on them one after another does not
                # add the timeouts together.
                limits[i] = (time.monotonic() + limit, limit)
            for future, i in futures.items():
                deadline_at, limit = limits[i]
                try:
                    results[i] = future.result(timeout=max(0.0, deadline_at - time.monotonic()))
                except FuturesTimeout:
                    results[i] = (
                        f"ERROR: the tool did not return within {limit:.0f}s and the "
                        "call was abandoned. This does **not** mean the operation "
                        "failed or the information does not exist; retry with a "
                        "smaller request."
                    )
                    future.cancel()
                except Exception as e:  # execute() already catches; this is the net
                    results[i] = f"ERROR: {type(e).__name__}: {e}"

    for i, call in enumerate(calls):  # strictly in the original order
        result = results[i]
        executed = i not in reasons
        if executed:
            emit("tool_result", name=call.name, result=result, step=state.step)
            try:
                parsed_args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError):
                parsed_args = {}
            try:
                result = hooks_mod.rewrite(call.name, parsed_args, result)
            except Exception as e:  # a broken hook must not sink the run
                emit("hook_error", name=call.name, point="after_tool_result", error=f"{type(e).__name__}: {e}")
            # Externalize: a big result goes to disk and the context keeps an excerpt
            # plus the path (the model can read it back with read_file). The trace
            # records the trimmed version — it represents what the model actually
            # saw; the full text lives on disk.
            spec = tools_mod.REGISTRY.get(call.name)
            if (
                run_dir is not None
                and len(result) > EXTERNALIZE_OVER
                and (spec is None or spec.externalize)
            ):
                full_len = len(result)
                result = ctx.externalize(result, run_dir, state.step, i, call.name)
                emit("externalized", name=call.name, chars=full_len, step=state.step)
        state.trace.append(
            ToolTrace(
                step=state.step,
                name=call.name,
                arguments=call.arguments,
                result=result,
                executed=executed,
                skip_reason=reasons.get(i, ""),
            )
        )
        state.messages.append(result_item(model, call, result))


def run(
    goal: str,
    model: Model,
    memory: Memory | None = None,
    max_steps: int = 8,
    budget: float = 0.05,
    max_tool_calls_per_step: int = 3,
    time_budget: float | None = 600.0,  # default 10 minutes; None = unlimited
    tool_timeout: float = 30.0,  # per tool call, in seconds
    context_limit: int = 30_000,  # compact once the context exceeds this many tokens
    run_dir: str | pathlib.Path | None = None,  # persistence + externalization dir
    resume: AgentState | None = None,  # continue from a previously saved state
    plan: bool = False,  # decompose the goal into a checklist first (one extra call)
    reflect: bool = False,  # write an experience/lesson note after a qualifying run
    skills: str | pathlib.Path | None = None,  # directory of skills; None = none
    hooks: str | pathlib.Path | None = None,  # project hooks.py; None = none
    subagents: bool = False,  # offer the delegate tool (a child run with its own context)
    subagent_max_steps: int = 4,
    exclude_tools: list[str] | None = None,  # names the model must not see this run
    approve: Callable[[ToolCall, Any], bool] = deny_all,  # approval policy
    today: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> AgentState:
    """Run one task and return the final state (answer, trace, cost, elapsed time).

    time_budget is the **wall-clock** ceiling in seconds, 600 by default, None for
    unlimited. It complements the money brake: dollars measure model compute, time
    measures human waiting — search throttling, backoff retries and slow networks
    burn time without burning money, and only the time brake catches those.

    Note: time is only checked **between turns**, so a single wedged tool call can
    still overshoot. That is what tool_timeout is for.
    """
    memory = memory or NullMemory()
    tools_mod.bind_memory(memory)  # let the remember tool write into this memory

    system = SYSTEM_PROMPT.format(
        max_tool_calls=max_tool_calls_per_step,
        today=today or date.today().isoformat(),  # the model has no idea what day it is
    )
    if recalled := memory.recall():
        system += f"\n\n{recalled}"

    # Skills put only their metadata here. The bodies stay out until asked for, which is
    # the whole point: this block is in every request, a body is in one.
    available_skills = skills_mod.discover(skills) if skills else []
    if catalog := skills_mod.catalog(available_skills):
        system += f"\n\n{catalog}"

    # run_dir=None means "do not persist and do not externalize" — what evals and
    # unit tests want, so they leave nothing behind in the repo.
    run_dir = pathlib.Path(run_dir) if run_dir else None

    def emit(event: str, **data: Any) -> None:
        if on_event:
            on_event(event, data)

    if resume is not None:
        # Resume: messages, steps, spend and elapsed time all carry over, and above
        # all the system message is **not rebuilt** — rebuilding changes the context
        # prefix and voids every prompt-cache entry earned so far.
        state = resume
        started_at = clock() - state.elapsed  # splice the already-spent time back on
    else:
        state = AgentState(
            goal=goal,
            max_steps=max_steps,
            remaining_budget=budget,
            max_tool_calls_per_step=max_tool_calls_per_step,
            time_budget=time_budget,
        )
        started_at = clock()
        state.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": goal},
        ]
        if plan:
            # One extra model call, before any tool exists, to write down what the
            # request actually asks for. Cheap insurance against finishing half a task.
            state.todo = plan_mod.decompose(goal, model)
            if state.todo:
                emit("planned", items=[t.text for t in state.todo])
    tools_mod.bind_todo(state.todo)  # let update_todo tick items off
    state.status = "running"

    # Prompt-cache grouping key, derived from the **context prefix**, so separate
    # runs with the same configuration can reuse each other's cache. The caching is
    # OpenAI's job; ours is only to keep the prefix stable (status lines are always
    # appended at the end) and to say which requests belong together. Note that a
    # prefix under ~1024 tokens never enters the cache at all.
    set_key = getattr(model, "set_cache_key", None)
    if set_key:
        set_key("teacup-agent-" + hashlib.sha256(state.messages[0]["content"].encode()).hexdigest()[:16])

    if available_skills:
        skills_mod.enable(available_skills, state)
        emit("skills", names=[s.name for s in available_skills])

    if hooks:
        # Loaded and torn down the same in-process, no-external-resource way skills
        # are: a project's hooks.py is source code with no connection to hold open,
        # unlike MCP/A2A's hubs.
        hooks_mod.load(hooks)
        emit("hooks", path=str(hooks))

    if subagents:
        # Registered per run, not globally: an unbound tool schema would sit in the
        # context prefix of every request for a capability the run cannot use.
        subagent_mod.enable(
            state,
            model,
            max_steps=subagent_max_steps,
            approve=approve,
            run_dir=run_dir,
            tool_timeout=tool_timeout,
            on_event=on_event,
        )

    hidden = set(exclude_tools or [])
    specs = [s for s in tools_mod.specs() if s["function"]["name"] not in hidden]

    try:
        return _loop(
            state, model, specs, emit, clock, started_at, context_limit,
            tool_timeout, run_dir, approve, memory, reflect,
        )
    finally:
        if subagents:
            subagent_mod.disable()
        if hooks:
            hooks_mod.disable()
        if available_skills:
            skills_mod.disable()


def _loop(
    state: AgentState,
    model: Model,
    specs: list[dict[str, Any]],
    emit: Callable[..., None],
    clock: Callable[[], float],
    started_at: float,
    context_limit: int,
    tool_timeout: float,
    run_dir: pathlib.Path | None,
    approve: Callable[[ToolCall, Any], bool],
    memory: Memory,
    reflect: bool,
) -> AgentState:
    """The loop itself. Split out only so run() can guarantee the teardown above."""
    while True:
        # ---- guards: steps / budget / time ---------------------------------
        state.elapsed = clock() - started_at
        if not state.can_continue():
            state.status = state.stop_reason()
            emit("stopped", reason=state.status)
            if state.step > 0:  # at least one turn ran, so do not leave empty-handed
                finalize(state, model, emit)
            break

        state.step += 1

        # ---- 0a. compact first if the context is over the limit -------------
        if state.context_tokens > context_limit:
            try:
                saved = ctx.compact(state, model, context_limit)
            except Exception as e:  # a failed compaction must not sink the task
                emit("error", message=f"compaction failed: {type(e).__name__}: {e}")
                saved = 0
            if saved:
                state.context_tokens = ctx.messages_tokens(state.messages)
                emit("compacted", saved_tokens=saved, now=state.context_tokens, step=state.step)

        # ---- 0. tell the model where it stands ------------------------------
        # It cannot choose between "dig further" and "wrap up" without knowing how
        # many steps and dollars are left.
        state.messages.append(status_note(state))

        # ---- 1. ask the model ----------------------------------------------
        # No tools on the final turn: wording can be ignored, an empty tool list
        # cannot.
        available = [] if state.step >= state.max_steps else specs
        try:
            reply = complete_with_retries(model, state.messages, available, emit)
        except Exception as e:  # external failure — record it instead of faking success
            state.status = "error"
            state.answer = f"model call failed: {type(e).__name__}: {e}"
            emit("error", message=state.answer)
            break

        state.charge(reply.cost)
        state.input_tokens_total += reply.input_tokens
        state.cached_tokens_total += reply.cached_tokens
        # Prefer the real token count; estimate when there is none (scripted model).
        state.context_tokens = reply.input_tokens or ctx.messages_tokens(state.messages)

        # ---- 2. write this turn's output back into the state (trap 1) -------
        # extend, not append: one Responses turn can produce several entries (a
        # reasoning item plus function_call items) and dropping one loses the
        # reasoning state.
        state.messages.extend(reply.items)

        # ---- 3. no tool calls = task complete (trap 3) ----------------------
        if not reply.tool_calls:
            # ...unless the checklist says otherwise. A model that "finishes" with an
            # untouched action item is the failure this check exists for: it gets one
            # push-back, once, and then the answer stands either way.
            outstanding = plan_mod.pending(state.todo)
            if outstanding and not state.completion_checked and state.step < state.max_steps:
                state.completion_checked = True
                state.messages.append(
                    {
                        "role": "system",
                        "content": COMPLETION_CHECK.format(
                            pending="\n".join(f"- {t.text}" for t in outstanding)
                        ),
                    }
                )
                emit("completion_check", pending=[t.text for t in outstanding])
                continue

            state.answer = reply.text
            state.status = "done"
            emit("answer", text=reply.text, step=state.step)
            break

        # ---- 4. run every tool call in parallel, refill in order (trap 2) ---
        execute_calls(state, reply.tool_calls, model, emit, tool_timeout, run_dir, approve)

        # ---- 5. persist: save every step, or there is nothing to resume from -
        # elapsed uses the value measured at the top of this turn, so we do not ask
        # the clock twice (one less source of non-determinism).
        if run_dir is not None:
            persist.save(state, run_dir)
        # Back to the top of the loop, handing the tool results to the model — this
        # is the difference between an agent and a single function call.

    state.elapsed = clock() - started_at
    if state.status != "done" and not state.answer:
        state.answer = f"(no final answer; stopped because: {state.status})"
    if reflect:
        # Before the final persist, so the reflection call's own cost is captured in
        # the state that gets written to disk.
        written = reflect_mod.maybe_record(state, model, memory)
        if written:
            emit("reflected", kinds=written)
    if run_dir is not None:  # the final state matters most for review and evals
        persist.save(state, run_dir)
        emit("saved", path=str(run_dir / persist.FILENAME))
    return state
