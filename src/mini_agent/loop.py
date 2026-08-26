"""Control Loop —— 把 Model / State / Tools / Memory 串起来的那 40 行。

一句话本质：

    LLM → tool call → tool result → LLM → ...

三个容易踩的坑，代码里都标了注释：
1. 带 tool_calls 的 assistant 消息必须**先** append，再 append 工具结果；
2. 一轮里可能有**多个** tool_call，每一个 id 都必须有对应的 role=tool 消息；
3. 终止条件不是玄学的 check_completion()，而是「模型不再要工具」+ 步数/预算上限。
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Callable

from mini_agent import tools as tools_mod
from mini_agent.memory import Memory, NullMemory
from mini_agent.model import Model, ToolCall, chat_tool_result
from mini_agent.state import AgentState, ToolTrace

SYSTEM_PROMPT = """你是一个会使用工具的自主 Agent，运行在**无人值守**的自动模式下。

今天是 {today}。

关于时效性（重要）：
- 你的训练数据截止时间早于今天，世界在那之后继续发生了变化。
- 检索结果与你的记忆冲突时，**以检索结果为准**。不要因为「这个数字比我印象中大得多」
  就判定信息不实 —— 那通常只说明你的记忆过期了。
- 正确做法是来源分级 + 交叉验证，而不是整体拒绝：
  一手来源（公司官网/公告/监管文件）> 主流媒体 > 二手聚合与 SEO 站点；
  多个独立来源一致就可以采信并标注来源，只有低质来源支持则标注「存疑」。
- 但仍然不许凭空编造具体数字。没查到就说没查到。

关于怎么提问（同样重要）：
- 检索词里的时间锚点要用**今天**来算，不要用你记忆里的年份。
  比如「最近半年」指的是今天往前推六个月这个区间，不是你印象中的某一年。
- 也不要把记忆里的具体事件（某轮融资金额、某个模型版本号）当作检索词的前提 ——
  那些多半已经过时，会把你带回旧新闻。先用宽泛的近期关键词摸清现状，再针对性深挖。

自动模式的硬约束：
- **没有人会回答你的提问。** 不要请求许可、不要问「是否需要我继续」、不要把待办清单
  交回给用户 —— 你自己想做的下一步，直接做。
- 每轮开头会告诉你剩余步数、预算和时间。**任何一项**见底都要立刻收尾，别只盯着最宽松的那个。
- 绝不能空手而归：即使信息不全，也要给出「目前能得出的最佳结论 + 置信度 + 待核实项」，
  而不是一份行动计划。

工作方式：
- 需要外部信息时就调用工具，不要凭空编造；一轮可以同时发起多个工具调用，但每轮最多执行 {max_tool_calls} 个，
  超出的会被拒绝执行并要求你下一轮重发 —— 所以请先挑最关键的几个。
- 工具返回以 ERROR: 开头时，说明调用有问题，请修正参数后重试，不要放弃。
- 不再调用工具 = 你认为任务已完成，这是循环的终止信号 —— 别用它来提问。
- 遇到值得长期保留的事实（用户偏好、稳定结论），用 remember 工具记下来。

请保持回答简洁，关键结论标注来源与置信度。"""


def status_note(state: AgentState) -> dict[str, Any]:
    """每轮开头告诉模型它的资源状况。

    为什么不写进 system prompt：那样会让上下文前缀每轮都变，
    prompt caching（roadmap #2）就废了。追加在末尾则不影响前缀。
    """
    left = state.max_steps - state.step  # 本轮之后还剩几轮
    time_left = state.time_left()
    # 时间和步数谁更紧张就听谁的 —— 第三次实测里模型被「预算还剩 91%」误导，
    # 忽略了步数已经见底。多个刹车并存时，必须把**最紧的那个**摆到台面上。
    tight_on_time = time_left is not None and time_left <= max(15.0, (state.time_budget or 0) * 0.25)

    if left <= 0:
        urgency = (
            "⚠ **这是最后一轮**，工具已被收走。立刻基于已有信息给出最终结论 —— "
            "包括已确认的结论、置信度、以及未能核实的项。绝不能空手而归。"
        )
    elif left <= 2 or tight_on_time:
        why = f"只剩 {left} 轮" if left <= 2 else f"只剩 {time_left:.0f} 秒"
        urgency = f"⚠ {why}，请开始收尾：最多再查一轮，然后必须给出结论。"
    else:
        urgency = "资源充足就继续查证，不足就立刻给出最终结论。"

    resources = f"第 {state.step}/{state.max_steps} 轮；预算剩余 ${state.remaining_budget:.4f}"
    if time_left is not None:
        resources += f"；剩余时间 {time_left:.0f} 秒"
    return {"role": "system", "content": f"[运行状态] {resources}。{urgency}"}


def finalize(state: AgentState, model: Model, emit: Callable[..., None]) -> None:
    """强制收尾轮：资源耗尽时，不给工具再问一次，把已有信息榨成结论。

    第三次实测运行的教训：模型把 8 轮全用在检索上，一个字的结论都没留下 ——
    十次检索的钱全白花。刹车不能只是「停」，还得把车上的东西卸下来。
    """
    state.messages.append(
        {
            "role": "system",
            "content": (
                f"[强制收尾] 资源已耗尽（{state.status}），工具不再可用。"
                "立刻基于已经获得的信息给出最终结论：已确认的结论 + 置信度 + 未能核实的项。"
            ),
        }
    )
    try:
        reply = model.complete(state.messages, [])  # 空工具清单 = 只能说话
    except Exception as e:
        emit("error", message=f"强制收尾失败：{type(e).__name__}: {e}")
        return
    state.charge(reply.cost)
    state.messages.extend(reply.items)
    if not reply.text:  # 没榨出东西就别声称抢救成功
        return
    state.answer = reply.text
    state.salvaged = True
    emit("salvaged", text=reply.text)


def result_item(model: Model, call: ToolCall, result: str) -> dict[str, Any]:
    """问模型后端要工具结果的形状；没实现就退回 Chat Completions 的老形状。"""
    fn = getattr(model, "tool_result_item", None)
    return fn(call, result) if fn else chat_tool_result(call, result)


def run(
    goal: str,
    model: Model,
    memory: Memory | None = None,
    max_steps: int = 8,
    budget: float = 0.05,
    max_tool_calls_per_step: int = 3,
    time_budget: float | None = 600.0,  # 默认 10 分钟；None = 不限
    today: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> AgentState:
    """跑一次任务，返回终态（含答案、留痕、花费、耗时）。

    time_budget 是**墙上时间**上限（秒），默认 600（10 分钟），None = 不限。它和预算刹车互补：
    钱衡量的是模型算力，时间衡量的是人的等待 —— 检索限流、退避重试、网络慢
    都只烧时间不烧钱，只有时间刹车拦得住。

    注意：时间只在**两轮之间**检查，单个工具调用卡死仍会超时（那需要给工具本身
    加超时，属于 roadmap #5 并行执行时一起做的事）。
    """
    memory = memory or NullMemory()
    tools_mod.bind_memory(memory)  # 让 remember 工具能写到这份记忆里

    system = SYSTEM_PROMPT.format(
        max_tool_calls=max_tool_calls_per_step,
        today=today or date.today().isoformat(),  # 模型不知道今天几号，必须告诉它
    )
    if recalled := memory.recall():
        system += f"\n\n{recalled}"

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
    state.status = "running"

    def emit(event: str, **data: Any) -> None:
        if on_event:
            on_event(event, data)

    specs = tools_mod.specs()

    while True:
        # ---- 守卫：步数 / 预算 / 时间 --------------------------------------
        state.elapsed = clock() - started_at
        if not state.can_continue():
            state.status = state.stop_reason()
            emit("stopped", reason=state.status)
            if state.step > 0:  # 跑过至少一轮，就别空手而归
                finalize(state, model, emit)
            break

        state.step += 1

        # ---- 0. 把资源状况告诉模型 -----------------------------------------
        # 它得知道自己还剩多少步、多少钱，才谈得上「继续挖还是收尾」。
        state.messages.append(status_note(state))

        # ---- 1. 问模型 ---------------------------------------------------
        # 最后一轮不给工具：措辞可以被无视，空的工具清单不能。
        available = [] if state.step >= state.max_steps else specs
        try:
            reply = model.complete(state.messages, available)
        except Exception as e:  # 网络/额度等外部故障，如实记录而不是假装成功
            state.status = "error"
            state.answer = f"模型调用失败：{type(e).__name__}: {e}"
            emit("error", message=state.answer)
            break

        state.charge(reply.cost)

        # ---- 2. 先把模型这轮的输出写回状态--------------------------
        # 用 extend 而不是 append：Responses API 一轮可能产出多个条目
        # （reasoning 项 + 多个 function_call 项），少带一个就丢了推理状态。
        state.messages.extend(reply.items)

        # ---- 3. 没有工具调用 = 任务完成----------------------------
        if not reply.tool_calls:
            state.answer = reply.text
            state.status = "done"
            emit("answer", text=reply.text, step=state.step)
            break

        # ---- 4. 每个 tool_call 都要执行并回填----------------------
        # 限流：只执行前 N 个，但**超出的那些也必须回一条 tool 消息**。
        # 少回一条，下一轮 API 就会因为 tool_call_id 没有对应结果而 400 ——
        # 所以这里是「拒绝执行」，不是「忽略」。
        cap = state.max_tool_calls_per_step
        if cap > 0 and len(reply.tool_calls) > cap:
            emit("throttled", requested=len(reply.tool_calls), cap=cap, step=state.step)

        for index, call in enumerate(reply.tool_calls):
            throttled = cap > 0 and index >= cap
            if throttled:
                result = (
                    f"ERROR: 本轮工具调用数已达上限（{cap} 个），该调用未执行。"
                    "请先看已返回的结果，若仍有必要，下一轮再发起（可合并成更少的查询）。"
                )
            else:
                emit("tool_call", name=call.name, arguments=call.arguments, step=state.step)
                result = tools_mod.execute(call.name, call.arguments)
                emit("tool_result", name=call.name, result=result, step=state.step)
            state.trace.append(
                ToolTrace(
                    step=state.step,
                    name=call.name,
                    arguments=call.arguments,
                    result=result,
                    executed=not throttled,
                )
            )
            # 工具结果的形状由模型后端决定：Chat 是 role="tool" 消息，
            # Responses 是 {"type": "function_call_output", ...}。
            state.messages.append(result_item(model, call, result))
        # 回到循环顶部，把工具结果一起交回模型 —— 这就是「Agent」和「一次函数调用」的区别

    state.elapsed = clock() - started_at
    if state.status != "done" and not state.answer:
        state.answer = f"（未得出最终答案，停止原因：{state.status}）"
    return state
