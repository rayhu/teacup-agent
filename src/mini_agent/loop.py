"""Control Loop —— 把 Model / State / Tools / Memory 串起来的那 40 行。

一句话本质：

    LLM → tool call → tool result → LLM → ...

三个容易踩的坑，代码里都标了注释：
1. 带 tool_calls 的 assistant 消息必须**先** append，再 append 工具结果；
2. 一轮里可能有**多个** tool_call，每一个 id 都必须有对应的 role=tool 消息；
3. 终止条件不是玄学的 check_completion()，而是「模型不再要工具」+ 步数/预算上限。
"""

from __future__ import annotations

from typing import Any, Callable

from mini_agent import tools as tools_mod
from mini_agent.memory import Memory, NullMemory
from mini_agent.model import Model, ToolCall, chat_tool_result
from mini_agent.state import AgentState, ToolTrace

SYSTEM_PROMPT = """你是一个会使用工具的助手。

工作方式：
- 需要外部信息时就调用工具，不要凭空编造；一轮可以同时发起多个工具调用，但每轮最多执行 {max_tool_calls} 个，
  超出的会被拒绝执行并要求你下一轮重发 —— 所以请先挑最关键的几个。
- 工具返回以 ERROR: 开头时，说明调用有问题，请修正参数后重试，不要放弃。
- 信息够了就直接给出最终答案，不要再调工具 —— 这也是本轮任务结束的信号。
- 遇到值得长期保留的事实（用户偏好、稳定结论），用 remember 工具记下来。

请保持回答简洁，并说明结论依据。"""


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
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> AgentState:
    """跑一次任务，返回终态（含答案、留痕、花费）。"""
    memory = memory or NullMemory()
    tools_mod.bind_memory(memory)  # 让 remember 工具能写到这份记忆里

    system = SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls_per_step)
    if recalled := memory.recall():
        system += f"\n\n{recalled}"

    state = AgentState(
        goal=goal,
        max_steps=max_steps,
        remaining_budget=budget,
        max_tool_calls_per_step=max_tool_calls_per_step,
    )
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
        # ---- 守卫：步数 / 预算 -------------------------------------------
        if not state.can_continue():
            state.status = state.stop_reason()
            emit("stopped", reason=state.status)
            break

        state.step += 1

        # ---- 1. 问模型 ---------------------------------------------------
        try:
            reply = model.complete(state.messages, specs)
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

    if state.status != "done" and not state.answer:
        state.answer = f"（未得出最终答案，停止原因：{state.status}）"
    return state
