# mini-agent 升级路线图

**基线评估（2026-08-25）**：内核不过时，工程层大约停在 2023 年底 / 2024 年初。
**进度**：#1 已完成，下一站 #4（上下文压缩）或 #5（并行工具）。

`LLM → tool call → tool result → LLM` 这个循环在 2026 年依然是所有 agent 的内核，
[`loop.py`](../src/mini_agent/loop.py) 没有一行是「老技术」。差的是外面那一整层生产工程。

这份路线图按**做完之后收益 / 改动成本**排序。每条都写清了：改哪里、怎么算做完、参考资料。
不必按顺序全做 —— 作为学习骨架，它现在这样恰好；每加一层，那 40 行循环就更难看清一点。

---

## P0 · 改一个类就能拿到的收益

### 1. 换用 Responses API ✅ 已完成（2026-08-25）

**现状**：[`model.py`](../src/mini_agent/model.py) 的 `OpenAIModel` 走 Chat Completions。
每轮把模型的推理过程丢掉，只把最终 message 塞回 `messages`。

**代价**：对 gpt-5 这类推理模型，跨工具调用的推理状态无法保留。OpenAI 官方迁移文档称，
同样的 prompt 下 Responses API 的 SWE-bench 成绩高约 3%；且新能力（内置 web_search、
code_interpreter 等 hosted tool）只在 Responses 上落地。

**怎么改**：只动 `OpenAIModel` 一个类，`loop.py` / `state.py` 一行不用改 —— 这正是当初
把模型抽象成 `complete(messages, tools) -> Reply` 的回报。要点：

- 工具定义是**扁平**的：`{"type": "function", "name": ..., "parameters": ...}`，
  不像 Chat Completions 套一层 `function`。
- 输出是 `response.output` 列表，工具调用项 `type == "function_call"`，用 `call_id`。
- 工具结果回传的形状是 `{"type": "function_call_output", "call_id": ..., "output": ...}`，
  不是 `role="tool"` 消息。
- 用 `previous_response_id` 串联上下文，就能保住推理状态。

**注意**：这会让「归一化的 Reply」多承担一点 —— `Reply.message` 目前假设是一条 chat 消息。
可能需要把它抽象成「要 append 回上下文的东西」。这是这条改动唯一的设计成本。

**验收**：
- `tests/test_model.py` 复制一份成 `test_responses_model.py`，用假 client 验证解析；
- 8 条 evals 全绿（它们只依赖 `Model` 接口，不该受影响）；
- 两个后端可通过 `--api {chat,responses}` 切换，同一任务跑通。

**实际落地情况**：

- 新增 `ResponsesModel`（[`model.py`](../src/mini_agent/model.py)），与 `OpenAIModel` 并存，
  用 `--api {responses,chat}` 切换，**默认 responses**。
- `loop.py` 的循环结构没变，但抽了两处形状差异出去：
  - `Reply.message` → `Reply.items`（一轮可能产出多条：reasoning + 多个 function_call），
    循环里改成 `state.messages.extend(reply.items)`；
  - 工具结果的形状交给模型后端决定（`Model.tool_result_item`），
    Chat 是 `role="tool"`，Responses 是 `function_call_output`。
- `evals.py` 的顺序不变量 `tool_results_follow_their_call()` 现在两种形状都认。
- 新增 `tests/test_responses_model.py`：假 client 验证解析 + 整条循环跑 Responses 形状。

**实测证据**（gpt-5-mini，同一任务两个后端各跑一次，共花费约 $0.002）：

```
### responses                    ### chat
  0. system                        0. system
  1. user                          1. user
  2. reasoning      ← 推理项       2. assistant (tool_calls=1)
  3. function_call                 3. tool
  4. function_call_output          4. assistant
  5. message
```

第 2 条 `reasoning` 会随下一轮 input 一起发回去 —— 这就是收益的来源，Chat 那边没有对应物。

**回头看**：预判的设计成本（`Reply.message` 的形状假设）确实是唯一的改动点，
`loop.py` 的控制流一行没动。那层 `Model` 抽象站住了。

**参考**：<https://developers.openai.com/api/docs/guides/migrate-to-responses>

---

### 2. Prompt caching

**现状**：长 system prompt + 越来越长的历史，每轮全价重发。

**怎么改**：让消息前缀保持稳定（system prompt 不要每轮拼入变动内容 —— 注意现在的
`memory.recall()` 是拼进 system 的，只要记忆没变就仍然稳定，别改成每轮拼时间戳之类）。
再在 `Reply.cost` 里区分 cached / uncached token 计价。

**验收**：同一任务连跑两次，第二次的 `remaining_budget` 消耗明显更低，且 snapshot 能报出缓存命中。

---

### 3. 重试、退避、超时

**现状**：[`loop.py`](../src/mini_agent/loop.py) 里模型调用一抛异常就 `status="error"` 整轮结束。
一次 429 或网络抖动就前功尽弃。

**怎么改**：在 `OpenAIModel.complete` 外面包一层指数退避（429/5xx/超时重试，4xx 不重试），
重试次数计入 state 便于观测。注意别和 `max_steps` 混淆：**重试不是一个 step**。

**验收**：新增单测，假 client 前两次抛 429、第三次成功 → 循环正常完成，`step` 不变。

---

## P1 · 决定「能不能跑长任务」

### 4. 上下文工程（压缩 / 外置）

**现状**：`state.messages` 是个无限增长的 list。跑 20-30 轮必然撑爆 context window，
而且越长越贵、模型注意力越散。

**怎么改**，两件事一起做：

- **压缩（compaction）**：token 数超过阈值时，把早期的工具结果摘要成一段，替换原始条目。
  保留最近 N 轮原文 + 目标 + 关键结论。
- **外置**：大块工具结果（网页正文、文件内容）写进 `runs/<id>/` 目录，上下文里只留
  「摘要 + 文件路径」，模型需要细节时用 `read_file` 取回。这比什么压缩算法都有效。

**验收**：造一个需要 15+ 轮的任务，跑完不爆 context；压缩前后 `state.snapshot()` 能报出
token 数下降；关键结论没有在压缩中丢失（这条要用 trajectory eval 兜，见 #7）。

---

### 5. 并行执行工具

**现状**：模型一轮并行发起 3 个搜索，[`loop.py`](../src/mini_agent/loop.py) 一个个串行跑。
接了真实联网检索后，这里就是最明显的墙上时间浪费。

**怎么改**：把工具执行改成 `asyncio.gather`（或线程池，工具多是同步 IO）。

**必须守住的不变量**：结果**回填顺序要和 `tool_calls` 顺序一致**，每个 `tool_call_id`
都要有对应结果 —— 也就是 `evals.py` 里 `tool_results_follow_their_call()` 锁的那条。
并发是最容易把这条打破的改动，改完先看这条用例。

**验收**：3 个各 sleep 1s 的假工具，一轮总耗时 ≈ 1s 而不是 3s；8 条 evals 全绿。

---

### 6. 人机确认门（HITL）与权限分级

**现状**：工具全是只读或无害的，所以没有审批环节。一旦加 `send_email`、`shell`、
写文件，这就是个能造成真实损失的洞。

**怎么改**：给 `Tool` 加一个 `requires_approval: bool`（或危险等级）。执行前 emit 一个
`approval_required` 事件，CLI 侧交互确认；非交互模式下默认拒绝并把「已拒绝」作为工具结果回传
（模型可以据此换个做法）。

**验收**：标记为危险的工具在自动模式下不会被执行，且循环不崩、消息协议依然完整。

---

## P2 · 从「能跑」到「可信」

### 7. Trajectory eval（轨迹评测）

**现状**：[`evals.py`](../src/mini_agent/evals.py) 的 8 条锁的是「循环协议对不对」——
必要，但属于最低档，等价于单元测试。

**2026 的 agent eval 长什么样**：评的是**整条轨迹**，不只是最终答案。同样一个正确答案，
3 步干净走到 vs 12 步瞎撞碰运气，不是一个分。常见维度：结果正确性、工具调用是否正确
（选对工具 + 参数对）、效率（步数 / token / 花费）、路径安全性，评分用 LLM-as-judge rubric。

**怎么改**：
- 造一个小任务集（10-20 条），每条给出期望结论要点和「合理的工具使用路径」；
- 用 `ScriptedModel` 覆盖协议层，用真实模型跑轨迹层（这部分要花钱，单独一个命令）；
- 加一个 judge：把 `state.trace` 和最终答案交给模型按 rubric 打分；
- 结果落盘成 `runs/`，能对比两次改动之间的回归。

**验收**：改一版 system prompt，能用数字说出「变好还是变坏」，而不是靠感觉。

**参考**：<https://qaskills.sh/blog/agent-trajectory-evaluation-guide-2026>

---

### 8. 可观测性与可恢复

**现状**：跑完就没了。崩了从头再来，出了问题只能看终端回滚。

**怎么改**：每次 run 落盘（`runs/<timestamp>/`：messages、trace、花费、终态），
加结构化日志 / OpenTelemetry span；再往前一步是 checkpoint 恢复 —— `AgentState` 本来就是
个 dataclass，序列化成 JSON 就能从中断处续跑。

**验收**：跑到一半 Ctrl-C，能从 checkpoint 续上，不重复已完成的工具调用。

---

## P3 · 打开生态

### 9. 接入 MCP

**现状**：工具是硬编码的 Python 函数，加一个工具就得改一次 [`tools.py`](../src/mini_agent/tools.py)。

**为什么值得**：MCP 已是 2026 工具接入的事实标准，接上之后现成的 server（文件系统、
数据库、GitHub、浏览器……）拿来就用，不用自己一个个写。2026 roadmap 的重点是传输层扩展性、
agent 间通信、治理与企业就绪。

**怎么改**：写一个 `mcp_tools.py`，从 MCP server 拉取工具清单，转成现有的 `Tool` 结构塞进
`REGISTRY` —— 因为 `tools.execute()` 的接口是「名字 + JSON 字符串参数」，和 MCP 的调用形状天然对齐，
`loop.py` 依然不用改。

**验收**：启动时连上一个本地 MCP server，其工具出现在 `tools.specs()` 里并能被模型正常调用。

**参考**：<https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/>

---

### 10. Subagent / 上下文隔离

**现状**：所有事情挤在一条上下文里。

**怎么改**：orchestrator-worker：主 agent 把子任务派给独立上下文的子 agent，子 agent 只把
**结论**回传（而不是把它读过的所有网页都带回主上下文）。这同时也是最有效的上下文压缩手段。

**验收**：一个需要读 5 个来源的任务，主上下文的 token 数显著低于单上下文版本，结论质量不降。

---

### 11. 更好的检索后端

**现状**：`search_web` 用 ddgs 抓 DuckDuckGo，免费免 key，但质量和稳定性都一般。

**选项**：模型侧内置的 hosted web search（走 Responses API，见 #1）、或专门的 agentic search API。
接口不用变，`search_web` 内部换后端即可 —— 三种模式（auto/web/offline）的结构已经预留好了。

---

## 刻意不做的事

- **不引入 agent 框架**（LangGraph 之类）。这个仓库的价值就在于那 40 行循环是**你自己的**，
  一眼能看完。套上框架，学习价值立刻归零。
- **不做多租户 / 服务化 / Web UI**。那是另一个项目的事。
- **不追求工具数量**。四个工具足够演示所有机制；真要工具，走 #9 的 MCP。

---

## 一句话优先级

想让它更**聪明** → 做 #1（Responses API）。
想让它能干**更久的活** → 做 #4（上下文压缩）。
想让它**更快** → 做 #5（并行工具）。
想让它**可信** → 做 #7（trajectory eval）。
想让它**有更多能力** → 做 #9（MCP）。
