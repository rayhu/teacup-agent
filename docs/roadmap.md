# mini-agent 升级路线图

**基线评估（2026-08-25）**：内核不过时，工程层大约停在 2023 年底 / 2024 年初。
**进度**：#1—#8 全部完成（除 #6 的细粒度权限外）。
另外从三次真实运行的复盘里补了三件 roadmap 上原本没有的事（见文末「实战补丁」）。
下一站 #9（MCP）—— 打开工具生态。

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

### 2. Prompt caching ✅ 已完成（2026-08-26）

**现状**：长 system prompt + 越来越长的历史，每轮全价重发。

**怎么改**：让消息前缀保持稳定（system prompt 不要每轮拼入变动内容 —— 注意现在的
`memory.recall()` 是拼进 system 的，只要记忆没变就仍然稳定，别改成每轮拼时间戳之类）。
再在 `Reply.cost` 里区分 cached / uncached token 计价。

**验收**：同一任务连跑两次，第二次的 `remaining_budget` 消耗明显更低，且 snapshot 能报出缓存命中。

**落地情况**：`PRICES` 改成三元组（输入 / 命中缓存的输入 / 输出），命中部分按十分之一计价；
两个后端都从 usage 里挖出 `cached_tokens`（Chat 在 `prompt_tokens_details`，
Responses 在 `input_tokens_details`）；`snapshot()` 报 `cache_hit`。
另外用 system prompt 的哈希做 `prompt_cache_key`，让同配置的多次运行互相复用缓存。

**实测**：gpt-5-mini 跑 5 轮带真实检索的任务，`cache_hit: 38%`。
**一个坑**：第一次测出来是 0%，查下来是首个请求约 972 token，**没到 OpenAI 约 1024 token 的起caching门槛**。
短任务显示 0% 是正常的，不是 bug。

---

### 3. 重试、退避、超时 ✅ 已完成（2026-08-26）

**现状**：[`loop.py`](../src/mini_agent/loop.py) 里模型调用一抛异常就 `status="error"` 整轮结束。
一次 429 或网络抖动就前功尽弃。

**怎么改**：在 `OpenAIModel.complete` 外面包一层指数退避（429/5xx/超时重试，4xx 不重试），
重试次数计入 state 便于观测。注意别和 `max_steps` 混淆：**重试不是一个 step**。

**验收**：新增单测，假 client 前两次抛 429、第三次成功 → 循环正常完成，`step` 不变。

**已完成的一半**：检索工具侧。实测 DuckDuckGo 连发 4-5 个查询就会限流，而 agent 恰好爱一轮甩好几个。
[`tools.py`](../src/mini_agent/tools.py) 现在有请求间隔（1.5s）+ 退避重试（1s/2s/4s），
并且**重试用尽后明确报 `ERROR:` 而不是静默降级成本地语料的「没有找到」**——
把「检索坏了」伪装成「查无此事」，模型会直接得出「世界上没有这件事」的结论。
**另一半也完成了**：`complete_with_retries()` 对 429/5xx/网络类错误退避重试（1s/2s），
4xx 直接抛出不浪费时间。关键细节：**重试不算一个 step** —— 步数衡量的是「模型做了几次决策」，
一次限流不该消耗 agent 的思考额度。另外单个工具调用也有超时了（`--tool-timeout`，默认 30s）。

---

## P1 · 决定「能不能跑长任务」

### 4. 上下文工程（压缩 / 外置）✅ 已完成（2026-08-26）

**现状**：`state.messages` 是个无限增长的 list。跑 20-30 轮必然撑爆 context window，
而且越长越贵、模型注意力越散。

**怎么改**，两件事一起做：

- **压缩（compaction）**：token 数超过阈值时，把早期的工具结果摘要成一段，替换原始条目。
  保留最近 N 轮原文 + 目标 + 关键结论。
- **外置**：大块工具结果（网页正文、文件内容）写进 `runs/<id>/` 目录，上下文里只留
  「摘要 + 文件路径」，模型需要细节时用 `read_file` 取回。这比什么压缩算法都有效。

**验收**：造一个需要 15+ 轮的任务，跑完不爆 context；压缩前后 `state.snapshot()` 能报出
token 数下降；关键结论没有在压缩中丢失（这条要用 trajectory eval 兜，见 #7）。

**落地情况**：新增 [`context.py`](../src/mini_agent/context.py)。

- 外置：>2000 字符的工具结果写进 `runs/<时间戳>/`，上下文只留 600 字符 + 路径 +
  「用 read_file 取回」。实测真实检索的 2022 字符结果被压到约 900 tokens 上下文，全文在盘上。
- 压缩：超 `--context-limit`（默认 30000）时把早期历史摘要成一条消息。
  保留 system 前缀、原始目标、最近 8 条。
- **最关键的实现细节是切点而不是摘要质量**：`safe_cut_points()` 复用了顺序不变量那套扫描，
  只在「没有悬空调用」处下刀，找不到就不压。拆散一对调用/结果的代价是下一轮直接 400。
- 判断依据优先用模型返回的真实 `usage.input_tokens`（顺手把它从 `Reply` 里暴露出来了），
  拿不到才退回字符估算。
- `state.snapshot()` 新增 `context_tokens` 和 `compactions`。

**遗留**：压缩本身要花一次模型调用的钱，目前每次都全量重摘；分层摘要（摘要的摘要）没做。

---

### 5. 并行执行工具 ✅ 已完成（2026-08-26）

**现状**：模型一轮并行发起 3 个搜索，[`loop.py`](../src/mini_agent/loop.py) 一个个串行跑。
接了真实联网检索后，这里就是最明显的墙上时间浪费。

**怎么改**：把工具执行改成 `asyncio.gather`（或线程池，工具多是同步 IO）。

**必须守住的不变量**：结果**回填顺序要和 `tool_calls` 顺序一致**，每个 `tool_call_id`
都要有对应结果 —— 也就是 `evals.py` 里 `tool_results_follow_their_call()` 锁的那条。
并发是最容易把这条打破的改动，改完先看这条用例。

**验收**：3 个各 sleep 1s 的假工具，一轮总耗时 ≈ 1s 而不是 3s；8 条 evals 全绿。

**落地情况**：`execute_calls()` 用线程池并发，结果按原顺序回填；
`tests/test_parallel.py` 里 3 个 0.3 秒的假工具 <0.6 秒跑完，顺序与不变量都验过。
单次超时也一并做了（`--tool-timeout`，默认 30s），超时返回 ERROR 结果而不是卡死整轮 ——
这就是时间刹车那里欠下的「单个工具卡死」问题。
`tools.py` 的检索限流全局量加了锁（并行下那是竞态）。

**意外发现**：真实检索只提速 1.39x（5.05s → 3.64s），因为瓶颈不是网络，
而是我们自己为了躲 DuckDuckGo 限流加的间隔。实测间隔 1.5s 并行要 8.3s、间隔 0 只要 4.7s。
已折中到 0.5s。**这项优化的天花板由 #11（换检索后端）决定，不由并发度决定** ——
先量再优化，不然会在错的地方使劲。

---

### 6. 人机确认门（HITL）与权限分级 ✅ 已完成（2026-08-26）

**现状**：工具全是只读或无害的，所以没有审批环节。一旦加 `send_email`、`shell`、
写文件，这就是个能造成真实损失的洞。

**怎么改**：给 `Tool` 加一个 `requires_approval: bool`（或危险等级）。执行前 emit 一个
`approval_required` 事件，CLI 侧交互确认；非交互模式下默认拒绝并把「已拒绝」作为工具结果回传
（模型可以据此换个做法）。

**验收**：标记为危险的工具在自动模式下不会被执行，且循环不崩、消息协议依然完整。

**落地情况**：`Tool.requires_approval` + `run(approve=...)` 回调，默认策略 `deny_all`。
CLI `--approve auto|deny|allow`：auto 有终端就问人、**没终端就拒绝**。
新增示例工具 `send_email`（写 outbox.jsonl，不真发）—— 正好是你原始笔记那份工具清单的最后一个，
也是第一个必须加门的：只读工具错了顶多浪费一次调用，这个错了信已经发出去了。

**几个刻意的选择**：
- 默认拒绝而不是默认放行。「没人看着就放行」是最危险的默认值，出事的恰恰是没人看着的那次。
- 批准检查在**提交线程池之前**串行做（它要么问人、要么直接拒绝，不能并发）。
- 被拒绝的调用同样回一条结果消息，内容明确「没有执行，别重发」。
- 只读工具绝不设这个标记：问多了会麻木，麻木了就闭眼点同意，反而更危险。
- 轨迹评测新增 `denied` 与 `retried_after_denial`（被拒后原样重发 = 没读懂拒绝）。

**未完成**：更细的权限分级（按参数判断，比如「只允许发给白名单域名」）。

---

## P2 · 从「能跑」到「可信」

### 7. Trajectory eval（轨迹评测）✅ 已完成（2026-08-26）

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

**落地情况**：新增 [`trajectory.py`](../src/mini_agent/trajectory.py)，输入就是 #8 落盘的
`runs/*/state.json`。两层：

- **机械指标**（零成本、确定性）：步数、工具调用/失败/**重复**、限流、压缩、耗时、token、
  缓存命中、`delivered`（真交付了结论还是空手）、`asks_user_back`（反过来问用户 —— 第一次
  实测失败的形态）、`unsupported_citations`（答案里出现但任何工具结果里都没有的链接）。
- **LLM 评委**：outcome / grounding / efficiency / honesty 各 0-5 + 总评 + 最该改的一点。
  解析不出 JSON 就报错，不假装打分。

**实测里最有意思的一幕**：评委给某次运行的 grounding 打 3 分，说「引用了 4 个来源但只检索了
一次，缺少证据」；而机械检查算出 `unsupported_citations = 0`，4 条链接全在那次检索结果里。
**评委看的是截断到 300 字符的摘要，机械检查看的是全文。**
结论：**先信确定性指标，再听评委的定性判断** —— 评委适合评「好不好」，不适合当事实核查员。

**未完成**：固定任务集 + 跨版本自动回归对比（现在还是一次跑一次评）。

**参考**：<https://qaskills.sh/blog/agent-trajectory-evaluation-guide-2026>

---

### 8. 可观测性与可恢复 ✅ 已完成（2026-08-26）

**现状**：跑完就没了。崩了从头再来，出了问题只能看终端回滚。

**怎么改**：每次 run 落盘（`runs/<timestamp>/`：messages、trace、花费、终态），
加结构化日志 / OpenTelemetry span；再往前一步是 checkpoint 恢复 —— `AgentState` 本来就是
个 dataclass，序列化成 JSON 就能从中断处续跑。

**验收**：跑到一半 Ctrl-C，能从 checkpoint 续上，不重复已完成的工具调用。

**落地情况**：新增 [`persist.py`](../src/mini_agent/persist.py)。每步把整个 `AgentState`
写进 `runs/<时间戳>/state.json`（临时文件 + 改名，避免残档），`--resume` 从那里接着跑。
恢复时**不重建 system 消息**（重建 = prompt cache 作废），命令行上限按「再给这么多」叠加。
`run_dir=None` 表示不落盘 —— 评测和单测走这条路，不往仓库里拉屎。

**顺带逮到一个真 bug**：写恢复用例时发现，强制收尾轮如果模型仍然硬发工具调用，
那些调用**没有结果消息**，消息协议就断了 —— 下一次请求（包括 `--resume`）直接 400。
现在收尾轮会给每个悬空调用补一条「已进入收尾阶段，工具不可用」的结果。
评测里加了一条用例盯着它。这类 bug 只有在真去做「恢复」时才会现形。

**未完成**：OpenTelemetry span 之类的结构化追踪。

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

---

## 实战补丁（来自三次真实运行的复盘）

这几条不在最初的路线图上，是跑真任务跑出来的。共同点：都不是循环写错了，
而是**模型缺少它做决策所需的信息**。

### A. 告诉模型今天几号 ✅

**症状**（第 1 次运行）：让它研究 Anthropic 近半年动态，检索返回了真实的 2026 年新闻，
它却判定「与公开信息量级严重不符，极可能不实」，拒绝采信，最后交回来一份**请示**而不是简报。

**根因**：system prompt 里没有日期。模型拿训练时代的记忆去衡量比自己新的信息，
于是系统性地不信任检索结果 —— 在变化快的领域，这会让检索能力直接失效。

**修法**：注入当天日期 + 明确「检索结果的时效性优先于你的先验」+ 要求做**来源分级**
（一手 > 主流媒体 > SEO 聚合）而不是整体拒绝。后来还补了一条：**检索词的时间锚点也要用今天算**，
不要拿记忆里的年份和事件当检索前提（第 2 次运行里它搜的是 "funding 2025"、"Claude 3.5"，
反而错过了上一次搜到过的营收报道）。

### B. 告诉模型它还剩多少资源 ✅

**症状**（第 1 次运行）：还剩 6 轮、97% 预算的情况下，它问「是否同意我再跑 2-3 组检索」——
而 CLI 是单轮的，**没有人能回答它**。

**根因**：`AgentState` 里 step / remaining_budget 一应俱全，却从没告诉过模型；
system prompt 也没说明它跑在无人值守模式下。

**修法**：每轮追加一条 `[运行状态]` 消息（追加在末尾，不破坏 prompt caching 前缀），
system prompt 里写明「没有人会回答你的提问，不要请求许可」。

### C. 触顶时必须强制收尾 ✅

**症状**（第 3 次运行）：修完 A、B 之后它变得很自主 —— 自主过头了。
8 轮全用来检索（而且检索得很好：anthropic.com/news、Bloomberg、NYT、FT 一手来源全找到了），
然后撞上 `max_steps`，输出「（未得出最终答案）」。**十次检索的钱全白花。**

**根因**：刹车只负责「停」，不负责「卸货」。而且状态行里预算显示很充裕（91%），
步数却已见底 —— 模型权衡时被前者误导了。

**修法**两层：
1. 最后一轮直接传**空的工具清单**（措辞可以被无视，空清单不能），并在状态行里升级措辞；
2. 真触顶时再问一次、同样不给工具，把已有信息榨成「结论 + 置信度 + 未核实项」，
   抢到东西才标记 `salvaged=True`。

**教训**：这三条都指向同一个更一般的原则 ——
**agent 的失败往往不在控制流，而在「模型不知道自己的处境」**。
状态存在 `AgentState` 里不等于模型知道；不告诉它，它就只能靠猜。

### D. 时间预算 ✅（2026-08-26）

**动机**：接了真实检索之后，限流间隔（1.5s）+ 退避重试（最多 7s）+ 网络延迟让墙上时间显著变长，
但循环对「跑了多久」完全无感 —— 钱可能只用了 8%，人已经等了两分钟。
钱衡量的是模型算力，时间衡量的是人的等待，两者不能互相代替。

**实现**：`run(time_budget=...)` / `--deadline 秒`，**默认 600 秒（10 分钟）**、填 0 不限，新状态 `out_of_time`，
超时同样走强制收尾轮（C）。`[运行状态]` 行现在会把步数/预算/时间里**最紧的那道**摆到模型面前。
`clock` 参数可注入假时钟，所以这条刹车在评测里是可复现的（`clock_values`），不靠 sleep。

**已知局限**：时间只在两轮之间检查，单个工具调用卡死仍会超时。
给工具本身加超时需要线程/异步，留到 #5 并行执行时一起做。
