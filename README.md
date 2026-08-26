# mini-agent

一个**最小但五脏俱全**的 AI Agent，用来把这个公式落成能跑的代码：

```
Agent = Model + State + Tools + Control Loop + Memory/Evals
```

每个部件都是真的在工作（没有空壳模块），但实现都刻意保持在几十行，方便逐个替换成真货。

## 快速开始

```bash
uv sync                      # 建环境（自动创建 .venv 与 uv.lock）

uv run mini-agent            # 离线跑一个 demo：不需要任何 API key，不花钱
uv run mini-agent "帮我算一下 (3200-450)*0.6，并查一下 CUDA 是什么"

uv run mini-agent --live "研究一下 NVIDIA 的 GPU 策略"          # 真实调用 OpenAI，需要 .env 里的 key
uv run mini-agent --live --api chat "..."                       # 想对比旧的 Chat Completions 路径

uv run python -m mini_agent.evals   # 跑评测（离线断言控制循环的正确性）
uv run pytest                        # 同一批用例 + 工具/记忆单测
```

`--live` 需要 `.env`：

```bash
cp .env.example .env   # 然后填入 OPENAI_API_KEY
```

## 目录结构

```
src/mini_agent/
├── model.py    Model        —— 唯一会思考的部件。OpenAIModel（真实）/ ScriptedModel（离线剧本）
├── state.py    State        —— goal / messages / step / remaining_budget / status + 工具留痕
├── tools.py    Tools        —— 函数 + JSON Schema + 安全执行（错误变成工具结果，而不是异常）
├── memory.py   Memory       —— 短期 = messages；长期 = memory.json（可被 remember 工具写入）
├── loop.py     Control Loop —— LLM → tool call → tool result → LLM
├── evals.py    Evals        —— 用脚本模型体检循环本身，零 key 零成本
└── cli.py                   —— 命令行入口
tests/                       —— pytest：evals 用例 + 工具/记忆单测
NOTES.md                     —— 你原来的学习笔记，逐段标注了对应实现
docs/roadmap.md              —— 这份实现距离当前生产级别的 agent 还差什么，以及按什么顺序补
```

## 五个部件都做了什么

| 部件 | 文件 | 当前实现 | 想加强时换成 |
| --- | --- | --- | --- |
| Model | `model.py` | Responses API（默认）+ Chat Completions，另有离线脚本模型 | Claude / 本地模型 / 多模型路由 |
| State | `state.py` | dataclass：步数、预算、状态机、留痕 | 落盘 checkpoint、可恢复运行 |
| Tools | `tools.py` | search_web（**真实联网**，DuckDuckGo 免 key）、calculate、read_file、remember | 浏览器、SQL、代码执行、发邮件 |
| Control Loop | `loop.py` | 单层循环 + 步数/预算/每轮工具数三道守卫 | 计划-执行分离、子 Agent、人工确认 |
| Memory | `memory.py` | JSON 文件 + 去重 + 只留最近 N 条 | 向量库、摘要压缩、按相关性召回 |
| Evals | `evals.py` | 7 条离线用例断言循环行为 | 加入真实任务集与打分模型 |

## search_web 的三种模式

联网检索走 [`ddgs`](https://pypi.org/project/ddgs/)（DuckDuckGo），**不需要任何 API key**，`uv sync` 时已随依赖装好。
用环境变量 `MINI_AGENT_SEARCH` 切换：

| 值 | 行为 | 用途 |
| --- | --- | --- |
| `auto`（默认） | 联网检索；失败则退回本地语料并注明原因 | 日常使用 |
| `web` | 只用联网检索；失败返回 `ERROR:` | 不允许拿离线语料充数的场景 |
| `offline` | 只查本地 3 条语料，零网络请求 | 评测、单测、演示 |

命令行用 `--search` 覆盖；默认值跟运行模式绑定：`--live` 时是 `auto`（真联网），离线 demo 是 `offline`（零网络、秒出）。

```bash
uv run mini-agent --live "研究 OpenAI 最近半年的融资和竞争格局"   # 默认就会真联网
MINI_AGENT_SEARCH=offline uv run mini-agent                      # 强制离线
```

**一条设计原则：宁可返回「没找到」，也不能返回错的东西。**
早期版本的离线匹配用 `any()`（查询里出现任一关键词就算命中），结果 "OpenAI strategy" 命中了 "nvidia gpu strategy" 这条语料，
模型差点拿 NVIDIA 的资料回答 OpenAI 的问题。现在改成 `all()`，并有回归测试盯着。
同理，`web` 模式下检索失败必须报 `ERROR:` 而不是返回空 —— 否则模型会把「工具坏了」理解成「世界上没有这件事」。

## 两个 OpenAI 后端

| | `--api responses`（默认） | `--api chat` |
| --- | --- | --- |
| 工具定义 | 扁平 `{"type":"function","name":...}` | 嵌套 `{"function":{...}}` |
| 一轮输出 | `output` 列表（reasoning 项 + 多个 function_call） | 单条 assistant 消息 |
| 调用 id | `call_id` | `id` |
| 结果回填 | `{"type":"function_call_output",...}` | `role="tool"` 消息 |
| 推理状态 | **跨工具调用保留** | 每轮丢弃 |

四处差异全封在 [`model.py`](src/mini_agent/model.py) 的两个类里，[`loop.py`](src/mini_agent/loop.py) 的控制流一行没动 ——
循环里只有两个泛化点：`state.messages.extend(reply.items)`（一轮可能产出多条），
以及工具结果的形状由后端的 `tool_result_item()` 决定。

同一任务实测（gpt-5-mini）：responses 的上下文里多出一条 `reasoning` 项，会随下一轮请求发回去；
chat 那边没有对应物。OpenAI 迁移文档称同 prompt 下 SWE-bench 高约 3%。

## 三道刹车

模型可能一轮甩出 10 个搜索、或者在工具之间反复横跳把预算烧光。循环里有三个独立的守卫：

| 守卫 | 参数 | 默认 | 触发后 |
| --- | --- | --- | --- |
| 轮数 | `--max-steps` | 8 | `status="max_steps"`，停机 |
| 预算 | `--budget`（美元） | 0.05 | `status="out_of_budget"`，停机 |
| 每轮工具调用数 | `--max-tool-calls` | 3（0 = 不限） | 只执行前 N 个，其余**退回下一轮** |

触顶不等于放弃：见下面「刹车不能只是停」。

第三道刹车有个必须做对的细节：**被拦下的调用也要回一条 `role="tool"` 消息**，内容是
「本轮已达上限，未执行，下一轮再发」。如果只是把它们丢掉，那些 `tool_call_id` 就没有对应结果，
下一轮请求会直接 400 —— 所以这里的语义是「拒绝执行」而不是「忽略」。模型读到这条会自己收敛，
把 10 个查询压成最关键的几个。`evals.py` 里有一条用例专门锁这个行为，
`state.snapshot()` 也会分别报告 `tool_calls`（真跑了几个）和 `throttled`（拦了几个）。

### 刹车不能只是「停」

实测教训：模型曾把 8 轮全用在检索上，一个字的结论都没给 —— 十次检索的钱白花。
所以刹车还要负责**把车上的东西卸下来**，两道机制：

1. **最后一轮收走工具**：`state.step >= max_steps` 时传空的工具清单。
   措辞可以被无视，空清单不能 —— 模型只剩「说话」这一个选择。
2. **强制收尾轮**：真的触顶（步数或预算）时，再问一次、同样不给工具，
   要求它把已有信息榨成「结论 + 置信度 + 未核实项」。抢救到东西才会把 `salvaged` 标为 true。

每轮开头还会追加一条 `[运行状态]` 消息，把「第几轮 / 剩多少预算」告诉模型 —— 它得知道自己的
处境才谈得上「继续挖还是收尾」。这条追加在**末尾**而不是写进 system prompt，
否则上下文前缀每轮都变，prompt caching 就废了。

```bash
uv run mini-agent --live --max-tool-calls 2 "研究 OpenAI 最近半年的融资和竞争格局"
```

## 控制循环里最容易踩的三个坑

代码里都有注释标记，也都被 `evals.py` 覆盖：

1. **顺序**：带 `tool_calls` 的 assistant 消息必须**先**写回 `messages`，再写工具结果；反了下一轮 API 会报错。
2. **数量**：一轮可能有多个 `tool_call`，**每个 `tool_call_id` 都要有**一条 `role="tool"` 消息，漏一个下一轮 400。
3. **终止**：没有神秘的 `check_completion()`。结束条件只有两种 —— 模型不再请求工具（= 给出最终答案），或步数/预算触顶。

另外：工具执行失败（坏 JSON、参数不对、工具本身抛异常）一律**把错误文本当作工具结果返回**给模型，让它自己改正。这不是防御性编程，这正是循环存在的意义。

## 原笔记里的两个 bug

`NOTES.md` 保留了改造前的伪代码。其中：

```python
result = fn(**item.arguments)   # ① arguments 是 JSON 字符串，不是 dict
                                # ② 结果没有回传给模型，也没有循环
```

① 少了 `json.loads()`；② 没有把工具结果 append 回消息再问一次模型 —— 所以那段代码是「一次函数调用」，还不是 Agent。差别就在 `loop.py` 那 40 行里。

## 它现在处在什么水平

内核不过时（2026 年的 agent 内核依然是这个循环），工程层大约停在 2023 年底 / 2024 年初：
已经补上 Responses API（#1），但仍没有 MCP、没有上下文压缩、工具串行执行、evals 只到单测级别。

差什么、为什么差、按什么顺序补 —— 见 [docs/roadmap.md](docs/roadmap.md)。
