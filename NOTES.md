# 原始学习笔记（原 main.py 的内容，原样保留）

> 这些是改造前的伪代码草稿。真正可运行的实现见 `agent/` 目录。
> 下面每段后面标注了它对应改造后的哪个文件。

## 1. 单次工具调用（Responses API）→ 对应 `agent/model.py` + `agent/tools.py`

```python
from openai import OpenAI

client = OpenAI()

def search_web(query):
    # your implementation
    return "search results"

tools = {"search_web": search_web}

response = client.responses.create(
    model="gpt-5",
    input="Research NVIDIA's latest GPU strategy",
    tools=[
        {
            "type": "function",
            "name": "search_web",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ],
)

for item in response.output:
    if item.type == "function_call":
        fn = tools[item.name]
        result = fn(**item.arguments)   # ← 这里有 bug：arguments 是 JSON 字符串
```

**两个关键问题**（改造后已修正）：
1. `item.arguments` 是 JSON **字符串**，必须 `json.loads()` 之后再 `**` 展开。
2. 工具结果没有回传给模型，也没有循环 —— 这只是「一次函数调用」，还不是 Agent。

## 2. 计划 → 对应 system prompt 里的策略说明

```python
plan = llm("Break this task into steps")
```

## 3. 控制循环 → 对应 `agent/loop.py`

```python
while not done:
    context = observe_state()
    action = model.decide(goal=goal, context=context, available_tools=tools)
    result = execute(action)
    update_state(result)
    done = check_completion()
```

`check_completion()` 在实现里被替换成两个明确条件：**模型不再请求工具** 或 **步数/预算耗尽**。

## 4. 工具清单 → 对应 `agent/tools.py` 的 REGISTRY

```python
tools = [web_search, browser, gmail_search, calendar_lookup, sql_query, python, send_email]
```

## 5. 状态 → 对应 `agent/state.py`

```python
state = {
    "goal": "...",
    "messages": [...],
    "research_results": [...],
    "current_step": 3,
    "remaining_budget": 2.31,
    "status": "researching",
}
```

## 6. 一句话本质

```
LLM → tool call → tool result → LLM
```
