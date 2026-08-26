# Original study notes (the contents of the old main.py, kept verbatim)

> These are the pseudo-code sketches from before the refactor. The runnable
> implementation lives in `src/mini_agent/`.
> Each section is annotated with the file that now implements it.

## 1. A single tool call (Responses API) -> `model.py` + `tools.py`

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
        result = fn(**item.arguments)   # <- bug: arguments is a JSON string
```

**Two problems** (both fixed in the implementation):

1. `item.arguments` is a JSON **string**; it must go through `json.loads()` before
   being splatted with `**`.
2. The tool result never goes back to the model, and there is no loop — so this is a
   single function call, not an agent.

## 2. Planning -> the strategy section of the system prompt

```python
plan = llm("Break this task into steps")
```

## 3. The control loop -> `loop.py`

```python
while not done:
    context = observe_state()
    action = model.decide(goal=goal, context=context, available_tools=tools)
    result = execute(action)
    update_state(result)
    done = check_completion()
```

`check_completion()` was replaced by two explicit conditions: **the model stopped
requesting tools**, or **a step / budget / time ceiling was hit**.

## 4. The tool list -> the REGISTRY in `tools.py`

```python
tools = [web_search, browser, gmail_search, calendar_lookup, sql_query, python, send_email]
```

## 5. State -> `state.py`

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

## 6. The whole thing in one line

```
LLM -> tool call -> tool result -> LLM
```
