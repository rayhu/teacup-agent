"""Hands-on demo of the approval gate: ask the model to send an email and watch what
each policy does.

It uses the scripted model, so it needs **no API key and costs nothing**. Run it as
often as you like:

    uv run python examples/approval_demo.py
"""

import json
import os
import pathlib
import tempfile

os.environ["TEACUP_AGENT_SEARCH"] = "offline"

from teacup_agent import loop  # noqa: E402
from teacup_agent.memory import NullMemory  # noqa: E402
from teacup_agent.model import ScriptedModel, assistant_calls, assistant_says  # noqa: E402

EMAIL = {
    "to": "boss@example.com",
    "subject": "Weekly report",
    "body": "Progress this week ...",
}


def script():
    """The fake model's script: request the email, then wrap up."""
    return [
        assistant_calls([("send_email", EMAIL)]),
        assistant_says("All done."),
    ]


POLICIES = {
    "default (unattended)": None,  # no approve argument = deny_all
    "human approved": lambda call, spec: True,
    "human refused": lambda call, spec: False,
    "recipient allowlist": lambda call, spec: json.loads(call.arguments)["to"].endswith(
        "@example.com"
    ),
}

os.chdir(tempfile.mkdtemp())  # send_email writes to the cwd; keep the repo clean
outbox = pathlib.Path("outbox.jsonl")

print(f"{'policy':24} {'executed':10} {'reason':10} mails in the outbox")
print("-" * 70)
for name, approve in POLICIES.items():
    outbox.unlink(missing_ok=True)
    kwargs = {} if approve is None else {"approve": approve}
    state = loop.run("send the weekly report to my boss", ScriptedModel(script()), memory=NullMemory(), **kwargs)

    trace = state.trace[0]
    sent = len(outbox.read_text().splitlines()) if outbox.exists() else 0
    print(f"{name:24} {str(trace.executed):10} {trace.skip_reason or '-':10} {sent}")

print("\nWhat the model receives when a call is denied:")
print(" ", loop.DENIED)
