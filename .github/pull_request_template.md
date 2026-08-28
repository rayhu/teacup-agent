<!-- The gate from docs/workflow.md, step 7: a human reads this before it lands. -->

## What this changes

Which roadmap item (or field patch) this is, and what counts as done for it.

## Verification

Paste the actual output. "Should work" is not a report; "the live path is unverified, no
API key in this shell" is.

```
uv run pytest
uv run python -m teacup_agent.evals
uv run teacup-agent
```

## Independent review

- [ ] Reviewed by an agent or person who did not write it, against `REVIEW.md`
- [ ] Findings answered — fixed, or argued in the thread rather than dropped

## If it is behavioural

Before-and-after numbers, or the reason there are none.
