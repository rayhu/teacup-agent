---
name: web-research
description: Research a question across several web sources and answer with graded, cited evidence. Load before any multi-source lookup.
---

## Procedure

1. **Scope first, in one line.** Write down what would count as an answer before
   searching. "Anthropic's last-six-months funding" is a scope; "research Anthropic" is
   not, and it is how runs burn eight turns and deliver nothing.

2. **Two or three queries, deliberately different.** Vary the angle, not the wording:
   the entity's own name, the event type, the number you expect to find. Four rephrasings
   of the same query return the same page four times.

3. **Grade every source as it arrives**, and keep the grade attached to the claim:

   | Tier | What it is | How to treat it |
   | --- | --- | --- |
   | Primary | company site, filing, regulator, official blog | citable on its own |
   | Major media | Reuters, Bloomberg, FT, CNBC, NYT | citable when it names its own source |
   | Aggregator / SEO | listicles, "2026 guide" content farms, republished posts | never citable alone; use only to find a primary source |

   A number that appears only on aggregators is not a fact yet. Say so.

4. **Cross-check anything load-bearing.** Two independent sources, or the claim carries
   "unverified" in the answer. Independent means different reporting, not the same wire
   story on two sites.

5. **Read the source when the claim matters.** A 300-character snippet is not the
   article. Fetch the page before quoting a number that a decision would rest on.

6. **Stop when the next search would not change the answer.** More searching is not more
   rigour; it is the most common way to run out of steps with nothing written down.

## Writing it up

- Lead with what is confirmed, then what is uncertain, then what you could not check.
- Every non-obvious claim carries a link and a confidence level.
- Name the gaps explicitly: "no primary source found for X" is a finding, not a failure.

## Failure modes to avoid

- **Trusting your memory over the search.** If a result conflicts with what you
  remember, the result is probably newer than you are.
- **Reading a failed search as an empty world.** A tool error means the channel broke,
  not that the fact does not exist. Retry or say it is unverified.
- **Delivering a plan instead of an answer.** If you ran out of room, give the best
  conclusion available with its confidence, not a list of what you would have done next.
