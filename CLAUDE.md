# CLAUDE.md

The rules for this repo live in one file, so that every agent working here — Claude Code,
Codex, whatever comes next — reads the same ones and no copy can drift out of date. The
line below imports that file. Keep it on a line of its own: written inline as part of a
sentence, `@AGENTS.md.` swallows the trailing period into the path and silently imports
nothing (measured against Claude Code 2.1.179 — the model then answers from an empty
ruleset without saying so).

@AGENTS.md
