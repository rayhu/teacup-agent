---
name: long-document
description: Work through a document too large for the context window, keeping only what matters. Load when a fetch or file read returns thousands of characters.
---

## Procedure

1. **Do not re-read it into context.** A large tool result has already been saved to a
   file and replaced with an excerpt plus its path. That excerpt is usually enough to
   decide what you need; reading the whole file back defeats the mechanism.

2. **Decide what you are looking for before opening it.** One or two concrete questions.
   Reading "to understand it" is how a context window fills with material you will never
   cite.

3. **Read in windows, extract as you go.** Use `read_file` on the saved path, and after
   each window write down the extracted claim in your own words with its location. The
   note is what you keep; the text is not.

4. **Delegate when the reading is bulky and the conclusion is small.** If answering needs
   several long pages and the details will not matter afterwards, hand it to a subagent
   with `delegate`. It reads in its own context and returns the conclusion, so the pages
   never enter yours. This costs more total tokens, so it is worth it when the alternative
   is carrying those pages for the rest of a long run.

5. **Quote precisely, and only what carries weight.** One sentence with its source beats
   a paragraph of paraphrase.

## Checks before you finish

- Every claim traceable to a location in the document.
- Anything you could not find stated as not found, rather than smoothed over.
