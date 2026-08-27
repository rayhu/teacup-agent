"""Thin entry point so `python main.py` works; the real usage is `uv run teacup-agent`.

The study notes that used to live in this file were moved to NOTES.md in full, with
each section annotated with the implementation file it corresponds to.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # works without installing

from teacup_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
