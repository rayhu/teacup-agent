"""方便直接 `python main.py` 的薄入口；正式用法是 `uv run mini-agent`。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # 未安装包时也能跑

from mini_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
