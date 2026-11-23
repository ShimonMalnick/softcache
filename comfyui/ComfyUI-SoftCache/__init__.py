import sys
from pathlib import Path

# In a clone of the softcache repo (this folder symlinked into ComfyUI/custom_nodes), use the repo's
# `softcache` package; otherwise use the pip-installed one.
_repo = Path(__file__).resolve().parents[2]
if (_repo / "softcache" / "method.py").exists():
    sys.path.insert(0, str(_repo))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
