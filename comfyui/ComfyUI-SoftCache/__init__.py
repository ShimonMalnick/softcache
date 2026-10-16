import sys
from pathlib import Path

# Use the `softcache` package from this repo's release/ dir (symlink this folder into
# ComfyUI/custom_nodes), unless it is already pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
