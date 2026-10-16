import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SEACACHE_COMMIT = "8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2"


def _seacache_dir() -> Path:
    """SeaCache's filter and staleness metric are its authors' code and are never copied into this
    package: a repo clone uses the pinned git submodule, anything else (pip install, or a clone made
    without --recursive) downloads the same commit once from upstream into ~/.cache/softcache."""
    submodule = Path(__file__).resolve().parents[1] / "third_party" / "SeaCache"
    if (submodule / "FLUX").exists():
        return submodule
    home = Path(os.environ.get("SOFTCACHE_HOME", Path.home() / ".cache" / "softcache"))
    target = home / f"SeaCache-{SEACACHE_COMMIT[:7]}"
    if not (target / "FLUX").exists():
        home.mkdir(parents=True, exist_ok=True)
        print(f"softcache: downloading SeaCache {SEACACHE_COMMIT[:7]} to {target}", file=sys.stderr)
        url = f"https://github.com/jiwoogit/SeaCache/archive/{SEACACHE_COMMIT}.tar.gz"
        with tempfile.TemporaryDirectory(dir=home) as tmp, urllib.request.urlopen(url) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as tar:
                for member in tar:   # the archive is one top-level dir; refuse anything that escapes it
                    if member.name.startswith(("/", "..")) or ".." in Path(member.name).parts:
                        raise RuntimeError(f"unsafe path in the SeaCache archive: {member.name}")
                    tar.extract(member, tmp)
            try:
                (Path(tmp) / f"SeaCache-{SEACACHE_COMMIT}").rename(target)
            except OSError:   # another process finished the same download first
                if not (target / "FLUX").exists():
                    raise
    return target


sys.path.insert(0, str(_seacache_dir() / "FLUX"))

from .method import Config  # noqa: E402
