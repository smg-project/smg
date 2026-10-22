"""Apply only the upstream TVMDerivedObject slots hunk to exact image source."""

import hashlib
import json
from pathlib import Path

SOURCE = Path("/opt/tilelang/3rdparty/tvm/python/tvm/runtime/support.py")
ORIGINAL_SHA = "f1c96958d39466a5bf00d7c31326ccb3d4eac6941dbf7126642460b7d0a1f702"
FIXED_SHA = "a1c63f6919c1c53646b882ee6c5b3f4dd1139c67e3a92c6b55537fef64894afe"
UPSTREAM = "https://github.com/apache/tvm/commit/44dbd138d51314309955fba8c3d1294e4a89da35"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def apply(source, output):
    original = source.read_bytes()
    if sha(original) != ORIGINAL_SHA:
        raise ValueError("TVM support source differs from exact unmodified image")
    needle = (
        b'    class TVMDerivedObject(metadata["cls"]):  # type: ignore\n'
        b'        """The derived object to avoid cyclic dependency."""\n\n'
    )
    if original.count(needle) != 1:
        raise ValueError("missing exact upstream slots patch context")
    fixed = original.replace(
        needle, needle + b'        __slots__ = ("__dict__", "__weakref__",)\n\n'
    )
    if sha(fixed) != FIXED_SHA:
        raise ValueError("unexpected source after slots backport")
    if output.exists():
        raise ValueError("refuse backport receipt overwrite")
    source.write_bytes(fixed)
    if sha(source.read_bytes()) != FIXED_SHA:
        raise ValueError("installed TVM slots source verification failed")
    output.write_text(
        json.dumps(
            {
                "path": str(source.resolve()),
                "original_sha256": ORIGINAL_SHA,
                "fixed_sha256": FIXED_SHA,
                "original_bytes": len(original),
                "fixed_bytes": len(fixed),
                "official_commit": UPSTREAM,
                "official_hunk_only": "runtime/support.py __slots__",
                "full_upstream_commit_applied": False,
                "native_binaries_changed": False,
                "package_metadata_or_requirements_changed": False,
            },
            indent=2,
        )
        + "\n"
    )


def verify_loaded(record):
    if record.get("sha256") != FIXED_SHA or record.get("path") != str(SOURCE):
        raise ValueError("actual imported TVM support differs from reviewed slots backport")


if __name__ == "__main__":
    apply(SOURCE, Path("diagnostic-results/tvm-slots-backport.json"))
