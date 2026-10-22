"""Install only the producer-bound Cairo runtime before importing its native wheel."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from reuse_foundation import require

OUT = Path("diagnostic-results")


def matches(path, expected):
    return (
        path.is_file()
        and path.stat().st_size <= 8 * 1024 * 1024
        and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    )


def command(argv):
    result = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ | {"DEBIAN_FRONTEND": "noninteractive"},
    )
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    binding = json.loads((OUT / "environment-binding.json").read_text())
    proof = binding["producer_native"]
    package = binding["cairo_runtime_package"]
    require(re.fullmatch(r"libcairo2=[0-9A-Za-z.+:~_-]+", package), "unknown Cairo runtime package")
    library = Path(proof["loaded_libcairo"]).resolve()
    require(
        str(library).startswith(("/usr/lib/", "/lib/"))
        and re.fullmatch(r"libcairo\.so(?:\.\d+)*", library.name),
        "unknown Cairo library path",
    )
    query = [
        "dpkg-query",
        "-W",
        "-f=${Package}=${Version}\n",
        "libcairo2",
        "libpng16-16",
        "libpixman-1-0",
        "libfontconfig1",
        "libfreetype6",
        "libuuid1",
        "libmount1",
        "libblkid1",
    ]
    record = {
        "package": package,
        "expected_library": str(library),
        "expected_sha256": proof["loaded_libcairo_sha256"],
        "before_packages": command(query),
        "apt": [],
    }
    try:
        if not matches(library, proof["loaded_libcairo_sha256"]):
            require(
                os.environ.get("GITHUB_ACTIONS") == "true"
                and os.geteuid() == 0
                and Path("/.dockerenv").is_file(),
                "runtime repair requires the root Actions container",
            )
            for argv in [
                ["apt-get", "update"],
                ["apt-get", "install", "-y", "--no-install-recommends", package],
            ]:
                result = command(argv)
                record["apt"].append(result)
                require(result["returncode"] == 0, "Cairo runtime package installation failed")
        require(
            matches(library, proof["loaded_libcairo_sha256"]),
            "Cairo runtime bytes differ from producer",
        )
        record["after_packages"] = command(query)
    finally:
        (OUT / "cairo-runtime.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
