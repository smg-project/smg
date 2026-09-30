#!/bin/bash
# Resolve the newest published nightly; derive the carrier tag from its date.
set -euo pipefail
python3 - "$@" <<'PYTHON'
import re
import sys
import subprocess

index = subprocess.check_output(
    ["curl", "-fsSL", "--retry", "3", "--max-time", "30",
     "https://lightseek.org/whl/nightly/tokenspeed/"], text=True
)
versions = re.findall(r"tokenspeed-(\d+\.\d+\.\d+\.post\d{8})-py3-none-any\.whl", index)
if not versions:
    raise RuntimeError("No dated TokenSpeed nightly wheels published")
version = max(versions, key=lambda value: tuple(map(int, re.findall(r"\d+", value))))
print(version if "--version" in sys.argv else f"ci-tokenspeed-{version.rsplit('post', 1)[1]}")
PYTHON
