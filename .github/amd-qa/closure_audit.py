"""Evidence for one temporary ROCm dependency-closure run; never launches a model."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def snapshot(output):
    constraints = []
    for value in os.environ.get("PIP_CONSTRAINT", "").split():
        path = Path(value)
        content = path.read_bytes()
        constraints.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "text": content.decode(),
            }
        )
    packages = sorted(
        (
            {
                "name": dist.metadata["Name"],
                "version": dist.version,
                "requires": dist.requires or [],
            }
            for dist in metadata.distributions()
        ),
        key=lambda dist: (dist["name"].lower(), dist["version"]),
    )
    record = {
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "source_sha": os.environ["GITHUB_SHA"],
        "checkout_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner": os.environ["RUNNER_NAME"],
        "engine": os.environ["E2E_ENGINE"],
        "configured_image": os.environ["SMG_QA_IMAGE"],
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "constraints": constraints,
        "packages": packages,
    }
    if record["source_sha"] != record["checkout_sha"]:
        raise SystemExit("Checkout does not match dispatched source")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Recorded {len(packages)} distributions in {output}")


def require_vllm(junit):
    required = json.loads(Path(__file__).with_name("vllm-required-cases.json").read_text())
    if len(required) != 14 or len(set(required)) != 14:
        raise SystemExit("Required vLLM test manifest must contain 14 distinct cases")
    cases = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        cases.setdefault(name, []).append(case)
    bad = []
    for name in required:
        found = cases.get(name, [])
        if len(found) != 1 or any(
            child.tag in {"failure", "error", "skipped", "rerunFailure", "rerunError"}
            for case in found
            for child in case
        ):
            bad.append(name)
    if bad:
        raise SystemExit("Required vLLM cases missing, duplicated or not passing: " + repr(bad))
    print("All 14 required installed-vLLM cases executed once and passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["snapshot", "require-vllm"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    (snapshot if args.mode == "snapshot" else require_vllm)(args.path)
