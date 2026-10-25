"""Require the installed-vLLM cases to execute once and pass in engine unit JUnit."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def require_vllm(junit):
    required = json.loads(Path(__file__).with_name("ci_vllm_required_cases.json").read_text())
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
    require_vllm(Path(sys.argv[1]))
