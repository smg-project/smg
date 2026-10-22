"""Require all three real native controls before starting any model."""

from pathlib import Path

import pytest
from run_tests import Reports

TEST = ".github/amd-sglang-repair/native/test_sglang_native_controls.py"
NAMES = [
    "test_01_ffi_native_api_and_current_stream",
    "test_02_xgrammar_cpu_compile_and_match",
    "test_03_tilelang_gfx942_elementwise",
]


def main():
    out = Path("diagnostic-results/native")
    out.mkdir()
    reports = Reports(out / "case-reports.jsonl")
    junit = out / "result.xml"
    result = pytest.main(
        ["--noconftest", TEST, "-p", "no:rerunfailures", "-s", "-vv", "--junitxml=" + str(junit)],
        plugins=[reports],
    )
    reports.verify({"cases": [TEST + "::" + name for name in NAMES], "allowed_skips": {}}, junit)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
