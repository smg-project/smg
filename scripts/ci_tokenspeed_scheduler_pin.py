"""Print the scheduler pin the installed TokenSpeed runtime nightly implies.

Retain the lower-bound pin for older nightlies. The 20261023 runtime imports
StorageConfig, which first appears in scheduler 0.1.42 although that runtime
declares >=0.1.41. Use its explicit compatible pair and check real imports
after installation. Prints ``tokenspeed-scheduler==<version>`` for ``pip install``; prints
nothing (a note on stderr) when the runtime declares no lower bound or exact
version, so the caller keeps pip's choice.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from importlib import metadata

from packaging.requirements import Requirement

RUNTIME = "tokenspeed"
SCHEDULER = "tokenspeed-scheduler"
KNOWN_SCHEDULERS = {"0.1.1.post20261023": "0.1.42"}

_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def scheduler_pin(requirements, runtime_version=None) -> str | None:
    """Select a known compatible pair, otherwise preserve the metadata pin."""
    for requirement in requirements or ():
        match = _REQUIREMENT.match(requirement.split(";", 1)[0].strip())
        if not match or _canonical(match.group(1)) != SCHEDULER:
            continue
        clauses = [clause.strip() for clause in match.group(2).split(",") if clause.strip()]
        for operator in ("==", ">="):
            for clause in clauses:
                if clause.startswith(operator) and not clause.startswith("==="):
                    if operator == ">=" and runtime_version in KNOWN_SCHEDULERS:
                        version = KNOWN_SCHEDULERS[runtime_version]
                        if version not in Requirement(requirement).specifier:
                            raise ValueError(
                                f"Known scheduler {version} conflicts with {requirement}"
                            )
                        return f"{SCHEDULER}=={version}"
                    return f"{SCHEDULER}=={clause[len(operator) :].strip()}"
        return None
    return None


def check_scheduler(runtime_version):
    expected = KNOWN_SCHEDULERS.get(runtime_version)
    if expected and metadata.version(SCHEDULER) != expected:
        raise RuntimeError(
            f"TokenSpeed {runtime_version} requires the CI scheduler pair {expected}"
        )
    module = importlib.import_module("tokenspeed_scheduler")
    required = (
        ("Scheduler", "StorageConfig", "StoragePressure", "StorageKind")
        if expected
        else ("Scheduler",)
    )
    for symbol in required:
        if getattr(module, symbol, None) is None:
            raise ImportError(
                f"tokenspeed_scheduler lacks {symbol} for TokenSpeed {runtime_version}"
            )


def main(check=False) -> int:
    try:
        runtime_version = metadata.version(RUNTIME)
        requirements = metadata.requires(RUNTIME)
    except metadata.PackageNotFoundError:
        print(f"{RUNTIME} is not installed; nothing to pin", file=sys.stderr)
        return 1
    if check:
        check_scheduler(runtime_version)
        return 0
    pin = scheduler_pin(requirements, runtime_version)
    if pin is None:
        print(
            f"{RUNTIME} declares no lower bound on {SCHEDULER}; keeping pip's choice",
            file=sys.stderr,
        )
        return 0
    print(pin)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    sys.exit(main(check=parser.parse_args().check))
