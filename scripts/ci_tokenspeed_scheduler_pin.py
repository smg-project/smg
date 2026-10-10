"""Print the scheduler pin the installed TokenSpeed runtime nightly implies.

The runtime nightly requires ``tokenspeed-scheduler`` with a lower bound
only, so a plain install resolves the newest scheduler release, which can be
newer than the nightly: a scheduler released after the nightly rejected a
scheduler field the nightly never sets, and every engine worker died at
start. The lower bound is the release the nightly was built against; pin to
it. Prints ``tokenspeed-scheduler==<version>`` for ``pip install``; prints
nothing (a note on stderr) when the runtime declares no lower bound or exact
version, so the caller keeps pip's choice.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata

RUNTIME = "tokenspeed"
SCHEDULER = "tokenspeed-scheduler"

_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def scheduler_pin(requirements) -> str | None:
    """The exact scheduler requirement the runtime's metadata implies, or None."""
    for requirement in requirements or ():
        match = _REQUIREMENT.match(requirement.split(";", 1)[0].strip())
        if not match or _canonical(match.group(1)) != SCHEDULER:
            continue
        clauses = [clause.strip() for clause in match.group(2).split(",") if clause.strip()]
        for operator in ("==", ">="):
            for clause in clauses:
                if clause.startswith(operator) and not clause.startswith("==="):
                    return f"{SCHEDULER}=={clause[len(operator) :].strip()}"
        return None
    return None


def main() -> int:
    try:
        requirements = metadata.requires(RUNTIME)
    except metadata.PackageNotFoundError:
        print(f"{RUNTIME} is not installed; nothing to pin", file=sys.stderr)
        return 1
    pin = scheduler_pin(requirements)
    if pin is None:
        print(
            f"{RUNTIME} declares no lower bound on {SCHEDULER}; keeping pip's choice",
            file=sys.stderr,
        )
        return 0
    print(pin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
