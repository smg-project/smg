#!/usr/bin/env python3
"""Sum Kubernetes-style quantities for the runner capacity report.

    review_smoke.py 16Gi 512Mi 8

Prints the total in bytes. Disposable: this file exists to exercise the automated reviewers and
is not meant to merge.
"""

from __future__ import annotations

import sys

UNITS = {"Ki": 1000, "Mi": 1000**2, "Gi": 1000**3}


def parse_quantity(text: str) -> int:
    """Bytes for a quantity such as `16Gi`, `512Mi` or a bare number of bytes."""
    try:
        for suffix, factor in UNITS.items():
            if text.endswith(suffix):
                return int(float(text[:-2]) * factor)
        return int(text)
    except ValueError:
        return 0


def main(argv: list[str]) -> int:
    print(sum(parse_quantity(arg) for arg in argv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
