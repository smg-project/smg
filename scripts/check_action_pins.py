#!/usr/bin/env python3
"""Enforce commit-SHA pins on third-party GitHub Actions.

A `uses:` reference like `actions/checkout@v7` points at a mutable tag: whoever
can move that tag executes code with this repository's token, on our runners.
Every third-party reference must therefore pin the full 40-hex commit SHA and
keep the human-readable version alongside it as a comment:

    uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1

The comment is load-bearing too: it is what Dependabot rewrites when it bumps a
pinned action, and what a human reads to know what is installed. A bare major
(`# v7`) is rejected because it cannot be checked against the SHA.

Local composite actions (`./…`), reusable-workflow refs to this repo, and
expression refs (`${{ … }}`) are exempt. Docker refs (`docker://…`) carry their
own digest discipline and are skipped here.

Usage:
    python scripts/check_action_pins.py           # scan .github/, exit 1 on violations
    python scripts/check_action_pins.py --test    # run the built-in case table
"""

import glob
import re
import sys

SCAN_GLOBS = [
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    ".github/actions/*/action.yml",
    ".github/actions/*/action.yaml",
]

# owner/repo whose refs are allowed to stay unpinned, each with a written
# reason. Every entry here is a check this script is NOT performing — keep it
# empty unless an action genuinely cannot be pinned.
UNPINNED_ALLOWED: dict[str, str] = {}

USES_RE = re.compile(r"^\s*(?:-\s+)?uses:\s*(.+?)\s*$")
# 40-hex ref, then a version comment with at least two dotted numeric
# components ("# v7.0.1", "# 1.98.0"). Suffixes like "-rc1" are accepted.
PINNED_RE = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^@\s]+)?"
    r"@[0-9a-f]{40}"
    r"\s+#\s*v?\d+(?:\.\d+)+\S*$"
)


def check_line(line: str) -> str | None:
    """Return a violation message for one line, or None if the line is fine."""
    m = USES_RE.match(line)
    if m is None:
        return None
    ref = m.group(1)
    # A quoted scalar must not hide a tag ref: unwrap before judging.
    if len(ref) >= 2 and ref[0] == ref[-1] and ref[0] in "\"'":
        ref = ref[1:-1].strip()
    if ref.startswith("./") or ref.startswith("docker://") or "${{" in ref:
        return None
    repo = ref.split("@", 1)[0].split("/")
    if len(repo) >= 2 and "/".join(repo[:2]) in UNPINNED_ALLOWED:
        return None
    if PINNED_RE.match(ref):
        return None
    if "@" not in ref:
        return f"missing a ref entirely: `{ref}`"
    target = ref.split("@", 1)[1].split()[0]
    if re.fullmatch(r"[0-9a-f]{40}", target):
        return f"pinned but missing the `# vX.Y.Z` version comment: `{ref}`"
    return f"mutable ref `{ref}` — pin the commit SHA and keep the version as a comment"


def scan() -> int:
    violations = []
    for pattern in SCAN_GLOBS:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    message = check_line(line)
                    if message is not None:
                        violations.append(f"{path}:{lineno}: {message}")
    if violations:
        print("\n".join(violations))
        print(
            "\nEvery third-party action must be pinned:"
            "\n    uses: owner/repo@<40-hex commit sha> # vX.Y.Z"
            "\nResolve a tag to its commit with:"
            "\n    gh api repos/OWNER/REPO/commits/TAG --jq .sha"
        )
        return 1
    return 0


# (line, expected_ok) — every rule and every near-miss this script must catch.
CASES: list[tuple[str, bool]] = [
    ("      uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1", True),
    ("      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1", True),
    ("      uses: dtolnay/rust-toolchain@f8be11a05b1d4f3fcebe6410cc16743212b999b0 # 1.98.0", True),
    ("      uses: owner/repo/subdir@3d3c42e5aac5ba805825da76410c181273ba90b1 # v1.2.3", True),
    ("      uses: ./.github/actions/setup-rust", True),
    ("      uses: ./.github/workflows/e2e-gpu-job.yml", True),
    ("      uses: ${{ matrix.action }}", True),
    ("      uses: docker://alpine:3.20", True),
    ("      not_a_uses_line: actions/checkout@v7", True),
    ("      uses: actions/checkout@v7", False),
    ('      uses: "actions/checkout@v7"', False),
    ("      uses: 'actions/checkout@v7'", False),
    ("      uses: actions/checkout@main", False),
    ("      uses: actions/checkout@3d3c42e5", False),
    ("      uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", False),
    ("      uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7", False),
    ("      uses: actions/checkout@3D3C42E5AAC5BA805825DA76410C181273BA90B1 # v7.0.1", False),
    ("      uses: actions/checkout", False),
]


def self_test() -> int:
    failures = 0
    for line, expected_ok in CASES:
        actual_ok = check_line(line) is None
        if actual_ok != expected_ok:
            failures += 1
            print(f"case failed (expected {'ok' if expected_ok else 'violation'}): {line!r}")
    print(f"self-test: {len(CASES) - failures}/{len(CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(self_test() if "--test" in sys.argv[1:] else scan())
