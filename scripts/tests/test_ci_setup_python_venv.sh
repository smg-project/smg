#!/bin/bash
# Tests for scripts/ci_setup_python_venv.sh.
#
# This script decides which interpreter every CI lane runs on, and its failure
# mode is silence: a venv built on the wrong Python does not error here, it
# surfaces much later as an unrelated-looking import failure in a different
# script. That is exactly how it went unnoticed that the bare-metal runners were
# on 3.10 while everything else was on 3.12.
#
# So the branches are exercised against stubbed `python3` and `uv` on PATH.
# Stubs keep this hermetic -- no interpreter downloads, no network, no apt --
# and let the failure paths be tested at all, which is impossible with a real
# interpreter that refuses to report the wrong version.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ci_setup_python_venv.sh"

PASS=0
FAIL=0
LAST_DIR=""
LAST_CODE=""

fail() {
    echo "  FAIL: $1"
    FAIL=$((FAIL + 1))
}

ok() {
    echo "  ok: $1"
    PASS=$((PASS + 1))
}

assert_eq() {
    local expected="$1" actual="$2" what="$3"
    if [ "$expected" = "$actual" ]; then
        ok "$what"
    else
        fail "$what (expected '$expected', got '$actual')"
    fi
}

# Build a sandbox with stubbed python3 + uv ahead of the real ones on PATH.
make_sandbox() {
    local dir="$1"
    mkdir -p "$dir/bin"

    cat > "$dir/bin/python3" <<'STUB'
#!/bin/bash
# Stub host interpreter. Creates a fake venv whose own python is another stub,
# so the version the venv reports can differ from the host's -- which is the
# case the assertions under test exist to catch.
make_venv() {
    local venv="$1"
    mkdir -p "$venv/bin"
    [ "${STUB_VENV_HAS_PIP:-1}" = "1" ] && touch "$venv/.has_pip"
    cat > "$venv/bin/python" <<'INNER'
#!/bin/bash
VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)"
case "$1" in
    -c) echo "${STUB_VENV_PYVER:-${STUB_HOST_PYVER:-3.12}}"; exit 0 ;;
    -m)
        case "$2" in
            pip) [ -f "$VENV_DIR/.has_pip" ] && { echo "pip 99.9"; exit 0; }; exit 1 ;;
            ensurepip)
                if [ "${STUB_ENSUREPIP_WORKS:-1}" = "1" ]; then touch "$VENV_DIR/.has_pip"; exit 0; fi
                exit 1 ;;
        esac ;;
esac
exit 0
INNER
    chmod +x "$venv/bin/python"
}

case "$1" in
    -c) echo "${STUB_HOST_PYVER:-3.12}"; exit 0 ;;
    -m)
        case "$2" in
            venv)
                if [ "$3" = "--help" ]; then exit "${STUB_VENV_MODULE_MISSING:-0}"; fi
                make_venv "$3"
                exit 0 ;;
        esac ;;
esac
exit 0
STUB

    cat > "$dir/bin/uv" <<'STUB'
#!/bin/bash
# Stub uv. Records that it was reached, so a test can assert the host-matches
# path did NOT fall through to provisioning.
echo "uv $*" >> "${STUB_UV_MARKER:-/dev/null}"
if [ "$1" = "venv" ]; then
    venv=".venv"
    mkdir -p "$venv/bin"
    # --seed is what puts pip in a uv venv; honour it, so a test can catch its
    # removal rather than discovering it in a downstream job.
    case "$*" in *--seed*) touch "$venv/.has_pip" ;; esac
    cat > "$venv/bin/python" <<'INNER'
#!/bin/bash
VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)"
case "$1" in
    -c) echo "${STUB_VENV_PYVER:-${CI_PYTHON_VERSION:-3.12}}"; exit 0 ;;
    -m)
        case "$2" in
            pip) [ -f "$VENV_DIR/.has_pip" ] && { echo "pip 99.9"; exit 0; }; exit 1 ;;
        esac ;;
esac
exit 0
INNER
    chmod +x "$venv/bin/python"
fi
exit 0
STUB

    chmod +x "$dir/bin/python3" "$dir/bin/uv"
}

# Run the script under test inside a fresh sandbox.
# Sets LAST_CODE and LAST_DIR. Deliberately not called via $(...): that runs the
# function in a subshell, where the globals it sets are discarded.
run_case() {
    local dir
    dir="$(mktemp -d)"
    make_sandbox "$dir"
    (
        cd "$dir"
        export PATH="$dir/bin:$PATH"
        export STUB_UV_MARKER="$dir/uv_called"
        unset GITHUB_PATH GITHUB_ENV
        set +e
        bash "$SCRIPT" > "$dir/out" 2> "$dir/err"
        echo $? > "$dir/code"
    )
    LAST_DIR="$dir"
    LAST_CODE="$(cat "$dir/code")"
}

echo "test: host already ships the pinned version"
STUB_HOST_PYVER=3.12 CI_PYTHON_VERSION=3.12 run_case
assert_eq "0" "$LAST_CODE" "exits clean"
if [ -f "$LAST_DIR/uv_called" ]; then
    fail "uv must not be used when the host already matches"
else
    ok "no uv provisioning (no-op path for green lanes)"
fi

echo "test: host ships a different version"
STUB_HOST_PYVER=3.10 CI_PYTHON_VERSION=3.12 run_case
assert_eq "0" "$LAST_CODE" "exits clean"
if grep -q "python install 3.12" "$LAST_DIR/uv_called" 2>/dev/null; then
    ok "provisions the pinned interpreter with uv"
else
    fail "expected 'uv python install 3.12'"
fi
if grep -q -- "--seed" "$LAST_DIR/uv_called" 2>/dev/null; then
    ok "seeds pip into the uv venv"
else
    fail "uv venv must pass --seed; downstream steps call 'python3 -m pip'"
fi

echo "test: venv lands on the wrong interpreter"
# The whole point of the assertion: creation succeeded, but on the wrong Python.
STUB_HOST_PYVER=3.12 STUB_VENV_PYVER=3.10 CI_PYTHON_VERSION=3.12 run_case
assert_eq "1" "$LAST_CODE" "fails loudly instead of returning a bad venv"
if grep -q "expected 3.12" "$LAST_DIR/err" 2>/dev/null; then
    ok "names both versions in the error"
else
    fail "error should say what it got and what it expected"
fi

echo "test: venv has no pip and ensurepip cannot fix it"
STUB_HOST_PYVER=3.12 CI_PYTHON_VERSION=3.12 STUB_VENV_HAS_PIP=0 STUB_ENSUREPIP_WORKS=0 run_case
assert_eq "1" "$LAST_CODE" "fails rather than deferring to a downstream pip call"
if grep -q "no pip" "$LAST_DIR/err" 2>/dev/null; then
    ok "error explains why pip is required"
else
    fail "error should mention pip"
fi

echo "test: missing pip is repaired by ensurepip"
STUB_HOST_PYVER=3.12 CI_PYTHON_VERSION=3.12 STUB_VENV_HAS_PIP=0 STUB_ENSUREPIP_WORKS=1 run_case
assert_eq "0" "$LAST_CODE" "recovers without failing the job"

echo "test: CI_PYTHON_VERSION override is honoured"
STUB_HOST_PYVER=3.10 CI_PYTHON_VERSION=3.10 run_case
assert_eq "0" "$LAST_CODE" "exits clean"
if [ -f "$LAST_DIR/uv_called" ]; then
    fail "an explicit 3.10 lane on a 3.10 host should not provision"
else
    ok "explicit older-interpreter lane uses the host directly"
fi

echo
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
