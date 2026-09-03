#!/usr/bin/env bash
# Print a directory that satisfies the `-lpython3.X` PyO3 emits for the
# `bindings/python` test binary, and exit non-zero with a diagnosis if no such
# directory can be produced.
#
# Why this is needed at all: `bindings/python` is a workspace member, so a bare
# `cargo test` builds a test binary for it. The maturin cdylib enables
# `extension-module`, which deliberately leaves the libpython symbols undefined
# for the interpreter to supply at dlopen time; the test binary has no
# interpreter and must link libpython for real.
#
# Two things get in the way:
#   * `sysconfig`'s LIBDIR is baked in when CPython is built. It is correct for
#     distro and pyenv interpreters, but actions/setup-python relocates its tool
#     cache, leaving that recorded path dangling.
#   * A CPython that ships only `libpython3.X.so.1.0` (no `-dev` package) has no
#     bare `.so` for `-lpython3.X` to resolve against.
#
# So: walk the plausible library directories, prefer one that already works, and
# otherwise build a scratch directory of symlinks rather than mutating the
# interpreter's own lib dir (which is often root-owned).
set -euo pipefail

python="${PYTHON:-python3}"

if ! "$python" -c 'import sysconfig, sys; sys.exit(0 if sysconfig.get_config_var("Py_ENABLE_SHARED") else 1)'; then
    echo "error: $("$python" -c 'import sys; print(sys.executable)') is a static CPython build" >&2
    echo "       (Py_ENABLE_SHARED unset); the bindings/python test binary cannot link libpython." >&2
    exit 1
fi

ldlib=$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY") or "")')
if [ -z "$ldlib" ]; then
    echo "error: sysconfig reports no LDLIBRARY for $python" >&2
    exit 1
fi

candidates=()
# Must not end on a failing test: under `set -e` a function that returns
# non-zero as its last command kills the script, which would turn "this
# candidate does not exist" -- the normal case for sysconfig's baked-in LIBDIR
# under actions/setup-python -- into a silent exit 1.
add_candidate() {
    if [ -n "${1:-}" ] && [ -d "$1" ]; then
        candidates+=("$1")
    fi
    return 0
}

add_candidate "$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
if [ -n "${pythonLocation:-}" ]; then
    add_candidate "${pythonLocation}/lib"
fi
add_candidate "$("$python" -c 'import sys; print(sys.base_prefix)')/lib"
multiarch=$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("MULTIARCH") or "")')
add_candidate "${multiarch:+/usr/lib/$multiarch}"

if [ ${#candidates[@]} -eq 0 ]; then
    echo "error: no plausible python library directory exists for $python" >&2
    exit 1
fi

for dir in "${candidates[@]}"; do
    if [ -e "$dir/$ldlib" ]; then
        echo "$dir"
        exit 0
    fi
done

for dir in "${candidates[@]}"; do
    real=$(find "$dir" -maxdepth 1 -name "$ldlib.*" -print -quit 2>/dev/null)
    [ -n "$real" ] || continue
    scratch="${CARGO_TARGET_DIR:-target}/libpython-link"
    mkdir -p "$scratch"
    # Both names: `-l` resolves the bare `.so`, while the loader resolves the
    # SONAME, so one symlink alone leaves the other lookup failing.
    ln -sf "$real" "$scratch/$ldlib"
    ln -sf "$real" "$scratch/$(basename "$real")"
    echo "$scratch"
    exit 0
done

echo "error: no shared libpython ($ldlib) found in any of: ${candidates[*]:-<none exist>}" >&2
exit 1
