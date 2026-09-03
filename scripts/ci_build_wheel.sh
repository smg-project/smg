#!/bin/bash
# Build Python wheel and Go FFI library in parallel
# This script is used by CI to build both artifacts concurrently.

set -euo pipefail

# Setup Rust environment
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install maturin and zig for manylinux cross-compilation
python3 -m pip install --upgrade pip maturin ziglang

# Start Go FFI build in background
echo "Starting Go FFI build in background..."
(cd bindings/golang && make build && echo "Go FFI: OK" && ls -la target/release/libsmg_go.*) &
GO_PID=$!

# Build Python wheel in foreground
echo "Building Python wheel..."
cd bindings/python
# `extension-module` is listed in pyproject.toml's `[tool.maturin] features`,
# but a command-line `--features` overrides that list rather than adding to it.
# Name it explicitly here: without it the wheel's .so links libpython and fails
# to load wherever that exact libpython is absent.
maturin build --profile ci --features extension-module,vendored-openssl --manylinux 2_28 --zig --out dist
echo "Python wheel: OK"
ls -lh dist/

# Backstop for the override above: every other `maturin --features` call site
# (release-pypi{,-dev}.yml, docker/Dockerfile, install-smg.sh, Makefile) has to
# repeat `extension-module`, and dropping it produces a wheel that imports fine
# on the build machine and fails everywhere else. Catch that here rather than at
# `pip install` time in a user's environment.
python3 - <<'PYEOF'
import pathlib, re, shutil, subprocess, sys, zipfile


def needed_libs(so: pathlib.Path) -> list[str]:
    """DT_NEEDED entries naming libpython, via whatever tool this image has."""
    for tool, args, pattern in (
        ("readelf", ["-d"], r"NEEDED.*\[(libpython[^\]]*)\]"),
        ("objdump", ["-p"], r"NEEDED\s+(libpython\S*)"),
    ):
        if shutil.which(tool):
            out = subprocess.run(
                [tool, *args, str(so)], capture_output=True, text=True, check=True
            ).stdout
            return re.findall(pattern, out)
    # No binutils in this image: the DT_NEEDED name lives verbatim in the
    # dynamic string table, so a byte scan of the stripped .so is a sound
    # (if blunt) stand-in.
    return [m.decode() for m in re.findall(rb"libpython[0-9.]*\.so[0-9.]*", so.read_bytes())]


wheels = sorted(pathlib.Path("dist").glob("*.whl"))
if not wheels:
    sys.exit("no wheel in dist/ to audit")
for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        modules = [n for n in archive.namelist() if n.endswith(".so")]
        if not modules:
            sys.exit(f"{wheel.name}: no extension module inside")
        for name in modules:
            so = pathlib.Path(archive.extract(name, "dist/.audit"))
            found = needed_libs(so)
            if found:
                sys.exit(
                    f"{wheel.name}: {name} links {sorted(set(found))} -- this "
                    "maturin invocation lost `--features extension-module`"
                )
            print(f"{wheel.name}: {name} does not link libpython")
PYEOF

# Wait for Go build to complete
echo "Waiting for Go FFI build..."
wait $GO_PID
GO_EXIT=$?
if [ $GO_EXIT -ne 0 ]; then
    echo "Go FFI build failed with exit code $GO_EXIT"
    exit $GO_EXIT
fi

echo "Both builds completed successfully"
