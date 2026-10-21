#!/bin/bash
set -e

# Install smg from source.
# Usage: install-smg.sh [path-to-smg-src]
# Default path: /tmp/smg-src

export MAKEFLAGS="-j$(nproc)"

SMG_SRC="${1:-/tmp/smg-src}"

# The gateway decodes JPEGs through libjpeg-turbo's TurboJPEG library when it
# can load libturbojpeg.so.0, so its pixels match an engine that decodes with
# Pillow; without the library it falls back to a pure-Rust decoder whose pixels
# differ by a few levels (the gateway warns about it at start-up). The runtime
# package is libturbojpeg on Ubuntu bases and libturbojpeg0 on Debian bases:
# take whichever this base offers instead of guessing from the distribution.
# apt-cache's output is translated, so it is read in the C locale. The same
# for libwebp, which decodes WebP files several times faster than the
# pure-Rust fallback with the same pixels: libwebp7 on current bases, libwebp6
# on older ones.
runtime_package() {
    local pkg
    for pkg in "$@"; do
        if LC_ALL=C apt-cache policy "$pkg" 2>/dev/null | grep -q '^  Candidate: [^(]'; then
            echo "$pkg"
            return 0
        fi
    done
    echo "install-smg.sh: none of $* is available on this base" >&2
    return 1
}

apt update -y
TURBOJPEG_PKG="$(runtime_package libturbojpeg libturbojpeg0)"
WEBP_PKG="$(runtime_package libwebp7 libwebp6)"
apt install -y git build-essential libssl-dev pkg-config unzip wget "${TURBOJPEG_PKG}" "${WEBP_PKG}" \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean
# The sonames the gateway loads have to resolve here, or every image built from
# this script would decode JPEGs and WebP files with the fallbacks without
# anyone noticing.
python3 -c 'import ctypes; ctypes.CDLL("libturbojpeg.so.0")'
python3 - <<'PY'
import ctypes
for name in ("libwebp.so.7", "libwebp.so.6"):
    try:
        ctypes.CDLL(name)
        break
    except OSError:
        pass
else:
    raise SystemExit("install-smg.sh: no libwebp soname resolves")
PY

cd /tmp && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v32.0/protoc-32.0-linux-x86_64.zip && \
    unzip protoc-32.0-linux-x86_64.zip -d /usr/local && \
    rm protoc-32.0-linux-x86_64.zip
protoc --version

# rustup installs under CARGO_HOME (/root/.cargo in the image). A base that
# already carries rustup keeps it: rustup-init over an existing /root/.rustup
# fails on some bases (a cross-device rename inside the image's layers), and
# rustup installs the toolchain rust-toolchain.toml pins at the first cargo
# call in the checkout anyway. The download and the installer are separate
# statements so that each fails on its own: a pipeline's status is its last
# command's, and `sh` reading an empty script exits 0.
CARGO_BIN="${CARGO_HOME:-${HOME}/.cargo}/bin"
if [ -x "${CARGO_BIN}/rustup" ] || command -v rustup >/dev/null 2>&1; then
    echo "install-smg.sh: rustup is already installed on this base; keeping it"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
    sh /tmp/rustup-init.sh -y
    rm /tmp/rustup-init.sh
fi
export PATH="${CARGO_BIN}:${PATH}"

# The version lines go to the build log on purpose, read from the source
# checkout so that rustup names (and installs, if need be) the toolchain
# rust-toolchain.toml pins, the one that builds smg below; one statement each:
# under `set -e` a failing member of an `&&` list other than the last does not
# stop the script, so a toolchain that does not answer must fail here rather
# than at the first cargo call.
cd "${SMG_SRC}"
rustc --version
cargo --version
protoc --version

pip install --no-cache-dir --upgrade pip \
    && pip install maturin --no-cache-dir --force-reinstall

cd "${SMG_SRC}/bindings/python"
ulimit -n 65536 && maturin build --release --features vendored-openssl --out dist
pip install --force-reinstall dist/*.whl

# Install smg-grpc-proto and smg-grpc-servicer from source so the image stays
# in sync with this repo. --force-reinstall overrides any stale version the
# engine base image may have preinstalled (e.g. lmsysorg/sglang ships an older
# smg-grpc-servicer pulled from PyPI at its own build time).
pip install --no-cache-dir --force-reinstall \
    "${SMG_SRC}/crates/grpc_client/python" \
    "${SMG_SRC}/grpc_servicer"
