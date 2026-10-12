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
# apt-cache's output is translated, so it is read in the C locale.
turbojpeg_runtime_package() {
    local pkg
    for pkg in libturbojpeg libturbojpeg0; do
        if LC_ALL=C apt-cache policy "$pkg" 2>/dev/null | grep -q '^  Candidate: [^(]'; then
            echo "$pkg"
            return 0
        fi
    done
    echo "install-smg.sh: neither libturbojpeg nor libturbojpeg0 is available on this base" >&2
    return 1
}

apt update -y
TURBOJPEG_PKG="$(turbojpeg_runtime_package)"
apt install -y git build-essential libssl-dev pkg-config unzip wget "${TURBOJPEG_PKG}" \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean
# The soname the gateway loads has to resolve here, or every image built from
# this script would decode JPEGs with the fallback without anyone noticing.
python3 -c 'import ctypes; ctypes.CDLL("libturbojpeg.so.0")'

cd /tmp && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v32.0/protoc-32.0-linux-x86_64.zip && \
    unzip protoc-32.0-linux-x86_64.zip -d /usr/local && \
    rm protoc-32.0-linux-x86_64.zip
protoc --version

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version && protoc --version

export PATH="/root/.cargo/bin:${PATH}"

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
