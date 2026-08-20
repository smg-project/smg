#!/bin/bash
set -e

# Install smg from source.
# Usage: install-smg.sh [path-to-smg-src]
# Default path: /tmp/smg-src

export MAKEFLAGS="-j$(nproc)"

SMG_SRC="${1:-/tmp/smg-src}"

apt update -y \
    && apt install -y git build-essential libssl-dev pkg-config unzip wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

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
ulimit -n 65536 && maturin build --release --locked --features vendored-openssl --out dist
pip install --force-reinstall dist/*.whl

# Install smg-grpc-proto and smg-grpc-servicer from source so the image stays
# in sync with this repo. --force-reinstall overrides any stale version the
# engine base image may have preinstalled (e.g. lmsysorg/sglang ships an older
# smg-grpc-servicer pulled from PyPI at its own build time).
pip install --no-cache-dir --force-reinstall \
    "${SMG_SRC}/crates/grpc_client/python" \
    "${SMG_SRC}/grpc_servicer"
