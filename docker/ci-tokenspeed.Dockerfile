# CI image carrying TokenSpeed nightly wheels (engine + kernel + scheduler)
# for the tokenspeed e2e lanes.
#
# The GPU e2e jobs run BARE on the k8s runner pods (no job container —
# vars.SMG_CI_GPU_CONTAINER_IMAGE has never been set, so the container: line
# in e2e-gpu-job.yml resolves to "no container"). This image is therefore a
# CARRIER, not a job container: at job time
# scripts/ci_fetch_tokenspeed_prebuilt.sh pulls it and docker-cp's the baked
# payload onto the runner, where scripts/ci_install_tokenspeed.sh upgrades
# TokenSpeed from the nightly index.
#
# Portability contract with the runner (Ubuntu 24.04 pods):
#   - base is ubuntu:24.04 so the baked venv's interpreter symlink resolves
#     to the same /usr/bin path on the runner (the venv-adoption check in
#     ci_setup_python_venv.sh re-validates the interpreter there);
#   - the payload lives under /opt/smg-ci (venv + stamp),
#     extracted to identical absolute paths;
#   - the CUDA toolkit and the Python dev headers are NOT part of the
#     payload; the install script apt-installs them on the runner when
#     missing.
#
# Built by ci-tokenspeed-image.yml using the published nightly version date.
# CUDA tooling supports runtime JIT; wheel installation needs no GPU.

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

WORKDIR /opt/smg-ci

# Copied ahead of the apt bootstrap so that bootstrap can retry and fail
# over to a mirror too.
COPY scripts/ci_retry.sh scripts/ci_apt_mirror.sh scripts/

# Build prerequisites the bare base lacks (the runner pods already carry
# these). python3 is 3.12 on noble, matching the CI interpreter pin.
# python3-dev provides Python.h for runtime JIT compilation.
ENV DEBIAN_FRONTEND=noninteractive
RUN bash scripts/ci_apt_mirror.sh \
    && bash scripts/ci_retry.sh 3 10 apt-get update \
    && bash scripts/ci_retry.sh 3 10 apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential pkg-config \
        python3 python3-dev python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/ci_setup_python_venv.sh scripts/ci_install_tokenspeed.sh scripts/ci_install_flashinfer.py scripts/ci_ensure_python_headers.sh scripts/
COPY .github/versions/tokenspeed.ref .github/versions/tokenspeed.ref

ARG TOKENSPEED_VERSION

# Bake nightly wheels; per-PR SMG gRPC packages are installed in each job.
RUN bash scripts/ci_setup_python_venv.sh \
    && TOKENSPEED_BUILD_ONLY=1 TOKENSPEED_VERSION="${TOKENSPEED_VERSION}" \
       bash scripts/ci_install_tokenspeed.sh \
    && if [ -x "$HOME/.local/bin/uv" ] && [ ! -x /usr/local/bin/uv ]; then \
           cp "$HOME/.local/bin/uv" /usr/local/bin/uv; \
       fi \
    && rm -rf /root/.cache/uv /root/.cache/pip /var/lib/apt/lists/*
