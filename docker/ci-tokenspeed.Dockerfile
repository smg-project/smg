# CI image carrying a fully-built TokenSpeed (engine + kernel + scheduler)
# for the tokenspeed e2e lanes.
#
# The GPU e2e jobs run BARE on the k8s runner pods (no job container —
# vars.SMG_CI_GPU_CONTAINER_IMAGE has never been set, so the container: line
# in e2e-gpu-job.yml resolves to "no container"). This image is therefore a
# CARRIER, not a job container: at job time
# scripts/ci_fetch_tokenspeed_prebuilt.sh pulls it and docker-cp's the baked
# payload onto the runner, where scripts/ci_install_tokenspeed.sh's stamp
# fast path takes over.
#
# Portability contract with the runner (Ubuntu 24.04 pods):
#   - base is ubuntu:24.04 so the baked venv's interpreter symlink resolves
#     to the same /usr/bin path on the runner (the venv-adoption check in
#     ci_setup_python_venv.sh re-validates the interpreter there);
#   - the payload lives under /opt/smg-ci (venv + stamp) and
#     /opt/tokenspeed-src (checkout the venv's editable installs point at),
#     extracted to identical absolute paths;
#   - the CUDA toolkit and the Python dev headers are NOT part of the
#     payload; the install script apt-installs them on the runner when
#     missing, same as the source path.
#
# Built by .github/workflows/ci-tokenspeed-image.yml on every bump of
# .github/versions/tokenspeed.ref or of the image tooling, and tagged
# ghcr.io/<owner>/smg:<tag> with the content-addressed tag printed by
# scripts/ci_tokenspeed_image_tag.sh. The build needs nvcc but no GPU
# (the kernel arch list is pinned in the install script), so it runs on the
# CPU/docker runner pool.

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# Build prerequisites the bare base lacks (the runner pods already carry
# these). python3 is 3.12 on noble, matching the CI interpreter pin.
# python3-dev: tokenspeed-scheduler's CMake does
# find_package(Python COMPONENTS Interpreter Development.Module), which
# needs Python.h — without it the configure step fails.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential pkg-config \
        python3 python3-dev python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/smg-ci

COPY scripts/ci_setup_python_venv.sh scripts/ci_install_tokenspeed.sh scripts/
COPY .github/versions/tokenspeed.ref .github/versions/tokenspeed.ref

# Keep the TokenSpeed checkout inside the image: the venv's editable installs
# point at it. uv is promoted to /usr/local/bin for anyone exec-ing into the
# image directly; the job-side fast path installs its own (per-job HOME
# differs from the build-time HOME).
RUN bash scripts/ci_setup_python_venv.sh \
    && TOKENSPEED_BUILD_ONLY=1 TOKENSPEED_DIR=/opt/tokenspeed-src \
       bash scripts/ci_install_tokenspeed.sh \
    && if [ -x "$HOME/.local/bin/uv" ] && [ ! -x /usr/local/bin/uv ]; then \
           cp "$HOME/.local/bin/uv" /usr/local/bin/uv; \
       fi \
    && rm -rf /root/.cache/uv /root/.cache/pip /var/lib/apt/lists/*
