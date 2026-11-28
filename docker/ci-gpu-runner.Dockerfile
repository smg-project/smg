# Multi-stage CI GPU environment builder (vLLM, SGLang, TRT-LLM) on CPU.
# No GPU is required for the build.
#
# Per-backend carrier targets (consumed on bare k8s GPU pods via
# ci_fetch_backend_prebuilt.sh and cached on node NVMe /models):
#   docker build -f docker/ci-gpu-runner.Dockerfile --target vllm \
#     --build-arg UV_CUDA_DRIVER_VERSION=580.65.06 -t <tag> .
#   docker build -f docker/ci-gpu-runner.Dockerfile --target sglang -t <tag> .
#   docker build -f docker/ci-gpu-runner.Dockerfile --target trtllm -t <tag> .
#
# Combined self-hosted runner image (default target `runner`, extending an
# Ubuntu 24.04 actions-runner image with all three fixed-path environments):
#   docker build -f docker/ci-gpu-runner.Dockerfile \
#     --build-arg BASE_IMAGE=<existing-runner-image> \
#     --build-arg UV_CUDA_DRIVER_VERSION=<target-driver-version> -t <tag> .
#
# Each Pod gets its own writable layer (or per-job NVMe reflink copy). The
# fixed-path environments remain separate; per-PR SMG packages are installed
# only after checkout.

ARG BASE_IMAGE=ghcr.io/actions/actions-runner:2.337.0@sha256:5036480998280bb21e32ade9fe1b02b493861ac314b62ba1aea320b94f56ec97
ARG CARRIER_BASE_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

FROM ${CARRIER_BASE_IMAGE} AS carrier-base

ENV DEBIAN_FRONTEND=noninteractive

COPY scripts/ci_retry.sh scripts/ci_apt_mirror.sh \
     scripts/ci_setup_python_venv.sh scripts/ci_prepared_backend_env.sh \
     scripts/ci_ensure_python_headers.sh scripts/ci_install_vllm.sh \
     scripts/ci_install_sglang.sh scripts/ci_install_trtllm.sh \
     scripts/ci_install_flashinfer_jit_cache.sh /opt/smg-ci/scripts/

RUN . /etc/os-release && test "$ID" = ubuntu && test "$VERSION_ID" = 24.04 \
    && test ! -x /usr/local/cuda/bin/nvcc && ! command -v nvcc \
    && bash /opt/smg-ci/scripts/ci_apt_mirror.sh \
    && bash /opt/smg-ci/scripts/ci_retry.sh 3 10 apt-get update \
    && bash /opt/smg-ci/scripts/ci_retry.sh 3 10 apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential pkg-config python3 python3-dev python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Per-backend carrier: vLLM ────────────────────────────────────────────────
FROM carrier-base AS vllm

ARG UV_CUDA_DRIVER_VERSION=580.65.06

RUN test -n "${UV_CUDA_DRIVER_VERSION}" \
    && mkdir -p /opt/smg-ci/vllm && cd /opt/smg-ci/vllm \
    && SMG_BUILD_PREPARED_ENV=1 bash ../scripts/ci_setup_python_venv.sh \
    && UV_BIN="$(PATH="$HOME/.local/bin:$PATH" command -v uv)" \
    && if [ "$UV_BIN" != /usr/local/bin/uv ]; then install -m 755 "$UV_BIN" /usr/local/bin/uv; fi \
    && SMG_BUILD_PREPARED_ENV=1 UV_CUDA_DRIVER_VERSION="${UV_CUDA_DRIVER_VERSION}" \
       bash ../scripts/ci_install_vllm.sh \
    && /usr/local/bin/uv cache clean \
    && rm -rf "$HOME/.cache/pip" /var/lib/apt/lists/*

# ── Per-backend carrier: SGLang (+ CUDA 13.0 nvcc) ───────────────────────────
FROM carrier-base AS sglang

RUN mkdir -p /opt/smg-ci/sglang && cd /opt/smg-ci/sglang \
    && SMG_BUILD_PREPARED_ENV=1 bash ../scripts/ci_setup_python_venv.sh \
    && UV_BIN="$(PATH="$HOME/.local/bin:$PATH" command -v uv)" \
    && if [ "$UV_BIN" != /usr/local/bin/uv ]; then install -m 755 "$UV_BIN" /usr/local/bin/uv; fi \
    && SMG_BUILD_PREPARED_ENV=1 bash ../scripts/ci_install_sglang.sh \
    && ln -sfn /usr/local/cuda-13.0 /usr/local/cuda \
    && test -x /usr/local/cuda/bin/nvcc \
    && /usr/local/bin/uv cache clean \
    && rm -rf "$HOME/.cache/pip" /var/lib/apt/lists/*

# ── Per-backend carrier: TRT-LLM (+ CUDA 13.0 toolkit) ───────────────────────
FROM carrier-base AS trtllm

RUN mkdir -p /opt/smg-ci/trtllm && cd /opt/smg-ci/trtllm \
    && bash ../scripts/ci_setup_python_venv.sh \
    && SMG_BUILD_PREPARED_ENV=1 bash ../scripts/ci_install_trtllm.sh \
    && ln -sfn /usr/local/cuda-13.0 /usr/local/cuda \
    && test -x /usr/local/cuda/bin/nvcc \
    && rm -rf "$HOME/.cache/pip" /var/lib/apt/lists/*

# ── Combined self-hosted GPU runner image (default target) ───────────────────
FROM ${BASE_IMAGE} AS runner

USER root
ARG RUNNER_USER=runner
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=carrier-base /opt/smg-ci/scripts/ /opt/smg-ci/scripts/
COPY --from=vllm /usr/local/bin/uv /usr/local/bin/uv
COPY --from=vllm /opt/smg-ci/vllm /opt/smg-ci/vllm
COPY --from=sglang /opt/smg-ci/sglang /opt/smg-ci/sglang
COPY --from=trtllm /opt/smg-ci/trtllm /opt/smg-ci/trtllm
COPY --from=trtllm /usr/local/cuda-13.0 /usr/local/cuda-13.0

# Install host runtime libraries required by all three baked environments
# (Python headers, FFmpeg for vLLM torchcodec, RDMA libs for Mooncake, OpenMPI
# + TensorRT runtime libs for TRT-LLM) and keep the /usr/local/cuda alias.
RUN . /etc/os-release && test "$ID" = ubuntu && test "$VERSION_ID" = 24.04 \
    && bash /opt/smg-ci/scripts/ci_retry.sh 3 10 curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm -f /tmp/cuda-keyring.deb \
    && bash /opt/smg-ci/scripts/ci_apt_mirror.sh \
    && bash /opt/smg-ci/scripts/ci_retry.sh 3 10 apt-get update \
    && bash /opt/smg-ci/scripts/ci_retry.sh 3 10 apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential pkg-config python3 python3-dev python3-venv python3-pip \
        ffmpeg libnuma-dev libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils \
        libopenmpi-dev libnvinfer10 \
    && ln -sfn /usr/local/cuda-13.0 /usr/local/cuda \
    && test -x /usr/local/cuda/bin/nvcc \
    && chown -R "${RUNNER_USER}" /opt/smg-ci/vllm /opt/smg-ci/sglang /opt/smg-ci/trtllm \
    && rm -rf /var/lib/apt/lists/*

USER ${RUNNER_USER}
