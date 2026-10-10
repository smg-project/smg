# Extend an existing Ubuntu 24.04 self-hosted runner image on a CPU builder.
# No GPU is required for the build.
#
# docker build -f docker/ci-gpu-runner.Dockerfile \
#   --build-arg BASE_IMAGE=<existing-runner-image> \
#   --build-arg UV_CUDA_DRIVER_VERSION=<target-driver-version> -t <local-tag> .
#
# Each Pod gets its own writable image layer. The two fixed-path environments
# remain separate; per-PR SMG packages are installed only after checkout.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root
ARG RUNNER_USER=runner
ARG UV_CUDA_DRIVER_VERSION

COPY scripts/ci_retry.sh scripts/ci_apt_mirror.sh \
     scripts/ci_setup_python_venv.sh scripts/ci_prepared_backend_env.sh \
     scripts/ci_ensure_python_headers.sh scripts/ci_install_vllm.sh \
     scripts/ci_install_sglang.sh scripts/ci_install_flashinfer_jit_cache.sh /opt/smg-ci/scripts/

# Preserve vLLM's runtime-only CUDA environment. The SGLang toolkit is kept
# at its versioned path and selected only by the SGLang setup action.
RUN . /etc/os-release && test "$ID" = ubuntu && test "$VERSION_ID" = 24.04 \
    && test -n "${UV_CUDA_DRIVER_VERSION}" \
    && test ! -x /usr/local/cuda/bin/nvcc && ! command -v nvcc \
    && bash /opt/smg-ci/scripts/ci_apt_mirror.sh \
    && bash /opt/smg-ci/scripts/ci_retry.sh 3 10 apt-get update \
    && bash /opt/smg-ci/scripts/ci_retry.sh 3 10 apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential pkg-config python3 python3-dev python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/smg-ci/vllm && cd /opt/smg-ci/vllm \
    && SMG_BUILD_PREPARED_ENV=1 bash ../scripts/ci_setup_python_venv.sh \
    && UV_BIN="$(PATH="$HOME/.local/bin:$PATH" command -v uv)" \
    && if [ "$UV_BIN" != /usr/local/bin/uv ]; then install -m 755 "$UV_BIN" /usr/local/bin/uv; fi \
    && SMG_BUILD_PREPARED_ENV=1 UV_CUDA_DRIVER_VERSION="${UV_CUDA_DRIVER_VERSION}" \
       bash ../scripts/ci_install_vllm.sh \
    && /usr/local/bin/uv cache clean \
    && chown -R "${RUNNER_USER}" /opt/smg-ci/vllm \
    && rm -rf "$HOME/.cache/pip" /var/lib/apt/lists/*

RUN mkdir /opt/smg-ci/sglang && cd /opt/smg-ci/sglang \
    && bash ../scripts/ci_setup_python_venv.sh \
    && SMG_BUILD_PREPARED_ENV=1 bash ../scripts/ci_install_sglang.sh \
    && if [ -L /usr/local/cuda ]; then rm /usr/local/cuda; fi \
    && test ! -x /usr/local/cuda/bin/nvcc && ! command -v nvcc \
    && /usr/local/bin/uv cache clean \
    && chown -R "${RUNNER_USER}" /opt/smg-ci/sglang \
    && rm -rf "$HOME/.cache/pip" /var/lib/apt/lists/*

USER ${RUNNER_USER}
