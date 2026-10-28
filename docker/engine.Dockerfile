# Parameterized engine image builder.
#
# Build args:
#   BASE_IMAGE_REF  - full image:tag to start FROM
#   ENGINE_BASE_STAGE - engine-base, or tokenspeed-nightly-base for CI carriers
#   ENGINE          - engine name: vllm | sglang | trtllm | tgl | tokenspeed
#   BACKEND         - SMG_DEFAULT_BACKEND value (defaults to ENGINE; tgl overrides to sglang)
#   ENGINE_REPO     - if set, engine source is cloned and install-<ENGINE>.sh runs
#   ENGINE_COMMIT   - commit/ref for ENGINE_REPO ("latest" = HEAD)
#   SMG_REPO        - SMG source repo URL
#   SMG_COMMIT      - commit/ref for SMG_REPO ("latest" = HEAD)
#
# Usage:
#   docker build --build-arg BASE_IMAGE_REF=lmsysorg/sglang:v0.5.10 \
#                --build-arg ENGINE=sglang \
#                --build-arg SMG_REPO=https://github.com/smg-project/smg \
#                --build-arg SMG_COMMIT=v1.1.0 \
#                -f docker/engine.Dockerfile .

ARG BASE_IMAGE_REF
ARG ENGINE_BASE_STAGE=engine-base

# ── sources stage: clone repos, stage install scripts ────────────────────────
FROM alpine:3.19@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1 AS sources
ARG ENGINE_REPO
ARG ENGINE_COMMIT
ARG SMG_REPO
ARG SMG_COMMIT
RUN apk add --no-cache git \
    && if [ -z "${SMG_REPO}" ] || [ -z "${SMG_COMMIT}" ]; then \
         echo "ERROR: SMG_REPO and SMG_COMMIT must be set" >&2; exit 1; fi \
    && if [ -n "${ENGINE_REPO}" ] && [ -n "${ENGINE_COMMIT}" ]; then \
         if [ "${ENGINE_COMMIT}" = "latest" ]; then \
           git clone --depth 1 "${ENGINE_REPO}" /opt/engine-src; \
         else \
           git clone "${ENGINE_REPO}" /opt/engine-src \
           && ( cd /opt/engine-src && git checkout "${ENGINE_COMMIT}" ); \
         fi; \
       else mkdir -p /opt/engine-src; fi \
    && if [ "${SMG_COMMIT}" = "latest" ]; then \
         git clone --depth 1 "${SMG_REPO}" /tmp/smg-src; \
       else \
         git clone "${SMG_REPO}" /tmp/smg-src \
         && ( cd /tmp/smg-src && git checkout "${SMG_COMMIT}" ); \
       fi
COPY scripts/installation/ /tmp/scripts/

FROM ${BASE_IMAGE_REF} AS engine-base

# The nightly carrier installs TokenSpeed into a venv. Use that interpreter for
# both SMG installation and runtime, and expose the carrier's CUDA toolkit.
FROM engine-base AS tokenspeed-nightly-base
ENV VIRTUAL_ENV=/opt/smg-ci/.venv \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/opt/smg-ci/.venv/bin:/usr/local/cuda-13.0/bin:/root/.cargo/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
    CPATH=/usr/local/cuda-13.0/include:/usr/local/cuda-13.0/include/cccl${CPATH:+:${CPATH}} \
    CPLUS_INCLUDE_PATH=/usr/local/cuda-13.0/include:/usr/local/cuda-13.0/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}} \
    C_INCLUDE_PATH=/usr/local/cuda-13.0/include:/usr/local/cuda-13.0/include/cccl${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}

# ── final stage: install SMG + conditionally install engine ──────────────────
FROM ${ENGINE_BASE_STAGE}

ARG ENGINE=sglang
ARG BACKEND
ARG ENGINE_REPO

ENV SMG_DEFAULT_BACKEND=${BACKEND:-${ENGINE}}

COPY --from=sources /opt/engine-src   /opt/engine-src
COPY --from=sources /tmp/smg-src      /opt/smg-src
COPY --from=sources /tmp/scripts/     /tmp/scripts/

# NGC TRT-LLM 1.3.0rc bases ship a debian-owned PyYAML with no pip RECORD
# file, which pip refuses to uninstall when install-smg.sh resolves smg's
# pyyaml dependency ("uninstall-no-record-file"). Shadow it with a pip-managed
# copy (--ignore-installed leaves the distro files alone); the smg install can
# then upgrade pyyaml normally. Scoped to trtllm so the other engine bases
# keep their exact pip behavior.
RUN if [ "${ENGINE}" = "trtllm" ]; then \
      pip install --no-cache-dir --ignore-installed pyyaml; \
    fi

# TokenSpeed prep. The base is a debian-system-Python image, so it needs the
# same debian-shadow treatment as the NGC trtllm base plus a gRPC cleanup:
#  1. pip/pyyaml are debian-owned with no pip RECORD file, so install-smg.sh's
#     `pip install --upgrade pip` (and the smg wheel's pyyaml dep) fail with
#     "Cannot uninstall … RECORD file not found". Shadow them with pip-managed
#     copies (--ignore-installed leaves the distro files alone) so the smg
#     install can upgrade them normally.
#  2. The base bakes in `tokenspeed-smg-grpc-proto` / `tokenspeed-smg-grpc-servicer`,
#     which claim the same `smg_grpc_proto` / `smg_grpc_servicer` import paths
#     that install-smg.sh reinstalls from source. Left in place they can shadow
#     the source installs and serve stale proto descriptors ("Method not
#     found!"). Drop them first so SMG's own gRPC modules win.
# Scoped to tokenspeed so the other engine bases keep their exact pip behavior.
RUN if [ "${ENGINE}" = "tokenspeed" ]; then \
      pip install --no-cache-dir --ignore-installed pip pyyaml; \
      pip uninstall -y tokenspeed-smg-grpc-proto tokenspeed-smg-grpc-servicer || true; \
    fi

# UCX reads /etc/ucx/ucx.conf (its compiled-in config directory) in every
# process that creates a UCX context, so a setting placed there reaches the
# engine's tensor-parallel workers whatever they inherit of the container's
# environment. The file raises the TCP transport's send segment from UCX's
# 8 KiB default (receive: 64 KiB) to 1 MiB: a prefill/decode pair over NIXL across
# hosts on TCP then pulls KV at the socket's rate instead of a fifth of it
# (docker/ucx.conf has the measurement). Only the tcp transport reads the two
# keys; RDMA and CUDA IPC paths are unchanged. A UCX built with another config
# directory (TRT-LLM's /usr/local/ucx) or an engine with no UCX never opens it.
COPY docker/ucx.conf /etc/ucx/ucx.conf

RUN bash /tmp/scripts/install-smg.sh /opt/smg-src

# TokenSpeed ships flashinfer's prebuilt kernel cache (flashinfer-jit-cache) pinned to the exact flashinfer-python
# version. When the two drift (an engine upgrade pulls a newer flashinfer-python and leaves the cache behind), flashinfer
# ignores the cache with a warning and JIT-compiles every kernel at the worker's first start, one compiler job per host CPU:
# on a 144-core node that start-up exceeded a 128 GiB container limit on both roles of a prefill/decode pair. The CI
# carrier re-aligns the three flashinfer artifacts after the TokenSpeed install (scripts/ci_install_flashinfer.py); this
# check turns any drift that reaches an engine image into a build failure instead of a start-up one: the versions of the
# three artifacts (flashinfer-cubin drifts the same silent way, with on-demand cubin downloads at first use) and then
# flashinfer's own acceptance of every cache provider the shim offers (it drops a provider whose version differs).
RUN if [ "${ENGINE}" = "tokenspeed" ]; then \
      python3 -c "import importlib.metadata as m, importlib.util, sys; fi = m.version('flashinfer-python'); jc = m.version('flashinfer-jit-cache') if importlib.util.find_spec('flashinfer_jit_cache') else None; cb = m.version('flashinfer-cubin') if importlib.util.find_spec('flashinfer_cubin') else None; ok = jc is not None and (jc == fi or jc.startswith(fi + '+')) and (cb is None or cb == fi); print('flashinfer-python', fi, 'flashinfer-jit-cache', jc, 'flashinfer-cubin', cb, 'match' if ok else 'MISMATCH'); sys.exit(0 if ok else 1)" \
      && python3 -c "import flashinfer.jit.env as env, flashinfer_jit_cache as jc, sys; ids = [p.provider_id for p in env.FLASHINFER_AOT_PROVIDERS]; n = len(list(jc.get_jit_cache_providers())); print('flashinfer AOT providers accepted:', ids, 'of', n); sys.exit(0 if ids and len(ids) == n else 1)" ; \
    fi

RUN case "${ENGINE}" in \
      vllm|sglang|trtllm|tgl|tokenspeed) ;; \
      *) echo "ERROR: Unknown ENGINE '${ENGINE}'" >&2; exit 1 ;; \
    esac \
    && if [ -n "${ENGINE_REPO}" ]; then \
         bash /tmp/scripts/install-${ENGINE}.sh /opt/engine-src; \
       fi

ENTRYPOINT ["smg"]
