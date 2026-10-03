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
FROM alpine:3.19 AS sources
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

RUN bash /tmp/scripts/install-smg.sh /opt/smg-src

RUN case "${ENGINE}" in \
      vllm|sglang|trtllm|tgl|tokenspeed) ;; \
      *) echo "ERROR: Unknown ENGINE '${ENGINE}'" >&2; exit 1 ;; \
    esac \
    && if [ -n "${ENGINE_REPO}" ]; then \
         bash /tmp/scripts/install-${ENGINE}.sh /opt/engine-src; \
       fi

ENTRYPOINT ["smg"]
