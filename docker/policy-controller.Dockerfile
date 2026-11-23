# The policy controller image: the standalone controller binary
# (crates/policy_controller, `smg-policy-controller`) and what it needs to
# run, nothing more: no Python and no media stack. The base and the toolchain
# are the gateway image's (docker/Dockerfile): the same pinned Ubuntu, and the
# Rust release rust-toolchain.toml names.
#
# From the repository root:
#   docker build --file docker/policy-controller.Dockerfile --tag smg-policy-controller .
#   docker run --rm --publish 30103:30103 \
#     --tmpfs /var/lib/smg/spec:uid=65532,gid=65532,mode=0700 \
#     smg-policy-controller --spec-store /var/lib/smg/spec
#
# A start names the source of its policy spec. The directory above is one the
# controller keeps, empty until the first apply and gone with the container:
# mount a volume there to keep it, or pass a document's file as --spec-file.
#
# An image for another architecture than the builder's (`--platform`) is
# cross-compiled: the build stage runs on the builder's own architecture, and
# the last stage's few commands alone run on the image's. Every build needs a
# builder that sets the platform arguments (BUILDPLATFORM, TARGETARCH and
# BUILDARCH).

FROM scratch AS local_src
COPY . /src

######################### BUILD IMAGE #########################
FROM --platform=$BUILDPLATFORM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852 AS build-image

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

# A C toolchain for the dependencies that compile C, protoc with the
# well-known types the engines' gRPC protocol imports, curl and the
# certificates for rustup and the crate registry.
RUN apt update -y \
    && apt install -y --no-install-recommends \
        build-essential ca-certificates curl libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# install rustup from rustup.rs; no default toolchain: the repository's
# rust-toolchain.toml selects the release once it is copied in (below)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal

WORKDIR /opt/smg

# install the pinned toolchain from rust-toolchain.toml alone, before the
# source is copied: the layer is rebuilt only when the pin changes
COPY --from=local_src /src/rust-toolchain.toml /opt/smg/rust-toolchain.toml
RUN rustup toolchain install \
    && rustc --version && cargo --version && protoc --version

# What a cross build needs, for an image of another architecture than the
# builder's: that architecture's C and C++ compilers and its C library (for
# the dependencies that compile C or C++, and for the link), its Rust standard
# library, and the linker's name for cargo. The one package named, the C++
# compiler's, brings the C compiler and the C library with it. /cross-target
# holds the Rust target's name for the build below. The builder sets both
# arguments; they are named here and not above, so the steps above are the
# same for every architecture.
ARG TARGETARCH
ARG BUILDARCH
RUN if [ "${TARGETARCH}" != "${BUILDARCH}" ]; then \
        case "${TARGETARCH}" in \
            arm64) cpu=aarch64 ;; \
            amd64) cpu=x86_64 ;; \
            *) echo "no cross build is named here for ${TARGETARCH}" >&2; exit 1 ;; \
        esac \
        && apt update -y \
        && apt install -y --no-install-recommends "g++-$(echo "${cpu}" | tr _ -)-linux-gnu" \
        && rm -rf /var/lib/apt/lists/* \
        && apt clean \
        && rustup target add "${cpu}-unknown-linux-gnu" \
        && printf '[target.%s]\nlinker = "%s"\n' "${cpu}-unknown-linux-gnu" "${cpu}-linux-gnu-gcc" \
            >> /root/.cargo/config.toml \
        && echo "${cpu}-unknown-linux-gnu" > /cross-target; \
    fi

COPY --from=local_src /src /opt/smg

# The workspace package whose binary the image runs: the stock controller, or
# with `--build-arg PACKAGE=<package>` a controller of your own built from the
# crate (a package that depends on smg-policy-controller; its binary carries
# the package's name and answers `--version`). The features are named with
# the crate, so the one line builds either: the listeners, the cache-event
# feed, the WebAssembly plugin host the crate turns on by default, and the
# Kubernetes Lease elector. The stock binary elects its leader over a Lease
# when a flag names one; the plugin host is compiled in for a controller of
# your own that runs it. The stock binary requires the listeners and the
# feed: without either feature cargo builds no binary and the install fails.
ARG PACKAGE=smg-policy-controller
# A cross build names its Rust target, and cargo then writes the binary under
# that name.
RUN target="$(cat /cross-target 2>/dev/null || true)" \
    && cargo build --release --locked ${target:+--target "${target}"} --package "${PACKAGE}" \
        --features smg-policy-controller/controller,smg-policy-controller/kubernetes,smg-policy-controller/worker-host \
    && install -D "target/${target:+${target}/}release/${PACKAGE}" /out/smg-policy-controller

######################### CONTROLLER IMAGE #########################
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852 AS policy-controller-image

# The root certificates are the one package the binary needs beyond the base
# (its shared libraries are the base's): the controller's HTTP client loads
# the system's roots as it is built, and with none the controller refuses to
# start. The user is the gateway image's: non-root, UID 65532 (the controller
# binds unprivileged ports only), named below by its number, which a kubelet
# can verify as non-root.
RUN apt update -y \
    && apt install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean \
    && useradd --system --no-log-init --no-create-home --uid 65532 smg

COPY --from=build-image /out/smg-policy-controller /usr/local/bin/smg-policy-controller

USER 65532

# The binary loads and runs in this stage, as this user, on the platform
# being built.
RUN ["/usr/local/bin/smg-policy-controller", "--version"]

# The allocator of an arm64 binary is built for 64 KiB pages, the largest an
# arm64 kernel runs with: a binary built for a smaller page ends at its start
# on such a kernel. The repository's cargo configuration sets the page for
# the target, and the allocator's statistics say the page it was built for.
# A controller of your own that registers no such allocator prints no
# statistics, and nothing is asked of it.
ARG TARGETARCH
RUN if [ "${TARGETARCH}" = "arm64" ]; then \
        stats="$(_RJEM_MALLOC_CONF=stats_print:true /usr/local/bin/smg-policy-controller --version 2>&1)" \
        && if printf '%s\n' "${stats}" | grep -qF "Begin jemalloc statistics"; then \
            printf '%s\n' "${stats}" | grep -F "Page size: 65536"; \
        fi; \
    fi

# The listeners' default ports: the gateways and the relays (30100), the
# operators (30101), the standbys (30102) and the HTTP admin listener with
# the probes and the metrics (30103).
EXPOSE 30100 30101 30102 30103

ENTRYPOINT ["/usr/local/bin/smg-policy-controller"]
