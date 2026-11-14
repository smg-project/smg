# The policy controller image: the standalone controller binary
# (crates/policy_controller, `smg-policy-controller`) and what it needs to
# run, nothing more: no Python and no media stack. The base and the toolchain
# are the gateway image's (docker/Dockerfile): the same pinned Ubuntu, and the
# Rust release rust-toolchain.toml names.
#
# From the repository root:
#   docker build --file docker/policy-controller.Dockerfile --tag smg-policy-controller .
#   docker run --rm --publish 30103:30103 smg-policy-controller

FROM scratch AS local_src
COPY . /src

######################### BUILD IMAGE #########################
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852 AS build-image

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

COPY --from=local_src /src /opt/smg

# The workspace package whose binary the image runs: the stock controller, or
# with `--build-arg PACKAGE=<package>` a controller of your own built from the
# crate (a package that depends on smg-policy-controller; its binary carries
# the package's name and answers `--version`). The features are named with
# the crate, so the one line builds either: the listeners, the WebAssembly
# plugin host the crate turns on by default, and the Kubernetes Lease elector
# for a controller of your own that runs it. The stock binary compiles the
# elector in and does not elect yet.
ARG PACKAGE=smg-policy-controller
RUN cargo build --release --locked --package "${PACKAGE}" \
        --features smg-policy-controller/controller,smg-policy-controller/kubernetes \
    && install -D "target/release/${PACKAGE}" /out/smg-policy-controller

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

# The listeners' default ports: the gateways and the relays (30100), the
# operators (30101), the standbys (30102) and the HTTP admin listener with
# the probes and the metrics (30103).
EXPOSE 30100 30101 30102 30103

ENTRYPOINT ["/usr/local/bin/smg-policy-controller"]
