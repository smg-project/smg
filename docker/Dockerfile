######################## FFMPEG IMAGE ########################
FROM ubuntu:24.04 AS ffmpeg-image

# TODO: Add a scheduled updater for these source-built media dependencies.
# Version and SHA256 must be updated together.
ARG FFMPEG_VERSION=8.0
ARG FFMPEG_SHA256=b2751fccb6cc4c77708113cd78b561059b6fa904b24162fa0be2d60273d27b8e
ARG FFMPEG_PREFIX=/opt/ffmpeg

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y \
    && apt install -y \
        ca-certificates \
        cmake \
        curl \
        build-essential \
        nasm \
        ninja-build \
        pkg-config \
        python3 \
        xz-utils \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# The sha256 pin makes any mirror integrity-safe; OSUOSL is independent
# infrastructure for when ffmpeg.org is unreachable.
RUN mkdir -p /tmp/ffmpeg \
    && { curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 --connect-timeout 30 \
            --max-time 120 --retry-max-time 300 \
            "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o /tmp/ffmpeg.tar.xz \
        || curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 --connect-timeout 30 \
            --max-time 120 --retry-max-time 300 \
            "https://ftp.osuosl.org/pub/blfs/conglomeration/ffmpeg/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o /tmp/ffmpeg.tar.xz; } \
    && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
    && tar -C /tmp/ffmpeg --strip-components=1 -xJf /tmp/ffmpeg.tar.xz \
    && cd /tmp/ffmpeg \
    && ./configure \
        --prefix=${FFMPEG_PREFIX} \
        --enable-shared \
        --disable-static \
        --enable-pic \
        --disable-doc \
        --disable-debug \
        --disable-ffplay \
        --disable-avdevice \
        --disable-network \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

ENV PATH="${FFMPEG_PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib"
ENV PKG_CONFIG_PATH="${FFMPEG_PREFIX}/lib/pkgconfig"

######################## OPENCV IMAGE ########################
FROM ffmpeg-image AS opencv-image

ARG OPENCV_VERSION=4.13.0
ARG OPENCV_SHA256=1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d
ARG FFMPEG_PREFIX=/opt/ffmpeg
ARG OPENCV_PREFIX=/opt/opencv

RUN mkdir -p /tmp/opencv \
    && curl -fsSL "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" -o /tmp/opencv.tar.gz \
    && echo "${OPENCV_SHA256}  /tmp/opencv.tar.gz" | sha256sum -c - \
    && tar -C /tmp/opencv --strip-components=1 -xzf /tmp/opencv.tar.gz \
    && cmake -S /tmp/opencv -B /tmp/opencv-build -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=${OPENCV_PREFIX} \
        -D CMAKE_PREFIX_PATH=${FFMPEG_PREFIX} \
        -D PKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_LIST=core,imgproc,videoio \
        -D BUILD_SHARED_LIBS=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_JAVA=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=OFF \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=OFF \
        -D WITH_GTK=OFF \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=OFF \
        -D WITH_TBB=OFF \
    && cmake --build /tmp/opencv-build --parallel "$(nproc)" \
    && cmake --install /tmp/opencv-build \
    && rm -rf /tmp/opencv /tmp/opencv-build /tmp/opencv.tar.gz

ENV LD_LIBRARY_PATH="${OPENCV_PREFIX}/lib:${FFMPEG_PREFIX}/lib"
ENV PKG_CONFIG_PATH="${OPENCV_PREFIX}/lib/pkgconfig:${FFMPEG_PREFIX}/lib/pkgconfig"

######################## BASE IMAGE ##########################
FROM ubuntu:24.04 AS base

ARG PYTHON_VERSION=3.12

# set the environment variables
ENV PATH="/opt/ffmpeg/bin:/root/.local/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="/opt/opencv/lib:/opt/ffmpeg/lib"
ENV PKG_CONFIG_PATH="/opt/opencv/lib/pkgconfig:/opt/ffmpeg/lib/pkgconfig"

# uv environment variables
ENV UV_HTTP_TIMEOUT=500
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_LINK_MODE="copy"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# install dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt update -y \
    && apt install -y curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

COPY --from=ffmpeg-image /opt/ffmpeg /opt/ffmpeg
COPY --from=opencv-image /opt/opencv /opt/opencv

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# install python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}

FROM scratch AS local_src
COPY . /src

######################### BUILD IMAGE #########################
FROM base AS build-image

# set the environment variables
ENV PATH="/root/.cargo/bin:${PATH}"

# install dependencies
RUN apt update -y \
    && apt install -y git build-essential clang libclang-dev libssl-dev patchelf pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# install rustup from rustup.rs
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version && protoc --version

# copy source code
COPY --from=local_src /src /opt/smg

# working directory
WORKDIR /opt/smg

# install maturin and build the wheel with vendored OpenSSL
RUN uv pip install maturin \
    && cargo clean \
    && rm -rf bindings/python/dist/ \
    && cd bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl,opencv-video --out dist \
    && rm -rf /root/.cache

######################### ROUTER IMAGE #########################
FROM base AS router-image
ARG TARGETARCH

# Copy the built package from the build image
COPY --from=build-image /opt/smg/bindings/python/dist/*.whl dist/

# Install the package and verify arm64's allocator configuration. BuildKit
# supplies TARGETARCH separately for each platform in a multi-arch build.
RUN uv pip install --force-reinstall dist/*.whl \
    && if [ "${TARGETARCH}" = "arm64" ]; then \
         jemalloc_stats="$(_RJEM_MALLOC_CONF=stats_print:true python3 -c 'from smg import smg_rs' 2>&1)" \
         && printf '%s\n' "${jemalloc_stats}" | grep -F "Page size: 65536"; \
       else \
         python3 -c 'from smg import smg_rs'; \
       fi

# Clean up unnecessary files to reduce the image size
RUN rm -rf /root/.cache dist/ \
    && apt purge -y --auto-remove curl

# Run as non-root user (SMG only binds to unprivileged ports 30000/29000)
# UID 65532 follows the nonroot convention used by distroless images.
# UID 1000 is already taken by the 'ubuntu' user in ubuntu:24.04.
RUN useradd --system --no-log-init --create-home --uid 65532 smg
USER smg

# Set the entrypoint to the main command
ENTRYPOINT ["python3", "-m", "smg.launch_router"]
