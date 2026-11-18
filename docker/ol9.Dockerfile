# Build from the repository root: docker build -f docker/ol9.Dockerfile -t smg:ol9 .
######################## FFMPEG IMAGE ########################
FROM oraclelinux:9@sha256:06f1c40f41e4026172cc2065103e313e454747ca48d40d165097c5297d764b38 AS ffmpeg-image

# TODO: Add a scheduled updater for these source-built media dependencies.
# Version and SHA256 must be updated together.
ARG FFMPEG_VERSION=8.0
ARG FFMPEG_SHA256=b2751fccb6cc4c77708113cd78b561059b6fa904b24162fa0be2d60273d27b8e
ARG FFMPEG_PREFIX=/opt/ffmpeg

RUN dnf install -y --setopt=install_weak_deps=False \
        --enablerepo=ol9_codeready_builder \
        ca-certificates cmake curl gcc gcc-c++ make nasm ninja-build \
        pkgconf-pkg-config python3 tar xz zlib-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

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
        -D CMAKE_INSTALL_LIBDIR=lib \
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
FROM oraclelinux:9@sha256:06f1c40f41e4026172cc2065103e313e454747ca48d40d165097c5297d764b38 AS base

# Use the distribution's Python and OpenSSL, including their security updates.
ARG PYTHON_VERSION=3.12
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:/opt/ffmpeg/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/opencv/lib:/opt/ffmpeg/lib"

RUN dnf upgrade -y \
    && dnf install -y --setopt=install_weak_deps=False \
        ca-certificates libstdc++ openssl-libs zlib \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-pip \
    && /usr/bin/python${PYTHON_VERSION} -m venv "${VIRTUAL_ENV}" \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && dnf clean all \
    && rm -rf /var/cache/dnf

COPY --from=ffmpeg-image /opt/ffmpeg /opt/ffmpeg
COPY --from=opencv-image /opt/opencv /opt/opencv

######################### BUILD IMAGE #########################
FROM base AS build-image
ARG PYTHON_VERSION=3.12
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PKG_CONFIG_PATH="/opt/opencv/lib/pkgconfig:/opt/ffmpeg/lib/pkgconfig"
ENV OPENSSL_NO_VENDOR=1

# protoc and the development packages are available from public OL9 repos.
RUN dnf install -y --setopt=install_weak_deps=False \
        --enablerepo=ol9_codeready_builder \
        clang clang-devel cmake gcc gcc-c++ git make openssl-devel \
        pkgconf-pkg-config protobuf-compiler protobuf-devel python${PYTHON_VERSION}-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --default-toolchain none --profile minimal \
    && rm /tmp/rustup-init.sh

WORKDIR /opt/smg
COPY rust-toolchain.toml ./
RUN rustup toolchain install \
    && rustc --version && cargo --version && protoc --version

COPY . .
RUN python3 -m pip install --no-cache-dir 'maturin>=1.0,<2.0' \
    && cd bindings/python \
    && ulimit -n 65536 \
    && maturin build --locked --release --features opencv-video --auditwheel skip --out dist

######################### ROUTER IMAGE #########################
FROM base AS router-image
ARG TARGETARCH

# SMG loads these decoders by their SONAME at runtime.
RUN dnf install -y --setopt=install_weak_deps=False \
        --enablerepo=ol9_codeready_builder turbojpeg libwebp \
    && dnf clean all \
    && rm -rf /var/cache/dnf

COPY --from=build-image /opt/smg/bindings/python/dist/*.whl /tmp/dist/
# Install and remove the wheel in one layer; no compiler or Rust toolchain is
# copied into the runtime. The base stage supplies the media and system libraries.
RUN python3 -m pip install --no-cache-dir /tmp/dist/*.whl \
    && python3 -m pip check \
    && python3 -c 'import ctypes; from smg import smg_rs; ctypes.CDLL("libturbojpeg.so.0"); ctypes.CDLL("libwebp.so.7")' \
    && if [ "${TARGETARCH}" = "arm64" ]; then \
         jemalloc_stats="$(_RJEM_MALLOC_CONF=stats_print:true python3 -c 'from smg import smg_rs' 2>&1)" \
         && printf '%s\n' "${jemalloc_stats}" | grep -F "Page size: 65536"; \
       fi \
    && rm -rf /tmp/dist /root/.cache

RUN useradd --system --no-log-init --create-home --uid 65532 smg
USER smg
ENTRYPOINT ["python3", "-m", "smg.launch_router"]
