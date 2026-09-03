# CI image for the TokenSpeed e2e lanes: the plain GPU CI container plus a
# fully-built TokenSpeed (engine + kernel + scheduler) baked into a venv.
#
# Built by .github/workflows/ci-tokenspeed-image.yml on every bump of
# .github/versions/tokenspeed.ref or of the image tooling, and tagged
# ghcr.io/<owner>/smg:<tag> with the content-addressed tag printed by
# scripts/ci_tokenspeed_image_tag.sh. The build needs nvcc but no GPU
# (the kernel arch list is pinned in the install script), so it runs on the
# CPU/docker runner pool.
#
# scripts/ci_install_tokenspeed.sh stamps the built ref at
# /opt/smg-ci/tokenspeed.ref; at job time the same script sees a matching
# stamp and skips the ~25-minute source build, leaving only the per-PR SMG
# gRPC glue install. scripts/ci_setup_python_venv.sh adopts the baked venv
# via the SMG_BAKED_VENV env below.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/smg-ci

COPY scripts/ci_setup_python_venv.sh scripts/ci_install_tokenspeed.sh scripts/
COPY .github/versions/tokenspeed.ref .github/versions/tokenspeed.ref

# Keep the TokenSpeed checkout inside the image: the venv's editable installs
# point at it. uv is promoted to /usr/local/bin because the per-job HOME
# differs from the build-time HOME, so ~/.local/bin would not be found.
RUN bash scripts/ci_setup_python_venv.sh \
    && TOKENSPEED_BUILD_ONLY=1 TOKENSPEED_DIR=/opt/tokenspeed-src \
       bash scripts/ci_install_tokenspeed.sh \
    && if [ -x "$HOME/.local/bin/uv" ] && [ ! -x /usr/local/bin/uv ]; then \
           cp "$HOME/.local/bin/uv" /usr/local/bin/uv; \
       fi \
    && rm -rf /root/.cache/uv /root/.cache/pip /var/lib/apt/lists/*

# Advertise the baked venv; scripts/ci_setup_python_venv.sh symlinks the
# per-job .venv to it when the interpreter matches the CI pin.
ENV SMG_BAKED_VENV=/opt/smg-ci/.venv
