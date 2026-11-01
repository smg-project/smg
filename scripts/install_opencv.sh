#!/usr/bin/env bash
#
# Install the system libraries needed to build the multimodal `opencv-video`
# feature (crate `llm-multimodal`, optional dep `opencv`).
#
# This feature is OFF by default, so a normal `cargo build` / `cargo test` does
# NOT need anything here. You only need it when building with `--all-features`
# or `--features opencv-video` — which is what `make check`, `make pre-commit`,
# and CI's lint job run. Without these libraries the build fails with:
#
#     Error: "Failed to find installed OpenCV package using pkg-config ..."
#
# CI (Debian) calls this script; it also works on macOS (Homebrew). For a quick
# local lint that skips this feature entirely, run clippy without
# `--all-features`:
#
#     cargo clippy --workspace --all-targets -- -D warnings
#
# Usage:
#   bash scripts/install_opencv.sh          # interactive on a dev machine
#   AUTO_INSTALL=1 bash scripts/install_opencv.sh   # non-interactive (CI)
#   DRY_RUN=1 bash scripts/install_opencv.sh        # print commands, run nothing

set -Eeuo pipefail
IFS=$'\n\t'

has_cmd() { command -v "$1" >/dev/null 2>&1; }

# Run a command, or just print it under DRY_RUN=1.
run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ( IFS=' '; echo "+ $*" )
  else
    "$@"
  fi
}

# CI sets CI=true; treat that as auto-confirm so the script never blocks a runner.
auto_install() { [[ "${AUTO_INSTALL:-0}" == "1" || "${CI:-}" == "true" ]]; }

confirm() {
  if auto_install; then
    return 0
  fi
  read -r -p "$1 [y/N] " response
  [[ "${response:-N}" =~ ^[Yy]$ ]]
}

install_apt() {
  local sudo=""
  has_cmd sudo && sudo="sudo"
  export DEBIAN_FRONTEND=noninteractive
  local retry=()
  if [[ "${CI:-}" == "true" ]]; then
    run bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_update.sh" $sudo
    retry=(bash "$(dirname "${BASH_SOURCE[0]}")/ci_retry.sh" 3 10)
  else
    run $sudo apt-get update
  fi
  run "${retry[@]}" $sudo apt-get install -y --no-install-recommends \
    clang \
    libclang-dev \
    cmake \
    ninja-build \
    pkg-config \
    libopencv-dev
}

install_brew() {
  # OpenCV + pkg-config let the `opencv` crate's build script discover the
  # libraries. libclang (for bindgen) ships with the Xcode Command Line Tools;
  # `brew install llvm` is the fallback if it is missing.
  run brew install opencv pkg-config
  if [[ "${DRY_RUN:-0}" != "1" ]] && ! has_cmd clang && [[ ! -e /Library/Developer/CommandLineTools/usr/lib/libclang.dylib ]]; then
    echo "libclang not found. Install the Xcode Command Line Tools (xcode-select --install)"
    echo "or run: brew install llvm  (then set LIBCLANG_PATH=\"\$(brew --prefix llvm)/lib\")"
  fi
}

main() {
  echo "Installing system libraries for the opencv-video feature..."

  if [[ "$(uname -s)" == "Darwin" ]]; then
    if has_cmd brew; then
      confirm "Install OpenCV via Homebrew?" && install_brew
    else
      echo "Homebrew not found. Install it from https://brew.sh, then run:"
      echo "  brew install opencv pkg-config"
      exit 1
    fi
  elif has_cmd apt-get; then
    confirm "Install OpenCV build dependencies via apt-get?" && install_apt
  else
    echo "Unsupported package manager. Install OpenCV development headers, libclang,"
    echo "pkg-config, and cmake manually for your distribution, for example:"
    echo "  dnf:    sudo dnf install opencv-devel clang-devel cmake pkgconf-pkg-config"
    echo "  pacman: sudo pacman -S opencv clang cmake pkgconf"
    exit 1
  fi

  if [[ "${DRY_RUN:-0}" != "1" ]] && has_cmd pkg-config; then
    echo "Done. Detected OpenCV: $(pkg-config --modversion opencv4 2>/dev/null || echo 'not on pkg-config path yet')"
  fi
}

main "$@"
