#!/usr/bin/env bash
# build_base.sh <rev> <out> [<target dir>]: the base revision's gateway binary for an A/B.
#
# A detached worktree of <rev> is built with the SAME cargo line as the head
# (`cargo build --release -p smg --bin smg`: the gateway alone, so the shared dependencies
# get the same feature set on both sides; building it together with other packages changes
# the binary through feature unification) into a target directory of its own (default
# target/gateway-perf/base-target: a full release build, slower than sharing the checkout's
# target directory but never served another tree's artifacts), and the binary is copied to
# <out> with the commit beside it. sccache, when the environment names it, shortens the build.
set -eu
rev=${1:?usage: build_base.sh <rev> <out> [<target dir>]}
out=${2:?usage: build_base.sh <rev> <out> [<target dir>]}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
target=${3:-$root/target/gateway-perf/base-target}
case "$target" in /*) ;; *) target=$root/$target ;; esac
case "$out" in /*) ;; *) out=$root/$out ;; esac
src=$root/target/gateway-perf/base-src
commit=$(git -C "$root" rev-parse --verify "$rev^{commit}")
if [ -e "$src" ]; then git -C "$root" worktree remove --force "$src"; fi
git -C "$root" worktree add --detach "$src" "$commit" >/dev/null
trap 'git -C "$root" worktree remove --force "$src" >/dev/null 2>&1 || true' EXIT
echo "build_base: $commit ($rev) -> $out (target $target)"
( cd "$src" && CARGO_TARGET_DIR=$target cargo build --release -p smg --bin smg )
mkdir -p "$(dirname "$out")"
cp "$target/release/smg" "$out"
echo "$commit" > "$out.commit"
echo "build_base: done ($(stat -c %s "$out") bytes)"
