# MoRI READ engine build inputs

This directory preserves the successful A11 full ROCm build recipe for
[public engine f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9](https://github.com/chunfangamd/vllm/commit/f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9).
It is a build recipe and provenance record, not a published container image.

`Dockerfile.rocm` is byte-identical to the successful recipe: the engine's
original Dockerfile plus `ENV MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16` in its
`dev` stage. `upstream-bake.hcl` is the engine's original bake file.
`bake.hcl` binds the three official immutable stack/runtime/test-deps images
and the actual gfx942 build arguments. It preserves the full own-source Csrc,
Rust and wheel build, installation, `test` and `test-smoke` targets. It does
not overlay Python onto an older engine or replace full compilation with smoke.

The source export must contain the exact f9 Git tree and SCM archival metadata.
From an existing source repository containing that commit and its verified tag
ancestry, prepare a fresh export (the destination must not already exist):

```bash
engine_commit=f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9
recipe_dir=/absolute/path/to/smg/docker/moriio-read
source_dir=/absolute/path/to/new-f9-export
mkdir "$source_dir"
git archive "$engine_commit" | tar -x -C "$source_dir"
cp "$recipe_dir/source.git_archival.txt" "$source_dir/.git_archival.txt"
```

The supplied archival metadata records the actual source/tag distance. It is
not a fabricated version override. The build calculates base SCM version
`0.34.0rc1.dev203+gf9e4ca965`; the installed ROCm distribution adds the legitimate
`.rocm100` suffix. See `receipts/scm.json`. Check the source tree against that
receipt before building. A different source checkout must not reuse this SCM
metadata or the validation claim.

For a static plan, run from that export using an existing Docker config and
explicit local endpoint. These commands do not provision a builder:

```bash
cd "$source_dir"
export CONTEXT="$source_dir" MORIIO_RECIPE_DIR="$recipe_dir"
export STACK=rocm STACK_WHEELS= AITER_SPEC= AITER_NIGHTLY= VLLM_VERSION=
export SCCACHE_BUCKET_NAME= PUSH=false CACHE= CACHE_TO= CACHE_CI=false
export TAG=smg-moriio-local:f9e4ca9653da
# Use an existing config; do not embed credentials in this recipe.
docker --config /absolute/path/to/docker-config --host unix:///var/run/docker.sock \
  buildx bake --allow="fs.read=$recipe_dir" \
  -f "$recipe_dir/upstream-bake.hcl" -f "$recipe_dir/bake.hcl" \
  --print test test-smoke
```

An independently admitted build uses the same two targets without `--print`,
with its reviewed builder, resource limits, deadlines and owned cleanup.
The overlay supplies immutable external contexts rather than rebuilding the
cold dependency stack. External build dependencies and tool downloads remain
part of the original recipe. This is not a deterministic bit-for-bit rebuild
claim. No build or publication is authorized merely by these instructions.

`receipts/{stack,runtime,testdeps}.json` are compact projections of actual
material inspections. Each records its original raw receipt SHA256/size,
canonical full file-map digest/count, native shared-object hashes, provider
names, wheel URLs, versions, compiler, Python SOABI, ordered loader candidates,
SDK and RDMA material. Raw receipts are not in this repository. Canonical JSON
uses sorted keys, compact separators and ASCII escaping. These projections
are not complete SBOMs. All four unobserved native Git HEADs remain `null`;
package versions or wheel filenames do not prove those source revisions.

`receipts/validation.json` binds the public source and actual bounded native
result: four original layer-wide DUMMY-weight cases, four passes, no skips,
517.756 seconds, 17 warnings, single-node two-GPU xGMI. The harness was pinned
separately; this build recipe does not claim that its original baked harness
contains subsequent harness fixes. It does not establish full-weight quality,
RDMA, heterogeneous TP, physical MR lifetime or SMG PD/normal-CI qualification.
The local image config and Node2 OCI manifest are historical local identifiers,
not public registry pull references. Publishing a pullable image remains open.
