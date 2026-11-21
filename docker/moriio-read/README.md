# MoRI READ engine build inputs

This directory preserves the successful A11 full ROCm build recipe for
[public engine f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9](https://github.com/chunfangamd/vllm/commit/f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9).
The recipe and provenance records are distinct from the candidate image produced
by the normal-main publisher described below.

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
not public registry pull references. Use a successful publisher artifact to
identify an actual registry candidate; a merge alone does not prove delivery.


## Normal-main candidate publication

[CI MoRI READ Candidate Image](../../.github/workflows/ci-moriio-read-image.yml)
runs its publisher only for a normal `smg-project/smg` push to `main` that changes
an actual build input, the publisher or its workflow. Documentation and receipt
changes alone do not publish. Same-repository PRs retain the broader recipe
integrity check and 13 stdlib publisher controls without Docker or publication.

The publisher builds both full targets, then exports the cached `test` target as
`ghcr.io/smg-project/smg:ci-moriio-read-<identity>-<run>-<attempt>`.
Find the actual candidate reference, immutable digest, completion status and
owned cleanup result in `result.json` inside the Actions artifact
`moriio-read-candidate-<run>-<attempt>`. `publication-metadata.json` records the
Buildx export; `candidate-identity.json` records the registry's raw OCI
index/manifest/config and layer descriptors. The result also records the exact
engine source commit and exported archive hash; the three `*-material.json`
files record the immutable dependency image metadata.

This runs on the existing Docker CPU route only after actual capacity, disk,
cgroup limits and no-GPU build-sandbox checks pass. Insufficient capacity,
registry permission, private-package visibility, anonymous metadata access or
owned-cleanup failure makes the operation fail. A failed operation may already
have pushed a candidate: inspect its result and registry receipt rather than
assuming failure means nothing was published. Main merge is not a guarantee
that a public pullable image exists.

Every output is an **unqualified candidate**. Binding its immutable digest to
`SMG_AMD_MORIIO_READ_IMAGE` requires a separately reviewed first-qualification
operation, preserving the full original SMG READ/RDMA suite, real model weights,
all topologies and assertions, effective installed-source preflight, device/NIC
requirements and shared-node admission/cleanup. That first run can establish
qualification; prior success of the same not-yet-enabled path is not required.
Only its actual complete results can support a qualified SMG PD claim. The
historical four DUMMY/xGMI passes above do not qualify a newly built digest or
replace this full READ/RDMA operation. No variable or deployment changes are
performed by the publisher.
