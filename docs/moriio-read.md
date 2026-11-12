# AMD MoRI READ image contract

The `vllm-amd-moriio-read` nightly PD row requires
`SMG_AMD_MORIIO_READ_IMAGE` to name a reviewed candidate image by immutable
registry digest (`repository[:tag]@sha256:<64 lowercase hex digits>`). The CPU
plan rejects an unset or malformed value and the v0.31.0 image digest below,
and captures the validated reference in the READ matrix. WRITE and other AMD
rows keep their existing image. No variable is enabled or image qualification
asserted here. A local Docker config ID is not a registry-pullable reference.

The image
`vllm/vllm-openai-rocm:v0.31.0@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f`
has ACK handling, but ignores `read_ack_fenced_hold` and can release expired
deferred producer sends after 60 seconds. Passing that flag cannot enable the
reviewed ACK-fenced producer hold in this older engine.

The expected sources are publicly retrievable at
[chunfangamd/vllm@f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9](https://github.com/chunfangamd/vllm/commit/f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9),
based on upstream `595b1514e837ba38df7ae51c411807783e227b78`.
This adds the ACK-fenced hold to adapted heterogeneous READ changes from
[vllm-project/vllm#62457](https://github.com/vllm-project/vllm/pull/62457).
Public fork source does not mean upstream merge or published image.
The portable successful build recipe and compact observed material/SCM receipts
are in [docker/moriio-read](../docker/moriio-read/README.md).

In a source checkout containing that commit, reproduce the two hashes
from the Git blobs without changing the working tree:

```bash
source_commit=f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9
for source_file in moriio_common.py moriio_connector.py; do
  git show "$source_commit:vllm/distributed/kv_transfer/kv_connector/v1/moriio/$source_file" | sha256sum
done
```

Compare the outputs in that order with `SOURCE_SHA256` in
`scripts/ci_moriio_read_image.py`.

After dependency installation and before model download or worker startup, the
reusable AMD vLLM job checks its effective installed package without importing
GPU modules. Every MoRI READ path, including an empty mode (default READ), lane
backend overrides and extra fleet backends, must match those reviewed common
and connector source hashes. Source shadow, missing files or mismatched bytes
fail the job. WRITE and non-MoRI paths are unchanged.

The two-source check identifies the reviewed producer-hold implementation; it
is **not** native ABI, physical transfer or model/quality qualification. A future
image requires actual admission, ROCm/driver/native MoRI/RDMA validation, the
original full topology suite and cleanup evidence. READ uses the full file with no
filter and explicit RDMA device declarations. No test, model, assertion, timeout
or shared-node lease/window contract is reduced by this preflight.

The locally built engine and separately pinned harness passed the original four
native cases: **4 passed, 0 skipped, 517.756 seconds, 17 warnings**. This used
Qwen3-32B architecture with four layers and wide **DUMMY weights**, two MI300X
GPUs on one node over xGMI, original graph settings/logit tolerance and both
corruption negative controls. The receipt retains the warnings; this is not a
runtime-cleanliness or performance claim. It does not qualify full real weights,
RDMA, heterogeneous TP, physical MR lifetime, SMG router PD or normal CI.

No public registry image reference is provided yet. Local config/OCI manifest
identifiers in the receipts must not be used as registry pull references.
Publishing a pullable image and executing the original SMG READ PD suite under
its existing admission/RDMA contract remain separate delivery steps.

## Normal nightly selection

`SMG_RUN_AMD_MORIIO_READ=true` independently opts the original READ/RDMA row
into normal schedules, only when `SMG_RUN_AMD_LEGS=true` too. An unset flag
selects no additional row. This change sets no variables; activation requires
separate review and readback of effective configuration, including the immutable
image and explicit InfiniBand device options. Invalid requested READ inputs fail
the CPU plan before GPU allocation rather than silently dropping the row.

The Monday–Saturday 09:45 UTC cron and Sunday 22:35 UTC cron each select one
additional READ row. Sunday 09:45 remains non-AMD only. Manual engine inputs and
the default manual NIXL/WRITE lanes retain their existing behavior. Selecting the
Sunday 22:35 row does not promise a 22:35 start: the existing mixed Sunday
calendar and full 390-minute lifecycle admission can defer it. Actual Router
completion, shared lease, resource ownership and cleanup remain required.

The first normal full-suite run supplies qualification results; passing that
same suite beforehand is not an activation prerequisite. The existing full
file, original model, assertions, 170-minute job / 150-minute test limits and
390-minute host lifecycle remain intact. Record every executed and skipped case
and the actual native transport: configured RDMA can select XGMI for local peers.
Source qualification does not lift the retained peer-memory restart skips.
Native peer-generation retirement, same-GPU restart/MR lifetime, real RDMA and
full-weight SMG PD acceptance remain explicit gaps until their original
contracts are demonstrated. A green run containing those skips cannot close
those gaps.
