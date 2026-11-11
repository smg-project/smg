# AMD MoRI READ image contract

The manual `vllm-amd-moriio-read` nightly PD row requires
`SMG_AMD_MORIIO_READ_IMAGE` to name a reviewed, qualified image by immutable
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

The expected sources come from unpublished local commit
`f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9` prepared for
[vllm-project/vllm](https://github.com/vllm-project/vllm), based on upstream commit
`595b1514e837ba38df7ae51c411807783e227b78`. The local patch adds the ACK-fenced
hold to the adapted heterogeneous READ changes from
[vllm-project/vllm#62457](https://github.com/vllm-project/vllm/pull/62457).
The local commit has not been published upstream; there is no public commit URL
for those exact source bytes yet. Providing a publicly retrievable source
reference remains a publication gap, separate from image qualification.

In a source checkout containing that local commit, reproduce the two hashes
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
original full topology suite and cleanup evidence. READ remains manual with no
filter and explicit RDMA device declarations. No test, model, assertion, timeout
or shared-node lease/window contract is reduced by this preflight.
