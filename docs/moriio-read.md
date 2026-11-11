# AMD MoRI READ image contract

The manual `vllm-amd-moriio-read` nightly PD row requires
`SMG_AMD_MORIIO_READ_IMAGE` to name a reviewed, qualified image by immutable
registry digest (`repository[:tag]@sha256:<64 lowercase hex digits>`). The CPU
plan rejects an unset/malformed value and the old749 digest, and captures the
validated reference in the READ matrix. WRITE and other AMD rows keep their
existing image. No variable is enabled or image qualification asserted here.
A local Docker config ID is not a registry-pullable image reference.

Image749 has ACK handling, but ignores `read_ack_fenced_hold` and can release
expired deferred producer sends after60seconds. Passing that flag cannot enable
the reviewed f9 ACK-fenced producer hold in this older engine.

After dependency installation and before model download/worker startup, the
reusable AMD vLLM job checks its effective installed package without importing
GPU modules. Every MoRI READ path, including empty-mode(defaultREAD), lane
backend overrides and extra fleet backends, must match the reviewed f9 common
and connector source hashes in `scripts/ci_moriio_read_image.py`. Source shadow,
missing files or mismatched bytes fail the job. WRITE/non-MoRI paths are unchanged.

The two-source check identifies the reviewed producer-hold implementation; it
is **not** native ABI, physical transfer or model/quality qualification. A future
image requires actual admission, ROCm/driver/native MoRI/RDMA validation, the
original full topology suite and cleanup evidence. READ remains manual with no
filter and explicit RDMA device declarations. No test/model/assertion/timeout
or shared-node lease/window contract is reduced by this preflight.
