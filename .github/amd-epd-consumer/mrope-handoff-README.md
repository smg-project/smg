This is a default-off source candidate for the fixed4420 Qwen3.5 EPD job.
No test, model, GPU run or resource admission has been performed for it.

The existing cached_tokens int32[16] row retains its64-byte shape, original
usage slots0-6 and tuple/pointer ordering. Opt-in writes a version tag in7
and signed delta in8, clearing both on every reuse. Decode validates the
existing bootstrap room first, then restores a separate request scalar.
It never fabricates MM features or sends images to decode.

Normal decode consumes the validated delta. A generated-tail extend must
start after the whole original prompt; prompt re-prefill and delta-only
speculative decode fail clearly. Existing full-MM and text paths retain
their original behavior. This does not establish generic model parity.

The caller deliberately does not set SMG_EPD_MROPE_DELTA_HANDOFF. A future
root-reviewed activation child must explicitly add that value=1 to this
job's container, retaining SMG_EPD_POSITION_DIAGNOSTIC=1. The worker's
existing Qwen model guard remains mandatory. All four admission flags
remain false; no producer/source map/native identity is replaced.

Opt-in selects mrope_manifest.json, reusing the original exact-source
overlay lock/atomic writes. It binds six engine files and two helpers;
the original position_manifest.json and default three-file mode remain
unchanged. Runtime import auditing requires the additional actual origins
for P/D. The offline analyzer must use --manifest mrope_manifest.json and
the new exact manifest SHA; old records cannot gain the new identity.

The prepared four CPU methods use real CPU tensors and exact captured
MetadataBuffers/ForwardBatch method ASTs with inert config/request objects.
They do not import SGLang, create a worker, transfer KV or run a model.
Root must separately admit execution with EPD_MROPE_ACTUAL_SOURCES pointing
at the hash-bound source map, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1.
Original8/four topologies/16 declared requests/oracles/3s/75-65/reruns2
remain unchanged. The first-topology receive stall is a separate open issue.
