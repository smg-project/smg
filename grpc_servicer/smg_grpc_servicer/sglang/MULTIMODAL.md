# Rust worker image and video processing

Select `SMG_SGLANG_MM_PROCESSOR=smg` together with the Rust SGLang servicer
(`SMG_SGLANG_SERVICER_IMPL=rust`). The router forwards image or video references
to workers advertising this capability. Workers use SMG's native media pipeline
and send SGLang-compatible tensors, expanded tokens and positional metadata to
the scheduler. Without this selection, existing router preprocessing remains in
use. Encode disaggregation cannot be combined with this worker processor.

References must identify immutable content for the lifetime of a request,
including retries and prefill/decode processing. Mixed image/video requests,
audio and per-item processing hints that cannot be represented by this wire
contract are rejected. SGLang drops image alpha; vLLM's white compositing policy
is unchanged. Video sampling preserves temporal grids, timestamps and offsets.
Video is admitted through the advertised native media-reference path only;
ordinary SGLang workers and router-side SGLang video tensors retain the existing
unsupported-modality refusal.

The native SGLang `smg` media processor supports GLM-5.3-Flash and MiniMax-M3 (including M3.1);
other model processors are rejected at startup.

MiniMax uses the SGLang checkpoint HF image area limits and float video resizing,
including upward frame-grid alignment before normalization. This differs from the
vLLM source-size policy, which remains unchanged. Engine support for the checkpoint's
quantization and Python processor API is required independently of SMG.

## Prefill/decode reuse

Both supported processors can advertise a versioned capability
fingerprint covering model, tokenizer, processor, dtype and effective media
limits. Both workers must advertise the same fingerprint. After preprocessing,
Prefill returns a control response containing expanded tokens, feature hashes,
position/grid metadata and a digest binding it to the original request, ordered
references and KV rendezvous. This response contains no pixels and does not
indicate that inference or KV transfer has completed.

Decode validates the contract, binding and tensor/range structure before using
this metadata. Other processors do not advertise P/D metadata reuse.
The router retains original media references and token anchors: an unsupported
worker, mismatched/invalid identity or five-second identity wait timeout uses
normal worker preprocessing. On timeout, Decode starts its own processing after
that wait: an input taking `T` seconds to preprocess can therefore take roughly
`5 seconds + T` before Decode is ready, instead of overlapping both processors.
This tradeoff can affect slow fetches and long videos; disable reuse with
`SMG_SGLANG_PD_MEDIA_REUSE=0` to retain parallel preprocessing. Legacy workers do not advertise reuse and receive
ordinary requests. Cancellation remains registered during preprocessing; a
cancelled blocking decoder retains its admission slot until it actually exits.
Worker-side fetch failures drain sibling tasks before releasing admission; this
may delay an error response until slow siblings finish. Router-side preprocessing
retains its existing fail-fast behavior.

## Configuration

All defaults below apply to the explicitly selected Rust media worker; explicit
thread settings are preserved.

| Variable | Default and behavior |
| --- | --- |
| `SMG_SGLANG_MM_MAX_INFLIGHT` | `64`; positive preprocessing admission limit |
| `SMG_SGLANG_MM_DTYPE` | Effective engine `ModelConfig.dtype`; override with `float32`, `float16` or `bfloat16` for the vision encoder. Unsupported effective dtypes warn and retain float32. |
| `SMG_SERVICER_WORKER_THREADS` | `8` when unset |
| `SMG_MM_THREADS` | `32` when unset |
| `SMG_MM_IMAGE_CONCURRENCY` | `8` when unset |
| `SMG_SGLANG_PD_MEDIA_REUSE` | Enabled for supported processors with a complete fingerprint; `0` disables reuse |
| `SMG_SGLANG_MM_HASH_THREADS` | `16` for supported processors, otherwise `0`; range `0..=32`, `0` disables off-runtime hashing and reuse |
| `SMG_SGLANG_MM_SHM_BROADCAST` | Enabled for this processor; `0` disables, `1` explicitly enables |

CPU hashing preserves SGLang's SHA256 feature identity. A nonzero/unknown
`SGLANG_MM_BUFFER_SIZE_MB` or enabled/unknown `SGLANG_MM_SKIP_COMPUTE_HASH`
retains engine hashing and disables identity reuse. Shared-memory broadcasts
require single-node TP, CPU features of at least 32 KiB and the engine's SHM
finalization support. Other configurations retain inline transport. Allocation
failure warns and retains inline transport; the engine owns collective
materialization and handle cleanup.

The `smg_servicer_media_identity_total` worker metric distinguishes processed, prepared, reused and
fallback events. These counters show path selection, not model accuracy or a
universal performance gain. Validate the effective dtype against the particular
vision encoder; model-specific correctness and performance require real-engine
testing, including tail latency and both transport paths.
