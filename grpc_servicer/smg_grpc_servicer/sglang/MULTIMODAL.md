# Rust worker image and video processing

Select `SMG_SGLANG_MM_PROCESSOR=smg` together with the Rust SGLang servicer
(`SMG_SGLANG_SERVICER_IMPL=rust`). The router forwards image or video references
to workers advertising this capability. Workers use SMG's native media pipeline
and send SGLang-compatible tensors, expanded tokens and positional metadata to
the scheduler. Without this selection, existing router preprocessing remains in
use. Encode disaggregation cannot be combined with this worker processor.

References must identify immutable content for the lifetime of a request,
including retries and prefill/decode processing. Audio and per-item processing hints that cannot be represented by this wire
contract are rejected. SGLang drops image alpha; vLLM's white compositing policy
is unchanged. Video sampling preserves temporal grids, timestamps and offsets.
Video is admitted through the advertised native media-reference path only;
ordinary SGLang workers and router-side SGLang video tensors retain the existing
unsupported-modality refusal.

The native SGLang `smg` media processor supports GLM-5.3-Flash and MiniMax-M3 (including M3.1);
other model processors are rejected at startup.

MiniMax uses the SGLang day-0 source image geometry, Pillow image and video resizing,
including nearest-grid frame alignment before normalization. This differs from the
vLLM source-size policy, which remains unchanged. Engine support for the checkpoint's
quantization and Python processor API is required independently of SMG.

## DeepSeek-V4.1 image batches

The router sends a DeepSeek-V4.1 image batch in the layout vLLM splits: one tensor
of every image's patches, with each image's grid (`vit_grid`), patch count
(`patches_per_image`) and token count (`types_per_image`) beside it. SGLang's V4.1
model encodes one image per item and reads that image's grid as `n_vit_h` /
`n_vit_w`, so both SGLang servicers cut the batch into one item per image, the
shape SGLang's own processor builds: the image's patch rows, its placeholder
range and its grid. A batch is a V4.1 batch when it carries `vit_grid`, which no
other family sends (GLM-5.3 Flash sends `patches_per_image` too). A V4.1 batch
is refused with `INVALID_ARGUMENT` before the scheduler when:

- its grids, patch counts, token counts and placeholders disagree (when it
  sends `types_per_image`, each span must be as long as its count);
- it carries no images, no patch counts, no pixels or no `im_token_id` (SGLang
  pads the image positions only when the request names the id);
- its pixel tensor is not rows of patches, each row with at least one value;
- it is labelled as a video batch (SGLang's V4.1 model encodes images only);
- its placeholders are out of prompt order or overlap (SGLang writes the items'
  embeddings over the placeholders in prompt order);
- its patch rows or spans do not fit the served model, when that model's
  geometry is known: each row must hold 3 x `vision_patch_size`^2 values, and
  each span must be as long as SGLang's V4.1 model makes the image's grid with
  `vision_downsample_ratio`.

The geometry is those two values in the model config SGLang loads from the
server arguments, with `--json-model-override-args` applied. The scheduler,
its model and the Python servicer each load it that way. The scheduler declares
the two values at its handshake for the Rust servicer and for a router
connected to it directly. The two geometry checks are skipped when the loaded
config is not a `deepseek_v41` one or does not hold both values as integers
above zero, and on the Rust side also when the scheduler's plugin predates the
two handshake fields, or when the scheduler ranks of a connection do not all
declare the same two values (a request without a pin is placed on a rank after
it is checked). The other checks do not need the geometry. Other families keep
one item per batch.

## Mixed images and videos

The native worker processor accepts mixed images and videos in either order.
It reuses the per-modality preprocessing pipeline and preserves each batch's
patch offsets, grids and tensors. The first batch uses `GenerateRequest.mm_inputs`;
`extra_mm_inputs` carries additional batches, following the vLLM protocol layout.
Upgrade the gateway, servicer and proto package together before using mixed inputs.
Older proto packages remain usable for text and single-modality requests; they do
not expose the additional batches needed for mixed inputs.

Mixed references are processed on both PD sides; the single-modality media identity
optimization is not used. Router-side video capability remains unchanged.

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
require single-node TP with `tp_size > 1` and the engine's SHM finalization
support. The 32 KiB threshold applies per CPU feature tensor: eligible tensors
use shared memory while smaller or non-CPU tensors in the same broadcast retain
inline transport. Other configurations retain inline transport. Allocation
failure warns and retains inline transport; the engine owns collective
materialization and handle cleanup.

The `smg_servicer_media_identity_total` worker metric distinguishes processed, prepared, reused and
fallback events. These counters show path selection, not model accuracy or a
universal performance gain. Validate the effective dtype against the particular
vision encoder; model-specific correctness and performance require real-engine
testing, including tail latency and both transport paths.
