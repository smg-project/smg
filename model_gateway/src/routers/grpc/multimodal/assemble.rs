//! Assembly: convert a [`MultimodalIntermediate`] into backend-specific
//! `MultimodalData` once the target backend is known (after worker selection).
//!
//! For TokenSpeed this also splits the batched preprocessing output into
//! per-item encoder inputs and per-item content hashes used for encode routing.

use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use anyhow::{Context, Result};
use llm_multimodal::{
    EncoderDtype, EncoderFieldLayouts, EncoderInput, EncoderInputView, FieldLayout, Modality,
    ModelSpecificValue, PlaceholderRange, PreprocessedEncoderInputs,
};
use smg_grpc_client::common_proto as common;
use tracing::{info, warn};

use super::{
    capability::ensure_backend_supports_modalities,
    log_mm_timing_enabled,
    serialize::{
        model_specific_to_tensor_bytes, serialize_encoder_input_owned, serialize_model_specific,
        serialize_tokenspeed_encoder_input, serialize_tokenspeed_encoder_input_owned, wire_dtype,
    },
    transport::{
        mm_encoder_input_dtype, mm_vllm_device_normalizes, mm_vllm_encoder_input_dtype,
        resolve_mm_shm_enabled, resolve_mm_shm_min_bytes,
    },
    MediaBatch, MultimodalIntermediate, PrecomputedMultimodalIntermediate, PromptBinding,
};
use crate::{
    routers::grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        context::WorkerSelection,
        proto_wrapper::{
            cleanup_tokenspeed_items_encoder_shm, SglangMultimodalData, TensorBytes,
            TokenSpeedModality, TokenSpeedMultimodalData, TokenSpeedMultimodalItem,
            TokenSpeedTensor, TrtllmMultimodalData, VllmMultimodalData,
        },
        MultimodalData,
    },
    worker::RuntimeType,
};

/// Assemble backend-specific multimodal data from the intermediate.
///
/// Called in request_building after worker selection, when the backend is known.
pub(crate) async fn assemble_multimodal_data(
    intermediate: MultimodalIntermediate,
    client: &BackendClient,
    workers: Option<&WorkerSelection>,
) -> Result<MultimodalData> {
    assemble_multimodal_data_impl(intermediate, client, workers, false).await
}

/// Assemble multimodal data for a prefill request whose item embeddings will
/// arrive out-of-band from encode workers.
pub(crate) async fn assemble_multimodal_data_after_encode(
    intermediate: MultimodalIntermediate,
    client: &BackendClient,
    workers: Option<&WorkerSelection>,
) -> Result<MultimodalData> {
    assemble_multimodal_data_impl(intermediate, client, workers, true).await
}

async fn assemble_multimodal_data_impl(
    intermediate: MultimodalIntermediate,
    client: &BackendClient,
    workers: Option<&WorkerSelection>,
    omit_prefill_pixels: bool,
) -> Result<MultimodalData> {
    validate_intermediate(&intermediate)?;
    // Defense in depth: worker selection already rejected unsupported (backend,
    // modality) combinations. Re-assert here against the same capability matrix
    // so assembly can never silently mishandle a modality the backend lacks.
    ensure_client_supports_intermediate(client, &intermediate)?;
    match client {
        BackendClient::Grpc(GrpcClient::Sglang(_)) => assemble_sglang_data(intermediate),
        BackendClient::Grpc(GrpcClient::Vllm(_)) => Ok(MultimodalData::Vllm(
            assemble_vllm_batches(intermediate, workers)?,
        )),
        BackendClient::Grpc(GrpcClient::Trtllm(_)) => {
            let batch = into_single_batch(intermediate, "TRT-LLM")?;
            Ok(MultimodalData::Trtllm(assemble_trtllm(batch)?))
        }
        BackendClient::Grpc(GrpcClient::TokenSpeed(_)) => {
            let options = intermediate
                .batches()
                .iter()
                .map(|batch| {
                    tokenspeed_assembly_options(
                        batch.media.modality(),
                        workers,
                        omit_prefill_pixels,
                    )
                })
                .collect::<Vec<_>>();
            Ok(MultimodalData::TokenSpeed(
                spawn_tokenspeed_assembly(intermediate, options).await?,
            ))
        }
        BackendClient::Grpc(GrpcClient::Mlx(_)) => {
            anyhow::bail!("MLX does not support multimodal inputs")
        }
        BackendClient::Zmq(client) => match client.runtime() {
            // connect() resolves Unspecified to vLLM before the handshake, so
            // no Unspecified arm here; a runtime without an arm is a hard
            // error below.
            RuntimeType::Vllm => {
                let mut data = assemble_vllm_batches(intermediate, workers)?;
                // The ZMQ translate reads tensor bytes inline; this wire has no
                // /dev/shm or RDMA pull on the engine side.
                data.shm_enabled = false;
                data.rdma_enabled = false;
                for batch in &mut data.extra_batches {
                    batch.shm_enabled = false;
                    batch.rdma_enabled = false;
                }
                Ok(MultimodalData::Vllm(data))
            }
            RuntimeType::TokenSpeed => {
                let options = intermediate
                    .batches()
                    .iter()
                    .map(|batch| TokenSpeedAssemblyOptions {
                        // The ZMQ translate sends tensor bytes inline
                        // (ext-encoded msgpack): assemble without /dev/shm so no
                        // handle ever needs re-inlining at conversion. The wire
                        // does support ShmTensorHandle; wiring that up is a
                        // follow-up optimization.
                        shm_enabled: false,
                        ..tokenspeed_assembly_options(
                            batch.media.modality(),
                            workers,
                            omit_prefill_pixels,
                        )
                    })
                    .collect::<Vec<_>>();
                Ok(MultimodalData::TokenSpeed(
                    spawn_tokenspeed_assembly(intermediate, options).await?,
                ))
            }
            // The same single batch the gRPC arm sends the servicer: the
            // adapter frames it as the scheduler's processor output.
            RuntimeType::Sglang => assemble_sglang_data(intermediate),
            runtime => anyhow::bail!(
                "multimodal inputs are not supported over the {runtime} ZMQ backend yet"
            ),
        },
    }
}

/// The SGLang payload, one batch, as both the gRPC and the ZMQ arm send it.
fn assemble_sglang_data(intermediate: MultimodalIntermediate) -> Result<MultimodalData> {
    let batch = into_single_batch(intermediate, "SGLang")?;
    Ok(MultimodalData::Sglang(assemble_sglang(batch)?))
}

/// Run TokenSpeed batch assembly on a blocking thread, guarding SHM output.
///
/// Both the gRPC and ZMQ TokenSpeed arms share this sequence; the
/// [`PendingTokenSpeedAssembly`] guard is what unlinks `/dev/shm` segments if
/// the request future is cancelled mid-assembly, so the dance must stay
/// identical for every caller.
async fn spawn_tokenspeed_assembly(
    intermediate: MultimodalIntermediate,
    options: Vec<TokenSpeedAssemblyOptions>,
) -> Result<TokenSpeedMultimodalData> {
    let batches = intermediate.into_batches();
    let pending = tokio::task::spawn_blocking(move || {
        assemble_tokenspeed_batches_owned(batches, options).map(PendingTokenSpeedAssembly::new)
    })
    .await
    .context("TokenSpeed multimodal assembly task failed")??;
    pending.into_inner()
}

/// Owns SHM-backed assembly output until the awaiting task accepts it.
///
/// Dropping a `spawn_blocking` join handle does not cancel its task. If the
/// request future is cancelled, Tokio drops the completed task output instead;
/// this guard unlinks any SHM files before that output is discarded.
struct PendingTokenSpeedAssembly {
    data: Option<TokenSpeedMultimodalData>,
}

impl PendingTokenSpeedAssembly {
    fn new(data: TokenSpeedMultimodalData) -> Self {
        Self { data: Some(data) }
    }

    fn into_inner(mut self) -> Result<TokenSpeedMultimodalData> {
        self.data
            .take()
            .context("pending TokenSpeed assembly is missing data")
    }
}

impl Drop for PendingTokenSpeedAssembly {
    fn drop(&mut self) {
        if let Some(data) = &self.data {
            cleanup_tokenspeed_items_encoder_shm(&data.items, None);
        }
    }
}

/// vLLM takes every modality batch of the request: the first travels as
/// `mm_inputs`, the rest as `extra_mm_inputs`, each assembled the same way.
fn assemble_vllm_batches(
    intermediate: MultimodalIntermediate,
    workers: Option<&WorkerSelection>,
) -> Result<VllmMultimodalData> {
    assemble_vllm_batches_with(intermediate, &VllmAssembly::for_selection(workers))
}

/// The wire choices of a vLLM assembly: the Router resolves them from the
/// worker selection and its transport settings; a worker processing its own
/// media names them directly (inline, in the engine's dtype).
#[derive(Debug, Clone)]
pub(super) struct VllmAssembly {
    /// The float dtype the primary tensor is written in (`float32`,
    /// `bfloat16`, `float16`).
    pub encoder_dtype: String,
    /// The engine rescales and normalizes pixels on device: it takes the
    /// pixels' own `uint8` bytes and would normalize floats twice.
    pub device_normalizes: bool,
    pub shm_enabled: bool,
    pub shm_min_bytes: usize,
}

impl VllmAssembly {
    fn for_selection(workers: Option<&WorkerSelection>) -> Self {
        Self {
            encoder_dtype: mm_vllm_encoder_input_dtype(workers),
            device_normalizes: mm_vllm_device_normalizes(workers),
            shm_enabled: resolve_mm_shm_enabled(workers, false),
            shm_min_bytes: resolve_mm_shm_min_bytes(workers),
        }
    }

    /// The dtype this assembly writes an image encoder input in, known
    /// before the images are preprocessed; `None` when the engine takes the
    /// pixels' own bytes.
    pub(super) fn image_wire_dtype(&self) -> Option<EncoderDtype> {
        if self.device_normalizes {
            return None;
        }
        let dtype = wire_dtype(&self.encoder_dtype, true);
        (dtype != EncoderDtype::Uint8).then_some(dtype)
    }
}

/// [`assemble_vllm_batches`] with the wire choices given.
pub(super) fn assemble_vllm_batches_with(
    intermediate: MultimodalIntermediate,
    assembly: &VllmAssembly,
) -> Result<VllmMultimodalData> {
    let mut batches = intermediate.into_batches().into_iter();
    let first = batches
        .next()
        .context("multimodal intermediate is missing its first batch")?;
    let mut data = assemble_vllm(first, assembly)?;
    data.extra_batches = batches
        .map(|batch| assemble_vllm(batch, assembly))
        .collect::<Result<Vec<_>>>()?;
    Ok(data)
}

/// SGLang and TRT-LLM take a single preprocessed batch (one modality). The
/// per-modality capability is enforced by
/// [`ensure_client_supports_intermediate`]; this only enforces the structural
/// single-batch constraint.
fn into_single_batch(
    intermediate: MultimodalIntermediate,
    backend: &str,
) -> Result<PrecomputedMultimodalIntermediate> {
    anyhow::ensure!(
        intermediate.batches().len() == 1,
        "{backend} multimodal path requires exactly one batch; got {} batches",
        intermediate.batches().len()
    );
    intermediate
        .into_batches()
        .pop()
        .context("multimodal intermediate is missing its sole batch")
}

/// Defense-in-depth capability assertion mirroring the early worker-selection
/// check, keyed on the concrete backend client. See
/// [`crate::routers::grpc::multimodal::capability::ensure_backend_supports_modalities`].
fn ensure_client_supports_intermediate(
    client: &BackendClient,
    intermediate: &MultimodalIntermediate,
) -> Result<()> {
    ensure_backend_supports_modalities(client.runtime_type(), intermediate)
}

fn assemble_sglang(
    intermediate: PrecomputedMultimodalIntermediate,
) -> Result<SglangMultimodalData> {
    // Pinned to float32: nothing on this wire reports which widths the worker
    // accepts, and the three names it does document are all four bytes or wider.
    let (pixel_values, pixel_values_shape, pixel_values_dtype) =
        serialize_encoder_input_owned(intermediate.preprocessed.encoder_input, "float32")?;
    let model_specific_tensors = serialize_model_specific(intermediate.preprocessed.model_specific);
    let MediaBatch::Images(images) = &intermediate.media else {
        anyhow::bail!("SGLang assembly requires an image batch");
    };
    // Sent alongside the preprocessed tensors: a worker that does its own
    // preprocessing has nothing else to work from.
    let image_data = images.iter().map(|f| f.raw_bytes.to_vec()).collect();
    let mm_placeholders = placeholders_for_bindings(&intermediate.bindings, true)?;

    Ok(SglangMultimodalData {
        image_data,
        pixel_values,
        pixel_values_shape,
        pixel_values_dtype,
        model_specific_tensors,
        im_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
    })
}

/// The dtype the primary tensor is written in: the engine's own float dtype,
/// or the pixels' own bytes for an engine that normalizes on device. A
/// preprocessor that produces normalized floats has no bytes to give such an
/// engine, which is a deployment mismatch, said once.
fn vllm_pixel_dtype<'a>(
    preprocessed: &PreprocessedEncoderInputs,
    assembly: &'a VllmAssembly,
) -> &'a str {
    if !assembly.device_normalizes {
        return &assembly.encoder_dtype;
    }
    if preprocessed.encoder_input.pixel_norm().is_some() {
        return "uint8";
    }
    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    WARNED.get_or_init(|| {
        warn!(
            "the engine normalizes pixels on device but this model's preprocessor produces \
             normalized floats, which the engine will normalize again; start the engine with \
             --mm-device-do-normalize=false"
        );
    });
    &assembly.encoder_dtype
}

fn assemble_vllm(
    intermediate: PrecomputedMultimodalIntermediate,
    assembly: &VllmAssembly,
) -> Result<VllmMultimodalData> {
    let dtype = vllm_pixel_dtype(&intermediate.preprocessed, assembly);
    let (pixel_values, pixel_values_shape, pixel_values_dtype) =
        serialize_encoder_input_owned(intermediate.preprocessed.encoder_input, dtype)?;
    let model_specific_tensors = serialize_model_specific(intermediate.preprocessed.model_specific);
    let (modality, mm_hashes) = match &intermediate.media {
        MediaBatch::Images(images) => (
            common::Modality::Image,
            images.iter().map(|frame| frame.hash.clone()).collect(),
        ),
        MediaBatch::Videos(videos) => (
            common::Modality::Video,
            videos.iter().map(|video| video.hash.clone()).collect(),
        ),
        MediaBatch::Audios(_) => {
            anyhow::bail!("vLLM assembly requires an image or video batch")
        }
    };
    let mm_placeholders = placeholders_for_bindings(&intermediate.bindings, false)?;
    // The primary tensor travels in the proto's `pixel_values` field but is
    // named by the model's forward kwarg in the layout keys and on the wire.
    let primary_key = intermediate
        .encoder_input_key
        .clone()
        .unwrap_or_else(|| "pixel_values".to_string());
    let (batched_keys, flat_keys) =
        vllm_field_layout_keys(&intermediate.field_layouts, &primary_key);

    Ok(VllmMultimodalData {
        pixel_values,
        pixel_values_shape,
        pixel_values_dtype,
        model_specific_tensors,
        im_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
        mm_hashes,
        batched_keys,
        flat_keys,
        keep_on_cpu_keys: intermediate.keep_on_cpu_keys,
        encoder_input_key: intermediate.encoder_input_key,
        modality,
        shm_enabled: assembly.shm_enabled,
        shm_min_bytes: assembly.shm_min_bytes,
        // vLLM workers cannot pull RDMA payloads yet.
        rdma_enabled: false,
        extra_batches: Vec::new(),
    })
}

/// Translate the neutral layout contract to vLLM's HF field names, naming the
/// primary tensor `primary_key` (`pixel_values` unless the spec renames it).
fn vllm_field_layout_keys(
    layouts: &EncoderFieldLayouts,
    primary_key: &str,
) -> (Vec<String>, HashMap<String, String>) {
    let mut batched_keys = PreprocessedEncoderInputs::batched_keys(&layouts.model_specific);
    let mut flat_keys = PreprocessedEncoderInputs::flat_keys(&layouts.model_specific);
    match &layouts.encoder_input {
        FieldLayout::Batched => batched_keys.push(primary_key.to_string()),
        FieldLayout::Flat { sizes_key } => {
            flat_keys.insert(primary_key.to_string(), sizes_key.clone());
        }
    }
    (batched_keys, flat_keys)
}

fn assemble_trtllm(
    intermediate: PrecomputedMultimodalIntermediate,
) -> Result<TrtllmMultimodalData> {
    let MediaBatch::Images(images) = &intermediate.media else {
        anyhow::bail!("TRT-LLM assembly requires an image batch");
    };
    let image_data = images.iter().map(|f| f.raw_bytes.to_vec()).collect();
    Ok(TrtllmMultimodalData { image_data })
}

#[cfg(test)]
fn assemble_tokenspeed(
    intermediate: &PrecomputedMultimodalIntermediate,
    workers: Option<&WorkerSelection>,
    skip_pixel_values: bool,
) -> Result<TokenSpeedMultimodalData> {
    validate_precomputed_batch(intermediate)?;
    let options =
        tokenspeed_assembly_options(intermediate.media.modality(), workers, skip_pixel_values);
    assemble_tokenspeed_with_options(intermediate, options, None)
}

pub(crate) fn assemble_tokenspeed_for_encode(
    intermediate: &MultimodalIntermediate,
    workers: Option<&WorkerSelection>,
) -> Result<TokenSpeedMultimodalData> {
    validate_intermediate(intermediate)?;
    let options = intermediate
        .batches()
        .iter()
        .map(|batch| tokenspeed_assembly_options(batch.media.modality(), workers, false))
        .collect();
    assemble_tokenspeed_batches(intermediate.batches(), options)
}

struct TokenSpeedAssemblyOptions {
    shm_enabled: bool,
    shm_min_bytes: usize,
    encoder_input_dtype: String,
    skip_pixel_values: bool,
}

fn tokenspeed_assembly_options(
    modality: Modality,
    workers: Option<&WorkerSelection>,
    skip_pixel_values: bool,
) -> TokenSpeedAssemblyOptions {
    TokenSpeedAssemblyOptions {
        shm_enabled: resolve_mm_shm_enabled(workers, skip_pixel_values),
        shm_min_bytes: resolve_mm_shm_min_bytes(workers),
        encoder_input_dtype: mm_encoder_input_dtype(modality, workers),
        skip_pixel_values,
    }
}

/// `owned_input` is the batch's encoder input when the caller handed it over
/// whole (see [`take_whole_encoder_input`]); its one item then moves those
/// bytes into its payload instead of writing a copy of its view.
fn assemble_tokenspeed_with_options(
    intermediate: &PrecomputedMultimodalIntermediate,
    options: TokenSpeedAssemblyOptions,
    mut owned_input: Option<EncoderInput>,
) -> Result<TokenSpeedMultimodalData> {
    let log_timing = log_mm_timing_enabled();
    let total_started = Instant::now();
    let TokenSpeedAssemblyOptions {
        shm_enabled,
        shm_min_bytes,
        encoder_input_dtype,
        skip_pixel_values,
    } = options;
    let modality = match intermediate.media.modality() {
        Modality::Image => TokenSpeedModality::Image,
        Modality::Video => TokenSpeedModality::Video,
        Modality::Audio => TokenSpeedModality::Audio,
        Modality::ImageEmbeds => TokenSpeedModality::Image,
    };

    let item_count = intermediate.media.len();
    // Build items imperatively so that if any step fails partway we can unlink
    // the /dev/shm segments already created for prior items' encoder inputs
    // (and this item's, once created). `?`/`collect` would drop those
    // `TokenSpeedTensor::Shm` handles without ever reaching the send-path
    // cleanup, leaking files until the next sweep.
    let mut ordered_bindings = intermediate.bindings.iter().collect::<Vec<_>>();
    ordered_bindings.sort_by_key(|binding| binding.prompt_ordinal);
    // Checked before the item loop, while a plain `?` still costs nothing: no
    // /dev/shm segment has been created yet.
    for binding in &ordered_bindings {
        ensure_structural_fallback_covers_features(intermediate, binding)?;
    }
    let mut items: Vec<TokenSpeedMultimodalItem> = Vec::with_capacity(item_count);
    for binding in ordered_bindings {
        let item_index = binding.item_index;
        let mm_placeholders = match placeholders_for_binding(binding, true) {
            Ok(value) => value,
            Err(error) => {
                cleanup_tokenspeed_items_encoder_shm(&items, None);
                return Err(error);
            }
        };
        let content_hash = match content_hash_for_item(intermediate, item_index) {
            Ok(value) => value,
            Err(error) => {
                cleanup_tokenspeed_items_encoder_shm(&items, None);
                return Err(error);
            }
        };
        let encoder_input_started = Instant::now();
        // EPD prefill: the embedding arrives over Mooncake and this item's
        // encoder_input is stripped downstream (clear_mm_pixel_values), so skip
        // the per-item slice + serialize entirely when skip_pixel_values is set.
        let encoder_input = if skip_pixel_values {
            TokenSpeedTensor::inline(Vec::new(), Vec::new(), encoder_input_dtype.clone())
        } else if let Some(input) = owned_input.take() {
            match serialize_tokenspeed_encoder_input_owned(
                input,
                &encoder_input_dtype,
                shm_enabled,
                shm_min_bytes,
            ) {
                Ok(value) => value,
                Err(error) => {
                    cleanup_tokenspeed_items_encoder_shm(&items, None);
                    return Err(error);
                }
            }
        } else {
            let item_encoder_input = match encoder_input_for_item(
                &intermediate.preprocessed,
                &intermediate.field_layouts.encoder_input,
                item_index,
            ) {
                Ok(value) => value,
                Err(error) => {
                    cleanup_tokenspeed_items_encoder_shm(&items, None);
                    return Err(error);
                }
            };
            match serialize_tokenspeed_encoder_input(
                &item_encoder_input,
                &encoder_input_dtype,
                shm_enabled,
                shm_min_bytes,
            ) {
                Ok(value) => value,
                Err(error) => {
                    cleanup_tokenspeed_items_encoder_shm(&items, None);
                    return Err(error);
                }
            }
        };
        let encoder_input_serialize_ms = encoder_input_started.elapsed().as_secs_f64() * 1000.0;
        let model_specific_started = Instant::now();
        let model_specific_tensors = match serialize_model_specific_for_item(
            &intermediate.preprocessed.model_specific,
            &intermediate.field_layouts.model_specific,
            item_index,
        ) {
            Ok(value) => value,
            Err(error) => {
                // `encoder_input` (possibly SHM) was created for this item but the
                // item isn't built; clean it plus all prior items.
                cleanup_tokenspeed_items_encoder_shm(&items, Some(&encoder_input));
                return Err(error);
            }
        };
        let model_specific_serialize_ms = model_specific_started.elapsed().as_secs_f64() * 1000.0;
        if log_timing {
            info!(
                modality = ?modality,
                item_index,
                encoder_input_dtype = %encoder_input.dtype,
                encoder_input_bytes = encoder_input.nbytes(),
                encoder_input_shape = ?encoder_input.shape,
                model_specific_tensor_count = model_specific_tensors.len(),
                encoder_input_serialize_ms,
                model_specific_serialize_ms,
                "smg_mm_timing assemble_tokenspeed_item"
            );
        }

        items.push(TokenSpeedMultimodalItem {
            modality,
            encoder_input,
            model_specific_tensors,
            placeholder_token_id: intermediate.placeholder_token_id,
            mm_placeholders,
            content_hash,
        });
    }

    if log_timing {
        info!(
            modality = ?modality,
            item_count = items.len(),
            total_ms = total_started.elapsed().as_secs_f64() * 1000.0,
            "smg_mm_timing assemble_tokenspeed"
        );
    }

    Ok(TokenSpeedMultimodalData {
        items,
        shm_enabled,
        shm_min_bytes,
    })
}

/// Batches the caller keeps: every item's encoder input is written from its
/// view.
fn assemble_tokenspeed_batches(
    batches: &[PrecomputedMultimodalIntermediate],
    options: Vec<TokenSpeedAssemblyOptions>,
) -> Result<TokenSpeedMultimodalData> {
    let none = batches.iter().map(|_| None).collect();
    assemble_tokenspeed_batches_with(batches, options, none)
}

/// Batches the caller hands over: a batch whose one item is the whole encoder
/// input moves that input into the item's payload instead of writing a copy.
fn assemble_tokenspeed_batches_owned(
    mut batches: Vec<PrecomputedMultimodalIntermediate>,
    options: Vec<TokenSpeedAssemblyOptions>,
) -> Result<TokenSpeedMultimodalData> {
    let owned = batches.iter_mut().map(take_whole_encoder_input).collect();
    assemble_tokenspeed_batches_with(&batches, options, owned)
}

/// The encoder input of a batch whose single item spans all of it, taken out
/// of the batch; an empty tensor stays in its place, which no per-item step
/// reads. `None` for every other batch, whose items are written from views.
fn take_whole_encoder_input(batch: &mut PrecomputedMultimodalIntermediate) -> Option<EncoderInput> {
    let [binding] = batch.bindings.as_slice() else {
        return None;
    };
    let item_index = binding.item_index;
    if batch.media.len() != 1 {
        return None;
    }
    let whole = batch.preprocessed.encoder_input.view();
    let item = encoder_input_for_item(
        &batch.preprocessed,
        &batch.field_layouts.encoder_input,
        item_index,
    )
    .ok()?;
    if whole.is_empty() || item.len() != whole.len() {
        return None;
    }
    Some(std::mem::replace(
        &mut batch.preprocessed.encoder_input,
        EncoderInput::F32(ndarray::ArrayD::zeros(ndarray::IxDyn(&[0]))),
    ))
}

fn assemble_tokenspeed_batches_with(
    batches: &[PrecomputedMultimodalIntermediate],
    options: Vec<TokenSpeedAssemblyOptions>,
    owned_inputs: Vec<Option<EncoderInput>>,
) -> Result<TokenSpeedMultimodalData> {
    anyhow::ensure!(
        batches.len() == options.len() && batches.len() == owned_inputs.len(),
        "multimodal batch/assembly option count mismatch"
    );

    let shm_enabled = options
        .first()
        .map(|opts| opts.shm_enabled)
        .unwrap_or(false);
    let shm_min_bytes = options.first().map(|opts| opts.shm_min_bytes).unwrap_or(0);
    let mut ordered_items: Vec<(usize, TokenSpeedMultimodalItem)> = Vec::new();

    for ((batch, options), owned_input) in batches.iter().zip(options).zip(owned_inputs) {
        match assemble_tokenspeed_with_options(batch, options, owned_input) {
            Ok(data) => {
                let mut ordinals = batch
                    .bindings
                    .iter()
                    .map(|binding| binding.prompt_ordinal)
                    .collect::<Vec<_>>();
                ordinals.sort_unstable();
                if ordinals.len() != data.items.len() {
                    cleanup_tokenspeed_items_encoder_shm(&data.items, None);
                    cleanup_ordered_tokenspeed_items(&ordered_items);
                    return Err(anyhow::anyhow!(
                        "TokenSpeed binding/item count mismatch for {}",
                        batch.media.modality()
                    ));
                }
                ordered_items.extend(ordinals.into_iter().zip(data.items));
            }
            Err(error) => {
                cleanup_ordered_tokenspeed_items(&ordered_items);
                return Err(error);
            }
        }
    }
    let items = into_prompt_order(ordered_items);

    Ok(TokenSpeedMultimodalData {
        items,
        shm_enabled,
        shm_min_bytes,
    })
}

fn cleanup_ordered_tokenspeed_items(items: &[(usize, TokenSpeedMultimodalItem)]) {
    for (_, item) in items {
        cleanup_tokenspeed_items_encoder_shm(std::slice::from_ref(item), None);
    }
}

fn into_prompt_order<T>(mut entries: Vec<(usize, T)>) -> Vec<T> {
    entries.sort_unstable_by_key(|(prompt_ordinal, _)| *prompt_ordinal);
    entries.into_iter().map(|(_, value)| value).collect()
}

fn validate_intermediate(intermediate: &MultimodalIntermediate) -> Result<()> {
    anyhow::ensure!(
        !intermediate.batches().is_empty(),
        "multimodal intermediate requires at least one batch"
    );
    let total_bindings = intermediate
        .batches()
        .iter()
        .map(|batch| batch.bindings.len())
        .sum::<usize>();
    let mut prompt_ordinals = HashSet::with_capacity(total_bindings);
    for batch in intermediate.batches() {
        validate_precomputed_batch(batch)?;
        for binding in &batch.bindings {
            anyhow::ensure!(
                binding.prompt_ordinal < total_bindings,
                "multimodal prompt ordinal {} is outside 0..{total_bindings}",
                binding.prompt_ordinal
            );
            anyhow::ensure!(
                prompt_ordinals.insert(binding.prompt_ordinal),
                "duplicate multimodal prompt ordinal {}",
                binding.prompt_ordinal
            );
        }
    }
    Ok(())
}

fn validate_precomputed_batch(intermediate: &PrecomputedMultimodalIntermediate) -> Result<()> {
    let modality = intermediate.media.modality();
    let media_count = intermediate.media.len();
    let token_count = intermediate.preprocessed.feature_token_counts.len();
    let binding_count = intermediate.bindings.len();
    anyhow::ensure!(
        media_count > 0,
        "precomputed {modality} batch requires at least one media item"
    );
    anyhow::ensure!(
        token_count == media_count,
        "precomputed multimodal token count mismatch: modality={modality}, token_count={token_count}, media_count={media_count}"
    );
    anyhow::ensure!(
        binding_count == media_count,
        "precomputed multimodal binding count mismatch: modality={modality}, binding_count={binding_count}, media_count={media_count}"
    );

    let encoder_rows = intermediate
        .preprocessed
        .encoder_input
        .shape()
        .first()
        .copied()
        .unwrap_or_default();
    match &intermediate.field_layouts.encoder_input {
        FieldLayout::Batched => anyhow::ensure!(
            encoder_rows == media_count,
            "precomputed {modality} batch carries {encoder_rows} encoder rows for {media_count} media items"
        ),
        FieldLayout::Flat { sizes_key } => {
            let sizes = tensor_sizes_from_model_specific(
                &intermediate.preprocessed.model_specific,
                sizes_key,
            )?;
            anyhow::ensure!(
                sizes.len() == media_count,
                "precomputed {modality} batch declares {} item sizes for {media_count} media items",
                sizes.len()
            );
            let covered = sizes
                .iter()
                .try_fold(0usize, |acc, &size| acc.checked_add(size))
                .context("flat encoder size total overflow")?;
            anyhow::ensure!(
                covered == encoder_rows,
                "precomputed {modality} item sizes cover {covered} of {encoder_rows} encoder rows"
            );
        }
    }

    let mut item_indices = HashSet::with_capacity(binding_count);
    for binding in &intermediate.bindings {
        anyhow::ensure!(
            binding.item_index < media_count,
            "precomputed {modality} binding item index {} is outside 0..{media_count}",
            binding.item_index
        );
        anyhow::ensure!(
            item_indices.insert(binding.item_index),
            "duplicate precomputed {modality} binding for item {}",
            binding.item_index
        );
        anyhow::ensure!(
            binding.structural.length > 0,
            "precomputed {modality} binding for item {} has an empty structural range",
            binding.item_index
        );
        let structural_end = binding
            .structural
            .offset
            .checked_add(binding.structural.length)
            .context("structural prompt range overflow")?;
        let mut reserved = 0usize;
        for patch in &binding.patches {
            let patch_end = patch
                .offset
                .checked_add(patch.length)
                .context("patch prompt range overflow")?;
            anyhow::ensure!(
                patch.length > 0
                    && patch.offset >= binding.structural.offset
                    && patch_end <= structural_end,
                "precomputed {modality} patch range ({}, {}) lies outside structural range ({}, {})",
                patch.offset,
                patch.length,
                binding.structural.offset,
                binding.structural.length
            );
            reserved = reserved
                .checked_add(patch.length)
                .context("patch prompt length overflow")?;
        }
        if !binding.patches.is_empty() {
            let features = *intermediate
                .preprocessed
                .feature_token_counts
                .get(binding.item_index)
                .with_context(|| {
                    format!(
                        "missing {modality} feature count for item {}",
                        binding.item_index
                    )
                })?;
            anyhow::ensure!(
                features_fit_reservation(reserved, features),
                "precomputed {modality} item {} reserves {reserved} prompt positions for {features} encoder features",
                binding.item_index
            );
        }
    }
    Ok(())
}

pub(crate) fn encode_routing_hashes(intermediate: &MultimodalIntermediate) -> Result<Vec<Vec<u8>>> {
    validate_intermediate(intermediate)?;
    let mut hashes = Vec::new();
    for batch in intermediate.batches() {
        for binding in &batch.bindings {
            hashes.push((
                binding.prompt_ordinal,
                content_hash_for_item(batch, binding.item_index)?,
            ));
        }
    }
    Ok(into_prompt_order(hashes))
}

fn encoder_input_for_item<'a>(
    preprocessed: &'a PreprocessedEncoderInputs,
    layout: &FieldLayout,
    item_index: usize,
) -> Result<EncoderInputView<'a>> {
    let input = preprocessed.encoder_input.view();
    match layout {
        FieldLayout::Batched => input.slice_axis0(item_index, 1),
        FieldLayout::Flat { sizes_key } => {
            let sizes = tensor_sizes_from_model_specific(&preprocessed.model_specific, sizes_key)?;
            let (start, len) = item_span(&sizes, item_index)?;
            input.slice_axis0(start, len)
        }
    }
}

fn serialize_model_specific_for_item(
    model_specific: &HashMap<String, ModelSpecificValue>,
    field_layouts: &HashMap<String, FieldLayout>,
    item_index: usize,
) -> Result<HashMap<String, TensorBytes>> {
    let mut serialized = HashMap::with_capacity(model_specific.len());
    for (key, value) in model_specific {
        let item_value = match field_layouts.get(key) {
            Some(FieldLayout::Batched) => value
                .slice_first_dim(item_index, 1)
                .with_context(|| format!("failed to slice model_specific tensor {key}"))?,
            Some(FieldLayout::Flat { sizes_key }) => {
                let sizes = tensor_sizes_from_model_specific(model_specific, sizes_key)?;
                let (start, len) = item_span(&sizes, item_index)?;
                value
                    .slice_first_dim(start, len)
                    .with_context(|| format!("failed to slice flat model_specific tensor {key}"))?
            }
            None => value.clone(),
        };
        if let Some(tensor) = model_specific_to_tensor_bytes(&item_value) {
            serialized.insert(key.clone(), tensor);
        } else {
            warn!(tensor_key = %key, "Dropping unsupported model_specific value during multimodal serialization");
        }
    }
    Ok(serialized)
}

fn placeholders_for_bindings(
    bindings: &[PromptBinding],
    prefer_patches: bool,
) -> Result<Vec<(u32, u32)>> {
    let mut ordered = bindings.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|binding| binding.prompt_ordinal);
    ordered
        .into_iter()
        .flat_map(|binding| {
            let ranges = if prefer_patches && !binding.patches.is_empty() {
                binding.patches.as_slice()
            } else {
                std::slice::from_ref(&binding.structural)
            };
            ranges.iter()
        })
        .map(placeholder_range_to_u32)
        .collect()
}

/// Check the ranges an item falls back to when it declares no patches.
///
/// An item without patches is sent as its whole structural range, so that
/// range has to hold exactly as many prompt positions as the item has encoder
/// features. Items that do declare patches are already covered by
/// [`validate_precomputed_batch`], and the vLLM path is exempt: it sends the
/// structural range on purpose and lets the backend pick the feature positions
/// out of it.
fn ensure_structural_fallback_covers_features(
    intermediate: &PrecomputedMultimodalIntermediate,
    binding: &PromptBinding,
) -> Result<()> {
    if !binding.patches.is_empty() {
        return Ok(());
    }
    let modality = intermediate.media.modality();
    let features = *intermediate
        .preprocessed
        .feature_token_counts
        .get(binding.item_index)
        .with_context(|| {
            format!(
                "missing {modality} feature count for item {}",
                binding.item_index
            )
        })?;
    anyhow::ensure!(
        features_fit_reservation(binding.structural.length, features),
        "precomputed {modality} item {} covers {} prompt positions for {features} encoder features",
        binding.item_index,
        binding.structural.length
    );
    Ok(())
}

/// Whether `features` encoder outputs can land in `reserved` prompt positions.
///
/// Usually one feature takes one position. Some models pack a fixed number of
/// encoder outputs into each embedding before placing it, so they reserve fewer
/// positions than they carry features, by a whole factor. Anything that is not
/// a whole factor is a genuine mismatch, which the engine answers by ending the
/// process rather than the request.
fn features_fit_reservation(reserved: usize, features: usize) -> bool {
    reserved > 0 && features >= reserved && features.is_multiple_of(reserved)
}

fn placeholders_for_binding(
    binding: &PromptBinding,
    prefer_patches: bool,
) -> Result<Vec<(u32, u32)>> {
    let ranges = if prefer_patches && !binding.patches.is_empty() {
        binding.patches.as_slice()
    } else {
        std::slice::from_ref(&binding.structural)
    };
    ranges.iter().map(placeholder_range_to_u32).collect()
}

fn placeholder_range_to_u32(range: &PlaceholderRange) -> Result<(u32, u32)> {
    Ok((
        u32::try_from(range.offset).context("multimodal placeholder offset exceeds u32")?,
        u32::try_from(range.length).context("multimodal placeholder length exceeds u32")?,
    ))
}

fn content_hash_for_item(
    intermediate: &PrecomputedMultimodalIntermediate,
    item_index: usize,
) -> Result<Vec<u8>> {
    let hash = match &intermediate.media {
        MediaBatch::Images(items) => items.get(item_index).map(|item| item.hash.as_str()),
        MediaBatch::Videos(items) => items.get(item_index).map(|item| item.hash.as_str()),
        MediaBatch::Audios(items) => items.get(item_index).map(|item| item.hash.as_str()),
    }
    .ok_or_else(|| {
        anyhow::anyhow!(
            "missing {} media item {item_index} for content hash",
            intermediate.media.modality()
        )
    })?;
    Ok(hash_hex_strings(std::iter::once(hash)))
}

fn tensor_sizes_from_model_specific(
    model_specific: &HashMap<String, ModelSpecificValue>,
    key: &str,
) -> Result<Vec<usize>> {
    let value = model_specific
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing flat sizes tensor {key}"))?;
    value
        .as_flat_sizes()
        .with_context(|| format!("invalid flat sizes tensor {key}"))
}

fn item_span(sizes: &[usize], item_index: usize) -> Result<(usize, usize)> {
    let len = *sizes
        .get(item_index)
        .ok_or_else(|| anyhow::anyhow!("missing flat size for item {item_index}"))?;
    let start = sizes[..item_index]
        .iter()
        .try_fold(0usize, |acc, &size| acc.checked_add(size))
        .ok_or_else(|| anyhow::anyhow!("flat size offset overflow"))?;
    Ok((start, len))
}

fn hash_hex_strings<'a>(hashes: impl Iterator<Item = &'a str>) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    for hash in hashes {
        hasher.update(hash.as_bytes());
    }
    hasher.finalize().as_bytes().to_vec()
}

#[cfg(test)]
mod tests;
