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
    EncoderFieldLayouts, FieldLayout, Modality, ModelSpecificValue, PlaceholderRange,
    PreprocessedEncoderInputs,
};
use ndarray::ArrayViewD;
use smg_grpc_client::common_proto as common;
use tracing::{info, warn};

use super::{
    capability::ensure_backend_supports_modalities,
    log_mm_timing_enabled,
    serialize::{
        model_specific_to_tensor_bytes, serialize_array_as_tokenspeed_tensor,
        serialize_encoder_input, serialize_model_specific, slice_array_axis0,
    },
    transport::{mm_encoder_input_dtype, resolve_mm_shm_enabled, resolve_mm_shm_min_bytes},
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
        BackendClient::Grpc(GrpcClient::Sglang(_)) => {
            let batch = into_single_batch(intermediate, "SGLang")?;
            Ok(MultimodalData::Sglang(assemble_sglang(batch)?))
        }
        BackendClient::Grpc(GrpcClient::Vllm(_)) => {
            let batch = into_single_batch(intermediate, "vLLM")?;
            Ok(MultimodalData::Vllm(assemble_vllm(batch, workers)?))
        }
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
            let batches = intermediate.into_batches();
            let pending = tokio::task::spawn_blocking(move || {
                assemble_tokenspeed_batches(&batches, options).map(PendingTokenSpeedAssembly::new)
            })
            .await
            .context("TokenSpeed multimodal assembly task failed")??;
            Ok(MultimodalData::TokenSpeed(pending.into_inner()?))
        }
        BackendClient::Grpc(GrpcClient::Mlx(_)) => {
            anyhow::bail!("MLX does not support multimodal inputs")
        }
        BackendClient::Smg(_) => {
            anyhow::bail!("WorkerInference v1 does not support multimodal inputs")
        }
        BackendClient::Zmq(client) => match client.runtime() {
            // connect() admits only vLLM/TokenSpeed runtimes over ZMQ, so no
            // Unspecified fallback: anything else is a hard error below.
            RuntimeType::Vllm => {
                let batch = into_single_batch(intermediate, "vLLM")?;
                let mut data = assemble_vllm(batch, workers)?;
                // The ZMQ translate reads tensor bytes inline; this wire has no
                // /dev/shm or RDMA pull on the engine side.
                data.shm_enabled = false;
                data.rdma_enabled = false;
                Ok(MultimodalData::Vllm(data))
            }
            runtime => anyhow::bail!(
                "multimodal inputs are not supported over the {runtime} ZMQ backend yet"
            ),
        },
    }
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

/// Backends other than TokenSpeed take a single preprocessed batch (one
/// modality). The per-modality capability is enforced by
/// [`ensure_client_supports_intermediate`]; this only enforces the structural
/// single-batch constraint (multiple modalities in one request are TokenSpeed-
/// only).
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
    let (pixel_values, pixel_values_shape) = serialize_encoder_input(&intermediate.preprocessed);
    let model_specific_tensors = serialize_model_specific(intermediate.preprocessed.model_specific);
    let MediaBatch::Images(images) = &intermediate.media else {
        anyhow::bail!("SGLang assembly requires an image batch");
    };
    let image_data = images.iter().map(|f| f.raw_bytes.to_vec()).collect();
    let mm_placeholders = placeholders_for_bindings(&intermediate.bindings, true)?;

    Ok(SglangMultimodalData {
        image_data,
        pixel_values,
        pixel_values_shape,
        model_specific_tensors,
        im_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
    })
}

fn assemble_vllm(
    intermediate: PrecomputedMultimodalIntermediate,
    workers: Option<&WorkerSelection>,
) -> Result<VllmMultimodalData> {
    let (pixel_values, pixel_values_shape) = serialize_encoder_input(&intermediate.preprocessed);
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
    let (batched_keys, flat_keys) = vllm_field_layout_keys(&intermediate.field_layouts);

    Ok(VllmMultimodalData {
        pixel_values,
        pixel_values_shape,
        model_specific_tensors,
        im_token_id: intermediate.placeholder_token_id,
        mm_placeholders,
        mm_hashes,
        batched_keys,
        flat_keys,
        keep_on_cpu_keys: intermediate.keep_on_cpu_keys,
        modality,
        shm_enabled: resolve_mm_shm_enabled(workers, false),
        shm_min_bytes: resolve_mm_shm_min_bytes(workers),
        // vLLM workers cannot pull RDMA payloads yet.
        rdma_enabled: false,
    })
}

/// Translate the neutral layout contract to vLLM's legacy HF field names.
fn vllm_field_layout_keys(layouts: &EncoderFieldLayouts) -> (Vec<String>, HashMap<String, String>) {
    let mut batched_keys = PreprocessedEncoderInputs::batched_keys(&layouts.model_specific);
    let mut flat_keys = PreprocessedEncoderInputs::flat_keys(&layouts.model_specific);
    match &layouts.encoder_input {
        FieldLayout::Batched => batched_keys.push("pixel_values".to_string()),
        FieldLayout::Flat { sizes_key } => {
            flat_keys.insert("pixel_values".to_string(), sizes_key.clone());
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
    assemble_tokenspeed_with_options(intermediate, options)
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

fn assemble_tokenspeed_with_options(
    intermediate: &PrecomputedMultimodalIntermediate,
    options: TokenSpeedAssemblyOptions,
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
            serialize_array_as_tokenspeed_tensor(
                &item_encoder_input,
                &encoder_input_dtype,
                shm_enabled,
                shm_min_bytes,
            )
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

fn assemble_tokenspeed_batches(
    batches: &[PrecomputedMultimodalIntermediate],
    options: Vec<TokenSpeedAssemblyOptions>,
) -> Result<TokenSpeedMultimodalData> {
    anyhow::ensure!(
        batches.len() == options.len(),
        "multimodal batch/assembly option count mismatch"
    );

    let shm_enabled = options
        .first()
        .map(|opts| opts.shm_enabled)
        .unwrap_or(false);
    let shm_min_bytes = options.first().map(|opts| opts.shm_min_bytes).unwrap_or(0);
    let mut ordered_items: Vec<(usize, TokenSpeedMultimodalItem)> = Vec::new();

    for (batch, options) in batches.iter().zip(options) {
        match assemble_tokenspeed_with_options(batch, options) {
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
) -> Result<ArrayViewD<'a, f32>> {
    match layout {
        FieldLayout::Batched => slice_array_axis0(&preprocessed.encoder_input, item_index, 1),
        FieldLayout::Flat { sizes_key } => {
            let sizes = tensor_sizes_from_model_specific(&preprocessed.model_specific, sizes_key)?;
            let (start, len) = item_span(&sizes, item_index)?;
            slice_array_axis0(&preprocessed.encoder_input, start, len)
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
mod tests {
    #[cfg(target_os = "linux")]
    use std::path::Path;
    use std::{mem::size_of, sync::Arc};

    use llm_multimodal::{
        audio::DecodedAudio, AudioClip, AudioSource, ImageDetail, ImageFrame, VideoClip,
    };
    use ndarray::{ArrayD, IxDyn};

    use super::*;
    #[cfg(target_os = "linux")]
    use crate::routers::grpc::proto_wrapper::write_tokenspeed_shm_with;

    #[cfg(target_os = "linux")]
    fn pending_tokenspeed_shm_assembly() -> (PendingTokenSpeedAssembly, std::path::PathBuf) {
        let handle = write_tokenspeed_shm_with(4, |output| {
            output.copy_from_slice(&[1, 2, 3, 4]);
            Ok(())
        })
        .unwrap();
        let path = Path::new("/dev/shm").join(&handle.name);
        let data = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::shm(handle, vec![2], "bfloat16".to_string()),
                model_specific_tensors: HashMap::new(),
                placeholder_token_id: None,
                mm_placeholders: vec![],
                content_hash: vec![],
            }],
            shm_enabled: true,
            shm_min_bytes: 0,
        };
        (PendingTokenSpeedAssembly::new(data), path)
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn pending_tokenspeed_assembly_cleans_shm_when_dropped() {
        let (pending, path) = pending_tokenspeed_shm_assembly();
        assert!(path.exists());

        drop(pending);

        assert!(!path.exists());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn pending_tokenspeed_assembly_transfers_shm_ownership() {
        let (pending, path) = pending_tokenspeed_shm_assembly();
        let data = pending.into_inner().unwrap();
        assert!(path.exists());

        cleanup_tokenspeed_items_encoder_shm(&data.items, None);

        assert!(!path.exists());
    }

    #[test]
    fn assemble_tokenspeed_splits_image_items() {
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "patches_per_image".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![2, 2],
                shape: vec![2],
            },
        );
        model_specific.insert(
            "image_grid_thw".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![1, 2, 3, 4, 5, 6],
                shape: vec![2, 3],
            },
        );

        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_shape_vec(
                IxDyn(&[4, 2]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
            feature_token_counts: vec![2, 2],
            item_sizes: vec![(1, 1), (1, 1)],
            model_specific,
        };

        let images = vec![
            Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                bytes::Bytes::from_static(b"a"),
                ImageDetail::Auto,
                llm_multimodal::ImageSource::InlineBytes,
                "hash-a".to_string(),
            )),
            Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                bytes::Bytes::from_static(b"b"),
                ImageDetail::Auto,
                llm_multimodal::ImageSource::InlineBytes,
                "hash-b".to_string(),
            )),
        ];

        let intermediate = PrecomputedMultimodalIntermediate {
            preprocessed,
            media: MediaBatch::Images(images),
            bindings: vec![
                PromptBinding {
                    item_index: 0,
                    prompt_ordinal: 0,
                    structural: PlaceholderRange {
                        offset: 10,
                        length: 2,
                    },
                    patches: vec![PlaceholderRange {
                        offset: 10,
                        length: 2,
                    }],
                },
                PromptBinding {
                    item_index: 1,
                    prompt_ordinal: 1,
                    structural: PlaceholderRange {
                        offset: 20,
                        length: 2,
                    },
                    patches: vec![PlaceholderRange {
                        offset: 20,
                        length: 2,
                    }],
                },
            ],
            placeholder_token_id: Some(151655),
            field_layouts: EncoderFieldLayouts::new(
                FieldLayout::flat("patches_per_image"),
                HashMap::from([
                    ("patches_per_image".to_string(), FieldLayout::Batched),
                    ("image_grid_thw".to_string(), FieldLayout::Batched),
                ]),
            ),
            keep_on_cpu_keys: vec![],
        };

        let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
        assert_eq!(assembled.items.len(), 2);

        let first = &assembled.items[0];
        assert_eq!(first.modality, TokenSpeedModality::Image);
        assert_eq!(first.encoder_input.shape, vec![2, 2]);
        // bf16 is the default TokenSpeed encoder_input wire dtype (2 bytes/elem).
        assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<u16>());
        assert_eq!(first.mm_placeholders, vec![(10, 2)]);
        assert_eq!(
            first.content_hash,
            hash_hex_strings(std::iter::once("hash-a"))
        );
        assert_eq!(
            first.model_specific_tensors["image_grid_thw"].shape,
            vec![1, 3]
        );
        assert_eq!(
            first.model_specific_tensors["patches_per_image"].shape,
            vec![1]
        );

        let second = &assembled.items[1];
        assert_eq!(second.encoder_input.shape, vec![2, 2]);
        assert_eq!(second.mm_placeholders, vec![(20, 2)]);
        assert_eq!(
            second.content_hash,
            hash_hex_strings(std::iter::once("hash-b"))
        );
        assert_eq!(
            second.model_specific_tensors["image_grid_thw"].shape,
            vec![1, 3]
        );
    }

    #[test]
    fn assemble_tokenspeed_splits_video_items() {
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "patches_per_video".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![2, 2],
                shape: vec![2],
            },
        );
        model_specific.insert(
            "video_grid_thw".to_string(),
            ModelSpecificValue::UintTensor {
                data: vec![1, 2, 3, 4, 5, 6],
                shape: vec![2, 3],
            },
        );

        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_shape_vec(
                IxDyn(&[4, 2]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
            feature_token_counts: vec![2, 2],
            item_sizes: vec![(1, 1), (1, 1)],
            model_specific,
        };

        let videos = vec![
            Arc::new(VideoClip::new(
                vec![image::DynamicImage::new_rgb8(1, 1)],
                bytes::Bytes::from_static(b"a"),
                llm_multimodal::VideoSource::InlineBytes,
                "video-hash-a".to_string(),
            )),
            Arc::new(VideoClip::new(
                vec![image::DynamicImage::new_rgb8(1, 1)],
                bytes::Bytes::from_static(b"b"),
                llm_multimodal::VideoSource::InlineBytes,
                "video-hash-b".to_string(),
            )),
        ];

        let intermediate = PrecomputedMultimodalIntermediate {
            preprocessed,
            media: MediaBatch::Videos(videos),
            bindings: vec![
                PromptBinding {
                    item_index: 0,
                    prompt_ordinal: 0,
                    structural: PlaceholderRange {
                        offset: 30,
                        length: 2,
                    },
                    patches: vec![PlaceholderRange {
                        offset: 30,
                        length: 2,
                    }],
                },
                PromptBinding {
                    item_index: 1,
                    prompt_ordinal: 1,
                    structural: PlaceholderRange {
                        offset: 40,
                        length: 2,
                    },
                    patches: vec![PlaceholderRange {
                        offset: 40,
                        length: 2,
                    }],
                },
            ],
            placeholder_token_id: Some(151656),
            field_layouts: EncoderFieldLayouts::new(
                FieldLayout::flat("patches_per_video"),
                HashMap::from([
                    ("patches_per_video".to_string(), FieldLayout::Batched),
                    ("video_grid_thw".to_string(), FieldLayout::Batched),
                ]),
            ),
            keep_on_cpu_keys: vec![],
        };

        let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
        assert_eq!(assembled.items.len(), 2);

        let first = &assembled.items[0];
        assert_eq!(first.modality, TokenSpeedModality::Video);
        assert_eq!(first.encoder_input.shape, vec![2, 2]);
        // bf16 is the default TokenSpeed encoder_input wire dtype (2 bytes/elem).
        assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<u16>());
        assert_eq!(first.mm_placeholders, vec![(30, 2)]);
        assert_eq!(
            first.content_hash,
            hash_hex_strings(std::iter::once("video-hash-a"))
        );
        assert_eq!(
            first.model_specific_tensors["video_grid_thw"].shape,
            vec![1, 3]
        );
        assert_eq!(
            first.model_specific_tensors["patches_per_video"].shape,
            vec![1]
        );

        let second = &assembled.items[1];
        assert_eq!(second.encoder_input.shape, vec![2, 2]);
        assert_eq!(second.mm_placeholders, vec![(40, 2)]);
        assert_eq!(
            second.content_hash,
            hash_hex_strings(std::iter::once("video-hash-b"))
        );
        assert_eq!(
            second.model_specific_tensors["video_grid_thw"].shape,
            vec![1, 3]
        );
    }

    #[test]
    fn assemble_tokenspeed_splits_audio_items_as_bfloat16_by_default() {
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "row_lengths".to_string(),
            ModelSpecificValue::IntTensor {
                data: vec![2, 2],
                shape: vec![2],
            },
        );

        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_shape_vec(
                IxDyn(&[4, 2]),
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            )
            .unwrap(),
            feature_token_counts: vec![2, 2],
            item_sizes: vec![(2, 2), (2, 2)],
            model_specific,
        };

        let audios = vec![
            Arc::new(AudioClip::new(
                bytes::Bytes::from_static(b"a"),
                DecodedAudio {
                    samples: Vec::new(),
                    sample_rate: 16_000,
                },
                AudioSource::InlineBytes,
                "audio-hash-a".to_string(),
            )),
            Arc::new(AudioClip::new(
                bytes::Bytes::from_static(b"b"),
                DecodedAudio {
                    samples: Vec::new(),
                    sample_rate: 16_000,
                },
                AudioSource::InlineBytes,
                "audio-hash-b".to_string(),
            )),
        ];

        let intermediate = PrecomputedMultimodalIntermediate {
            preprocessed,
            media: MediaBatch::Audios(audios),
            bindings: vec![
                PromptBinding {
                    item_index: 0,
                    prompt_ordinal: 0,
                    structural: PlaceholderRange {
                        offset: 30,
                        length: 2,
                    },
                    patches: vec![PlaceholderRange {
                        offset: 30,
                        length: 2,
                    }],
                },
                PromptBinding {
                    item_index: 1,
                    prompt_ordinal: 1,
                    structural: PlaceholderRange {
                        offset: 40,
                        length: 2,
                    },
                    patches: vec![PlaceholderRange {
                        offset: 40,
                        length: 2,
                    }],
                },
            ],
            placeholder_token_id: Some(42),
            field_layouts: EncoderFieldLayouts::new(
                FieldLayout::flat("row_lengths"),
                HashMap::from([("row_lengths".to_string(), FieldLayout::Batched)]),
            ),
            keep_on_cpu_keys: vec![],
        };

        let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
        assert_eq!(assembled.items.len(), 2);

        let first = &assembled.items[0];
        assert_eq!(first.modality, TokenSpeedModality::Audio);
        assert_eq!(first.encoder_input.dtype, "bfloat16");
        assert_eq!(first.encoder_input.shape, vec![2, 2]);
        assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<u16>());
        assert_eq!(first.mm_placeholders, vec![(30, 2)]);
        assert_eq!(
            first.content_hash,
            hash_hex_strings(std::iter::once("audio-hash-a"))
        );
        assert_eq!(first.model_specific_tensors["row_lengths"].shape, vec![1]);

        let second = &assembled.items[1];
        assert_eq!(second.encoder_input.dtype, "bfloat16");
        assert_eq!(second.encoder_input.shape, vec![2, 2]);
        assert_eq!(second.mm_placeholders, vec![(40, 2)]);
        assert_eq!(
            second.content_hash,
            hash_hex_strings(std::iter::once("audio-hash-b"))
        );
    }

    #[test]
    fn tokenspeed_epd_items_and_hashes_follow_prompt_binding_order() {
        let one_item_inputs = || PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap(),
            feature_token_counts: vec![1],
            item_sizes: vec![(1, 1)],
            model_specific: HashMap::new(),
        };
        let image_batch = PrecomputedMultimodalIntermediate {
            preprocessed: one_item_inputs(),
            media: MediaBatch::Images(vec![Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                bytes::Bytes::from_static(b"image"),
                ImageDetail::Auto,
                llm_multimodal::ImageSource::InlineBytes,
                "image-hash".to_string(),
            ))]),
            // Deliberately earlier by offset but later by explicit ordinal.
            bindings: vec![PromptBinding {
                item_index: 0,
                prompt_ordinal: 2,
                structural: PlaceholderRange {
                    offset: 1,
                    length: 1,
                },
                patches: vec![],
            }],
            placeholder_token_id: Some(10),
            field_layouts: EncoderFieldLayouts::default(),
            keep_on_cpu_keys: vec![],
        };
        let video_batch = PrecomputedMultimodalIntermediate {
            preprocessed: one_item_inputs(),
            media: MediaBatch::Videos(vec![Arc::new(VideoClip::new(
                vec![image::DynamicImage::new_rgb8(1, 1)],
                bytes::Bytes::from_static(b"video"),
                llm_multimodal::VideoSource::InlineBytes,
                "video-hash".to_string(),
            ))]),
            bindings: vec![PromptBinding {
                item_index: 0,
                prompt_ordinal: 1,
                structural: PlaceholderRange {
                    offset: 50,
                    length: 1,
                },
                patches: vec![],
            }],
            placeholder_token_id: Some(30),
            field_layouts: EncoderFieldLayouts::default(),
            keep_on_cpu_keys: vec![],
        };
        let audio_batch = PrecomputedMultimodalIntermediate {
            preprocessed: one_item_inputs(),
            media: MediaBatch::Audios(vec![Arc::new(AudioClip::new(
                bytes::Bytes::from_static(b"audio"),
                DecodedAudio {
                    samples: Vec::new(),
                    sample_rate: 16_000,
                },
                AudioSource::InlineBytes,
                "audio-hash".to_string(),
            ))]),
            bindings: vec![PromptBinding {
                item_index: 0,
                prompt_ordinal: 0,
                structural: PlaceholderRange {
                    offset: 100,
                    length: 1,
                },
                patches: vec![],
            }],
            placeholder_token_id: Some(20),
            field_layouts: EncoderFieldLayouts::default(),
            keep_on_cpu_keys: vec![],
        };
        let intermediate =
            MultimodalIntermediate::try_new(vec![image_batch, video_batch, audio_batch]).unwrap();
        let routing_hashes = encode_routing_hashes(&intermediate).unwrap();
        let assembled = assemble_tokenspeed_for_encode(&intermediate, None).unwrap();

        assert_eq!(assembled.items.len(), 3);
        assert_eq!(assembled.items[0].modality, TokenSpeedModality::Audio);
        assert_eq!(assembled.items[0].mm_placeholders, vec![(100, 1)]);
        assert_eq!(assembled.items[1].modality, TokenSpeedModality::Video);
        assert_eq!(assembled.items[1].mm_placeholders, vec![(50, 1)]);
        assert_eq!(assembled.items[2].modality, TokenSpeedModality::Image);
        assert_eq!(assembled.items[2].mm_placeholders, vec![(1, 1)]);
        assert_eq!(
            routing_hashes,
            assembled
                .items
                .iter()
                .map(|item| item.content_hash.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn qwen_audio_batched_layout_slices_items_not_mel_bins() {
        let mut model_specific = HashMap::new();
        model_specific.insert(
            "feature_attention_mask".to_string(),
            ModelSpecificValue::int_2d(vec![1, 1, 1, 1, 1, 1, 0, 0], 2, 4),
        );
        model_specific.insert(
            "audio_feature_lengths".to_string(),
            ModelSpecificValue::int_1d(vec![4, 2]),
        );
        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_shape_vec(
                IxDyn(&[2, 3, 4]),
                (0..24).map(|value| value as f32).collect(),
            )
            .unwrap(),
            feature_token_counts: vec![1, 1],
            item_sizes: vec![(3, 4), (3, 2)],
            model_specific,
        };
        let layouts = EncoderFieldLayouts::new(
            FieldLayout::Batched,
            HashMap::from([
                ("feature_attention_mask".to_string(), FieldLayout::Batched),
                ("audio_feature_lengths".to_string(), FieldLayout::Batched),
            ]),
        );

        let second = encoder_input_for_item(&preprocessed, &layouts.encoder_input, 1).unwrap();
        assert_eq!(second.shape(), &[1, 3, 4]);
        assert_eq!(
            second.iter().copied().collect::<Vec<_>>(),
            (12..24).map(|v| v as f32).collect::<Vec<_>>()
        );

        let extras = serialize_model_specific_for_item(
            &preprocessed.model_specific,
            &layouts.model_specific,
            1,
        )
        .unwrap();
        assert_eq!(extras["feature_attention_mask"].shape, vec![1, 4]);
        assert_eq!(extras["audio_feature_lengths"].shape, vec![1]);
    }

    #[test]
    fn vllm_layout_adapter_restores_legacy_primary_field_name() {
        let flat = EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_image"),
            HashMap::from([("image_grid_thw".to_string(), FieldLayout::Batched)]),
        );
        let (batched_keys, flat_keys) = vllm_field_layout_keys(&flat);
        assert_eq!(batched_keys, vec!["image_grid_thw"]);
        assert_eq!(
            flat_keys.get("pixel_values").map(String::as_str),
            Some("patches_per_image")
        );

        let batched = EncoderFieldLayouts::default();
        let (batched_keys, flat_keys) = vllm_field_layout_keys(&batched);
        assert_eq!(batched_keys, vec!["pixel_values"]);
        assert!(flat_keys.is_empty());
    }
}
