//! The encoder-cache handoff checks of an encode-prefill-decode leg, as the
//! Python servicer runs them per connector: a consumer's pixel-less batches
//! need a usable handle for every media item, a producer's completion carries
//! the request's media identity, and a CPU connector's producer must return a
//! fetch address for every item it encoded.

use std::collections::HashMap;

use serde_json::Value;
use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;

use super::VllmModelInfo;

const CPU_CONNECTOR: &str = "ECCPUConnector";
const MOONCAKE_CONNECTOR: &str = "ECMooncakeConnector";

/// The per-item tensors a prefill side needs besides pixels: the M-RoPE
/// grids and the video timing that goes with them, in both spellings (the
/// Python servicer's `GRID_KEYS`, the Router's `VLLM_MROPE_GRID_KEYS`).
const GRID_KEYS: [&str; 4] = [
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "video_second_per_grid",
];

/// The refusal of a consumer's leg whose handles do not cover its pixel-less
/// media, in the Python servicer's words.
const HANDLES_SHORT: &str = "multimodal payload carries grid tensors but no pixel_values and \
                             ec_transfer_params_json is not usable for every pixel-less media \
                             item; a pixel-less leg requires a handoff";

/// Whether the engine consumes encoder caches (the prefill side), off the
/// launcher's role.
pub(super) fn is_consumer(model: &VllmModelInfo) -> bool {
    matches!(model.ec_role.as_str(), "ec_consumer" | "ec_both")
}

/// Whether the engine produces encoder caches (the encoder), off the
/// launcher's role.
pub(super) fn is_producer(model: &VllmModelInfo) -> bool {
    matches!(model.ec_role.as_str(), "ec_producer" | "ec_both")
}

/// Every multimodal batch of the request, `mm_inputs` first (a request
/// mixing modalities carries one batch per modality).
fn batches(req: &vllm::GenerateRequest) -> impl Iterator<Item = &vllm::MultimodalInputs> {
    req.mm_inputs.iter().chain(req.extra_mm_inputs.iter())
}

/// A CPU connector's handle names the address the encoding is fetched from.
fn valid_cpu_handle(handle: Option<&Value>) -> bool {
    let Some(handle) = handle.and_then(Value::as_object) else {
        return false;
    };
    handle
        .get("peer_host")
        .and_then(Value::as_str)
        .is_some_and(|host| !host.is_empty())
        && handle
            .get("peer_port")
            .and_then(Value::as_u64)
            .is_some_and(|port| (1..=65535).contains(&port))
}

/// The consumer side's check of the encoder's handles before the engine sees
/// the request: every pixel-less batch carries its hashes, and per connector
/// each hash has what the engine fetches by (a CPU connector's address, a
/// Mooncake transfer id per occurrence; any other connector takes a non-empty
/// object). A leg with encoder-cache parameters may not rely on remote KV
/// alone, as the Python servicer rules; one without pixel-less batches needs
/// no handle.
pub(super) fn check_consumer_handoff(
    req: &vllm::GenerateRequest,
    model: &VllmModelInfo,
    params: &Value,
) -> Result<(), Status> {
    let missing: Vec<&vllm::MultimodalInputs> = batches(req)
        .filter(|batch| batch.pixel_values.is_none())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let named = params.as_object().is_some_and(|object| !object.is_empty());
    if is_consumer(model) && named && handles_cover(&missing, &model.ec_connector, params) {
        Ok(())
    } else {
        Err(Status::invalid_argument(HANDLES_SHORT))
    }
}

fn handles_cover(missing: &[&vllm::MultimodalInputs], connector: &str, params: &Value) -> bool {
    if missing.iter().any(|batch| batch.mm_hashes.is_empty())
        && matches!(connector, CPU_CONNECTOR | MOONCAKE_CONNECTOR)
    {
        return false;
    }
    match connector {
        CPU_CONNECTOR => missing
            .iter()
            .flat_map(|batch| batch.mm_hashes.iter())
            .all(|hash| valid_cpu_handle(params.get(hash))),
        MOONCAKE_CONNECTOR => {
            // Mooncake consumes one transfer id per occurrence of a hash.
            let mut ids: Vec<&str> = params
                .get("ec_items")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter(|item| {
                            item.get("transfer_id")
                                .and_then(Value::as_str)
                                .is_some_and(|id| !id.is_empty())
                        })
                        .filter_map(|item| item.get("mm_hash").and_then(Value::as_str))
                        .collect()
                })
                .unwrap_or_default();
            missing
                .iter()
                .flat_map(|batch| batch.mm_hashes.iter())
                .all(|hash| match ids.iter().position(|id| id == hash) {
                    Some(at) => {
                        ids.swap_remove(at);
                        true
                    }
                    None => false,
                })
        }
        _ => true,
    }
}

/// A producer's completion carries the media identity the Router hands the
/// prefill side: the prompt's token ids and the request's batches without
/// their pixels (the hashes, the placeholders, the image token id, the
/// modality and the grid tensors with inline payloads), for a request
/// whose batches all carry pixels and hashes. A media-reference request gets
/// its identity from the media stage instead; a request without batches, or
/// a pixel-less one (the consumer's leg), gets none.
pub(super) fn producer_identity(
    req: &vllm::GenerateRequest,
    model: &VllmModelInfo,
) -> Option<vllm::MediaIdentity> {
    if !is_producer(model) || batches(req).next().is_none() {
        return None;
    }
    if batches(req).any(|batch| batch.pixel_values.is_none() || batch.mm_hashes.is_empty()) {
        return None;
    }
    let prompt_token_ids = match req.input.as_ref() {
        Some(vllm::generate_request::Input::Tokenized(tokenized)) => tokenized.input_ids.clone(),
        _ => return None,
    };
    let mut stripped = batches(req).map(without_pixels);
    Some(vllm::MediaIdentity {
        prompt_token_ids,
        mm_inputs: stripped.next(),
        extra_mm_inputs: stripped.collect(),
    })
}

/// The batch as the identity carries it, the shape the Python servicer's
/// `build_media_identity` and the Router's PD-decode identity give it: the
/// hashes, the placeholders, the image token id and the modality as sent,
/// the grid tensors with inline payloads (a shared-memory segment is unlinked
/// once the adapter has read it, so a handle to it would reach the prefill
/// side dead) with the size tensors of the flat ones, every other tensor
/// dropped with the pixels, and the key lists trimmed to what stays.
fn without_pixels(batch: &vllm::MultimodalInputs) -> vllm::MultimodalInputs {
    let inline = |key: &str| {
        batch
            .model_specific_tensors
            .get(key)
            .filter(|tensor| matches!(tensor.payload, Some(vllm::tensor_data::Payload::Inline(_))))
            .cloned()
    };
    let mut tensors: HashMap<String, vllm::TensorData> = GRID_KEYS
        .iter()
        .filter_map(|key| inline(key).map(|tensor| (key.to_string(), tensor)))
        .collect();
    let mut flat_keys = HashMap::new();
    for (key, sizes_key) in &batch.flat_keys {
        if !tensors.contains_key(key) {
            continue;
        }
        match inline(sizes_key) {
            Some(sizes) => {
                tensors.insert(sizes_key.clone(), sizes);
                flat_keys.insert(key.clone(), sizes_key.clone());
            }
            None => {
                tensors.remove(key);
            }
        }
    }
    let retained = |keys: &[String]| -> Vec<String> {
        keys.iter()
            .filter(|key| tensors.contains_key(*key))
            .cloned()
            .collect()
    };
    let batched_keys = retained(&batch.batched_keys);
    let keep_on_cpu_keys = retained(&batch.keep_on_cpu_keys);
    vllm::MultimodalInputs {
        pixel_values: None,
        batched_keys,
        keep_on_cpu_keys,
        flat_keys,
        model_specific_tensors: tensors,
        ..batch.clone()
    }
}

/// A CPU connector's producer must return a fetch address for every media
/// item of the identity: the connector may evict an encoding before the
/// completion and return metadata without one, and the prefill side has no
/// pixels to recompute it, so the completion is refused before it becomes a
/// handoff (the Python servicer's rule). Any other connector's answer passes.
pub(super) fn check_producer_handles(
    model: &VllmModelInfo,
    identity: &vllm::MediaIdentity,
    complete: &vllm::GenerateComplete,
) -> Result<(), Status> {
    if model.ec_connector != CPU_CONNECTOR {
        return Ok(());
    }
    let handles: Value = complete
        .ec_transfer_params_json
        .as_deref()
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or(Value::Null);
    let covered = identity
        .mm_inputs
        .iter()
        .chain(identity.extra_mm_inputs.iter())
        .flat_map(|batch| batch.mm_hashes.iter())
        .all(|hash| valid_cpu_handle(handles.get(hash)));
    if covered {
        Ok(())
    } else {
        Err(Status::internal(
            "Encoder returned an incomplete ECCPU transfer handle",
        ))
    }
}
