//! The encoder-cache handoff checks of an encode-prefill-decode request, as the
//! Python servicer runs them per connector: a consumer's pixel-less batches
//! need a usable handle for every media item, a producer's completion carries
//! the request's media identity, and a CPU connector's producer must return a
//! fetch address for every item it encoded.

use serde_json::Value;
use smg_grpc_client::{media_identity, vllm_proto as vllm};
use tonic::Status;

use super::VllmModelInfo;

const CPU_CONNECTOR: &str = "ECCPUConnector";
const MOONCAKE_CONNECTOR: &str = "ECMooncakeConnector";

/// The refusal of a consumer's request whose handles do not cover its pixel-less
/// media, in the Python servicer's words.
const HANDLES_SHORT: &str = "multimodal payload carries grid tensors but no pixel_values and \
                             ec_transfer_params_json is not usable for every pixel-less media \
                             item; a pixel-less request requires a handoff";

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
/// object). A request with encoder-cache parameters may not rely on remote KV
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
/// their pixels (the one rule of `smg_grpc_client::media_identity`,
/// which the Router's decode side reads by too), for a request
/// whose batches all carry pixels and hashes. A media-reference request gets
/// its identity from the media stage instead; a request without batches, or
/// a pixel-less one (the consumer's request), gets none.
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
    let mut stripped = batches(req).filter_map(media_identity::identity);
    Some(vllm::MediaIdentity {
        prompt_token_ids,
        mm_inputs: stripped.next(),
        extra_mm_inputs: stripped.collect(),
    })
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
