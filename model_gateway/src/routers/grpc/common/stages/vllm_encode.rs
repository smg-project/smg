//! vLLM's encoder completes before prefill consumes its metadata-only prompt.

use axum::response::Response;
use serde_json::json;
use smg_grpc_client::vllm_proto as vllm;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        context::WorkerSelection,
        proto_wrapper::{ProtoGenerateRequest, ProtoGenerateResponse},
        utils::tonic_ext::TonicStatusExt,
    },
};

pub(super) async fn encode(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) -> Result<(), Response> {
    if request.sampling_n() != 1 {
        return Err(error::bad_request(
            "ec_multiple_samples",
            "vLLM EPD currently requires n=1",
        ));
    }
    let WorkerSelection::Disaggregated {
        encode_assignments: Some(assignments),
        prefill,
        ..
    } = workers
    else {
        return Err(error::service_unavailable(
            "ec_encoder_unavailable",
            "No vLLM encoder selected",
        ));
    };
    let [assignment] = assignments.as_slice() else {
        return Err(error::internal_error(
            "ec_encoder_assignment",
            "vLLM requires one encoder per prompt",
        ));
    };
    if !prefill
        .metadata()
        .spec
        .labels
        .get("supports_ec_transfer")
        .is_some_and(|value| value == "true")
    {
        return Err(error::service_unavailable(
            "ec_consumer_unsupported",
            "Prefill worker does not support EC transfer",
        ));
    }
    let ProtoGenerateRequest::Vllm(original) = request else {
        return Err(error::internal_error(
            "ec_request_type",
            "Expected a vLLM request",
        ));
    };
    let mut encoder = original.clone();
    encoder.request_id = format!("{}-encode-{}", encoder.request_id, Uuid::now_v7());
    encoder.stream = false;
    encoder.kv_transfer_params = None;
    encoder.kv_transfer_params_json = None;
    encoder.data_parallel_rank = assignment.worker.dp_rank().map(|rank| rank as i32);
    encoder.sampling_params = Some(vllm::SamplingParams {
        max_tokens: Some(1),
        n: 1,
        ignore_eos: true,
        ..Default::default()
    });
    let mut params = json!({});
    // Mooncake PUSH needs the exact selected DP replica, not the HTTP endpoint.
    // PULL connectors leave this list empty and return their handles from E.
    if let Some(addrs) = prefill.metadata().spec.labels.get("ec_zmq_addrs") {
        let addrs: Vec<String> = serde_json::from_str(addrs)
            .map_err(|e| error::internal_error("ec_consumer_address", e.to_string()))?;
        let rank = prefill.dp_rank().unwrap_or(0);
        let addr = addrs.get(rank).ok_or_else(|| {
            error::service_unavailable(
                "ec_consumer_address",
                "No EC destination for the selected DP rank",
            )
        })?;
        params["consumer_zmq"] = json!(addr);
        let count = encoder
            .media_refs
            .as_ref()
            .map_or(0, |refs| refs.items.len());
        params["ec_items"] = (0..count)
            .map(|_| json!({"transfer_id": Uuid::now_v7().to_string()}))
            .collect();
    }
    encoder.ec_transfer_params_json = Some(params.to_string());
    let backend = assignment
        .worker
        .get_backend_client()
        .await
        .map_err(|e| error::service_unavailable("ec_encoder_connection", e.to_string()))?
        .ok_or_else(|| {
            error::service_unavailable("ec_encoder_connection", "Encoder is not connected")
        })?;
    let mut client = backend.as_ref().clone();
    let mut stream = client
        .generate(ProtoGenerateRequest::Vllm(encoder))
        .await
        .map_err(|e| e.to_http_error("ec_encode_failed", e.message().to_string()))?;
    while let Some(reply) = stream.next().await {
        let reply =
            reply.map_err(|e| e.to_http_error("ec_encode_failed", e.message().to_string()))?;
        let ProtoGenerateResponse::Vllm(reply) = reply else {
            return Err(error::internal_error(
                "ec_encode_response",
                "Expected a vLLM encoder reply",
            ));
        };
        let Some(vllm::generate_response::Response::Complete(complete)) = reply.response else {
            continue;
        };
        if complete.finish_reason == "error" {
            return Err(error::bad_gateway("ec_encode_failed", "Encoder failed"));
        }
        let identity = complete.media_identity.ok_or_else(|| {
            error::bad_gateway("ec_metadata_missing", "Encoder returned no media identity")
        })?;
        let handles = complete.ec_transfer_params_json.ok_or_else(|| {
            error::bad_gateway(
                "ec_handles_missing",
                "Encoder returned no EC transfer handles",
            )
        })?;
        let mut handles: serde_json::Value = serde_json::from_str(&handles)
            .map_err(|e| error::bad_gateway("ec_handles_invalid", e.to_string()))?;
        if !handles.is_object() {
            return Err(error::bad_gateway(
                "ec_handles_invalid",
                "Expected an EC handle object",
            ));
        }
        if let Some(items) = params
            .get_mut("ec_items")
            .and_then(serde_json::Value::as_array_mut)
        {
            let mut features: Vec<_> = identity
                .mm_inputs
                .as_ref()
                .into_iter()
                .chain(identity.extra_mm_inputs.iter())
                .flat_map(|mm| {
                    mm.mm_placeholders
                        .iter()
                        .zip(&mm.mm_hashes)
                        .map(|(range, hash)| (range.offset, hash))
                })
                .collect();
            features.sort_by_key(|(offset, _)| *offset);
            if features.len() != items.len() {
                return Err(error::bad_gateway(
                    "ec_item_count",
                    "Encoder media count changed",
                ));
            }
            for (item, (_, hash)) in items.iter_mut().zip(features) {
                item["mm_hash"] = json!(hash);
            }
            // A hash-keyed response collapses duplicates; retain every push ID.
            handles["ec_items"] = json!(items);
        }
        if !request.apply_media_identity(&identity) {
            return Err(error::bad_gateway(
                "ec_metadata_invalid",
                "Encoder returned an invalid media identity",
            ));
        }
        if let ProtoGenerateRequest::Vllm(request) = request {
            request.ec_transfer_params_json = Some(handles.to_string());
        }
        stream.mark_completed();
        return Ok(());
    }
    Err(error::bad_gateway(
        "ec_encode_incomplete",
        "Encoder closed without a completion",
    ))
}
