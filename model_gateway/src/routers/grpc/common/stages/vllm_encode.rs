//! vLLM's encoder completes before prefill consumes its metadata-only prompt.

use axum::response::Response;
use serde_json::json;
use smg_grpc_client::vllm_proto as vllm;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        context::WorkerSelection,
        multimodal::{
            ec_push::{apply_completion, push_destination, PushCleanup},
            worker_supports_ec_transfer,
        },
        proto_wrapper::{ProtoGenerateRequest, ProtoGenerateResponse},
        utils::tonic_ext::TonicStatusExt,
    },
};

/// Encode one prompt, validating every handoff before removing its media.
pub(super) async fn encode(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) -> Result<PushCleanup, Response> {
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
    if !worker_supports_ec_transfer(prefill.as_ref()) {
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
    if let Some(addr) = push_destination(prefill.as_ref())? {
        params["consumer_zmq"] = json!(addr);
        let count = encoder.media_refs.as_ref().map_or_else(
            || {
                encoder
                    .mm_inputs
                    .iter()
                    .chain(&encoder.extra_mm_inputs)
                    .map(|mm| mm.mm_hashes.len())
                    .sum()
            },
            |refs| refs.items.len(),
        );
        params["ec_items"] = (0..count)
            .map(|_| json!({"transfer_id": Uuid::now_v7().to_string()}))
            .collect();
    }
    encoder.ec_transfer_params_json = Some(params.to_string());
    let cleanup = PushCleanup::new(&params);
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
        let connector = assignment
            .worker
            .metadata()
            .spec
            .labels
            .get("ec_connector")
            .map(String::as_str)
            .unwrap_or("");
        apply_completion(request, complete, &mut params, connector)?;
        stream.mark_completed();
        return Ok(cleanup);
    }
    Err(error::bad_gateway(
        "ec_encode_incomplete",
        "Encoder closed without a completion",
    ))
}
