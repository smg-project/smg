//! SGLang metadata adaptation; membership, placement and admission belong to SMG.
use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::response::Response;
use futures::future::try_join_all;
use llm_multimodal::{ImageDetail, MediaContentPart};
use prost::Message;
use sha2::{Digest, Sha256};
use smg_grpc_client::{
    common_proto as common,
    sglang_encoder_proto::{EncodeRequest, ImageOption},
    sglang_managed::{capabilities_compatible, proto, Limits, LAYOUT, VERSION},
    sglang_proto, vllm_proto as vllm,
};
use smg_worker::identity::basic::{Worker, WorkerLoadGuard, WorkerType};
use tokio::sync::oneshot;
use tonic::Status;
use uuid::Uuid;

use crate::{
    config::types::PdPairingMode,
    routers::{
        common::{overload, snapshot::RoutingPool},
        grpc::{
            backend_client::{get_backend_client, BackendClient},
            client::GrpcClient,
            context::{RequestContext, RequestType, WorkerSelection},
            multimodal::{
                assemble_media_refs, check_forwarded_references, media_plan_chat,
                media_plan_messages, MmRefsError,
            },
            proto_wrapper::ProtoGenerateResponse,
            utils::tonic_ext::TonicStatusExt,
        },
    },
};

/// Worker label key carrying the JSON-encoded SGLang managed EPD capability contract.
pub const CAPABILITY: &str = "sglang_prepared";

/// Bounded handoff budget for E encode, P layout export, and D admission; once D is
/// submitted, total decode duration (streaming or buffered) follows `--grpc-request-timeout-secs`.
const PREPARED_TIMEOUT: Duration = Duration::from_secs(120);

/// Maximum total media reference bytes accepted ahead of encoder dispatch (256 MiB encoder
/// control budget minus 1 MiB protobuf/JSON framing headroom).
const MAX_MEDIA_REF_BYTES: usize = 255 * 1024 * 1024;

/// Failure preparing a request for SGLang managed EPD, preserving [`MmRefsError`] codes.
#[derive(Debug)]
pub(crate) enum PrepareError {
    /// Media reference validation or assembly refusal with its own stable error code.
    MediaRefs(MmRefsError),
    /// SGLang EPD capability or payload-budget refusal.
    Capability(Status),
}

impl PrepareError {
    /// Convert into an HTTP response while preserving `MmRefsError` error codes.
    pub(crate) fn into_response(self) -> Response {
        match self {
            Self::MediaRefs(err) => err.into_response(),
            Self::Capability(status) => {
                status.to_http_error("sglang_epd_capability", status.message().to_owned())
            }
        }
    }
}

/// Tonic URIs use HTTP schemes; SMG registration uses a plain `grpc://` hint.
pub(crate) fn worker_registration_url(url: &str) -> String {
    url.strip_prefix("http://")
        .map_or_else(|| url.to_owned(), |address| format!("grpc://{address}"))
}

/// Read capabilities obtained by the ordinary worker registration workflow.
pub(crate) fn capability(worker: &dyn Worker) -> Option<proto::Capabilities> {
    serde_json::from_str(worker.metadata().spec.labels.get(CAPABILITY)?).ok()
}

/// Before placement, require a compatible E for this P/D, not merely the same engine.
pub(crate) fn compatible(encoder: &dyn Worker, worker: &dyn Worker) -> bool {
    let (Some(e), Some(p)) = (capability(encoder), capability(worker)) else {
        return false;
    };
    compatible_contracts(encoder, &e, worker, &p)
}

/// Verify that `encoder` and a P or D `worker` share runtime and compatible prepared contracts.
fn compatible_contracts(
    encoder: &dyn Worker,
    e: &proto::Capabilities,
    worker: &dyn Worker,
    p: &proto::Capabilities,
) -> bool {
    let expected_role = match worker.worker_type() {
        WorkerType::Prefill => "prefill",
        WorkerType::Decode => "decode",
        _ => return false,
    };
    encoder.metadata().spec.runtime_type == worker.metadata().spec.runtime_type
        && capabilities_compatible(e, p, expected_role)
}

/// Verify that an E/P/D triplet shares compatible prepared contracts and matching KV backends.
fn compatible_pair(
    encoder: &dyn Worker,
    e: &proto::Capabilities,
    prefill: &dyn Worker,
    p: &proto::Capabilities,
    decode: &dyn Worker,
    d: &proto::Capabilities,
) -> bool {
    compatible_contracts(encoder, e, prefill, p)
        && compatible_contracts(encoder, e, decode, d)
        && p.kv_backend == d.kv_backend
}

/// Use the same contracts, pairing rules, and overload-fallback semantics for readiness
/// and request placement.
pub(crate) fn pipeline_ready(
    workers: &[Arc<dyn Worker>],
    mode: PdPairingMode,
    overload_shed: bool,
) -> bool {
    let workers: Vec<_> = workers
        .iter()
        .filter(|w| {
            w.is_available() || (!overload_shed && overload::vetoed_by_overload_alone(w.as_ref()))
        })
        .filter_map(|w| capability(w.as_ref()).map(|c| (w, c)))
        .collect();
    let available = |pool: RoutingPool| {
        workers
            .iter()
            .filter(move |(w, _)| pool.matches(w.as_ref()))
    };
    available(RoutingPool::GrpcEncode).any(|(e, ec)| {
        available(RoutingPool::GrpcPrefill).any(|(p, pc)| {
            available(RoutingPool::GrpcDecode).any(|(d, dc)| {
                compatible_pair(e.as_ref(), ec, p.as_ref(), pc, d.as_ref(), dc)
                    && p.pd_pairing().mismatch(d.pd_pairing(), mode).is_none()
            })
        })
    })
}

/// Validated media references, standard OpenAI image detail options, routing digest,
/// and handoff deadline captured ahead of preparation for one SGLang EPD request.
#[derive(Clone)]
pub(crate) struct PreparedInput {
    refs: vllm::MediaRefs,
    details: Vec<Option<ImageDetail>>,
    digest: Vec<u8>,
    deadline: Instant,
}

/// Only media enters prepared-input; text keeps ordinary Generate semantics.
pub(crate) fn prepare(ctx: &RequestContext) -> Result<Option<PreparedInput>, PrepareError> {
    let (plan, unsupported_flags) = match &ctx.input.request_type {
        RequestType::Chat(body) => (
            media_plan_chat(&body.messages),
            body.session_params.is_some() || body.return_hidden_states,
        ),
        RequestType::Messages(body) => (media_plan_messages(&body.messages), false),
        _ => return Ok(None),
    };
    if plan.is_empty() {
        return Ok(None);
    }
    let mut details = Vec::with_capacity(plan.parts().len());
    for part in plan.parts() {
        match part {
            MediaContentPart::ImageUrl { detail, .. }
            | MediaContentPart::ImageBase64 { detail, .. } => details.push(*detail),
            MediaContentPart::AudioUrl { .. }
            | MediaContentPart::AudioBase64 { .. }
            | MediaContentPart::AudioData { .. } => {
                return Err(PrepareError::Capability(Status::invalid_argument(
                    "backend sglang does not support audio inputs",
                )))
            }
            _ => details.push(None),
        }
    }
    let max_items = Limits::default().images;
    if unsupported_flags || details.len() > max_items {
        return Err(PrepareError::Capability(Status::invalid_argument(format!(
            "prepared-input requires at most {max_items} media items without session params or hidden states"
        ))));
    }
    check_forwarded_references(&plan).map_err(PrepareError::MediaRefs)?;
    let refs = assemble_media_refs(&plan).map_err(PrepareError::MediaRefs)?;
    let mut total_bytes: usize = 0;
    for item in &refs.items {
        total_bytes = total_bytes.saturating_add(item.url.len());
    }
    if total_bytes > MAX_MEDIA_REF_BYTES {
        return Err(PrepareError::Capability(Status::invalid_argument(
            "managed media references exceed the 255 MiB control payload limit",
        )));
    }
    // Hash media references and detail options into a single prompt routing key.
    let mut hasher = Sha256::new();
    for (item, detail) in refs.items.iter().zip(&details) {
        hasher.update(item.modality.to_le_bytes());
        hasher.update((item.url.len() as u64).to_le_bytes());
        hasher.update(item.url.as_bytes());
        hasher.update([match detail {
            None | Some(ImageDetail::Auto) => 0,
            Some(ImageDetail::Low) => 1,
            Some(ImageDetail::High) => 2,
        }]);
    }
    Ok(Some(PreparedInput {
        refs,
        details,
        digest: hasher.finalize().to_vec(),
        deadline: Instant::now() + PREPARED_TIMEOUT,
    }))
}

impl PreparedInput {
    /// Whole-prompt assignment preserves occurrence order and native batching.
    pub(crate) fn hashes(&self) -> Vec<Vec<u8>> {
        vec![self.digest.clone()]
    }

    /// Freeze the workers already chosen by the common EPD policies and populate prepared metadata.
    pub(crate) fn build(
        self,
        generation: &mut sglang_proto::GenerateRequest,
        workers: &WorkerSelection,
    ) -> Result<NativeEncode, Status> {
        let prefill = workers
            .prefill_worker()
            .ok_or_else(|| Status::internal("missing P"))?;
        let decode = workers
            .decode_worker()
            .ok_or_else(|| Status::internal("missing D"))?;
        let encoder = &workers
            .encode_assignments()
            .and_then(|a| a.first())
            .ok_or_else(|| Status::unavailable("missing E"))?
            .worker;
        let p = capability(prefill.as_ref())
            .ok_or_else(|| Status::unavailable("missing P capability"))?;
        let d = capability(decode.as_ref())
            .ok_or_else(|| Status::unavailable("missing D capability"))?;
        let e = capability(encoder.as_ref())
            .ok_or_else(|| Status::unavailable("missing E capability"))?;
        if !compatible_pair(
            encoder.as_ref(),
            &e,
            prefill.as_ref(),
            &p,
            decode.as_ref(),
            &d,
        ) {
            return Err(Status::failed_precondition(
                "selected prepared-input contracts differ",
            ));
        }
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| Status::deadline_exceeded("prepared request expired"))?;
        let attempt = Uuid::now_v7().to_string();
        let params = generation
            .disaggregated_params
            .as_ref()
            .ok_or_else(|| Status::internal("missing shared PD rendezvous"))?;
        let prefill_rid = format!("{attempt}-p");
        let decode_rid = format!("{attempt}-d");
        let binding = proto::InputBinding {
            prefill_rid: prefill_rid.clone(),
            decode_rid: decode_rid.clone(),
            bootstrap: Some(proto::Bootstrap {
                host: params.bootstrap_host.clone(),
                port: params.bootstrap_port as u32,
                room: i64::from(params.bootstrap_room),
            }),
            model_revision: p.model_revision,
            processor_contract: p.processor_contract,
            layout_contract: LAYOUT.into(),
        };
        let (mut image_urls, mut image_details, mut video_urls) =
            (Vec::new(), Vec::new(), Vec::new());
        for (item, detail) in self.refs.items.into_iter().zip(self.details) {
            if item.modality == common::Modality::Video as i32 {
                video_urls.push(item.url);
            } else {
                image_urls.push(item.url);
                image_details.push(detail);
            }
        }
        let image_count = image_urls.len() as u32;
        let video_count = video_urls.len() as u32;
        let num_parts = u32::from(!image_urls.is_empty()) + u32::from(!video_urls.is_empty());
        let mut requests = Vec::with_capacity(num_parts as usize);
        let push_part = |requests: &mut Vec<EncodeRequest>,
                         modality: &str,
                         mm_items: Vec<String>,
                         item_options: Vec<ImageOption>| {
            let part_idx = requests.len() as u32;
            requests.push(EncodeRequest {
                req_id: format!("{prefill_rid}_local_part_{part_idx}"),
                mm_items,
                num_parts,
                part_idx,
                prefill_host: p.bootstrap_host.clone(),
                embedding_port: vec![],
                item_options,
                modality: modality.into(),
            });
        };
        if !image_urls.is_empty() {
            let has_options = image_details
                .iter()
                .any(|detail| matches!(detail, Some(ImageDetail::Low | ImageDetail::High)));
            let item_options = if has_options {
                image_details
                    .iter()
                    .map(|detail| ImageOption {
                        detail: match detail {
                            Some(ImageDetail::Low) => "low",
                            Some(ImageDetail::High) => "high",
                            Some(ImageDetail::Auto) => "auto",
                            None => "",
                        }
                        .into(),
                    })
                    .collect()
            } else {
                Vec::new()
            };
            push_part(&mut requests, "image", image_urls, item_options);
        }
        if !video_urls.is_empty() {
            push_part(&mut requests, "video", video_urls, Vec::new());
        }
        tracing::debug!(
            logical_request_id = %generation.request_id,
            prefill_rid = %prefill_rid,
            decode_rid = %decode_rid,
            num_parts,
            bootstrap_room = params.bootstrap_room,
            "Prepared EPD plan request mapping"
        );
        generation.prepared_control = proto::PrefillStart {
            version: VERSION,
            timeout_ms: remaining.as_millis() as u64,
            binding: Some(binding),
            encoder_receive_endpoint: e.receive_endpoint,
            image_count,
            video_count,
        }
        .encode_to_vec();
        generation.mm_inputs = None;
        generation.extra_mm_inputs.clear();
        Ok(NativeEncode {
            worker: Arc::clone(encoder),
            requests,
            timeout: remaining,
            // Hold a load guard for the full lifetime of the encode RPC so load-aware
            // encoder policies observe in-flight encode concurrency in real time.
            _load_guard: WorkerLoadGuard::with_key(Arc::clone(encoder), None),
        })
    }
}

/// One prompt is assigned to one native encoder, using its cached worker connection.
pub(crate) struct NativeEncode {
    worker: Arc<dyn Worker>,
    requests: Vec<EncodeRequest>,
    timeout: Duration,
    _load_guard: WorkerLoadGuard,
}

impl NativeEncode {
    /// Send all modality encode parts concurrently to the selected SGLang encoder worker.
    pub(crate) async fn dispatch(self) -> Result<(), String> {
        let backend = get_backend_client(self.worker.as_ref())
            .await
            .map_err(|e| e.to_string())?
            .ok_or("encoder connection unavailable")?;
        let BackendClient::Grpc(GrpcClient::Sglang(client)) = backend.as_ref() else {
            return Err("encoder requires SGLang gRPC".into());
        };
        try_join_all(
            self.requests
                .into_iter()
                .map(|request| client.encode_native(request, self.timeout)),
        )
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
    }
}

/// Split one frozen control into P and D requests before either RPC starts.
pub(crate) fn split_requests(
    p: &mut sglang_proto::GenerateRequest,
    d: &mut sglang_proto::GenerateRequest,
) -> Result<Option<Duration>, Status> {
    if p.prepared_control.is_empty() {
        return Ok(None);
    }
    let control = proto::PrefillStart::decode(p.prepared_control.as_slice())
        .map_err(|_| Status::invalid_argument("invalid prepared control"))?;
    let binding = control
        .binding
        .ok_or_else(|| Status::invalid_argument("missing prepared binding"))?;
    if control.timeout_ms == 0 {
        return Err(Status::deadline_exceeded("prepared request expired"));
    }
    p.request_id.clone_from(&binding.prefill_rid);
    if let Some(params) = &mut p.sampling_params {
        params.max_new_tokens = Some(1);
        params.min_new_tokens = 0;
    }
    d.request_id.clone_from(&binding.decode_rid);
    if let Some(tokens) = &mut d.tokenized {
        tokens.input_ids.clear();
        tokens.original_text.clear();
    }
    Ok(Some(Duration::from_millis(control.timeout_ms)))
}

/// Recompute D's remaining handoff budget after waiting for P's layout export.
pub(crate) fn refresh_decode_timeout(
    decode: &mut sglang_proto::GenerateRequest,
    deadline: tokio::time::Instant,
) -> Result<(), Status> {
    let remaining_ms = deadline
        .checked_duration_since(tokio::time::Instant::now())
        .map(|d| d.as_millis() as u64)
        .filter(|&ms| ms > 0)
        .ok_or_else(|| Status::deadline_exceeded("prepared request expired"))?;
    let mut control = proto::PrefillStart::decode(decode.prepared_control.as_slice())
        .map_err(|_| Status::invalid_argument("invalid prepared control"))?;
    control.timeout_ms = remaining_ms;
    decode.prepared_control = control.encode_to_vec();
    Ok(())
}

/// Receive P's prepared layout bytes before `deadline` and forward them to D.
pub(crate) async fn relay_prefill_layout(
    deadline: tokio::time::Instant,
    first: impl Future<Output = Option<Result<ProtoGenerateResponse, Status>>>,
    tx: oneshot::Sender<Vec<u8>>,
) -> Result<(), Status> {
    let item = tokio::time::timeout_at(deadline, first)
        .await
        .map_err(|_| Status::deadline_exceeded("prepared layout expired"))?;
    let input = match item {
        Some(Ok(ProtoGenerateResponse::Sglang(response)))
            if response.response.is_none() && !response.prepared_input.is_empty() =>
        {
            response.prepared_input
        }
        Some(Err(error)) => return Err(error),
        _ => return Err(Status::data_loss("P did not export prepared input")),
    };
    let _ = tx.send(input);
    Ok(())
}

/// Wait for P's exported layout before `deadline`, validate its size, and attach it to D.
pub(crate) async fn attach_decode_layout(
    deadline: tokio::time::Instant,
    rx: oneshot::Receiver<Vec<u8>>,
    decode: &mut sglang_proto::GenerateRequest,
) -> Result<(), Status> {
    let bytes = tokio::time::timeout_at(deadline, rx)
        .await
        .map_err(|_| Status::deadline_exceeded("prepared layout expired"))?
        .map_err(|_| Status::cancelled("P layout export failed"))?;
    if bytes.len() > Limits::default().manifest_bytes {
        return Err(Status::resource_exhausted("prepared input exceeds limit"));
    }
    refresh_decode_timeout(decode, deadline)?;
    decode.prepared_input = bytes;
    Ok(())
}

/// Build a sample managed EPD capability descriptor for unit tests.
#[cfg(test)]
pub(crate) fn test_capability(role: WorkerType, tp_size: u32) -> proto::Capabilities {
    proto::Capabilities {
        version: VERSION,
        role: match role {
            WorkerType::Encode => "encoder",
            WorkerType::Prefill => "prefill",
            WorkerType::Decode => "decode",
            WorkerType::Regular => panic!("expected an EPD role"),
        }
        .into(),
        model_revision: "weights".into(),
        processor_contract: "qwen3-vl".into(),
        layout_contract: LAYOUT.into(),
        embedding_dtype: "bfloat16".into(),
        kv_backend: "mooncake".into(),
        embedding_backend: "mooncake".into(),
        topology_contract: format!("single-host-tp{tp_size}"),
        tp_size,
        ..Default::default()
    }
}

/// Construct minimal `SharedComponents` for prepared-input unit tests.
#[cfg(test)]
pub(crate) fn test_components() -> Arc<super::context::SharedComponents> {
    use llm_tokenizer::TokenizerRegistry;

    use crate::{
        config::ParserChoice,
        routers::{
            common::snapshot::WorkerRegistry,
            grpc::{
                context::SharedComponents, regular::symphony::template::Templates,
                utils::ParserResolver,
            },
        },
    };
    let tokenizers = Arc::new(TokenizerRegistry::new());
    Arc::new(SharedComponents {
        templates: Arc::new(Templates::new(Arc::clone(&tokenizers))),
        tokenizer_registry: tokenizers,
        worker_registry: Arc::new(WorkerRegistry::new()),
        tool_parser_factory: Default::default(),
        reasoning_parser_factory: Default::default(),
        parser_resolver: ParserResolver::disabled(),
        multimodal: None,
        tokenizer_autoload: true,
        tokenizer_load_failures: Arc::default(),
        parser: ParserChoice::Legacy,
    })
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use openai_protocol::worker::WorkerStatus;
    use smg_worker::{
        identity::basic::{ConnectionMode, RuntimeType},
        test_support::no_health_check,
    };

    use super::*;
    use crate::routers::{
        common::snapshot::WorkerRegistry,
        error::extract_error_code_from_response,
        grpc::{backend_client::BasicWorkerBuilder, context::EncodeWorkerAssignment},
    };

    fn worker_with_contract(
        role: &str,
        runtime: RuntimeType,
        update: impl FnOnce(&mut proto::Capabilities),
    ) -> Arc<dyn Worker> {
        let worker_type = match role {
            "encoder" => WorkerType::Encode,
            "prefill" => WorkerType::Prefill,
            _ => WorkerType::Decode,
        };
        let mut cap = test_capability(worker_type, 1);
        cap.bootstrap_host = "10.0.0.8".into();
        cap.bootstrap_port = 8998;
        cap.receive_endpoint = "10.0.0.9:5000".into();
        cap.ready = true;
        update(&mut cap);
        Arc::new(
            BasicWorkerBuilder::new(format!("grpc://{role}.service:30000"))
                .runtime_type(runtime)
                .connection_mode(ConnectionMode::Grpc)
                .worker_type(worker_type)
                .health_config(no_health_check())
                .label(CAPABILITY, serde_json::to_string(&cap).unwrap())
                .build(),
        )
    }

    fn ready_workers(make: impl Fn(&str) -> Arc<dyn Worker>) -> [Arc<dyn Worker>; 3] {
        let registry = WorkerRegistry::new();
        ["encoder", "prefill", "decode"].map(|role| {
            let worker = make(role);
            let id = registry.register(Arc::clone(&worker)).unwrap();
            registry.transition_status(&id, WorkerStatus::Ready);
            worker
        })
    }

    fn selection([encoder, prefill, decode]: [Arc<dyn Worker>; 3]) -> WorkerSelection {
        WorkerSelection::Disaggregated {
            encode_assignments: Some(vec![EncodeWorkerAssignment {
                item_index: 0,
                worker: encoder,
            }]),
            prefill,
            decode: decode.into(),
            runtime_type: RuntimeType::Sglang,
        }
    }

    fn base_gen_request() -> sglang_proto::GenerateRequest {
        sglang_proto::GenerateRequest {
            request_id: "client".into(),
            disaggregated_params: Some(sglang_proto::DisaggregatedParams {
                bootstrap_host: "prefill.service".into(),
                bootstrap_port: 8998,
                bootstrap_room: 17,
            }),
            ..Default::default()
        }
    }

    fn prepared() -> PreparedInput {
        PreparedInput {
            refs: vllm::MediaRefs {
                items: vec![vllm::MediaRef {
                    modality: common::Modality::Image as i32,
                    url: "data:image/png;base64,AAAA".into(),
                }],
            },
            details: vec![None],
            digest: vec![1; 32],
            deadline: Instant::now() + Duration::from_secs(30),
        }
    }

    #[test]
    fn pipeline_ready_and_build_enforce_contract_and_pairing_compatibility() {
        let ok = ready_workers(|role| worker_with_contract(role, RuntimeType::Sglang, |_| {}));
        assert!(pipeline_ready(&ok, PdPairingMode::Lenient, false));
        assert!(pipeline_ready(&ok, PdPairingMode::Off, false));

        // Overloaded encoders remain ready when `--worker-overload-shed` is off (steering-only),
        // matching selection's overload fallback, and report not-ready when shedding is enabled.
        let overloaded_registry = WorkerRegistry::new();
        for w in &ok {
            overloaded_registry.register(Arc::clone(w)).unwrap();
        }
        overloaded_registry.set_worker_overloaded(&ok[0], true);
        assert!(pipeline_ready(&ok, PdPairingMode::Lenient, false));
        assert!(!pipeline_ready(&ok, PdPairingMode::Lenient, true));

        let mismatched_kv = ready_workers(|role| {
            worker_with_contract(role, RuntimeType::Sglang, |cap| {
                if role == "decode" {
                    cap.kv_backend = "nixl".into();
                }
            })
        });
        assert!(!pipeline_ready(&mismatched_kv, PdPairingMode::Off, false));
        assert_eq!(
            prepared()
                .build(&mut base_gen_request(), &selection(mismatched_kv))
                .err()
                .unwrap()
                .code(),
            tonic::Code::FailedPrecondition
        );

        let mismatched_embedding = ready_workers(|role| {
            worker_with_contract(role, RuntimeType::Sglang, |cap| {
                if role == "prefill" {
                    cap.embedding_backend = "zmq_to_scheduler".into();
                }
            })
        });
        assert!(!pipeline_ready(
            &mismatched_embedding,
            PdPairingMode::Lenient,
            false
        ));
    }

    #[test]
    fn split_requests_preserves_decode_min_tokens_and_refreshes_timeout() {
        let workers = selection(ready_workers(|role| {
            worker_with_contract(role, RuntimeType::Sglang, |_| {})
        }));
        let mut request = base_gen_request();
        request.sampling_params = Some(sglang_proto::SamplingParams {
            max_new_tokens: Some(64),
            min_new_tokens: 30,
            ..Default::default()
        });
        prepared().build(&mut request, &workers).unwrap();
        let mut prefill = request.clone();
        let mut decode = request;
        assert!(split_requests(&mut prefill, &mut decode).unwrap().is_some());
        let p_params = prefill.sampling_params.as_ref().unwrap();
        assert_eq!(
            (p_params.max_new_tokens, p_params.min_new_tokens),
            (Some(1), 0)
        );
        let d_params = decode.sampling_params.as_ref().unwrap();
        assert_eq!(
            (d_params.max_new_tokens, d_params.min_new_tokens),
            (Some(64), 30)
        );

        let short_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        refresh_decode_timeout(&mut decode, short_deadline).unwrap();
        let refreshed = proto::PrefillStart::decode(decode.prepared_control.as_slice()).unwrap();
        assert!(refreshed.timeout_ms > 0 && refreshed.timeout_ms <= 5_000);
        let expired = refresh_decode_timeout(&mut decode, tokio::time::Instant::now())
            .expect_err("expired deadline must be rejected");
        assert_eq!(expired.code(), tonic::Code::DeadlineExceeded);

        // Ordinary PD requests have empty prepared_control and bypass splitting untouched.
        let mut plain_p = base_gen_request();
        let mut plain_d = base_gen_request();
        assert!(split_requests(&mut plain_p, &mut plain_d)
            .unwrap()
            .is_none());
        assert_eq!(plain_p.request_id, "client");
        assert_eq!(plain_d.request_id, "client");
    }

    #[test]
    fn native_encode_holds_encoder_load_guard_until_dropped() {
        let triplet = ready_workers(|role| worker_with_contract(role, RuntimeType::Sglang, |_| {}));
        let encoder = Arc::clone(&triplet[0]);
        assert_eq!(encoder.load(), 0);
        let plan = prepared()
            .build(&mut base_gen_request(), &selection(triplet))
            .unwrap();
        assert_eq!(encoder.load(), 1);
        drop(plan);
        assert_eq!(encoder.load(), 0);
    }

    fn prepare_chat_json(value: serde_json::Value) -> Result<Option<PreparedInput>, PrepareError> {
        let req = Arc::new(serde_json::from_value(value).unwrap());
        let ctx = RequestContext::for_chat(req, None, "qwen3-vl".into(), test_components());
        prepare(&ctx)
    }

    fn prepare_messages_json(
        value: serde_json::Value,
    ) -> Result<Option<PreparedInput>, PrepareError> {
        let req = Arc::new(serde_json::from_value(value).unwrap());
        let ctx = RequestContext::for_messages_carrying_its_tools(
            req,
            None,
            "qwen3-vl".into(),
            test_components(),
        );
        prepare(&ctx)
    }

    #[test]
    fn prepare_supports_detail_tools_video_and_messages() {
        let plain = prepare_chat_json(serde_json::json!({
            "model": "qwen3-vl",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
            ]}]
        }))
        .unwrap()
        .unwrap();
        let detailed = prepare_chat_json(serde_json::json!({
            "model": "qwen3-vl",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": "https://example.test/b.png", "detail": "low"}}
            ]}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        }))
        .unwrap()
        .unwrap();
        assert_ne!(plain.hashes(), detailed.hashes());

        let workers = selection(ready_workers(|role| {
            worker_with_contract(role, RuntimeType::Sglang, |_| {})
        }));
        let plain_encode = plain.build(&mut base_gen_request(), &workers).unwrap();
        assert!(plain_encode.requests[0].item_options.is_empty());
        assert_eq!(plain_encode.requests[0].modality, "image");
        let encoded = detailed
            .build(&mut base_gen_request(), &workers)
            .unwrap()
            .requests
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(
            encoded
                .item_options
                .iter()
                .map(|o| o.detail.as_str())
                .collect::<Vec<_>>(),
            vec!["high", "low"]
        );

        let msg_prepared = prepare_messages_json(serde_json::json!({
            "model": "qwen3-vl",
            "max_tokens": 32,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AQ=="}},
                    {"type": "tool_result", "tool_use_id": "t1", "content": [
                        {"type": "image", "source": {"type": "url", "url": "https://example.test/tool.png"}}
                    ]}
                ]
            }]
        }))
        .unwrap()
        .unwrap();
        let msg_req = msg_prepared
            .build(&mut base_gen_request(), &workers)
            .unwrap()
            .requests
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(
            msg_req.mm_items,
            vec![
                "data:image/png;base64,AQ==",
                "https://example.test/tool.png"
            ]
        );

        // Text-only requests (even with n > 1) bypass prepared-input completely.
        assert!(prepare_chat_json(serde_json::json!({
            "model": "qwen3-vl",
            "n": 2,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
        .is_none());

        let multi_n = prepare_chat_json(serde_json::json!({
            "model": "qwen3-vl",
            "n": 3,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
            ]}]
        }))
        .unwrap()
        .unwrap();
        assert_eq!(multi_n.refs.items.len(), 1);

        let sixty_four: Vec<_> = (0..64)
            .map(|i| serde_json::json!({"type": "image_url", "image_url": {"url": format!("https://example.test/{i}.png")}}))
            .collect();
        let many_images = prepare_chat_json(serde_json::json!({
            "model": "qwen3-vl",
            "messages": [{"role": "user", "content": sixty_four}]
        }))
        .unwrap()
        .unwrap();
        assert_eq!(many_images.refs.items.len(), 64);

        let video_only = prepare_chat_json(serde_json::json!({
            "model": "qwen3-vl",
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4"}}
            ]}]
        }))
        .unwrap()
        .unwrap();
        let mut video_gen = base_gen_request();
        let video_enc = video_only.build(&mut video_gen, &workers).unwrap();
        let video_ctrl =
            proto::PrefillStart::decode(video_gen.prepared_control.as_slice()).unwrap();
        assert_eq!((video_ctrl.image_count, video_ctrl.video_count), (0, 1));
        assert_eq!(video_enc.requests.len(), 1);
        assert_eq!(
            (
                video_enc.requests[0].modality.as_str(),
                video_enc.requests[0].num_parts,
                video_enc.requests[0].part_idx
            ),
            ("video", 1, 0)
        );

        let mixed = prepare_chat_json(serde_json::json!({
            "model": "qwen3-vl",
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4"}},
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png", "detail": "high"}}
            ]}]
        }))
        .unwrap()
        .unwrap();
        let mut mixed_gen = base_gen_request();
        let mixed_enc = mixed.build(&mut mixed_gen, &workers).unwrap();
        let mixed_ctrl =
            proto::PrefillStart::decode(mixed_gen.prepared_control.as_slice()).unwrap();
        assert_eq!((mixed_ctrl.image_count, mixed_ctrl.video_count), (1, 1));
        assert_eq!(mixed_enc.requests.len(), 2);
        assert_eq!(
            (
                mixed_enc.requests[0].modality.as_str(),
                mixed_enc.requests[0].part_idx,
                mixed_enc.requests[0].num_parts,
                mixed_enc.requests[0].item_options.len()
            ),
            ("image", 0, 2, 1)
        );
        assert_eq!(
            (
                mixed_enc.requests[1].modality.as_str(),
                mixed_enc.requests[1].part_idx,
                mixed_enc.requests[1].num_parts,
                mixed_enc.requests[1].item_options.len()
            ),
            ("video", 1, 2, 0)
        );

        for (bad, expect_capability) in [
            (
                serde_json::json!({"model": "qwen3-vl", "return_hidden_states": true, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]}]}),
                true,
            ),
            (
                serde_json::json!({"model": "qwen3-vl", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": " https://example.test/a.png"}}]}]}),
                false,
            ),
            (
                serde_json::json!({"model": "qwen3-vl", "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "https://example.test/a.mp4", "fps": 2.0}}]}]}),
                false,
            ),
            (
                serde_json::json!({"model": "qwen3-vl", "messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "https://example.test/a.wav"}}]}]}),
                true,
            ),
        ] {
            let err = prepare_chat_json(bad).err().unwrap();
            let resp = err.into_response();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let code = extract_error_code_from_response(&resp);
            if expect_capability {
                assert_eq!(code, "sglang_epd_capability");
            } else {
                assert_ne!(code, "sglang_epd_capability");
            }
        }
    }
}
