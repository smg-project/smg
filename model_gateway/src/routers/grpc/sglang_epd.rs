//! SGLang metadata adaptation; membership, placement and admission belong to SMG.
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use llm_multimodal::{media::MAX_LONG_SIDE_PIXEL_FACTOR, ImageDetail, MediaContentPart};
use prost::Message;
use sha2::{Digest, Sha256};
use smg_grpc_client::{
    sglang_encoder_proto::{EncodeRequest, ImageOption},
    sglang_managed::{proto, Limits, LAYOUT, VERSION},
    sglang_proto,
};
use smg_worker::identity::basic::{Worker, WorkerType};
use tonic::Status;
use uuid::Uuid;

use crate::{
    config::types::PdPairingMode,
    routers::{
        common::snapshot::RoutingPool,
        grpc::{
            backend_client::{get_backend_client, BackendClient},
            client::GrpcClient,
            context::{RequestContext, RequestType, WorkerSelection},
            multimodal::{media_plan_chat, media_plan_messages, MediaPlan},
        },
    },
};

pub const CAPABILITY: &str = "sglang_prepared";

/// Bounded handoff budget for E encode, P layout export, and D admission; once D is
/// submitted, total decode duration (streaming or buffered) follows `--grpc-request-timeout-secs`.
const PREPARED_TIMEOUT: Duration = Duration::from_secs(120);

/// Tonic URIs use HTTP schemes; SMG registration uses a plain `grpc://` hint.
pub(crate) fn worker_registration_url(url: &str) -> String {
    if let Some(address) = url.strip_prefix("http://") {
        format!("grpc://{address}")
    } else {
        url.to_owned()
    }
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

/// Check protocol, revision, topology, dtype, and embedding backend compatibility between E and P/D.
fn compatible_contracts(
    encoder: &dyn Worker,
    e: &proto::Capabilities,
    worker: &dyn Worker,
    p: &proto::Capabilities,
) -> bool {
    encoder.metadata().spec.runtime_type == worker.metadata().spec.runtime_type
        && e.version == VERSION
        && p.version == VERSION
        && [e, p].iter().all(|c| {
            matches!(c.tp_size, 1 | 2 | 4 | 8)
                && c.topology_contract == format!("single-host-tp{}", c.tp_size)
        })
        && e.role == "encoder"
        && match worker.worker_type() {
            WorkerType::Prefill => p.role == "prefill",
            WorkerType::Decode => p.role == "decode",
            _ => false,
        }
        && e.model_revision == p.model_revision
        && e.processor_contract == p.processor_contract
        && e.layout_contract == LAYOUT
        && p.layout_contract == LAYOUT
        && e.embedding_dtype == p.embedding_dtype
        && (p.role == "decode" || e.embedding_backend == p.embedding_backend)
}

/// Check full E/P/D contract compatibility including shared KV transfer backend between P and D.
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

/// Use the same contracts and pairing rules for readiness and request placement.
pub(crate) fn pipeline_ready(workers: &[Arc<dyn Worker>], mode: PdPairingMode) -> bool {
    let workers: Vec<_> = workers
        .iter()
        .filter(|w| w.is_available())
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedImage {
    url: String,
    detail: Option<ImageDetail>,
    max_long_side_pixel: Option<u32>,
}

pub(crate) struct PreparedInput {
    images: Vec<PreparedImage>,
    digest: Vec<u8>,
    deadline: Instant,
}

/// Only media enters prepared-input; text keeps ordinary Generate semantics.
pub(crate) fn prepare(ctx: &RequestContext) -> Result<Option<PreparedInput>, Status> {
    let (plan, unsupported_flags) = match &ctx.input.request_type {
        RequestType::Chat(body) => (
            media_plan_chat(&body.messages),
            body.n.unwrap_or(1) != 1 || body.session_params.is_some() || body.return_hidden_states,
        ),
        RequestType::Messages(body) => (media_plan_messages(&body.messages), false),
        _ => return Ok(None),
    };
    if plan.is_empty() {
        return Ok(None);
    }
    let images = prepared_images_from_plan(plan)?;
    if unsupported_flags || images.len() > Limits::default().images {
        return Err(Status::invalid_argument(
            "prepared-input v1 requires at most 32 images, n=1, and no session/hidden states",
        ));
    }
    const MAX_IMAGE_REF_BYTES: usize = 63 * 1024 * 1024;
    let total_bytes: usize = images.iter().map(|img| img.url.len()).sum();
    if total_bytes > MAX_IMAGE_REF_BYTES {
        return Err(Status::invalid_argument(
            "managed image references exceed the 64 MiB encoder control budget",
        ));
    }
    let mut hasher = Sha256::new();
    for image in &images {
        hasher.update((image.url.len() as u64).to_le_bytes());
        hasher.update(image.url.as_bytes());
        hasher.update([match image.detail {
            None | Some(ImageDetail::Auto) => 0,
            Some(ImageDetail::Low) => 1,
            Some(ImageDetail::High) => 2,
        }]);
        hasher.update(image.max_long_side_pixel.unwrap_or(0).to_le_bytes());
    }
    Ok(Some(PreparedInput {
        images,
        digest: hasher.finalize().to_vec(),
        deadline: Instant::now() + PREPARED_TIMEOUT,
    }))
}

fn validate_long_side_cap(cap: Option<u32>) -> Result<(), Status> {
    if let Some(value) = cap {
        if value == 0 || !value.is_multiple_of(MAX_LONG_SIDE_PIXEL_FACTOR) {
            return Err(Status::invalid_argument(format!(
                "managed image max_long_side_pixel ({value}) must be a positive multiple of {MAX_LONG_SIDE_PIXEL_FACTOR}"
            )));
        }
    }
    Ok(())
}

/// Extract occurrence order without decoding, fetching, or examining image bytes.
fn prepared_images_from_plan(plan: MediaPlan) -> Result<Vec<PreparedImage>, Status> {
    plan.into_parts()
        .into_iter()
        .map(|part| match part {
            MediaContentPart::ImageUrl {
                url,
                detail,
                max_long_side_pixel,
                ..
            } => {
                if url.is_empty() {
                    return Err(Status::invalid_argument("managed image URL is missing"));
                }
                validate_long_side_cap(max_long_side_pixel)?;
                Ok(PreparedImage {
                    url: url.as_ref().clone(),
                    detail,
                    max_long_side_pixel,
                })
            }
            MediaContentPart::ImageBase64 {
                media_type,
                data,
                detail,
                max_long_side_pixel,
                ..
            } => {
                if data.is_empty() {
                    return Err(Status::invalid_argument(
                        "managed image base64 payload is missing",
                    ));
                }
                validate_long_side_cap(max_long_side_pixel)?;
                Ok(PreparedImage {
                    url: format!("data:{media_type};base64,{data}"),
                    detail,
                    max_long_side_pixel,
                })
            }
            _ => Err(Status::invalid_argument(
                "managed EPD supports text and images only",
            )),
        })
        .collect()
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
        let encoder_req_id = format!("{prefill_rid}_local_part_0");
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
        let has_options = self.images.iter().any(|img| {
            img.max_long_side_pixel.is_some()
                || matches!(img.detail, Some(ImageDetail::Low | ImageDetail::High))
        });
        let item_options = if has_options {
            self.images
                .iter()
                .map(|img| ImageOption {
                    detail: match img.detail {
                        Some(ImageDetail::Low) => "low",
                        Some(ImageDetail::High) => "high",
                        Some(ImageDetail::Auto) => "auto",
                        None => "",
                    }
                    .into(),
                    max_long_side_pixel: img.max_long_side_pixel.unwrap_or(0),
                })
                .collect()
        } else {
            Vec::new()
        };
        let request = EncodeRequest {
            req_id: encoder_req_id.clone(),
            mm_items: self.images.iter().map(|img| img.url.clone()).collect(),
            num_parts: 1,
            part_idx: 0,
            prefill_host: p.bootstrap_host.clone(),
            embedding_port: vec![],
            item_options,
        };
        tracing::debug!(
            logical_request_id = %generation.request_id,
            prefill_rid = %prefill_rid,
            decode_rid = %decode_rid,
            encoder_req_id = %encoder_req_id,
            bootstrap_room = params.bootstrap_room,
            "Prepared EPD plan request mapping"
        );
        generation.prepared_control = proto::PrefillStart {
            version: VERSION,
            timeout_ms: remaining.as_millis() as u64,
            binding: Some(binding),
            encoder_receive_endpoint: e.receive_endpoint,
            image_count: self.images.len() as u32,
        }
        .encode_to_vec();
        generation.mm_inputs = None;
        generation.extra_mm_inputs.clear();
        Ok(NativeEncode {
            worker: Arc::clone(encoder),
            request,
            timeout: remaining,
        })
    }
}

/// One prompt is assigned to one native encoder, using its cached worker connection.
pub(crate) struct NativeEncode {
    worker: Arc<dyn Worker>,
    request: EncodeRequest,
    timeout: Duration,
}

impl NativeEncode {
    /// Send the bounded `Encode` RPC to the selected native encoder worker.
    pub(crate) async fn dispatch(self) -> Result<(), String> {
        let backend = get_backend_client(self.worker.as_ref())
            .await
            .map_err(|e| e.to_string())?
            .ok_or("encoder connection unavailable")?;
        let BackendClient::Grpc(GrpcClient::Sglang(client)) = backend.as_ref() else {
            return Err("encoder requires SGLang gRPC".into());
        };
        client
            .encode_native(self.request, self.timeout)
            .await
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

#[cfg(test)]
mod tests {
    use openai_protocol::worker::WorkerStatus;
    use smg_worker::{
        identity::basic::{ConnectionMode, RuntimeType, WorkerType},
        test_support::no_health_check,
    };

    use super::*;
    use crate::routers::{
        common::snapshot::WorkerRegistry,
        grpc::{
            backend_client::BasicWorkerBuilder,
            context::{EncodeWorkerAssignment, SharedComponents},
        },
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
            decode,
            runtime_type: RuntimeType::Sglang,
        }
    }

    fn prepared() -> PreparedInput {
        PreparedInput {
            images: vec![PreparedImage {
                url: "data:image/png;base64,AAAA".into(),
                detail: None,
                max_long_side_pixel: None,
            }],
            digest: vec![1; 32],
            deadline: Instant::now() + PREPARED_TIMEOUT,
        }
    }

    #[test]
    fn build_and_readiness_enforce_contracts_and_hosts() {
        let workers = selection(ready_workers(|role| {
            worker_with_contract(role, RuntimeType::Sglang, |_| {})
        }));
        let mut request = sglang_proto::GenerateRequest {
            request_id: "client".into(),
            disaggregated_params: Some(sglang_proto::DisaggregatedParams {
                bootstrap_host: "prefill.service".into(),
                bootstrap_port: 8998,
                bootstrap_room: 17,
            }),
            ..Default::default()
        };
        let plan = prepared().build(&mut request, &workers).unwrap();
        let control = proto::PrefillStart::decode(request.prepared_control.as_slice()).unwrap();
        assert_eq!(
            control.binding.unwrap().bootstrap.unwrap().host,
            "prefill.service"
        );
        assert_eq!(plan.request.prefill_host, "10.0.0.8");

        for broken in ["prefill-role", "decode-role", "prefill-kv", "decode-kv"] {
            let workers = ready_workers(|role| {
                worker_with_contract(role, RuntimeType::Sglang, |c| {
                    if broken == format!("{role}-role") {
                        c.role = "encoder".into();
                    }
                    if broken == format!("{role}-kv") {
                        c.kv_backend = "other".into();
                    }
                })
            });
            assert!(!pipeline_ready(&workers, PdPairingMode::Off), "{broken}");
            let error = prepared()
                .build(
                    &mut sglang_proto::GenerateRequest::default(),
                    &selection(workers),
                )
                .err()
                .unwrap();
            assert_eq!(error.code(), tonic::Code::FailedPrecondition, "{broken}");
        }
    }

    #[test]
    fn split_requests_resets_prefill_min_new_tokens_and_preserves_decode_params() {
        let workers = selection(ready_workers(|role| {
            worker_with_contract(role, RuntimeType::Sglang, |_| {})
        }));
        let mut request = sglang_proto::GenerateRequest {
            request_id: "client".into(),
            sampling_params: Some(sglang_proto::SamplingParams {
                min_new_tokens: 30,
                max_new_tokens: Some(64),
                ..Default::default()
            }),
            disaggregated_params: Some(sglang_proto::DisaggregatedParams {
                bootstrap_host: "prefill.service".into(),
                bootstrap_port: 8998,
                bootstrap_room: 17,
            }),
            ..Default::default()
        };
        prepared().build(&mut request, &workers).unwrap();
        let mut prefill = request.clone();
        let mut decode = request;
        assert!(split_requests(&mut prefill, &mut decode).unwrap().is_some());
        let p_params = prefill.sampling_params.as_ref().unwrap();
        assert_eq!(p_params.max_new_tokens, Some(1));
        assert_eq!(p_params.min_new_tokens, 0);
        let d_params = decode.sampling_params.as_ref().unwrap();
        assert_eq!(d_params.max_new_tokens, Some(64));
        assert_eq!(d_params.min_new_tokens, 30);

        let short_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        refresh_decode_timeout(&mut decode, short_deadline).unwrap();
        let refreshed = proto::PrefillStart::decode(decode.prepared_control.as_slice()).unwrap();
        assert!(refreshed.timeout_ms > 0 && refreshed.timeout_ms <= 5_000);
        let expired = refresh_decode_timeout(&mut decode, tokio::time::Instant::now())
            .expect_err("expired deadline must be rejected");
        assert_eq!(expired.code(), tonic::Code::DeadlineExceeded);
    }

    fn test_components() -> Arc<SharedComponents> {
        use llm_tokenizer::TokenizerRegistry;

        use crate::{
            config::ParserChoice,
            routers::{
                common::snapshot::WorkerRegistry,
                grpc::{regular::symphony::template::Templates, utils::ParserResolver},
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

    fn prepare_chat_json(value: serde_json::Value) -> Result<Option<PreparedInput>, Status> {
        let req = Arc::new(serde_json::from_value(value).unwrap());
        let ctx = RequestContext::for_chat(req, None, "glm53_flash".into(), test_components());
        prepare(&ctx)
    }

    fn prepare_messages_json(value: serde_json::Value) -> Result<Option<PreparedInput>, Status> {
        let req = Arc::new(serde_json::from_value(value).unwrap());
        let ctx = RequestContext::for_messages_carrying_its_tools(
            req,
            None,
            "glm53_flash".into(),
            test_components(),
        );
        prepare(&ctx)
    }

    #[test]
    fn prepare_supports_detail_resize_tools_and_messages() {
        let plain = prepare_chat_json(serde_json::json!({
            "model": "glm53_flash",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
            ]}]
        }))
        .unwrap()
        .unwrap();
        let detailed = prepare_chat_json(serde_json::json!({
            "model": "glm53_flash",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png", "detail": "high", "max_long_side_pixel": 504}},
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
        let base_req = || sglang_proto::GenerateRequest {
            request_id: "client".into(),
            disaggregated_params: Some(sglang_proto::DisaggregatedParams {
                bootstrap_host: "prefill.service".into(),
                bootstrap_port: 8998,
                bootstrap_room: 17,
            }),
            ..Default::default()
        };
        assert!(plain
            .build(&mut base_req(), &workers)
            .unwrap()
            .request
            .item_options
            .is_empty());
        let encoded = detailed.build(&mut base_req(), &workers).unwrap().request;
        assert_eq!(
            encoded.mm_items,
            vec!["https://example.test/a.png", "https://example.test/b.png"]
        );
        assert_eq!(
            encoded.item_options,
            vec![
                ImageOption {
                    detail: "high".into(),
                    max_long_side_pixel: 504
                },
                ImageOption {
                    detail: "low".into(),
                    max_long_side_pixel: 0
                },
            ]
        );

        let msgs = prepare_messages_json(serde_json::json!({
            "model": "glm53_flash",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AQ=="}},
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.test/tool.png"}}
                ]}
            ]}],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
        }))
        .unwrap()
        .unwrap();
        let msg_req = msgs.build(&mut base_req(), &workers).unwrap().request;
        assert_eq!(
            msg_req.mm_items,
            vec![
                "data:image/png;base64,AQ==",
                "https://example.test/tool.png"
            ]
        );

        // Text-only stays un-prepared even when n > 1.
        assert!(prepare_chat_json(serde_json::json!({
            "model": "glm53_flash",
            "n": 2,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
        .is_none());

        for bad in [
            serde_json::json!({"model": "glm53_flash", "n": 2, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]}]}),
            serde_json::json!({"model": "glm53_flash", "return_hidden_states": true, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]}]}),
            serde_json::json!({"model": "glm53_flash", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/a.png", "max_long_side_pixel": 0}}]}]}),
            serde_json::json!({"model": "glm53_flash", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/a.png", "max_long_side_pixel": 500}}]}]}),
            serde_json::json!({"model": "glm53_flash", "messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "https://example.test/a.wav"}}]}]}),
        ] {
            let err = prepare_chat_json(bad).err().unwrap();
            assert_eq!(err.code(), tonic::Code::InvalidArgument);
        }
    }
}
