use async_trait::async_trait;
use axum::{
    body::to_bytes,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    model_card::ModelCard, worker::WorkerStatus,
};
use serde_json::{json, Value};
use smg_worker::identity::basic::WorkerType;

use super::*;
use crate::{
    middleware::{RouteRequestMeta, TenantKey, TenantRequestMeta},
    routers::grpc::backend_client::BasicWorkerBuilder,
};

fn worker(
    registry: &WorkerRegistry,
    port: u16,
    model: &str,
    role: WorkerType,
    connection: ConnectionMode,
    runtime: RuntimeType,
) {
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new(format!("http://127.0.0.1:{port}"))
                .model(ModelCard::new(model).with_alias(format!("{model}-alias")))
                .worker_type(role)
                .connection_mode(connection)
                .runtime_type(runtime)
                .build(),
        ))
        .unwrap();
}

fn registry() -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());
    worker(
        &registry,
        9601,
        "base",
        WorkerType::Regular,
        ConnectionMode::Grpc,
        RuntimeType::Sglang,
    );
    registry
}

#[test]
fn model_suffix_selects_a_preloaded_adapter_and_preserves_exact_names() {
    let registry = registry();
    for base in ["base", "base-alias"] {
        let mut model = format!("{base}:adapter");
        let mut path = None;
        assert_eq!(
            normalize_model(&registry, &mut model, &mut path).unwrap(),
            Some(base.to_string())
        );
        assert_eq!(model, base);
        assert_eq!(path.as_deref(), Some("adapter"));
    }
    worker(
        &registry,
        9602,
        "base:revision",
        WorkerType::Regular,
        ConnectionMode::Grpc,
        RuntimeType::Sglang,
    );
    for literal in [
        "base:revision",
        "base:revision-alias",
        "unknown:provider:model",
    ] {
        let mut model = literal.to_string();
        let mut path = None;
        assert!(normalize_model(&registry, &mut model, &mut path)
            .unwrap()
            .is_none());
        assert_eq!(model, literal);
        assert!(path.is_none());
    }
    let mut model = "base:revision:adapter".to_string();
    let mut path = None;
    assert_eq!(
        normalize_model(&registry, &mut model, &mut path).unwrap(),
        Some("base:revision".to_string())
    );
}

#[test]
fn conflicting_or_empty_adapter_names_are_refused_before_routing() {
    let registry = registry();
    for (model, path) in [
        ("base:", None),
        ("base:  ", None),
        ("base:adapter", Some("other")),
    ] {
        assert_eq!(
            normalize_model(
                &registry,
                &mut model.to_string(),
                &mut path.map(str::to_string)
            )
            .unwrap_err()
            .status(),
            StatusCode::BAD_REQUEST
        );
    }
    let mut model = "base:adapter".to_string();
    let mut path = Some("adapter".to_string());
    assert!(normalize_model(&registry, &mut model, &mut path)
        .unwrap()
        .is_some());
}

#[test]
fn non_sglang_and_http_model_ids_are_not_reinterpreted() {
    for (connection, runtime) in [
        (ConnectionMode::Grpc, RuntimeType::Vllm),
        (ConnectionMode::Http, RuntimeType::Sglang),
    ] {
        let registry = WorkerRegistry::new();
        worker(
            &registry,
            9603,
            "base",
            WorkerType::Regular,
            connection,
            runtime,
        );
        let mut model = "base:adapter".to_string();
        let mut path = None;
        assert!(normalize_model(&registry, &mut model, &mut path)
            .unwrap()
            .is_none());
        assert_eq!(model, "base:adapter");
        assert!(path.is_none());
    }
}

#[derive(Debug)]
struct Capture(&'static str);

#[async_trait]
impl RouterTrait for Capture {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn router_type(&self) -> &'static str {
        self.0
    }
    async fn route_generate(
        &self,
        _: Option<&HeaderMap>,
        _: &TenantRequestMeta,
        body: GenerateRequest,
        model: &str,
    ) -> Response {
        Json(json!({"family": self.0, "model": model, "body_model": body.model, "lora_path": body.lora_path.or(body.lora_id), "stream": body.stream})).into_response()
    }
    async fn route_chat(
        &self,
        _: Option<&HeaderMap>,
        _: &TenantRequestMeta,
        body: Arc<ChatCompletionRequest>,
        model: &str,
    ) -> Response {
        Json(json!({"family": self.0, "model": model, "body_model": body.model, "lora_path": body.lora_path, "stream": body.stream})).into_response()
    }
    async fn route_completion(
        &self,
        _: Option<&HeaderMap>,
        _: &TenantRequestMeta,
        body: CompletionRequest,
        model: &str,
    ) -> Response {
        Json(json!({"family": self.0, "model": model, "body_model": body.model, "lora_path": body.lora_path, "stream": body.stream})).into_response()
    }
}

#[tokio::test]
async fn openai_chat_and_completion_suffixes_route_by_base_across_grpc_families() {
    for igw in [false, true] {
        for (id, label, roles) in [
            (router_ids::GRPC_REGULAR, "grpc", vec![WorkerType::Regular]),
            (
                router_ids::GRPC_PD,
                "grpc_pd",
                vec![WorkerType::Prefill, WorkerType::Decode],
            ),
            (
                router_ids::GRPC_EPD,
                "grpc_epd",
                vec![WorkerType::Encode, WorkerType::Prefill, WorkerType::Decode],
            ),
        ] {
            let registry = Arc::new(WorkerRegistry::new());
            for (index, role) in roles.into_iter().enumerate() {
                worker(
                    &registry,
                    9610 + index as u16,
                    "base",
                    role,
                    ConnectionMode::Grpc,
                    RuntimeType::Sglang,
                );
            }
            worker(
                &registry,
                9620,
                "base",
                WorkerType::Regular,
                ConnectionMode::Http,
                RuntimeType::Sglang,
            );
            let mut gateway = Gateway::new(registry);
            gateway.enable_igw = igw;
            gateway.register_router(router_ids::HTTP_REGULAR, Arc::new(Capture("http")));
            gateway.register_router(id.clone(), Arc::new(Capture(label)));
            if !igw {
                gateway.set_default_router(id);
            }
            let meta = RouteRequestMeta::new(TenantKey::from("test"));
            for stream in [false, true] {
                for (model, path) in [
                    ("base-alias:adapter", None),
                    ("base-alias", Some("adapter")),
                ] {
                    let chat = serde_json::from_value(
                    json!({"model": model, "messages": [], "stream": stream, "lora_path": path}),
                )
                .unwrap();
                    let completion = serde_json::from_value(
                    json!({"model": model, "prompt": "hi", "stream": stream, "lora_path": path}),
                )
                .unwrap();
                    for response in [
                        gateway.route_chat(None, &meta, Arc::new(chat), model).await,
                        gateway
                            .route_completion(None, &meta, completion, model)
                            .await,
                    ] {
                        assert_eq!(response.status(), StatusCode::OK);
                        let response: Value = serde_json::from_slice(
                            &to_bytes(response.into_body(), usize::MAX).await.unwrap(),
                        )
                        .unwrap();
                        assert_eq!(
                            response,
                            json!({"family": label, "model": "base-alias", "body_model": "base-alias", "lora_path": "adapter", "stream": stream})
                        );
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn native_selectors_pin_mixed_fleets_and_preserve_http_only_routing() {
    for grpc in [true, false] {
        let registry = if grpc {
            registry()
        } else {
            Arc::new(WorkerRegistry::new())
        };
        worker(
            &registry,
            9630,
            "base",
            WorkerType::Regular,
            ConnectionMode::Http,
            RuntimeType::Sglang,
        );
        let mut gateway = Gateway::new(registry);
        gateway.enable_igw = true;
        gateway.register_router(router_ids::HTTP_REGULAR, Arc::new(Capture("http")));
        if grpc {
            gateway.register_router(router_ids::GRPC_REGULAR, Arc::new(Capture("grpc")));
        }
        let meta = RouteRequestMeta::new(TenantKey::from("test"));
        for field in ["lora_path", "lora_id"] {
            let mut request = json!({"model": "base", "text": "hi"});
            request[field] = json!("adapter");
            let response = gateway
                .route_generate(
                    None,
                    &meta,
                    serde_json::from_value(request).unwrap(),
                    "base",
                )
                .await;
            let response: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                    .unwrap();
            assert_eq!(response["family"], if grpc { "grpc" } else { "http" });
            assert_eq!(response["lora_path"], "adapter");
        }
    }
}

/// An adapter request goes only to a family whose SGLang workers can serve.
/// The regular family's SGLang worker is down, so the PD family gets every
/// request. The ready vLLM worker gives no weight, because adapter requests
/// do not go to it.
#[test]
fn an_adapter_request_goes_to_the_family_that_can_serve_it() {
    use RuntimeType::{Sglang, Vllm};
    use WorkerStatus::{NotReady, Ready};
    let registry = Arc::new(WorkerRegistry::new());
    for (port, role, runtime, status) in [
        (9640, WorkerType::Regular, Sglang, NotReady),
        (9641, WorkerType::Regular, Vllm, Ready),
        (9642, WorkerType::Prefill, Sglang, Ready),
        (9643, WorkerType::Decode, Sglang, Ready),
    ] {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(format!("http://127.0.0.1:{port}"))
                    .model(ModelCard::new("base"))
                    .worker_type(role)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(runtime)
                    .status(status)
                    .build(),
            ))
            .unwrap();
    }
    let mut gateway = Gateway::new(registry);
    gateway.enable_igw = true;
    gateway.register_router(router_ids::GRPC_REGULAR, Arc::new(Capture("grpc")));
    gateway.register_router(router_ids::GRPC_PD, Arc::new(Capture("grpc_pd")));

    for draw in 0..100 {
        let router = gateway
            .select_router_for_adapter("base", Some("adapter"), EpdDemand::Pair)
            .expect("the pair serves the adapter");
        assert_eq!(router.router_type(), "grpc_pd", "draw {draw}");
    }
}

/// The adapter draw sees the request's media. It counts only SGLang
/// workers, and SGLang has no encode role, so the EPD router cannot use the
/// ready encode worker. An adapter request with media goes to the regular
/// family, and the EPD family still takes text.
#[test]
fn an_adapter_request_with_media_needs_an_encode_worker_the_epd_router_can_use() {
    let registry = Arc::new(WorkerRegistry::new());
    for (port, role) in [
        (9650, WorkerType::Regular),
        (9651, WorkerType::Encode),
        (9652, WorkerType::Prefill),
        (9653, WorkerType::Decode),
    ] {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(format!("http://127.0.0.1:{port}"))
                    .model(ModelCard::new("base"))
                    .worker_type(role)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Sglang)
                    .status(WorkerStatus::Ready)
                    .build(),
            ))
            .unwrap();
    }
    let mut gateway = Gateway::new(registry);
    gateway.enable_igw = true;
    gateway.register_router(router_ids::GRPC_REGULAR, Arc::new(Capture("grpc")));
    gateway.register_router(router_ids::GRPC_EPD, Arc::new(Capture("grpc_epd")));

    let draw = |epd| {
        gateway
            .select_router_for_adapter("base", Some("adapter"), epd)
            .expect("a family serves the adapter")
            .router_type()
    };
    for draw_index in 0..100 {
        assert_eq!(
            draw(EpdDemand::PairAndEncoder { vision: true }),
            "grpc",
            "draw {draw_index}"
        );
    }
    let text: Vec<_> = (0..100).map(|_| draw(EpdDemand::Pair)).collect();
    assert!(
        text.contains(&"grpc_epd"),
        "the EPD family took no text: {text:?}"
    );
}

/// Every family of the adapter draw is a gRPC family. A generate request with
/// media still goes to one of them, which refuses it with its own `400`, as
/// before.
#[test]
fn an_adapter_generate_request_with_media_still_goes_to_a_grpc_family() {
    let registry = Arc::new(WorkerRegistry::new());
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("http://127.0.0.1:9660")
                .model(ModelCard::new("base"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Sglang)
                .status(WorkerStatus::Ready)
                .build(),
        ))
        .unwrap();
    let mut gateway = Gateway::new(registry);
    gateway.enable_igw = true;
    gateway.register_router(router_ids::GRPC_REGULAR, Arc::new(Capture("grpc")));
    let router = gateway
        .select_router_for_adapter("base", Some("adapter"), EpdDemand::GrpcRefused)
        .expect("the gRPC family answers the refusal");
    assert_eq!(router.router_type(), "grpc");
}
