//! Store startup and routing under the policy controller.

use std::{
    error::Error,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::Response,
};
use openai_protocol::{
    model_card::ModelCard,
    transcription::{AudioFile, TranscriptionRequest},
    worker::WorkerStatus,
};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use smg_policy_controller::{
    document::{operator::SpecRejected, AppliedSpec, PolicySpec, RoutingSettingsError, SpecError},
    engine::{Candidate, RequestFacts, Role, Scorer, TieBand, WireModes, WorkerId},
    stages::{Catalog, CatalogError},
    store::{WorkerFacts, WorkerStatus as RowStatus, WorkersStore},
    tracking::Lifecycle,
};
use smg_worker::{connection::engine::EngineConnector, identity::basic::ConnectionMode};

use super::{
    operators::document_bytes, refused, start, start_with, undecided::ROUTER_TYPE, StartError,
};
use crate::{
    app_context::AppContext,
    config::{ManualAssignmentMode, PolicyConfig, RouterConfig, RoutingMode},
    middleware::{RouteRequestMeta, TenantKey},
    routers::{
        gateway::Gateway,
        grpc::backend_client::{BackendConnector, BasicWorkerBuilder},
        http::router::Router,
        RouterTrait,
    },
    server::ServerConfig,
};

type Case = Result<(), Box<dyn Error>>;

/// The refusal's code as a client reads it. A literal: the product's
/// constant may not move it in silence.
const BY_NAME: &str = "not_decided_by_policy_controller";

#[tokio::test]
async fn refused_names_its_shape_under_the_one_code() -> Case {
    let answer = refused("a test request");
    assert_eq!(answer.status(), StatusCode::NOT_IMPLEMENTED);
    let bytes = to_bytes(answer.into_body(), 1024).await?;
    let error: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(error["error"]["code"], BY_NAME);
    assert_eq!(
        error["error"]["message"],
        "The policy controller is picked for this gateway and does not decide a test request yet"
    );
    Ok(())
}

/// `config` with the policy controller picked.
fn picked(mut config: RouterConfig) -> RouterConfig {
    config.policy_controller = true;
    config
}

#[test]
fn the_classic_policies_start_no_store() -> Case {
    assert!(start(&RouterConfig::default())?.is_none());
    Ok(())
}

#[test]
fn the_store_starts_with_the_document_the_routing_configuration_means() -> Case {
    let config = picked(RouterConfig::default());
    let store = start(&config)?.ok_or("the controller was picked and no store started")?;
    let world = store.world();
    assert_eq!(world.spec().generation(), 1);
    assert_eq!(
        world.spec().digest(),
        Some(config.policy_document()?.digest()?)
    );
    Ok(())
}

#[test]
fn the_store_starts_on_the_given_document() -> Case {
    // A document that names no generation is the store's first.
    for (written, runs) in [(2, 2), (0, 1)] {
        let mut document = PolicySpec::preset("round_robin")?;
        document.generation = written;
        let digest = document.digest()?;
        let config = picked(RouterConfig {
            policy_spec: Some(document),
            ..RouterConfig::default()
        });
        let store = start(&config)?.ok_or("the controller was picked and no store started")?;
        let world = store.world();
        assert_eq!(world.spec().generation(), runs);
        assert_eq!(world.spec().digest(), Some(digest));
    }
    Ok(())
}

#[test]
fn a_start_whose_flags_and_document_disagree_is_refused_by_name() -> Case {
    let config = picked(RouterConfig {
        policy: PolicyConfig::RoundRobin,
        policy_spec: Some(PolicySpec::preset("random")?),
        ..RouterConfig::default()
    });
    let refusal = start(&config)
        .err()
        .ok_or("the start took a document that disagrees with the flags")?;
    assert!(matches!(refusal, StartError::Document(_)), "{refusal:?}");
    let message = refusal.to_string();
    assert!(
        message.contains("disagree at profiles.round_robin:"),
        "{message}"
    );
    Ok(())
}

#[test]
fn a_setting_no_document_says_stops_the_start_by_its_name() -> Case {
    let mut config = picked(RouterConfig::default());
    config.routing_key_override.enabled = true;
    for supplied in [None, Some(PolicySpec::preset("round_robin")?)] {
        config.policy_spec = supplied;
        let refusal = start(&config)
            .err()
            .ok_or("the start took a setting no document says")?;
        assert!(matches!(refusal, StartError::Document(_)), "{refusal:?}");
        assert_eq!(
            refusal.to_string(),
            "routing_key_override.enabled = true: the controller holds no recorded form of the sticky sessions' rule yet"
        );
    }
    Ok(())
}

#[test]
fn a_document_this_build_does_not_run_stops_the_start_by_its_name() -> Case {
    let mut policy = PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 1 << 20);
    if let PolicyConfig::CacheAware { overlap_decay, .. } = &mut policy {
        *overlap_decay = 0.5;
    }
    let mut config = picked(RouterConfig::default());
    config.policy = policy;
    let refusal = start(&config)
        .err()
        .ok_or("the start took a document this build does not run")?;
    assert!(matches!(refusal, StartError::Spec(_)), "{refusal:?}");
    assert_eq!(
        refusal.to_string(),
        "profiles.cache_aware.score[1].prefix-affinity: the prefix-affinity decay is not in this build"
    );
    Ok(())
}

#[test]
fn a_view_that_reads_the_gateways_own_header_stops_the_start_by_its_name() -> Case {
    // The tenant header is the gateway's own: the view of a document this
    // gateway hosts may not read it. The manual policy's view reads the
    // routing key headers.
    let mut config = picked(RouterConfig::default());
    config.policy = PolicyConfig::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: ManualAssignmentMode::Random,
    };
    let tenant_header = config.tenant_resolution.tenant_header_name.clone();
    config.routing_key_override.headers = vec![tenant_header.clone()];
    let refusal = start(&config)
        .err()
        .ok_or("the start took a view that reads the tenant header")?;
    assert!(matches!(refusal, StartError::Spec(_)), "{refusal:?}");
    assert_eq!(
        refusal.to_string(),
        format!("view.headers[0]: header {tenant_header:?} is on the deny-list")
    );
    Ok(())
}

/// A request body from its JSON.
fn body<T: DeserializeOwned>(json: Value) -> T {
    serde_json::from_value(json).expect("a request body")
}

/// The status of an answer and the code of its error object, empty where
/// the answer holds none.
async fn refusal(answer: Response) -> (StatusCode, String) {
    let status = answer.status();
    let bytes = to_bytes(answer.into_body(), usize::MAX)
        .await
        .expect("the answer's body");
    let error: Value = serde_json::from_slice(&bytes).unwrap_or_default();
    let code = error["error"]["code"].as_str().unwrap_or_default();
    (status, code.to_owned())
}

/// What each of `answers` read: its ask, its status and its error's code.
async fn read(answers: Vec<(&'static str, Response)>) -> Vec<(&'static str, StatusCode, String)> {
    let mut read = Vec::new();
    for (ask, answer) in answers {
        let (status, code) = refusal(answer).await;
        read.push((ask, status, code));
    }
    read
}

/// A chat request for `model`.
fn chat(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "hi"}]})
}

/// Every route of the router trait, asked of `router` for `model` in the
/// trait's order. The messages route is asked twice: its second request
/// asks for no token, which the gateway routes by a rule of its own.
async fn every_route(router: &dyn RouterTrait, model: &str) -> Vec<(&'static str, Response)> {
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let empty = || Request::new(Body::empty());
    let text = json!({"model": model, "input": "hi"});
    let message = |max_tokens: u32| {
        let mut request = chat(model);
        request["max_tokens"] = json!(max_tokens);
        request
    };
    let audio = AudioFile {
        bytes: bytes::Bytes::new(),
        file_name: "a.wav".to_owned(),
        content_type: None,
    };
    let decisions = json!({
        "model": model, "input": "evidence",
        "questions": [{"type": "predicate", "instructions": "Is it evidence?"}]
    });
    let systemone = json!({
        "model": model, "state": {"ticket": "evidence"},
        "questions": {"evidence": {
            "type": "noul", "criteria": {"true": "Has evidence", "false": "No evidence"}
        }}
    });
    let rerank = json!({"query": "q", "documents": ["d"], "model": model});
    vec![
        ("health_generate", router.health_generate(empty()).await),
        ("get_server_info", router.get_server_info(empty()).await),
        ("get_model_info", router.get_model_info(empty()).await),
        (
            "route_generate",
            router
                .route_generate(None, &meta, body(json!({"text": "hi"})), model)
                .await,
        ),
        (
            "route_chat",
            router
                .route_chat(None, &meta, Arc::new(body(chat(model))), model)
                .await,
        ),
        (
            "route_completion",
            router
                .route_completion(
                    None,
                    &meta,
                    body(json!({"model": model, "prompt": "hi"})),
                    model,
                )
                .await,
        ),
        (
            "route_responses",
            router
                .route_responses(None, &meta, body(text.clone()), model)
                .await,
        ),
        (
            "cancel_response",
            router.cancel_response(None, &meta, "resp_1").await,
        ),
        (
            "route_embeddings",
            router
                .route_embeddings(None, &meta, body(text.clone()), model)
                .await,
        ),
        (
            "route_classify",
            router.route_classify(None, &meta, body(text), model).await,
        ),
        (
            "route_decisions",
            router
                .route_decisions(None, &meta, body(decisions), model)
                .await,
        ),
        (
            "route_systemone",
            router
                .route_systemone(None, &meta, body(systemone), model)
                .await,
        ),
        (
            "route_audio_transcriptions",
            router
                .route_audio_transcriptions(
                    None,
                    &meta,
                    &TranscriptionRequest::default(),
                    audio,
                    model,
                )
                .await,
        ),
        (
            "route_rerank",
            router.route_rerank(None, &meta, body(rerank), model).await,
        ),
        (
            "route_messages",
            router
                .route_messages(None, &meta, Arc::new(body(message(1))), model)
                .await,
        ),
        (
            "route_messages, no token asked",
            router
                .route_messages(None, &meta, Arc::new(body(message(0))), model)
                .await,
        ),
        (
            "route_messages_count_tokens",
            router
                .route_messages_count_tokens(None, &meta, body(chat(model)), model)
                .await,
        ),
        (
            "route_interactions",
            router
                .route_interactions(None, &meta, body(json!({"input": "hi"})), Some(model))
                .await,
        ),
        (
            "route_realtime_session",
            router
                .route_realtime_session(None, &body(json!({"type": "realtime"})))
                .await,
        ),
        (
            "route_realtime_client_secret",
            router
                .route_realtime_client_secret(None, &body(json!({"session": {"type": "realtime"}})))
                .await,
        ),
        (
            "route_realtime_transcription_session",
            router
                .route_realtime_transcription_session(None, &body(json!({"type": "transcription"})))
                .await,
        ),
        (
            "route_realtime_ws",
            router.route_realtime_ws(empty(), model).await,
        ),
        (
            "route_realtime_webrtc",
            router.route_realtime_webrtc(empty(), model).await,
        ),
    ]
}

/// The adapter request shapes: the model names the adapter, or the
/// request's own field does.
async fn adapter_asks(router: &dyn RouterTrait) -> Vec<(&'static str, Response)> {
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let named = chat("base:adapter");
    let mut by_field = chat("base");
    by_field["lora_path"] = json!("adapter");
    let completion = json!({"model": "base:adapter", "prompt": "hi"});
    let generate = json!({"text": "hi", "lora_path": "adapter"});
    vec![
        (
            "chat, the model names the adapter",
            router
                .route_chat(None, &meta, Arc::new(body(named)), "base:adapter")
                .await,
        ),
        (
            "chat, the request's field names the adapter",
            router
                .route_chat(None, &meta, Arc::new(body(by_field)), "base")
                .await,
        ),
        (
            "completion, the model names the adapter",
            router
                .route_completion(None, &meta, body(completion), "base:adapter")
                .await,
        ),
        (
            "generate, the request's field names the adapter",
            router
                .route_generate(None, &meta, body(generate), "base")
                .await,
        ),
    ]
}

/// The three provider modes, each over one provider's worker.
fn provider_modes() -> [RoutingMode; 3] {
    let worker_urls = vec!["https://provider.example".to_owned()];
    [
        RoutingMode::OpenAI {
            worker_urls: worker_urls.clone(),
        },
        RoutingMode::Anthropic {
            worker_urls: worker_urls.clone(),
        },
        RoutingMode::Gemini { worker_urls },
    ]
}

#[test]
fn a_provider_mode_starts_no_store_under_the_setting() -> Case {
    // A provider mode places its requests by no routing policy: the setting
    // moves nothing, and the gateway starts as the classic line does.
    for mode in provider_modes() {
        let config = picked(RouterConfig {
            mode,
            ..RouterConfig::default()
        });
        assert!(
            start(&config)?.is_none(),
            "the {} mode started a store",
            config.mode_type()
        );
    }
    Ok(())
}

#[test]
fn a_document_beside_a_provider_mode_is_refused_by_name() -> Case {
    for mode in provider_modes() {
        let config = running(
            RouterConfig {
                mode,
                ..RouterConfig::default()
            },
            PolicySpec::preset("random")?,
        );
        let refusal = start(&config)
            .err()
            .ok_or("a document ran beside a provider mode")?;
        assert!(matches!(refusal, StartError::Document(_)), "{refusal}");
        assert!(
            refusal
                .to_string()
                .contains("a provider mode places its requests by no routing policy"),
            "{refusal}"
        );
    }
    Ok(())
}

#[test]
fn a_provider_mode_of_a_multi_router_gateway_starts_the_store() -> Case {
    // A multi-router gateway mounts the routers of self-hosted workers
    // beside the provider's. The setting stays in force for them, on the
    // document the routing settings mean under the regular mode of many
    // routers.
    for mode in provider_modes() {
        let config = picked(RouterConfig {
            mode,
            enable_igw: true,
            ..RouterConfig::default()
        });
        let store = start(&config)?.ok_or("a self-hosted request lost the controller")?;
        assert_eq!(
            store.world().spec().digest(),
            Some(
                picked(RouterConfig {
                    enable_igw: true,
                    ..RouterConfig::default()
                })
                .policy_document()?
                .digest()?
            ),
            "the {} mode",
            config.mode_type()
        );
    }
    Ok(())
}

#[test]
fn a_provider_mode_of_a_multi_router_gateway_runs_a_given_document() -> Case {
    // The compare of a given document with the routing settings reads the
    // multi-router setting too. Without it a provider mode has no document
    // to compare with, and the start would stop whatever was given.
    for mode in provider_modes() {
        let mut document = PolicySpec::preset("round_robin")?;
        document.generation = 2;
        let digest = document.digest()?;
        let config = running(
            RouterConfig {
                mode,
                enable_igw: true,
                ..RouterConfig::default()
            },
            document,
        );
        let store = start(&config)?.ok_or("the given document started no store")?;
        let world = store.world();
        assert_eq!(world.spec().generation(), 2);
        assert_eq!(
            world.spec().digest(),
            Some(digest),
            "the {} mode",
            config.mode_type()
        );
    }
    Ok(())
}

#[test]
fn a_controller_address_beside_a_provider_mode_stops_the_start() -> Case {
    // Nothing of such a gateway is placed through a controller. The
    // validator refuses the address by name. A context built over a
    // configuration that skipped it stops here: the address is never
    // dropped in silence.
    for mode in provider_modes() {
        let config = picked(RouterConfig {
            mode,
            policy_controller_address: Some("controller:30100".to_owned()),
            ..RouterConfig::default()
        });
        let refusal = start(&config)
            .err()
            .ok_or("the controller's address was dropped in silence")?;
        assert!(matches!(refusal, StartError::Document(_)), "{refusal}");
        assert!(
            refusal
                .to_string()
                .contains("a provider mode places its requests by no routing policy"),
            "{refusal}"
        );
    }
    Ok(())
}

/// A configuration with no health probe behind its cases.
fn quiet() -> RouterConfig {
    let mut config = RouterConfig::default();
    config.health_check.disable_health_check = true;
    config
}

async fn build_gateway(config: RouterConfig) -> Result<Arc<Gateway>, String> {
    Ok(Box::pin(build(config)).await?.1)
}

/// The context a start makes for `config`, and the gateway over it.
async fn build(config: RouterConfig) -> Result<(Arc<AppContext>, Arc<Gateway>), String> {
    let context = Box::pin(AppContext::from_config(config.clone(), 30, None, None)).await?;
    let context = Arc::new(context);
    let gateway = Box::pin(Gateway::from_config(
        &ServerConfig::minimal(config),
        &context,
    ))
    .await?;
    Ok((context, gateway))
}

/// The asks the controller decides.
const TYPED: [&str; 16] = [
    "route_generate",
    "route_chat",
    "route_completion",
    "route_responses",
    "route_embeddings",
    "route_classify",
    "route_decisions",
    "route_systemone",
    "route_audio_transcriptions",
    "route_rerank",
    "route_messages",
    "route_messages_count_tokens",
    "chat, the model names the adapter",
    "chat, the request's field names the adapter",
    "completion, the model names the adapter",
    "generate, the request's field names the adapter",
];

/// What `gateway` reads to every route and every adapter ask.
async fn every_answer(gateway: &Gateway) -> Vec<(&'static str, StatusCode, String)> {
    let mut answers = every_route(gateway, "m").await;
    answers.extend(adapter_asks(gateway).await);
    read(answers).await
}

/// With an empty registry, the asks the controller decides reach a model
/// miss. Every other route picks its worker by no routing policy and reads
/// as the classic line reads it: none is refused by the controller's name.
async fn decided_or_classic(config: RouterConfig) -> Case {
    let enable_igw = config.enable_igw;
    let mut classic = config.clone();
    classic.policy_controller = false;
    let gateway = build_gateway(config).await?;
    assert_eq!(gateway.router_count() > 1, enable_igw);
    let classic = build_gateway(classic).await?;

    let answers = Box::pin(every_answer(gateway.as_ref())).await;
    let classic = Box::pin(every_answer(classic.as_ref())).await;
    let (decided, other): (Vec<_>, Vec<_>) = answers
        .into_iter()
        .partition(|(ask, _, _)| TYPED.contains(ask));
    assert_eq!(
        decided,
        TYPED.map(|ask| (ask, StatusCode::NOT_FOUND, "model_not_found".to_owned())),
    );
    let classic: Vec<_> = classic
        .into_iter()
        .filter(|(ask, _, _)| !TYPED.contains(ask))
        .collect();
    assert_eq!(other.len(), 11);
    assert_eq!(other, classic);
    Ok(())
}

const SCORER: &str = "steady-score";

#[derive(Debug)]
struct SteadyScore;

impl Lifecycle for SteadyScore {}

impl Scorer for SteadyScore {
    fn name(&self) -> &'static str {
        SCORER
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand(0.0)
    }
}

fn host_stages() -> Result<Catalog, CatalogError> {
    let mut stages = Catalog::builtin();
    stages.scorer(SCORER, || Box::new(SteadyScore))?;
    Ok(stages)
}

fn scorer_document(config: &RouterConfig, name: &str) -> Result<PolicySpec, RoutingSettingsError> {
    let mut document = config.policy_document()?;
    let profile = document
        .profiles
        .get_mut("random")
        .expect("the random profile");
    profile.score = vec![name.into()];
    profile.pick = "best".into();
    Ok(document)
}

/// `config` with the policy controller picked and `document` given, as a
/// host gives one.
fn running(mut config: RouterConfig, document: PolicySpec) -> RouterConfig {
    config.policy_spec = Some(document);
    picked(config)
}

#[test]
fn the_store_starts_with_a_registered_scorer() -> Case {
    let document = scorer_document(&RouterConfig::default(), SCORER)?;
    let digest = document.digest()?;
    let generation = document.generation;
    let config = running(RouterConfig::default(), document);
    let started = start_with(&config, Some(host_stages()?))?
        .ok_or("the controller was picked and no store started")?;
    let world = started.store.world();
    assert_eq!(world.spec().generation(), generation);
    assert_eq!(world.spec().digest(), Some(digest));
    Ok(())
}

/// A hosted document passes the door's bound at the start: one whose YAML
/// is past it stops the start by its size. The print option answers by the
/// same function.
#[test]
fn a_document_past_the_doors_bound_stops_the_start_by_its_size() -> Case {
    let mut document = RouterConfig::default().policy_document()?;
    let stock = document.bindings[0].clone();
    let many = (0..4096).map(|n| {
        let mut binding = stock.clone();
        binding.matcher.model = format!("tenant-{n:04}/model");
        binding
    });
    document.bindings.splice(0..0, many);
    let config = running(RouterConfig::default(), document);
    let written = serde_yaml::to_string(&config.policy_document_in_force()?)?.len();
    let bound = document_bytes();
    assert!(written > bound, "{written}");
    let refusal = start_with(&config, None)
        .err()
        .ok_or("the start took a document past the door's bound")?;
    assert_eq!(
        refusal.to_string(),
        format!("the document is {written} bytes; at most {bound}")
    );
    Ok(())
}

/// A given document's `kind` line is no part of its YAML, so the door never
/// reads it: the start refuses it by name, by the door's own word.
#[test]
fn a_given_documents_kind_line_stops_the_start_by_name() -> Case {
    let document: PolicySpec = serde_yaml::from_str(
        "apiVersion: smg.policy/v1\nkind: PolicySpec\nprofiles:\n  preview:\n    score: [least-load]\n    pick: best\n",
    )?;
    let refusal = start_with(&running(RouterConfig::default(), document), None)
        .err()
        .ok_or("the start took a document with a kind line")?;
    assert!(
        matches!(
            &refusal,
            StartError::Spec(SpecRejected::Invalid(SpecError::Kind(kind))) if kind == "PolicySpec"
        ),
        "{refusal:?}"
    );
    Ok(())
}

#[test]
fn the_stock_store_names_the_unknown_scorer() -> Case {
    let document = scorer_document(&RouterConfig::default(), SCORER)?;
    let refusal = start_with(&running(RouterConfig::default(), document), None)
        .err()
        .ok_or("the stock start took a scorer it does not hold")?;
    assert!(matches!(
        refusal,
        StartError::Spec(SpecRejected::Invalid(SpecError::UnknownScorer { .. }))
    ));
    assert_eq!(
        refusal.to_string(),
        format!("profile \"random\": unknown scorer {SCORER:?}")
    );
    Ok(())
}

/// A host may build its context from a configuration it never validated.
/// A document handed beside the setting off stops the start in the
/// validator's words, and no catalog reads it: the classic policies do not
/// serve in its place in silence.
#[test]
fn a_document_beside_the_setting_off_stops_the_start() -> Case {
    let mut config = running(
        RouterConfig::default(),
        scorer_document(&RouterConfig::default(), "unknown-score")?,
    );
    assert!(matches!(
        start_with(&config, Some(host_stages()?)),
        Err(StartError::Spec(SpecRejected::Invalid(
            SpecError::UnknownScorer { .. }
        )))
    ));

    config.policy_controller = false;
    let refusal = start_with(&config, Some(host_stages()?))
        .err()
        .ok_or("the start dropped a given document")?;
    assert_eq!(
        refusal.to_string(),
        "Invalid value for field 'policy_spec': the given document - a policy document is run by the policy controller: set --policy-controller on"
    );
    Ok(())
}

/// The path is written with a configuration and the document is not. A
/// configuration that holds the path alone would run the routing settings'
/// document: the start refuses it in the validator's words.
#[test]
fn a_path_with_no_loaded_document_stops_the_start() -> Case {
    let config = picked(RouterConfig {
        policy_spec_path: Some("routing.yaml".to_owned()),
        ..RouterConfig::default()
    });
    let refusal = start(&config)
        .err()
        .ok_or("the start ran the routing settings' document")?;
    assert_eq!(
        refusal.to_string(),
        "Invalid value for field 'policy_spec': routing.yaml - the policy document of this path is not loaded: the builder reads it, or set policy_spec"
    );
    Ok(())
}

#[test]
fn the_gateways_tenant_header_is_denied_with_a_host_catalog() -> Case {
    let mut config = RouterConfig::default();
    let tenant_header = "x-account-scope";
    config.tenant_resolution.tenant_header_name = tenant_header.into();
    let mut document = scorer_document(&config, SCORER)?;
    document.view.headers.push(tenant_header.into());
    let stages = host_stages()?;
    AppliedSpec::validate_in(&document, &stages, &[])?;

    let refusal = start_with(&running(config, document), Some(stages))
        .err()
        .ok_or("the start took a view that reads the tenant header")?;
    assert!(matches!(
        refusal,
        StartError::Spec(SpecRejected::Invalid(SpecError::DeniedHeader { .. }))
    ));
    assert_eq!(
        refusal.to_string(),
        format!("view.headers[0]: header {tenant_header:?} is on the deny-list")
    );
    Ok(())
}

#[tokio::test]
async fn a_hosts_catalog_and_document_reach_the_store_through_the_context() -> Case {
    let document = scorer_document(&quiet(), SCORER)?;
    let digest = document.digest()?;
    let config = running(quiet(), document);

    let stages = Some(host_stages()?);
    let hosted =
        AppContext::from_config_with_extensions(config.clone(), 30, None, None, stages, None)
            .await?;
    let store = hosted
        .policy_store
        .as_ref()
        .ok_or("the controller was picked and no store started")?;
    assert_eq!(store.world().spec().digest(), Some(digest));

    // The stock start holds no stage of a host, and says so by name.
    let refusal = AppContext::from_config(config, 30, None, None).await.err();
    assert_eq!(
        refusal,
        Some(format!("profile \"random\": unknown scorer {SCORER:?}"))
    );
    Ok(())
}

#[tokio::test]
async fn under_the_setting_one_router_decides_typed_requests_and_leaves_the_rest_classic() -> Case {
    Box::pin(decided_or_classic(picked(quiet()))).await
}

#[tokio::test]
async fn under_the_setting_many_routers_decide_typed_requests_and_leave_the_rest_classic() -> Case {
    let mut config = picked(quiet());
    config.enable_igw = true;
    Box::pin(decided_or_classic(config)).await
}

/// The router asks the policy engine for its typed requests; the router's own
/// cases hold that.
#[tokio::test]
async fn the_regular_router_is_built_under_the_setting() -> Case {
    let gateway = build_gateway(picked(quiet())).await?;
    let router = gateway
        .select_router_for_request(None)
        .ok_or("the regular router is missing")?;
    assert_eq!(router.router_type(), "regular");
    Ok(())
}

/// The HTTP pair router asks the policy engine for each attempt's pair; the
/// router's own cases hold that. No classic pair policy is built beside it.
#[tokio::test]
async fn the_http_pair_router_is_built_under_the_setting() -> Case {
    let mut config = picked(quiet());
    config.mode = RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    let (context, gateway) = build(config).await?;
    let router = gateway
        .select_router_for_request(None)
        .ok_or("the HTTP pair router is missing")?;
    assert_eq!(router.router_type(), "pd");
    let policies = &context.policy_registry;
    let unset = policies.get_default_policy();
    assert!(Arc::ptr_eq(&policies.get_prefill_policy(), &unset));
    assert!(Arc::ptr_eq(&policies.get_decode_policy(), &unset));
    Ok(())
}

/// The HTTP pair router reads what its roles' policy settings keep. Under
/// the setting the registry holds no role policy, and its main policy here
/// reads no target, so a role's own setting alone refuses a request that
/// names its worker. Under settings that keep no target the request goes on
/// to the policy engine.
#[tokio::test]
async fn the_http_pair_router_refuses_an_explicit_target_a_roles_setting_keeps() -> Case {
    let mut named = HeaderMap::new();
    named.insert("x-smg-target-worker", "0".parse()?);
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let hashing = || Some(PolicyConfig::ConsistentHashing);
    for (prefill_policy, decode_policy, kept) in [
        (hashing(), None, true),
        (None, hashing(), true),
        (None, None, false),
    ] {
        let mut config = picked(quiet());
        config.policy = PolicyConfig::RoundRobin;
        config.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy,
            decode_policy,
        };
        let gateway = build_gateway(config).await?;
        let router = gateway
            .select_router_for_request(None)
            .ok_or("the HTTP pair router is missing")?;
        let routed = router
            .route_generate(Some(&named), &meta, body(json!({"text": "hi"})), "model-a")
            .await;
        let (status, code) = refusal(routed).await;
        assert_eq!(code == BY_NAME, kept, "{status} {code}");
    }
    Ok(())
}

/// A gateway of many routers mounts every family over its one document.
/// That document plans pairs under a regular configuration and under a
/// pair configuration alike, so the HTTP pair family serves under both.
/// Under the manual policy the document plans single workers alone, and
/// the family keeps its refusal by name: a pair asked there would be
/// decided as a single. Under a controller's address it refuses by name
/// though the document plans pairs: the pair is the controller's to decide
/// there.
#[tokio::test]
async fn many_routers_mount_the_http_pair_router_where_the_document_plans_pairs() -> Case {
    use crate::routers::{factory::router_ids, RouterFactory};

    let mut regular = picked(quiet());
    regular.enable_igw = true;
    let mut pair = regular.clone();
    pair.mode = RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    // A socket that never listens: the follower dials it and tells nothing.
    let socket = tokio::net::TcpSocket::new_v4()?;
    socket.bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))?;
    let mut followed = pair.clone();
    followed.policy_controller_address = Some(socket.local_addr()?.to_string());
    let mut manual = regular.clone();
    manual.policy = PolicyConfig::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: ManualAssignmentMode::Random,
    };
    for (config, serves) in [
        (regular, true),
        (pair, true),
        (manual, false),
        (followed, false),
    ] {
        let context = Arc::new(AppContext::from_config(config.clone(), 30, None, None).await?);
        let routers = RouterFactory::create_igw_routers(&config.policy, &context).await;
        let (_, _, built) = routers
            .into_iter()
            .find(|(id, ..)| *id == router_ids::HTTP_PD)
            .ok_or("the HTTP pair family has no entry")?;
        let router = built?;
        if serves {
            assert_eq!(router.router_type(), "pd");
        } else {
            assert_refusing_router(router.as_ref(), "HTTP pair, many routers").await?;
        }
        if let Some(follower) = &context.policy_follower {
            let follower = follower.lock().take().ok_or("the follower was taken")?;
            follower.stop().await;
        }
    }
    Ok(())
}

/// The router's selection stage asks the policy engine; the stage's cases hold
/// that.
#[tokio::test]
async fn the_regular_grpc_router_is_built_under_the_setting() -> Case {
    let mut config = picked(quiet());
    config.connection_mode = ConnectionMode::Grpc;
    let gateway = build_gateway(config).await?;
    let router = gateway
        .select_router_for_request(None)
        .ok_or("the regular gRPC router is missing")?;
    assert_eq!(router.router_type(), "grpc");
    Ok(())
}

/// The pair router's selection stage asks the policy engine; the stage's cases
/// and the pipeline's serving cases hold that. No classic pair policy is built
/// beside it.
#[tokio::test]
async fn the_grpc_pair_router_is_built_under_the_setting() -> Case {
    let mut config = picked(quiet());
    config.connection_mode = ConnectionMode::Grpc;
    config.mode = RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    let (context, gateway) = build(config).await?;
    let router = gateway
        .select_router_for_request(None)
        .ok_or("the gRPC pair router is missing")?;
    assert_eq!(router.router_type(), "grpc_pd");
    let policies = &context.policy_registry;
    let unset = policies.get_default_policy();
    assert!(Arc::ptr_eq(&policies.get_prefill_policy(), &unset));
    assert!(Arc::ptr_eq(&policies.get_decode_policy(), &unset));
    Ok(())
}

/// A gateway of many routers mounts every family over its one document.
/// That document plans pairs under a regular configuration and under a
/// pair configuration alike, so the gRPC pair family serves under both.
/// Under the manual policy the document plans single workers alone, and
/// the family keeps its refusal by name: a pair asked there would be
/// decided as a single.
#[tokio::test]
async fn many_routers_mount_the_grpc_pair_router_where_the_document_plans_pairs() -> Case {
    use crate::routers::{factory::router_ids, RouterFactory};

    let mut regular = picked(quiet());
    regular.enable_igw = true;
    let mut pair = regular.clone();
    pair.mode = RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    let mut manual = regular.clone();
    manual.policy = PolicyConfig::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: ManualAssignmentMode::Random,
    };
    for (config, serves) in [(regular, true), (pair, true), (manual, false)] {
        let context = Arc::new(AppContext::from_config(config.clone(), 30, None, None).await?);
        let routers = RouterFactory::create_igw_routers(&config.policy, &context).await;
        let (_, _, built) = routers
            .into_iter()
            .find(|(id, ..)| *id == router_ids::GRPC_PD)
            .ok_or("the gRPC pair family has no entry")?;
        let router = built?;
        if serves {
            assert_eq!(router.router_type(), "grpc_pd");
        } else {
            assert_refusing_router(router.as_ref(), "gRPC pair, many routers").await?;
        }
    }
    Ok(())
}

/// Under a controller's address a pair is the controller's to decide, and
/// no ask reaches it for a pair yet. The gRPC pair family refuses by name
/// there, as it did, though the gateway's document plans pairs.
#[tokio::test]
async fn a_grpc_pair_gateway_that_follows_a_controller_refuses_by_name() -> Case {
    let socket = tokio::net::TcpSocket::new_v4()?;
    socket.bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))?;
    let mut config = picked(quiet());
    config.connection_mode = ConnectionMode::Grpc;
    config.mode = RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    config.policy_controller_address = Some(socket.local_addr()?.to_string());
    let (context, gateway) = build(config).await?;
    assert!(context.follows_controller());
    let router = gateway
        .select_router_for_request(None)
        .ok_or("the router is missing")?;
    assert_refusing_router(router.as_ref(), "gRPC pair under an address").await
}

async fn assert_refusing_router(router: &dyn RouterTrait, family: &str) -> Case {
    assert_eq!(router.router_type(), ROUTER_TYPE, "{family}");
    let answer = router.health_generate(Request::new(Body::empty())).await;
    assert_eq!(answer.status(), StatusCode::NOT_IMPLEMENTED, "{family}");
    let bytes = to_bytes(answer.into_body(), 1024).await?;
    let error: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(error["error"]["code"], BY_NAME, "{family}");
    assert_eq!(
        error["error"]["message"],
        "The policy controller is picked for this gateway and does not decide requests for this router yet",
        "{family}"
    );
    Ok(())
}

#[tokio::test]
async fn the_pair_routers_refuse_by_name() -> Case {
    let mut pair = picked(quiet());
    pair.mode = RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    let mut grpc_pair = pair.clone();
    grpc_pair.connection_mode = ConnectionMode::Grpc;
    let mut encode_pair = grpc_pair.clone();
    encode_pair.mode = RoutingMode::EncodePrefillDecode {
        encode_urls: vec![],
        prefill_urls: vec![],
        decode_urls: vec![],
        encode_policy: None,
        prefill_policy: None,
        decode_policy: None,
    };
    for (family, config) in [("HTTP pair", pair), ("gRPC encode pair", encode_pair)] {
        let (context, gateway) = build(config).await?;
        let router = gateway
            .select_router_for_request(None)
            .ok_or("the router is missing")?;
        // The HTTP pair router asks the policy engine and refuses nothing. Its
        // own case holds what it is built as.
        if family == "HTTP pair" {
            assert_eq!(router.router_type(), "pd");
            continue;
        }
        assert_refusing_router(router.as_ref(), family).await?;
        // No classic pair policy was built for the refused family.
        let policies = &context.policy_registry;
        let unset = policies.get_default_policy();
        assert!(
            Arc::ptr_eq(&policies.get_prefill_policy(), &unset),
            "{family}"
        );
        assert!(
            Arc::ptr_eq(&policies.get_decode_policy(), &unset),
            "{family}"
        );
        // An encode policy nobody registered is made anew at each read. A
        // registered one is one object.
        assert!(
            !Arc::ptr_eq(&policies.get_encode_policy(), &policies.get_encode_policy()),
            "{family}"
        );
    }

    Ok(())
}

/// A provider's router picks its worker by no routing policy, so the
/// setting leaves its mount alone. A gateway of many routers and the
/// factory's own mount both give the router of the classic line.
#[cfg(feature = "provider-openai")]
#[tokio::test]
async fn a_providers_router_mounts_as_on_the_classic_line() -> Case {
    use smg_external_router::known::OPENAI;
    use smg_worker::identity::basic::RuntimeType;

    use crate::routers::RouterFactory;

    let model = "external-model";
    let mut mounts = Vec::new();
    for mut config in [quiet(), picked(quiet())] {
        config.enable_igw = true;
        let context = Arc::new(AppContext::from_config(config.clone(), 30, None, None).await?);
        let worker = BasicWorkerBuilder::new("http://127.0.0.1:1")
            .model(ModelCard::new(model))
            .runtime_type(RuntimeType::External)
            .status(WorkerStatus::Ready)
            .build();
        context
            .worker_registry
            .register(Arc::new(worker))
            .ok_or("the provider worker did not register")?;
        let gateway = Gateway::from_config(&ServerConfig::minimal(config), &context).await?;
        let routed = gateway
            .select_router_for_request(Some(model))
            .ok_or("the provider router is missing")?;
        let mounted = RouterFactory::create_external_router(OPENAI.backend, &context).await?;
        mounts.push((routed.router_type(), mounted.router_type()));
    }
    assert_eq!(mounts[0], mounts[1]);
    assert_ne!(mounts[1].0, ROUTER_TYPE);
    Ok(())
}

#[tokio::test]
async fn the_classic_policies_build_the_gateway_object() -> Case {
    let config = quiet();
    let context = Arc::new(AppContext::from_config(config.clone(), 30, None, None).await?);
    assert!(context.policy_store.is_none());
    let gateway = Gateway::from_config(&ServerConfig::minimal(config), &context).await?;
    let router = gateway
        .select_router_for_request(None)
        .ok_or("the regular router is missing")?;
    assert_eq!(router.router_type(), "regular");
    Ok(())
}

#[tokio::test]
async fn a_starts_context_holds_the_store_where_the_controller_was_picked() -> Case {
    let hosted = AppContext::from_config(picked(quiet()), 30, None, None).await?;
    let generation = hosted
        .policy_store
        .as_ref()
        .map(|store| store.world().spec().generation());
    assert_eq!(generation, Some(1));
    let mut unsaid = picked(quiet());
    unsaid.routing_key_override.enabled = true;
    let refusal = AppContext::from_config(unsaid, 30, None, None).await.err();
    assert_eq!(
        refusal.as_deref(),
        Some("routing_key_override.enabled = true: the controller holds no recorded form of the sticky sessions' rule yet")
    );
    Ok(())
}

#[tokio::test]
async fn a_starts_context_holds_the_policy_engine_where_it_holds_the_store() -> Case {
    let hosted = Arc::new(AppContext::from_config(picked(quiet()), 30, None, None).await?);
    assert!(hosted.policy_store.is_some());
    assert!(hosted.policy_engine.is_some());
    // A worker the registry holds and the store has no row for. The
    // controller answers for it; the classic policy would have sent to it.
    let worker = BasicWorkerBuilder::new("http://127.0.0.1:1")
        .model(ModelCard::new("model-a"))
        .status(WorkerStatus::Ready)
        .build();
    hosted
        .worker_registry
        .register(Arc::new(worker))
        .ok_or("the worker did not register")?;
    let router = Router::new(&hosted).await?;
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let routed = router
        .route_generate(None, &meta, body(json!({"text": "hi"})), "model-a")
        .await;
    assert_eq!(
        refusal(routed).await,
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "no_available_workers".to_owned()
        )
    );

    let classic = AppContext::from_config(quiet(), 30, None, None).await?;
    assert!(classic.policy_store.is_none());
    assert!(classic.policy_engine.is_none());
    Ok(())
}

#[tokio::test]
async fn a_starts_context_holds_the_feed_where_it_holds_the_store() -> Case {
    let hosted = AppContext::from_config(picked(quiet()), 30, None, None).await?;
    assert!(hosted.policy_store.is_some());
    let feed = hosted.policy_feed.as_ref().ok_or("the store has no feed")?;
    assert!(Arc::ptr_eq(
        feed.objects.registry(),
        &hosted.worker_registry
    ));

    let classic = AppContext::from_config(quiet(), 30, None, None).await?;
    assert!(classic.policy_store.is_none());
    assert!(classic.policy_feed.is_none());
    Ok(())
}

/// Under a controller's address the gateway probes no worker, so a row's
/// status is its worker's. Elsewhere a worker's status is the gateway's own
/// to write, and a row's change leaves it. On both paths the worker dials
/// through the gateway's wrap of the connector the gateway was started
/// with: the context's connector, which asks the handed one.
#[tokio::test]
async fn a_rows_status_is_its_workers_under_a_controllers_address_alone() -> Case {
    use smg_worker::workflow::WorkflowHost as _;

    const URL: &str = "http://worker:8000";
    let socket = tokio::net::TcpSocket::new_v4()?;
    socket.bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))?;
    let address = socket.local_addr()?.to_string();
    // A connector of the case's own: each path's worker holds the
    // gateway's wrap of the one handed over, and no stock one by name.
    let handed: Arc<dyn EngineConnector> = Arc::new(BackendConnector);
    let mut dials_through_it = Vec::new();
    for following in [false, true] {
        let mut config = picked(quiet());
        config.policy_controller_address = following.then(|| address.clone());
        let connector = Some(Arc::clone(&handed));
        let context =
            AppContext::from_config_with_extensions(config, 30, None, None, None, connector)
                .await?;
        assert_eq!(context.follows_controller(), following);
        let store = context.policy_store.as_ref().ok_or("no store started")?;
        let feed = context
            .policy_feed
            .as_ref()
            .ok_or("the store has no feed")?;

        // A row written past the door, as a watch writes it, and then its
        // promotion.
        let row = store
            .upsert(WorkerFacts {
                key: URL.to_owned(),
                engine: URL.to_owned(),
                role: Role::Single,
                wire_modes: WireModes::HTTP,
                models: vec!["model".to_owned()],
                dp_size: 1,
                ..WorkerFacts::default()
            })
            .id();
        feed.settle();
        let worker = context
            .worker_registry
            .get_by_url(URL)
            .ok_or("the row has no worker")?;
        let connector = worker
            .engine_connector()
            .ok_or("the worker has no connector")?;
        let wrap = context.engine_connector();
        assert_eq!(format!("{wrap:?}"), "Fronting(BackendConnector)");
        dials_through_it.push(Arc::ptr_eq(&connector, &wrap));
        assert!(!worker.is_healthy());
        let revision = store.get(row).ok_or("the row left")?.revision;
        store
            .set_status(row, RowStatus::Ready, revision)
            .ok_or("the status did not move")?;
        feed.settle();
        assert_eq!(worker.is_healthy(), following);

        if let Some(follower) = &context.policy_follower {
            let follower = follower.lock().take().ok_or("the follower was taken")?;
            follower.stop().await;
        }
    }
    assert_eq!(dials_through_it, [true, true]);
    Ok(())
}

/// A context that decides in its own process, with one worker written at
/// its feed's door under `url`: the context and the worker's row.
async fn context_with_a_worker(url: &str) -> Result<(AppContext, WorkerId), Box<dyn Error>> {
    use smg_policy_controller::{
        store::{HealthPolicy, Maker},
        worker_host::{Expect, Wrote},
    };

    let context = AppContext::from_config(picked(quiet()), 30, None, None).await?;
    let feed = context
        .policy_feed
        .as_ref()
        .ok_or("the store has no feed")?;
    let wrote = feed
        .door
        .found(&Maker::Control, Expect::Nothing, url, &|_| WorkerFacts {
            key: url.to_owned(),
            engine: url.to_owned(),
            role: Role::Single,
            wire_modes: WireModes::HTTP,
            models: vec!["model-a".to_owned()],
            dp_size: 1,
            max_running: Some(64),
            health: HealthPolicy {
                probes: false,
                ..HealthPolicy::default()
            },
            ..WorkerFacts::default()
        });
    let Ok(Wrote::Registered(row)) = wrote else {
        return Err(format!("expected a registration, got {wrote:?}").into());
    };
    Ok((context, row))
}

/// The policy engine the context builds stands over the feed's own holder. A
/// worker written at the context's feed door is placed by the context's policy
/// engine, on the object the holder laid in the context's registry.
#[tokio::test]
async fn a_worker_written_at_the_contexts_door_is_placed_by_its_policy_engine() -> Case {
    use std::time::Instant;

    use super::engine::Ask;

    const URL: &str = "http://127.0.0.1:1";
    let (context, row) = context_with_a_worker(URL).await?;
    let policy_engine = context
        .policy_engine
        .as_ref()
        .ok_or("the store has no policy engine")?;
    let placed = policy_engine
        .place(&Ask::of(b"r1", "model-a"), Instant::now())
        .await
        .map_err(|_| "the context's policy engine placed no worker")?;
    assert_eq!(placed.id(), row);
    let laid = context
        .worker_registry
        .get_by_url(URL)
        .ok_or("the registry holds no worker at the URL")?;
    assert!(Arc::ptr_eq(placed.worker(), &laid));
    Ok(())
}

/// The context starts its policy engine's timer. A lease placed through the
/// context's policy engine a minute ago, whose dispatch is never named, is
/// ended at the timer's next pass: no later request has to end it.
#[tokio::test]
async fn a_built_contexts_timer_ends_a_lease_whose_dispatch_was_never_named() -> Case {
    use std::time::{Duration, Instant};

    use smg_policy_controller::store::{GatewayId, LeasesStore};

    use super::{engine::Ask, upkeep};

    let (context, _) = context_with_a_worker("http://127.0.0.1:1").await?;
    let policy_engine = context
        .policy_engine
        .as_ref()
        .ok_or("the store has no policy engine")?;
    let store = context
        .policy_store
        .as_ref()
        .ok_or("the context holds no store")?;
    let issued = Instant::now()
        .checked_sub(Duration::from_secs(60))
        .ok_or("the clock is under a minute old")?;
    let placed = policy_engine
        .place(&Ask::of(b"r1", "model-a"), issued)
        .await
        .map_err(|_| "the context's policy engine placed no worker")?;
    assert_eq!(LeasesStore::live(&**store, GatewayId::LOCAL), 1);

    // The clock is held and moved a tick at a step. The timer's task may
    // take its first turn inside the first step, and its tick then falls
    // a millisecond past that step.
    tokio::time::pause();
    for _ in 0..3 {
        tokio::time::advance(upkeep::EVERY).await;
    }
    tokio::task::yield_now().await;
    assert_eq!(LeasesStore::live(&**store, GatewayId::LOCAL), 0);
    drop(placed);
    Ok(())
}

/// Under a controller's address the store mirrors that controller's rows.
/// The context holds the feed for the follower, and its worker workflows
/// get none: a write of the gateway's own would set the mirror apart. A
/// local worker's registration writes no row there.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[tokio::test]
async fn a_followed_gateway_hands_its_worker_workflows_no_feed() -> Case {
    let hosted = AppContext::from_config(picked(quiet()), 30, None, None).await?;
    assert!(hosted.door_feed().is_some());

    let mut config = picked(quiet());
    config.policy_controller_address = Some("127.0.0.1:1".to_owned());
    let followed = AppContext::from_config(config, 30, None, None).await?;
    assert!(followed.follows_controller());
    assert!(followed.policy_feed.is_some());
    assert!(followed.door_feed().is_none());
    Ok(())
}
