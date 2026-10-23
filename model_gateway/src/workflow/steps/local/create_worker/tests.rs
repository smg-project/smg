use super::*;

/// Under `--parser symphony`, the registration of a model's first worker whose labels select
/// no reader (none, or a reasoning name no table answers to) says that the registry's table
/// reads the model, not what the gateway's flags name, naming both; the model's next worker,
/// a worker whose labels select a reader, a model the registry does not know, flags that name
/// the registry's own table or nothing, and the legacy parser choice, say nothing.
#[test]
fn the_registrys_table_over_the_flags_is_said_once_per_model_at_registration() {
    let flags = Names {
        tool: Some("mistral_v13".to_string()),
        reasoning: None,
    };
    let known = ModelCard::new("Qwen/Qwen3-8B");
    let registry = WorkerRegistry::new();
    let said = |card: &ModelCard, url: &str, parser, flags: &Names, registry: &WorkerRegistry| {
        registry_reads_over_the_flags(card, url, parser, flags, registry)
    };

    let first = said(
        &known,
        "grpc://a:1",
        ParserChoice::Symphony,
        &flags,
        &registry,
    )
    .expect("the model's first worker says so");
    assert!(
        first.contains("'Qwen/Qwen3-8B'")
            && first.contains("'qwen3'")
            && first.contains("the table 'mistral_v13'")
            && first.contains("grpc://a:1"),
        "{first}"
    );
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("grpc://a:1")
                .model(known.clone())
                .build(),
        ))
        .expect("register");
    assert_eq!(
        said(
            &known,
            "grpc://b:1",
            ParserChoice::Symphony,
            &flags,
            &registry
        ),
        None,
        "the model's second worker"
    );

    let fresh = WorkerRegistry::new();
    let mut labeled = known.clone();
    labeled.reasoning_parser = Some("qwen3".to_string());
    assert_eq!(
        said(
            &labeled,
            "grpc://a:1",
            ParserChoice::Symphony,
            &flags,
            &fresh
        ),
        None,
        "a worker whose labels select a reader"
    );
    assert_eq!(
        said(
            &ModelCard::new("acme/ministral-3-14b-instruct-w4a16"),
            "grpc://a:1",
            ParserChoice::Symphony,
            &flags,
            &fresh
        ),
        None,
        "a model the registry does not know"
    );
    let own = Names {
        tool: Some("qwen".to_string()),
        reasoning: None,
    };
    assert_eq!(
        said(&known, "grpc://a:1", ParserChoice::Symphony, &own, &fresh),
        None,
        "the registry's own table"
    );
    // A label that selects nothing (a reasoning name no table answers to) falls through to
    // the registry as no label does, so the worker says it too.
    let mut idle_label = known.clone();
    idle_label.reasoning_parser = Some("deepseek_r1".to_string());
    assert!(
        said(
            &idle_label,
            "grpc://a:1",
            ParserChoice::Symphony,
            &flags,
            &fresh
        )
        .is_some_and(|message| {
            message.contains("'qwen3'") && message.contains("whose labels select no reader")
        }),
        "a label that selects nothing"
    );
    assert_eq!(
        said(
            &known,
            "grpc://a:1",
            ParserChoice::Symphony,
            &Names::default(),
            &fresh
        ),
        None,
        "no flags"
    );
    assert_eq!(
        said(&known, "grpc://a:1", ParserChoice::Legacy, &flags, &fresh),
        None,
        "the legacy parsers"
    );
}

fn make_app_context() -> Arc<crate::app_context::AppContext> {
    app_context_with(
        crate::config::RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked(),
    )
}

fn app_context_with(
    router_config: crate::config::RouterConfig,
) -> Arc<crate::app_context::AppContext> {
    use crate::{
        middleware::{AuthConfig, TokenBucket},
        observability::inflight_tracker::InFlightRequestTracker,
        routers::{
            common::{openai_bridge, realtime::RealtimeRegistry},
            grpc::multimodal::MultimodalConfigRegistry,
        },
        worker::WorkerService,
    };

    let registry = Arc::new(WorkerRegistry::new());
    let job_queue = Arc::new(std::sync::OnceLock::new());

    Arc::new(crate::app_context::AppContext {
        gateway_auth: AuthConfig::new(None),
        client: crate::clients::http_client(),
        router_config: router_config.clone(),
        rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
        rate_limit_manager: None,
        worker_registry: Arc::clone(&registry),
        prefill_admission: None,
        policy_registry: Arc::new(crate::policies::PolicyRegistry::new(
            router_config.policy.clone(),
        )),
        reasoning_parser_factory: None,
        tool_parser_factory: None,
        gateway: None,
        response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
        conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        conversation_item_storage: Arc::new(
            smg_data_connector::MemoryConversationItemStorage::new(),
        ),
        worker_monitor: None,
        configured_reasoning_parser: None,
        configured_tool_parser: None,
        worker_job_queue: Arc::clone(&job_queue),
        workflow_engines: Arc::new(std::sync::OnceLock::new()),
        mcp_orchestrator: Arc::new(std::sync::OnceLock::new()),
        mcp_format_registry: openai_bridge::FormatRegistry::new(),
        tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
        multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
        tokenizer_load_failures: Arc::default(),
        wasm_manager: None,
        worker_client_cache: Arc::new(crate::worker::WorkerHttpClientCache::new(&router_config)),
        worker_service: Arc::new(WorkerService::new(registry, job_queue, router_config)),
        inflight_tracker: InFlightRequestTracker::new(),
        kv_event_monitor: None,
        rl: None,
        realtime_registry: Arc::new(RealtimeRegistry::new()),
        webrtc_bind_addr: None,
        webrtc_stun_server: None,
    })
}

/// A worker registered through the API keeps every spec field the
/// request carried, not only the ones this step copies by hand: the
/// provider, the KV block size, the resilience overrides (the per-model
/// retry config is derived from them at registration), the pairing
/// protocol and the load-monitor interval all have consumers.
#[tokio::test]
async fn api_registered_worker_keeps_the_fields_the_spec_accepted() {
    use openai_protocol::worker::{ProviderType, ResilienceUpdate, WorkerModels};
    use wfaas::WorkflowInstanceId;

    use crate::workflow::steps::create_worker_workflow_data;

    let mut spec = WorkerSpec::new("http://worker:8000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.provider = Some(ProviderType::OpenAI);
    spec.kv_block_size = Some(64);
    spec.pairing_protocol = Some("cluster-blue".to_string());
    spec.load_monitor_interval_secs = Some(3);
    spec.resilience = ResilienceUpdate {
        max_retries: Some(7),
        ..Default::default()
    };
    spec.labels = HashMap::from([
        ("kv_role".to_string(), "kv_consumer".to_string()),
        ("tier".to_string(), "gold".to_string()),
    ]);

    let mut data =
        create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, make_app_context());
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Http);
    data.http_client_handle = Some(Arc::new(crate::clients::http_client()));
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let worker = &ctx.data.actual_workers.as_ref().unwrap()[0];
    let spec = &worker.metadata().spec;
    assert_eq!(spec.url, "http://worker:8000");
    assert_eq!(spec.provider, Some(ProviderType::OpenAI));
    assert_eq!(spec.kv_block_size, Some(64));
    assert_eq!(spec.pairing_protocol.as_deref(), Some("cluster-blue"));
    assert_eq!(spec.load_monitor_interval_secs, Some(3));
    assert_eq!(spec.resilience.max_retries, Some(7));
    assert_eq!(worker.resilience().retry.max_retries, 7);
    // Resolved by this step rather than copied: the KV transfer keys
    // leave the labels for their own fields, and no DP placement is set.
    assert_eq!(spec.kv_role.as_deref(), Some("kv_consumer"));
    assert!(!spec.labels.contains_key("kv_role"));
    assert_eq!(spec.labels["tier"], "gold");
    assert_eq!(spec.dp_rank, None);
    assert_eq!(spec.dp_size, None);
}

/// What a registration records as engine-discovered is the engine's
/// report minus every key the operator's config sets, whatever value the
/// config carries: a config label equal to the engine's value is still
/// the operator's, so a later refresh does not rewrite it as the engine's.
#[tokio::test]
async fn a_config_label_equal_to_the_engines_value_is_not_recorded_as_discovered() {
    use openai_protocol::worker::WorkerModels;
    use wfaas::WorkflowInstanceId;

    use crate::workflow::steps::create_worker_workflow_data;

    let mut spec = WorkerSpec::new("http://worker:8000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.labels = HashMap::from([
        ("page_size".to_string(), "256".to_string()),
        ("tier".to_string(), "gold".to_string()),
    ]);

    let mut data =
        create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, make_app_context());
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Http);
    data.http_client_handle = Some(Arc::new(crate::clients::http_client()));
    data.discovered_labels = HashMap::from([
        ("page_size".to_string(), "256".to_string()),
        ("max_num_seqs".to_string(), "8".to_string()),
    ]);
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let worker = &ctx.data.actual_workers.as_ref().unwrap()[0];
    let labels = &worker.metadata().spec.labels;
    assert_eq!(labels["page_size"], "256");
    assert_eq!(labels["tier"], "gold");
    assert_eq!(labels["max_num_seqs"], "8");
    assert_eq!(
        *worker.metadata().discovered_labels,
        HashMap::from([("max_num_seqs".to_string(), "8".to_string())])
    );
}

/// On a dp-aware registration the engine's report is mapped to the rank's
/// space before the config's keys leave it: a DP label the config also
/// names still sizes the window share the record carries, so a later
/// refresh reads that share, not the whole engine window, as the engine's.
#[tokio::test]
async fn a_dp_label_the_config_also_sets_still_sizes_the_recorded_window_share() {
    use openai_protocol::worker::WorkerModels;
    use wfaas::WorkflowInstanceId;

    use crate::workflow::steps::create_worker_workflow_data;

    let mut spec = WorkerSpec::new("http://worker:8000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.labels = HashMap::from([("enable_dp_attention".to_string(), "true".to_string())]);
    let router_config = crate::config::RouterConfig::builder()
        .worker_startup_timeout_secs(1)
        .dp_aware(true)
        .build_unchecked();

    let mut data = create_worker_workflow_data(
        spec,
        WorkerRegistrationMode::Upsert,
        app_context_with(router_config),
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Http);
    data.http_client_handle = Some(Arc::new(crate::clients::http_client()));
    data.discovered_labels = HashMap::from([
        ("enable_dp_attention".to_string(), "true".to_string()),
        ("max_running_requests".to_string(), "256".to_string()),
    ]);
    data.dp_info = Some(DpInfo {
        dp_size: 8,
        model_id: "m".to_string(),
    });
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let workers = ctx.data.actual_workers.as_ref().unwrap();
    assert_eq!(workers.len(), 8);
    for worker in workers {
        let labels = &worker.metadata().spec.labels;
        assert_eq!(labels["enable_dp_attention"], "true");
        assert_eq!(labels["max_running_requests"], "32");
        assert_eq!(
            *worker.metadata().discovered_labels,
            HashMap::from([("max_running_requests".to_string(), "32".to_string())])
        );
    }
}

/// The width the worker's labels use sizes the record too when only the
/// config names it: an operator's DP flag over an engine that reports no
/// such label still leaves the engine's share, not its whole window, in
/// the record, so a refresh reads the share as the engine's.
#[tokio::test]
async fn a_dp_flag_only_the_config_sets_sizes_the_recorded_window_share_too() {
    use openai_protocol::worker::WorkerModels;
    use wfaas::WorkflowInstanceId;

    use crate::workflow::steps::create_worker_workflow_data;

    let mut spec = WorkerSpec::new("http://worker:8000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.labels = HashMap::from([("enable_dp_attention".to_string(), "true".to_string())]);
    let router_config = crate::config::RouterConfig::builder()
        .worker_startup_timeout_secs(1)
        .dp_aware(true)
        .build_unchecked();

    let mut data = create_worker_workflow_data(
        spec,
        WorkerRegistrationMode::Upsert,
        app_context_with(router_config),
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Http);
    data.http_client_handle = Some(Arc::new(crate::clients::http_client()));
    data.discovered_labels =
        HashMap::from([("max_running_requests".to_string(), "256".to_string())]);
    data.dp_info = Some(DpInfo {
        dp_size: 8,
        model_id: "m".to_string(),
    });
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let workers = ctx.data.actual_workers.as_ref().unwrap();
    assert_eq!(workers.len(), 8);
    for worker in workers {
        assert_eq!(worker.metadata().spec.labels["max_running_requests"], "32");
        assert_eq!(
            *worker.metadata().discovered_labels,
            HashMap::from([("max_running_requests".to_string(), "32".to_string())])
        );
    }
}

/// A loopback port nothing listens on.
fn closed_loopback_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.local_addr().expect("address").port()
}

/// A registration context for a disaggregated worker at a loopback URL.
fn worker_context(
    worker_type: WorkerType,
    bootstrap_port: Option<u16>,
    registration_mode: WorkerRegistrationMode,
) -> WorkflowContext<WorkerWorkflowData> {
    worker_context_on(
        worker_type,
        bootstrap_port,
        registration_mode,
        RuntimeType::Unspecified,
        None,
    )
}

/// A registration context for a disaggregated worker of `runtime` on
/// `kv_connector`, at a loopback URL.
fn worker_context_on(
    worker_type: WorkerType,
    bootstrap_port: Option<u16>,
    registration_mode: WorkerRegistrationMode,
    runtime: RuntimeType,
    kv_connector: Option<&str>,
) -> WorkflowContext<WorkerWorkflowData> {
    use openai_protocol::worker::WorkerModels;
    use wfaas::WorkflowInstanceId;

    use crate::workflow::steps::create_worker_workflow_data;

    let mut spec = WorkerSpec::new("http://127.0.0.1:18000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.worker_type = worker_type;
    spec.bootstrap_port = bootstrap_port;
    spec.runtime_type = runtime;
    spec.kv_connector = kv_connector.map(str::to_string);
    let mut data = create_worker_workflow_data(spec, registration_mode, make_app_context());
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Http);
    data.http_client_handle = Some(Arc::new(crate::clients::http_client()));
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// The probe's count once it lands: the probe runs off the path, within
/// its own timeout.
async fn wait_for_probe_count(handle: &metrics_exporter_prometheus::PrometheusHandle) -> String {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let rendered = handle.render();
        if let Some(line) = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
        {
            return line.to_string();
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the failed probe is counted within its timeout:\n{rendered}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// This step builds a prefill worker with the bootstrap port and host its
/// spec carries and probes nothing: the port is probed once the worker is
/// registered (the register step), so a workflow that fails between the
/// two is not reported as a registered worker with a dead port.
#[tokio::test]
async fn a_prefill_workers_bootstrap_port_is_probed_once_the_worker_is_registered() {
    use crate::workflow::steps::RegisterWorkersStep;

    let closed_port = closed_loopback_port();
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut ctx = worker_context(
        WorkerType::Prefill,
        Some(closed_port),
        WorkerRegistrationMode::Upsert,
    );
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let worker = Arc::clone(&ctx.data.actual_workers.as_ref().unwrap()[0]);
    assert_eq!(worker.bootstrap_port(), Some(closed_port));
    assert_eq!(worker.bootstrap_host(), "127.0.0.1");
    // Built, not registered: nothing is probed yet.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !handle.render().contains("smg_pd_rendezvous_failures_total"),
        "{}",
        handle.render()
    );

    assert_eq!(
        RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let counted = wait_for_probe_count(&handle).await;
    assert!(
        counted.contains(r#"leg="prefill""#)
            && counted.contains(r#"phase="probe""#)
            && counted.ends_with(" 1"),
        "{counted}"
    );
}

/// A worker whose registration the register step refuses never joined the
/// registry, so its port is not probed and no probe failure is counted
/// for it; a decode worker without a bootstrap port is not probed either,
/// since no partner is handed a port for it.
#[tokio::test]
async fn a_worker_the_registry_refuses_or_a_decode_worker_without_a_bootstrap_port_is_not_probed() {
    use crate::workflow::steps::RegisterWorkersStep;

    let closed_port = closed_loopback_port();
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);

    // The replacement of a worker the registry no longer holds is refused.
    let refused = WorkerRegistrationMode::ReplaceById {
        worker_id: "worker-gone".to_string(),
        expected_revision: 0,
    };
    let mut ctx = worker_context(WorkerType::Prefill, Some(closed_port), refused);
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    assert!(RegisterWorkersStep.execute(&mut ctx).await.is_err());

    let mut ctx = worker_context(WorkerType::Decode, None, WorkerRegistrationMode::Upsert);
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    assert_eq!(
        RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !handle.render().contains("smg_pd_rendezvous_failures_total"),
        "{}",
        handle.render()
    );
}

/// Whether something on this host answers on the default bootstrap port:
/// the tests of a worker registered without a port cannot tell a probe that
/// found a listener there from one that never ran, so they stand down.
fn something_listens_on_the_default_bootstrap_port() -> bool {
    use std::net::{SocketAddr, TcpStream};

    let address = SocketAddr::from(([127, 0, 0, 1], crate::worker::DEFAULT_BOOTSTRAP_PORT));
    TcpStream::connect_timeout(&address, Duration::from_millis(500)).is_ok()
}

/// A prefill worker registered without a bootstrap port (a `POST /workers`
/// without the field, a discovered pod without the annotation) is handed
/// the default port on the request path, so the default port is what is
/// probed for it: with nothing listening there, the failed probe is
/// counted once the worker is registered, like a registered port's.
#[tokio::test]
async fn a_prefill_worker_registered_without_a_bootstrap_port_is_probed_on_the_default_port() {
    use crate::workflow::steps::RegisterWorkersStep;

    if something_listens_on_the_default_bootstrap_port() {
        return;
    }
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut ctx = worker_context(WorkerType::Prefill, None, WorkerRegistrationMode::Upsert);
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let worker = Arc::clone(&ctx.data.actual_workers.as_ref().unwrap()[0]);
    assert_eq!(
        worker.bootstrap_port(),
        None,
        "the spec keeps what was registered"
    );
    assert_eq!(
        RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let counted = wait_for_probe_count(&handle).await;
    assert!(
        counted.contains(r#"leg="prefill""#)
            && counted.contains(r#"phase="probe""#)
            && counted.ends_with(" 1"),
        "{counted}"
    );
}

/// A prefill worker on a KV store registered with the bootstrap port the
/// rendezvous recipe carries: no partner is ever handed it, so nothing is
/// probed and no probe failure is counted once the worker is registered.
#[tokio::test]
async fn a_store_prefill_workers_registered_bootstrap_port_is_not_probed() {
    use crate::workflow::steps::RegisterWorkersStep;

    let closed_port = closed_loopback_port();
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut ctx = worker_context_on(
        WorkerType::Prefill,
        Some(closed_port),
        WorkerRegistrationMode::Upsert,
        RuntimeType::Vllm,
        Some("MooncakeStoreConnector"),
    );
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let worker = Arc::clone(&ctx.data.actual_workers.as_ref().unwrap()[0]);
    assert_eq!(
        worker.bootstrap_port(),
        Some(closed_port),
        "the spec keeps what was registered"
    );
    assert_eq!(
        worker.metadata().spec.kv_connector.as_deref(),
        Some("MooncakeStoreConnector")
    );
    assert_eq!(
        RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    // A probe of the closed loopback port is refused and counted at once;
    // wait longer than that before reading nothing.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let rendered = handle.render();
    assert!(
        !rendered.contains("smg_pd_rendezvous_failures_total"),
        "{rendered}"
    );
}

/// The dedicated `kv_role` field follows the same precedence as the
/// connector and engine id: it wins over the legacy label, and an empty
/// value falls back to the label. A `POST /workers` carrying a top-level
/// `kv_role` and no label used to lose the role here.
#[test]
fn dedicated_kv_role_wins_over_the_label() {
    let mut spec = WorkerSpec::new("http://worker:8080");
    spec.kv_role = Some("kv_producer".to_string());
    let mut labels = HashMap::new();
    let (_, role, _) = take_kv_transfer_metadata(&spec, &mut labels);
    assert_eq!(role.as_deref(), Some("kv_producer"));

    labels.insert("kv_role".to_string(), "kv_consumer".to_string());
    let (_, role, _) = take_kv_transfer_metadata(&spec, &mut labels);
    assert_eq!(role.as_deref(), Some("kv_producer"));
    assert!(!labels.contains_key("kv_role"));

    spec.kv_role = Some(String::new());
    labels.insert("kv_role".to_string(), "kv_consumer".to_string());
    let (_, role, _) = take_kv_transfer_metadata(&spec, &mut labels);
    assert_eq!(role.as_deref(), Some("kv_consumer"));
}

#[test]
fn dedicated_kv_metadata_wins_and_legacy_labels_are_removed() {
    let mut spec = WorkerSpec::new("http://worker:8080");
    spec.kv_connector = Some("NixlConnector".to_string());
    spec.kv_engine_id = Some("engine-direct".to_string());
    let mut labels = HashMap::from([
        ("kv_connector".to_string(), "legacy".to_string()),
        ("kv_role".to_string(), "kv_consumer".to_string()),
        ("kv_engine_id".to_string(), "engine-legacy".to_string()),
    ]);

    let metadata = take_kv_transfer_metadata(&spec, &mut labels);
    assert_eq!(
        metadata,
        (
            Some("NixlConnector".to_string()),
            Some("kv_consumer".to_string()),
            Some("engine-direct".to_string()),
        )
    );
    assert!(labels.is_empty());

    spec.kv_connector = Some(String::new());
    spec.kv_engine_id = Some(String::new());
    labels.insert("kv_connector".to_string(), "legacy".to_string());
    labels.insert("kv_engine_id".to_string(), "engine-legacy".to_string());
    let metadata = take_kv_transfer_metadata(&spec, &mut labels);
    assert_eq!(
        (metadata.0.as_deref(), metadata.2.as_deref()),
        (Some("legacy"), Some("engine-legacy"))
    );
}

#[test]
fn dp_expansion_pins_only_multi_rank_widths() {
    let multi = DpInfo {
        dp_size: 3,
        model_id: "m".to_string(),
    };
    assert_eq!(
        dp_rank_expansion(Some(&multi), ConnectionMode::Grpc, "grpc://w:1").unwrap(),
        vec![Some((0, 3)), Some((1, 3)), Some((2, 3))]
    );

    // Width 1 is a capability-only advertisement: no placement exists,
    // and a rank-0 pin would still alter engine scheduling.
    let single = DpInfo {
        dp_size: 1,
        model_id: "m".to_string(),
    };
    assert_eq!(
        dp_rank_expansion(Some(&single), ConnectionMode::Grpc, "grpc://w:1").unwrap(),
        vec![None]
    );
    assert_eq!(
        dp_rank_expansion(None, ConnectionMode::Grpc, "grpc://w:1").unwrap(),
        vec![None]
    );
}

#[test]
fn normalize_url_preserves_existing_schemes() {
    assert_eq!(
        normalize_url("http://localhost:30000", ConnectionMode::Http),
        "http://localhost:30000"
    );
    assert_eq!(
        normalize_url("https://localhost:30000", ConnectionMode::Http),
        "https://localhost:30000"
    );
    assert_eq!(
        normalize_url("grpc://localhost:30001", ConnectionMode::Grpc),
        "grpc://localhost:30001"
    );
    assert_eq!(
        normalize_url("grpcs://localhost:30001", ConnectionMode::Grpc),
        "grpcs://localhost:30001"
    );
}

#[test]
fn normalize_url_adds_scheme_for_bare_urls() {
    assert_eq!(
        normalize_url("localhost:30000", ConnectionMode::Http),
        "http://localhost:30000"
    );
    assert_eq!(
        normalize_url("localhost:30001", ConnectionMode::Grpc),
        "grpc://localhost:30001"
    );
}

#[test]
fn degenerate_overload_overrides_are_rejected_at_registration() {
    // Silent acceptance would strand the worker permanently vetoed; the
    // ConfigValidator rules apply to spec blocks too.
    let mut spec = WorkerSpec::new("http://worker:8080");
    spec.overload.waiting_requests = Some(0);
    let err =
        validate_overload_overrides(&spec).expect_err("waiting_requests = 0 must be rejected");
    assert!(err.contains("overload.waiting_requests"), "{err}");
    assert!(err.contains("http://worker:8080"), "{err}");

    for bad in [0.0, -0.5, 1.5, f64::NAN] {
        let mut spec = WorkerSpec::new("http://worker:8080");
        spec.overload.token_usage = Some(bad);
        let err = validate_overload_overrides(&spec)
            .expect_err("degenerate token_usage must be rejected");
        assert!(err.contains("(0.0, 1.0]"), "{err}");
    }

    // The excluded ends' valid neighbors and an empty block pass.
    let mut spec = WorkerSpec::new("http://worker:8080");
    spec.overload.waiting_requests = Some(1);
    spec.overload.token_usage = Some(1.0);
    assert!(validate_overload_overrides(&spec).is_ok());
    assert!(validate_overload_overrides(&WorkerSpec::new("x")).is_ok());
}

#[test]
fn zmq_handshake_override_is_rejected_off_the_zmq_path() {
    let mut spec = WorkerSpec::new("http://worker:8080");
    spec.zmq_handshake_address = Some("tcp://127.0.0.1:30500".to_string());

    for mode in [ConnectionMode::Http, ConnectionMode::Grpc] {
        let err = validate_zmq_handshake_override(&spec, mode)
            .expect_err("non-ZMQ workers must reject the handshake override");
        assert!(err.contains("zmq_handshake_address"), "{err}");
    }
    // On the ZMQ path the override is legitimate; unset is always fine.
    assert!(validate_zmq_handshake_override(&spec, ConnectionMode::Zmq).is_ok());
    assert!(validate_zmq_handshake_override(&WorkerSpec::new("x"), ConnectionMode::Http).is_ok());
}

fn alias_map(entries: &[(&str, &str)]) -> HashMap<String, String> {
    entries
        .iter()
        .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
        .collect()
}

#[test]
fn model_alias_map_attaches_matching_aliases_to_discovered_card() {
    // The service-discovery path supplies no user card, so the card is
    // built from the discovered model ID alone; the router-level alias
    // map is the only way it can gain aliases.
    let spec = WorkerSpec::new("http://worker:8080");
    let aliases = alias_map(&[
        ("GLM-5.2-Coding", "GLM-5.2"),
        ("other-alias", "other-model"),
    ]);

    let card = build_model_card("GLM-5.2", &spec, &HashMap::new(), &aliases);

    assert_eq!(card.id, "GLM-5.2");
    assert_eq!(card.aliases, vec!["GLM-5.2-Coding".to_string()]);
}

#[test]
fn service_discovery_keeps_served_model_name_canonical() {
    // Service discovery supplies an empty model list. Metadata discovery
    // supplies the backend's served_model_name.
    let spec = WorkerSpec::new("http://worker:8080");
    let labels = HashMap::from([
        ("served_model_name".to_string(), "GLM-5.2".to_string()),
        ("model_id".to_string(), "GLM-5.2-Coding".to_string()),
        ("model_path".to_string(), "unrelated-alias".to_string()),
    ]);
    let aliases = alias_map(&[
        ("GLM-5.2-Coding", "GLM-5.2"),
        ("unrelated-alias", "other-model"),
    ]);

    let model_id = resolve_model_id(&spec, &labels);
    let card = build_model_card(model_id, &spec, &labels, &aliases);
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(&spec.url).model(card).build());
    let registry = WorkerRegistry::new();
    registry.register(worker.clone()).unwrap();

    assert_eq!(worker.model_id(), "GLM-5.2");
    assert_eq!(registry.get_by_model("GLM-5.2").len(), 1);
    assert_eq!(registry.get_by_model("GLM-5.2-Coding").len(), 1);
    assert_eq!(
        registry.resolve_model_alias("GLM-5.2-Coding").as_deref(),
        Some("GLM-5.2")
    );
    assert!(registry.get_by_model("unrelated-alias").is_empty());
}

#[test]
fn model_alias_map_is_case_sensitive_and_skips_self_reference() {
    let spec = WorkerSpec::new("http://worker:8080");
    // Wrong-case canonical must not match; an alias equal to the model ID
    // must not be attached (it would shadow the canonical entry).
    let aliases = alias_map(&[("glm-5.2-coding", "glm-5.2"), ("GLM-5.2", "GLM-5.2")]);

    let card = build_model_card("GLM-5.2", &spec, &HashMap::new(), &aliases);

    assert!(card.aliases.is_empty());
}

fn zmq_worker(url: &str, handshake_override: Option<&str>) -> Arc<dyn Worker> {
    let mut spec = WorkerSpec::new(url);
    spec.connection_mode = ConnectionMode::Zmq;
    spec.zmq_handshake_address = handshake_override.map(str::to_string);
    Arc::new(BasicWorkerBuilder::from_spec(spec).build())
}

/// The ZMQ link's request bound resolves as the HTTP worker client's
/// does: the worker's own `http_pool.timeout_secs` when set, else the
/// router's `request_timeout_secs`.
#[test]
fn zmq_request_timeout_takes_the_workers_pool_timeout_over_the_routers() {
    let mut spec = WorkerSpec::new("ipc:///tmp/smg-zmq/bound.ipc");
    spec.connection_mode = ConnectionMode::Zmq;
    assert_eq!(zmq_request_timeout(&spec, 1800), Duration::from_secs(1800));
    spec.http_pool.timeout_secs = Some(7200);
    assert_eq!(zmq_request_timeout(&spec, 1800), Duration::from_secs(7200));
}

#[test]
fn zmq_handshake_collision_is_rejected_at_registration() {
    // Two ipc paths hashing to the same derived port would leave the second
    // worker's handshake bind failing forever with a bare transport error.
    let registry = WorkerRegistry::new();
    let first = "ipc:///tmp/smg-zmq/a.ipc";
    let handshake = zmq_handshake_address(first, None).expect("derived handshake address");
    registry.register(zmq_worker(first, None)).unwrap();

    // Same derived address reached through an explicit override.
    let err = validate_zmq_handshake_address(
        &registry,
        "ipc:///tmp/smg-zmq/b.ipc",
        Some(&handshake),
        ConnectionMode::Zmq,
    )
    .expect_err("a colliding handshake address must be rejected");
    assert!(err.contains(&handshake), "message was: {err}");
    assert!(err.contains(first), "message was: {err}");

    // A distinct path derives a distinct address; re-registering the same
    // URL (update path) is not a collision with itself.
    validate_zmq_handshake_address(
        &registry,
        "ipc:///tmp/smg-zmq/b.ipc",
        None,
        ConnectionMode::Zmq,
    )
    .expect("a distinct ipc path must be accepted");
    validate_zmq_handshake_address(&registry, first, None, ConnectionMode::Zmq)
        .expect("re-registering the same worker URL must be accepted");
    validate_zmq_handshake_address(&registry, "grpc://worker:8080", None, ConnectionMode::Grpc)
        .expect("non-ZMQ workers are not affected");
}

#[test]
fn malformed_zmq_handshake_address_is_rejected_at_registration() {
    // A handshake address that can never bind must fail here, with the
    // reason named — not silently register a worker whose every connect
    // attempt dies on a bare transport error.
    let registry = WorkerRegistry::new();

    // The engine dials a TCP handshake, so an ipc:// override is unusable.
    let err = validate_zmq_handshake_address(
        &registry,
        "ipc:///tmp/smg-zmq/a.ipc",
        Some("ipc:///tmp/smg-zmq/handshake.sock"),
        ConnectionMode::Zmq,
    )
    .expect_err("a non-tcp:// handshake override must be rejected");
    assert!(err.contains("tcp://"), "message was: {err}");
    assert!(
        err.contains("ipc:///tmp/smg-zmq/handshake.sock"),
        "message was: {err}"
    );

    // A ZMQ worker whose base URL is not an ipc:// path derives nothing.
    let err =
        validate_zmq_handshake_address(&registry, "http://worker:8080", None, ConnectionMode::Zmq)
            .expect_err("a non-ipc:// ZMQ base URL must be rejected");
    assert!(err.contains("ipc://"), "message was: {err}");

    // A well-formed tcp:// override on an ipc:// base is the supported form.
    validate_zmq_handshake_address(
        &registry,
        "ipc:///tmp/smg-zmq/a.ipc",
        Some("tcp://127.0.0.1:30500"),
        ConnectionMode::Zmq,
    )
    .expect("a tcp:// override must be accepted");

    // A peer that reached the registry with an unusable address is skipped,
    // not inherited: it cannot claim a handshake address to collide with.
    registry
        .register(zmq_worker("http://peer:8080", None))
        .expect("registering the malformed peer");
    validate_zmq_handshake_address(
        &registry,
        "ipc:///tmp/smg-zmq/a.ipc",
        None,
        ConnectionMode::Zmq,
    )
    .expect("an unusable peer address must not block a valid registration");
}

#[test]
fn zmq_workers_keep_health_checks_enabled() {
    // The ZMQ probe drives the handshake and evicts dead clients, so
    // disabling it would brick the worker on engine restart.
    let disabled = HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    };

    let resolved = resolve_zmq_health_config(
        disabled.clone(),
        ConnectionMode::Zmq,
        "ipc:///tmp/smg-zmq/a.ipc",
    );
    assert!(!resolved.disable_health_check);
    assert_eq!(
        resolved.check_interval_secs, disabled.check_interval_secs,
        "only the disable flag is overridden"
    );

    assert!(
        resolve_zmq_health_config(disabled, ConnectionMode::Http, "http://w:1")
            .disable_health_check,
        "other transports keep the operator's choice"
    );
}

#[test]
fn zmq_disaggregated_workers_are_rejected_as_a_create_worker_failure() {
    for worker_type in [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode] {
        let err = validate_zmq_worker_type(
            ConnectionMode::Zmq,
            worker_type,
            "ipc:///tmp/smg-zmq/ts0.ipc",
        )
        .expect_err("ZMQ encode/prefill/decode must be rejected");
        match err {
            WorkflowError::StepFailed { step_id, message } => {
                assert_eq!(step_id, StepId::new("create_worker"));
                assert!(
                    message.contains(&worker_type.to_string()),
                    "message was: {message}"
                );
            }
            other => panic!("expected StepFailed, got {other:?}"),
        }
    }

    // Regular is the one role the ZMQ lane serves; every worker type over
    // the other transports stays accepted.
    validate_zmq_worker_type(
        ConnectionMode::Zmq,
        WorkerType::Regular,
        "ipc:///tmp/smg-zmq/ts0.ipc",
    )
    .expect("ZMQ regular workers are the supported case");
    for mode in [ConnectionMode::Http, ConnectionMode::Grpc] {
        for worker_type in [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode] {
            validate_zmq_worker_type(mode, worker_type, "grpc://worker:8080")
                .expect("non-ZMQ transports serve every worker type");
        }
    }
}

#[test]
fn zmq_data_parallel_is_rejected_as_a_create_worker_failure() {
    // dp_size > 1 over ZMQ is the only rejected combination, and it must
    // surface as a create_worker StepFailed.
    let err = validate_zmq_dp(ConnectionMode::Zmq, 2, "ipc:///tmp/smg-zmq/ts0.ipc")
        .expect_err("dp_size > 1 over ZMQ must be rejected");
    match err {
        WorkflowError::StepFailed { step_id, message } => {
            assert_eq!(step_id, StepId::new("create_worker"));
            assert!(message.contains("dp_size=2"), "message was: {message}");
        }
        other => panic!("expected StepFailed, got {other:?}"),
    }
}

#[test]
fn single_engine_zmq_and_data_parallel_grpc_are_accepted() {
    // Single-engine ZMQ is the supported ZMQ shape; gRPC/HTTP data
    // parallelism is untouched by the ZMQ guard.
    validate_zmq_dp(ConnectionMode::Zmq, 1, "ipc:///tmp/smg-zmq/ts0.ipc")
        .expect("single-engine ZMQ must be accepted");
    validate_zmq_dp(ConnectionMode::Grpc, 4, "grpc://worker:8080")
        .expect("gRPC data parallelism must be accepted");
    validate_zmq_dp(ConnectionMode::Http, 4, "http://worker:8080")
        .expect("HTTP data parallelism must be accepted");
}

#[test]
fn model_alias_map_does_not_duplicate_user_provided_alias() {
    // POST /workers can already carry aliases in the spec; the router map
    // must merge, not duplicate, so repeated registration stays stable.
    let mut spec = WorkerSpec::new("http://worker:8080");
    let card_with_alias = ModelCard::new("GLM-5.2").with_alias("GLM-5.2-Coding");
    spec.models = vec![card_with_alias].into();
    let aliases = alias_map(&[("GLM-5.2-Coding", "GLM-5.2"), ("glm-5.2", "GLM-5.2")]);

    let card = build_model_card("GLM-5.2", &spec, &HashMap::new(), &aliases);

    assert_eq!(
        card.aliases
            .iter()
            .filter(|a| *a == "GLM-5.2-Coding")
            .count(),
        1
    );
    assert!(card.aliases.contains(&"glm-5.2".to_string()));
    assert_eq!(card.aliases.len(), 2);
}

#[test]
fn parser_override_labels_flow_into_card() {
    let spec = WorkerSpec::new("http://worker:8080");
    let labels = HashMap::from([
        ("tool_parser".to_string(), "json".to_string()),
        ("reasoning_parser".to_string(), "basic".to_string()),
        ("model_type".to_string(), "llama".to_string()),
    ]);

    let card = build_model_card("m", &spec, &labels, &HashMap::new());
    assert_eq!(card.tool_parser.as_deref(), Some("json"));
    assert_eq!(card.reasoning_parser.as_deref(), Some("basic"));

    // Empty label values are treated as unset.
    let labels = HashMap::from([("tool_parser".to_string(), String::new())]);
    let card = build_model_card("m", &spec, &labels, &HashMap::new());
    assert_eq!(card.tool_parser, None);
}

#[test]
fn explicit_card_parser_wins_over_labels() {
    // A user-provided WorkerSpec card keeps its parser fields; labels
    // only fill gaps (same precedence as tokenizer_path).
    let mut spec = WorkerSpec::new("http://worker:8080");
    spec.models = openai_protocol::worker::WorkerModels::Single(Box::new(
        ModelCard::new("m").with_tool_parser("pythonic"),
    ));
    let labels = HashMap::from([
        ("tool_parser".to_string(), "json".to_string()),
        ("reasoning_parser".to_string(), "basic".to_string()),
    ]);

    let card = build_model_card("m", &spec, &labels, &HashMap::new());
    assert_eq!(card.tool_parser.as_deref(), Some("pythonic"));
    // The explicit card left reasoning unset — the label fills it.
    assert_eq!(card.reasoning_parser.as_deref(), Some("basic"));
}

/// An engine's own reasoning parser rides in its own label
/// (`engine_reasoning_parser`), not in `reasoning_parser`, the per-model
/// parser override registration validates against the gateway's registry:
/// the engine's parser names are the engine's (`openai_gptoss`, `kimi_k2`),
/// which the gateway may not have, and a shared name must not outrank the
/// configured `--reasoning-parser`. Such a worker registers, its card
/// carries no override, and its grammar start reads as advertised.
#[test]
fn an_engines_advertised_reasoning_parser_is_not_a_parser_override() {
    use tool_parser::GrammarStart;

    use crate::routers::grpc::client::ServerInfo;

    let tool_factory = tool_parser::ParserFactory::default();
    let reasoning_factory = reasoning_parser::ParserFactory::default();
    let spec = WorkerSpec::new("grpc://engine:50051");
    for name in ["openai_gptoss", "kimi_k2", "qwen3", "glm45"] {
        let info = ServerInfo::Vllm(Box::new(
            smg_grpc_client::vllm_proto::GetServerInfoResponse {
                engine_reasoning_parser: name.to_string(),
                constrained_decoding_start: "after_reasoning".to_string(),
                ..Default::default()
            },
        ));
        let labels = info.to_labels();
        assert_eq!(
            labels.get("engine_reasoning_parser").map(String::as_str),
            Some(name)
        );
        let card = build_model_card("m", &spec, &labels, &HashMap::new());
        assert_eq!(
            card.reasoning_parser, None,
            "{name}: no override from the engine's fact"
        );
        assert_eq!(card.tool_parser, None);
        validate_parser_overrides(
            &card,
            "grpc://engine:50051",
            ParserChoice::Legacy,
            Some(&tool_factory),
            Some(&reasoning_factory),
        )
        .unwrap_or_else(|e| panic!("{name}: the worker must register: {e}"));
        assert_eq!(
            GrammarStart::from_label(labels.get(GrammarStart::LABEL).map(String::as_str)),
            GrammarStart::AfterReasoning,
            "{name}: the grammar start reads as advertised"
        );
    }
}

#[test]
fn a_symphony_table_is_a_parser_override_under_parser_symphony_alone() {
    let tool_factory = tool_parser::ParserFactory::default();
    let reasoning_factory = reasoning_parser::ParserFactory::default();
    let mut card = ModelCard::new("m");
    card.tool_parser = Some("ling".to_string());
    card.reasoning_parser = Some("ling".to_string());
    let check = |card: &ModelCard, parser| {
        validate_parser_overrides(
            card,
            "http://w:1",
            parser,
            Some(&tool_factory),
            Some(&reasoning_factory),
        )
    };
    assert!(check(&card, ParserChoice::Symphony).is_ok());
    let err = check(&card, ParserChoice::Legacy).expect_err("a table is no legacy parser");
    assert!(
        err.contains("'ling'") && err.contains("--parser symphony"),
        "{err}"
    );
    // A table no legacy parser shares binds the other override to the same table.
    card.reasoning_parser = Some("qwen3".to_string());
    let err = check(&card, ParserChoice::Symphony).expect_err("bound to the table");
    assert!(err.contains("same table"), "{err}");
    // A name both registries have passes under either parser.
    card.tool_parser = Some("kimi_k3".to_string());
    card.reasoning_parser = Some("kimi_k3".to_string());
    assert!(check(&card, ParserChoice::Symphony).is_ok());
    assert!(check(&card, ParserChoice::Legacy).is_ok());
}

#[test]
fn unknown_parser_override_fails_validation() {
    let tool_factory = tool_parser::ParserFactory::default();
    let reasoning_factory = reasoning_parser::ParserFactory::default();

    // No overrides → nothing to validate.
    let card = ModelCard::new("m");
    assert!(validate_parser_overrides(
        &card,
        "http://w:1",
        ParserChoice::Legacy,
        Some(&tool_factory),
        Some(&reasoning_factory)
    )
    .is_ok());

    // Known tool parser passes; unknown names fail loudly.
    let card = ModelCard::new("m").with_tool_parser("json");
    assert!(validate_parser_overrides(
        &card,
        "http://w:1",
        ParserChoice::Legacy,
        Some(&tool_factory),
        None
    )
    .is_ok());

    let card = ModelCard::new("m").with_tool_parser("definitely-not-a-parser");
    let err = validate_parser_overrides(
        &card,
        "http://w:1",
        ParserChoice::Legacy,
        Some(&tool_factory),
        None,
    )
    .expect_err("unknown tool parser must fail registration");
    assert!(err.contains("definitely-not-a-parser"), "{err}");

    let card = ModelCard::new("m").with_reasoning_parser("definitely-not-a-parser");
    let err = validate_parser_overrides(
        &card,
        "http://w:1",
        ParserChoice::Legacy,
        None,
        Some(&reasoning_factory),
    )
    .expect_err("unknown reasoning parser must fail registration");
    assert!(err.contains("definitely-not-a-parser"), "{err}");

    // Absent factories skip validation (parsers unused in that config).
    let card = ModelCard::new("m").with_tool_parser("definitely-not-a-parser");
    assert!(
        validate_parser_overrides(&card, "http://w:1", ParserChoice::Legacy, None, None).is_ok()
    );
}
