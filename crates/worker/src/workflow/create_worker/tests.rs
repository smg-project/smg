//! Cases for building a local registration's workers.

use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    health::{circuit_breaker::CircuitBreakerConfig, overload::OverloadThresholds},
    identity::basic::BasicWorker,
    jobs::WorkerJobAsker,
    settings::RetryConfig,
    test_support::{labels_of, BasicWorkerBuilder, ScriptedConnector, WorkerRegistry},
    worker::{OverloadUpdate, ProviderType, ResilienceUpdate, WorkerModels},
    workflow::{
        dp_info_from_labels,
        host::testing::{registration_settings, ProbingHost},
    },
};

type Data = WorkerRegistrationWorkflowData<ProbingHost>;

fn local_data(spec: WorkerSpec, mode: ConnectionMode, host: Arc<ProbingHost>) -> Data {
    let mut data = WorkerRegistrationWorkflowData::new(
        spec,
        WorkerRegistrationMode::Upsert,
        host,
        WorkerJobAsker::Api,
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(mode);
    data
}

fn ordinary_host() -> Arc<ProbingHost> {
    ProbingHost::new(false, ScriptedConnector::default())
}

#[tokio::test]
async fn a_host_with_no_gateway_type_creates_a_worker() {
    let host = ordinary_host();
    let mut spec = WorkerSpec::new("worker-a:9000");
    spec.labels = labels_of(&[("tier", "configured")]);
    let mut data = local_data(spec, ConnectionMode::Grpc, Arc::clone(&host));
    data.discovered_labels = labels_of(&[
        ("served_model_name", "m"),
        ("tier", "reported"),
        ("dp_size", "2"),
    ]);
    data.dp_info = dp_info_from_labels(&data.discovered_labels).unwrap();
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );

    let workers = ctx.data.actual_workers.as_ref().unwrap();
    assert_eq!(workers.len(), 2);
    assert_eq!(
        ctx.data.final_labels,
        labels_of(&[
            ("served_model_name", "m"),
            ("tier", "configured"),
            ("dp_size", "2"),
        ])
    );
    let connector = host.engine_connector();
    for (rank, worker) in workers.iter().enumerate() {
        assert_eq!(worker.url(), format!("grpc://worker-a:9000@{rank}"));
        assert_eq!(worker.metadata().base_url(), "grpc://worker-a:9000");
        assert_eq!(worker.dp_rank(), Some(rank));
        assert_eq!(worker.dp_size(), Some(2));
        assert_eq!(worker.model_id(), "m");
        assert_eq!(worker.metadata().spec.labels, ctx.data.final_labels);
        assert!(Arc::ptr_eq(&worker.engine_connector().unwrap(), &connector));
    }
    assert_eq!(host.connects(), 0);
    assert!(host.worker_registry().is_empty());
}

#[tokio::test]
async fn the_creation_reads_its_bounds_from_the_settings() {
    let settings = crate::settings::RegistrationSettings {
        model_path: Some("/models/local".to_string()),
        model_aliases: labels_of(&[("m", "/models/local")]),
        retry: RetryConfig {
            max_retries: 19,
            ..RetryConfig::default()
        },
        retries_enabled: false,
        circuit_breaker: CircuitBreakerConfig {
            failure_threshold: 17,
            ..CircuitBreakerConfig::default()
        },
        circuit_breaker_enabled: false,
        overload_defaults: OverloadThresholds {
            waiting_requests: Some(7),
            token_usage: Some(0.5),
        },
        health: HealthCheckConfig {
            timeout_secs: 13,
            ..HealthCheckConfig::default()
        },
        health_endpoint: "/working".to_string(),
        zmq_startup_timeout_secs: 29,
        request_timeout_secs: 37,
        ..registration_settings()
    };
    let host = ProbingHost::with_settings(settings, ScriptedConnector::default());
    let mut ctx = WorkflowContext::new(
        WorkflowInstanceId::new(),
        local_data(
            WorkerSpec::new("/tmp/worker-a.ipc"),
            ConnectionMode::Zmq,
            Arc::clone(&host),
        ),
    );

    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );

    let worker = &ctx.data.actual_workers.as_ref().unwrap()[0];
    let metadata = worker.metadata();
    let settings = host.registration_settings();
    assert_eq!(worker.model_id(), "/models/local");
    assert_eq!(worker.models()[0].aliases, ["m"]);
    assert_eq!(worker.url(), "ipc:///tmp/worker-a.ipc");
    assert_eq!(metadata.spec.runtime_type, RuntimeType::Sglang);
    assert_eq!(metadata.health_endpoint, "/working");
    assert_eq!(metadata.health_config, settings.health);
    assert_eq!(metadata.zmq_startup_timeout, Duration::from_secs(29));
    assert_eq!(metadata.zmq_request_timeout, Some(Duration::from_secs(37)));
    assert_eq!(metadata.overload, settings.overload_defaults);
    assert_eq!(worker.resilience().retry, settings.retry);
    assert!(!worker.resilience().retry_enabled);
    assert!(!worker.resilience().circuit_breaker_enabled);
    let worker = worker.as_any().downcast_ref::<BasicWorker>().unwrap();
    assert_eq!(worker.circuit_breaker_config(), settings.circuit_breaker);
}

#[tokio::test]
async fn a_local_creation_without_a_host_fails_naming_the_host() {
    let mut data = local_data(
        WorkerSpec::new("http://worker-a:8000"),
        ConnectionMode::Http,
        ordinary_host(),
    );
    data.host = None;
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    let error = CreateLocalWorkerStep.execute(&mut ctx).await.unwrap_err();
    assert!(matches!(&error, WorkflowError::ContextValueNotFound(name) if name == "host"));
    assert!(
        !<CreateLocalWorkerStep as StepExecutor<Data>>::is_retryable(
            &CreateLocalWorkerStep,
            &error
        )
    );

    ctx.data.worker_kind = Some(WorkerKind::External);
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Skip
    );
}

#[tokio::test]
async fn creation_refuses_a_runtime_without_a_zmq_wire() {
    let mut spec = WorkerSpec::new("ipc:///tmp/worker-a.ipc");
    spec.runtime_type = RuntimeType::Generic;
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    let mut ctx = WorkflowContext::new(
        WorkflowInstanceId::new(),
        local_data(spec, ConnectionMode::Zmq, ordinary_host()),
    );

    let error = CreateLocalWorkerStep.execute(&mut ctx).await.unwrap_err();
    assert_eq!(
        error.to_string(),
        "Step failed: create_worker - ZMQ worker ipc:///tmp/worker-a.ipc has unsupported runtime \
         generic: only vllm, tokenspeed and sglang are supported over the ZMQ direct backend"
    );
    assert!(ctx.data.actual_workers.is_none());
}

#[tokio::test]
async fn create_only_checks_the_hosts_registry() {
    let url = "http://worker-a:8000";
    let host = ordinary_host();
    let worker = crate::test_support::ready_worker(url, "m");
    host.worker_registry().register(worker).unwrap();
    let mut data = local_data(WorkerSpec::new(url), ConnectionMode::Http, host);
    data.registration_mode = WorkerRegistrationMode::CreateOnly;
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    let error = CreateLocalWorkerStep.execute(&mut ctx).await.unwrap_err();

    assert_eq!(
        error.to_string(),
        "Step failed: create_worker - Worker http://worker-a:8000 already exists"
    );
    assert!(ctx.data.actual_workers.is_none());
}

/// A worker registered through the API keeps every spec field the
/// request carried, not only the ones this step copies by hand: the
/// provider, the KV block size, the resilience overrides (the per-model
/// retry config is derived from them at registration), the pairing
/// protocol and the load-monitor interval all have consumers.
#[tokio::test]
async fn api_registered_worker_keeps_the_fields_the_spec_accepted() {
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

    let data = local_data(spec, ConnectionMode::Http, ordinary_host());
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
    let mut spec = WorkerSpec::new("http://worker:8000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.labels = HashMap::from([
        ("page_size".to_string(), "256".to_string()),
        ("tier".to_string(), "gold".to_string()),
    ]);

    let mut data = local_data(spec, ConnectionMode::Http, ordinary_host());
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

/// A registration of one model under the DP label, as a local HTTP worker
/// before the engine's report is set on it.
fn ranked_local_http_data() -> Data {
    let mut spec = WorkerSpec::new("http://worker:8000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.labels = HashMap::from([("enable_dp_attention".to_string(), "true".to_string())]);
    local_data(spec, ConnectionMode::Http, ordinary_host())
}

/// On a registration of several ranks the engine's report is mapped to the
/// rank's space before the config's keys leave it: a DP label the config also
/// names still sizes the window share the record carries, so a later
/// refresh reads that share, not the whole engine window, as the engine's.
#[tokio::test]
async fn a_dp_label_the_config_also_sets_still_sizes_the_recorded_window_share() {
    let mut data = ranked_local_http_data();
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
    let mut data = ranked_local_http_data();
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

/// A registration refused for a worker whose address carries userinfo
/// names the address through the mask: the credential is in no refusal
/// text (the job log prints the refusal beside the masked worker).
#[tokio::test]
async fn a_refusal_names_the_workers_address_through_the_mask() {
    let mut spec = WorkerSpec::new("http://user:secret@worker:8000");
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.overload = OverloadUpdate {
        waiting_requests: Some(0),
        ..Default::default()
    };
    let data = local_data(spec, ConnectionMode::Http, ordinary_host());
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    let text = CreateLocalWorkerStep
        .execute(&mut ctx)
        .await
        .expect_err("the overload override is refused")
        .to_string();
    assert!(text.contains("http://***@worker:8000"), "{text}");
    assert!(
        !text.contains("secret") && !text.contains("user:"),
        "{text}"
    );
}

/// The failure of creation for a worker that reports these labels.
async fn failure_of_a_worker_reporting(labels: &[(&str, &str)]) -> String {
    let mut data = local_data(
        WorkerSpec::new("http://worker:8000"),
        ConnectionMode::Http,
        ordinary_host(),
    );
    data.discovered_labels = labels_of(labels);
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);
    CreateLocalWorkerStep
        .execute(&mut ctx)
        .await
        .unwrap_err()
        .to_string()
}

/// A `constrained_decoding` value the worker reports with a control
/// character in it is refused in a sentence that names the label and
/// does not repeat the value. A value on one line that names no policy
/// is refused by name, as before.
#[tokio::test]
async fn a_reported_decoding_policy_with_a_control_character_fails_the_registration() {
    assert_eq!(
        failure_of_a_worker_reporting(&[
            ("served_model_name", "m"),
            ("constrained_decoding", "sometimes\nINJECTED"),
        ])
        .await,
        "Step failed: create_worker - Worker http://worker:8000: \
         constrained_decoding must not contain control characters"
    );
    assert_eq!(
        failure_of_a_worker_reporting(&[
            ("served_model_name", "m"),
            ("constrained_decoding", "sometimes"),
        ])
        .await,
        "Step failed: create_worker - Worker http://worker:8000: constrained_decoding must be \
         'requested' or 'always', not 'sometimes'"
    );
}

/// An empty label value is no KV field, as an empty spec value is: a
/// registration carrying `kv_connector: ""` or `kv_role: ""` leaves the
/// field unset for a later report to fill, instead of holding `Some("")`
/// that nothing fills.
#[test]
fn an_empty_kv_label_value_is_no_field() {
    let spec = WorkerSpec::new("http://worker:8080");
    let mut labels = HashMap::from([
        ("kv_connector".to_string(), String::new()),
        ("kv_role".to_string(), String::new()),
        ("kv_engine_id".to_string(), String::new()),
    ]);
    assert_eq!(
        take_kv_transfer_metadata(&spec, &mut labels),
        (None, None, None)
    );
    assert!(labels.is_empty(), "the keys leave the labels all the same");
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
        dp_rank_expansion(Some(&multi)),
        vec![Some((0, 3)), Some((1, 3)), Some((2, 3))]
    );

    // Width 1 is a capability-only advertisement: no placement exists,
    // and a rank-0 pin would still alter engine scheduling.
    let single = DpInfo {
        dp_size: 1,
        model_id: "m".to_string(),
    };
    assert_eq!(dp_rank_expansion(Some(&single)), vec![None]);
    assert_eq!(dp_rank_expansion(None), vec![None]);
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
    let refused = [
        (RuntimeType::Vllm, WorkerType::Prefill),
        (RuntimeType::Vllm, WorkerType::Decode),
        (RuntimeType::Vllm, WorkerType::Encode),
        (RuntimeType::Sglang, WorkerType::Prefill),
        (RuntimeType::Sglang, WorkerType::Decode),
        (RuntimeType::Sglang, WorkerType::Encode),
    ];
    for (runtime, worker_type) in refused {
        let err = validate_zmq_worker_type(
            ConnectionMode::Zmq,
            runtime,
            worker_type,
            "ipc:///tmp/smg-zmq/ts0.ipc",
        )
        .expect_err("a role the ZMQ link cannot serve must be rejected");
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

    // Every runtime with a ZMQ link serves regular workers. One supports
    // disaggregation on that link. The other transports serve every role.
    for runtime in [RuntimeType::Vllm, RuntimeType::TokenSpeed] {
        validate_zmq_worker_type(
            ConnectionMode::Zmq,
            runtime,
            WorkerType::Regular,
            "ipc:///tmp/smg-zmq/ts0.ipc",
        )
        .expect("ZMQ regular workers are the supported case");
    }
    for worker_type in [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode] {
        validate_zmq_worker_type(
            ConnectionMode::Zmq,
            RuntimeType::TokenSpeed,
            worker_type,
            "ipc:///tmp/smg-zmq/ts0.ipc",
        )
        .expect("a runtime that supports disaggregation serves every role");
    }
    for mode in [ConnectionMode::Http, ConnectionMode::Grpc] {
        for worker_type in [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode] {
            validate_zmq_worker_type(mode, RuntimeType::Sglang, worker_type, "grpc://worker:8080")
                .expect("non-ZMQ transports serve every worker type");
        }
    }
}

/// A ZMQ engine group given a count becomes one worker per rank, as a gRPC
/// engine does, on a host whose settings name nothing of it, and the rank
/// workers hold one connection slot: the group's engines dial one socket set.
#[tokio::test]
async fn a_zmq_group_given_a_count_becomes_rank_workers_sharing_one_connection() {
    let mut spec = WorkerSpec::new("ipc:///tmp/smg-zmq/ts0.ipc");
    spec.runtime_type = RuntimeType::TokenSpeed;
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
    spec.dp_size = Some(3);
    let mut data = local_data(spec, ConnectionMode::Zmq, ordinary_host());
    data.dp_info = Some(DpInfo {
        dp_size: 3,
        model_id: "m".to_string(),
    });
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);

    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );

    let workers = ctx.data.actual_workers.as_ref().unwrap();
    let ranks: Vec<&BasicWorker> = workers
        .iter()
        .map(|worker| worker.as_any().downcast_ref::<BasicWorker>().unwrap())
        .collect();
    assert_eq!(
        ranks.iter().map(|rank| rank.url()).collect::<Vec<_>>(),
        [
            "ipc:///tmp/smg-zmq/ts0.ipc@0",
            "ipc:///tmp/smg-zmq/ts0.ipc@1",
            "ipc:///tmp/smg-zmq/ts0.ipc@2",
        ]
    );
    for (index, rank) in ranks.iter().enumerate() {
        assert_eq!(rank.dp_rank(), Some(index));
        assert_eq!(rank.base_url(), "ipc:///tmp/smg-zmq/ts0.ipc");
        assert_eq!(rank.metadata.zmq_engine_count(), Some(3));
        assert!(Arc::ptr_eq(
            &rank.engine_connection,
            &ranks[0].engine_connection
        ));
    }
}

/// The rank workers of a group register at `{base}@{rank}`: a second
/// create-only registration of the group finds them there and is refused.
#[tokio::test]
async fn create_only_refuses_a_group_whose_rank_workers_are_registered() {
    let base = "ipc:///tmp/smg-zmq/ts0.ipc";
    let host = ordinary_host();
    let group = |host: Arc<ProbingHost>| {
        let mut spec = WorkerSpec::new(base);
        spec.runtime_type = RuntimeType::TokenSpeed;
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        spec.dp_size = Some(2);
        let mut data = local_data(spec, ConnectionMode::Zmq, host);
        data.dp_info = Some(DpInfo {
            dp_size: 2,
            model_id: "m".to_string(),
        });
        data.registration_mode = WorkerRegistrationMode::CreateOnly;
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    };

    let mut first = group(Arc::clone(&host));
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut first).await.unwrap(),
        StepResult::Success
    );
    for worker in first.data.actual_workers.take().unwrap() {
        host.worker_registry().register(worker).unwrap();
    }

    let mut second = group(host);
    let error = CreateLocalWorkerStep
        .execute(&mut second)
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Step failed: create_worker - Worker ipc:///tmp/smg-zmq/ts0.ipc already exists"
    );
    assert!(second.data.actual_workers.is_none());
}

/// A registration that read no shape, where the address's rank workers
/// stand: a server info that failed to read leaves none, and the rank
/// workers are at `{base}@{rank}`, where the base address finds no worker.
/// A create is refused as a duplicate and any other registration by name.
/// Neither builds a plain worker beside the ranks.
#[tokio::test]
async fn a_registration_with_no_shape_builds_no_plain_worker_beside_standing_ranks() {
    let base = "http://worker:8000";
    let host = ordinary_host();
    for rank in 0..2 {
        let worker = BasicWorkerBuilder::new(base)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .dp_config(rank, 2)
            .build();
        host.worker_registry().register(Arc::new(worker)).unwrap();
    }
    let shapeless = |mode: WorkerRegistrationMode| {
        let mut spec = WorkerSpec::new(base);
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        let mut data = local_data(spec, ConnectionMode::Http, Arc::clone(&host));
        data.registration_mode = mode;
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    };

    let mut create = shapeless(WorkerRegistrationMode::CreateOnly);
    let error = CreateLocalWorkerStep
        .execute(&mut create)
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Step failed: create_worker - Worker http://worker:8000 already exists"
    );
    assert!(create.data.actual_workers.is_none());

    let mut upsert = shapeless(WorkerRegistrationMode::Upsert);
    let error = CreateLocalWorkerStep
        .execute(&mut upsert)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("stands as 2 rank workers and this registration read no data-parallel ranks"),
        "{error}"
    );
    assert!(upsert.data.actual_workers.is_none());
}

/// A registration over what stands replaces every worker of the address or
/// changes nothing. One that reads two ranks where a plain worker stands
/// (its first server info failed to read) is refused by name, and so is one
/// that reads two ranks where four stand: each would leave a worker beside
/// its own. One that reads four where two stand replaces both and adds two.
#[tokio::test]
async fn a_registration_over_what_stands_leaves_no_worker_beside_its_own() {
    let base = "http://worker:8000";
    let standing_ranks = |size: usize| {
        let host = ordinary_host();
        for rank in 0..size {
            let worker = BasicWorkerBuilder::new(base)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("m"))
                .dp_config(rank, size)
                .build();
            host.worker_registry().register(Arc::new(worker)).unwrap();
        }
        host
    };
    let plain = ordinary_host();
    let worker = BasicWorkerBuilder::new(base)
        .worker_type(WorkerType::Regular)
        .model(ModelCard::new("m"))
        .build();
    plain.worker_registry().register(Arc::new(worker)).unwrap();

    for (host, read, said) in [
        (
            plain,
            2,
            Some("stands as one worker and this registration read 2 data-parallel ranks"),
        ),
        (
            standing_ranks(4),
            2,
            Some("stands as 4 rank workers and this registration read 2 data-parallel ranks"),
        ),
        (standing_ranks(2), 4, None),
    ] {
        let mut spec = WorkerSpec::new(base);
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        let mut data = local_data(spec, ConnectionMode::Http, host);
        data.dp_info = Some(DpInfo {
            dp_size: read,
            model_id: "m".to_string(),
        });
        data.registration_mode = WorkerRegistrationMode::Upsert;
        let mut upsert = WorkflowContext::new(WorkflowInstanceId::new(), data);

        let result = CreateLocalWorkerStep.execute(&mut upsert).await;

        let built = upsert.data.actual_workers.as_ref().map(Vec::len);
        match said {
            Some(said) => {
                let error = result.unwrap_err().to_string();
                assert!(error.contains(said), "{error}");
                assert!(error.contains("nothing is changed"), "{error}");
                assert_eq!(built, None, "{said}");
            }
            None => {
                assert_eq!(result.unwrap(), StepResult::Success);
                assert_eq!(built, Some(read), "a wider engine replaces what stands");
            }
        }
    }
}

/// What this node registered is held by its address, the registry step's
/// own key, and not by the rank a worker states. Two workers built from the
/// address alone stand at `{base}@{rank}` and state no rank. An upsert that
/// reads the same two ranks replaces both. A registration that reads no
/// ranks is refused, since its plain worker would stand beside them, and
/// the refusal counts them as rank workers.
#[tokio::test]
async fn standing_workers_are_held_by_their_addresses_not_their_stated_ranks() {
    let base = "http://worker:8000";
    let host = ordinary_host();
    for rank in 0..2 {
        let imported = BasicWorkerBuilder::new(format!("{base}@{rank}"))
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .build();
        assert_eq!(imported.dp_rank(), None, "built from the address alone");
        // Registered here: this node owns them.
        host.worker_registry().register(Arc::new(imported)).unwrap();
    }
    let upsert = |read: Option<usize>| {
        let mut spec = WorkerSpec::new(base);
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        let mut data = local_data(spec, ConnectionMode::Http, Arc::clone(&host));
        data.dp_info = read.map(|dp_size| DpInfo {
            dp_size,
            model_id: "m".to_string(),
        });
        data.registration_mode = WorkerRegistrationMode::Upsert;
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    };

    let mut same_shape = upsert(Some(2));
    assert_eq!(
        CreateLocalWorkerStep
            .execute(&mut same_shape)
            .await
            .unwrap(),
        StepResult::Success
    );
    let built: Vec<String> = same_shape
        .data
        .actual_workers
        .unwrap()
        .iter()
        .map(|worker| worker.url().to_string())
        .collect();
    assert_eq!(built, ["http://worker:8000@0", "http://worker:8000@1"]);

    let mut no_size = upsert(None);
    let error = CreateLocalWorkerStep
        .execute(&mut no_size)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(
            "stands as 2 rank workers and this registration read no data-parallel ranks: it \
             would leave http://worker:8000@0 and 1 more beside its own, so nothing is changed"
        ),
        "{error}"
    );
    assert!(no_size.data.actual_workers.is_none());
}

/// A worker that stands under another spelling of the address is one the
/// registry step would not replace: an upsert under the other spelling is
/// refused, and the refusal names the worker that would stay. The two
/// spellings differ in the form of an IPv6 address, or in the host's letter
/// case alone.
#[tokio::test]
async fn an_upsert_under_another_spelling_of_a_standing_address_is_refused() {
    for (stands, asked) in [
        ("http://[::1]:8000", "http://[0:0:0:0:0:0:0:1]:8000"),
        ("http://Worker:8000", "http://worker:8000"),
    ] {
        let host = ordinary_host();
        let standing = BasicWorkerBuilder::new(stands)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .build();
        host.worker_registry().register(Arc::new(standing)).unwrap();
        let mut spec = WorkerSpec::new(asked);
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        let mut data = local_data(spec, ConnectionMode::Http, host);
        data.registration_mode = WorkerRegistrationMode::Upsert;
        let mut upsert = WorkflowContext::new(WorkflowInstanceId::new(), data);

        let error = CreateLocalWorkerStep
            .execute(&mut upsert)
            .await
            .unwrap_err()
            .to_string();

        assert!(
            error.contains(&format!(
                "stands as one worker and this registration read no data-parallel ranks: it \
                 would leave {stands} beside its own, so nothing is changed"
            )),
            "{error}"
        );
        assert!(upsert.data.actual_workers.is_none());
    }
}

/// A peer's worker, imported from its published state, holds no registration
/// of this node back: the node takes the addresses it is configured with.
/// Imports with no spec stand at `{base}@{rank}` and state no rank, and a
/// same-shape upsert builds over both. A peer's rank workers do not stop a
/// registration that read no ranks, and a peer's plain worker does not stop
/// the upsert of this node's own ranks. In those two the engine stands twice
/// until the peer's state changes, as it did before the rule.
#[tokio::test]
async fn a_peers_imported_workers_hold_no_registration_back() {
    use crate::registry::RemoteWorkerState;

    let base = "http://worker:8000";
    let import = |host: &Arc<ProbingHost>, url: &str| {
        let registry = host.worker_registry();
        registry.on_remote_worker_state(&RemoteWorkerState {
            worker_id: format!("peer-{url}"),
            model_id: "m".to_string(),
            url: url.to_string(),
            health: true,
            version: 1,
            spec: Vec::new(),
        });
        let id = registry.get_id_by_url(url).expect("the import");
        assert_eq!(registry.origin_of(&id), Some(WorkerOrigin::Mesh));
    };
    let built_by_upsert = |host: &Arc<ProbingHost>, read: Option<usize>| {
        let mut spec = WorkerSpec::new(base);
        spec.models = WorkerModels::Single(Box::new(ModelCard::new("m")));
        let mut data = local_data(spec, ConnectionMode::Http, Arc::clone(host));
        data.dp_info = read.map(|dp_size| DpInfo {
            dp_size,
            model_id: "m".to_string(),
        });
        data.registration_mode = WorkerRegistrationMode::Upsert;
        let mut upsert = WorkflowContext::new(WorkflowInstanceId::new(), data);
        async move {
            let result = CreateLocalWorkerStep.execute(&mut upsert).await;
            assert_eq!(result.unwrap(), StepResult::Success);
            let built = upsert.data.actual_workers.expect("the built workers");
            built
                .iter()
                .map(|worker| worker.url().to_string())
                .collect::<Vec<String>>()
        }
    };
    let ranks = ["http://worker:8000@0", "http://worker:8000@1"];

    let peers_ranks = ordinary_host();
    for url in ranks {
        import(&peers_ranks, url);
        let imported = peers_ranks.worker_registry().get_by_url(url).unwrap();
        assert_eq!(imported.dp_rank(), None, "built from the address alone");
    }
    assert_eq!(built_by_upsert(&peers_ranks, Some(2)).await, ranks);
    assert_eq!(built_by_upsert(&peers_ranks, None).await, [base]);

    let own_ranks = ordinary_host();
    for rank in 0..2 {
        let worker = BasicWorkerBuilder::new(base)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .dp_config(rank, 2)
            .build();
        own_ranks
            .worker_registry()
            .register(Arc::new(worker))
            .unwrap();
    }
    import(&own_ranks, base);
    assert_eq!(built_by_upsert(&own_ranks, Some(2)).await, ranks);
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

/// The rank's labels when every label was discovered (none set by hand).
fn discovered(
    engine: &HashMap<String, String>,
    dp: Option<(usize, usize)>,
) -> HashMap<String, String> {
    labels_for_rank(engine, dp, &HashMap::new())
}

/// A window the operator set by hand on the worker is the worker's only
/// window, whichever name the operator used: the merge drops a
/// discovered window under the other name (the reader prefers
/// `max_num_seqs`, which would otherwise override the operator's
/// figure), on an attention-DP engine, a plain data-parallel one and an
/// unexpanded worker alike; the other labels merge as before, the
/// operator's on top.
#[test]
fn a_window_the_operator_set_by_hand_is_the_workers_only_window() {
    let by_hand = labels_of(&[("max_running_requests", "512"), ("served_model_name", "m")]);
    let replicas = labels_of(&[
        ("max_num_seqs", "256"),
        ("dp_size", "4"),
        ("served_model_name", "engine"),
    ]);
    let merged = merge_labels(&replicas, &by_hand);
    assert_eq!(
        merged,
        labels_of(&[
            ("max_running_requests", "512"),
            ("dp_size", "4"),
            ("served_model_name", "m"),
        ])
    );
    assert_eq!(labels_for_rank(&merged, Some((1, 4)), &by_hand), merged);
    assert_eq!(labels_for_rank(&merged, None, &by_hand), merged);

    let by_hand = labels_of(&[("max_num_seqs", "512")]);
    let attention = labels_of(&[("max_running_requests", "16384"), ("attn_dp_size", "8")]);
    let merged = merge_labels(&attention, &by_hand);
    assert_eq!(merged.get("max_running_requests"), None);
    let rank = labels_for_rank(&merged, Some((0, 8)), &by_hand);
    assert_eq!(rank.get("max_num_seqs").map(String::as_str), Some("512"));

    let nothing_by_hand = labels_of(&[("served_model_name", "m")]);
    let both = labels_of(&[("max_running_requests", "2048"), ("max_num_seqs", "256")]);
    let merged = merge_labels(&both, &nothing_by_hand);
    assert_eq!(
        merged.get("max_running_requests").map(String::as_str),
        Some("2048")
    );
    assert_eq!(merged.get("max_num_seqs").map(String::as_str), Some("256"));

    // The context window is a family of its own under the same rule: a
    // limit set by hand drops the engine's figure under the names the
    // reader prefers to it, and leaves the running window alone.
    let by_hand = labels_of(&[("max_model_len", "2048")]);
    let engine = labels_of(&[
        ("max_context_length", "131072"),
        ("context_length", "131072"),
        ("max_num_seqs", "256"),
    ]);
    assert_eq!(
        merge_labels(&engine, &by_hand),
        labels_of(&[("max_model_len", "2048"), ("max_num_seqs", "256")])
    );
}

/// An engine that splits its running window across its attention-DP
/// workers advertises the whole engine's window; each of its rank
/// workers gets its share, whichever way the engine's release spells
/// the split; the other labels ride along. Replicas of attention-DP
/// groups (`dp_size` next to a resolved `attn_dp_size`) each run the
/// whole window: the share is by the attention-DP width, not by the
/// dp-aware width of every rank.
#[test]
fn an_attention_dp_engines_window_is_each_ranks_share() {
    let engine = labels_of(&[
        ("max_running_requests", "16384"),
        ("enable_dp_attention", "true"),
        ("dp_size", "8"),
    ]);
    let rank = discovered(&engine, Some((3, 8)));
    assert_eq!(
        rank.get("max_running_requests").map(String::as_str),
        Some("2048")
    );
    assert_eq!(rank.get("dp_size").map(String::as_str), Some("8"));
    let newer = labels_of(&[("max_running_requests", "16384"), ("attn_dp_size", "8")]);
    assert_eq!(
        discovered(&newer, Some((0, 8)))
            .get("max_running_requests")
            .map(String::as_str),
        Some("2048")
    );
    let replicas_of_groups = labels_of(&[
        ("max_running_requests", "4096"),
        ("dp_size", "2"),
        ("attn_dp_size", "4"),
    ]);
    assert_eq!(
        discovered(&replicas_of_groups, Some((5, 8)))
            .get("max_running_requests")
            .map(String::as_str),
        Some("1024")
    );
}

/// Plain data parallelism runs the whole flag on every rank, an engine
/// whose figure is per rank already keeps it, and a worker that stands
/// for the whole engine (no expansion, or a width of one) keeps the
/// whole window.
#[test]
fn a_per_rank_window_and_an_unexpanded_worker_keep_their_labels() {
    let replicas = labels_of(&[("max_running_requests", "2048"), ("dp_size", "8")]);
    assert_eq!(discovered(&replicas, Some((1, 8))), replicas);
    let per_rank = labels_of(&[("max_num_seqs", "256"), ("dp_size", "4")]);
    assert_eq!(discovered(&per_rank, Some((0, 4))), per_rank);
    let attention = labels_of(&[
        ("max_running_requests", "16384"),
        ("enable_dp_attention", "true"),
    ]);
    assert_eq!(discovered(&attention, None), attention);
    assert_eq!(discovered(&attention, Some((0, 1))), attention);
}

/// A configured or discovered `constrained_decoding` label must name a
/// policy; anything else fails the registration with the reason.
#[test]
fn a_constrained_decoding_label_must_name_a_policy() {
    let mut labels = HashMap::new();
    let check = |labels: &HashMap<String, String>, step: &str| {
        validate_constrained_decoding_label(labels, "http://w:8000", step)
    };
    assert!(check(&labels, "create_worker").is_ok());
    labels.insert("constrained_decoding".to_string(), "always".to_string());
    assert!(check(&labels, "create_worker").is_ok());
    labels.insert("constrained_decoding".to_string(), "sometimes".to_string());
    let message = check(&labels, "create_worker").unwrap_err().to_string();
    assert!(message.contains("http://w:8000"), "{message}");
    assert!(message.contains("'requested' or 'always'"), "{message}");
    let update = check(&labels, "update_worker_properties").unwrap_err();
    assert!(
        format!("{update:?}").contains("update_worker_properties"),
        "{update}"
    );
}
