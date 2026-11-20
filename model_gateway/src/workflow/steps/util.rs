//! The gateway's connector under the worker crate's registration steps: a
//! mock gRPC engine for the cases that dial one, and the cases of the reach,
//! the runtime detection and the metadata read on the gateway's own
//! connector and a served engine.

use std::sync::Arc;

use smg_worker::identity::basic::RuntimeType;

use crate::routers::grpc::backend_client::BackendConnector;

/// A mock gRPC engine on an ephemeral loopback port for the probes' tests,
/// answering its health check as its overrides say (healthy until told
/// otherwise): its `grpc://` URL and those overrides.
#[cfg(test)]
pub(crate) async fn mock_grpc_engine() -> (String, Arc<mock_worker::overrides::WorkerOverrides>) {
    mock_grpc_engine_on(mock_worker::config::SchedulerWire::TokenSpeed).await
}

/// [`mock_grpc_engine`] speaking `wire`'s proto, for the reads that pick a
/// client by the runtime's name: the engine's model info and server info
/// answer as its overrides say (served until told otherwise).
#[cfg(test)]
pub(crate) async fn mock_grpc_engine_on(
    wire: mock_worker::config::SchedulerWire,
) -> (String, Arc<mock_worker::overrides::WorkerOverrides>) {
    serve_mock(wire, Vec::new()).await
}

/// [`mock_grpc_engine_on`] whose worker states `value` of `name` under
/// `unit` in every load report it makes: the engine's `grpc://` URL.
#[cfg(test)]
pub(crate) async fn mock_grpc_engine_stating(
    wire: mock_worker::config::SchedulerWire,
    name: &str,
    unit: &str,
    value: f64,
) -> String {
    let stated = smg_grpc_client::common_proto::NamedValue {
        name: name.to_owned(),
        unit: unit.to_owned(),
        value,
        age_ms: None,
    };
    let (url, _) = serve_mock(wire, vec![stated]).await;
    url
}

/// A mock gRPC engine on `wire` whose one worker states `values` by name in
/// every load report it makes.
#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "test helper: the mock engine's serve task runs for the test process's lifetime"
)]
async fn serve_mock(
    wire: mock_worker::config::SchedulerWire,
    values: Vec<smg_grpc_client::common_proto::NamedValue>,
) -> (String, Arc<mock_worker::overrides::WorkerOverrides>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the mock engine");
    let port = listener
        .local_addr()
        .expect("the mock engine's address")
        .port();
    let config = Arc::new(mock_worker::config::Config {
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        grpc_wire: wire,
        model_id: "probe-model".to_string(),
        tokenizer_path: "probe-model".to_string(),
        // The mock names a gRPC worker by its port.
        worker_load_values: [(format!("grpc:{port}"), values)].into(),
        ..mock_worker::config::Config::default()
    });
    // A port an earlier test's engine served on keeps that engine's
    // overrides; this engine starts from fresh ones.
    let overrides = mock_worker::overrides::fresh_for_port(port);
    tokio::spawn(mock_worker::grpc::serve_with_listener(config, listener));
    (format!("grpc://127.0.0.1:{port}"), overrides)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use smg_worker::{
        identity::{basic::ConnectionMode, endpoint::Endpoint},
        jobs::WorkerJobAsker,
        workflow::{
            discover_engine_labels, grpc_health_answer, grpc_reachable_url, http_health_url,
            try_grpc_reachable,
        },
    };

    use super::*;
    use crate::server::http_client;

    /// The worker crate's gRPC metadata read on the gateway's connector.
    async fn fetch_grpc_metadata(
        url: &str,
        runtime_type: &str,
        strict: bool,
    ) -> Result<(HashMap<String, String>, String), String> {
        let connector = BackendConnector::shared();
        discover_engine_labels(
            &http_client(),
            connector.as_ref(),
            url,
            ConnectionMode::Grpc,
            runtime_type,
            None,
            strict,
        )
        .await
        .map(|(labels, runtime)| (labels, runtime.unwrap_or_default()))
    }

    /// An endpoint that cannot be reached at the transport level must
    /// short-circuit before the per-runtime fan-out, so one unreachable worker
    /// costs one dial rather than five.
    ///
    /// TCP port 0 cannot be owned by a listener: binding it asks the OS for an
    /// ephemeral nonzero port. This makes the local transport failure
    /// deterministic without a bind-then-drop port-reuse race.
    /// The fan-out aggregate can only be constructed after all runtime probes
    /// run, so the transport error proves the gate returned first.
    #[tokio::test]
    async fn unreachable_transport_short_circuits_the_runtime_fanout() {
        let connector = BackendConnector::shared();
        let err = try_grpc_reachable(connector.as_ref(), "grpc://127.0.0.1:0", 1)
            .await
            .expect_err("closed local endpoint should not be reachable");

        assert!(
            err.starts_with("gRPC connection"),
            "expected the transport-gate error, got: {err}"
        );
        assert!(
            !err.contains("gRPC not reachable"),
            "fan-out aggregate was produced, so the gate did not short-circuit: {err}"
        );
        for runtime in ["sglang", "vllm", "trtllm", "mlx", "tokenspeed"] {
            assert!(
                !err.contains(runtime),
                "error names {runtime}, so a per-runtime probe ran: {err}"
            );
        }
    }

    /// An engine that answers its health check but reports itself not healthy
    /// (its model still loading, say) is not reachable yet: the runtime it
    /// speaks answers with the engine's reason, the fan-out fails with it,
    /// and passes once the engine reports healthy. This is the worker crate's
    /// reach on the gateway's connector and a served engine; the single
    /// check's own text is that crate's case.
    #[tokio::test]
    async fn an_engine_that_reports_not_healthy_is_not_reachable_yet() {
        let (url, engine) = mock_grpc_engine().await;
        engine.set_serving(false);
        let connector = BackendConnector::shared();

        let answer = grpc_health_answer(connector.as_ref(), &url, 5, RuntimeType::TokenSpeed)
            .await
            .expect("the runtime answers its health RPC whether or not the engine is healthy");
        assert!(!answer.healthy);
        assert_eq!(answer.message, "not serving");
        grpc_health_answer(connector.as_ref(), &url, 5, RuntimeType::Vllm)
            .await
            .expect_err("another runtime's service does not answer there");
        let err = try_grpc_reachable(connector.as_ref(), &url, 5)
            .await
            .expect_err("the fan-out must not count a not-healthy answer as reachable");
        assert!(
            err.contains(
                "tokenspeed=gRPC health check reports the engine not healthy: not serving"
            ),
            "the fan-out error does not carry the engine's reason: {err}"
        );

        engine.set_serving(true);
        try_grpc_reachable(connector.as_ref(), &url, 5)
            .await
            .expect("a healthy engine is reachable");
    }

    /// A runtime that answers its health RPC with `healthy: false` (its
    /// engine still loading) is identified all the same, by the worker
    /// crate's step on the gateway's connector; an endpoint nobody answers
    /// on names every runtime's reason.
    #[tokio::test]
    async fn grpc_detection_identifies_a_runtime_that_reports_not_healthy() {
        use openai_protocol::worker::WorkerSpec;
        use smg_worker::{
            identity::basic::ConnectionMode,
            jobs::WorkerRegistrationMode,
            workflow::{DetectRuntimeStep, WorkerKind},
        };
        use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

        use crate::{
            app_context::testing::create_test_app_context,
            workflow::steps::create_worker_workflow_data,
        };

        async fn detect(url: &str, timeout_secs: u64) -> Result<Option<String>, String> {
            let mut spec = WorkerSpec::new(url);
            spec.health.timeout_secs = Some(timeout_secs);
            let mut data = create_worker_workflow_data(
                spec,
                WorkerRegistrationMode::Upsert,
                create_test_app_context(),
                WorkerJobAsker::Api,
            );
            data.worker_kind = Some(WorkerKind::Local);
            data.connection_mode = Some(ConnectionMode::Grpc);
            let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
            DetectRuntimeStep
                .execute(&mut context)
                .await
                .map(|_| context.data.detected_runtime_type.clone())
                .map_err(|error| error.to_string())
        }

        let (url, engine) = mock_grpc_engine().await;
        engine.set_serving(false);
        assert_eq!(
            detect(&url, 5).await.unwrap().as_deref(),
            Some("tokenspeed")
        );

        let err = detect("grpc://127.0.0.1:0", 1)
            .await
            .expect_err("nothing answers on a closed endpoint");
        for runtime in ["sglang", "vllm", "trtllm", "tokenspeed", "mlx"] {
            assert!(
                err.contains(&format!("{runtime}: gRPC connection")),
                "the error does not carry {runtime}'s reason: {err}"
            );
        }
    }

    /// For an address the door takes, the host and port the probes dial are
    /// the ones every line names. Where a client and the gateway's own type
    /// read an address apart (a caller's rank suffix, whose digits a client
    /// dials as the host), the door refuses it.
    #[test]
    fn the_probes_dial_the_host_a_line_names() {
        use crate::config::validate_worker_url;
        let dialled = |address: &str| {
            let health = http_health_url(address).expect(address);
            let health = url::Url::parse(&health).expect(address);
            format!(
                "{}:{}",
                health.host_str().expect(address),
                health.port_or_known_default().expect(address)
            )
        };
        for (address, host_and_port) in [
            ("http://user:s3cret@127.0.0.1:8000", "127.0.0.1:8000"),
            ("http://worker:3000/v1", "worker:3000"),
            ("https://[::1]:8443", "[::1]:8443"),
        ] {
            validate_worker_url(address).expect(address);
            assert_eq!(dialled(address), host_and_port);
            let named = Endpoint::redacted_str(address);
            assert!(named.contains(host_and_port), "{address}: {named}");
        }
        for (address, named, dialled_at) in [
            (
                "http://127.0.0.1:35099@2130706433",
                "127.0.0.1:35099",
                "127.0.0.1:80",
            ),
            (
                "http://user:s3cret@127.0.0.1:52605@2130706433",
                "127.0.0.1:52605",
                "127.0.0.1:80",
            ),
            (
                "http://user:s3cret@127.0.0.1:57697@3",
                "127.0.0.1:57697",
                "0.0.0.3:80",
            ),
        ] {
            assert!(Endpoint::redacted_str(address).contains(named), "{address}");
            assert_eq!(dialled(address), dialled_at, "{address}");
            assert!(
                validate_worker_url(address).is_err(),
                "{address} is refused at the door"
            );
        }
        // The gRPC channel reads the digits as its host too.
        let address = "grpc://127.0.0.1:61845@2130706433";
        let channel: http::Uri = grpc_reachable_url(address)
            .expect(address)
            .parse()
            .expect(address);
        assert_eq!(channel.host(), Some("2130706433"));
        assert!(Endpoint::redacted_str(address).contains("127.0.0.1:61845"));
        assert!(
            validate_worker_url(address).is_err(),
            "{address} is refused at the door"
        );
        // To the gateway's type these two hold no userinfo, only a host and
        // its rank: a token alone, a user and a password of digits. A line
        // would name each whole.
        for address in [
            "http://s3cret-token@8080",
            "http://s3cret-user:8080@2130706433",
        ] {
            assert_eq!(Endpoint::redacted_str(address), address);
            assert!(
                validate_worker_url(address).is_err(),
                "{address} is refused at the door"
            );
        }
    }

    fn label<'a>(labels: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
        labels.get(key).map(String::as_str)
    }

    /// The crate's mock gRPC engine of each proto, as the probes' tests use
    /// it: its model id names its model, tokenizer and served name.
    const MOCK_MODEL: &str = "probe-model";
    /// The mock engine's context length, as it reports it.
    const MOCK_CONTEXT: &str = "32768";

    /// The gRPC read of each proto: the model info's fields become labels,
    /// the server info adds its own (a vLLM engine's KV connector, an SGLang
    /// engine's version), and the runtime read is the one asked for.
    #[tokio::test]
    async fn the_grpc_read_takes_the_model_info_and_the_server_info_of_each_proto() {
        use mock_worker::config::SchedulerWire;

        use crate::workflow::steps::util::mock_grpc_engine_on;

        let (vllm, engine) = mock_grpc_engine_on(SchedulerWire::Vllm).await;
        let server_label = with_server_label(&engine, "vllm");
        let (labels, runtime) = fetch_grpc_metadata(&vllm, "vllm", true)
            .await
            .expect("the vLLM read");
        assert_eq!(runtime, "vllm");
        assert_eq!(label(&labels, "model_path"), Some(MOCK_MODEL));
        assert_eq!(label(&labels, "tokenizer_path"), Some(MOCK_MODEL));
        assert_eq!(label(&labels, "served_model_name"), Some(MOCK_MODEL));
        assert_eq!(label(&labels, "max_context_length"), Some(MOCK_CONTEXT));
        assert_eq!(label(&labels, server_label), Some("MooncakeConnector"));

        let (sglang, engine) = mock_grpc_engine_on(SchedulerWire::Sglang).await;
        let server_label = with_server_label(&engine, "sglang");
        let (labels, runtime) = fetch_grpc_metadata(&sglang, "sglang", true)
            .await
            .expect("the SGLang read");
        assert_eq!(runtime, "sglang");
        assert_eq!(label(&labels, "model_path"), Some(MOCK_MODEL));
        assert_eq!(label(&labels, "tokenizer_path"), Some(MOCK_MODEL));
        assert_eq!(label(&labels, "served_model_name"), Some(MOCK_MODEL));
        assert_eq!(label(&labels, "max_context_length"), Some(MOCK_CONTEXT));
        assert_eq!(label(&labels, server_label), Some("mock"));
    }

    /// A server-info label each proto's engine reports when its server info
    /// is served, so a read without one shows the gap. The vLLM mock sends
    /// an empty KV connector, which the flattening drops, so the connector
    /// is patched in first; the SGLang mock sends its version. The served
    /// read's test takes its label here too, so the gap proofs' label is
    /// one a served read is shown to carry.
    fn with_server_label(
        engine: &mock_worker::overrides::WorkerOverrides,
        runtime: &str,
    ) -> &'static str {
        if runtime == "vllm" {
            let mut patch = serde_json::Map::new();
            patch.insert(
                "kv_connector".to_string(),
                serde_json::Value::String("MooncakeConnector".to_string()),
            );
            engine.patch_info(&patch);
            "kv_connector"
        } else {
            "version"
        }
    }

    /// An engine without a server info (an `Unimplemented` answer) keeps its
    /// model labels on a plain read and on a strict one, and has no server
    /// label: the call is a gap, not a failure, on both protos.
    #[tokio::test]
    async fn an_engine_without_server_info_keeps_its_model_labels_on_a_plain_and_a_strict_read() {
        use mock_worker::{config::SchedulerWire, overrides::InfoAnswer};

        use crate::workflow::steps::util::mock_grpc_engine_on;

        let mut engines = Vec::new();
        for (wire, runtime) in [
            (SchedulerWire::Vllm, "vllm"),
            (SchedulerWire::Sglang, "sglang"),
        ] {
            let (url, engine) = mock_grpc_engine_on(wire).await;
            let server_label = with_server_label(&engine, runtime);
            engine.set_server_info(InfoAnswer::Unimplemented);
            engines.push((url, runtime, server_label));
        }
        for (url, runtime, server_label) in &engines {
            for strict in [false, true] {
                let (labels, _) = fetch_grpc_metadata(url, runtime, strict)
                    .await
                    .unwrap_or_else(|error| panic!("{runtime} strict={strict}: {error}"));
                assert_eq!(
                    label(&labels, "model_path"),
                    Some(MOCK_MODEL),
                    "{runtime} strict={strict}"
                );
                assert_eq!(
                    label(&labels, server_label),
                    None,
                    "{runtime} strict={strict}: {labels:?}"
                );
            }
        }
    }

    /// A server info that fails leaves a gap on a plain read (the model
    /// labels, no server label) and fails a strict read, named, on both
    /// protos.
    #[tokio::test]
    async fn a_failing_server_info_leaves_a_gap_for_registration_and_fails_a_strict_read_by_name() {
        use mock_worker::{config::SchedulerWire, overrides::InfoAnswer};

        use crate::workflow::steps::util::mock_grpc_engine_on;

        let mut engines = Vec::new();
        for (wire, runtime) in [
            (SchedulerWire::Vllm, "vllm"),
            (SchedulerWire::Sglang, "sglang"),
        ] {
            let (url, engine) = mock_grpc_engine_on(wire).await;
            let server_label = with_server_label(&engine, runtime);
            engine.set_server_info(InfoAnswer::Failing);
            engines.push((url, runtime, server_label));
        }
        for (url, runtime, server_label) in &engines {
            let (labels, _) = fetch_grpc_metadata(url, runtime, false)
                .await
                .unwrap_or_else(|error| panic!("{runtime}: {error}"));
            assert_eq!(label(&labels, "model_path"), Some(MOCK_MODEL), "{runtime}");
            assert_eq!(label(&labels, server_label), None, "{runtime}: {labels:?}");

            let error = fetch_grpc_metadata(url, runtime, true)
                .await
                .expect_err("a strict read fails on a failing server info");
            assert!(
                error.starts_with("Failed to fetch gRPC server info: "),
                "{runtime}: {error}"
            );
        }
    }

    /// A model info that fails fails the read, named, strict or not: the
    /// model's fields are what registration reads, so there is no gap to
    /// leave, on both protos.
    #[tokio::test]
    async fn a_failing_model_info_fails_the_grpc_read_by_name() {
        use mock_worker::{config::SchedulerWire, overrides::InfoAnswer};

        use crate::workflow::steps::util::mock_grpc_engine_on;

        let mut engines = Vec::new();
        for (wire, runtime) in [
            (SchedulerWire::Vllm, "vllm"),
            (SchedulerWire::Sglang, "sglang"),
        ] {
            let (url, engine) = mock_grpc_engine_on(wire).await;
            engine.set_model_info(InfoAnswer::Failing);
            engines.push((url, runtime));
        }
        for (url, runtime) in &engines {
            for strict in [false, true] {
                let error = fetch_grpc_metadata(url, runtime, strict)
                    .await
                    .expect_err("a read of an engine whose model info fails");
                assert!(
                    error.starts_with("Failed to fetch gRPC model info: "),
                    "{runtime} strict={strict}: {error}"
                );
            }
        }
    }

    /// An engine that refuses the connection fails the read at the dial,
    /// named, strict or not.
    #[tokio::test]
    async fn an_engine_that_refuses_the_connection_fails_the_grpc_read_at_the_dial() {
        use smg_worker::test_support::RefusedAddress;

        let refused = RefusedAddress::reserve();
        let url = format!("grpc://{}", refused.addr());
        for runtime in ["vllm", "sglang"] {
            for strict in [false, true] {
                let error = fetch_grpc_metadata(&url, runtime, strict)
                    .await
                    .expect_err("a read of an engine that refuses the connection");
                assert!(
                    error.starts_with("Failed to connect to gRPC: "),
                    "{runtime} strict={strict}: {error}"
                );
            }
        }
    }
}
