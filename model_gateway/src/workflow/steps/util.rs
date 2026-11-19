//! The gateway's edge of the worker crate's reach helpers, for the steps of
//! a registration that still stand here: they name a runtime by its text.

use std::sync::Arc;

// The gRPC base URL stands beside the address type, where the worker module
// reads it too; its path here is kept for the steps.
pub(crate) use smg_worker::identity::endpoint::grpc_base_url;
use smg_worker::{connection::engine::EngineConnection, identity::basic::RuntimeType};

use crate::routers::grpc::backend_client::BackendConnector;

/// A connection to the engine at `grpc_url`, spoken to as `runtime_type`,
/// made by the gateway's connector. The detection reads an engine through
/// the worker crate's port, as the worker runtime does.
pub(crate) async fn connect_grpc_as(
    grpc_url: &str,
    runtime_type: &str,
) -> Result<Arc<dyn EngineConnection>, String> {
    let runtime: RuntimeType = runtime_type
        .parse()
        .map_err(|_| format!("Unknown runtime type: {runtime_type}"))?;
    let connector = BackendConnector::shared();
    connector.connect_grpc(grpc_url, runtime).await
}

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
#[expect(
    clippy::disallowed_methods,
    reason = "test helper: the mock engine's serve task runs for the test process's lifetime"
)]
pub(crate) async fn mock_grpc_engine_on(
    wire: mock_worker::config::SchedulerWire,
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
    use smg_worker::{
        identity::endpoint::Endpoint,
        workflow::{grpc_health_answer, grpc_reachable_url, http_health_url, try_grpc_reachable},
    };

    use super::*;

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
}
