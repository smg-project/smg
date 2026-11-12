//! Data Parallel (DP) information discovery step: the width an engine
//! advertised at metadata discovery, read as the ranks `create_worker`
//! registers. A gRPC engine's width is already in the discovered labels
//! (`GetServerInfo`), a ZMQ worker's on the spec as a grouped worker; an
//! HTTP engine is asked for its width here when the labels carry none and
//! the width decides the workers it registers as (`create_worker`'s rule),
//! since metadata discovery reads tolerantly and a width the gateway could
//! not read is not a width of one.

use std::collections::HashMap;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    create_worker::{resolve_model_id, resolved_runtime_type, takes_http_rank_workers},
    discover_metadata::read_width_labels,
};
use crate::{
    worker::{basic::merge_labels, endpoint::Endpoint, ConnectionMode, UNKNOWN_MODEL_ID},
    workflow::data::{WorkerKind, WorkerWorkflowData},
};

/// DP (Data Parallel) information for a worker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

/// Build DP info from the discovered metadata labels (gRPC server info
/// normalized by metadata discovery, or an HTTP engine's `/server_info`
/// flattened on the HTTP path): `Ok(None)` for an engine that advertises
/// no width, `Err` for a width that is present but unusable (not a number,
/// zero, an overflowing product).
pub fn dp_info_from_labels(labels: &HashMap<String, String>) -> Result<Option<DpInfo>, String> {
    let width = |name: &str| {
        labels
            .get(name)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid {name} in discovered metadata: {value}"))
            })
            .transpose()
    };
    let Some(dp_size) = routing_dp_size(width("dp_size")?, width("attn_dp_size")?)? else {
        return Ok(None);
    };

    Ok(Some(DpInfo {
        dp_size,
        model_id: model_id_from_labels(labels),
    }))
}

/// The routing width the labels advertise: `None` when neither width is
/// named, an error for a width that is named and unusable.
fn routing_dp_size(
    dp_size: Option<usize>,
    attn_dp_size: Option<usize>,
) -> Result<Option<usize>, String> {
    if dp_size.is_none() && attn_dp_size.is_none() {
        return Ok(None);
    }
    // Newer SGLang separates model replicas from attention-DP groups; older
    // servers report their routing width in dp_size alone.
    dp_size
        .unwrap_or(1)
        .checked_mul(attn_dp_size.unwrap_or(1))
        .filter(|&width| width > 0)
        .map(Some)
        .ok_or_else(|| "DP sizes must be positive and their product must fit usize".to_string())
}

/// Best-effort model identity from discovered metadata labels.
fn model_id_from_labels(labels: &HashMap<String, String>) -> String {
    labels
        .get("model_id")
        .or_else(|| labels.get("served_model_name"))
        .filter(|s| !s.is_empty())
        .cloned()
        .or_else(|| {
            labels
                .get("model_path")
                .and_then(|p| p.split('/').rfind(|s| !s.is_empty()))
                .map(str::to_string)
        })
        .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string())
}

/// Whether the engine's width decides the workers it registers as: the
/// decision `create_worker` takes for an HTTP engine (its runtime, the policy
/// the registration will assign, the rank scheduler's rule), read here from
/// the labels so far. Only then is a width the gateway could not read worth
/// a refusal; everywhere else the engine is one worker whatever its endpoint
/// answers, as before.
fn width_decides_the_workers(data: &WorkerWorkflowData) -> WorkflowResult<bool> {
    if data.connection_mode != Some(ConnectionMode::Http) {
        return Ok(false);
    }
    let app_context = data
        .app_context
        .as_ref()
        .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
    let labels = merge_labels(&data.discovered_labels, &data.config.labels);
    Ok(takes_http_rank_workers(
        app_context,
        &data.config,
        ConnectionMode::Http,
        resolved_runtime_type(data),
        resolve_model_id(&data.config, &labels),
        &labels,
    ))
}

/// Step 2b: Discover DP (Data Parallel) information, on every registration.
///
/// A gRPC or HTTP engine's width is the `dp_size` (times `attn_dp_size`)
/// label metadata discovery read, or, for an HTTP engine whose labels carry
/// none and whose width decides its workers, the one its width endpoint
/// answers here (a read that fails fails the registration after the step's
/// retries); an engine that reports none, or one, is one worker
/// (`create_worker` registers a rank per unit of width above one). A ZMQ
/// worker reports nothing: data parallelism over ZMQ is the grouped worker
/// (`dp_size: N` on the spec, one socket set, N engines dialing in), whose
/// connector places per rank itself, so there is no width to expand.
pub struct DiscoverDPInfoStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DiscoverDPInfoStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let url = context.data.config.url.clone();
        if context.data.connection_mode == Some(ConnectionMode::Zmq) {
            debug!(
                "ZMQ worker {} is one worker (a grouped worker places its own ranks); no DP \
                 discovery",
                Endpoint::redacted_str(&url)
            );
            return Ok(StepResult::Success);
        }

        let mut dp_info = dp_info_from_labels(&context.data.discovered_labels);
        if matches!(dp_info, Ok(None)) && width_decides_the_workers(&context.data)? {
            // Where the width decides the workers: metadata discovery reads
            // tolerantly, so the read of the endpoint that carries the width
            // may have failed, and a width the gateway could not read is not
            // a width of one. It is read here, where a failed read fails the
            // registration after the step's retries; an endpoint the engine
            // does not offer is one worker.
            let client = context.data.http_client("discover_dp_info")?;
            let api_key = context.data.config.api_key.clone();
            match read_width_labels(&client, &url, api_key.as_deref()).await {
                Ok(Some(labels)) => {
                    for (name, value) in labels {
                        context.data.discovered_labels.entry(name).or_insert(value);
                    }
                    dp_info = dp_info_from_labels(&context.data.discovered_labels);
                }
                Ok(None) => {}
                Err(e) => {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("discover_dp_info"),
                        message: format!("width of {}: {e}", Endpoint::redacted_str(&url)),
                    });
                }
            }
        }

        match dp_info {
            Ok(Some(dp_info)) => {
                debug!(
                    "Discovered DP size {} for {} (model: {})",
                    dp_info.dp_size,
                    Endpoint::redacted_str(&url),
                    dp_info.model_id
                );
                context.data.dp_info = Some(dp_info);
            }
            // The usual case: an engine that is not data-parallel advertises
            // no width. One worker, nothing to log above debug.
            Ok(None) => debug!(
                "No DP size for {}; one worker",
                Endpoint::redacted_str(&url)
            ),
            // A width the engine advertised but the gateway cannot use: one
            // worker, and the operator is told, since the pin is lost.
            Err(reason) => warn!(
                "Unusable DP size for {} ({reason}); registering one worker",
                Endpoint::redacted_str(&url)
            ),
        }
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::{WorkerSpec, WorkerType};
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::testing::app_context_for_tests,
        config::{PolicyConfig, RouterConfig},
        worker::WorkerRegistry,
        workflow::{
            data::WorkerRegistrationMode,
            steps::{
                create_worker_workflow_data,
                local::test_support::{labels_of, serve_json_routes},
            },
        },
    };

    /// A local registration after metadata discovery: the transport decided,
    /// the engine's labels read, no DP info yet. No flag on the gateway's
    /// configuration says anything about data parallelism any more.
    fn discovered(
        url: &str,
        connection_mode: ConnectionMode,
        discovered_labels: HashMap<String, String>,
    ) -> WorkflowContext<WorkerWorkflowData> {
        discovered_under(
            RouterConfig::default(),
            WorkerSpec::new(url),
            connection_mode,
            discovered_labels,
        )
    }

    /// `discovered`, under a gateway configuration and a spec of the test's
    /// choice.
    fn discovered_under(
        router_config: RouterConfig,
        spec: WorkerSpec,
        connection_mode: ConnectionMode,
        discovered_labels: HashMap<String, String>,
    ) -> WorkflowContext<WorkerWorkflowData> {
        let app_context =
            app_context_for_tests(router_config, Arc::new(WorkerRegistry::new()), None);
        let mut data =
            create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, app_context);
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(connection_mode);
        data.discovered_labels = discovered_labels;
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    async fn dp_size_after_the_step(
        ctx: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> Option<usize> {
        assert_eq!(
            DiscoverDPInfoStep.execute(ctx).await.unwrap(),
            StepResult::Success
        );
        ctx.data.dp_info.as_ref().map(|info| info.dp_size)
    }

    /// The width a gRPC engine advertised becomes its DP info on every
    /// registration: nothing in the configuration gates it.
    #[tokio::test]
    async fn a_grpc_width_label_becomes_dp_info_without_any_flag() {
        let mut ctx = discovered(
            "grpc://worker:8000",
            ConnectionMode::Grpc,
            labels_of(&[("dp_size", "4"), ("served_model_name", "m")]),
        );
        assert_eq!(dp_size_after_the_step(&mut ctx).await, Some(4));
        assert_eq!(ctx.data.dp_info.as_ref().unwrap().model_id, "m");
    }

    /// An HTTP engine's width is read from the labels metadata discovery
    /// flattened out of its server info, the same read as a gRPC engine's: no
    /// second request to the engine when the labels carry it, and the
    /// attention-DP split counts.
    #[tokio::test]
    async fn an_http_width_label_becomes_dp_info_from_the_discovered_labels() {
        let mut ctx = discovered(
            "http://worker:8000",
            ConnectionMode::Http,
            labels_of(&[("dp_size", "2"), ("attn_dp_size", "2")]),
        );
        assert_eq!(dp_size_after_the_step(&mut ctx).await, Some(4));
    }

    /// An engine that advertises no width is one worker: the step succeeds
    /// with no DP info, over gRPC and HTTP alike (an HTTP engine whose
    /// metadata endpoint carries no width included).
    #[tokio::test]
    async fn a_missing_width_is_one_worker() {
        for (url, mode) in [
            ("grpc://worker:8000", ConnectionMode::Grpc),
            ("http://worker:8000", ConnectionMode::Http),
        ] {
            let mut ctx = discovered(url, mode, labels_of(&[("served_model_name", "m")]));
            assert_eq!(dp_size_after_the_step(&mut ctx).await, None, "{url}");
        }
    }

    /// The policy under which an HTTP engine's width decides its workers.
    fn cache_aware() -> RouterConfig {
        RouterConfig {
            policy: PolicyConfig::cache_aware(0.5, 32, 1.5, 60, 1000),
            ..RouterConfig::default()
        }
    }

    /// A width endpoint that fails with an answer other than 404 (the old
    /// form answers 501, as a nested gateway's does).
    fn failing_width_endpoint() -> Vec<(&'static str, http::StatusCode, serde_json::Value)> {
        vec![(
            "/get_server_info",
            http::StatusCode::NOT_IMPLEMENTED,
            serde_json::json!({"error": "not implemented"}),
        )]
    }

    /// A local HTTP registration with no width among the discovered labels:
    /// the gateway's configuration, the worker's type and the runtime
    /// detection recorded (`None`: detection ran out of retries) as given, the
    /// engine's endpoints as `routes` serves them.
    async fn http_without_a_width(
        router_config: RouterConfig,
        worker_type: WorkerType,
        runtime: Option<&str>,
        routes: Vec<(&'static str, http::StatusCode, serde_json::Value)>,
    ) -> WorkflowContext<WorkerWorkflowData> {
        let mut spec = WorkerSpec::new(serve_json_routes(routes).await);
        spec.worker_type = worker_type;
        let mut ctx = discovered_under(
            router_config,
            spec,
            ConnectionMode::Http,
            labels_of(&[("served_model_name", "m")]),
        );
        ctx.data.detected_runtime_type = runtime.map(str::to_string);
        ctx.data.http_client_handle = Some(Arc::new(crate::server::http_client()));
        ctx
    }

    /// Where the width decides the workers (an HTTP engine of the runtime that
    /// takes rank workers, under cache-aware), an engine whose labels carry
    /// none (its width endpoint failed the tolerant metadata read) is asked
    /// again here: a width it answers with is its DP info and the labels take
    /// it, an endpoint it does not offer is one worker, a failed read fails
    /// the step, so the engine is never registered as one unpinned worker by
    /// a read that failed. An engine whose detection ran out of retries and
    /// whose spec names no runtime is the default local runtime here as at
    /// the expansion: asked, and refused on a failed read.
    #[tokio::test]
    async fn an_engines_width_is_read_here_where_it_decides_the_workers() {
        use http::StatusCode;

        let answering = vec![(
            "/server_info",
            StatusCode::OK,
            serde_json::json!({"dp_size": 4, "model_path": "org/model-w"}),
        )];
        let mut ctx = http_without_a_width(
            cache_aware(),
            WorkerType::Regular,
            Some("sglang"),
            answering,
        )
        .await;
        assert_eq!(dp_size_after_the_step(&mut ctx).await, Some(4));
        let labels = &ctx.data.discovered_labels;
        assert_eq!(labels.get("dp_size").map(String::as_str), Some("4"));
        assert_eq!(
            labels.get("model_path").map(String::as_str),
            Some("org/model-w")
        );
        assert_eq!(ctx.data.dp_info.as_ref().unwrap().model_id, "m");

        let mut ctx =
            http_without_a_width(cache_aware(), WorkerType::Regular, Some("sglang"), vec![]).await;
        assert_eq!(dp_size_after_the_step(&mut ctx).await, None);

        for runtime in [Some("sglang"), None] {
            let mut ctx = http_without_a_width(
                cache_aware(),
                WorkerType::Regular,
                runtime,
                failing_width_endpoint(),
            )
            .await;
            let error = DiscoverDPInfoStep
                .execute(&mut ctx)
                .await
                .expect_err("a failed width read fails the step");
            assert!(
                error.to_string().contains("server_info"),
                "{runtime:?}: {error}"
            );
            assert!(ctx.data.dp_info.is_none());
        }
    }

    /// Where the width decides nothing the engine is not asked, and a failing
    /// width endpoint registers it as one worker, as before: under the default
    /// policy, for a prefill worker under the rank scheduler, and for a
    /// runtime that takes no rank workers.
    #[tokio::test]
    async fn an_engine_whose_width_decides_nothing_is_not_asked() {
        let under_the_scheduler = RouterConfig {
            dp_minimum_tokens_scheduler: true,
            ..cache_aware()
        };
        for (case, router_config, worker_type, runtime) in [
            (
                "the default policy",
                RouterConfig::default(),
                WorkerType::Regular,
                Some("sglang"),
            ),
            (
                "the rank scheduler",
                under_the_scheduler,
                WorkerType::Prefill,
                Some("sglang"),
            ),
            (
                "a generic worker",
                cache_aware(),
                WorkerType::Regular,
                Some("generic"),
            ),
            (
                "another runtime",
                cache_aware(),
                WorkerType::Regular,
                Some("vllm"),
            ),
        ] {
            let mut ctx = http_without_a_width(
                router_config,
                worker_type,
                runtime,
                failing_width_endpoint(),
            )
            .await;
            assert_eq!(dp_size_after_the_step(&mut ctx).await, None, "{case}");
        }
    }

    /// A width the engine advertised but the gateway cannot use (not a
    /// number, zero, an overflowing product) is one worker too, said at
    /// warn: the registration goes on without the pin.
    #[tokio::test]
    async fn an_unusable_width_is_one_worker() {
        for labels in [
            labels_of(&[("dp_size", "abc")]),
            labels_of(&[("dp_size", "0")]),
            labels_of(&[("dp_size", &usize::MAX.to_string()), ("attn_dp_size", "2")]),
        ] {
            let mut ctx = discovered("grpc://worker:8000", ConnectionMode::Grpc, labels);
            assert_eq!(dp_size_after_the_step(&mut ctx).await, None);
        }
    }

    /// A ZMQ worker reports no DP info whatever its spec says: a grouped
    /// worker is one worker whose connector places its ranks, so a
    /// `dp_size` of four on the spec is no width to expand.
    #[tokio::test]
    async fn a_zmq_worker_reports_no_dp_info() {
        let mut ctx = discovered(
            "ipc:///tmp/smg-zmq/engine.ipc",
            ConnectionMode::Zmq,
            labels_of(&[("model_id", "m")]),
        );
        ctx.data.config.dp_size = Some(4);
        assert_eq!(dp_size_after_the_step(&mut ctx).await, None);
    }

    #[test]
    fn dp_info_from_labels_reads_dp_size_and_model() {
        for widths in [
            vec![("dp_size", "4")],
            vec![("dp_size", "1"), ("attn_dp_size", "4")],
            vec![("attn_dp_size", "4")],
            vec![("dp_size", "2"), ("attn_dp_size", "2")],
        ] {
            let mut metadata = labels_of(&widths);
            metadata.insert("served_model_name".to_string(), "m".to_string());
            let info = dp_info_from_labels(&metadata).unwrap().unwrap();
            assert_eq!(info.dp_size, 4);
            assert_eq!(info.model_id, "m");
        }
    }

    #[test]
    fn dp_info_from_labels_model_path_fallback() {
        let info = dp_info_from_labels(&labels_of(&[("dp_size", "2"), ("model_path", "org/repo")]))
            .unwrap()
            .unwrap();
        assert_eq!(info.model_id, "repo");
        let info =
            dp_info_from_labels(&labels_of(&[("dp_size", "2"), ("model_path", "org/repo/")]))
                .unwrap()
                .unwrap();
        assert_eq!(info.model_id, "repo");
    }

    #[test]
    fn dp_info_from_labels_unknown_model_when_absent() {
        let info = dp_info_from_labels(&labels_of(&[("dp_size", "2")]))
            .unwrap()
            .unwrap();
        assert_eq!(info.model_id, UNKNOWN_MODEL_ID);
    }

    #[test]
    fn dp_info_from_labels_is_none_without_a_width_and_rejects_a_bad_one() {
        assert!(dp_info_from_labels(&labels_of(&[])).unwrap().is_none());
        assert!(dp_info_from_labels(&labels_of(&[("dp_size", "0")])).is_err());
        assert!(dp_info_from_labels(&labels_of(&[("dp_size", "abc")])).is_err());
        assert!(
            dp_info_from_labels(&labels_of(&[("dp_size", "1"), ("attn_dp_size", "0")])).is_err()
        );
        assert!(
            dp_info_from_labels(&labels_of(&[("dp_size", "1"), ("attn_dp_size", "abc")])).is_err()
        );
        let overflow = labels_of(&[("dp_size", &usize::MAX.to_string()), ("attn_dp_size", "2")]);
        assert!(dp_info_from_labels(&overflow).is_err());
    }
}
