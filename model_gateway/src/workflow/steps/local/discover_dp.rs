//! Data-parallel discovery: the rank count a registering engine reports.
//!
//! A data-parallel engine registers as one worker per rank, so the routing
//! policy's pick among the rank workers is the rank choice and every rank
//! carries its own load. The discovery runs on every registration; no
//! configuration turns it on (`--dp-aware` is accepted and ignored for one
//! release). The rank count is read from the labels the metadata discovery
//! flattened: a gRPC engine's `GetServerInfo`, an SGLang HTTP engine's
//! `/server_info` (the one HTTP source of a rank count; every other HTTP
//! runtime registers as one worker). An engine that reports none registers
//! as one worker; a rank count missing from an SGLang engine's labels is
//! re-read strictly, so a failed read retries the registration instead of
//! settling the engine on one worker. A gRPC rank worker's load is pushed by
//! its servicer; an HTTP rank worker's comes from the gateway's polls, and
//! each request to it carries its `data_parallel_rank`, as before. A ZMQ
//! engine's data parallelism is the grouped worker (`dp_size` on the spec:
//! one worker, one socket set, N engines dialing in), never an expansion.
use std::collections::HashMap;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_metadata::server_info_labels;
use crate::{
    worker::{endpoint::Endpoint, ConnectionMode, UNKNOWN_MODEL_ID},
    workflow::data::{WorkerKind, WorkerWorkflowData},
};

/// DP (Data Parallel) information for a worker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

/// The rank count and model of a registering engine, from the labels its
/// metadata discovery flattened (a gRPC engine's `GetServerInfo`, an HTTP
/// engine's `/server_info`): `Ok(None)` when the engine reports no width,
/// `Err` for a width it reports wrongly (zero, unparsable, overflowing).
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

fn routing_dp_size(
    dp_size: Option<usize>,
    attn_dp_size: Option<usize>,
) -> Result<Option<usize>, String> {
    if dp_size.is_none() && attn_dp_size.is_none() {
        return Ok(None);
    }
    // Newer engine releases separate model replicas from attention-DP groups;
    // older servers report their routing width in dp_size alone.
    dp_size
        .unwrap_or(1)
        .checked_mul(attn_dp_size.unwrap_or(1))
        .filter(|&width| width > 0)
        .map(Some)
        .ok_or_else(|| "DP sizes must be positive and their product must fit usize".to_string())
}

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

/// Step 2b: the data-parallel rank count of a local worker, read from the
/// labels metadata discovery flattened its handshake into. Set for a gRPC
/// or HTTP engine that reports one; left unset for a ZMQ worker and for an
/// engine that reports none, which register as one worker.
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

        let config = &context.data.config;
        let url = Endpoint::redacted_str(&config.url);
        let labels = &context.data.discovered_labels;
        let one_worker = |why: &str| debug!("{why} for {url}; registering as one worker");
        let dp_info = match context.data.connection_mode {
            // A ZMQ engine's data parallelism is the grouped worker (`dp_size`
            // on the spec), never an expansion.
            Some(ConnectionMode::Zmq) => {
                one_worker("No rank expansion over ZMQ");
                return Ok(StepResult::Success);
            }
            // A gRPC engine reports its rank count through GetServerInfo,
            // flattened into the labels by the metadata discovery.
            Some(ConnectionMode::Grpc) => match dp_info_from_labels(labels) {
                Ok(Some(info)) => info,
                Ok(None) => {
                    one_worker("No DP size reported");
                    return Ok(StepResult::Success);
                }
                Err(e) => {
                    warn!("Invalid DP info for {url} ({e}); registering as one worker");
                    return Ok(StepResult::Success);
                }
            },
            // Over HTTP the one source of a rank count is SGLang's
            // /server_info; every other runtime (vLLM, TRT-LLM, MLX, a nested
            // gateway registered as `generic`, whose /get_server_info is one
            // of its engines') registers as one worker. The registration's
            // tolerant metadata read leaves a gap where /server_info failed,
            // so a rank count missing from the labels is re-read strictly: a
            // failed read fails the step, which is retried, rather than
            // settling the engine on one worker; an engine that answers
            // without a rank count is one worker.
            _ => {
                if context.data.detected_runtime_type.as_deref() != Some("sglang") {
                    one_worker("No rank discovery for this HTTP runtime");
                    return Ok(StepResult::Success);
                }
                match dp_info_from_labels(labels) {
                    Ok(Some(info)) => info,
                    Err(e) => {
                        warn!("Invalid DP info for {url} ({e}); registering as one worker");
                        return Ok(StepResult::Success);
                    }
                    Ok(None) => {
                        let client = context.data.http_client("discover_dp_info")?;
                        let read =
                            server_info_labels(&client, &config.url, config.api_key.as_deref())
                                .await
                                .map_err(|e| WorkflowError::StepFailed {
                                    step_id: StepId::new("discover_dp_info"),
                                    message: format!("Failed to read the rank count of {url}: {e}"),
                                })?;
                        match read.as_ref().map(dp_info_from_labels) {
                            Some(Ok(Some(info))) => info,
                            Some(Err(e)) => {
                                warn!("Invalid DP info for {url} ({e}); registering as one worker");
                                return Ok(StepResult::Success);
                            }
                            Some(Ok(None)) | None => {
                                one_worker("No DP size in /server_info");
                                return Ok(StepResult::Success);
                            }
                        }
                    }
                }
            }
        };

        debug!(
            "Discovered DP size {} for {url} (model: {})",
            dp_info.dp_size, dp_info.model_id
        );

        context.data.dp_info = Some(dp_info);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::WorkerSpec;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::testing::app_context_for_tests,
        config::RouterConfig,
        worker::WorkerRegistry,
        workflow::{
            data::WorkerRegistrationMode,
            steps::{create_worker_workflow_data, local::test_support::labels_of},
        },
    };

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
            let info = dp_info_from_labels(&metadata).unwrap().expect("a width");
            assert_eq!(info.dp_size, 4);
            assert_eq!(info.model_id, "m");
        }
    }

    #[test]
    fn dp_info_from_labels_model_path_fallback() {
        let info = dp_info_from_labels(&labels_of(&[("dp_size", "2"), ("model_path", "org/repo")]))
            .unwrap()
            .expect("a width");
        assert_eq!(info.model_id, "repo");
        let info =
            dp_info_from_labels(&labels_of(&[("dp_size", "2"), ("model_path", "org/repo/")]))
                .unwrap()
                .expect("a width");
        assert_eq!(info.model_id, "repo");
    }

    #[test]
    fn dp_info_from_labels_unknown_model_when_absent() {
        let info = dp_info_from_labels(&labels_of(&[("dp_size", "2")]))
            .unwrap()
            .expect("a width");
        assert_eq!(info.model_id, UNKNOWN_MODEL_ID);
    }

    #[test]
    fn dp_info_from_labels_leaves_a_missing_width_and_rejects_a_wrong_one() {
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

    /// The step run over a local worker of `mode` at `url`, its runtime as
    /// detected, whose metadata discovery left `labels`, under a
    /// configuration with `dp_aware` as given: what it leaves for the create
    /// step, or the step's error.
    async fn discovered(
        mode: ConnectionMode,
        runtime: &str,
        url: &str,
        labels: &[(&str, &str)],
        dp_aware: bool,
    ) -> WorkflowResult<Option<DpInfo>> {
        let router_config = RouterConfig {
            dp_aware,
            ..RouterConfig::default()
        };
        let mut data = create_worker_workflow_data(
            WorkerSpec::new(url),
            WorkerRegistrationMode::Upsert,
            app_context_for_tests(router_config, Arc::new(WorkerRegistry::new()), None),
        );
        data.worker_kind = Some(WorkerKind::Local);
        data.connection_mode = Some(mode);
        data.detected_runtime_type = Some(runtime.to_string());
        data.discovered_labels = labels_of(labels);
        let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);
        assert_eq!(
            DiscoverDPInfoStep.execute(&mut ctx).await?,
            StepResult::Success
        );
        Ok(ctx.data.dp_info)
    }

    const GRPC: (ConnectionMode, &str, &str) =
        (ConnectionMode::Grpc, "tokenspeed", "grpc://worker:8000");
    /// A closed port: an HTTP engine nothing answers for.
    const HTTP: (ConnectionMode, &str, &str) =
        (ConnectionMode::Http, "sglang", "http://127.0.0.1:1");

    /// Discovery runs on every registration, whatever the retired flag says:
    /// a gRPC or SGLang HTTP engine that reports four ranks leaves a four-rank
    /// `dp_info` for the create step; one that reports a single rank leaves a
    /// single-rank one; one that reports a wrong width, a gRPC engine that
    /// reports none, an HTTP engine of another runtime and a ZMQ worker leave
    /// none (one worker each).
    #[tokio::test]
    async fn ranks_are_discovered_for_grpc_and_http_engines_without_any_flag() {
        let four = [("dp_size", "4"), ("model_id", "m")];
        for dp_aware in [false, true] {
            for (mode, runtime, url) in [GRPC, HTTP] {
                let info = discovered(mode, runtime, url, &four, dp_aware)
                    .await
                    .unwrap()
                    .expect("the engine's rank count is discovered");
                assert_eq!(
                    (info.dp_size, info.model_id.as_str()),
                    (4, "m"),
                    "{mode:?} dp_aware {dp_aware}"
                );
                let one = discovered(mode, runtime, url, &[("dp_size", "1")], dp_aware)
                    .await
                    .unwrap()
                    .expect("a single rank is reported too");
                assert_eq!(one.dp_size, 1, "{mode:?}");
                assert!(
                    discovered(mode, runtime, url, &[("dp_size", "0")], dp_aware)
                        .await
                        .unwrap()
                        .is_none(),
                    "{mode:?}: a wrong width, one worker"
                );
            }
            assert!(discovered(GRPC.0, GRPC.1, GRPC.2, &[], dp_aware)
                .await
                .unwrap()
                .is_none());
            assert!(
                discovered(ConnectionMode::Http, "generic", HTTP.2, &four, dp_aware)
                    .await
                    .unwrap()
                    .is_none(),
                "another HTTP runtime registers as one worker whatever its labels say"
            );
            assert!(discovered(
                ConnectionMode::Zmq,
                "tokenspeed",
                "zmq://worker:8000",
                &four,
                dp_aware
            )
            .await
            .unwrap()
            .is_none());
        }
    }

    /// An SGLang HTTP engine whose labels carry no rank count is re-read
    /// strictly: a failed read fails the step (the registration retries), an
    /// engine answering without a width is one worker, one answering a width
    /// is expanded.
    #[tokio::test]
    async fn a_missing_http_rank_count_is_re_read_strictly() {
        let failed = discovered(HTTP.0, HTTP.1, HTTP.2, &[], false).await;
        assert!(failed.is_err(), "{failed:?}");
        for (body, want) in [
            (r#"{"model_path":"org/m"}"#, None),
            (r#"{"model_path":"org/m","dp_size":4}"#, Some(4)),
        ] {
            let (url, server) = serve_json(body).await;
            let info = discovered(ConnectionMode::Http, "sglang", &url, &[], false)
                .await
                .unwrap();
            server.abort();
            assert_eq!(info.as_ref().map(|info| info.dp_size), want, "{body}");
            if let Some(info) = info {
                assert_eq!(info.model_id, "m");
            }
        }
    }

    /// A local server answering `body` as JSON at every path.
    async fn serve_json(body: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        use axum::{http::header::CONTENT_TYPE, Router};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only server; the handle is aborted at the test's end"
        )]
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .fallback(move || async move { ([(CONTENT_TYPE, "application/json")], body) }),
            )
            .await
            .unwrap();
        });
        (format!("http://127.0.0.1:{port}"), server)
    }
}
