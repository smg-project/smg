//! The read of a local worker's data-parallel (DP) shape, as a step of its
//! registration.

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::Client;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    data::{DpInfo, WorkerKind, WorkerRegistrationWorkflowData},
    host::{require_host, WorkflowHost},
    metadata::get_server_info,
    registration::DISCOVER_DP_STEP,
    runtimes::reports_dp,
};
use crate::{
    identity::{basic::ConnectionMode, endpoint::Endpoint},
    worker::WorkerSpec,
    UNKNOWN_MODEL_ID,
};

/// The DP info of the worker at `url`, read from its server info.
async fn get_dp_info(client: &Client, url: &str, api_key: Option<&str>) -> Result<DpInfo, String> {
    let info = get_server_info(client, url, api_key).await?;

    let dp_size = routing_dp_size(info.dp_size, info.attn_dp_size).map_err(|error| {
        format!(
            "Invalid DP metadata from {}: {error}",
            Endpoint::redacted_str(url)
        )
    })?;

    let model_id = info
        .model_id
        .filter(|s| !s.is_empty())
        .or_else(|| info.served_model_name.filter(|s| !s.is_empty()))
        .or_else(|| {
            info.model_path
                .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
        })
        .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

    Ok(DpInfo { dp_size, model_id })
}

/// Build DP info from gRPC-discovered metadata labels, normalized by metadata discovery.
pub fn dp_info_from_labels(labels: &HashMap<String, String>) -> Result<DpInfo, String> {
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
    let dp_size = routing_dp_size(width("dp_size")?, width("attn_dp_size")?)?;

    Ok(DpInfo {
        dp_size,
        model_id: model_id_from_labels(labels),
    })
}

fn routing_dp_size(dp_size: Option<usize>, attn_dp_size: Option<usize>) -> Result<usize, String> {
    if dp_size.is_none() && attn_dp_size.is_none() {
        return Err("no DP size in discovered metadata".to_string());
    }
    // A newer engine reports model replicas apart from attention-DP groups;
    // an older one reports its routing width in dp_size alone.
    dp_size
        .unwrap_or(1)
        .checked_mul(attn_dp_size.unwrap_or(1))
        .filter(|&width| width > 0)
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

/// DP info for a ZMQ worker. There is nothing to read: its engines dial in
/// later and serve no metadata route, so the spec's own `dp_size` (N engines
/// dialing one socket set) is the answer. Reporting it lets the creation
/// register one worker per rank where the DP read is on.
fn zmq_dp_info(config: &WorkerSpec, labels: &HashMap<String, String>) -> DpInfo {
    DpInfo {
        dp_size: config.dp_size.unwrap_or(1).max(1),
        model_id: model_id_from_labels(labels),
    }
}

/// Reads a local worker's DP shape into `dp_info`, where the host's settings
/// turn the read on: how many ranks the worker runs, and the model they
/// serve. A worker left with no shape registers as one plain worker.
///
/// A gRPC worker's shape stands in the labels the metadata read left; with
/// no DP size among them it is left with none. An HTTP worker's is read
/// from its server info, where its runtime's row of the runtime table says
/// the size stands there, and a read that fails fails the step; a worker of
/// any other runtime is left with none. A ZMQ worker's shape is its spec's.
pub struct DiscoverDpInfoStep;

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRegistrationWorkflowData<H>> for DiscoverDpInfoStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRegistrationWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let host = require_host(context.data.host.as_ref())?;

        if !host.registration_settings().dp_aware {
            debug!(
                "Worker {} is not DP-aware, skipping DP discovery",
                Endpoint::redacted_str(&config.url)
            );
            return Ok(StepResult::Success);
        }

        debug!(
            "Discovering DP info for {} (DP-aware)",
            Endpoint::redacted_str(&config.url)
        );

        let dp_info = match context.data.connection_mode {
            // A gRPC worker reports its DP size in its server info, already
            // flattened into the discovered labels. A missing label means
            // the engine cannot honor a rank pin: the worker registers as
            // one plain worker and its registration does not fail.
            Some(ConnectionMode::Grpc) => {
                match dp_info_from_labels(&context.data.discovered_labels) {
                    Ok(info) => info,
                    Err(e) => {
                        warn!(
                            "No DP info for {} ({e}); registering as a single \
                         non-DP worker",
                            Endpoint::redacted_str(&config.url)
                        );
                        return Ok(StepResult::Success);
                    }
                }
            }
            // A ZMQ worker has no metadata route to read: its DP shape is
            // the spec's own `dp_size` (N engines dialing one socket set),
            // which the creation expands into rank workers. Engines that
            // only declare their count at the handshake are not known yet:
            // the worker registers as one grouped worker.
            Some(ConnectionMode::Zmq) => {
                if config.dp_size.is_none() {
                    warn!(
                        "No DP size for ZMQ worker {}: its ranks register as one grouped \
                         worker, and no routing policy chooses among them; give the \
                         engine count (--zmq-engine-count, or `dp_size` on the worker spec)",
                        Endpoint::redacted_str(&config.url)
                    );
                }
                zmq_dp_info(config, &context.data.discovered_labels)
            }
            _ => {
                // Over HTTP the DP size stands in the server info of the
                // runtimes whose row of the table says so. No other runtime
                // reports it there, and a runtime nobody detected has no row.
                let runtime = context.data.detected_runtime_type.as_deref();
                if !runtime.is_some_and(reports_dp) {
                    warn!(
                        "DP discovery is not supported for {} HTTP workers ({}), skipping",
                        runtime.unwrap_or("unknown"),
                        Endpoint::redacted_str(&config.url)
                    );
                    return Ok(StepResult::Success);
                }
                let client = context.data.http_client(DISCOVER_DP_STEP)?;
                get_dp_info(&client, &config.url, config.api_key.as_deref())
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: StepId::new(DISCOVER_DP_STEP),
                        message: format!("Failed to get DP info: {e}"),
                    })?
            }
        };

        debug!(
            "Discovered DP size {} for {} (model: {})",
            dp_info.dp_size,
            Endpoint::redacted_str(&config.url),
            dp_info.model_id
        );

        context.data.dp_info = Some(dp_info);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests;
