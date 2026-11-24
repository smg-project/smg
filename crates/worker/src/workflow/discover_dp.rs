//! The read of a local worker's data-parallel (DP) shape, as a step of its
//! registration.

use std::collections::HashMap;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    data::{DpInfo, WorkerKind, WorkerRegistrationWorkflowData},
    host::WorkflowHost,
    runtimes::reports_dp,
};
use crate::{
    identity::{basic::ConnectionMode, endpoint::Endpoint},
    worker::WorkerSpec,
    UNKNOWN_MODEL_ID,
};

/// The DP info the discovered metadata labels hold, as the metadata read
/// normalized them: none where they name no DP size, an error where they
/// name one that cannot be used (not a number, zero, a product past `usize`).
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
    let dp_size = routing_dp_size(width("dp_size")?, width("attn_dp_size")?)?;

    Ok(dp_size.map(|dp_size| DpInfo {
        dp_size,
        model_id: model_id_from_labels(labels),
    }))
}

/// The routing width of an engine that names `dp_size`, `attn_dp_size` or
/// both: none where it names neither.
fn routing_dp_size(
    dp_size: Option<usize>,
    attn_dp_size: Option<usize>,
) -> Result<Option<usize>, String> {
    if dp_size.is_none() && attn_dp_size.is_none() {
        return Ok(None);
    }
    // A newer engine reports model replicas apart from attention-DP groups;
    // an older one reports its routing width in dp_size alone.
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

/// DP info for a ZMQ worker. There is nothing to read: its engines dial in
/// later and serve no metadata route, so the spec's own `dp_size` (N engines
/// dialing one socket set) is the answer. Reporting it lets the creation
/// register one worker per rank.
fn zmq_dp_info(config: &WorkerSpec, labels: &HashMap<String, String>) -> DpInfo {
    DpInfo {
        dp_size: config.dp_size.unwrap_or(1).max(1),
        model_id: model_id_from_labels(labels),
    }
}

/// The DP shape the labels hold for the worker at `url`. Labels that name no
/// size hold none: an engine that runs one rank, or one that cannot take a
/// rank pin. A size that cannot be used holds none either, and is logged:
/// the engine named ranks the registration cannot register.
fn dp_info_of(labels: &HashMap<String, String>, url: &str) -> Option<DpInfo> {
    match dp_info_from_labels(labels) {
        Ok(Some(info)) => Some(info),
        Ok(None) => {
            debug!(
                "No DP size among the labels of {}; registering as a single non-DP worker",
                Endpoint::redacted_str(url)
            );
            None
        }
        Err(e) => {
            warn!(
                "Unusable DP info for {} ({e}); registering as a single non-DP worker",
                Endpoint::redacted_str(url)
            );
            None
        }
    }
}

/// Reads a local worker's DP shape into `dp_info`, on every registration:
/// how many ranks the worker runs, and the model they serve. A worker left
/// with no shape registers as one plain worker, and nothing the read finds
/// fails the registration.
///
/// A gRPC worker's shape stands in the labels the metadata read left. An
/// HTTP worker's stands there too, and counts where its runtime's row of the
/// runtime table says its server info reports the size: a worker of any
/// other runtime, or of none detected, is left with no shape whatever its
/// labels hold, since a host that fronts an engine answers with that
/// engine's labels and is one worker. A ZMQ worker's shape is its spec's.
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
        let labels = &context.data.discovered_labels;

        let dp_info = match context.data.connection_mode {
            // A gRPC worker reports its DP size in its server info, already
            // flattened into the discovered labels. A missing label means
            // the engine cannot honor a rank pin: the worker registers as
            // one plain worker.
            Some(ConnectionMode::Grpc) => dp_info_of(labels, &config.url),
            // A ZMQ worker has no metadata route to read: its DP shape is
            // the spec's own `dp_size` (N engines dialing one socket set),
            // which the creation expands into rank workers. Engines that
            // only declare their count at the handshake are not known yet:
            // the worker registers as one worker, grouped where they
            // declare several, and no routing policy chooses among them.
            Some(ConnectionMode::Zmq) => {
                if config.dp_size.is_none() {
                    debug!(
                        "No DP size for ZMQ worker {}: it registers as one worker; give the \
                         engine count (--zmq-engine-count, or `dp_size` on the worker spec) \
                         to register one worker per engine",
                        Endpoint::redacted_str(&config.url)
                    );
                }
                Some(zmq_dp_info(config, labels))
            }
            _ => {
                // Over HTTP the DP size stands in the server info of the
                // runtimes whose row of the table says so, which the
                // metadata read flattened into the labels. No other runtime
                // reports it there, and a runtime nobody detected has no row.
                let runtime = context.data.detected_runtime_type.as_deref();
                if runtime.is_some_and(reports_dp) {
                    dp_info_of(labels, &config.url)
                } else {
                    debug!(
                        "No DP size is read for {} HTTP workers ({}); registering as a single \
                         non-DP worker",
                        runtime.unwrap_or("unknown"),
                        Endpoint::redacted_str(&config.url)
                    );
                    None
                }
            }
        };

        if let Some(dp_info) = &dp_info {
            debug!(
                "Discovered DP size {} for {} (model: {})",
                dp_info.dp_size,
                Endpoint::redacted_str(&config.url),
                dp_info.model_id
            );
        }

        context.data.dp_info = dp_info;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests;
