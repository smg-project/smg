//! Metadata discovery step for local workers.

use std::collections::HashMap;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    data::{WorkerKind, WorkerRegistrationWorkflowData},
    host::{require_host, WorkflowHost},
    metadata::{discover_engine_labels, fetch_grpc_metadata},
    registration::DISCOVER_METADATA_STEP,
    runtimes::DEFAULT_RUNTIME,
};
use crate::identity::{basic::ConnectionMode, endpoint::Endpoint};

/// Reads what a local worker's engine reports about itself into
/// `discovered_labels`, over HTTP or gRPC. A read that fails leaves the
/// labels empty and does not fail the registration.
///
/// A ZMQ engine reports its metadata at the handshake, which runs after this
/// step: nothing is read for it here, and what a host knows of such a worker
/// ahead of the handshake is written by a step of the host's own.
pub struct DiscoverMetadataStep;

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRegistrationWorkflowData<H>> for DiscoverMetadataStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRegistrationWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;

        debug!(
            "Discovering metadata for {} ({:?})",
            Endpoint::redacted_str(&config.url),
            connection_mode
        );

        let (discovered_labels, detected_runtime) = match connection_mode {
            ConnectionMode::Http => {
                let runtime = context
                    .data
                    .detected_runtime_type
                    .as_deref()
                    .unwrap_or_else(|| {
                        warn!(
                            "No detected_runtime_type for {}, defaulting to {}",
                            Endpoint::redacted_str(&config.url),
                            DEFAULT_RUNTIME
                        );
                        DEFAULT_RUNTIME.as_str()
                    });
                let client = context.data.http_client(DISCOVER_METADATA_STEP)?;
                let connector = require_host(context.data.host.as_ref())?.engine_connector();
                discover_engine_labels(
                    &client,
                    connector.as_ref(),
                    &config.url,
                    ConnectionMode::Http,
                    runtime,
                    config.api_key.as_deref(),
                    false,
                )
                .await
            }
            ConnectionMode::Grpc => {
                let config_runtime = config.runtime_type.to_string();
                let runtime_type = context
                    .data
                    .detected_runtime_type
                    .as_deref()
                    .unwrap_or(&config_runtime);
                let connector = require_host(context.data.host.as_ref())?.engine_connector();
                fetch_grpc_metadata(connector.as_ref(), &config.url, runtime_type, false)
                    .await
                    .map(|(labels, rt)| (labels, Some(rt)))
            }
            // A ZMQ engine reports its metadata at the handshake, which runs
            // after this step: nothing is read here.
            ConnectionMode::Zmq => return Ok(StepResult::Success),
        }
        .unwrap_or_else(|e| {
            warn!(
                "Failed to fetch metadata for {}: {}",
                Endpoint::redacted_str(&config.url),
                e
            );
            (HashMap::new(), None)
        });

        debug!(
            "Discovered {} labels for {}",
            discovered_labels.len(),
            Endpoint::redacted_str(&config.url)
        );
        context.data.discovered_labels = discovered_labels;
        if let Some(runtime) = detected_runtime {
            context.data.detected_runtime_type = Some(runtime);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests;
