//! Client acquisition: get gRPC clients from selected workers.
//!
//! Called once at ingress and again per retry attempt after worker
//! re-selection.

use std::sync::Arc;

use axum::response::Response;
use tracing::error;

use crate::{
    routers::{
        error,
        grpc::{
            backend_client::BackendClient,
            context::{ClientSelection, WorkerSelection},
        },
    },
    worker::Worker,
};

/// Client acquisition distinguishes a pre-dispatch overload observation from
/// transport/configuration failures. The pipeline may safely reselect for the
/// former because no backend RPC has started; it must return the latter.
///
/// The rendered response is boxed so the overload path — the common one — does
/// not carry a response-sized payload through every `Result` in the pipeline.
pub(crate) enum ClientAcquisitionError {
    Overloaded(Arc<dyn Worker>),
    Response(Box<Response>),
}

fn first_overloaded_worker(workers: &WorkerSelection) -> Option<Arc<dyn Worker>> {
    match workers {
        WorkerSelection::Single { worker } => worker.is_overloaded().then(|| Arc::clone(worker)),
        WorkerSelection::Disaggregated {
            encode_assignments,
            prefill,
            decode,
            ..
        } => [prefill, decode]
            .into_iter()
            .find(|worker| worker.is_overloaded())
            .cloned()
            .or_else(|| {
                encode_assignments.iter().flatten().find_map(|assignment| {
                    assignment
                        .worker
                        .is_overloaded()
                        .then(|| Arc::clone(&assignment.worker))
                })
            }),
    }
}

/// Acquire backend clients for the selected workers, bracketing any lazy
/// connection awaits with overload checks. This closes the selection-to-build
/// window without sending an inference request to a newly vetoed worker.
pub(crate) async fn acquire_clients(
    workers: &WorkerSelection,
) -> Result<ClientSelection, ClientAcquisitionError> {
    if let Some(worker) = first_overloaded_worker(workers) {
        return Err(ClientAcquisitionError::Overloaded(worker));
    }

    match workers {
        WorkerSelection::Single { worker } => {
            let client = get_backend_client_from_worker(worker)
                .await
                .map_err(|response| ClientAcquisitionError::Response(Box::new(response)))?;
            if let Some(worker) = first_overloaded_worker(workers) {
                return Err(ClientAcquisitionError::Overloaded(worker));
            }
            Ok(ClientSelection::Single { client })
        }
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => {
            let prefill_client = get_backend_client_from_worker(prefill)
                .await
                .map_err(|response| ClientAcquisitionError::Response(Box::new(response)))?;
            let decode_client = get_backend_client_from_worker(decode)
                .await
                .map_err(|response| ClientAcquisitionError::Response(Box::new(response)))?;
            // Re-check after any lazy connection awaits. Every assigned leg,
            // encode included, must still be open immediately before request
            // construction begins.
            if let Some(worker) = first_overloaded_worker(workers) {
                return Err(ClientAcquisitionError::Overloaded(worker));
            }

            Ok(ClientSelection::Disaggregated {
                prefill: prefill_client,
                decode: decode_client,
            })
        }
    }
}

async fn get_backend_client_from_worker(
    worker: &Arc<dyn Worker>,
) -> Result<BackendClient, Response> {
    // Get cached client from worker (or create one if not cached yet)
    let client_arc = worker
        .get_backend_client()
        .await
        .map_err(|e| {
            error!(
                function = "get_backend_client_from_worker",
                error = %e,
                "Failed to get backend client from worker"
            );
            error::internal_error(
                "get_backend_client_failed",
                format!("Failed to get backend client: {e}"),
            )
        })?
        .ok_or_else(|| {
            error!(
                function = "get_backend_client_from_worker",
                "Selected worker has no gRPC/ZMQ backend client"
            );
            error::internal_error(
                "worker_not_configured_for_backend",
                "Selected worker is not configured for a gRPC/ZMQ backend",
            )
        })?;

    Ok((*client_arc).clone())
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, WorkerType};

    #[tokio::test]
    async fn overload_is_reported_as_reselectable_before_client_creation() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:9999")
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        worker.set_overloaded(true);

        let result = acquire_clients(&WorkerSelection::Single {
            worker: Arc::clone(&worker),
        })
        .await;

        match result {
            Err(ClientAcquisitionError::Overloaded(observed)) => {
                assert!(Arc::ptr_eq(&observed, &worker));
            }
            _ => panic!("overload must be distinguished from a client acquisition error"),
        }
    }
}
