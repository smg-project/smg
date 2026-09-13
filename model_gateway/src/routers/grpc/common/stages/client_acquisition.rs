//! Client acquisition: get gRPC clients from selected workers.
//!
//! Called once at ingress and again per retry attempt after worker
//! re-selection.

use std::sync::Arc;

use axum::response::Response;
use tracing::error;

use crate::{
    routers::{
        common::overload,
        error,
        grpc::{
            backend_client::BackendClient,
            context::{ClientSelection, WorkerSelection},
        },
    },
    worker::Worker,
};

/// Acquire backend clients for the selected workers, with a dispatch-time
/// overload re-check: one relaxed atomic read per already-chosen worker,
/// closing the window between selection and dispatch in which a load report
/// can flip the veto.
pub(crate) async fn acquire_clients(
    workers: &WorkerSelection,
    model_id: &str,
) -> Result<ClientSelection, Response> {
    match workers {
        WorkerSelection::Single { worker } => {
            if let Some(shed) = overload::shed_if_worker_overloaded(worker.as_ref(), model_id) {
                return Err(shed);
            }
            let client = get_backend_client_from_worker(worker).await?;
            Ok(ClientSelection::Single { client })
        }
        WorkerSelection::Disaggregated {
            encode_assignments,
            prefill,
            decode,
            ..
        } => {
            // Every assigned leg, encode included: an encode worker is
            // vetoed at selection through the same filter, so leaving it out
            // of the re-check would be the one dispatch path that can send
            // to a worker known to be over the ceiling.
            if let Some(shed) = overload::shed_if_worker_overloaded(prefill.as_ref(), model_id)
                .or_else(|| overload::shed_if_worker_overloaded(decode.as_ref(), model_id))
                .or_else(|| {
                    encode_assignments.iter().flatten().find_map(|assignment| {
                        overload::shed_if_worker_overloaded(assignment.worker.as_ref(), model_id)
                    })
                })
            {
                return Err(shed);
            }
            let prefill_client = get_backend_client_from_worker(prefill).await?;
            let decode_client = get_backend_client_from_worker(decode).await?;

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
