//! WASM HTTP API Routes
//!
//! Provides REST API endpoints for managing WASM modules:
//! - POST /wasm - Add modules
//! - DELETE /wasm/:uuid - Remove a module
//! - GET /wasm - List all modules with metrics
//!
//! A refusal answers in the control plane's envelope (`{"error", "code"}`),
//! the shape the worker routes use; a load's per-module results keep theirs.

use std::{sync::Arc, time::Duration};

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use smg_auth::AuditResource;
use uuid::Uuid;

use crate::{
    server::AppState,
    wasm::{
        WasmMetrics, WasmModuleAddRequest, WasmModuleAddResponse, WasmModuleAddResult,
        WasmModuleListResponse,
    },
    worker::admin_json::{refusal, AdminJson},
    workflow::{
        job_queue::{Job, JobQueue},
        WasmModuleConfigRequest,
    },
};

/// Wait for job completion by polling job status
/// Returns the job result message if successful
async fn wait_for_job_completion(
    job_queue: &JobQueue,
    status_key: &str,
    timeout_duration: Duration,
) -> Result<String, String> {
    let start = std::time::Instant::now();
    let mut poll_interval = Duration::from_millis(100);
    let max_poll_interval = Duration::from_millis(2000);
    let poll_backoff = Duration::from_millis(200);

    loop {
        if start.elapsed() > timeout_duration {
            return Err(format!("Job timeout after {}s", timeout_duration.as_secs()));
        }

        if let Some(job_status) = job_queue.get_status(status_key) {
            match job_status.status.as_str() {
                "pending" | "processing" => {
                    tokio::time::sleep(poll_interval).await;
                    poll_interval = (poll_interval + poll_backoff).min(max_poll_interval);
                    continue;
                }
                "failed" => {
                    let error_msg = job_status
                        .message
                        .unwrap_or_else(|| "Unknown error".to_string());
                    job_queue.remove_status(status_key);
                    return Err(error_msg);
                }
                _ => {
                    // Should not happen, but handle gracefully
                    job_queue.remove_status(status_key);
                    return Err("Unexpected job status".to_string());
                }
            }
        } else {
            // Job completed successfully (status was removed by record_job_completion)
            // We need to get the result from the job execution
            // Since job queue removes status on success, we can't get the result here
            // We'll need to query the wasm manager to find the module by name
            // For now, return a success message and let caller extract UUID from manager
            return Ok("Job completed successfully".to_string());
        }
    }
}

/// 503 in the envelope: the gateway runs without the WASM module manager.
fn no_wasm_manager() -> Response {
    refusal(
        StatusCode::SERVICE_UNAVAILABLE,
        "SERVICE_UNAVAILABLE",
        "The WASM module manager is not enabled on this gateway".to_string(),
    )
}

/// 503 in the envelope: the job queue the module jobs run on is not up.
fn no_job_queue() -> Response {
    refusal(
        StatusCode::SERVICE_UNAVAILABLE,
        "SERVICE_UNAVAILABLE",
        "The job queue is not initialized".to_string(),
    )
}

pub(crate) async fn add_wasm_module(
    State(state): State<Arc<AppState>>,
    AdminJson(config): AdminJson<WasmModuleAddRequest>,
) -> Response {
    let Some(_) = state.context.wasm_manager.as_ref() else {
        return no_wasm_manager();
    };

    let Some(job_queue) = state.context.worker_job_queue.get() else {
        return no_job_queue();
    };

    let mut status = StatusCode::OK;
    let mut modules = config.modules.clone();

    for module in &mut modules {
        let wasm_config = WasmModuleConfigRequest {
            descriptor: module.clone(),
        };

        let job = Job::AddWasmModule {
            config: Box::new(wasm_config),
        };

        let worker_url = job.worker_url().to_string();

        // Submit job to queue
        match job_queue.submit(job).await {
            Ok(()) => {
                // Wait for job completion (timeout: 5 minutes)
                let timeout = Duration::from_secs(300);
                match wait_for_job_completion(job_queue, &worker_url, timeout).await {
                    Ok(_) => {
                        // Job completed successfully, but we need to get the UUID
                        // Since job queue removes status on success, we need to query
                        // the workflow engine or wasm manager to get the UUID
                        // For now, let's try to get it from the wasm manager by name
                        if let Some(wasm_manager) = state.context.wasm_manager.as_ref() {
                            if let Ok(all_modules) = wasm_manager.get_modules() {
                                if let Some(registered_module) = all_modules
                                    .iter()
                                    .find(|m| m.module_meta.name == module.name)
                                {
                                    module.add_result = Some(WasmModuleAddResult::Success(
                                        registered_module.module_uuid,
                                    ));
                                } else {
                                    module.add_result = Some(WasmModuleAddResult::Error(
                                        "Module registered but UUID not found".to_string(),
                                    ));
                                    status = StatusCode::BAD_REQUEST;
                                }
                            } else {
                                module.add_result = Some(WasmModuleAddResult::Error(
                                    "Failed to query registered modules".to_string(),
                                ));
                                status = StatusCode::BAD_REQUEST;
                            }
                        } else {
                            module.add_result = Some(WasmModuleAddResult::Error(
                                "WASM manager not available".to_string(),
                            ));
                            status = StatusCode::BAD_REQUEST;
                        }
                    }
                    Err(e) => {
                        module.add_result = Some(WasmModuleAddResult::Error(e));
                        status = StatusCode::BAD_REQUEST;
                    }
                }
            }
            Err(e) => {
                module.add_result = Some(WasmModuleAddResult::Error(format!(
                    "Failed to submit job: {e}"
                )));
                status = StatusCode::BAD_REQUEST;
            }
        }
    }

    let added: Vec<String> = modules
        .iter()
        .filter_map(|module| match &module.add_result {
            Some(WasmModuleAddResult::Success(uuid)) => Some(uuid.to_string()),
            Some(WasmModuleAddResult::Error(_)) | None => None,
        })
        .collect();
    let response = (status, Json(WasmModuleAddResponse { modules })).into_response();
    if added.is_empty() {
        return response;
    }
    AuditResource::name(response, added.join(","))
}

pub async fn remove_wasm_module(
    State(state): State<Arc<AppState>>,
    Path(module_uuid_str): Path<String>,
) -> Response {
    let Ok(module_uuid) = Uuid::parse_str(&module_uuid_str) else {
        return refusal(
            StatusCode::BAD_REQUEST,
            "BAD_REQUEST",
            format!("'{module_uuid_str}' is not a module uuid"),
        );
    };

    let Some(wasm_manager) = state.context.wasm_manager.as_ref() else {
        return no_wasm_manager();
    };

    let Some(job_queue) = state.context.worker_job_queue.get() else {
        return no_job_queue();
    };

    match wasm_manager.get_module(module_uuid) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return refusal(
                StatusCode::NOT_FOUND,
                "WASM_MODULE_NOT_FOUND",
                format!("No WASM module with uuid {module_uuid}"),
            )
        }
        Err(e) => {
            return refusal(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_SERVER_ERROR",
                format!("Failed to read the module table: {e}"),
            )
        }
    }

    use crate::workflow::WasmModuleRemovalRequest;

    let removal_request = WasmModuleRemovalRequest::new(module_uuid);

    let job = Job::RemoveWasmModule {
        request: Box::new(removal_request),
    };

    let worker_url = job.worker_url().to_string();

    // Submit job to queue
    match job_queue.submit(job).await {
        Ok(()) => {
            // Wait for job completion (timeout: 1 minute)
            let timeout = Duration::from_secs(60);
            match wait_for_job_completion(job_queue, &worker_url, timeout).await {
                Ok(_) => AuditResource::name(
                    (StatusCode::OK, "Module removed successfully").into_response(),
                    module_uuid.to_string(),
                ),
                Err(e) => refusal(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "INTERNAL_SERVER_ERROR",
                    e,
                ),
            }
        }
        Err(e) => refusal(
            StatusCode::SERVICE_UNAVAILABLE,
            "SERVICE_UNAVAILABLE",
            format!("Failed to submit job: {e}"),
        ),
    }
}

pub async fn list_wasm_modules(State(state): State<Arc<AppState>>) -> Response {
    let Some(wasm_manager) = state.context.wasm_manager.as_ref() else {
        return no_wasm_manager();
    };
    let modules = match wasm_manager.get_modules() {
        Ok(modules) => modules,
        Err(e) => {
            return refusal(
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_SERVER_ERROR",
                format!("Failed to read the module table: {e}"),
            )
        }
    };
    let (total, success, failed, total_time_ms, max_time_ms) = wasm_manager.get_metrics();
    let average_execution_time_ms = if total > 0 {
        Some(total_time_ms as f64 / total as f64)
    } else {
        None
    };
    let metrics = WasmMetrics {
        total_executions: total,
        successful_executions: success,
        failed_executions: failed,
        total_execution_time_ms: total_time_ms,
        max_execution_time_ms: max_time_ms,
        average_execution_time_ms,
    };
    let response = WasmModuleListResponse { modules, metrics };
    (StatusCode::OK, Json(response)).into_response()
}
