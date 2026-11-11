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
        job_queue::{Job, JobOutcomeError},
        WasmModuleConfigRequest,
    },
};

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
        // The uuid is assigned here, so the answer names the module the job
        // registered and two modules of one name never share an answer.
        let module_uuid = Uuid::now_v7();
        let job = Job::AddWasmModule {
            config: Box::new(WasmModuleConfigRequest::new(module.clone(), module_uuid)),
        };
        match job_queue
            .submit_and_wait(job, Duration::from_secs(300))
            .await
        {
            Ok(_) => module.add_result = Some(WasmModuleAddResult::Success(module_uuid)),
            Err(e) => {
                module.add_result = Some(WasmModuleAddResult::Error(e.to_string()));
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

    match job_queue
        .submit_and_wait(job, Duration::from_secs(60))
        .await
    {
        Ok(_) => AuditResource::name(
            (StatusCode::OK, "Module removed successfully").into_response(),
            module_uuid.to_string(),
        ),
        Err(error) => job_refusal(error),
    }
}

/// The status and code of a removal the queue did not finish: no queue is a
/// 503, a wait past its bound a 504 (the job may still finish; the client
/// re-reads `GET /wasm`), a job that ran and failed a 500.
fn job_refusal_parts(error: &JobOutcomeError) -> (StatusCode, &'static str) {
    match error {
        JobOutcomeError::NotQueued(_) => (StatusCode::SERVICE_UNAVAILABLE, "SERVICE_UNAVAILABLE"),
        JobOutcomeError::TimedOut(_) => (StatusCode::GATEWAY_TIMEOUT, "GATEWAY_TIMEOUT"),
        JobOutcomeError::Failed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_SERVER_ERROR"),
        JobOutcomeError::InFlight(_) => (StatusCode::CONFLICT, "CONFLICT"),
    }
}

fn job_refusal(error: JobOutcomeError) -> Response {
    let (status, code) = job_refusal_parts(&error);
    refusal(status, code, error.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A removal the queue did not finish answers by what happened: no queue
    /// 503, a wait past its bound 504, a failed job 500.
    #[test]
    fn a_removal_the_queue_did_not_finish_answers_by_what_happened() {
        let cases = [
            (
                JobOutcomeError::NotQueued("no queue".into()),
                StatusCode::SERVICE_UNAVAILABLE,
                "SERVICE_UNAVAILABLE",
            ),
            (
                JobOutcomeError::TimedOut(Duration::from_secs(60)),
                StatusCode::GATEWAY_TIMEOUT,
                "GATEWAY_TIMEOUT",
            ),
            (
                JobOutcomeError::Failed("the job failed".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_SERVER_ERROR",
            ),
            (
                JobOutcomeError::InFlight("0f1e2d3c".into()),
                StatusCode::CONFLICT,
                "CONFLICT",
            ),
        ];
        for (error, status, code) in cases {
            assert_eq!(job_refusal_parts(&error), (status, code), "{error}");
        }
    }
}
