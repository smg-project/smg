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
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use smg_auth::AuditResource;
use uuid::Uuid;

use crate::{
    admin::{
        json::{refusal, AdminJson},
        params::AdminPath,
    },
    server::AppState,
    wasm::{
        WasmMetrics, WasmModuleAddRequest, WasmModuleAddResponse, WasmModuleAddResult,
        WasmModuleListResponse,
    },
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

    let mut failures = Vec::new();
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
                let (failure, text) = add_failure(&e, module_uuid);
                module.add_result = Some(WasmModuleAddResult::Error(text));
                failures.push(failure);
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
    let response = (
        load_status(failures),
        Json(WasmModuleAddResponse { modules }),
    )
        .into_response();
    if added.is_empty() {
        return response;
    }
    AuditResource::name(response, added.join(","))
}

pub(crate) async fn remove_wasm_module(
    State(state): State<Arc<AppState>>,
    AdminPath(module_uuid_str): AdminPath<String>,
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

/// The status and code of a module job the queue did not finish: no queue is a
/// 503, a wait past its bound a 504 (the job may still finish; the client
/// re-reads `GET /wasm`), a job that ran and failed a 500, a removal while
/// another caller waits on one for the same module a 409.
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

/// The answer's status for a load's module failures: the highest among
/// them, so one module refused by the queue (503) beside one that failed
/// to load (400) answers 503; a load with no failure answers 200.
fn load_status(failures: impl IntoIterator<Item = StatusCode>) -> StatusCode {
    failures.into_iter().fold(StatusCode::OK, StatusCode::max)
}

/// The status and the result text of a module whose add job did not end
/// well. A job that ran and failed is the module's own failure, a 400. A queue
/// that took no job or a wait past its bound keep the removal route's status
/// (503, 504): the client retries or re-reads `GET /wasm` instead of fixing
/// its request. A timed-out job may still register the module, so its text
/// names the uuid the late registration carries.
fn add_failure(error: &JobOutcomeError, module_uuid: Uuid) -> (StatusCode, String) {
    let status = match error {
        JobOutcomeError::Failed(_) => StatusCode::BAD_REQUEST,
        _ => job_refusal_parts(error).0,
    };
    let text = match error {
        JobOutcomeError::TimedOut(_) => {
            format!("{error}; a registration that ends late carries the uuid {module_uuid}")
        }
        _ => error.to_string(),
    };
    (status, text)
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
#[path = "../../tests/unit/wasm/route/tests.rs"]
mod tests;
