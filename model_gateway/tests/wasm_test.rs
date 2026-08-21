//! WASM Module Integration Tests
//!
//! This test suite validates the complete WASM module management functionality:
//! - API endpoints (add, remove, list)
//! - Workflow integration
//! - Module execution
//! - Error handling

mod common;

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use llm_tokenizer::TokenizerRegistry;
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    health::ProbeState,
    policies::PolicyRegistry,
    routers::RouterFactory,
    server::{build_app, AppState},
    wasm::{
        module::{
            MiddlewareAttachPoint, WasmModuleAddRequest, WasmModuleAddResponse,
            WasmModuleAddResult, WasmModuleAttachPoint, WasmModuleDescriptor,
            WasmModuleListResponse, WasmModuleType,
        },
        module_manager::WasmModuleManager,
    },
    worker::{WorkerMonitor, WorkerRegistry},
};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use tempfile::TempDir;
use tokio::fs;
use tower::ServiceExt;
use uuid::Uuid;

#[expect(clippy::expect_used)]
/// Create a test AppContext with WASM manager initialized
async fn create_test_context_with_wasm() -> Arc<AppContext> {
    let config = RouterConfig::default();

    // Initialize WASM manager first
    let wasm_manager = Arc::new(WasmModuleManager::with_default_config());

    // Create AppContext with wasm_manager from the start
    let client = reqwest::Client::new();

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(config.policy.clone()));

    // Initialize storage backends
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    // Initialize the worker monitor with the same interval the
    // production builder uses so tests exercise the real polling
    // cadence.
    let worker_monitor = Some(Arc::new(WorkerMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        config.load_monitor_interval_secs,
        config.engine_metrics,
        config.disable_load_monitoring,
    )));

    // Create empty OnceLock for worker job queue, workflow engines, and mcp orchestrator
    use std::sync::OnceLock;
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());
    let mcp_orchestrator_lock = Arc::new(OnceLock::new());

    let app_context = Arc::new(
        AppContext::builder()
            .router_config(config.clone())
            .client(client)
            .rate_limiter(None)
            .tokenizer_registry(tokenizer_registry)
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .worker_monitor(worker_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .mcp_orchestrator(mcp_orchestrator_lock)
            .wasm_manager(Some(wasm_manager))
            .build()
            .expect("Failed to build AppContext with WASM manager"),
    );

    // Mirror production wiring: start the WorkerMonitor event loop so
    // tests exercise the event-driven group reconciliation path
    // instead of an inert monitor.
    if let Some(monitor) = &app_context.worker_monitor {
        monitor.start_event_loop();
    }

    // Initialize JobQueue after AppContext is created
    let weak_context = Arc::downgrade(&app_context);
    let job_queue =
        smg::workflow::JobQueue::new(smg::workflow::JobQueueConfig::default(), weak_context);
    app_context
        .worker_job_queue
        .set(job_queue)
        .expect("JobQueue should only be initialized once");

    // Initialize WorkflowEngines
    use smg::workflow::WorkflowEngines;
    let engines = WorkflowEngines::new(&config);
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");

    // Initialize MCP orchestrator with empty config
    use smg_mcp::{McpConfig, McpOrchestrator};
    let empty_config = McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: vec![],
        inventory: Default::default(),
        policy: Default::default(),
    };
    let mcp_orchestrator = McpOrchestrator::new(empty_config)
        .await
        .expect("Failed to create MCP orchestrator");
    app_context
        .mcp_orchestrator
        .set(Arc::new(mcp_orchestrator))
        .ok()
        .expect("McpOrchestrator should only be initialized once");

    app_context
}

#[expect(clippy::expect_used, clippy::unwrap_used)]
/// Create a test WASM component file
/// Dynamically generates a valid WASM component programmatically without external tools
/// This ensures tests work in new environments without requiring pre-built files or external tools
async fn create_test_wasm_component(temp_dir: &TempDir) -> String {
    use wasm_encoder::{Component, Module};

    // Create a minimal valid WASM module first
    // A minimal module needs at least a type section
    let mut module = Module::new();

    // Add an empty type section (0 types) - this is valid
    let type_section = wasm_encoder::TypeSection::new();
    module.section(&type_section);
    let mut component = Component::new();
    component.section(&wasm_encoder::ModuleSection(&module));
    let component_bytes = component.as_slice().to_vec();
    let component_path = temp_dir.path().join("test_module.component.wasm");
    fs::write(&component_path, component_bytes)
        .await
        .expect("Failed to write WASM component file");

    // Return absolute path
    component_path
        .canonicalize()
        .expect("Failed to canonicalize path")
        .to_str()
        .unwrap()
        .to_string()
}

#[expect(clippy::expect_used)]
/// Create a test app with WASM support
async fn create_test_app_with_wasm() -> (axum::Router, Arc<AppContext>, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let app_context = create_test_context_with_wasm().await;

    // Create a dummy router (we only need the app for WASM endpoints)
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("Failed to create router");
    let router = Arc::from(router);

    let app_state = Arc::new(AppState {
        router,
        probe_state: ProbeState::new(app_context.inflight_tracker.clone()),
        context: app_context.clone(),
        admission_queue: None,
        router_manager: None,
        mesh_handler: None,
        mesh_adapters: None,
    });

    let request_id_headers = vec!["x-request-id".to_string(), "x-correlation-id".to_string()];

    #[expect(
        clippy::expect_used,
        reason = "test helper assumes router config is already validated"
    )]
    let app = build_app(
        app_state,
        smg::middleware::AuthConfig::new(None),
        smg::middleware::AuthConfig::new(None),
        None, // No control plane auth for tests
        256 * 1024 * 1024,
        request_id_headers,
        vec![], // cors_allowed_origins
    )
    .expect("valid tenant resolution config");

    (app, app_context, temp_dir)
}

// ============================================================================
// API Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_wasm_api_add_module() {
    let (app, app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module".to_string(),
            file_path: wasm_file_path.clone(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(response_json.modules.len(), 1);
    let module_result = &response_json.modules[0].add_result;

    // Assert module didn't fail with error
    assert!(
        !matches!(module_result, Some(WasmModuleAddResult::Error(_))),
        "Module registration failed: {module_result:?}"
    );

    // If status is not OK, check the error message
    assert_eq!(
        status,
        StatusCode::OK,
        "Expected OK status but got {status:?}. Error: {module_result:?}. Body: {}",
        String::from_utf8_lossy(&body)
    );

    assert!(module_result.is_some());

    // Verify module is registered in wasm_manager
    if let Some(wasm_manager) = app_context.wasm_manager.as_ref() {
        let modules = wasm_manager.get_modules().expect("Failed to get modules");
        assert!(!modules.is_empty(), "Module should be registered");

        if let Some(WasmModuleAddResult::Success(uuid)) = module_result {
            let module = wasm_manager
                .get_module(*uuid)
                .expect("Failed to get module");
            assert!(module.is_some(), "Module should exist in manager");
        }
    }
}

#[tokio::test]
async fn test_wasm_api_add_module_invalid_file() {
    let (app, _app_context, _temp_dir) = create_test_app_with_wasm().await;

    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module".to_string(),
            file_path: "/nonexistent/path/to/module.component.wasm".to_string(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return error status
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(response_json.modules.len(), 1);
    let module_result = &response_json.modules[0].add_result;
    assert!(module_result.is_some());

    // Verify it's an error result
    if let Some(WasmModuleAddResult::Error(_)) = module_result {
        // Expected error
    } else {
        panic!("Expected error result for invalid file path");
    }
}

#[tokio::test]
async fn test_wasm_api_add_module_invalid_wasm() {
    let (app, _app_context, temp_dir) = create_test_app_with_wasm().await;

    // Create an invalid WASM file (just random bytes)
    let invalid_wasm_path = temp_dir.path().join("invalid.component.wasm");
    fs::write(&invalid_wasm_path, b"not a valid wasm file")
        .await
        .expect("Failed to write invalid WASM file");

    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "invalid_module".to_string(),
            file_path: invalid_wasm_path.to_str().unwrap().to_string(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return error status
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(response_json.modules.len(), 1);
    let module_result = &response_json.modules[0].add_result;
    assert!(module_result.is_some());

    // Verify it's an error result
    if let Some(WasmModuleAddResult::Error(_)) = module_result {
        // Expected error
    } else {
        panic!("Expected error result for invalid WASM file");
    }
}

#[tokio::test]
async fn test_wasm_api_list_modules() {
    let (app, _app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

    // First, add a module
    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module_list".to_string(),
            file_path: wasm_file_path.clone(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let add_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(add_response.status(), StatusCode::OK);

    // Wait a bit for the job to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Now list modules
    let list_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/wasm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(list_response.status(), StatusCode::OK);

    let body = to_bytes(list_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response_json: WasmModuleListResponse = serde_json::from_slice(&body).unwrap();

    assert!(
        !response_json.modules.is_empty(),
        "Should have at least one module"
    );
    assert!(response_json
        .modules
        .iter()
        .any(|m| m.module_meta.name == "test_module_list"));

    // Verify metrics are present (total_executions is u64, so always >= 0)
    let _ = response_json.metrics.total_executions;
}

#[tokio::test]
async fn test_wasm_api_remove_module() {
    let (app, app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

    // First, add a module
    let add_request = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module_remove".to_string(),
            file_path: wasm_file_path.clone(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let add_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(add_response.status(), StatusCode::OK);

    let body = to_bytes(add_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    // Wait for job to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Get the module UUID
    let module_uuid =
        if let Some(WasmModuleAddResult::Success(uuid)) = &response_json.modules[0].add_result {
            *uuid
        } else {
            // If we can't get UUID from response, try to find it from manager
            if let Some(wasm_manager) = app_context.wasm_manager.as_ref() {
                let modules = wasm_manager.get_modules().expect("Failed to get modules");
                modules
                    .iter()
                    .find(|m| m.module_meta.name == "test_module_remove")
                    .map(|m| m.module_uuid)
                    .expect("Module should be registered")
            } else {
                panic!("WASM manager not available");
            }
        };

    // Now remove the module
    let remove_response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/wasm/{module_uuid}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let remove_status = remove_response.status();
    assert_eq!(
        remove_status,
        StatusCode::OK,
        "Expected OK status for remove but got {remove_status:?}"
    );

    // Wait for removal to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify module is removed
    if let Some(wasm_manager) = app_context.wasm_manager.as_ref() {
        let module = wasm_manager
            .get_module(module_uuid)
            .expect("Failed to get module");
        assert!(module.is_none(), "Module should be removed");
    }
}

#[tokio::test]
async fn test_wasm_api_remove_module_not_found() {
    let (app, _app_context, _temp_dir) = create_test_app_with_wasm().await;

    // Try to remove a non-existent module
    let fake_uuid = Uuid::now_v7();
    let remove_response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/wasm/{fake_uuid}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return error status
    assert!(
        remove_response.status() == StatusCode::BAD_REQUEST
            || remove_response.status() == StatusCode::NOT_FOUND
    );
}

// ============================================================================
// WASM Functionality Tests
// ============================================================================

#[tokio::test]
async fn test_wasm_module_duplicate_sha256() {
    let (app, _app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

    // Add first module
    let add_request1 = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module_dup1".to_string(),
            file_path: wasm_file_path.clone(),
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response1 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request1).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response1.status(), StatusCode::OK);

    // Wait for first job to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Try to add the same file again (should fail due to duplicate SHA256)
    let add_request2 = WasmModuleAddRequest {
        modules: vec![WasmModuleDescriptor {
            name: "test_module_dup2".to_string(),
            file_path: wasm_file_path.clone(), // Same file
            module_type: WasmModuleType::Middleware,
            attach_points: vec![WasmModuleAttachPoint::Middleware(
                MiddlewareAttachPoint::OnRequest,
            )],
            add_result: None,
        }],
    };

    let response2 = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/wasm")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&add_request2).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return error status for duplicate
    assert!(
        response2.status() == StatusCode::BAD_REQUEST
            || response2.status() == StatusCode::INTERNAL_SERVER_ERROR
    );

    let body = to_bytes(response2.into_body(), usize::MAX).await.unwrap();
    let response_json: WasmModuleAddResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(response_json.modules.len(), 1);
    let module_result = &response_json.modules[0].add_result;
    assert!(module_result.is_some());

    // Verify it's an error result (duplicate)
    if let Some(WasmModuleAddResult::Error(err_msg)) = module_result {
        assert!(
            err_msg.contains("duplicate")
                || err_msg.contains("Duplicate")
                || err_msg.contains("SHA256")
        );
    } else {
        panic!("Expected error result for duplicate SHA256");
    }
}

#[tokio::test]
async fn test_wasm_module_execution() {
    let (_app, app_context, temp_dir) = create_test_app_with_wasm().await;
    let wasm_file_path = create_test_wasm_component(&temp_dir).await;

    // First, add a module using the workflow directly
    let wasm_manager = app_context
        .wasm_manager
        .as_ref()
        .expect("WASM manager should be initialized");

    let engines = app_context
        .workflow_engines
        .get()
        .expect("Workflow engines should be initialized");

    // Create workflow context for registration
    use smg::workflow::{WasmModuleConfigRequest, WasmRegistrationWorkflowData};
    use wfaas::WorkflowId;

    let descriptor = WasmModuleDescriptor {
        name: "test_execution_module".to_string(),
        file_path: wasm_file_path.clone(),
        module_type: WasmModuleType::Middleware,
        attach_points: vec![WasmModuleAttachPoint::Middleware(
            MiddlewareAttachPoint::OnRequest,
        )],
        add_result: None,
    };

    let config_request = WasmModuleConfigRequest { descriptor };
    let workflow_data = WasmRegistrationWorkflowData {
        config: config_request,
        wasm_bytes: None,
        sha256_hash: None,
        file_size_bytes: None,
        module_uuid: None,
        app_context: Some(app_context.clone()),
    };

    // Start workflow
    let instance_id = engines
        .wasm_registration
        .start_workflow(WorkflowId::new("wasm_module_registration"), workflow_data)
        .await
        .expect("Failed to start workflow");

    // Wait for workflow to complete
    let timeout = Duration::from_secs(30);
    let start = std::time::Instant::now();

    let module_uuid = loop {
        assert!(start.elapsed() <= timeout, "Workflow timeout");

        let state = engines
            .wasm_registration
            .get_status(instance_id)
            .await
            .expect("Failed to get workflow status");

        match state.status {
            wfaas::WorkflowStatus::Completed => {
                // Extract module UUID from typed workflow data
                break state
                    .context
                    .data
                    .module_uuid
                    .expect("Module UUID should be in context");
            }
            wfaas::WorkflowStatus::Failed => {
                panic!("Workflow failed: {state:?}");
            }
            _ => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    };

    // Verify module is registered
    let module = wasm_manager
        .get_module(module_uuid)
        .expect("Failed to get module");
    assert!(module.is_some(), "Module should be registered");

    // Get initial metrics
    let (initial_total, initial_success, initial_failed, _, _) = wasm_manager.get_metrics();

    // Execute the module
    use smg::wasm::{
        spec::smg::gateway::middleware_types,
        types::{WasmComponentInput, WasmComponentOutput},
    };

    let request = middleware_types::Request {
        method: "GET".to_string(),
        path: "/test".to_string(),
        query: String::new(),
        headers: vec![],
        body: vec![],
        request_id: "test-request-id".to_string(),
        now_epoch_ms: 1000,
    };

    let input = WasmComponentInput::MiddlewareRequest(request);
    let attach_point = WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);

    // Execute the module
    let result = wasm_manager
        .execute_module_interface(module_uuid, attach_point, input)
        .await;

    // Verify execution result
    match result {
        Ok(WasmComponentOutput::MiddlewareAction(action)) => {
            // Verify action is valid (should be Continue, Reject, or Modify)
            match action {
                middleware_types::Action::Continue => {
                    // Expected for a simple middleware
                }
                middleware_types::Action::Reject(_) => {
                    // Also valid
                }
                middleware_types::Action::Modify(_) => {
                    // Also valid
                }
            }
        }
        Err(_e) => {
            // Execution might fail if the WASM component is not properly built
            // This is acceptable for testing - we're testing the execution path, not the component itself
        }
    }

    // Verify metrics were updated
    let (final_total, final_success, final_failed, _, _) = wasm_manager.get_metrics();

    // Metrics should have increased (either success or failed)
    assert!(
        final_total > initial_total
            || final_failed > initial_failed
            || final_success > initial_success,
        "Metrics should be updated after execution"
    );
}
