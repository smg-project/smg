//! Load balancing policy FFI bindings for Go SDK
//!
//! This module provides FFI functions to create and use load balancing policies
//! from the model_gateway crate. It enables the Go SDK to distribute requests
//! across multiple gRPC workers using the same policy implementations as the
//! Rust gateway.

use std::{
    any::Any,
    ffi::{CStr, CString},
    os::raw::c_char,
    ptr,
    sync::{
        atomic::{AtomicU8, AtomicUsize, Ordering},
        Arc,
    },
};

use async_trait::async_trait;
use llm_tokenizer::{create_tokenizer_from_file, traits::Tokenizer};
use openai_protocol::{
    chat::ChatCompletionRequest,
    worker::{HealthCheckConfig, WorkerSpec, WorkerStatus},
};
use smg::{
    policies::{
        BucketPolicy, CacheAwarePolicy, LoadBalancingPolicy, PowerOfTwoPolicy,
        RandomPolicy, RoundRobinPolicy, SelectWorkerInfo,
    },
    routers::grpc::{backend_client::BackendClient, utils::process_chat_messages},
    worker::{
        circuit_breaker::{CircuitBreaker, CircuitState},
        resilience::ResolvedResilience,
        worker::{RuntimeType, WorkerMetadata, WorkerRoutingKeyLoad},
        ConnectionMode, OverloadThresholds, Worker, WorkerResult, WorkerType,
    },
};
use smg_grpc_client::sglang_scheduler::{SglangGenerateRequestOptions, SglangSchedulerClient};
use tokio::sync::Mutex as TokioMutex;
use uuid::Uuid;

use super::{
    error::{set_error_message, SglErrorCode},
    grpc_converter::sgl_grpc_response_converter_create,
    runtime::RUNTIME,
    stream::SglangStreamHandle,
    tokenizer::TokenizerHandle,
    utils::chat_requires_reasoning,
};

/// FFI worker that implements the gateway's `Worker` trait so policies
/// can select workers using their real selection logic (not a fallback).
pub struct GrpcWorker {
    pub(crate) client: Arc<SglangSchedulerClient>,
    pub(crate) endpoint: String,
    pub(crate) status: AtomicU8,
    pub(crate) load: AtomicUsize,
    pub(crate) processed: AtomicUsize,
    pub(crate) circuit_breaker: CircuitBreaker,
    pub(crate) metadata: WorkerMetadata,
    pub(crate) routing_key_load: WorkerRoutingKeyLoad,
    pub(crate) api_key: Option<String>,
    pub(crate) http_client: Arc<reqwest::Client>,
    pub(crate) resilience: ResolvedResilience,
}

impl GrpcWorker {
    pub fn new(client: Arc<SglangSchedulerClient>, endpoint: String) -> Self {
        let mut spec = WorkerSpec::new(endpoint.clone());
        spec.connection_mode = ConnectionMode::Grpc;
        spec.runtime_type = RuntimeType::Sglang;

        let metadata = WorkerMetadata {
            spec: Arc::new(spec),
            health_config: HealthCheckConfig::default(),
            health_endpoint: "/health".to_string(),
            overload: OverloadThresholds::default(),
        };
        Self {
            client,
            routing_key_load: WorkerRoutingKeyLoad::new(&endpoint),
            endpoint,
            status: AtomicU8::new(WorkerStatus::Ready as u8),
            load: AtomicUsize::new(0),
            processed: AtomicUsize::new(0),
            circuit_breaker: CircuitBreaker::new(),
            metadata,
            api_key: None,
            http_client: Arc::new(reqwest::Client::new()),
            resilience: ResolvedResilience::default(),
        }
    }
}

impl std::fmt::Debug for GrpcWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcWorker")
            .field("endpoint", &self.endpoint)
            .field(
                "status",
                &WorkerStatus::from_u8(self.status.load(Ordering::Relaxed)),
            )
            .finish()
    }
}

#[async_trait]
impl Worker for GrpcWorker {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn url(&self) -> &str {
        &self.endpoint
    }

    fn api_key(&self) -> Option<&String> {
        self.api_key.as_ref()
    }

    fn worker_type(&self) -> &WorkerType {
        &self.metadata.spec.worker_type
    }

    fn connection_mode(&self) -> &ConnectionMode {
        &self.metadata.spec.connection_mode
    }

    fn status(&self) -> WorkerStatus {
        WorkerStatus::from_u8(self.status.load(Ordering::Relaxed))
    }

    fn set_status(&self, status: WorkerStatus) {
        self.status.store(status as u8, Ordering::Relaxed);
    }

    fn revision(&self) -> u64 {
        0
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        // FFI workers don't do their own health checks
        Ok(())
    }

    // FFI workers don't run the state machine — these counters are unused.
    // The Go SDK manages worker health via direct set_status() calls
    // through the `sgl_multi_client_set_worker_health` FFI entry point.
    fn consecutive_failures_increment(&self) -> usize {
        0
    }
    fn consecutive_failures_reset(&self) {}
    fn consecutive_successes_increment(&self) -> usize {
        0
    }
    fn consecutive_successes_reset(&self) {}
    fn total_pending_probes(&self) -> usize {
        0
    }
    fn total_pending_probes_increment(&self) -> usize {
        0
    }
    fn total_pending_probes_reset(&self) {}

    fn load(&self) -> usize {
        self.load.load(Ordering::Relaxed)
    }

    fn increment_load(&self) {
        self.load.fetch_add(1, Ordering::Relaxed);
    }

    fn decrement_load(&self) {
        self.load.fetch_sub(1, Ordering::Relaxed);
    }

    fn routing_key_load(&self) -> usize {
        self.routing_key_load.value()
    }

    fn routing_key_inflight(&self, routing_key: &str) -> usize {
        self.routing_key_load.key_inflight(routing_key)
    }

    fn increment_routing_key_load(&self, routing_key: &str) {
        self.routing_key_load.increment(routing_key);
    }

    fn decrement_routing_key_load(&self, routing_key: &str) {
        self.routing_key_load.decrement(routing_key);
    }

    fn processed_requests(&self) -> usize {
        self.processed.load(Ordering::Relaxed)
    }

    fn increment_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
    }

    fn metadata(&self) -> &WorkerMetadata {
        &self.metadata
    }

    fn circuit_breaker_state(&self) -> CircuitState {
        self.circuit_breaker.state()
    }

    fn circuit_breaker_can_execute(&self) -> bool {
        self.circuit_breaker.can_execute()
    }

    fn record_circuit_breaker_outcome(&self, success: bool) {
        self.circuit_breaker.record_outcome(success);
    }

    fn resilience(&self) -> &ResolvedResilience {
        &self.resilience
    }

    fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    fn http_client_handle_if_initialized(&self) -> Option<Arc<reqwest::Client>> {
        Some(Arc::clone(&self.http_client))
    }

    async fn get_backend_client(&self) -> WorkerResult<Option<Arc<BackendClient>>> {
        // Not used by policies — the FFI layer handles the backend directly.
        Ok(None)
    }

    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        Ok(self.is_healthy())
    }

    async fn zmq_health_check(&self) -> WorkerResult<bool> {
        Ok(self.is_healthy())
    }

    async fn http_health_check(&self) -> WorkerResult<bool> {
        Ok(self.is_healthy())
    }
}

/// Handle for a multi-worker client with load balancing.
///
/// Workers implement the gateway's `Worker` trait so that the real
/// `LoadBalancingPolicy::select_worker` is used — no fallback logic needed.
pub struct MultiWorkerClientHandle {
    /// Workers as trait objects so policies can use them directly
    pub(crate) workers: Vec<Arc<dyn Worker>>,
    /// Concrete workers for accessing the gRPC client
    pub(crate) grpc_workers: Vec<Arc<GrpcWorker>>,
    pub(crate) policy: Arc<dyn LoadBalancingPolicy>,
    pub(crate) tokenizer_path: String,
}

impl MultiWorkerClientHandle {
    /// Select a worker using the configured policy.
    ///
    /// Delegates to `LoadBalancingPolicy::select_worker` with real `Arc<dyn Worker>`
    /// objects, so all policies (round_robin, random, cache_aware, etc.) work natively.
    pub fn select_worker(&self, info: &SelectWorkerInfo) -> Option<Arc<GrpcWorker>> {
        let idx = self.policy.select_worker(&self.workers, info)?;
        Some(Arc::clone(&self.grpc_workers[idx]))
    }
}

/// Create a multi-worker client with load balancing
///
/// # Arguments
/// * `endpoints` - Comma-separated list of gRPC endpoints (e.g., "grpc://host1:20000,grpc://host2:20001")
/// * `tokenizer_path` - Path to tokenizer directory
/// * `policy_name` - Load balancing policy name ("round_robin", "random", "cache_aware")
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to MultiWorkerClientHandle on success, null on failure
///
/// # Safety
/// - All string arguments must be valid null-terminated C strings
/// - Caller owns the returned handle and must free it with `sgl_multi_client_free`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_create(
    endpoints: *const c_char,
    tokenizer_path: *const c_char,
    policy_name: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut MultiWorkerClientHandle {
    if endpoints.is_null() || tokenizer_path.is_null() || policy_name.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return ptr::null_mut();
    }

    let endpoints_str = match CStr::from_ptr(endpoints).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in endpoints");
            return ptr::null_mut();
        }
    };

    let tokenizer_path_str = match CStr::from_ptr(tokenizer_path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in tokenizer_path");
            return ptr::null_mut();
        }
    };

    let policy_name_str = match CStr::from_ptr(policy_name).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in policy_name");
            return ptr::null_mut();
        }
    };

    // Parse endpoints
    let endpoint_list: Vec<&str> = endpoints_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if endpoint_list.is_empty() {
        set_error_message(error_out, "No valid endpoints provided");
        return ptr::null_mut();
    }

    // Create policy
    //
    // Supported policies are those that work with the SDK's SelectWorkerInfo
    // (request_text + tokens). Policies requiring HTTP headers or a pre-computed
    // hash ring are not supported because the SDK operates below the HTTP layer.
    //
    // TODO: Support consistent_hashing, prefix_hash, and manual policies.
    // These require SelectWorkerInfo.headers and/or SelectWorkerInfo.hash_ring
    // which are not available in the SDK. To support them we would need to:
    // - Forward HTTP headers from the Go HTTP server through the FFI boundary
    // - Build and cache a HashRing from the worker list (like WorkerRegistry does)
    let policy: Arc<dyn LoadBalancingPolicy> = match policy_name_str {
        "round_robin" | "roundrobin" => Arc::new(RoundRobinPolicy::new()),
        "random" => Arc::new(RandomPolicy::new()),
        "power_of_two" | "poweroftwo" => Arc::new(PowerOfTwoPolicy::new()),
        "cache_aware" | "cacheaware" => Arc::new(CacheAwarePolicy::new()),
        "bucket" => Arc::new(BucketPolicy::new()),
        "consistent_hashing" | "consistenthashing" | "prefix_hash" | "prefixhash" | "manual" => {
            set_error_message(
                error_out,
                &format!(
                    "Policy '{policy_name_str}' is not supported in the SDK. It requires HTTP headers \
                     and/or a hash ring which are not available at the FFI layer. \
                     Supported policies: round_robin, random, power_of_two, cache_aware, bucket"
                ),
            );
            return ptr::null_mut();
        }
        _ => {
            set_error_message(
                error_out,
                &format!(
                    "Unknown policy: '{policy_name_str}'. \
                     Supported policies: round_robin, random, power_of_two, cache_aware, bucket"
                ),
            );
            return ptr::null_mut();
        }
    };

    // Create gRPC clients for all endpoints
    let mut grpc_workers = Vec::with_capacity(endpoint_list.len());
    let mut workers: Vec<Arc<dyn Worker>> = Vec::with_capacity(endpoint_list.len());
    for endpoint in endpoint_list {
        let client =
            match RUNTIME.block_on(async { SglangSchedulerClient::connect(endpoint).await }) {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    set_error_message(error_out, &format!("Failed to connect to {endpoint}: {e}"));
                    return ptr::null_mut();
                }
            };
        let grpc_worker = Arc::new(GrpcWorker::new(client, endpoint.to_string()));
        workers.push(Arc::clone(&grpc_worker) as Arc<dyn Worker>);
        grpc_workers.push(grpc_worker);
    }

    Box::into_raw(Box::new(MultiWorkerClientHandle {
        workers,
        grpc_workers,
        policy,
        tokenizer_path: tokenizer_path_str,
    }))
}

/// Free a multi-worker client handle
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`, or null
/// - `handle` must not be used after this call
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_free(handle: *mut MultiWorkerClientHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Get the number of workers in the multi-worker client
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_worker_count(
    handle: *mut MultiWorkerClientHandle,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    (*handle).grpc_workers.len()
}

/// Get the number of healthy workers in the multi-worker client
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_healthy_count(
    handle: *mut MultiWorkerClientHandle,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    (*handle)
        .grpc_workers
        .iter()
        .filter(|w| w.is_healthy())
        .count()
}

/// Mark a worker as unhealthy by index
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_set_worker_health(
    handle: *mut MultiWorkerClientHandle,
    worker_index: usize,
    healthy: bool,
) -> SglErrorCode {
    if handle.is_null() {
        return SglErrorCode::InvalidArgument;
    }
    let client = &*handle;
    if worker_index >= client.grpc_workers.len() {
        return SglErrorCode::InvalidArgument;
    }
    // The Go SDK is the source of truth for FFI worker health, so we
    // map its boolean directly without the legacy `set_healthy` guard
    // (which only demoted from `Ready`). FFI workers never run the
    // local state machine, so a `Pending` start state can be flipped
    // straight to `Ready` or `NotReady` here.
    let status = if healthy {
        WorkerStatus::Ready
    } else {
        WorkerStatus::NotReady
    };
    client.grpc_workers[worker_index].set_status(status);
    SglErrorCode::Success
}

/// Get the policy name
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
/// - Returned string must be freed with `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_policy_name(
    handle: *mut MultiWorkerClientHandle,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let policy_name = (*handle).policy.name();
    match CString::new(policy_name) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Get the tokenizer path from the multi-worker client
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_multi_client_create`
/// - Returned string must be freed with `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_tokenizer_path(
    handle: *mut MultiWorkerClientHandle,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    match CString::new((*handle).tokenizer_path.as_str()) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Send a chat completion request using load-balanced worker selection
///
/// # Arguments
/// * `client_handle` - Multi-worker client handle
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `stream_handle_out` - Pointer to receive stream handle
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `client_handle` must be a valid pointer returned by `sgl_multi_client_create`
/// - `request_json` must be a valid null-terminated C string containing valid JSON
/// - `stream_handle_out` must be a valid pointer to writable memory
/// - Caller owns the stream handle and must free it with `sgl_stream_free`
#[no_mangle]
pub unsafe extern "C" fn sgl_multi_client_chat_completion_stream(
    client_handle: *mut MultiWorkerClientHandle,
    request_json: *const c_char,
    stream_handle_out: *mut *mut SglangStreamHandle,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if client_handle.is_null() || request_json.is_null() || stream_handle_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    let multi_client = &*client_handle;

    // Create tokenizer
    let tokenizer: Arc<dyn Tokenizer> =
        match create_tokenizer_from_file(&multi_client.tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                set_error_message(error_out, &format!("Failed to create tokenizer: {e}"));
                return SglErrorCode::TokenizationError;
            }
        };

    // Parse OpenAI ChatCompletionRequest
    let mut chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {e}"));
            return SglErrorCode::ParsingError;
        }
    };

    // Process messages and apply chat template
    let processed_messages = match process_chat_messages(&chat_request, tokenizer.as_ref(), None) {
        Ok(msgs) => msgs,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to process messages: {e}"));
            return SglErrorCode::TokenizationError;
        }
    };

    // Tokenize
    let token_ids = match tokenizer.encode(&processed_messages.text, false) {
        Ok(encoding) => encoding.token_ids().to_vec(),
        Err(e) => {
            set_error_message(error_out, &format!("Failed to tokenize: {e}"));
            return SglErrorCode::TokenizationError;
        }
    };
    let prompt_tokens = token_ids.len() as u32;

    // Select a worker using the real policy with request context
    let select_info = SelectWorkerInfo {
        request_text: Some(&processed_messages.text),
        tokens: Some(&token_ids),
        ..Default::default()
    };
    let worker = match multi_client.select_worker(&select_info) {
        Some(w) => w,
        None => {
            set_error_message(error_out, "No healthy workers available");
            return SglErrorCode::UnknownError;
        }
    };

    // Track load so policies like cache_aware and power_of_two can make informed decisions
    worker.increment_load();

    let client = Arc::clone(&worker.client);

    // Generate tool constraints if needed
    let registry = super::runtime::PARSER_FACTORY.registry();
    let tool_constraint = if let (Some(tools), Some(tool_choice)) = (
        chat_request.tools.as_ref(),
        chat_request.tool_choice.as_ref(),
    ) {
        match registry.generate_tool_constraint(None, tools, tool_choice) {
            Ok(Some(c)) => Some(c.to_tuple()),
            Ok(None) => None,
            Err(e) => {
                set_error_message(
                    error_out,
                    &format!("Failed to generate tool constraints: {e}"),
                );
                return SglErrorCode::ParsingError;
            }
        }
    } else {
        None
    };

    // Derive skip_special_tokens from constraint type
    if tool_constraint
        .as_ref()
        .is_none_or(|c| c.0 != "json_schema")
        && chat_request.tools.is_some()
        && !matches!(
            chat_request.tool_choice,
            Some(openai_protocol::common::ToolChoice::Value(
                openai_protocol::common::ToolChoiceValue::None
            ))
        )
    {
        chat_request.skip_special_tokens = false;
    }

    // Build GenerateRequest
    let request_id = format!("chatcmpl-{}", Uuid::now_v7());
    let require_reasoning = chat_requires_reasoning(&chat_request, tokenizer.as_ref());
    let proto_request = match client.build_generate_request_from_chat(
        request_id.clone(),
        &chat_request,
        processed_messages.text,
        token_ids,
        SglangGenerateRequestOptions {
            multimodal_inputs: None, // multimodal not supported in golang bindings
            tool_call_constraint: tool_constraint,
            require_reasoning,
        },
    ) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to build generate request: {e}"));
            return SglErrorCode::ParsingError;
        }
    };

    // Send request and get stream
    let stream = match RUNTIME.block_on(async { client.generate(proto_request).await }) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to send request: {e}"));
            return SglErrorCode::UnknownError;
        }
    };

    // Create response converter
    // Use and_then with CString::new to safely handle potential null bytes in JSON strings
    let tools_json = chat_request
        .tools
        .as_ref()
        .and_then(|t| serde_json::to_string(t).ok())
        .and_then(|s| CString::new(s).ok())
        .map(|s| s.into_raw());
    let tool_choice_json = chat_request
        .tool_choice
        .as_ref()
        .and_then(|tc| serde_json::to_string(tc).ok())
        .and_then(|s| CString::new(s).ok())
        .map(|s| s.into_raw());
    let stop_json = chat_request
        .stop
        .as_ref()
        .and_then(|s| serde_json::to_string(s).ok())
        .and_then(|s| CString::new(s).ok())
        .map(|s| s.into_raw());
    let stop_token_ids_json = chat_request
        .stop_token_ids
        .as_ref()
        .and_then(|ids| serde_json::to_string(ids).ok())
        .and_then(|s| CString::new(s).ok())
        .map(|s| s.into_raw());

    // Create tokenizer handle for converter
    let tokenizer_handle = Box::into_raw(Box::new(TokenizerHandle {
        tokenizer: Arc::clone(&tokenizer),
    }));

    let model_cstr = match CString::new(chat_request.model.clone()) {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid model name: contains null byte");
            let _ = Box::from_raw(tokenizer_handle);
            return SglErrorCode::InvalidArgument;
        }
    };
    let request_id_cstr = match CString::new(request_id.clone()) {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid request ID: contains null byte");
            let _ = Box::from_raw(tokenizer_handle);
            return SglErrorCode::InvalidArgument;
        }
    };

    let converter = sgl_grpc_response_converter_create(
        tokenizer_handle,
        model_cstr.as_ptr(),
        request_id_cstr.as_ptr(),
        tools_json.unwrap_or(ptr::null_mut()),
        tool_choice_json.unwrap_or(ptr::null_mut()),
        stop_json.unwrap_or(ptr::null_mut()),
        stop_token_ids_json.unwrap_or(ptr::null_mut()),
        if chat_request.skip_special_tokens {
            1
        } else {
            0
        },
        error_out,
    );

    // Free temporary tokenizer handle (converter now owns the tokenizer)
    let _ = Box::from_raw(tokenizer_handle);

    if converter.is_null() {
        return SglErrorCode::MemoryError;
    }

    // Clean up temporary CStrings
    if let Some(ptr) = tools_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = tool_choice_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = stop_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = stop_token_ids_json {
        let _ = CString::from_raw(ptr);
    }

    // Create converter handle and set initial_prompt_tokens
    let mut converter_handle = *Box::from_raw(converter);
    converter_handle.initial_prompt_tokens = Some(prompt_tokens);

    // Create stream handle with worker reference for load tracking
    *stream_handle_out = Box::into_raw(Box::new(SglangStreamHandle {
        stream: Arc::new(TokioMutex::new(stream)),
        converter: Arc::new(TokioMutex::new(converter_handle)),
        client: Arc::clone(&client),
        prompt_tokens,
        worker: Some(Arc::clone(&worker)),
    }));

    SglErrorCode::Success
}
