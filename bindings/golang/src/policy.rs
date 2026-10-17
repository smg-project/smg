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
use openai_protocol::worker::{HealthCheckConfig, WorkerSpec, WorkerStatus};
use smg::{
    policies::{
        BucketPolicy, CacheAwarePolicy, LoadBalancingPolicy, PowerOfTwoPolicy, RandomPolicy,
        RoundRobinPolicy, SelectWorkerInfo,
    },
    routers::grpc::backend_client::BackendClient,
    worker::{
        circuit_breaker::{CircuitBreaker, CircuitState},
        resilience::ResolvedResilience,
        worker::{RuntimeType, WorkerMetadata, WorkerRoutingKeyLoad},
        ConnectionMode, OverloadThresholds, PdPairing, Worker, WorkerResult, WorkerType,
    },
};
use smg_grpc_client::sglang_scheduler::SglangSchedulerClient;

use super::{
    chat_stream::{prepare_chat, start_chat_stream},
    error::{set_error_message, SglErrorCode},
    runtime::RUNTIME,
    stream::SglangStreamHandle,
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
            pd_pairing: PdPairing::derive(&spec),
            spec: Arc::new(spec),
            health_config: HealthCheckConfig::default(),
            health_endpoint: "/health".to_string(),
            overload: OverloadThresholds::default(),
            http2: false,
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
            http_client: Arc::new(smg::http_client::client()),
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

    fn try_increment_load(&self, max: usize) -> bool {
        self.load
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1).filter(|next| *next <= max)
            })
            .is_ok()
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

/// One request's unit of load on the worker a policy selected: counted on
/// the worker when taken, given back when this drops. The stream handle
/// carries it until the caller frees the stream; a request that fails
/// before its stream opens drops it on the way out, so a worker whose
/// engine refuses requests carries no load from them (a count that would
/// steer `power_of_two` and `cache_aware` away from it after it recovers).
pub(crate) struct HeldLoad {
    worker: Arc<GrpcWorker>,
}

impl HeldLoad {
    /// Count one request on `worker`.
    pub(crate) fn take(worker: Arc<GrpcWorker>) -> Self {
        worker.increment_load();
        Self { worker }
    }

    /// Give the load back; the worker, for what a stream still counts on it.
    pub(crate) fn release(self) -> Arc<GrpcWorker> {
        let worker = Arc::clone(&self.worker);
        drop(self);
        worker
    }
}

impl Drop for HeldLoad {
    fn drop(&mut self) {
        self.worker.decrement_load();
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
    // The workers' HTTP clients use the process-level TLS provider the
    // gateway binary installs at start-up; this library has no start-up.
    smg::tls::install_crypto_provider();
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

    let prepared = match prepare_chat(request_str, tokenizer.as_ref()) {
        Ok(prepared) => prepared,
        Err((code, message)) => {
            set_error_message(error_out, &message);
            return code;
        }
    };

    // Select a worker using the real policy with request context
    let select_info = SelectWorkerInfo {
        request_text: Some(&prepared.text),
        tokens: Some(&prepared.token_ids),
        ..Default::default()
    };
    let worker = match multi_client.select_worker(&select_info) {
        Some(w) => w,
        None => {
            set_error_message(error_out, "No healthy workers available");
            return SglErrorCode::UnknownError;
        }
    };

    // The request counts on its worker from the selection on, so
    // power_of_two and cache_aware see it: the stream carries the count
    // until the caller frees it, and a failure before the stream opens
    // gives it back on the way out.
    let load = HeldLoad::take(Arc::clone(&worker));

    match start_chat_stream(&worker.client, &tokenizer, prepared, Some(load), error_out) {
        Ok(handle) => {
            *stream_handle_out = Box::into_raw(Box::new(handle));
            SglErrorCode::Success
        }
        Err(code) => code,
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use serde_json::{json, Value};
    use smg::policies::RoundRobinPolicy;
    use tokio::{net::TcpListener, task::JoinHandle};

    use super::{super::memory::sgl_free_string, *};

    /// A listener that takes connections and never answers: a client's
    /// `connect` succeeds against it (the HTTP/2 preface is written, not
    /// read back), and once it is closed every request on the connection
    /// fails at the send.
    struct SilentListener {
        endpoint: String,
        accepting: JoinHandle<()>,
    }

    impl SilentListener {
        fn bind() -> Self {
            let listener = RUNTIME
                .block_on(TcpListener::bind("127.0.0.1:0"))
                .expect("bind loopback");
            let endpoint = format!("grpc://{}", listener.local_addr().expect("local addr"));
            let accepting = RUNTIME.spawn(async move {
                let mut open = Vec::new();
                while let Ok((socket, _)) = listener.accept().await {
                    open.push(socket);
                }
            });
            Self {
                endpoint,
                accepting,
            }
        }

        /// Close the listener and every connection it took.
        fn close(self) {
            self.accepting.abort();
            let _ = RUNTIME.block_on(self.accepting);
        }
    }

    fn connect(endpoint: &str) -> Arc<SglangSchedulerClient> {
        // The worker's HTTP client wants the process-level TLS provider the
        // create entry point installs.
        smg::tls::install_crypto_provider();
        Arc::new(
            RUNTIME
                .block_on(SglangSchedulerClient::connect(endpoint))
                .expect("connect"),
        )
    }

    /// A policy client over one worker at `endpoint`, with the test
    /// tokenizer, and that worker.
    fn client_over(
        client: &Arc<SglangSchedulerClient>,
        endpoint: &str,
    ) -> (MultiWorkerClientHandle, Arc<GrpcWorker>) {
        let worker = Arc::new(GrpcWorker::new(Arc::clone(client), endpoint.to_string()));
        let handle = MultiWorkerClientHandle {
            workers: vec![Arc::clone(&worker) as Arc<dyn Worker>],
            grpc_workers: vec![Arc::clone(&worker)],
            policy: Arc::new(RoundRobinPolicy::new()),
            tokenizer_path: "mock".to_string(),
        };
        (handle, worker)
    }

    /// The chat entry point's answer to a request that fails: its code and
    /// message.
    fn failed_chat(
        handle: &mut MultiWorkerClientHandle,
        request: &Value,
    ) -> (SglErrorCode, String) {
        let request = CString::new(request.to_string()).unwrap();
        let mut stream: *mut SglangStreamHandle = ptr::null_mut();
        let mut error: *mut c_char = ptr::null_mut();
        // SAFETY: every pointer is a live local; the message is taken back
        // once, through the entry point that frees it.
        unsafe {
            let code = sgl_multi_client_chat_completion_stream(
                handle,
                request.as_ptr(),
                &mut stream,
                &mut error,
            );
            assert!(stream.is_null(), "no stream handle on a failure");
            let message = if error.is_null() {
                String::new()
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            };
            sgl_free_string(error);
            (code, message)
        }
    }

    /// The load a request holds is counted when taken and given back when
    /// it drops, or when it is released for the stream's bookkeeping.
    #[test]
    fn a_held_load_is_given_back_when_it_drops_or_is_released() {
        let listener = SilentListener::bind();
        let client = connect(&listener.endpoint);
        let worker = Arc::new(GrpcWorker::new(client, listener.endpoint.clone()));
        let held = HeldLoad::take(Arc::clone(&worker));
        let other = HeldLoad::take(Arc::clone(&worker));
        assert_eq!(worker.load(), 2);
        drop(held);
        assert_eq!(worker.load(), 1);
        assert!(Arc::ptr_eq(&other.release(), &worker));
        assert_eq!(worker.load(), 0);
        listener.close();
    }

    /// A request that fails after the policy selected its worker, at any of
    /// the three points between the selection and the open stream (the
    /// tool constraint, the generate request, the send), leaves the
    /// worker's load as it found it, and counts nothing as processed.
    #[test]
    fn a_request_that_fails_after_selection_gives_the_workers_load_back() {
        let mut listener = Some(SilentListener::bind());
        let endpoint = listener.as_ref().unwrap().endpoint.clone();
        let client = connect(&endpoint);

        let plain = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        let mut unknown_function = plain.clone();
        unknown_function["tools"] = json!([{"type": "function", "function":
            {"name": "get_time", "parameters": {"type": "object"}}}]);
        unknown_function["tool_choice"] =
            json!({"type": "function", "function": {"name": "missing"}});
        let mut two_constraints = plain.clone();
        two_constraints["response_format"] = json!({"type": "json_object"});
        two_constraints["ebnf"] = json!("root ::= \"a\"");
        let cases = [
            (
                "tool constraints",
                unknown_function,
                SglErrorCode::ParsingError,
                "Failed to generate tool constraints: ",
            ),
            (
                "generate request",
                two_constraints,
                SglErrorCode::ParsingError,
                "Failed to build generate request: ",
            ),
            (
                "send",
                plain,
                SglErrorCode::UnknownError,
                "Failed to send request: ",
            ),
        ];
        let mut after = Vec::new();
        for (point, request, expected_code, prefix) in cases {
            if point == "send" {
                // The engine goes away under the connected client.
                listener.take().unwrap().close();
            }
            let (mut handle, worker) = client_over(&client, &endpoint);
            for _ in 0..3 {
                let (code, message) = failed_chat(&mut handle, &request);
                assert_eq!(code, expected_code, "{point}: {message}");
                assert!(message.starts_with(prefix), "{point}: {message}");
            }
            after.push((point, worker.load(), worker.processed_requests()));
        }
        assert_eq!(
            after,
            [
                ("tool constraints", 0, 0),
                ("generate request", 0, 0),
                ("send", 0, 0),
            ],
            "(failure point, load, processed) after three failed requests"
        );
    }
}
