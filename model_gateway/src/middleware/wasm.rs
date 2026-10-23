//! WASM plugin middleware.
//!
//! Dispatches request and response through every WASM module attached at
//! the corresponding `Middleware::OnRequest` / `Middleware::OnResponse`
//! point. Streaming responses skip the OnResponse phase to avoid buffering
//! arbitrary bodies into memory.

use std::{sync::Arc, time::Duration};

use axum::{
    body::{Body, HttpBody},
    extract::{Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use tracing::{error, warn};

use super::request_id::{generate_request_id, RequestId};
use crate::{
    routers::error,
    server::AppState,
    wasm::{
        module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
        module_manager::WasmModuleManager,
        spec::{
            apply_modify_action_to_headers, build_wasm_headers_from_axum_headers,
            smg::gateway::middleware_types::{
                Action, Request as WasmRequest, Response as WasmResponse,
            },
        },
        types::WasmComponentInput,
    },
};

pub async fn wasm_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if !app_state.context.router_config.enable_wasm {
        return next.run(request).await;
    }
    let Some(wasm_manager) = &app_state.context.wasm_manager else {
        return next.run(request).await;
    };

    // The request id the extensions carry, else one generated for the hooks.
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_else(|| generate_request_id(request.uri().path()));

    let request = match run_request_hook(wasm_manager, request, &request_id).await {
        RequestHook::Run(request) => request,
        RequestHook::RunUnhooked(request) => return next.run(request).await,
        RequestHook::Answer(response) => return response,
    };
    let response = next.run(request).await;
    run_response_hook(wasm_manager, response).await
}

/// What the request hook decided for a request.
enum RequestHook {
    /// Run the request as the modules left it, then the response hook.
    Run(Request<Body>),
    /// Run the request untouched and skip the response hook: the modules of
    /// the request hook could not be listed.
    RunUnhooked(Request<Body>),
    /// Answer with this response and do not run the request: a module
    /// rejected it, or its body could not be read.
    Answer(Response),
}

/// Run the modules attached at `OnRequest` over `request`: the request is
/// taken apart (its extensions kept aside), its body read within the
/// manager's body limit, each module shown the method, path, query, headers
/// and body as the modules before it left them, and the request put back
/// together with the modifications and its original extensions.
async fn run_request_hook(
    wasm_manager: &WasmModuleManager,
    request: Request<Body>,
    request_id: &str,
) -> RequestHook {
    let attach_point = WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);
    let modules = match wasm_manager.get_modules_by_attach_point(attach_point.clone()) {
        Ok(modules) => modules,
        Err(e) => {
            error!("Failed to get WASM modules for OnRequest: {}", e);
            return RequestHook::RunUnhooked(request);
        }
    };
    if modules.is_empty() {
        return RequestHook::Run(request);
    }

    // Decompose request to preserve extensions across reconstruction
    let (parts, body) = request.into_parts();
    let method = parts.method;
    let uri = parts.uri;
    let mut headers = parts.headers;
    let extensions = parts.extensions;

    let max_body_size = wasm_manager.get_max_body_size();
    let mut body_bytes = match axum::body::to_bytes(body, max_body_size).await {
        Ok(bytes) => bytes.to_vec(),
        Err(e) if super::body_timeout::crossed_payload_limit(&e) => {
            // The edge limit or the WASM cap: the gateway's answer for a body
            // over a limit, not a read failure.
            warn!("Request body over the limit before WASM processing: {}", e);
            return RequestHook::Answer(error::create_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                super::body_timeout::REQUEST_BODY_TOO_LARGE,
                "Request body exceeded the payload limit",
            ));
        }
        Err(e) => {
            error!("Failed to read request body for WASM processing: {}", e);
            return RequestHook::Answer(error::bad_request(
                "wasm_request_body_unreadable",
                format!("Failed to read request body: {e}"),
            ));
        }
    };

    // Pre-compute strings once before the loop to avoid repeated allocations
    let method_str = method.to_string();
    let path_str = uri.path().to_string();
    let query_str = uri.query().unwrap_or("").to_string();

    for module in modules {
        let wasm_request = WasmRequest {
            method: method_str.clone(),
            path: path_str.clone(),
            query: query_str.clone(),
            headers: build_wasm_headers_from_axum_headers(&headers),
            body: body_bytes.clone(),
            request_id: request_id.to_string(),
            now_epoch_ms: now_epoch_ms(),
        };
        let Some(action) = wasm_manager
            .execute_module_for_attach_point(
                &module,
                attach_point.clone(),
                WasmComponentInput::MiddlewareRequest(wasm_request),
            )
            .await
        else {
            continue;
        };
        match action {
            Action::Continue => {}
            Action::Reject(status) => {
                return RequestHook::Answer(
                    StatusCode::from_u16(status)
                        .unwrap_or(StatusCode::BAD_REQUEST)
                        .into_response(),
                );
            }
            Action::Modify(modify) => {
                apply_modify_action_to_headers(&mut headers, &modify);
                if let Some(new_body) = modify.body_replace {
                    body_bytes = new_body;
                }
            }
        }
    }

    // Reconstruct request with modifications, preserving original extensions
    let mut final_request = Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::from(body_bytes))
        .unwrap_or_else(|_| Request::new(Body::empty()));
    *final_request.headers_mut() = headers;
    *final_request.extensions_mut() = extensions;
    RequestHook::Run(final_request)
}

/// Run the modules attached at `OnResponse` over `response`: a streaming
/// response (the hook would have to buffer the whole stream), or one known
/// before the read to be over the manager's body limit (a declared length,
/// a body already in memory), passes untouched; otherwise the body is read
/// within that limit, each module shown the status, headers and body as the
/// modules before it left them, and the response put back together with
/// the modifications.
async fn run_response_hook(wasm_manager: &WasmModuleManager, response: Response) -> Response {
    let attach_point = WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse);
    let modules = match wasm_manager.get_modules_by_attach_point(attach_point.clone()) {
        Ok(modules) => modules,
        Err(e) => {
            error!("Failed to get WASM modules for OnResponse: {}", e);
            return response;
        }
    };
    if modules.is_empty() {
        return response;
    }

    // Skip WASM OnResponse processing for streaming responses to avoid
    // buffering the entire stream into memory (breaks SSE, causes OOM on large streams).
    if is_streaming(&response) {
        warn!("Skipping WASM OnResponse for streaming response; OnResponse modules do not apply to streaming");
        return response;
    }

    // A response the WASM cap cannot hold passes through unhooked, as a
    // streaming one does: the upstream answered, and the modules do not see
    // bodies over the cap. A declared length or a body already in memory
    // says so before the read.
    let max_body_size = wasm_manager.get_max_body_size();
    if known_to_exceed(&response, max_body_size) {
        warn!(
            "Skipping WASM OnResponse for a response over the body limit ({} bytes); OnResponse modules do not apply to it",
            max_body_size
        );
        return response;
    }

    // Extract response data once before processing modules
    let mut status = response.status();
    let mut headers = response.headers().clone();
    let mut body_bytes = match axum::body::to_bytes(response.into_body(), max_body_size).await {
        Ok(bytes) => bytes.to_vec(),
        Err(e) if super::body_timeout::crossed_payload_limit(&e) => {
            // A size unknown before the read, over the cap: the body is
            // consumed by now, so the answer is a refusal, under its own code.
            error!(
                "Response body over the WASM body limit ({} bytes) whose size was not known before the read; the response cannot be relayed",
                max_body_size
            );
            return error::internal_error(
                "wasm_response_body_too_large",
                "Response body exceeded the WASM middleware's body limit",
            );
        }
        Err(e) => {
            error!("Failed to read response body for WASM processing: {}", e);
            return error::internal_error(
                "wasm_response_body_unreadable",
                "Failed to read response body",
            );
        }
    };

    for module in modules {
        let wasm_response = WasmResponse {
            status: status.as_u16(),
            headers: build_wasm_headers_from_axum_headers(&headers),
            body: body_bytes.clone(),
        };
        let Some(action) = wasm_manager
            .execute_module_for_attach_point(
                &module,
                attach_point.clone(),
                WasmComponentInput::MiddlewareResponse(wasm_response),
            )
            .await
        else {
            continue;
        };
        match action {
            Action::Continue => {}
            Action::Reject(status_code) => {
                let status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::BAD_REQUEST);
                return rebuild_response(status, headers, body_bytes);
            }
            Action::Modify(modify) => {
                if let Some(new_status) = modify.status {
                    status = StatusCode::from_u16(new_status).unwrap_or(status);
                }
                apply_modify_action_to_headers(&mut headers, &modify);
                if let Some(new_body) = modify.body_replace {
                    body_bytes = new_body;
                }
            }
        }
    }

    // Reconstruct final response with all modifications
    rebuild_response(status, headers, body_bytes)
}

/// Whether `response` is known to be over `max_body_size` before it is read:
/// by the length it declares, or by the lower bound of its body's size hint,
/// exact for a body already in memory and zero for a stream.
fn known_to_exceed(response: &Response, max_body_size: usize) -> bool {
    let declared: Option<usize> = response
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());
    declared.is_some_and(|length| length > max_body_size)
        || response.body().size_hint().lower() > max_body_size as u64
}

/// Whether `response` streams: an event stream, newline-delimited JSON or a
/// chunked transfer, none of which the response hook may buffer.
fn is_streaming(response: &Response) -> bool {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream") || ct.contains("application/x-ndjson"))
        || response
            .headers()
            .get(header::TRANSFER_ENCODING)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|te| te.contains("chunked"))
}

/// The response the modules left: `status`, `headers` and `body`.
fn rebuild_response(status: StatusCode, headers: HeaderMap, body: Vec<u8>) -> Response {
    let mut response = Response::builder()
        .status(status)
        .body(Body::from(body))
        .unwrap_or_else(|_| Response::new(Body::empty()));
    *response.headers_mut() = headers;
    response
}

/// The wall clock in milliseconds since the epoch, as the modules read it.
fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_millis(0))
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use std::{io, sync::Arc};

    use axum::{
        body::{Body, Bytes},
        extract::Request,
        http::{header, StatusCode},
        response::Response,
    };
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use uuid::Uuid;

    use super::{is_streaming, run_request_hook, run_response_hook, RequestHook};
    use crate::wasm::{
        module::{MiddlewareAttachPoint, WasmModule, WasmModuleAttachPoint, WasmModuleMeta},
        module_manager::WasmModuleManager,
        WasmRuntimeConfig,
    };

    /// A request extension the hook must carry across its reconstruction.
    #[derive(Clone, Debug, PartialEq)]
    struct Marker(&'static str);

    /// A manager with one module at both middleware hooks whose component is
    /// valid but exports nothing, so running it yields no action: the hooks
    /// walk their whole path (list, take apart, read, run, put back
    /// together) with no module changing the message.
    fn manager_with_an_inert_module() -> WasmModuleManager {
        manager_with_an_inert_module_under(WasmRuntimeConfig::default())
    }

    /// [`manager_with_an_inert_module`] under `config` (a small body cap, say).
    fn manager_with_an_inert_module_under(config: WasmRuntimeConfig) -> WasmModuleManager {
        use wasm_encoder::{Component, Module, ModuleSection, TypeSection};
        let mut module = Module::new();
        module.section(&TypeSection::new());
        let mut component = Component::new();
        component.section(&ModuleSection(&module));
        let bytes = component.as_slice().to_vec();
        let manager = WasmModuleManager::new(config);
        manager
            .register_module_internal(WasmModule {
                module_uuid: Uuid::now_v7(),
                module_meta: WasmModuleMeta {
                    name: "inert".to_string(),
                    file_path: String::new(),
                    sha256_hash: [0; 32],
                    size_bytes: bytes.len() as u64,
                    created_at: 0,
                    last_accessed_at: 0,
                    access_count: 0,
                    attach_points: vec![
                        WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest),
                        WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse),
                    ],
                    wasm_bytes: Arc::new(bytes),
                },
            })
            .expect("the inert module registers");
        manager
    }

    async fn body_of(body: Body) -> Vec<u8> {
        axum::body::to_bytes(body, usize::MAX)
            .await
            .expect("the body reads")
            .to_vec()
    }

    /// The request hook puts a request back together with its extensions,
    /// method, uri, headers and body when no module changes it.
    #[tokio::test]
    async fn the_request_hook_keeps_the_requests_extensions_and_parts() {
        let manager = manager_with_an_inert_module();
        let mut request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions?x=1")
            .header("x-kept", "yes")
            .body(Body::from("{\"k\":1}"))
            .expect("a request");
        request.extensions_mut().insert(Marker("kept"));

        let RequestHook::Run(request) = run_request_hook(&manager, request, "req-1").await else {
            panic!("the request runs");
        };
        assert_eq!(manager.get_metrics().0, 1, "the module ran once");
        assert_eq!(request.extensions().get::<Marker>(), Some(&Marker("kept")));
        assert_eq!(request.method(), "POST");
        assert_eq!(request.uri(), "/v1/chat/completions?x=1");
        assert_eq!(
            request.headers().get("x-kept").map(|v| v.as_bytes()),
            Some(&b"yes"[..])
        );
        assert_eq!(body_of(request.into_body()).await, b"{\"k\":1}");
    }

    /// A manager without a module at the request hook passes the request on
    /// as it is.
    #[tokio::test]
    async fn the_request_hook_passes_a_request_through_an_empty_hook() {
        let manager = WasmModuleManager::with_default_config();
        let request = Request::builder()
            .uri("/v1/models")
            .body(Body::from("body"))
            .expect("a request");
        let RequestHook::Run(request) = run_request_hook(&manager, request, "req-2").await else {
            panic!("the request runs");
        };
        assert_eq!(request.uri(), "/v1/models");
        assert_eq!(body_of(request.into_body()).await, b"body");
    }

    /// A streaming response passes the response hook untouched, and no
    /// module runs: an event stream, newline-delimited JSON, or a chunked
    /// transfer.
    #[tokio::test]
    async fn the_response_hook_leaves_a_streaming_response_alone() {
        let manager = manager_with_an_inert_module();
        for (name, value) in [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CONTENT_TYPE, "application/x-ndjson"),
            (header::TRANSFER_ENCODING, "chunked"),
        ] {
            let response = Response::builder()
                .status(StatusCode::OK)
                .header(name.clone(), value)
                .header("x-stream", "1")
                .body(Body::from("data: x\n\n"))
                .expect("a response");
            let response = run_response_hook(&manager, response).await;
            assert_eq!(manager.get_metrics().0, 0, "{value}: no module ran");
            assert_eq!(response.status(), StatusCode::OK, "{value}");
            assert_eq!(
                response.headers().get(&name).map(|v| v.as_bytes()),
                Some(value.as_bytes()),
                "{value}"
            );
            assert_eq!(
                response.headers().get("x-stream").map(|v| v.as_bytes()),
                Some(&b"1"[..])
            );
            assert_eq!(
                body_of(response.into_body()).await,
                b"data: x\n\n",
                "{value}"
            );
        }
    }

    /// A plain response is read and put back together with its status,
    /// headers and body when no module changes it.
    #[tokio::test]
    async fn the_response_hook_rebuilds_a_plain_response_unchanged() {
        let manager = manager_with_an_inert_module();
        let response = Response::builder()
            .status(StatusCode::CREATED)
            .header(header::CONTENT_TYPE, "application/json")
            .header("x-kept", "yes")
            .body(Body::from("{\"ok\":true}"))
            .expect("a response");
        let response = run_response_hook(&manager, response).await;
        assert_eq!(manager.get_metrics().0, 1, "the module ran once");
        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(
            response.headers().get("x-kept").map(|v| v.as_bytes()),
            Some(&b"yes"[..])
        );
        assert_eq!(body_of(response.into_body()).await, b"{\"ok\":true}");
    }

    /// The streaming check reads the content type and the transfer encoding.
    #[test]
    fn a_response_streams_by_its_content_type_or_transfer_encoding() {
        let with = |name: header::HeaderName, value: &str| {
            Response::builder()
                .header(name, value)
                .body(Body::empty())
                .expect("a response")
        };
        assert!(is_streaming(&with(
            header::CONTENT_TYPE,
            "text/event-stream; charset=utf-8"
        )));
        assert!(is_streaming(&with(
            header::CONTENT_TYPE,
            "application/x-ndjson"
        )));
        assert!(is_streaming(&with(header::TRANSFER_ENCODING, "chunked")));
        assert!(!is_streaming(&with(
            header::CONTENT_TYPE,
            "application/json"
        )));
        assert!(!is_streaming(&Response::new(Body::empty())));
    }

    /// The status, the `X-SMG-Error-Code` header and the JSON body of
    /// `response`.
    async fn envelope_of(response: Response) -> (StatusCode, String, serde_json::Value) {
        let status = response.status();
        let code = response
            .headers()
            .get("x-smg-error-code")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let body: serde_json::Value =
            serde_json::from_slice(&body_of(response.into_body()).await).expect("a JSON body");
        (status, code, body)
    }

    fn unreadable_body() -> Body {
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(1);
        tx.try_send(Err(io::Error::other("the stream broke")))
            .expect("the channel takes one item");
        drop(tx);
        Body::from_stream(ReceiverStream::new(rx))
    }

    /// A request body that cannot be read answers in the gateway's error
    /// envelope, not a bare text under `error`: a 400 with the code
    /// `wasm_request_body_unreadable` and the envelope's `type`.
    #[tokio::test]
    async fn the_request_hook_answers_an_unreadable_body_with_the_error_envelope() {
        let manager = manager_with_an_inert_module();
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .body(unreadable_body())
            .expect("a request");
        let RequestHook::Answer(response) = run_request_hook(&manager, request, "req-1").await
        else {
            panic!("an unreadable body is answered, not forwarded");
        };
        let (status, code, body) = envelope_of(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(code, "wasm_request_body_unreadable");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "wasm_request_body_unreadable");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with("Failed to read request body")),
            "{body}"
        );
    }

    /// A response body that cannot be read answers in the gateway's error
    /// envelope: a 500 with the code `wasm_response_body_unreadable` and the
    /// envelope's `type`.
    #[tokio::test]
    async fn the_response_hook_answers_an_unreadable_body_with_the_error_envelope() {
        let manager = manager_with_an_inert_module();
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(unreadable_body())
            .expect("a response");
        let response = run_response_hook(&manager, response).await;
        let (status, code, body) = envelope_of(response).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(code, "wasm_response_body_unreadable");
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "wasm_response_body_unreadable");
        assert_eq!(body["error"]["message"], "Failed to read response body");
    }

    fn small_cap() -> WasmRuntimeConfig {
        WasmRuntimeConfig {
            max_body_size: 1024,
            ..WasmRuntimeConfig::default()
        }
    }

    /// A request body over a limit (the edge limit or the WASM cap) is the
    /// gateway's 413 `request_body_too_large`, not a read failure.
    #[tokio::test]
    async fn the_request_hook_answers_a_body_over_the_cap_with_the_gateways_413() {
        let manager = manager_with_an_inert_module_under(small_cap());
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .body(Body::from(vec![b'x'; 4096]))
            .expect("a request");
        let RequestHook::Answer(response) = run_request_hook(&manager, request, "req-2").await
        else {
            panic!("a body over the cap is answered, not forwarded");
        };
        let (status, code, body) = envelope_of(response).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(code, "request_body_too_large");
        assert_eq!(body["error"]["code"], "request_body_too_large");
    }

    /// A response that declares a length over the cap passes through the hook
    /// untouched, as a streaming response does.
    #[tokio::test]
    async fn the_response_hook_passes_a_declared_body_over_the_cap_through() {
        let manager = manager_with_an_inert_module_under(small_cap());
        let payload = vec![b'y'; 4096];
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, payload.len())
            .body(Body::from(payload.clone()))
            .expect("a response");
        let response = run_response_hook(&manager, response).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_of(response.into_body()).await, payload);
    }

    /// A response whose body is already in memory, over the cap, passes
    /// through as a declared one does though it declares no length: a typed
    /// JSON response carries none before it is written out.
    #[tokio::test]
    async fn the_response_hook_passes_an_in_memory_body_over_the_cap_through() {
        let manager = manager_with_an_inert_module_under(small_cap());
        let payload = vec![b'y'; 4096];
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(payload.clone()))
            .expect("a response");
        assert!(response.headers().get(header::CONTENT_LENGTH).is_none());
        let response = run_response_hook(&manager, response).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_of(response.into_body()).await, payload);
    }

    /// A response over the cap whose size is not known before the read (a
    /// stream) cannot be relayed once read: its refusal carries its own code,
    /// not the unreadable one.
    #[tokio::test]
    async fn the_response_hook_names_a_streamed_body_over_the_cap() {
        let manager = manager_with_an_inert_module_under(small_cap());
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(4);
        for _ in 0..3 {
            tx.try_send(Ok(Bytes::from(vec![b'z'; 1024])))
                .expect("the channel takes the chunks");
        }
        drop(tx);
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from_stream(ReceiverStream::new(rx)))
            .expect("a response");
        let response = run_response_hook(&manager, response).await;
        let (status, code, body) = envelope_of(response).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(code, "wasm_response_body_too_large");
        assert_eq!(body["error"]["code"], "wasm_response_body_too_large");
    }
}
