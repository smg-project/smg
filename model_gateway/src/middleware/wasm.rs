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
    http::{header, response::Parts, HeaderMap, HeaderValue, StatusCode},
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
                Action, ModifyAction, Request as WasmRequest, Response as WasmResponse,
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
    let mut body_replaced = false;
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
                body_replaced |= apply_modify(&mut headers, &mut body_bytes, modify);
            }
        }
    }

    // Reconstruct request with modifications, preserving original extensions
    frame_body(&mut headers, body_bytes.len(), body_replaced);
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

    // Take the response apart once before the modules run; the parts carry
    // the extensions the layers inside handed the edge (the admission permit,
    // the request span, the keep-alive), which ride on into the rebuilt one.
    let (mut parts, body) = response.into_parts();
    let mut status = parts.status;
    let mut headers = std::mem::take(&mut parts.headers);
    let mut body_replaced = false;
    let mut body_bytes = match axum::body::to_bytes(body, max_body_size).await {
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
                return rebuild_response(parts, status, headers, body_bytes, body_replaced);
            }
            Action::Modify(modify) => {
                if let Some(new_status) = modify.status {
                    status = StatusCode::from_u16(new_status).unwrap_or(status);
                }
                body_replaced |= apply_modify(&mut headers, &mut body_bytes, modify);
            }
        }
    }

    // Reconstruct final response with all modifications
    rebuild_response(parts, status, headers, body_bytes, body_replaced)
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

/// The response the modules left: `status`, `headers` and `body` on the
/// original `parts`, so the extensions the layers inside handed the edge
/// ride on; the headers frame the body as it is now.
fn rebuild_response(
    mut parts: Parts,
    status: StatusCode,
    mut headers: HeaderMap,
    body: Vec<u8>,
    body_replaced: bool,
) -> Response {
    frame_body(&mut headers, body.len(), body_replaced);
    parts.status = status;
    parts.headers = headers;
    Response::from_parts(parts, Body::from(body))
}

/// Apply a module's `Modify` to the headers and body in hand, and say whether
/// the body was replaced. A replacement first drops the content encoding
/// declared for the body it replaces, then the module's own header edits
/// apply, so a module that encodes its replacement and says so keeps its
/// word; a body left in place keeps its encoding.
fn apply_modify(headers: &mut HeaderMap, body: &mut Vec<u8>, modify: ModifyAction) -> bool {
    if modify.body_replace.is_some() {
        headers.remove(header::CONTENT_ENCODING);
    }
    apply_modify_action_to_headers(headers, &modify);
    match modify.body_replace {
        Some(new_body) => {
            *body = new_body;
            true
        }
        None => false,
    }
}

/// The framing of a body the hook forwards whole: its length is the body's
/// (an upstream's declaration described the body it replaced), and a
/// transfer encoding no longer applies to a body held in memory; the
/// content encoding is the body's own, settled when a module's `Modify` was
/// applied. An empty body no module touched keeps the length it came with:
/// a HEAD answer or a 304 declares the entity it stands for, and a bodiless
/// request declares nothing.
fn frame_body(headers: &mut HeaderMap, len: usize, body_replaced: bool) {
    if body_replaced || len > 0 {
        headers.insert(header::CONTENT_LENGTH, HeaderValue::from(len));
    }
    headers.remove(header::TRANSFER_ENCODING);
}

/// The wall clock in milliseconds since the epoch, as the modules read it.
fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_millis(0))
        .as_millis() as u64
}

#[cfg(test)]
mod tests;
