use std::{io, sync::Arc};

use axum::{
    body::{Body, Bytes},
    extract::Request,
    http::{header, HeaderMap, StatusCode},
    response::Response,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use super::{
    apply_modify, frame_body, is_streaming, rebuild_response, run_request_hook, run_response_hook,
    RequestHook,
};
use crate::wasm::{
    module::{MiddlewareAttachPoint, WasmModule, WasmModuleAttachPoint, WasmModuleMeta},
    module_manager::WasmModuleManager,
    spec::smg::gateway::middleware_types::{Header, ModifyAction},
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
    let RequestHook::Answer(response) = run_request_hook(&manager, request, "req-1").await else {
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
    let RequestHook::Answer(response) = run_request_hook(&manager, request, "req-2").await else {
        panic!("a body over the cap is answered, not forwarded");
    };
    let (status, code, body) = envelope_of(response).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(code, "request_body_too_large");
    assert_eq!(body["error"]["code"], "request_body_too_large");
}

/// A body the hook forwards is framed as it is now: the length is the
/// body's, not the one the upstream declared for the body a module
/// replaced; a transfer encoding no longer applies to a body held whole;
/// the content encoding is not the framing's to touch.
#[test]
fn a_forwarded_body_is_framed_as_it_is_now() {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_LENGTH, "5".parse().unwrap());
    headers.insert(header::TRANSFER_ENCODING, "chunked".parse().unwrap());
    frame_body(&mut headers, 12, true);
    assert_eq!(headers[header::CONTENT_LENGTH], "12");
    assert!(headers.get(header::TRANSFER_ENCODING).is_none());

    let mut kept = HeaderMap::new();
    kept.insert(header::CONTENT_ENCODING, "gzip".parse().unwrap());
    frame_body(&mut kept, 7, false);
    assert_eq!(kept[header::CONTENT_LENGTH], "7");
    assert_eq!(kept[header::CONTENT_ENCODING], "gzip");

    let mut head = HeaderMap::new();
    head.insert(header::CONTENT_LENGTH, "5".parse().unwrap());
    frame_body(&mut head, 0, false);
    assert_eq!(
        head[header::CONTENT_LENGTH],
        "5",
        "a HEAD answer keeps the entity's length"
    );
    let mut bodiless = HeaderMap::new();
    frame_body(&mut bodiless, 0, false);
    assert!(bodiless.get(header::CONTENT_LENGTH).is_none());
    let mut emptied = HeaderMap::new();
    emptied.insert(header::CONTENT_LENGTH, "5".parse().unwrap());
    frame_body(&mut emptied, 0, true);
    assert_eq!(
        emptied[header::CONTENT_LENGTH],
        "0",
        "a body a module emptied is framed empty"
    );
}

fn modify(body: Option<&[u8]>, set: &[(&str, &str)]) -> ModifyAction {
    ModifyAction {
        status: None,
        headers_set: set
            .iter()
            .map(|(name, value)| Header {
                name: (*name).to_string(),
                value: (*value).to_string(),
            })
            .collect(),
        headers_add: vec![],
        headers_remove: vec![],
        body_replace: body.map(<[u8]>::to_vec),
    }
}

/// A module's replacement carries its own encoding: the one declared for
/// the body it replaces goes, a declaration the module makes alongside
/// its replacement stays, and a body left in place keeps its encoding.
#[test]
fn a_modules_replacement_carries_its_own_encoding() {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_ENCODING, "gzip".parse().unwrap());
    let mut body = b"gzipped upstream".to_vec();
    assert!(apply_modify(
        &mut headers,
        &mut body,
        modify(Some(b"plain".as_slice()), &[])
    ));
    assert!(headers.get(header::CONTENT_ENCODING).is_none());
    assert_eq!(body, b"plain");

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_ENCODING, "br".parse().unwrap());
    let mut body = b"brotli upstream".to_vec();
    let encoded = modify(
        Some(b"\x1f\x8b gzipped by the module".as_slice()),
        &[("content-encoding", "gzip")],
    );
    assert!(apply_modify(&mut headers, &mut body, encoded));
    assert_eq!(headers[header::CONTENT_ENCODING], "gzip");
    assert_eq!(body, b"\x1f\x8b gzipped by the module");

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_ENCODING, "gzip".parse().unwrap());
    let mut body = b"left in place".to_vec();
    assert!(!apply_modify(
        &mut headers,
        &mut body,
        modify(None, &[("x-seen", "1")])
    ));
    assert_eq!(headers[header::CONTENT_ENCODING], "gzip");
    assert_eq!(headers["x-seen"], "1");
    assert_eq!(body, b"left in place");
}

/// A rejection is rebuilt from the response's parts too: the status is
/// the module's, the body is framed, and the extensions stay.
#[test]
fn a_rejection_keeps_the_responses_extensions() {
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_LENGTH, "40")
        .body(Body::empty())
        .expect("a response");
    response.extensions_mut().insert(Marker("permit"));
    let (parts, _) = response.into_parts();
    let headers = parts.headers.clone();
    let rejected = rebuild_response(parts, StatusCode::FORBIDDEN, headers, b"no".to_vec(), true);
    assert_eq!(rejected.status(), StatusCode::FORBIDDEN);
    assert_eq!(rejected.headers()[header::CONTENT_LENGTH], "2");
    assert_eq!(
        rejected.extensions().get::<Marker>(),
        Some(&Marker("permit"))
    );
}

/// The response the hook rebuilds declares the body it carries, whatever
/// the upstream declared for the body before the modules ran, and keeps
/// the extensions the layers inside handed the edge.
#[tokio::test]
async fn the_response_hook_frames_the_body_it_forwards_and_keeps_the_extensions() {
    let manager = manager_with_an_inert_module();
    let payload = b"twenty bytes of body".to_vec();
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_LENGTH, "3")
        .body(Body::from(payload.clone()))
        .expect("a response");
    response.extensions_mut().insert(Marker("permit"));
    let response = run_response_hook(&manager, response).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_LENGTH],
        payload.len().to_string()
    );
    assert_eq!(
        response.extensions().get::<Marker>(),
        Some(&Marker("permit"))
    );
    assert_eq!(body_of(response.into_body()).await, payload);
}

/// The request the hook forwards declares the body it carries, not the
/// length the client declared for the body before the modules ran.
#[tokio::test]
async fn the_request_hook_frames_the_body_it_forwards() {
    let manager = manager_with_an_inert_module();
    let payload = b"{\"model\":\"m\"}".to_vec();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_LENGTH, "3")
        .body(Body::from(payload.clone()))
        .expect("a request");
    let RequestHook::Run(request) = run_request_hook(&manager, request, "req").await else {
        panic!("the hook forwards the request");
    };
    assert_eq!(
        request.headers()[header::CONTENT_LENGTH],
        payload.len().to_string()
    );
    assert_eq!(body_of(request.into_body()).await, payload);
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
