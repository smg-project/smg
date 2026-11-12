//! Request ID middleware — pulls an existing request ID from configured
//! headers or generates an OpenAI-compatible one, stores it in extensions,
//! and echoes it back via `x-request-id`; a request in the Messages API's
//! shape gets the same id as `request-id` too, the header its SDKs read,
//! unless the response already carries one (a proxying route's relayed
//! upstream id, which is kept).
//!
//! An inbound id is honoured when it is one to
//! `MAX_INBOUND_REQUEST_ID_LEN` visible-ASCII characters. Anything else
//! (longer, empty, a space or control character, a byte outside ASCII) is
//! dropped for a generated id: the value is neither echoed nor logged, and
//! the header on the request is rewritten to carry the chosen id as its only
//! value (so a worker that reads or forwards it sees the gateway's id, and a
//! dropped or repeated value travels no further); a client cannot size the
//! response header, the access-log records or the worker's view of its
//! request. The first configured header that is present decides, by its
//! first value.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{ready, Context, Poll},
};

use axum::{
    extract::Request,
    http::{HeaderName, HeaderValue},
    response::Response,
};
use pin_project_lite::pin_project;
use rand::RngExt;
// Re-export RequestId from auth crate for backward compatibility
pub use smg_auth::RequestId;
use tower::{Layer, Service};

use super::messages_errors::wants_messages_envelope;

/// Alphanumeric characters for request ID generation (as bytes for O(1) indexing)
const REQUEST_ID_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
/// The random characters after the prefix of a fresh id.
const RANDOM_PART_LEN: usize = 24;
/// The response header that echoes the id the gateway chose: parsed once,
/// so putting it on a response copies a reference to its text, not the text.
static X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
/// The header the Messages API's SDKs read the request id from.
pub(crate) static REQUEST_ID: HeaderName = HeaderName::from_static("request-id");

/// The longest inbound request id the gateway honours, in bytes. The ids
/// proxies and clients commonly mint are 32 to 55 characters (a hex request
/// id, a UUID, a W3C trace parent), and a load balancer's trace id with its
/// lineage stays under 100; 128 holds every one of them with room and still
/// bounds what a client can make the gateway echo and log.
pub(super) const MAX_INBOUND_REQUEST_ID_LEN: usize = 128;

/// Whether an inbound request id is one the gateway honours: one to
/// [`MAX_INBOUND_REQUEST_ID_LEN`] visible-ASCII characters, no space among
/// them. The HTTP parser lets a horizontal tab through, so a header value
/// that parses as a string is not enough. The rule for every id the gateway
/// echoes: the message-thread route applies it to a client's `rid`.
pub(crate) fn is_acceptable_inbound_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_INBOUND_REQUEST_ID_LEN
        && value.bytes().all(|byte| (b'!'..=b'~').contains(&byte))
}

/// Generate OpenAI-compatible request ID based on endpoint.
pub(super) fn generate_request_id(path: &str) -> String {
    // Suffix compares, not substring scans; `/chat/completions` must precede
    // `/completions`.
    let prefix = if path.ends_with("/chat/completions") {
        "chatcmpl-"
    } else if path.ends_with("/completions") {
        "cmpl-"
    } else if path.ends_with("/generate") {
        "gnt-"
    } else if path.ends_with("/responses") {
        "resp-"
    } else if path.ends_with("/messages") {
        "msg_"
    } else {
        "req-"
    };

    // The random part in OpenAI's form, written after the prefix into the one
    // String the id is: its exact capacity up front, no second buffer and no
    // growth on the way. Byte indexing (O(1)) instead of chars().nth() (O(n)).
    let mut rng = rand::rng();
    let mut id = String::with_capacity(prefix.len() + RANDOM_PART_LEN);
    id.push_str(prefix);
    for _ in 0..RANDOM_PART_LEN {
        let idx = rng.random_range(0..REQUEST_ID_CHARS.len());
        id.push(REQUEST_ID_CHARS[idx] as char);
    }
    id
}

/// Tower Layer for request ID middleware
#[derive(Clone)]
pub struct RequestIdLayer {
    headers: Arc<Vec<HeaderName>>,
}

impl RequestIdLayer {
    /// `headers`: the request headers consulted for an inbound id, in order.
    /// They are parsed once here; a name that does not parse could never
    /// match a request and is skipped.
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers: Arc::new(
                headers
                    .iter()
                    .filter_map(|name| HeaderName::try_from(name.as_str()).ok())
                    .collect(),
            ),
        }
    }
}

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdMiddleware {
            inner,
            headers: self.headers.clone(),
        }
    }
}

/// Tower Service for request ID middleware
#[derive(Clone)]
pub struct RequestIdMiddleware<S> {
    inner: S,
    headers: Arc<Vec<HeaderName>>,
}

/// A generated id for a request to `path`, with the header value that
/// carries it (the generated ids are alphanumeric, so the value always
/// builds; the fallback is never reached).
fn fresh_id(path: &str) -> (String, HeaderValue) {
    let id = generate_request_id(path);
    let value = HeaderValue::from_str(&id)
        .unwrap_or_else(|_| HeaderValue::from_static("invalid-request-id"));
    (id, value)
}

impl<S> Service<Request> for RequestIdMiddleware<S>
where
    S: Service<Request, Response = Response>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = RequestIdFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request) -> Self::Future {
        let headers = self.headers.clone();

        // The first configured header that is present decides, by its first
        // value: that value when the gateway honours it, else a fresh id. A
        // dropped value is neither echoed nor logged, and the header is
        // rewritten to carry the chosen id as its only value: the HTTP routers
        // forward request-id headers to the worker, and the worker's log and
        // response id should name the same request as the gateway's.
        let inbound = headers.iter().find_map(|header_name| {
            req.headers()
                .get(header_name)
                .map(|header_value| (header_name, header_value))
        });
        let (request_id, header_value) = match inbound {
            Some((header_name, header_value)) => {
                let (id, value) = match header_value.to_str() {
                    Ok(id) if is_acceptable_inbound_id(id) => (id.to_owned(), header_value.clone()),
                    _ => {
                        tracing::debug!(
                            target: "smg::request",
                            header = %header_name,
                            length = header_value.len(),
                            max = MAX_INBOUND_REQUEST_ID_LEN,
                            "inbound request id dropped: not visible ASCII within the length bound; a fresh id is used"
                        );
                        fresh_id(req.uri().path())
                    }
                };
                req.headers_mut().insert(header_name, value.clone());
                (id, value)
            }
            None => fresh_id(req.uri().path()),
        };

        // Insert request ID into request extensions for other middleware/handlers to use
        req.extensions_mut().insert(RequestId(request_id));

        // The Anthropic SDKs read the request id from a `request-id` header
        // and nothing else (`_request_id` on a success, `request_id` on an
        // error), so a request in the Messages API's shape is answered with
        // that header beside `x-request-id`, both carrying the one id. A
        // response that already carries one (a proxying route relaying its
        // upstream's) keeps it: that is the id the upstream's support asks
        // for. Every other response keeps its bytes.
        let speaks_messages = wants_messages_envelope(req.headers(), req.uri().path());

        // The route's own future with the echo beside it, not a box around both.
        RequestIdFuture {
            inner: self.inner.call(req),
            header_value: Some(header_value),
            speaks_messages,
        }
    }
}

pin_project! {
    /// The request-id layer's future: the route's future, then the chosen id
    /// echoed on the response (`header_value`, until it is on the response);
    /// `speaks_messages` is whether the request had the Messages API's shape,
    /// which is answered with the id as `request-id` too.
    pub struct RequestIdFuture<F> {
        #[pin]
        inner: F,
        header_value: Option<HeaderValue>,
        speaks_messages: bool,
    }
}

impl<F, E> Future for RequestIdFuture<F>
where
    F: Future<Output = Result<Response, E>>,
{
    type Output = Result<Response, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let mut response = ready!(this.inner.poll(cx))?;
        // The id the gateway chose, echoed to the client.
        if let Some(value) = this.header_value.take() {
            if *this.speaks_messages {
                response
                    .headers_mut()
                    .entry(&REQUEST_ID)
                    .or_insert_with(|| value.clone());
            }
            response.headers_mut().insert(&X_REQUEST_ID, value);
        }
        Poll::Ready(Ok(response))
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, routing::get, Router};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    #[test]
    fn generated_ids_use_endpoint_prefixes() {
        for (path, prefix) in [
            ("/v1/chat/completions", "chatcmpl-"),
            ("/v1/completions", "cmpl-"),
            ("/generate", "gnt-"),
            ("/v1/responses", "resp-"),
            ("/v1/messages", "msg_"),
            ("/v1/embeddings", "req-"),
        ] {
            assert!(
                generate_request_id(path).starts_with(prefix),
                "path {path} must generate a {prefix} id"
            );
        }
    }

    #[test]
    fn the_inbound_id_bound_is_one_to_128_visible_ascii_characters() {
        for accepted in ["a", "!~", &"x".repeat(MAX_INBOUND_REQUEST_ID_LEN)] {
            assert!(is_acceptable_inbound_id(accepted), "{accepted:?}");
        }
        for dropped in [
            "",
            " ",
            "a b",
            "tab\there",
            "caf\u{e9}",
            "\u{7f}",
            &"x".repeat(MAX_INBOUND_REQUEST_ID_LEN + 1),
        ] {
            assert!(!is_acceptable_inbound_id(dropped), "{dropped:?}");
        }
    }

    /// The id the handler found in the request extensions, the first
    /// `x-request-id` header value it received (`-` when absent) and how many
    /// values of that header it received come back as the body,
    /// `extension|header|count`; the layer echoes the id it chose in
    /// `x-request-id`.
    fn app() -> Router {
        app_with(vec!["x-request-id".into(), "x-correlation-id".into()])
    }

    /// [`app`] with its own list of configured request-id headers.
    fn app_with(names: Vec<String>) -> Router {
        Router::new()
            .route("/v1/chat/completions", get(ids_seen))
            .route("/v1/messages", get(ids_seen))
            .route("/v1/models", get(ids_seen))
            .layer(RequestIdLayer::new(names))
    }

    /// The handler of [`app_with`]'s routes.
    async fn ids_seen(request: Request) -> String {
        let extension = request
            .extensions()
            .get::<RequestId>()
            .map(|id| id.0.clone())
            .unwrap_or_default();
        let header = request
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("-");
        let count = request.headers().get_all("x-request-id").iter().count();
        format!("{extension}|{header}|{count}")
    }

    /// (the echoed `x-request-id`, the id the handler saw in the extensions,
    /// the first `x-request-id` header value the handler received, how many
    /// values it received) for a request carrying `headers`.
    async fn ids_for(headers: &[(&str, &str)]) -> (String, String, String, usize) {
        let mut request = Request::builder().uri("/v1/chat/completions");
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let response = app()
            .oneshot(request.body(Body::empty()).expect("request"))
            .await
            .expect("response");
        let echoed = response.headers()["x-request-id"]
            .to_str()
            .expect("visible ASCII")
            .to_owned();
        let body = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        let body = String::from_utf8(body.to_vec()).expect("utf-8");
        let mut parts = body.split('|');
        let (extension, header, count) = (
            parts.next().expect("extension"),
            parts.next().expect("header"),
            parts.next().expect("count"),
        );
        (
            echoed,
            extension.to_owned(),
            header.to_owned(),
            count.parse().expect("a count"),
        )
    }

    fn is_fresh_chat_id(id: &str) -> bool {
        id.strip_prefix("chatcmpl-")
            .is_some_and(|rest| rest.len() == 24 && rest.bytes().all(|b| b.is_ascii_alphanumeric()))
    }

    #[tokio::test]
    async fn an_inbound_id_within_the_bound_is_honoured() {
        for id in [
            "req_client_1",
            "4bf92f3577b34da6a3ce929d0e0e4736",
            &"a".repeat(128),
        ] {
            let (echoed, seen, header, count) = ids_for(&[("x-request-id", id)]).await;
            assert_eq!(echoed, id, "the echoed id");
            assert_eq!(seen, id, "the id the handler saw");
            assert_eq!(
                (header.as_str(), count),
                (id, 1),
                "the header reaches the handler as is"
            );
        }
    }

    #[tokio::test]
    async fn an_inbound_id_over_the_bound_is_replaced_by_a_fresh_one() {
        let long = "R".repeat(129);
        let (echoed, seen, header, _) = ids_for(&[("x-request-id", &long)]).await;
        assert_eq!(echoed, seen, "the handler and the client see the same id");
        assert!(is_fresh_chat_id(&echoed), "a fresh id, got {echoed:?}");
        assert!(
            !echoed.contains("RRRR"),
            "nothing of the inbound value survives"
        );
        assert_eq!(
            header, echoed,
            "the request header carries the fresh id, so a worker reading or forwarding it sees the gateway's id"
        );
        // The first configured header that is present decides; a second one
        // is not promoted in its place.
        let (echoed, _, _, _) =
            ids_for(&[("x-request-id", &long), ("x-correlation-id", "corr-1")]).await;
        assert!(is_fresh_chat_id(&echoed), "a fresh id, got {echoed:?}");
    }

    #[tokio::test]
    async fn an_inbound_id_that_is_not_visible_ascii_is_replaced() {
        for value in ["tab\there", "two words", ""] {
            let (echoed, seen, header, _) = ids_for(&[("x-request-id", value)]).await;
            assert_eq!(echoed, seen, "the handler and the client see the same id");
            assert!(
                is_fresh_chat_id(&echoed),
                "a fresh id for {value:?}, got {echoed:?}"
            );
            assert_eq!(
                header, echoed,
                "the request header is rewritten for {value:?}"
            );
        }
    }

    /// A repeated request-id header forwards every value; after the layer
    /// the request carries exactly one, the honoured first value or the
    /// fresh id, and an unchecked second value goes no further.
    #[tokio::test]
    async fn a_repeated_request_id_header_keeps_only_the_chosen_id() {
        let long = "R".repeat(129);
        let (echoed, seen, header, count) =
            ids_for(&[("x-request-id", "ok-first"), ("x-request-id", &long)]).await;
        assert_eq!((echoed.as_str(), seen.as_str()), ("ok-first", "ok-first"));
        assert_eq!(
            (header.as_str(), count),
            ("ok-first", 1),
            "one value, the honoured one"
        );
        let (echoed, _, header, count) =
            ids_for(&[("x-request-id", &long), ("x-request-id", "ok-second")]).await;
        assert!(
            is_fresh_chat_id(&echoed),
            "the first value decides: {echoed:?}"
        );
        assert_eq!(
            (header.as_str(), count),
            (echoed.as_str(), 1),
            "one value, the fresh id"
        );
    }

    /// A request in the Messages API's shape (a Messages route, or any route
    /// under the `anthropic-version` header the Anthropic SDKs send on every
    /// call) is answered with the id as `request-id` as well: the header those
    /// SDKs read for `_request_id` and for an error's `request_id`. Every other
    /// response carries `x-request-id` alone, as before.
    #[tokio::test]
    async fn a_messages_api_request_is_echoed_its_id_as_request_id_too() {
        for (path, version, as_request_id) in [
            ("/v1/messages", None, true),
            ("/v1/models", Some("2023-06-01"), true),
            ("/v1/chat/completions", Some("2023-06-01"), true),
            ("/v1/models", None, false),
            ("/v1/chat/completions", None, false),
        ] {
            let mut request = Request::builder()
                .uri(path)
                .header("x-request-id", "req_client_1");
            if let Some(version) = version {
                request = request.header("anthropic-version", version);
            }
            let response = app()
                .oneshot(request.body(Body::empty()).expect("request"))
                .await
                .expect("response");
            assert_eq!(response.headers()["x-request-id"], "req_client_1", "{path}");
            assert_eq!(
                response
                    .headers()
                    .get("request-id")
                    .map(|value| value.to_str().expect("visible ASCII")),
                as_request_id.then_some("req_client_1"),
                "{path} under anthropic-version {version:?}"
            );
        }

        // A fresh id is echoed under both names too, the same value.
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/v1/messages")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let echoed = &response.headers()["x-request-id"];
        assert!(echoed.to_str().expect("visible ASCII").starts_with("msg_"));
        assert_eq!(&response.headers()["request-id"], echoed);
    }

    /// A response that already carries a `request-id` (a proxying route
    /// relaying its upstream's) keeps it; `x-request-id` still names the id
    /// the gateway chose.
    #[tokio::test]
    async fn a_request_id_already_on_the_response_is_kept() {
        let app = Router::new()
            .route(
                "/v1/messages",
                get(|| async { ([("request-id", "req_upstream_1")], "relayed") }),
            )
            .layer(RequestIdLayer::new(vec!["x-request-id".into()]));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/messages")
                    .header("x-request-id", "req_client_1")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.headers()["request-id"], "req_upstream_1");
        assert_eq!(response.headers()["x-request-id"], "req_client_1");
    }

    /// A configured name that does not parse could never match a request;
    /// it is skipped and the next configured header is consulted.
    #[tokio::test]
    async fn a_configured_name_that_does_not_parse_is_skipped() {
        let app = app_with(vec!["bad name".into(), "x-request-id".into()]);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/chat/completions")
                    .header("x-request-id", "ok-1")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.headers()["x-request-id"], "ok-1");
    }
}
