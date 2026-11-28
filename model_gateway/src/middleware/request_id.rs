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
//!
//! Under a policy spec that keeps a sample of its decisions the gateway
//! also restates the chosen id on the request, under the header
//! `x-smg-request-id` (`restate`). The engine reads what stands there
//! as the gateway's own id of the request, by the name a placement
//! states to it. That
//! write is no part of this layer: the server installs it as a layer of its
//! own inside the edge layers, behind this one, and under no other
//! configuration, so this layer is the same for every gateway. The write
//! replaces whatever a client sent under that name, and no worker is sent
//! the header: the routers forward an allow-list of request headers, and
//! this name is not on it. Inside the gateway it is a request header like
//! any other: a request plugin sees it and can change it, and a policy spec
//! that names it as a header reads what stands there. Under a spec that
//! keeps no sample, and in a gateway that places with its classic policies,
//! nothing is written there, and a client's value stays as it came. A
//! configuration that names that header for a use of its own is served as
//! for any header it names so. Named as a request id header, it is read
//! and rewritten by this layer where it is the first configured header
//! that is present.

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
/// The request header the gateway restates a request's chosen id under,
/// where its policy spec keeps a sample ([`restate`]). No route sends it to
/// a worker: the routers' allow-list of request headers passes the
/// request-id headers by their names and by the prefix `x-request-id-`,
/// and this name is neither.
pub(crate) static OWN_REQUEST_ID: HeaderName = HeaderName::from_static("x-smg-request-id");

/// Restates the id the layer chose for `request` on the request itself,
/// under [`OWN_REQUEST_ID`], as the one value there: whatever a client sent
/// under that name is replaced. A request that holds no chosen id, because
/// no request-id layer ran in front of this, keeps no value under the name
/// either: a client's value never stands behind this function.
///
/// The server installs it as a layer of its own inside the edge layers,
/// and only where the gateway runs Grand Central under a policy spec that
/// keeps a sample of its decisions (`own_request_id_layer` in `server.rs`):
/// the engine reads a host's id of a request only where it keeps a
/// decision. That choice is made once, where the applications are built. A
/// policy spec put in force later on the running engine does not move it.
pub(crate) fn restate(mut request: Request) -> Request {
    let id = request
        .extensions()
        .get::<RequestId>()
        .and_then(|id| HeaderValue::from_str(&id.0).ok());
    let headers = request.headers_mut();
    match id {
        Some(id) => {
            headers.insert(&OWN_REQUEST_ID, id);
        }
        None => {
            headers.remove(&OWN_REQUEST_ID);
        }
    }
    request
}

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
#[path = "../../tests/unit/middleware/request_id/tests.rs"]
mod tests;
