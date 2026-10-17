//! Opt-in observation of the calls the engine clients make.
//!
//! Every engine client sends its RPCs through a [`TapChannel`]: the channel
//! [`connect_channel`] builds, plus a check for a process-wide [`CallTap`].
//! The client is built with its endpoint as the origin, so each request's URI
//! names the endpoint and the method.
//! With no tap set, the default, a call costs one atomic load and its bodies
//! are not touched. With a tap set, each call is offered to it. For a call the
//! tap takes, every gRPC message of the request and of the response reaches
//! the call's [`CallRecord`] as its encoded protobuf, in order, and the record
//! learns how the call ended. The bytes the caller and the engine see do not
//! change.

use std::{
    error::Error,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, MutexGuard, PoisonError, RwLock,
    },
    task::{ready, Context, Poll},
};

use bytes::{Buf, Bytes, BytesMut};
use http::{uri::PathAndQuery, Uri};
use http_body::{Frame, SizeHint};
use tonic::{
    body::Body,
    transport::{channel::ResponseFuture, Channel},
    Status,
};
use tower_service::Service;

use crate::channel::{connect_channel, normalize_grpc_endpoint};

/// Takes the calls it wants to observe. Set it with [`set_call_tap`].
pub trait CallTap: Send + Sync + 'static {
    /// A call is starting. Return a record to observe it, or `None` to leave
    /// it untouched. Runs on the caller's task, before the call is sent.
    fn begin(&self, call: &CallStart<'_>) -> Option<Box<dyn CallRecord>>;
}

/// The call offered to [`CallTap::begin`].
#[non_exhaustive]
pub struct CallStart<'a> {
    /// The endpoint's scheme and authority (a `grpc` scheme reads as `http`)
    /// and the gRPC method path, e.g.
    /// `http://10.0.0.7:30000/vllm.grpc.engine.VllmEngine/Generate`.
    pub uri: &'a Uri,
    /// The request metadata.
    pub metadata: &'a http::HeaderMap,
}

/// Observes one call.
///
/// A message is the payload of one gRPC frame, as on the wire. The clients
/// negotiate no compression, so that is the encoded protobuf.
pub trait CallRecord: Send + 'static {
    /// The client sent `message`.
    fn request(&mut self, message: &[u8]);
    /// The engine sent `message`.
    fn response(&mut self, message: &[u8]);
    /// The call ended. Called once, after every message.
    fn end(self: Box<Self>, end: CallEnd);
}

/// How a call ended.
#[derive(Debug)]
pub enum CallEnd {
    /// The engine ended the call with this status, `Code::Ok` when it
    /// completed.
    Status(Status),
    /// The call failed below gRPC: it could not be sent, the connection
    /// broke, or the response ended without a status.
    Failed(String),
    /// The caller dropped the call before the engine ended it.
    Dropped,
}

static TAP_SET: AtomicBool = AtomicBool::new(false);
static TAP: RwLock<Option<Arc<dyn CallTap>>> = RwLock::new(None);

/// Set the process-wide tap, or clear it with `None`. Calls already started
/// keep the records they began with.
pub fn set_call_tap(tap: Option<Arc<dyn CallTap>>) {
    let mut slot = TAP.write().unwrap_or_else(PoisonError::into_inner);
    TAP_SET.store(tap.is_some(), Ordering::Release);
    *slot = tap;
}

fn begin(call: &CallStart<'_>) -> Option<Arc<TappedCall>> {
    if !TAP_SET.load(Ordering::Acquire) {
        return None;
    }
    let slot = TAP.read().unwrap_or_else(PoisonError::into_inner);
    let record = slot.as_ref()?.begin(call)?;
    Some(Arc::new(TappedCall(Mutex::new(Some(record)))))
}

/// The channel the engine clients send through: a [`Channel`] whose calls
/// are offered to the [`CallTap`]. It is no larger than the channel.
#[derive(Clone, Debug)]
pub struct TapChannel(Channel);

// Every engine client holds one, and callers keep clients in enums sized by
// them: tapping must not grow a client.
const _: () = assert!(size_of::<TapChannel>() == size_of::<Channel>());

impl TapChannel {
    /// [`connect_channel`] to `endpoint`. Also returns the origin to build
    /// the client with (`with_origin`), which puts the endpoint in each
    /// call's [`CallStart::uri`].
    pub async fn connect(endpoint: &str) -> Result<(Self, Uri), Box<dyn Error + Send + Sync>> {
        let channel = connect_channel(endpoint).await?;
        // Scheme and authority only: tonic prefixes an origin's path to every
        // method path, and the channel replaces the rest with its own.
        let mut origin = normalize_grpc_endpoint(endpoint)
            .parse::<Uri>()?
            .into_parts();
        origin.path_and_query = Some(PathAndQuery::from_static("/"));
        Ok((Self(channel), Uri::from_parts(origin)?))
    }
}

impl Service<http::Request<Body>> for TapChannel {
    type Response = http::Response<Body>;
    type Error = tonic::transport::Error;
    type Future = TapFuture;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.0.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let call = begin(&CallStart {
            uri: request.uri(),
            metadata: request.headers(),
        });
        let request = match &call {
            Some(call) => request.map(|body| TapBody::wrap(body, Side::Request, Arc::clone(call))),
            None => request,
        };
        TapFuture {
            inner: self.0.call(request),
            call,
        }
    }
}

/// The response future of a [`TapChannel`] call.
pub struct TapFuture {
    inner: ResponseFuture,
    call: Option<Arc<TappedCall>>,
}

impl Future for TapFuture {
    type Output = Result<http::Response<Body>, tonic::transport::Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let result = ready!(Pin::new(&mut self.inner).poll(cx));
        let Some(call) = self.call.take() else {
            return Poll::Ready(result);
        };
        Poll::Ready(match result {
            Ok(response) => Ok(tap_response(response, call)),
            Err(error) => {
                call.end(CallEnd::Failed(error_chain(&error)));
                Err(error)
            }
        })
    }
}

fn tap_response(response: http::Response<Body>, call: Arc<TappedCall>) -> http::Response<Body> {
    // Without a body to follow, the headers end the call: a trailers-only
    // response carries the status there, and tonic fails a non-200 response
    // without reading its body.
    if let Some(status) = Status::from_header_map(response.headers()) {
        call.end(CallEnd::Status(status));
        return response;
    }
    if response.status() != http::StatusCode::OK {
        call.end(CallEnd::Failed(format!(
            "HTTP status {}",
            response.status()
        )));
        return response;
    }
    response.map(|body| TapBody::wrap(body, Side::Response, call))
}

fn error_chain(error: &dyn Error) -> String {
    let mut chain = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        chain.push_str(": ");
        chain.push_str(&cause.to_string());
        source = cause.source();
    }
    chain
}

/// One tapped call's record, shared by its request and response bodies.
/// Dropping the last handle before the call ended ends it as dropped.
struct TappedCall(Mutex<Option<Box<dyn CallRecord>>>);

impl TappedCall {
    fn lock(&self) -> MutexGuard<'_, Option<Box<dyn CallRecord>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn end(&self, end: CallEnd) {
        let record = self.lock().take();
        if let Some(record) = record {
            record.end(end);
        }
    }
}

impl Drop for TappedCall {
    fn drop(&mut self) {
        self.end(CallEnd::Dropped);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Request,
    Response,
}

/// A body whose gRPC messages are handed to the call's record as they pass.
struct TapBody {
    inner: Body,
    side: Side,
    messages: Deframer,
    call: Arc<TappedCall>,
}

impl TapBody {
    fn wrap(inner: Body, side: Side, call: Arc<TappedCall>) -> Body {
        Body::new(Self {
            inner,
            side,
            messages: Deframer::default(),
            call,
        })
    }
}

impl http_body::Body for TapBody {
    type Data = Bytes;
    type Error = Status;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Status>>> {
        let this = &mut *self;
        let frame = ready!(Pin::new(&mut this.inner).poll_frame(cx));
        match &frame {
            Some(Ok(frame)) => {
                if let Some(data) = frame.data_ref() {
                    if let Some(record) = this.call.lock().as_deref_mut() {
                        let side = this.side;
                        this.messages.push(data, |message| match side {
                            Side::Request => record.request(message),
                            Side::Response => record.response(message),
                        });
                    }
                } else if let Some(trailers) = frame.trailers_ref() {
                    this.call.end(match Status::from_header_map(trailers) {
                        Some(status) => CallEnd::Status(status),
                        None => CallEnd::Failed("trailers without a grpc-status".to_string()),
                    });
                }
            }
            Some(Err(status)) => this.call.end(CallEnd::Failed(status.to_string())),
            None if this.side == Side::Response => this.call.end(CallEnd::Failed(
                "response ended without a status".to_string(),
            )),
            None => {}
        }
        Poll::Ready(frame)
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// gRPC frame header: a compression flag, then the message length as a
/// big-endian `u32`.
const HEADER_LEN: usize = 5;

/// Cuts a body's data into messages. Data frames split and join messages
/// freely; the start of a message whose end has not arrived waits in
/// `pending`.
#[derive(Default)]
struct Deframer {
    pending: BytesMut,
}

impl Deframer {
    fn push(&mut self, data: &[u8], mut deliver: impl FnMut(&[u8])) {
        let buffered = !self.pending.is_empty();
        if buffered {
            self.pending.extend_from_slice(data);
        }
        let bytes = if buffered { &self.pending[..] } else { data };
        let mut at = 0;
        while let Some((message, len)) = bytes.get(at..).and_then(first_message) {
            deliver(message);
            at += len;
        }
        if buffered {
            self.pending.advance(at);
        } else {
            self.pending
                .extend_from_slice(data.get(at..).unwrap_or_default());
        }
    }
}

/// The first whole message in `bytes` and its framed length.
fn first_message(bytes: &[u8]) -> Option<(&[u8], usize)> {
    let len = u32::from_be_bytes(bytes.get(1..HEADER_LEN)?.try_into().ok()?);
    let end = HEADER_LEN.checked_add(usize::try_from(len).ok()?)?;
    Some((bytes.get(HEADER_LEN..end)?, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(message: &[u8]) -> Vec<u8> {
        let mut framed = vec![0];
        framed.extend_from_slice(&u32::try_from(message.len()).unwrap().to_be_bytes());
        framed.extend_from_slice(message);
        framed
    }

    fn deframe(chunks: &[&[u8]]) -> Vec<Vec<u8>> {
        let mut deframer = Deframer::default();
        let mut messages = Vec::new();
        for chunk in chunks {
            deframer.push(chunk, |message| messages.push(message.to_vec()));
        }
        messages
    }

    #[test]
    fn messages_split_at_any_byte_are_reassembled() {
        let messages = [Vec::new(), b"abc".to_vec(), vec![9; 300]];
        let wire: Vec<u8> = messages.iter().flat_map(|m| frame(m)).collect();
        for split in 0..=wire.len() {
            let (head, tail) = wire.split_at(split);
            assert_eq!(deframe(&[head, tail]), messages, "split at {split}");
        }
    }

    #[test]
    fn messages_fed_byte_by_byte_are_reassembled() {
        let messages = [b"abc".to_vec(), Vec::new(), vec![7; 70_000]];
        let wire: Vec<u8> = messages.iter().flat_map(|m| frame(m)).collect();
        let bytes: Vec<&[u8]> = wire.chunks(1).collect();
        assert_eq!(deframe(&bytes), messages);
    }

    #[test]
    fn partial_message_is_not_delivered() {
        let wire = frame(b"abcdef");
        assert!(deframe(&[&wire[..wire.len() - 1]]).is_empty());
        assert!(deframe(&[&wire[..3]]).is_empty());
    }
}
