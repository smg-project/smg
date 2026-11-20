//! A placement as the regular HTTP router holds it, to the end of its
//! response's body.

use std::{
    ops::Deref,
    pin::Pin,
    sync::{
        atomic::{AtomicU16, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Instant,
};

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use http_body::{Frame, SizeHint};
use smg_policy_controller::lease::TransportClass;

use super::{transport_class, End, Placed};
use crate::{middleware::metrics::BodyProgress, routers::common::attempt_ledger::UpstreamAnswer};

/// Response extension: the gateway made this answer behind the worker's own
/// head: a body cut short, one over the limit, or a read past the request's
/// time. The worker was reached, so the end is not a failed connection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BehindAnswer;

/// Whether the worker answered `response`'s request, whole or its head
/// alone.
fn answered(response: &Response) -> bool {
    let marks = response.extensions();
    marks.get::<UpstreamAnswer>().is_some() || marks.get::<BehindAnswer>().is_some()
}

/// A placement with what the regular HTTP router alone knows of its send:
/// the instant, and the mark its streamed answer's relay writes at a cut.
/// The host's [`Placed`] keeps neither, so a route that needs none carries
/// none.
pub(crate) struct Sent {
    placed: Placed,
    at: Option<Instant>,
    cut: Option<Arc<AtomicU16>>,
}

impl From<Placed> for Sent {
    fn from(placed: Placed) -> Self {
        Self {
            placed,
            at: None,
            cut: None,
        }
    }
}

impl Deref for Sent {
    type Target = Placed;

    fn deref(&self) -> &Placed {
        &self.placed
    }
}

impl Sent {
    /// The request was sent to the worker at `now`. Said once.
    pub(crate) fn dispatched(&mut self, now: Instant) {
        self.at = Some(now);
        self.placed.dispatched(now);
    }

    /// The mark for the relay of this placement's streamed answer: the relay
    /// writes the status of a cut's class on it, and the body's end reads it.
    pub(crate) fn cut_mark(&mut self) -> Arc<AtomicU16> {
        Arc::clone(self.cut.get_or_insert_with(Arc::default))
    }

    /// The status of the cut's class, where the relay marked one.
    fn cut(&self) -> Option<StatusCode> {
        StatusCode::from_u16(self.cut.as_ref()?.load(Ordering::Relaxed)).ok()
    }

    /// The request ended with a whole answer of `status` at `now`.
    /// `answered` says the worker gave it itself. Before dispatch, the
    /// placement's drop gives the lease up as abandoned.
    pub(crate) fn ended(self, status: StatusCode, answered: bool, now: Instant) {
        self.end(status, transport_class(status, answered), None, now);
    }

    /// The request ended with the whole answer `response` at `now`. Its
    /// marks say whether the worker answered: its own answer, or one the
    /// gateway made behind the worker's head ([`BehindAnswer`]).
    pub(crate) fn ended_by(self, response: &Response, now: Instant) {
        self.ended(response.status(), answered(response), now);
    }

    /// Say the placement's end where it was sent.
    fn end(
        mut self,
        status: StatusCode,
        class: TransportClass,
        first: Option<Instant>,
        now: Instant,
    ) {
        if let Some(at) = self.at {
            self.placed
                .completed(End::http(status, class, at, first, now), now);
        }
    }

    /// `response` with this placement held to the end of its body. The body
    /// reports once: when it has gone out in full, or at its error frame. A
    /// body dropped before either is cancelled.
    pub(crate) fn held_to_the_end_of(self, response: Response) -> Response {
        let progress = BodyProgress::new(&response);
        let (parts, body) = response.into_parts();
        let body = Body::new(Ride {
            sent: Some(self),
            inner: body,
            status: parts.status,
            first: None,
            progress,
        });
        Response::from_parts(parts, body)
    }
}

/// The class of a body's end. Where the relay marked a cut, the cut's
/// status decides, whether the body then ended cleanly or with an error
/// frame. Else a whole body ends by its head's status and a failed one as a
/// server error.
fn class_at_the_end(failed: bool, cut: Option<StatusCode>, status: StatusCode) -> TransportClass {
    match (cut, failed) {
        (Some(cut), _) => transport_class(cut, true),
        (None, false) => transport_class(status, true),
        (None, true) => TransportClass::ServerError,
    }
}

/// The regular router wraps the body because it shares the relay with the
/// pair router. A relay dedicated to the regular router can carry the
/// placement and the load guard together as [`Held`](super::Held).
///
/// The placement drops before the body and its guard. A dispatched
/// placement reports once: when the body has gone out in full, or at an
/// error frame, where a server drops the body and polls no further. A body
/// has gone out in full by the edge body's rule ([`BodyProgress`]): at its
/// last frame, or once its data adds up to the length the response
/// declared, since an HTTP/1.1 server asks for no frame after that. On an
/// early drop the handle reports the cancellation. The body reads each
/// frame's length and none of its bytes.
struct Ride {
    sent: Option<Sent>,
    inner: Body,
    status: StatusCode,
    first: Option<Instant>,
    progress: BodyProgress,
}

impl http_body::Body for Ride {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let this = self.get_mut();
        let next = Pin::new(&mut this.inner).poll_frame(cx);
        let failed = match &next {
            Poll::Pending => return next,
            Poll::Ready(Some(Ok(frame))) => {
                if frame.is_data() {
                    this.first.get_or_insert_with(Instant::now);
                }
                if !this.progress.frame_out(frame, this.inner.is_end_stream()) {
                    return next;
                }
                false
            }
            Poll::Ready(Some(Err(_))) => true,
            Poll::Ready(None) => false,
        };
        if let Some(sent) = this.sent.take() {
            let class = class_at_the_end(failed, sent.cut(), this.status);
            sent.end(this.status, class, this.first, Instant::now());
        }
        next
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::{io, time::Duration};

    use futures_util::stream;
    use http_body_util::BodyExt;

    use super::*;

    #[tokio::test]
    async fn the_body_times_its_first_data_frame_once() {
        let frames = [
            Err(io::Error::other("body failed")),
            Ok(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ];
        let response = Response::new(Body::from_stream(stream::iter(frames)));
        let mut ride = Ride {
            sent: None,
            progress: BodyProgress::new(&response),
            inner: response.into_body(),
            status: StatusCode::OK,
            first: None,
        };
        assert!(matches!(ride.frame().await, Some(Err(_))));
        assert_eq!(ride.first, None);
        assert!(matches!(ride.frame().await, Some(Ok(_))));
        let first = ride.first.expect("the first frame was timed");
        let earlier = first - Duration::from_millis(1);
        ride.first = Some(earlier);
        assert!(matches!(ride.frame().await, Some(Ok(_))));
        assert_eq!(ride.first, Some(earlier));
    }

    #[test]
    fn an_answer_behind_the_workers_head_counts_as_answered() {
        let marked = |mark: Option<bool>| {
            let mut response = Response::new(Body::empty());
            match mark {
                Some(true) => drop(response.extensions_mut().insert(UpstreamAnswer)),
                Some(false) => drop(response.extensions_mut().insert(BehindAnswer)),
                None => {}
            }
            answered(&response)
        };
        assert_eq!(
            [marked(None), marked(Some(true)), marked(Some(false))],
            [false, true, true]
        );
    }

    #[test]
    fn a_cut_the_relay_marked_decides_the_class_of_either_end() {
        use TransportClass::{Ok as Whole, ServerError, Timeout};
        const OK: StatusCode = StatusCode::OK;
        let cut = |status| Some(StatusCode::from_u16(status).expect("a status"));
        for (failed, mark, class) in [
            (false, None, Whole),
            (true, None, ServerError),
            (false, cut(502), ServerError),
            (true, cut(502), ServerError),
            (false, cut(504), Timeout),
            (true, cut(504), Timeout),
        ] {
            assert_eq!(
                class_at_the_end(failed, mark, OK),
                class,
                "{failed} {mark:?}"
            );
        }
    }
}
