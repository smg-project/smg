//! A placement as an HTTP router holds it, to the end of its response's
//! body.

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
use http_body::{Body as _, Frame, SizeHint};
use parking_lot::Mutex;
use smg_policy_controller::{lease::TransportClass, ReleaseReason};
use smg_worker::identity::basic::Worker;

use super::{transport_class, Ask, End, Host, Placed};
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

/// The end of the drain of a pair's prefill, which the drain shares with
/// the placement. The drain outlives the attempt that sent it, and it can
/// outlive the answer's body. A placement whose answer ended first is left
/// here, and the drain says the prefill's end on it.
#[derive(Default)]
pub(crate) struct DrainEnd {
    /// The status the drain ended with. Zero while it runs.
    status: AtomicU16,
    /// The placement of an answer that ended ahead of the drain.
    left: Mutex<Option<Sent>>,
}

impl DrainEnd {
    /// The drain ended with `status`. A placement left here hears its
    /// prefill's end now. One whose answer still runs reads the status at
    /// its next frame, or at its end.
    pub(crate) fn tell(&self, status: StatusCode) {
        let left = {
            let mut left = self.left.lock();
            self.status.store(status.as_u16(), Ordering::Relaxed);
            left.take()
        };
        if let Some(mut sent) = left {
            sent.prefill_ended(status, Instant::now());
        }
    }

    /// The status the drain ended with, where it has ended.
    fn told(&self) -> Option<StatusCode> {
        StatusCode::from_u16(self.status.load(Ordering::Relaxed)).ok()
    }

    /// Takes `sent`, whose answer ended. A drain that has ended is told on
    /// it here. One that still runs keeps it and tells its own end. The
    /// status is read under the lock the drain writes it under, so exactly
    /// one of the two says the prefill's end.
    fn leave(&self, mut sent: Sent) {
        let mut left = self.left.lock();
        match self.told() {
            Some(status) => {
                drop(left);
                sent.prefill_ended(status, Instant::now());
            }
            None => *left = Some(sent),
        }
    }
}

/// A placement with what an HTTP router alone knows of its send:
/// the instant, and the mark its streamed answer's relay writes at a cut.
/// The host's [`Placed`] keeps neither, so a route that needs none carries
/// none.
pub(crate) struct Sent {
    placed: Placed,
    at: Option<Instant>,
    /// Whether [`Self::dispatched`] was said: the request's own dispatch, or
    /// a pair's prefill's. A pair's prefill that was not sent is said no
    /// end, and the placement's drop releases its side.
    own_sent: bool,
    /// The end of the drain of a pair's prefill, while the drain owes it.
    drained: Option<Arc<DrainEnd>>,
    cut: Option<Arc<AtomicU16>>,
}

impl From<Placed> for Sent {
    fn from(placed: Placed) -> Self {
        Self {
            placed,
            at: None,
            own_sent: false,
            drained: None,
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
        self.own_sent = true;
        self.placed.dispatched(now);
    }

    /// A pair's decode request was sent at `now`. Said once. Where the
    /// prefill was not sent, the placement's time counts from here.
    pub(crate) fn decode_dispatched(&mut self, now: Instant) {
        if self.at.is_none() {
            self.at = Some(now);
        }
        self.placed.decode_dispatched(now);
    }

    /// Asks the pair's decode again at `now`, behind its prefill's answer
    /// ([`Placed::decode_again`]).
    pub(crate) fn decode_again(
        &mut self,
        host: &Host,
        ask: &Ask<'_>,
        holds: impl Fn(&Arc<dyn Worker>) -> bool,
        now: Instant,
    ) -> Option<&Arc<dyn Worker>> {
        self.placed.decode_again(host, ask, holds, now)
    }

    /// A pair's prefill answered `status` at `now`, ahead of the end of the
    /// answer its decode writes. Said once, and of a prefill that was sent.
    pub(crate) fn prefill_ended(&mut self, status: StatusCode, now: Instant) {
        if let (true, Some(at)) = (self.own_sent, self.at) {
            let class = transport_class(status, true);
            self.placed
                .completed(End::http(status, class, at, None, now), now);
        }
    }

    /// The end of the drain of this pair's prefill, for the drain to tell
    /// ([`DrainEnd::tell`]). The answer's body reads it at each frame, and
    /// an answer that ends ahead of the drain leaves the placement to it.
    pub(crate) fn drain_end(&mut self) -> Arc<DrainEnd> {
        Arc::clone(self.drained.get_or_insert_with(Arc::default))
    }

    /// Say the prefill's end where its drain has told one. Said once: the
    /// drain's end is given up with it. The clock is read only for a told
    /// end.
    fn tell_drained(&mut self) {
        if let Some(status) = self.drained.as_deref().and_then(DrainEnd::told) {
            self.drained = None;
            self.prefill_ended(status, Instant::now());
        }
    }

    /// The answer's body was dropped ahead of its end, at `now`: its client
    /// left. Where the prefill's drain still owes its end, the decode is
    /// given up here and the placement is left to the drain. Elsewhere the
    /// placement's drop gives up what it owes.
    fn body_dropped(mut self, now: Instant) {
        if let Some(drain) = self.drained.take() {
            self.placed.decode_released(ReleaseReason::Abandoned, now);
            drain.leave(self);
        }
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

    /// Say the placement's end where it was sent. A pair's answer is its
    /// decode's: the decode ends with it, and so does a prefill whose router
    /// said no end ahead of it ([`Self::prefill_ended`]). A drained prefill
    /// ends with its drain: the placement goes to the drain's end, which
    /// tells it at once or when the drain ends ([`DrainEnd::leave`]). A
    /// prefill that was not sent hears no end: the drop releases its side.
    fn end(
        mut self,
        status: StatusCode,
        class: TransportClass,
        first: Option<Instant>,
        now: Instant,
    ) {
        let Some(at) = self.at else {
            return;
        };
        let end = End::http(status, class, at, first, now);
        match self.drained.take() {
            Some(drain) => {
                self.placed.decode_completed(end, now);
                drain.leave(self);
            }
            None => {
                if self.own_sent {
                    self.placed.completed(end, now);
                }
                self.placed.decode_completed(end, now);
            }
        }
    }

    /// `response` with this placement held to the end of its body. The body
    /// reports once: when it has gone out in full, or at its error frame. A
    /// body dropped before either is cancelled. A response with nothing to
    /// send has gone out in full at its head, where a server asks its body
    /// for no frame: its lease ends here, as the answer it was.
    pub(crate) fn held_to_the_end_of(self, response: Response) -> Response {
        let progress = BodyProgress::new(&response);
        let status = response.status();
        if declares_no_body(status) || progress.out_in_full(response.body().is_end_stream()) {
            let class = class_at_the_end(false, self.cut(), status);
            self.end(status, class, None, Instant::now());
            return response;
        }
        let (parts, body) = response.into_parts();
        let body = Body::new(Ride {
            sent: Some(self),
            inner: body,
            status,
            first: None,
            progress,
        });
        Response::from_parts(parts, body)
    }
}

/// Whether a server writes no body under `status`, whatever the response
/// holds: an informational status, 204 and 304.
fn declares_no_body(status: StatusCode) -> bool {
    status.is_informational() || matches!(status, StatusCode::NO_CONTENT | StatusCode::NOT_MODIFIED)
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
/// early drop the handle reports the cancellation, and a prefill whose
/// drain still runs is left to the drain ([`Sent::body_dropped`]). The body
/// reads each frame's length and none of its bytes. A response with nothing
/// to send gets no such body: its lease ended at its head.
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
                    if let Some(sent) = &mut this.sent {
                        sent.tell_drained();
                    }
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

impl Drop for Ride {
    /// A body dropped with its placement still in hand never reached its
    /// end: the placement hears that.
    fn drop(&mut self) {
        if let Some(sent) = self.sent.take() {
            sent.body_dropped(Instant::now());
        }
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
