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

use axum::{
    body::{Body, BodyDataStream},
    http::StatusCode,
    response::Response,
};
use futures_util::Stream;
use smg_policy_controller::lease::TransportClass;

use super::{transport_class, End, Placed};

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

    /// `response` with this placement held to the end of its body. The
    /// stream reports at EOF, using the time of the first data frame, or at
    /// the body's error frame. A body dropped before either is cancelled.
    pub(crate) fn held_to_the_end_of(self, response: Response) -> Response {
        let (parts, body) = response.into_parts();
        let body = Body::from_stream(Ride {
            sent: Some(self),
            inner: body.into_data_stream(),
            status: parts.status,
            first: None,
        });
        Response::from_parts(parts, body)
    }
}

/// The regular router wraps the body because it shares the relay with the
/// pair router. A relay dedicated to the regular router can carry the
/// placement and the load guard together as [`Held`](super::Held).
///
/// The placement drops before the body and its guard. A dispatched
/// placement reports once: at EOF, by the head's status or by the cut the
/// relay marked, or at an error frame, where a server drops the body and
/// polls no further. On an early drop the handle reports the cancellation.
/// The stream reads the frame results without reading their bytes.
struct Ride {
    sent: Option<Sent>,
    inner: BodyDataStream,
    status: StatusCode,
    first: Option<Instant>,
}

impl Stream for Ride {
    type Item = <BodyDataStream as Stream>::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let next = Pin::new(&mut this.inner).poll_next(cx);
        let class = match &next {
            Poll::Ready(Some(Ok(_))) => {
                this.first.get_or_insert_with(Instant::now);
                return next;
            }
            Poll::Ready(Some(Err(_))) => TransportClass::ServerError,
            Poll::Ready(None) => {
                let cut = this.sent.as_ref().and_then(Sent::cut);
                transport_class(cut.unwrap_or(this.status), true)
            }
            Poll::Pending => return next,
        };
        if let Some(sent) = this.sent.take() {
            sent.end(this.status, class, this.first, Instant::now());
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use std::{io, time::Duration};

    use bytes::Bytes;
    use futures_util::{stream, StreamExt};

    use super::*;

    #[tokio::test]
    async fn the_stream_times_its_first_data_frame_once() {
        let frames = [
            Err(io::Error::other("body failed")),
            Ok(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ];
        let mut ride = Ride {
            sent: None,
            inner: Body::from_stream(stream::iter(frames)).into_data_stream(),
            status: StatusCode::OK,
            first: None,
        };
        assert!(matches!(ride.next().await, Some(Err(_))));
        assert_eq!(ride.first, None);
        assert!(matches!(ride.next().await, Some(Ok(_))));
        let first = ride.first.expect("the first frame was timed");
        let earlier = first - Duration::from_millis(1);
        ride.first = Some(earlier);
        assert!(matches!(ride.next().await, Some(Ok(_))));
        assert_eq!(ride.first, Some(earlier));
    }
}
