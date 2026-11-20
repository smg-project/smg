//! A placement held by the regular HTTP router's response body.

use std::{
    pin::Pin,
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

impl Placed {
    /// `response` with this placement held to the end of its body. The
    /// stream reports at EOF, using the time of the first data frame, or at
    /// the body's error frame. A body dropped before either is cancelled.
    pub(crate) fn held_to_the_end_of(self, response: Response) -> Response {
        let (parts, body) = response.into_parts();
        let body = Body::from_stream(Ride {
            placed: Some(self),
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
    placed: Option<Placed>,
    inner: BodyDataStream,
    status: StatusCode,
    first: Option<Instant>,
}

impl Ride {
    /// Say the placement's end, once, where it was sent.
    fn end(&mut self, class: TransportClass) {
        let Some(mut placed) = self.placed.take() else {
            return;
        };
        if let Some(sent) = placed.sent {
            let now = Instant::now();
            placed.completed(End::http(self.status, class, sent, self.first, now), now);
        }
    }
}

impl Stream for Ride {
    type Item = <BodyDataStream as Stream>::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let next = Pin::new(&mut this.inner).poll_next(cx);
        match &next {
            Poll::Ready(Some(Ok(_))) if this.first.is_none() => this.first = Some(Instant::now()),
            Poll::Ready(Some(Err(_))) => this.end(TransportClass::ServerError),
            Poll::Ready(None) => {
                let cut = this.placed.as_ref().and_then(Placed::cut);
                this.end(transport_class(cut.unwrap_or(this.status), true));
            }
            _ => {}
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
            placed: None,
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
