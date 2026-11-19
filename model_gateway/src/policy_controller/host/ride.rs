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
    /// stream reports at EOF, using the time of the first data frame.
    /// A body dropped before EOF is cancelled.
    pub(crate) fn held_to_the_end_of(self, response: Response) -> Response {
        let (parts, body) = response.into_parts();
        let body = Body::from_stream(Ride {
            placed: Some(self),
            inner: body.into_data_stream(),
            status: parts.status,
            first: None,
            failed: false,
        });
        Response::from_parts(parts, body)
    }
}

/// The regular router wraps the body because it shares the relay with the
/// pair router. A relay dedicated to the regular router can carry the
/// placement and the load guard together as [`Held`](super::Held).
///
/// The placement drops before the body and its guard. A dispatched
/// placement reports once at EOF. On an early drop the handle reports
/// the cancellation.
/// The stream reads the frame results without reading their bytes.
struct Ride {
    placed: Option<Placed>,
    inner: BodyDataStream,
    status: StatusCode,
    first: Option<Instant>,
    failed: bool,
}

impl Stream for Ride {
    type Item = <BodyDataStream as Stream>::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let next = Pin::new(&mut this.inner).poll_next(cx);
        match &next {
            Poll::Ready(Some(Ok(_))) if this.first.is_none() => this.first = Some(Instant::now()),
            Poll::Ready(Some(Err(_))) => this.failed = true,
            Poll::Ready(None) => {
                let Some(mut placed) = this.placed.take() else {
                    return next;
                };
                let Some(sent) = placed.sent else {
                    return next;
                };
                let class = if this.failed {
                    TransportClass::ServerError
                } else {
                    transport_class(this.status, true)
                };
                let now = Instant::now();
                placed.completed(End::http(this.status, class, sent, this.first, now), now);
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
            failed: false,
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
