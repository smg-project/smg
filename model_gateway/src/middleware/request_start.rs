//! The request's entry stamp: the one clock read the edge takes when a
//! request comes in, which the layers inside share instead of reading the
//! clock again. The metrics layer, the outermost layer with a clock, takes
//! it and puts it on the request's extensions; the access log counts its
//! latency from it and the in-flight guard ages the request from it. Each
//! keeps a read of its own for a request that reaches it without the stamp
//! (a bare stack in a test). The stamps inside the admission layers (the
//! acceptance at tenant resolution, the request-body deadline) stay reads of
//! their own: a wait for a permit sits between the edge and them.
//!
//! The stamp is on the runtime's clock ([`clock_now`]): the wall clock in
//! production, the paused clock of a test that advances time itself, which
//! is also the clock the acceptance stamp is read on.

use std::time::Instant;

use http::Extensions;

/// When the request came in at the edge.
#[derive(Clone, Copy, Debug)]
pub struct RequestStarted(Instant);

impl RequestStarted {
    /// A stamp taken now, on the runtime's clock.
    pub fn now() -> Self {
        Self(clock_now())
    }

    /// A stamp taken at `at`.
    pub fn at(at: Instant) -> Self {
        Self(at)
    }

    /// The instant the request came in.
    pub fn instant(self) -> Instant {
        self.0
    }

    /// The stamp a request carries, when the edge put one on it.
    pub fn of(extensions: &Extensions) -> Option<Instant> {
        extensions.get::<Self>().map(|started| started.0)
    }
}

/// The runtime's clock: the wall clock in production, the paused clock of a
/// test that advances time itself.
pub fn clock_now() -> Instant {
    tokio::time::Instant::now().into_std()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn a_request_without_the_stamp_reads_as_none() {
        assert!(RequestStarted::of(&Extensions::new()).is_none());
    }

    #[test]
    fn the_stamp_a_request_carries_is_the_instant_it_was_taken_at() {
        let at = Instant::now() - Duration::from_secs(3);
        let mut extensions = Extensions::new();
        extensions.insert(RequestStarted::at(at));
        assert_eq!(RequestStarted::of(&extensions), Some(at));
    }

    #[tokio::test(start_paused = true)]
    async fn the_stamp_is_on_the_runtime_s_clock() {
        let before = RequestStarted::now();
        tokio::time::advance(Duration::from_secs(5)).await;
        let after = RequestStarted::now();
        assert_eq!(
            after.instant().duration_since(before.instant()),
            Duration::from_secs(5)
        );
    }
}
