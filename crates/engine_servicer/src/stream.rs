//! The merged response stream of one request: its choices polled together,
//! ended early by the `Abort` RPC or by shutdown. Engine-neutral: each
//! servicer's `Generate` builds one over its own choice streams, and the
//! cancellation semantics ([`CancelReason`]) are decided here once.

use std::{
    collections::VecDeque,
    future::Future,
    ops::ControlFlow,
    pin::Pin,
    task::{Context, Poll},
};

use futures::{stream::SelectAll, Stream};
use tokio::sync::oneshot;
use tonic::Status;

use crate::requests::{
    settle_shutdown, CancelReason, HeldTerminal, Registration, SHUTDOWN_MESSAGE,
};

/// A choice of a request's stream as the `Abort` RPC sees it (the shutdown's
/// view of it is [`HeldTerminal`]).
pub(crate) trait Choice:
    HeldTerminal + Stream<Item = Result<<Self as HeldTerminal>::Response, Status>> + Unpin
{
    /// End this choice on the caller's abort: the terminal it was about to
    /// yield, the engine's parked one, or a synthesized `abort` one; `None`
    /// once it has already ended. Dropping the engine stream aborts the
    /// engine side.
    fn abort(&mut self) -> Option<Self::Response>;
}

/// Poll the engine stream behind a choice. `Continue` carries the engine's
/// next item; `Break` carries what the choice's own `poll_next` returns in
/// its place: pending, the end of the stream, or its error, the engine
/// stream dropped on the last two so the choice stays ended.
pub(crate) fn poll_engine_item<S, T, R>(
    inner: &mut Option<S>,
    cx: &mut Context<'_>,
) -> ControlFlow<Poll<Option<Result<R, Status>>>, T>
where
    S: Stream<Item = Result<T, Status>> + Unpin,
{
    let Some(stream) = inner.as_mut() else {
        return ControlFlow::Break(Poll::Ready(None));
    };
    match Pin::new(stream).poll_next(cx) {
        Poll::Pending => ControlFlow::Break(Poll::Pending),
        Poll::Ready(None) => {
            *inner = None;
            ControlFlow::Break(Poll::Ready(None))
        }
        Poll::Ready(Some(Err(status))) => {
            *inner = None;
            ControlFlow::Break(Poll::Ready(Some(Err(status))))
        }
        Poll::Ready(Some(Ok(item))) => ControlFlow::Continue(item),
    }
}

/// The merged response stream of one request: its choices polled together
/// (interleaved, each tagged with its `index`), ended early by the `Abort`
/// RPC or by shutdown. `H` is whatever else must live as long as the stream
/// (a servicer's load-tracker guard); nothing, by default.
pub(crate) struct RequestStream<C: Choice, H = ()> {
    choices: SelectAll<C>,
    cancel: oneshot::Receiver<CancelReason>,
    /// Set once aborted: the terminal `Complete` of every choice still open
    /// at that point, yielded before the stream ends.
    aborted: Option<VecDeque<C::Response>>,
    /// The status a shutdown ends the stream with once `aborted` is drained,
    /// when it cut a choice still open; `None` ends it plainly.
    cut: Option<Status>,
    _registration: Registration,
    _held: H,
}

impl<C: Choice> RequestStream<C> {
    /// The stream over `choices`, ended early by `cancel` (the registry's
    /// handle for the request), keeping the request registered for as long
    /// as it lives.
    pub(crate) fn new(
        choices: SelectAll<C>,
        cancel: oneshot::Receiver<CancelReason>,
        registration: Registration,
    ) -> Self {
        Self {
            choices,
            cancel,
            aborted: None,
            cut: None,
            _registration: registration,
            _held: (),
        }
    }

    /// The same stream, holding `held` for its lifetime.
    pub(crate) fn holding<H>(self, held: H) -> RequestStream<C, H> {
        RequestStream {
            choices: self.choices,
            cancel: self.cancel,
            aborted: self.aborted,
            cut: self.cut,
            _registration: self._registration,
            _held: held,
        }
    }
}

impl<C: Choice, H: Unpin> Stream for RequestStream<C, H>
where
    C::Response: Unpin,
{
    type Item = Result<C::Response, Status>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(aborted) = this.aborted.as_mut() {
                if let Some(response) = aborted.pop_front() {
                    return Poll::Ready(Some(Ok(response)));
                }
                // The completions the choices already held went out; a
                // shutdown that cut a choice still open ends the stream with
                // its status, a plain abort just ends it.
                return Poll::Ready(this.cut.take().map(Err));
            }
            // The cancellation names its cause. An `Abort` RPC is the caller's
            // own abort: the request ends as the Python servicer ends it, with
            // each open choice's `Complete(finish_reason = "abort")`. A
            // shutdown with the request still open is its engine side going
            // away under it (the lifecycle owner stopping the pair after the
            // engine exited, a drain whose grace ran out): no completion the
            // caller could read as a short answer, so the stream ends with
            // UNAVAILABLE, which the Router classes as the worker's failure.
            // Either way dropping the choices aborts the engine side.
            if let Poll::Ready(reason) = Pin::new(&mut this.cancel).poll(cx) {
                if reason == Ok(CancelReason::Shutdown) {
                    // A choice the engine (or a stop string) finished before
                    // this poll keeps its completion; only a choice still
                    // open is cut, and only then does the stream end with the
                    // status (see `settle_shutdown`).
                    let (finished, cut) = settle_shutdown(&mut this.choices);
                    this.aborted = Some(finished);
                    this.cut = cut.then(|| Status::unavailable(SHUTDOWN_MESSAGE));
                    continue;
                }
                // The abort, or a sender dropped without a word (the registry
                // entry went away): the completion it always was.
                let aborted = this.choices.iter_mut().filter_map(C::abort).collect();
                this.choices.clear();
                this.aborted = Some(aborted);
                continue;
            }
            return Pin::new(&mut this.choices).poll_next(cx);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::Duration,
    };

    use futures::StreamExt;
    use tonic::Code;

    use super::*;
    use crate::requests::RequestRegistry;

    /// A choice in the shape of the servicers' own: a scripted engine stream
    /// (`None` once ended), which may hold the engine's parked terminal, and
    /// which stays pending once its script ran out (the engine still
    /// generating).
    struct Scripted {
        inner: Option<Script>,
        parked: Option<&'static str>,
    }

    impl Scripted {
        fn open(items: &[&'static str]) -> Self {
            Self {
                inner: Some(Script(items.iter().map(|item| Ok(*item)).collect())),
                parked: None,
            }
        }

        fn parked(items: &[&'static str], terminal: &'static str) -> Self {
            Self {
                parked: Some(terminal),
                ..Self::open(items)
            }
        }
    }

    impl Stream for Scripted {
        type Item = Result<&'static str, Status>;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            let item = match poll_engine_item(&mut this.inner, cx) {
                ControlFlow::Continue(item) => item,
                ControlFlow::Break(poll) => return poll,
            };
            Poll::Ready(Some(Ok(item)))
        }
    }

    /// An engine stream: the scripted items, then pending for as long as the
    /// choice is open.
    struct Script(VecDeque<Result<&'static str, Status>>);

    impl Stream for Script {
        type Item = Result<&'static str, Status>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            match self.get_mut().0.pop_front() {
                Some(item) => Poll::Ready(Some(item)),
                None => Poll::Pending,
            }
        }
    }

    impl HeldTerminal for Scripted {
        type Response = &'static str;

        fn finished(&mut self) -> Option<&'static str> {
            let terminal = self.parked.take()?;
            self.inner = None;
            Some(terminal)
        }

        fn is_open(&self) -> bool {
            self.inner.is_some()
        }
    }

    impl Choice for Scripted {
        fn abort(&mut self) -> Option<&'static str> {
            self.inner.take()?;
            Some(self.parked.take().unwrap_or("abort"))
        }
    }

    fn request(choices: Vec<Scripted>) -> (Arc<RequestRegistry>, RequestStream<Scripted>) {
        let registry = Arc::new(RequestRegistry::default());
        let (registration, cancel) = registry.register("r1").expect("a fresh id");
        let mut all = SelectAll::new();
        for choice in choices {
            all.push(choice);
        }
        (registry, RequestStream::new(all, cancel, registration))
    }

    #[test]
    fn poll_engine_item_hands_over_the_items_and_ends_the_choice_with_the_stream() {
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut inner: Option<futures::stream::Iter<std::vec::IntoIter<Result<u8, Status>>>> =
            Some(futures::stream::iter(vec![
                Ok(1),
                Err(Status::internal("engine")),
            ]));
        assert!(matches!(
            poll_engine_item::<_, u8, u8>(&mut inner, &mut cx),
            ControlFlow::Continue(1)
        ));
        match poll_engine_item::<_, u8, u8>(&mut inner, &mut cx) {
            ControlFlow::Break(Poll::Ready(Some(Err(status)))) => {
                assert_eq!(status.code(), Code::Internal);
            }
            other => panic!("the engine's error ends the choice: {other:?}"),
        }
        assert!(inner.is_none(), "the engine stream is dropped on its error");
        assert!(matches!(
            poll_engine_item::<_, u8, u8>(&mut inner, &mut cx),
            ControlFlow::Break(Poll::Ready(None))
        ));

        let mut ended: Option<futures::stream::Iter<std::vec::IntoIter<Result<u8, Status>>>> =
            Some(futures::stream::iter(Vec::new()));
        assert!(matches!(
            poll_engine_item::<_, u8, u8>(&mut ended, &mut cx),
            ControlFlow::Break(Poll::Ready(None))
        ));
        assert!(ended.is_none(), "the engine stream is dropped at its end");

        let mut pending: Option<futures::stream::Pending<Result<u8, Status>>> =
            Some(futures::stream::pending());
        assert!(matches!(
            poll_engine_item::<_, u8, u8>(&mut pending, &mut cx),
            ControlFlow::Break(Poll::Pending)
        ));
        assert!(pending.is_some(), "a pending engine stream stays");
    }

    #[tokio::test]
    async fn the_choices_items_pass_through_and_the_stream_ends_with_them() {
        let (registry, mut stream) = request(vec![Scripted::open(&["a", "b"])]);
        assert_eq!(stream.next().await.unwrap().unwrap(), "a");
        assert_eq!(stream.next().await.unwrap().unwrap(), "b");
        // The script ran out with the choice open: the stream waits on the
        // engine (and on the cancel), as a live request does.
        assert!(
            tokio::time::timeout(Duration::from_millis(20), stream.next())
                .await
                .is_err()
        );
        assert_eq!(
            registry.len(),
            1,
            "registered for as long as the stream lives"
        );
        drop(stream);
        assert_eq!(registry.len(), 0, "the registration goes with the stream");
    }

    #[tokio::test]
    async fn an_abort_ends_every_open_choice_with_its_abort_completion() {
        let (registry, mut stream) = request(vec![
            Scripted::open(&["a"]),
            Scripted::parked(&[], "length"),
        ]);
        assert_eq!(stream.next().await.unwrap().unwrap(), "a");
        assert_eq!(registry.abort(&["r1".to_string()]).unwrap(), 1);
        let mut completions = vec![
            stream.next().await.unwrap().unwrap(),
            stream.next().await.unwrap().unwrap(),
        ];
        completions.sort_unstable();
        assert_eq!(completions, ["abort", "length"]);
        assert!(
            stream.next().await.is_none(),
            "a plain abort ends the stream without a status"
        );
    }

    #[tokio::test]
    async fn a_shutdown_keeps_held_completions_and_cuts_an_open_choice_with_unavailable() {
        let (registry, mut stream) =
            request(vec![Scripted::open(&[]), Scripted::parked(&[], "stop")]);
        registry.cancel_all().unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap(), "stop");
        let status = stream
            .next()
            .await
            .unwrap()
            .expect_err("the open choice was cut");
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(status.message(), SHUTDOWN_MESSAGE);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn a_shutdown_with_nothing_open_ends_the_stream_without_a_status() {
        let (registry, mut stream) = request(vec![Scripted::parked(&[], "length")]);
        registry.cancel_all().unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap(), "length");
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn what_the_stream_holds_lives_as_long_as_it_does() {
        struct Held(Arc<AtomicBool>);
        impl Drop for Held {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let (_registry, stream) = request(vec![Scripted::open(&["a"])]);
        let mut stream = stream.holding(Held(Arc::clone(&dropped)));
        assert_eq!(stream.next().await.unwrap().unwrap(), "a");
        assert!(!dropped.load(Ordering::SeqCst));
        drop(stream);
        assert!(dropped.load(Ordering::SeqCst));
    }
}
