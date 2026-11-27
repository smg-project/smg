use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use futures::StreamExt;
use tonic::Code;

use super::*;
use crate::{
    metrics::{Identity, ServicerMetrics},
    requests::RequestRegistry,
};

/// The observation of a request on a servicer that exposes no metrics.
fn request_metrics() -> Arc<RequestMetrics> {
    let identity = Identity {
        model: "served".to_string(),
        wire: "test",
        ranks: 1,
        max_model_len: 100,
        matches_string_stops: true,
        processes_media: false,
    };
    ServicerMetrics::new(&identity, false)
        .expect("metrics without a listener")
        .request(Instant::now(), None, 0, None, 1)
}

/// A choice in the shape of the servicers' own: a scripted engine stream
/// (`None` once ended), which may hold the engine's parked terminal, and
/// which stays pending once its script ran out (the engine still
/// generating).
struct Scripted {
    inner: Option<Script>,
    parked: Option<&'static str>,
    metrics: ChoiceMetrics,
}

impl Scripted {
    fn open(items: &[&'static str]) -> Self {
        Self {
            inner: Some(Script(items.iter().map(|item| Ok(*item)).collect())),
            parked: None,
            metrics: request_metrics().choice(),
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
        let item = match poll_engine_item(&mut this.inner, &mut this.metrics, cx) {
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
    let mut all = Choices::new();
    for choice in choices {
        all.push(choice);
    }
    (
        registry,
        RequestStream::new(all, cancel, registration, request_metrics()),
    )
}

/// One choice is polled in place and dropped once it ended, as the merge
/// drops an ended stream; a second choice makes the merge.
#[test]
fn one_choice_is_polled_in_place_and_two_are_merged() {
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);

    let mut one = Choices::new();
    assert!(one.iter().next().is_none());
    one.push(Scripted::open(&["a"]));
    assert!(matches!(one, Choices::One(Some(_))));
    assert_eq!(one.iter().count(), 1);
    assert!(matches!(
        Pin::new(&mut one).poll_next(&mut cx),
        Poll::Ready(Some(Ok("a")))
    ));
    assert!(Pin::new(&mut one).poll_next(&mut cx).is_pending());
    one.clear();
    assert!(one.iter_mut().next().is_none());
    assert!(matches!(
        Pin::new(&mut one).poll_next(&mut cx),
        Poll::Ready(None)
    ));

    let mut ended = Scripted::open(&[]);
    ended.inner = None;
    let mut one = Choices::new();
    one.push(ended);
    assert!(matches!(
        Pin::new(&mut one).poll_next(&mut cx),
        Poll::Ready(None)
    ));
    assert!(one.iter().next().is_none(), "an ended choice is dropped");

    let mut two = Choices::new();
    two.push(Scripted::open(&["a"]));
    two.push(Scripted::open(&["b"]));
    assert!(matches!(two, Choices::Many(_)));
    assert_eq!(two.iter().count(), 2);
    let mut seen = Vec::new();
    for _ in 0..2 {
        if let Poll::Ready(Some(Ok(item))) = Pin::new(&mut two).poll_next(&mut cx) {
            seen.push(item);
        }
    }
    seen.sort_unstable();
    assert_eq!(seen, ["a", "b"]);
    assert!(Pin::new(&mut two).poll_next(&mut cx).is_pending());
}

#[test]
fn poll_engine_item_hands_over_the_items_and_ends_the_choice_with_the_stream() {
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut metrics = request_metrics().choice();
    let mut inner: Option<futures::stream::Iter<std::vec::IntoIter<Result<u8, Status>>>> =
        Some(futures::stream::iter(vec![
            Ok(1),
            Err(Status::internal("engine")),
        ]));
    assert!(matches!(
        poll_engine_item::<_, u8, u8>(&mut inner, &mut metrics, &mut cx),
        ControlFlow::Continue(1)
    ));
    match poll_engine_item::<_, u8, u8>(&mut inner, &mut metrics, &mut cx) {
        ControlFlow::Break(Poll::Ready(Some(Err(status)))) => {
            assert_eq!(status.code(), Code::Internal);
        }
        other => panic!("the engine's error ends the choice: {other:?}"),
    }
    assert!(inner.is_none(), "the engine stream is dropped on its error");
    assert!(matches!(
        poll_engine_item::<_, u8, u8>(&mut inner, &mut metrics, &mut cx),
        ControlFlow::Break(Poll::Ready(None))
    ));

    let mut ended: Option<futures::stream::Iter<std::vec::IntoIter<Result<u8, Status>>>> =
        Some(futures::stream::iter(Vec::new()));
    assert!(matches!(
        poll_engine_item::<_, u8, u8>(&mut ended, &mut metrics, &mut cx),
        ControlFlow::Break(Poll::Ready(None))
    ));
    assert!(ended.is_none(), "the engine stream is dropped at its end");

    let mut pending: Option<futures::stream::Pending<Result<u8, Status>>> =
        Some(futures::stream::pending());
    assert!(matches!(
        poll_engine_item::<_, u8, u8>(&mut pending, &mut metrics, &mut cx),
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
    let (registry, mut stream) = request(vec![Scripted::open(&[]), Scripted::parked(&[], "stop")]);
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
