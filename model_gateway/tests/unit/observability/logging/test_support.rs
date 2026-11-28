use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use tracing::{subscriber::DefaultGuard, Dispatch, Level};
use tracing_subscriber::{layer::SubscriberExt, Layer, Registry};

/// A test's subscriber: this thread's default until the value drops,
/// seeing every line the thread writes meanwhile.
///
/// `tracing` caches per callsite whether any subscriber wants it, and
/// while it knows of a single live dispatcher it rebuilds the entry of a
/// callsite registered for the first time from the registering thread's
/// default alone. A test running beside this one with no subscriber of
/// its own, writing a line at such a callsite first, then caches it as
/// never enabled: the macro skips the line on every thread, this one
/// included, before any subscriber is asked, until the next dispatcher
/// registers. The second dispatcher held here keeps every rebuild over
/// all live dispatchers, the test's among them.
#[must_use = "dropping the value uninstalls the subscriber"]
pub(crate) struct TestSubscriber {
    _guard: DefaultGuard,
    _second: Dispatch,
}

/// `layer` over a registry as this thread's subscriber until the value
/// drops.
pub(crate) fn test_subscriber<L>(layer: L) -> TestSubscriber
where
    L: Layer<Registry> + Send + Sync + 'static,
{
    let second = Dispatch::new(Registry::default());
    let guard = tracing::subscriber::set_default(Registry::default().with(layer));
    TestSubscriber {
        _guard: guard,
        _second: second,
    }
}

/// A callsite first met on a thread with no subscriber, after this thread
/// took its own, still reaches this thread's subscriber.
#[test]
fn a_callsite_first_met_on_another_thread_reaches_this_threads_subscriber() {
    fn span_is_live() -> bool {
        !tracing::info_span!("first met on another thread").is_disabled()
    }
    let _subscriber = test_subscriber(tracing_subscriber::layer::Identity::new());
    std::thread::spawn(span_is_live)
        .join()
        .expect("the other thread");
    assert!(
        span_is_live(),
        "the callsite reads this thread's subscriber as absent"
    );
}

/// Every event of the `smg::response` target, as (level, message): the
/// lines the response-side middleware writes, as a test reads them.
#[derive(Clone, Default)]
pub(crate) struct ResponseLines(Arc<Mutex<Vec<(Level, String)>>>);

impl ResponseLines {
    pub(crate) fn lines(&self) -> Vec<(Level, String)> {
        self.0.lock().expect("lines").clone()
    }
}

impl<S: tracing::Subscriber> Layer<S> for ResponseLines {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if event.metadata().target() == "smg::response" {
            self.0.lock().expect("lines").push(line_of(event));
        }
    }
}

/// Every event at WARN or ERROR, as (level, message): the lines an
/// operator's default level shows, as a test reads them to prove which
/// of them an answer writes.
#[derive(Clone, Default)]
pub(crate) struct LinesAtWarnOrAbove(Arc<Mutex<Vec<(Level, String)>>>);

impl LinesAtWarnOrAbove {
    pub(crate) fn lines(&self) -> Vec<(Level, String)> {
        self.0.lock().expect("lines").clone()
    }
}

impl<S: tracing::Subscriber> Layer<S> for LinesAtWarnOrAbove {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if matches!(*event.metadata().level(), Level::WARN | Level::ERROR) {
            self.0.lock().expect("lines").push(line_of(event));
        }
    }
}

/// `event` as (level, message).
fn line_of(event: &tracing::Event<'_>) -> (Level, String) {
    struct Message(String);
    impl tracing::field::Visit for Message {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }
    let mut message = Message(String::new());
    event.record(&mut message);
    (*event.metadata().level(), message.0)
}

/// Counts, on this thread's subscriber, the ERROR events and the refusal
/// lines (the events carrying a `suppressed` field): what a router's tests
/// read to prove a refusal is written at WARN within the allowance, never
/// at ERROR.
pub(crate) struct RefusalLineCounter {
    pub(crate) errors: Arc<AtomicUsize>,
    pub(crate) refusal_lines: Arc<AtomicUsize>,
}

impl<S: tracing::Subscriber> Layer<S> for RefusalLineCounter {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if *event.metadata().level() == Level::ERROR {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        if event.metadata().fields().field("suppressed").is_some() {
            self.refusal_lines.fetch_add(1, Ordering::Relaxed);
        }
    }
}
