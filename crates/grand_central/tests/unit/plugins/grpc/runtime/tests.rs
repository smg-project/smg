use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    future::Future,
    path::PathBuf,
    pin::Pin,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use tokio::{
    sync::{mpsc, watch},
    task::JoinSet,
};

use super::{
    number, observe, run_commands, Command, Ended, Fault, Lane, LaneEvents, LaneState, ObserveLane,
    Opened, Peer, Requests, Runtime, Said, SendError, Spoken, Stand, Turn, Wait, Witness,
    OBSERVE_WINDOW, PEER_TEXT, PROMISE_MICROS, RECONNECT_PAUSE, STAND, WINDOW,
};
use crate::{
    plugins::{grpc::config::Transport, shared::OpenDecisions},
    service::proto,
    tracking::lease,
};

/// What a lane tells its plugin, counted.
#[derive(Default)]
struct Counted {
    late: AtomicU64,
}

impl LaneEvents for Counted {
    fn connected(&self) {}
    fn stream_failed(&self, _why: &str, _at: Instant) {}
    fn late_answer(&self) {
        self.late.fetch_add(1, Ordering::Relaxed);
    }
    fn observe_failed(&self, _why: &str) {}
}

/// A peer nothing listens behind, none of its streams settled.
fn nowhere() -> Arc<Peer> {
    peer(Spoken::default())
}

/// A plugin's version as one settled `Decide` stream fixed it.
fn standing_on(version: u32) -> Spoken {
    Spoken {
        stand: watch::Sender::new(Stand {
            fixed: version,
            standing: 1,
            settles: 1,
        }),
    }
}

/// A peer nothing listens behind, its `Decide` streams as `spoken` says.
fn peer(spoken: Spoken) -> Arc<Peer> {
    Arc::new(Peer {
        name: "p".to_owned(),
        transport: Transport::Unix {
            path: PathBuf::from("/nonexistent/p.sock"),
            peer_uid: None,
        },
        config: Vec::new(),
        exposure: proto::Exposure::default(),
        counters: Vec::new(),
        connect: Duration::from_secs(1),
        answer_bytes: super::answer_bytes(128, 0),
        versions: 1..=2,
        spoken,
    })
}

/// A lane nothing listens behind, not asked to connect yet.
fn cold_lane(events: &Arc<Counted>) -> Lane {
    let peer = nowhere();
    let events: Arc<dyn LaneEvents> = Arc::clone(events) as _;
    let weak: Weak<dyn LaneEvents> = Arc::downgrade(&events);
    Lane::new(peer, weak, Arc::new(Witness::new()))
}

/// An open lane with no stream behind it: the frames stay in `rx`.
fn open_lane(
    events: &Arc<Counted>,
    capacity: usize,
) -> (Lane, mpsc::Receiver<proto::PluginRequest>) {
    let lane = cold_lane(events);
    let (tx, rx) = mpsc::channel(capacity);
    lane.set_ready(tx, 7);
    (lane, rx)
}

fn answer(decision_id: u64) -> proto::PluginDecision {
    proto::PluginDecision {
        decision_id,
        ..proto::PluginDecision::default()
    }
}

fn answered(wait: Wait) -> Option<u64> {
    match wait {
        Wait::Answered(_, decision) => Some(decision.decision_id),
        Wait::Missed | Wait::Broken => None,
    }
}

#[test]
fn two_frames_in_flight_each_find_their_own_answer() {
    let events = Arc::new(Counted::default());
    let (lane, _rx) = open_lane(&events, 16);
    let (first, second) = (lane.next_id(), lane.next_id());
    assert_ne!(first, 0, "a frame's id is never zero");
    lane.send(first, proto::PluginDecide::default())
        .expect("sent");
    lane.send(second, proto::PluginDecide::default())
        .expect("sent");
    // The second side's answer arrives first and waits in its place.
    lane.deliver(answer(second));
    lane.deliver(answer(first));
    let soon = Instant::now() + Duration::from_secs(5);
    assert_eq!(answered(lane.wait_answer(first, soon)), Some(first));
    assert_eq!(answered(lane.wait_answer(second, soon)), Some(second));
    assert_eq!(events.late.load(Ordering::Relaxed), 0);
}

#[test]
fn a_missed_or_forgotten_frame_frees_its_place_and_its_answer_is_late() {
    let events = Arc::new(Counted::default());
    let (lane, _rx) = open_lane(&events, 16);
    let missed = lane.next_id();
    lane.send(missed, proto::PluginDecide::default())
        .expect("sent");
    assert!(matches!(
        lane.wait_answer(missed, Instant::now()),
        Wait::Missed
    ));
    lane.deliver(answer(missed));
    let forgotten = lane.next_id();
    lane.send(forgotten, proto::PluginDecide::default())
        .expect("sent");
    lane.forget(forgotten);
    lane.deliver(answer(forgotten));
    // An id of zero names no frame, whatever the free places hold.
    lane.deliver(answer(0));
    assert_eq!(events.late.load(Ordering::Relaxed), 3);
}

#[test]
fn a_frame_past_the_window_is_not_sent_and_a_freed_place_is_taken_again() {
    let events = Arc::new(Counted::default());
    let (lane, mut rx) = open_lane(&events, WINDOW * 2);
    let ids: Vec<u64> = (0..WINDOW).map(|_| lane.next_id()).collect();
    for &id in &ids {
        lane.send(id, proto::PluginDecide::default()).expect("sent");
    }
    let over = lane.next_id();
    assert_eq!(
        lane.send(over, proto::PluginDecide::default()).err(),
        Some(SendError::Full)
    );
    lane.deliver(answer(ids[0]));
    let soon = Instant::now() + Duration::from_secs(5);
    assert_eq!(answered(lane.wait_answer(ids[0], soon)), Some(ids[0]));
    lane.send(over, proto::PluginDecide::default())
        .expect("a place is free");
    let mut frames = 0;
    while rx.try_recv().is_ok() {
        frames += 1;
    }
    assert_eq!(
        frames,
        WINDOW + 1,
        "the refused frame never entered the stream"
    );
}

#[test]
fn a_stream_that_goes_breaks_every_frame_in_flight_and_refuses_the_next() {
    let events = Arc::new(Counted::default());
    let (lane, _rx) = open_lane(&events, 16);
    let (first, second) = (lane.next_id(), lane.next_id());
    lane.send(first, proto::PluginDecide::default())
        .expect("sent");
    lane.send(second, proto::PluginDecide::default())
        .expect("sent");
    lane.fail("the plugin ended the stream".into());
    let soon = Instant::now() + Duration::from_secs(5);
    assert!(matches!(lane.wait_answer(first, soon), Wait::Broken));
    assert!(matches!(lane.wait_answer(second, soon), Wait::Broken));
    assert_eq!(
        lane.send(lane.next_id(), proto::PluginDecide::default())
            .err(),
        Some(SendError::NotReady)
    );
    assert!(lane.stages().is_none());
}

/// The witness on instants of the case's own: `ms` after its zero.
fn at(witness: &Witness, ms: u64) -> Instant {
    witness.origin + Duration::from_millis(ms)
}

#[test]
fn a_thread_not_seen_for_longer_than_its_promise_has_stalled() {
    let promise = Duration::from_micros(PROMISE_MICROS);
    let witness = Witness::new();
    // The first frame in flight starts the promise: the idle time
    // before it is no gap.
    let sent = at(&witness, 10_000);
    witness.took_flight(sent);
    assert!(!witness.stalled_between(sent, sent + promise));
    assert!(
        witness.stalled_between(sent, sent + promise + Duration::from_micros(1)),
        "the gap stands: the thread has not run since"
    );
    // The thread runs a tick later than its promise: the gap is
    // closed and kept, and the stamps after it are in time.
    let back = sent + promise + Duration::from_millis(6);
    witness.stamp_at(back);
    witness.stamp_at(back + Duration::from_millis(1));
    let later = back + Duration::from_millis(2);
    witness.stamp_at(later);
    assert!(
        witness.stalled_between(sent, later),
        "a frame in flight across it"
    );
    assert!(
        witness.stalled_between(back - Duration::from_micros(1), later),
        "a frame sent a microsecond before it ended"
    );
    assert!(
        !witness.stalled_between(back, later),
        "a frame sent in the microsecond it ended"
    );
    assert!(
        !witness.stalled_between(back + Duration::from_micros(1), later),
        "a frame sent after it"
    );
    assert!(
        !witness.stalled_between(at(&witness, 1), at(&witness, 9_999)),
        "a frame that landed before it"
    );
    // The gap began in the microsecond the thread last ran: an answer
    // owed only until then is not crossed, and one owed a microsecond
    // longer is.
    assert!(
        !witness.stalled_between(at(&witness, 1), sent),
        "a frame that landed in the microsecond it began"
    );
    assert!(
        witness.stalled_between(at(&witness, 1), sent + Duration::from_micros(1)),
        "a frame that landed a microsecond into it"
    );
}

#[test]
fn the_first_frame_in_flight_starts_the_promise_and_a_later_one_does_not() {
    let promise = Duration::from_micros(PROMISE_MICROS);
    let witness = Witness::new();
    let first = at(&witness, 1_000);
    witness.took_flight(first);
    // A second frame takes flight while the first is out: the thread
    // owed a run since the first, and the second moves nothing.
    let second = first + promise;
    witness.took_flight(second);
    assert!(witness.stalled_between(second, second + Duration::from_micros(1)));
    // Both land; the next first frame starts the promise over, and
    // the stamp that follows in time closes no gap.
    witness.landed();
    witness.landed();
    let third = at(&witness, 60_000);
    witness.took_flight(third);
    assert!(!witness.stalled_between(third, third + promise));
    witness.stamp_at(third + promise);
    assert!(!witness.stalled_between(third, third + promise));
}

/// An answer's arrival is a run of the I/O thread: the gap it came
/// across is kept for the frame that waited, and a frame sent after
/// it starts in time, in the gap's closing microsecond too. Nothing
/// ticks here: no I/O thread runs, and this thread stands for it. A
/// try whose runner held the thread past the promise between the
/// arrival and the next frame says nothing of the witness and is made
/// again.
#[test]
fn an_arriving_answer_stamps_the_witness_and_keeps_the_gap_before_it() {
    let events = Arc::new(Counted::default());
    let until = Instant::now() + Duration::from_secs(60);
    loop {
        let (lane, _rx) = open_lane(&events, 16);
        let first = lane.next_id();
        let sent = lane
            .send(first, proto::PluginDecide::default())
            .expect("sent");
        std::thread::sleep(Duration::from_micros(PROMISE_MICROS + 2_000));
        let before = Instant::now();
        lane.deliver(answer(first));
        // The first frame is still in its place: this one starts no
        // promise of its own.
        let after = lane
            .send(lane.next_id(), proto::PluginDecide::default())
            .expect("sent");
        if after.saturating_duration_since(before) > Duration::from_micros(PROMISE_MICROS) {
            assert!(Instant::now() < until, "the runner held every try");
            continue;
        }
        assert!(
            !lane.stalled_between(after, after),
            "the delivery was a run of the thread"
        );
        let soon = Instant::now() + Duration::from_secs(5);
        let Wait::Answered(arrived, _) = lane.wait_answer(first, soon) else {
            panic!("the first frame was answered");
        };
        assert!(
            lane.stalled_between(sent, arrived),
            "the answer came across a gap past the promise"
        );
        // The gap ends at the arrival, exclusive: a frame that took
        // flight in its closing microsecond is not crossed, and one a
        // microsecond earlier is.
        assert!(
            !lane.stalled_between(arrived, arrived),
            "a frame in the gap's closing microsecond"
        );
        assert!(
            lane.stalled_between(arrived - Duration::from_micros(1), arrived),
            "a frame a microsecond before the gap closed"
        );
        // The gap began in the microsecond the first frame took
        // flight: an answer owed only until then is not crossed, and
        // one owed a microsecond longer is.
        assert!(
            !lane.stalled_between(sent, sent),
            "an answer owed until the gap's first microsecond"
        );
        assert!(
            lane.stalled_between(sent, sent + Duration::from_micros(1)),
            "an answer owed a microsecond into the gap"
        );
        break;
    }
}

/// The ticker stamps the witness while a frame is in flight and
/// parks with none: an idle runtime wakes for nothing.
#[test]
fn the_ticker_stamps_while_a_frame_is_in_flight_and_parks_with_none() {
    let runtime = Runtime::new(&OpenDecisions::new()).expect("a runtime");
    let witness = Arc::clone(&runtime.witness);
    let ran = || witness.ran.load(Ordering::Relaxed);
    let advances = |within: Duration| {
        let (from, until) = (ran(), Instant::now() + within);
        while ran() == from && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(1));
        }
        ran() != from
    };
    assert!(
        !advances(Duration::from_millis(30)),
        "no frame in flight: nothing ticks"
    );
    witness.took_flight(Instant::now());
    let sent = ran();
    assert!(
        advances(Duration::from_secs(5)),
        "a frame in flight: a tick"
    );
    assert!(advances(Duration::from_secs(5)), "and the next");
    assert!(ran() > sent);
    witness.landed();
    // The tick in flight when the frame landed may still stamp.
    std::thread::sleep(Duration::from_millis(10));
    assert!(
        !advances(Duration::from_millis(30)),
        "the frame landed: the ticker parks"
    );
}

/// A runtime makes its lanes: a lane counts its frames on its maker's
/// witness, the one that runtime's I/O thread stamps, and on no other's.
#[test]
fn a_runtimes_lane_counts_its_frames_on_that_runtimes_witness() {
    let (maker, other) = (
        Runtime::new(&OpenDecisions::new()).expect("a runtime"),
        Runtime::new(&OpenDecisions::new()).expect("another runtime"),
    );
    let events = Arc::new(Counted::default());
    let listener: Arc<dyn LaneEvents> = Arc::clone(&events) as _;
    let lane = maker.lane(nowhere(), Arc::downgrade(&listener));
    assert!(Arc::ptr_eq(&lane.witness, &maker.witness));
    assert!(lane.is_of(&maker));
    assert!(!lane.is_of(&other));
}

/// A lane asked for its stream through another runtime than its maker
/// would count its flights where no thread stamps: a debug build stops at
/// the ask.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the runtime that made it")]
fn a_lane_asked_through_another_runtime_than_its_maker_stops_a_debug_build() {
    let (maker, other) = (
        Runtime::new(&OpenDecisions::new()).expect("a runtime"),
        Runtime::new(&OpenDecisions::new()).expect("another runtime"),
    );
    let events = Arc::new(Counted::default());
    let listener: Arc<dyn LaneEvents> = Arc::clone(&events) as _;
    let lane = Arc::new(maker.lane(nowhere(), Arc::downgrade(&listener)));
    lane.connect(&other);
}

#[test]
fn a_lane_counts_its_frames_in_flight_on_the_witness_until_each_lands() {
    let events = Arc::new(Counted::default());
    let (lane, _rx) = open_lane(&events, 1);
    let flying = || lane.witness.in_flight.load(Ordering::Relaxed);
    let (answered_id, missed, forgotten, refused) = (
        lane.next_id(),
        lane.next_id(),
        lane.next_id(),
        lane.next_id(),
    );
    lane.send(answered_id, proto::PluginDecide::default())
        .expect("sent");
    assert_eq!(flying(), 1);
    // The stream takes one frame: the next is not sent and never flew.
    assert_eq!(
        lane.send(refused, proto::PluginDecide::default()).err(),
        Some(SendError::Full)
    );
    assert_eq!(flying(), 1);
    lane.deliver(answer(answered_id));
    let soon = Instant::now() + Duration::from_secs(5);
    assert_eq!(
        answered(lane.wait_answer(answered_id, soon)),
        Some(answered_id)
    );
    assert_eq!(flying(), 0, "an answered frame landed");
    let (lane, _rx) = open_lane(&events, 16);
    let flying = || lane.witness.in_flight.load(Ordering::Relaxed);
    lane.send(missed, proto::PluginDecide::default())
        .expect("sent");
    lane.send(forgotten, proto::PluginDecide::default())
        .expect("sent");
    assert_eq!(flying(), 2);
    assert!(matches!(
        lane.wait_answer(missed, Instant::now()),
        Wait::Missed
    ));
    assert_eq!(flying(), 1, "a missed frame landed");
    lane.forget(forgotten);
    lane.forget(forgotten);
    assert_eq!(flying(), 0, "a forgotten frame landed, once");
    lane.forget(0);
    assert_eq!(flying(), 0, "an id of zero names no frame");
    // A frame still out when its lane goes lands with it.
    let witness = Arc::clone(&lane.witness);
    lane.send(lane.next_id(), proto::PluginDecide::default())
        .expect("sent");
    assert_eq!(witness.in_flight.load(Ordering::Relaxed), 1);
    drop(lane);
    assert_eq!(witness.in_flight.load(Ordering::Relaxed), 0);
}

/// The I/O loop takes an ended stream's task out of its set: a stream
/// that ends and reopens for the life of a process leaves no task behind
/// for each end. Here the connect finds no socket, so the lane's task
/// fails the lane and ends.
#[tokio::test]
async fn an_ended_streams_task_leaves_the_loops_set() {
    let (commands, taken) = mpsc::unbounded_channel();
    let mut looping = JoinSet::new();
    looping.spawn(run_commands(taken, Arc::new(Witness::new())));
    let events = Arc::new(Counted::default());
    let lane = Arc::new(cold_lane(&events));
    assert!(commands.send(Command::Connect(Arc::clone(&lane))).is_ok());
    let mut waits = 0;
    while lane.failure().is_none() {
        waits += 1;
        assert!(waits < 10_000, "the connect to no socket never failed");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // The task ended with the lane's failure: the loop's next turn takes
    // it out.
    for _ in 0..3 {
        tokio::task::yield_now().await;
    }
    drop(commands);
    let streams = looping
        .join_next()
        .await
        .expect("the loop's task")
        .expect("the loop ends with its commands");
    assert_eq!(streams.len(), 1, "the witness's ticker alone is left");
}

/// What a lane tells its plugin, with the instant of each failed stream
/// kept.
#[derive(Default)]
struct Dated {
    failed: parking_lot::Mutex<Vec<Instant>>,
}

impl LaneEvents for Dated {
    fn connected(&self) {}
    fn stream_failed(&self, _why: &str, at: Instant) {
        self.failed.lock().push(at);
    }
    fn late_answer(&self) {}
    fn observe_failed(&self, _why: &str) {}
}

/// A failed stream's telling carries the instant its lane dated the
/// failure by, the one its pause runs from: the count on the breaker
/// reads that instant and not the telling's own, which a held I/O thread
/// moves. Here the connect finds no socket.
#[tokio::test]
async fn a_failed_streams_telling_carries_the_instant_its_lane_was_dated_by() {
    let (commands, taken) = mpsc::unbounded_channel();
    let mut looping = JoinSet::new();
    looping.spawn(run_commands(taken, Arc::new(Witness::new())));
    let events = Arc::new(Dated::default());
    let listener: Arc<dyn LaneEvents> = Arc::clone(&events) as _;
    let lane = Arc::new(Lane::new(
        nowhere(),
        Arc::downgrade(&listener),
        Arc::new(Witness::new()),
    ));
    assert!(commands.send(Command::Connect(Arc::clone(&lane))).is_ok());
    let mut waits = 0;
    while events.failed.lock().is_empty() {
        waits += 1;
        assert!(waits < 10_000, "the connect to no socket was never told");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let dated = match &*lane.state.lock() {
        LaneState::Failed(_, at) => Some(*at),
        _ => None,
    };
    assert_eq!(
        events.failed.lock().first().copied(),
        dated,
        "the telling carries the lane's own instant"
    );
}

/// What an `Observe` lane tells its plugin: each telling's words with
/// the loop's clock, and a panic at each of the first `panics`.
struct Told {
    panics: usize,
    told: parking_lot::Mutex<Vec<(tokio::time::Instant, String)>>,
}

impl LaneEvents for Told {
    fn connected(&self) {}
    fn stream_failed(&self, _why: &str, _at: Instant) {}
    fn late_answer(&self) {}
    fn observe_failed(&self, why: &str) {
        let tellings = {
            let mut told = self.told.lock();
            told.push((tokio::time::Instant::now(), why.to_owned()));
            told.len()
        };
        assert!(tellings > self.panics, "telling {tellings} panics");
    }
}

/// A lane nothing listens behind, with its listener, whose first
/// `panics` tellings panic, and its queue's sender. Its plugin's
/// `Decide` streams stand on version 1, so its calls open.
fn told(panics: usize) -> (Arc<Told>, ObserveLane, mpsc::Sender<proto::ObserveRequest>) {
    let events = Arc::new(Told {
        panics,
        told: parking_lot::Mutex::new(Vec::new()),
    });
    let listener: Arc<dyn LaneEvents> = Arc::clone(&events) as _;
    let (raise, raised) = mpsc::channel(4);
    let lane = ObserveLane::new(peer(standing_on(1)), Arc::downgrade(&listener), raised);
    (events, lane, raise)
}

/// The error lines a case's own subscriber heard on its thread.
#[derive(Clone, Default)]
struct Errors(Arc<AtomicU64>);

impl tracing::Subscriber for Errors {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() == tracing::Level::ERROR
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// A failure says its line once an outage and a panicked turn once a
/// streak. One edge arms both again: the end of a call that stood.
#[test]
fn the_end_of_a_call_that_stood_arms_both_lines() {
    let mut said = Said::default();
    assert!(said.ended(false), "an outage's first failure");
    assert!(!said.ended(false));
    assert!(said.panicked(), "a streak's first panic");
    assert!(!said.panicked());
    assert!(!said.ended(false), "a panicked turn arms nothing");
    assert!(said.ended(true), "the end of a call that stood");
    assert!(said.panicked(), "the streak is armed with the outage");
    assert!(!said.ended(false));
    assert!(!said.panicked());
}

/// One step of a case's script of `Observe` calls.
enum Step {
    /// No call opens.
    Refused,
    /// The opener panics.
    Panics,
    /// A call opens and ends by its outcome this long after.
    Ends(Duration),
    /// A call opens, lets its receiver go this long after as an event
    /// is raised, and says no outcome.
    Goes(Duration),
}

type Scripted = Pin<Box<dyn Future<Output = Result<(), tonic::Status>>>>;

/// What a script of calls saw: the loop's clock at each open, and the
/// number of each event a call was sent.
#[derive(Default)]
struct Ran {
    opens: RefCell<Vec<tokio::time::Instant>>,
    taken: Rc<RefCell<Vec<u64>>>,
}

/// The events task of `lane` over a script of calls, then an hour of
/// the loop's paused clock in which no call opens.
async fn scripted(
    lane: &ObserveLane,
    raise: &mpsc::Sender<proto::ObserveRequest>,
    steps: Vec<Step>,
) -> Ran {
    let ran = Ran::default();
    let mut steps = steps.into_iter();
    let open = |_version: u32| {
        ran.opens.borrow_mut().push(tokio::time::Instant::now());
        let (step, raise, taken) = (steps.next(), raise.clone(), Rc::clone(&ran.taken));
        async move {
            let (tx, mut rx) = mpsc::channel::<proto::ObserveRequest>(OBSERVE_WINDOW);
            let (after, goes) = match step {
                None => return std::future::pending().await,
                Some(Step::Refused) => return Err(Cow::Borrowed("refused")),
                Some(Step::Panics) => panic!("the opener panics"),
                Some(Step::Ends(after)) => (after, false),
                Some(Step::Goes(after)) => (after, true),
            };
            let call: Scripted = Box::pin(async move {
                let end = tokio::time::sleep(after);
                tokio::pin!(end);
                loop {
                    tokio::select! {
                        () = &mut end => break,
                        Some(sent) = rx.recv() => {
                            if let Some(proto::observe_request::Request::Event(event)) =
                                sent.request
                            {
                                taken.borrow_mut().push(event.seq);
                            }
                        }
                    }
                }
                if goes {
                    drop(rx);
                    raise.try_send(raised()).expect("the queue takes an event");
                    return std::future::pending().await;
                }
                Err(tonic::Status::unavailable("the plugin went"))
            });
            Ok::<Opened<Scripted>, Cow<'static, str>>((call, tx))
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(3_600), observe(lane, open)).await;
    ran
}

/// An `Observe` call names the version its plugin's `Decide` streams
/// last fixed. None opens before the first settle, which opens one. It
/// lives on when the last stream ends. Once a settle fixes another
/// version it is ended, its answer awaited, and the next call names that
/// one, in the same instant and with nothing told: no call failed.
#[tokio::test(start_paused = true)]
async fn an_observe_call_names_the_version_its_plugins_decide_streams_fixed() {
    let events = Arc::new(Told {
        panics: 0,
        told: parking_lot::Mutex::new(Vec::new()),
    });
    let listener: Arc<dyn LaneEvents> = Arc::clone(&events) as _;
    let (_raise, raised) = mpsc::channel(4);
    let lane = ObserveLane::new(nowhere(), Arc::downgrade(&listener), raised);
    // Each call stands until its request stream ends, then answers.
    let (named, answered) = (RefCell::new(Vec::new()), Rc::new(Cell::new(0)));
    let open = |version: u32| {
        named
            .borrow_mut()
            .push((tokio::time::Instant::now(), version));
        let answered = Rc::clone(&answered);
        async move {
            let (tx, mut rx) = mpsc::channel::<proto::ObserveRequest>(OBSERVE_WINDOW);
            let call: Scripted = Box::pin(async move {
                while rx.recv().await.is_some() {}
                answered.set(answered.get() + 1);
                Ok::<(), tonic::Status>(())
            });
            Ok::<Opened<Scripted>, Cow<'static, str>>((call, tx))
        }
    };
    let (spoken, minute) = (&lane.peer.spoken, Duration::from_secs(60));
    let settles = async {
        tokio::time::sleep(minute).await;
        assert!(named.borrow().is_empty(), "no stream has settled");
        let first = spoken.settle(1).expect("the first settle");
        tokio::time::sleep(minute).await;
        drop(first);
        tokio::time::sleep(minute).await;
        let _second = spoken.settle(2).expect("the next settle fixes it anew");
        tokio::time::sleep(minute).await;
    };
    let start = tokio::time::Instant::now();
    tokio::select! {
        () = observe(&lane, open) => panic!("the plugin is loaded"),
        () = settles => {}
    }
    assert_eq!(
        named.into_inner(),
        [(start + minute, 1), (start + minute * 3, 2)],
        "one call a version, each opened as its settle fell"
    );
    assert!(events.told.lock().is_empty(), "no call failed");
    assert_eq!(answered.get(), 1, "the ended call's answer was awaited");
}

/// One event an `Observe` call was written: the version the call named,
/// the event's number, whether it is a left word, and the instant.
type Heard = (u32, u64, bool, tokio::time::Instant);

/// A lane nothing listens behind, none of its plugin's `Decide` streams
/// settled, with its listener and its queue's sender.
fn unsettled() -> (Arc<Told>, ObserveLane, mpsc::Sender<proto::ObserveRequest>) {
    let events = Arc::new(Told {
        panics: 0,
        told: parking_lot::Mutex::new(Vec::new()),
    });
    let listener: Arc<dyn LaneEvents> = Arc::clone(&events) as _;
    let (raise, raised) = mpsc::channel(4);
    let lane = ObserveLane::new(nowhere(), Arc::downgrade(&listener), raised);
    (events, lane, raise)
}

/// One `Observe` call that stands until the host ends it, and says each
/// event it was written: an opener's answer.
fn standing_call(
    version: u32,
    heard: Rc<RefCell<Vec<Heard>>>,
) -> impl Future<Output = Result<Opened<Scripted>, Cow<'static, str>>> {
    use crate::plugins::grpc::view;
    let (tx, mut rx) = mpsc::channel::<proto::ObserveRequest>(OBSERVE_WINDOW);
    let call: Scripted = Box::pin(async move {
        while let Some(written) = rx.recv().await {
            if let Some(proto::observe_request::Request::Event(event)) = written.request {
                let left = event
                    .event
                    .as_ref()
                    .is_some_and(|word| view::since(word) == 2);
                let now = tokio::time::Instant::now();
                heard.borrow_mut().push((version, event.seq, left, now));
            }
        }
        Ok(())
    });
    std::future::ready(Ok((call, tx)))
}

/// A membership word as the door raises it, behind no refusal.
fn word(change: &lease::Membership<'_>) -> proto::ObserveRequest {
    use crate::plugins::grpc::view;
    proto::ObserveRequest {
        request: Some(proto::observe_request::Request::Event(
            proto::LifecycleEvent {
                seq: 0,
                event: Some(view::membership(change)),
            },
        )),
    }
}

/// A word raised behind a settle that moves the version to 2 is written
/// by the call that names 2: a left word, and a word every version
/// carries alike. Here a call that named 1 still stands as the settle
/// falls, and the word is raised in the same instant: the call's next
/// turn finds the word and the move together, round after round. The
/// call on 1 is written nothing, the call on 2 the word, numbered, and
/// no call failed.
#[tokio::test(start_paused = true)]
async fn a_word_raised_behind_a_move_is_written_by_the_call_that_names_it() {
    use crate::{engine::view::WorkerId, tracking::lease::Membership};
    let (models, minute) = (["m".to_owned()], Duration::from_secs(60));
    for round in 0..32 {
        // A left word in one round, a relabelling in the next.
        let is_left = round % 2 == 0;
        let (events, lane, raise) = unsettled();
        let heard = Rc::new(RefCell::new(Vec::new()));
        let open = |version| standing_call(version, Rc::clone(&heard));
        let spoken = &lane.peer.spoken;
        let settles = async {
            let first = spoken.settle(1).expect("the first settle");
            tokio::time::sleep(minute).await;
            drop(first);
            let _second = spoken.settle(2).expect("the next settle fixes it anew");
            let raised = if is_left {
                Membership::Left {
                    id: WorkerId(7),
                    models: &models,
                }
            } else {
                Membership::Relabelled(WorkerId(7))
            };
            raise.try_send(word(&raised)).expect("the queue takes it");
            tokio::time::sleep(minute).await;
        };
        tokio::select! {
            () = observe(&lane, open) => panic!("the plugin is loaded"),
            () = settles => {}
        }
        let heard: Vec<_> = heard
            .borrow()
            .iter()
            .map(|&(version, seq, left, _)| (version, seq, left))
            .collect();
        assert_eq!(heard, [(2, 1, is_left)], "the call on 2 writes the word");
        assert!(events.told.lock().is_empty(), "no call failed");
    }
}

/// The staging of the four cases below. The last stream on 1 ends and
/// the call that named 1 lives on. A minute later a left word is raised
/// and a relabelling behind it, and a frame and a half after that a
/// stream settles on `next`: inside a frame of the kept word's wait, so
/// the settle itself must wake it. Where it does not stand, that stream
/// ends before the events task looks. It says that settle's instant and
/// what the calls were written; no call failed.
async fn a_stream_settles_behind_a_left_word(
    next: u32,
    stands: bool,
) -> (tokio::time::Instant, Vec<Heard>) {
    use crate::{engine::view::WorkerId, tracking::lease::Membership};
    let (models, minute) = (["m".to_owned()], Duration::from_secs(60));
    let (events, lane, raise) = unsettled();
    let heard = Rc::new(RefCell::new(Vec::new()));
    let open = |version| standing_call(version, Rc::clone(&heard));
    let (spoken, settled) = (&lane.peer.spoken, Cell::new(None));
    let settles = async {
        let first = spoken.settle(1).expect("the first settle");
        tokio::time::sleep(minute).await;
        drop(first);
        tokio::time::sleep(minute).await;
        let left = Membership::Left {
            id: WorkerId(7),
            models: &models,
        };
        let relabelled = Membership::Relabelled(WorkerId(7));
        raise.try_send(word(&left)).expect("the queue takes it");
        raise
            .try_send(word(&relabelled))
            .expect("the queue takes it");
        tokio::time::sleep(lane.peer.connect * 3 / 2).await;
        let settle = spoken.settle(next).expect("the next settle");
        let _next = stands.then_some(settle);
        settled.set(Some(tokio::time::Instant::now()));
        tokio::time::sleep(minute).await;
    };
    tokio::select! {
        () = observe(&lane, open) => panic!("the plugin is loaded"),
        () = settles => {}
    }
    assert!(events.told.lock().is_empty(), "no call failed");
    let heard = heard.borrow().clone();
    (settled.get().expect("the settle fell"), heard)
}

/// A left word raised while no stream stands waits for the stream that
/// settles next, though the events call that named 1 still stands as
/// it is raised. Here that stream settles on 2: the call that names 2
/// writes the word and the relabelling behind it, in their order and
/// numbered, as the settle falls. The call on 1 was written neither.
#[tokio::test(start_paused = true)]
async fn a_left_word_raised_with_no_stream_standing_is_written_once_one_settles_on_2() {
    let (settled, heard) = a_stream_settles_behind_a_left_word(2, true).await;
    assert_eq!(heard, [(2, 1, true, settled), (2, 2, false, settled)]);
}

/// The same word where the next stream settles on 1: no stream of the
/// plugin carries it, so it is let go and takes no number. The
/// relabelling behind it waited with it: the call that opens as the
/// settle falls writes it, the first event.
#[tokio::test(start_paused = true)]
async fn a_left_word_raised_with_no_stream_standing_is_let_go_once_one_settles_on_1() {
    let (settled, heard) = a_stream_settles_behind_a_left_word(1, true).await;
    assert_eq!(heard, [(1, 1, false, settled)]);
}

/// A stream that settles on 1 and ends before the events task looks
/// still says what becomes of a kept left word. The word is let go and
/// takes no number, and the call that opens as the settle falls writes
/// the relabelling behind it, the first event.
#[tokio::test(start_paused = true)]
async fn a_kept_left_word_is_let_go_by_a_settle_on_1_whose_stream_has_ended() {
    let (settled, heard) = a_stream_settles_behind_a_left_word(1, false).await;
    assert_eq!(heard, [(1, 1, false, settled)]);
}

/// The same where the stream settled on 2: the call that names 2 opens
/// as the settle falls and writes the word first, though no stream
/// stands.
#[tokio::test(start_paused = true)]
async fn a_kept_left_word_is_written_behind_a_settle_on_2_whose_stream_has_ended() {
    let (settled, heard) = a_stream_settles_behind_a_left_word(2, false).await;
    assert_eq!(heard, [(2, 1, true, settled), (2, 2, false, settled)]);
}

/// A call that stood and was ended by the host arms the outage's line
/// again, as one that ended by itself does. Here a call fails at once
/// and says the line, the next stands until a settle fixes another
/// version, and the call that names that one fails at once. Behind a
/// stand that is a new outage, and its line is said; a millisecond
/// short of one it is the same outage, and quiet.
#[tokio::test(start_paused = true)]
async fn a_call_that_stood_and_was_ended_for_a_move_arms_the_outage_line() {
    let short = STAND - Duration::from_millis(1);
    for (stand, lines) in [(short, 1), (STAND, 2)] {
        let (events, lane, _raise) = unsettled();
        let opens = Cell::new(0);
        // The second call alone stands, until the host ends it.
        let open = |_version: u32| {
            opens.set(opens.get() + 1);
            let stands = opens.get() == 2;
            async move {
                let (tx, mut rx) = mpsc::channel::<proto::ObserveRequest>(OBSERVE_WINDOW);
                let call: Scripted = Box::pin(async move {
                    if stands {
                        while rx.recv().await.is_some() {}
                        return Ok(());
                    }
                    Err(tonic::Status::unavailable("the plugin went"))
                });
                Ok::<Opened<Scripted>, Cow<'static, str>>((call, tx))
            }
        };
        let (spoken, frame) = (&lane.peer.spoken, lane.peer.connect);
        let settles = async {
            let first = spoken.settle(1).expect("the first settle");
            // The second call opens a frame behind the first one's end.
            tokio::time::sleep(frame + stand).await;
            drop(first);
            let _second = spoken.settle(2).expect("the next settle fixes it anew");
            tokio::time::sleep(frame / 2).await;
        };
        tokio::select! {
            () = observe(&lane, open) => panic!("the plugin is loaded"),
            () = settles => {}
        }
        assert_eq!(opens.get(), 3, "one failed, one stood, one failed");
        let told = events.told.lock();
        assert_eq!(told.len(), lines, "a stand of {stand:?}: {told:?}");
    }
}

/// A word taken ahead of the move's signal is kept all the same: it is
/// judged by the plugin's version as it stands. Here a call that named
/// 1 takes a left word while a stream stands on 2 and no signal comes:
/// the call yields, the word kept and no number taken. The call that
/// names 2 writes it first, numbered.
#[tokio::test(start_paused = true)]
async fn a_word_taken_ahead_of_the_moves_signal_is_kept_for_the_call_that_carries_it() {
    use crate::{engine::view::WorkerId, tracking::lease::Membership};
    let (models, hour) = (["m".to_owned()], Duration::from_secs(3_600));
    let left = Membership::Left {
        id: WorkerId(7),
        models: &models,
    };
    let (raise, mut queue) = mpsc::channel(4);
    raise.try_send(word(&left)).expect("the queue takes it");
    let (spoken, mut kept) = (standing_on(2), None);
    let (mut sent, mut lost) = (0, 0);
    let heard = Rc::new(RefCell::new(Vec::new()));
    for version in [1, 2] {
        let (call, tx) = standing_call(version, Rc::clone(&heard))
            .await
            .expect("a call opens");
        let requests = Requests {
            tx,
            version,
            spoken: &spoken,
            kept: &mut kept,
        };
        let (queue, never) = (&mut queue, std::future::pending());
        let turn = super::forward(call, requests, queue, &mut sent, &mut lost, hour, never);
        // The call on 2 stands: an hour ends its turn here.
        match (version, tokio::time::timeout(hour, turn).await) {
            (1, Ok(Turn::Moved { stood: false })) => {
                assert!(kept.is_some(), "the word is kept");
                assert_eq!((sent, lost), (0, 0), "and took no number");
            }
            (2, Err(_)) => assert!(kept.is_none(), "the word is written"),
            _ => panic!("the call on {version} ended another way"),
        }
    }
    let heard: Vec<_> = heard
        .borrow()
        .iter()
        .map(|&(version, seq, left, _)| (version, seq, left))
        .collect();
    assert_eq!(heard, [(2, 1, true)], "the call on 2 writes the word");
    assert_eq!((sent, lost), (1, 0));
}

/// An event that finds its call's window full is written by the next
/// call. Here the call that named 1 is written a full window and reads
/// none of it, two more events are raised, and a settle moves the
/// version to 2. The call that names 2 writes both, numbered behind the
/// window: the one the first call held took no number there.
#[tokio::test(start_paused = true)]
async fn an_event_that_finds_its_call_full_is_written_by_the_next_call() {
    use crate::{engine::view::WorkerId, tracking::lease::Membership};
    let (events, lane, raise) = unsettled();
    let heard = Rc::new(RefCell::new(Vec::new()));
    let open = |version| {
        let heard = Rc::clone(&heard);
        async move {
            if version == 2 {
                return standing_call(version, heard).await;
            }
            // The call on 1 stands and reads nothing: its window fills.
            let (tx, unread) = mpsc::channel::<proto::ObserveRequest>(OBSERVE_WINDOW);
            let call: Scripted = Box::pin(async move {
                let _unread = unread;
                std::future::pending().await
            });
            Ok((call, tx))
        }
    };
    let (spoken, minute) = (&lane.peer.spoken, Duration::from_secs(60));
    let relabelled = Membership::Relabelled(WorkerId(7));
    let settles = async {
        let first = spoken.settle(1).expect("the first settle");
        for _ in 0..OBSERVE_WINDOW + 2 {
            let raised = raise.send(word(&relabelled)).await;
            raised.expect("the queue takes it");
        }
        tokio::time::sleep(minute).await;
        drop(first);
        let _second = spoken.settle(2).expect("the next settle fixes it anew");
        tokio::time::sleep(minute).await;
    };
    tokio::select! {
        () = observe(&lane, open) => panic!("the plugin is loaded"),
        () = settles => {}
    }
    assert!(events.told.lock().is_empty(), "no call failed");
    let window = u64::try_from(OBSERVE_WINDOW).expect("a small window");
    let numbers: Vec<_> = heard
        .borrow()
        .iter()
        .map(|&(version, seq, ..)| (version, seq))
        .collect();
    assert_eq!(numbers, [(2, window + 1), (2, window + 2)]);
}

/// A word kept while the version fixed already carries it is written,
/// though no stream stands and none has settled since. Here a stream
/// settled on 2 and ended, and a call that named 1 then took a left
/// word: the word is kept under the settles of that instant. The next
/// call names 2 and writes the word first.
#[tokio::test(start_paused = true)]
async fn a_word_kept_under_a_version_that_carries_it_is_written_with_no_stream_standing() {
    use crate::{engine::view::WorkerId, tracking::lease::Membership};
    let models = ["m".to_owned()];
    let left = Membership::Left {
        id: WorkerId(7),
        models: &models,
    };
    let (events, lane, _raise) = unsettled();
    let mut queue = lane.queue.lock().take().expect("the lane's queue");
    drop(lane.peer.spoken.settle(2).expect("the settle"));
    let mut kept = Some(super::Kept {
        word: word(&left),
        settles: 1,
    });
    let heard = Rc::new(RefCell::new(Vec::new()));
    let mut open = |version| standing_call(version, Rc::clone(&heard));
    let (mut sent, mut lost, mut said) = (0, 0, Said::default());
    let task = super::observe_streams(
        &lane, &mut queue, &mut kept, &mut sent, &mut lost, &mut said, &mut open,
    );
    // The call on 2 stands: a minute ends the case's look at it.
    let stood = tokio::time::timeout(Duration::from_secs(60), task).await;
    assert!(stood.is_err(), "the call on 2 stands");
    let heard: Vec<_> = heard
        .borrow()
        .iter()
        .map(|&(version, seq, left, _)| (version, seq, left))
        .collect();
    assert_eq!(heard, [(2, 1, true)], "the call on 2 writes the kept word");
    assert!(events.told.lock().is_empty(), "no call failed");
}

/// One call's turn where a stream stands on protocol 1 and nothing
/// moves its plugin's version: how the call ended, or `None` once the
/// plugin is dropped.
async fn forward<C, T>(
    call: C,
    tx: mpsc::Sender<proto::ObserveRequest>,
    queue: &mut mpsc::Receiver<proto::ObserveRequest>,
    sent: &mut u64,
    lost: &mut u64,
    frame: Duration,
) -> Option<Ended>
where
    C: Future<Output = Result<T, tonic::Status>>,
{
    let (spoken, mut kept) = (standing_on(1), None);
    let requests = Requests {
        tx,
        version: 1,
        spoken: &spoken,
        kept: &mut kept,
    };
    let never = std::future::pending();
    match super::forward(call, requests, queue, sent, lost, frame, never).await {
        Turn::Ended(ended) => Some(ended),
        Turn::Dropped | Turn::Moved { .. } => None,
    }
}

/// The first stream to settle fixes its plugin's version. While one
/// stands a hello offers that version alone, and an ack that names
/// another comes back with the fixed one. The end of the last stream
/// frees the version, and the next settle fixes it anew.
#[test]
fn the_first_settle_fixes_a_plugins_version_until_its_last_stream_ends() {
    let (spoken, host) = (Spoken::default(), 1..=2);
    assert_eq!(spoken.offer(&host), (1, 2), "no stream stands");
    assert!(!spoken.stands_under(2), "none stands: the write decides");
    let first = spoken.settle(2).expect("the first settle fixes it");
    assert_eq!(spoken.offer(&host), (2, 2), "one stands: its version alone");
    assert!(!spoken.stands_under(2), "a stream on 2 hears a word of 2");
    let second = spoken.settle(2).expect("a second stream joins it");
    assert_eq!(spoken.settle(1).err(), Some(2), "an ack names another");
    drop(first);
    assert_eq!(spoken.offer(&host), (2, 2), "one stands yet");
    drop(second);
    assert_eq!(spoken.offer(&host), (1, 2), "the last end frees it");
    let _anew = spoken.settle(1).expect("the next settle fixes it anew");
    assert_eq!(spoken.offer(&host), (1, 1));
    assert!(
        spoken.stands_under(2) && !spoken.stands_under(1),
        "a stream on 1 hears no word of 2"
    );
}

fn raised() -> proto::ObserveRequest {
    proto::ObserveRequest {
        request: Some(proto::observe_request::Request::Event(
            proto::LifecycleEvent::default(),
        )),
    }
}

/// A turn that finds the call's receiver gone, ahead of the call's own
/// outcome, ends the call as that outcome does. Here the call takes one
/// event and lets its receiver go, a second event is raised, and the
/// outcome comes a moment later: the turn waits for it, and the end says
/// the outcome, dated where the turn saw it. The second event takes no
/// number: it rests in the task's keep for the next call. An event
/// carried is no stand.
#[tokio::test(start_paused = true)]
async fn a_send_that_sees_a_calls_end_first_says_the_calls_outcome() {
    let (raise, mut queue) = mpsc::channel(4);
    raise.try_send(raised()).expect("the queue takes an event");
    let (tx, mut rx) = mpsc::channel(OBSERVE_WINDOW);
    let call = async move {
        assert!(rx.recv().await.is_some(), "the call takes an event");
        drop(rx);
        raise.try_send(raised()).expect("the queue takes an event");
        tokio::time::sleep(Duration::from_millis(1)).await;
        Err::<(), _>(tonic::Status::unavailable("the plugin went"))
    };
    let (mut sent, mut lost) = (0, 0);
    let before = tokio::time::Instant::now();
    let ended = forward(call, tx, &mut queue, &mut sent, &mut lost, RECONNECT_PAUSE)
        .await
        .expect("the plugin is loaded");
    assert!(ended.why.contains("the plugin went"), "{}", ended.why);
    assert_eq!(ended.at, before, "the end is dated where the turn saw it");
    assert!(!ended.stood, "an event carried is no stand");
    assert_eq!(sent, 1, "the second event took no number");
    assert!(queue.try_recv().is_err(), "and left the queue for the keep");
}

/// An event taken as the call's receiver goes is kept for the next call
/// and takes no number. Here a turn waits for an event. The receiver
/// goes, an event is raised, and the call says its outcome a moment
/// later: the event is kept, unnumbered.
#[tokio::test(start_paused = true)]
async fn an_event_taken_as_the_calls_receiver_goes_is_kept_for_the_next_call() {
    let (raise, mut queue) = mpsc::channel(4);
    let (tx, rx) = mpsc::channel(OBSERVE_WINDOW);
    let call = async {
        tokio::time::sleep(Duration::from_millis(2)).await;
        Err::<(), _>(tonic::Status::unavailable("the plugin went"))
    };
    let goes = async {
        tokio::time::sleep(Duration::from_millis(1)).await;
        drop(rx);
        raise.try_send(raised()).expect("the queue takes an event");
        std::future::pending::<()>().await;
    };
    let (spoken, mut kept) = (standing_on(1), None);
    let (mut sent, mut lost) = (0, 0);
    let requests = Requests {
        tx,
        version: 1,
        spoken: &spoken,
        kept: &mut kept,
    };
    let (queue, frame, never) = (&mut queue, RECONNECT_PAUSE, std::future::pending());
    let turn = super::forward(call, requests, queue, &mut sent, &mut lost, frame, never);
    let turn = tokio::select! {
        turn = turn => turn,
        () = goes => unreachable!("the receiver's end never ends"),
    };
    assert!(matches!(turn, Turn::Ended(_)), "the call ended");
    assert!(kept.is_some(), "the event is kept");
    assert_eq!(sent, 0, "and took no number");
}

/// A word the call's version has no kind for is judged though the
/// call's receiver has gone. Here a call that named 1 waits for an
/// event, and a stream stands on 1. The receiver goes and a left word
/// is raised: no stream of the plugin hears the word, so it is let go,
/// as it is under a call that stands. Kept, it would wait for the next
/// settle with the events behind it.
#[tokio::test(start_paused = true)]
async fn a_left_word_taken_as_the_calls_receiver_goes_is_let_go_under_a_stream_on_1() {
    use crate::{engine::view::WorkerId, tracking::lease::Membership};
    let models = ["m".to_owned()];
    let left = Membership::Left {
        id: WorkerId(7),
        models: &models,
    };
    let (raise, mut queue) = mpsc::channel(4);
    let (tx, rx) = mpsc::channel(OBSERVE_WINDOW);
    let call = async {
        tokio::time::sleep(Duration::from_millis(2)).await;
        Err::<(), _>(tonic::Status::unavailable("the plugin went"))
    };
    let goes = async {
        tokio::time::sleep(Duration::from_millis(1)).await;
        drop(rx);
        raise.try_send(word(&left)).expect("the queue takes a word");
        std::future::pending::<()>().await;
    };
    let (spoken, mut kept) = (standing_on(1), None);
    let (mut sent, mut lost) = (0, 0);
    let requests = Requests {
        tx,
        version: 1,
        spoken: &spoken,
        kept: &mut kept,
    };
    let (queue, frame, never) = (&mut queue, RECONNECT_PAUSE, std::future::pending());
    let turn = super::forward(call, requests, queue, &mut sent, &mut lost, frame, never);
    let turn = tokio::select! {
        turn = turn => turn,
        () = goes => unreachable!("the receiver's end never ends"),
    };
    assert!(matches!(turn, Turn::Ended(_)), "the call ended");
    assert!(kept.is_none(), "the word is let go: a stream stands on 1");
    assert_eq!(sent, 0, "and took no number");
}

/// A dropped plugin's call ends though its window is full. Here the
/// call's peer reads nothing: the window fills and one more event waits
/// for the stream's room. The plugin is then dropped. The wait for room
/// is cut each frame and reads the drop there, so the task ends within
/// two frames of it.
#[tokio::test(start_paused = true)]
async fn a_dropped_plugins_call_ends_though_its_window_is_full() {
    let (raise, mut queue) = mpsc::channel(OBSERVE_WINDOW + 1);
    for _ in 0..=OBSERVE_WINDOW {
        raise.try_send(raised()).expect("the queue takes an event");
    }
    let (tx, rx) = mpsc::channel(OBSERVE_WINDOW);
    // The call's peer holds its receiver and reads nothing.
    let call = async move {
        let _held = rx;
        std::future::pending::<Result<(), tonic::Status>>().await
    };
    let (mut sent, mut lost) = (0, 0);
    let ended = {
        let turn = forward(call, tx, &mut queue, &mut sent, &mut lost, RECONNECT_PAUSE);
        tokio::pin!(turn);
        let waits = tokio::time::timeout(Duration::from_secs(1), &mut turn).await;
        assert!(
            waits.is_err(),
            "the window is full: the turn waits for room"
        );
        drop(raise);
        tokio::time::timeout(2 * RECONNECT_PAUSE, &mut turn).await
    };
    assert!(
        matches!(ended, Ok(None)),
        "the dropped plugin's call ends within two frames"
    );
    assert_eq!(
        u64::try_from(OBSERVE_WINDOW).ok(),
        Some(sent),
        "the window's events were written"
    );
}

/// A word the call's version has no kind for is judged though the
/// call's window is full. Here the call that named 1 is written a full
/// window and reads none of it, no stream stands, and a left word is
/// raised behind the window: the word is kept and the call yields to
/// the next one. Behind a wait for room the word would stay unread, and
/// the events behind it with it.
#[tokio::test(start_paused = true)]
async fn a_left_word_is_judged_though_its_calls_window_is_full() {
    use crate::{engine::view::WorkerId, tracking::lease::Membership};
    let models = ["m".to_owned()];
    let left = Membership::Left {
        id: WorkerId(7),
        models: &models,
    };
    let (raise, mut queue) = mpsc::channel(OBSERVE_WINDOW + 1);
    for _ in 0..OBSERVE_WINDOW {
        raise.try_send(raised()).expect("the queue takes an event");
    }
    raise
        .try_send(word(&left))
        .expect("the queue takes the word");
    let (tx, rx) = mpsc::channel(OBSERVE_WINDOW);
    // The call's peer holds its receiver and reads nothing.
    let call = async move {
        let _held = rx;
        std::future::pending::<Result<(), tonic::Status>>().await
    };
    // A stream settled on 1 and ended: the version holds, none stands.
    let spoken = Spoken::default();
    drop(spoken.settle(1).expect("the settle"));
    let (mut kept, mut sent, mut lost) = (None, 0, 0);
    let requests = Requests {
        tx,
        version: 1,
        spoken: &spoken,
        kept: &mut kept,
    };
    let (queue, frame, never) = (&mut queue, RECONNECT_PAUSE, std::future::pending());
    let turn = super::forward(call, requests, queue, &mut sent, &mut lost, frame, never);
    let turn = tokio::time::timeout(Duration::from_secs(60), turn).await;
    assert!(
        matches!(turn, Ok(Turn::Moved { .. })),
        "the call yields within the minute"
    );
    assert!(kept.is_some(), "the word is kept for the next call");
    assert_eq!(
        u64::try_from(OBSERVE_WINDOW).ok(),
        Some(sent),
        "the events ahead of the word were written"
    );
}

/// An event that waited for its call's room is written in that call
/// once the peer reads. Here the window fills, one more event waits, and
/// the peer then reads every event and ends the call: the event that
/// waited took the next number.
#[tokio::test(start_paused = true)]
async fn an_event_that_waited_for_room_is_written_once_its_call_reads() {
    let (raise, mut queue) = mpsc::channel(OBSERVE_WINDOW + 1);
    for _ in 0..=OBSERVE_WINDOW {
        raise.try_send(raised()).expect("the queue takes an event");
    }
    let (tx, mut rx) = mpsc::channel(OBSERVE_WINDOW);
    // The call's peer reads nothing for a second, then every event.
    let call = async move {
        tokio::time::sleep(Duration::from_secs(1)).await;
        for _ in 0..=OBSERVE_WINDOW {
            if rx.recv().await.is_none() {
                return Err(tonic::Status::aborted("the stream ended early"));
            }
        }
        Ok(())
    };
    let (mut sent, mut lost) = (0, 0);
    let turn = forward(call, tx, &mut queue, &mut sent, &mut lost, RECONNECT_PAUSE);
    let ended = tokio::time::timeout(Duration::from_secs(60), turn).await;
    assert!(matches!(ended, Ok(Some(_))), "the call read and ended");
    assert_eq!(
        u64::try_from(OBSERVE_WINDOW + 1).ok(),
        Some(sent),
        "the event that waited was written, numbered"
    );
}

/// A call that lets its receiver go and says no outcome within the frame
/// is said as that, at the frame and no later, and dated where the send
/// saw it.
#[tokio::test(start_paused = true)]
async fn a_call_gone_with_no_outcome_is_said_at_the_frame() {
    let (raise, mut queue) = mpsc::channel(4);
    raise.try_send(raised()).expect("the queue takes an event");
    let (tx, rx) = mpsc::channel(OBSERVE_WINDOW);
    drop(rx);
    let call = std::future::pending::<Result<(), tonic::Status>>();
    let (mut sent, mut lost) = (0, 0);
    let before = tokio::time::Instant::now();
    let ended = forward(call, tx, &mut queue, &mut sent, &mut lost, RECONNECT_PAUSE)
        .await
        .expect("the plugin is loaded");
    assert_eq!(ended.why, "the observe stream took no more events");
    assert_eq!(ended.at, before, "the end is dated where the send saw it");
    let waited = before.elapsed();
    assert!(
        (RECONNECT_PAUSE..RECONNECT_PAUSE * 2).contains(&waited),
        "the wait for the outcome was {waited:?}, not one frame"
    );
}

/// A count carried late moves no number back. Two threads raise at a
/// full queue, and the one that read the count first enters the queue
/// last: the counts handed here are the ones the events carried.
#[test]
fn a_count_carried_late_moves_no_number_back() {
    let (mut sent, mut lost) = (0_u64, 0_u64);
    assert_eq!(number(&mut sent, &mut lost, 0), 1);
    // Two refusals fell before the second event was raised.
    assert_eq!(number(&mut sent, &mut lost, 2), 4);
    // The third was raised before them and queued after the second.
    assert_eq!(number(&mut sent, &mut lost, 0), 5);
    // One more refusal: a gap of one, behind the third.
    assert_eq!(number(&mut sent, &mut lost, 3), 7);
}

/// A word the call's version has no kind for is not written and takes no
/// number, and the refusals it carried still count. Here a stream on
/// protocol 1 is handed a left word raised behind two refusals, then a
/// relabelling raised ahead of them: the call hears the relabelling
/// alone, numbered behind both.
#[tokio::test(start_paused = true)]
async fn a_word_the_calls_version_has_no_kind_for_takes_no_number_and_keeps_its_refusals() {
    use crate::{engine::view::WorkerId, plugins::grpc::view, tracking::lease::Membership};
    let word = |change: Membership<'_>, refusals| proto::ObserveRequest {
        request: Some(proto::observe_request::Request::Event(
            proto::LifecycleEvent {
                seq: refusals,
                event: Some(view::membership(&change)),
            },
        )),
    };
    let models = ["m".to_owned()];
    let left = Membership::Left {
        id: WorkerId(7),
        models: &models,
    };
    let (raise, mut queue) = mpsc::channel(4);
    raise.try_send(word(left, 2)).expect("the queue takes it");
    raise
        .try_send(word(Membership::Relabelled(WorkerId(7)), 0))
        .expect("the queue takes it");
    let (tx, mut rx) = mpsc::channel::<proto::ObserveRequest>(OBSERVE_WINDOW);
    let heard = RefCell::new(Vec::new());
    let told = &heard;
    let call = async move {
        let event = rx.recv().await.expect("the call takes an event");
        let Some(proto::observe_request::Request::Event(event)) = event.request else {
            panic!("an event, not a hello");
        };
        let Some(proto::lifecycle_event::Event::Membership(change)) = event.event else {
            panic!("a membership word");
        };
        told.borrow_mut().push((event.seq, change.kind));
        Err::<(), _>(tonic::Status::unavailable("the plugin went"))
    };
    let (mut sent, mut lost) = (0, 0);
    let ended = forward(call, tx, &mut queue, &mut sent, &mut lost, RECONNECT_PAUSE)
        .await
        .expect("the plugin is loaded");
    assert!(ended.why.contains("the plugin went"), "{}", ended.why);
    let relabelled = proto::membership_change::Kind::Relabelled as i32;
    assert_eq!(heard.into_inner(), [(3, relabelled)]);
    assert_eq!((sent, lost), (1, 2), "the left word took no number");
}

/// An outage is the failures with no call that stood between them, and
/// says one line. Here a call fails at once, a connect is refused and a
/// call ends a millisecond short of a stand: one outage. A call that
/// stood then ends and says the line again, and a failure after it is
/// quiet.
#[tokio::test(start_paused = true)]
async fn an_outage_says_one_line_until_a_call_stood() {
    let (events, lane, raise) = told(0);
    let quick = Duration::from_millis(5);
    let steps = vec![
        Step::Ends(quick),
        Step::Refused,
        Step::Ends(STAND - Duration::from_millis(1)),
        Step::Ends(STAND),
        Step::Ends(quick),
    ];
    let ran = scripted(&lane, &raise, steps).await;
    let opens = ran.opens.borrow();
    let told = events.told.lock();
    assert_eq!(told.len(), 2, "{told:?}");
    // A refused open is asked for again one frame later, never at once.
    assert_eq!(opens[2].duration_since(opens[1]), lane.peer.connect);
    assert_eq!(told[0].0.duration_since(opens[0]), quick);
    assert_eq!(told[1].0.duration_since(opens[3]), STAND);
}

/// A telling that panics is told once an outage: the lines' state is
/// held outside the turn and moves before the telling. Here the first
/// telling panics, and the refusals of the turns after it are the same
/// outage: told again only where a call that stood has ended.
#[tokio::test(start_paused = true)]
async fn a_telling_that_panics_is_told_once_an_outage() {
    // A subscriber of the case's own counts this thread's error lines.
    // A second one stays registered beside it: with one alone a line is
    // on or off by the thread that reaches it first, and a case beside
    // this one reaches it under none.
    let errors = Errors::default();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(errors.clone());
    let (events, lane, raise) = told(1);
    let steps = vec![
        Step::Refused,
        Step::Refused,
        Step::Refused,
        Step::Ends(STAND),
    ];
    let ran = scripted(&lane, &raise, steps).await;
    assert_eq!(ran.opens.borrow().len(), 5, "every step was asked for");
    assert_eq!(events.told.lock().len(), 2, "an outage, then a stand's end");
    assert_eq!(errors.0.load(Ordering::Relaxed), 1, "one turn panicked");
}

/// A turn that panicked is started again a pause later, never at once
/// (the I/O thread carries every stream and would spin), and a streak
/// of such turns says one line: the turns with no call that stood
/// between them. Here the opener panics three times, a call stands and
/// ends, and the opener panics twice more: two streaks, two lines.
#[tokio::test(start_paused = true)]
async fn a_streak_of_panicked_turns_says_one_line_until_a_call_stood() {
    // A subscriber of the case's own counts this thread's error lines.
    // A second one stays registered beside it: with one alone a line is
    // on or off by the thread that reaches it first, and a case beside
    // this one reaches it under none.
    let errors = Errors::default();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(errors.clone());
    let (_events, lane, raise) = told(0);
    let steps = vec![
        Step::Panics,
        Step::Panics,
        Step::Panics,
        Step::Ends(STAND),
        Step::Panics,
        Step::Panics,
    ];
    let ran = scripted(&lane, &raise, steps).await;
    let opens = ran.opens.borrow();
    assert_eq!(opens.len(), 7, "every step was asked for");
    for pair in opens.windows(2).take(3) {
        assert_eq!(pair[1].duration_since(pair[0]), RECONNECT_PAUSE);
    }
    assert_eq!(errors.0.load(Ordering::Relaxed), 2, "one line a streak");
}

/// An `Observe` task that panics is not the end of the plugin's events:
/// the queue and the numbers are held outside the turn. Here an event
/// is raised and the opener panics; the call of the next turn is sent
/// that event, with the first number.
#[tokio::test(start_paused = true)]
async fn an_observe_task_that_panics_is_started_again() {
    let (_events, lane, raise) = told(0);
    raise.try_send(raised()).expect("the queue takes an event");
    let steps = vec![Step::Panics, Step::Ends(Duration::from_millis(5))];
    let ran = scripted(&lane, &raise, steps).await;
    assert_eq!(
        *ran.taken.borrow(),
        [1],
        "the event raised before the panic"
    );
}

/// The pause before the next call runs from the instant the end was
/// seen: a turn's wait for the call's outcome is inside it. Here a call
/// lets its receiver go and says no outcome, and the next call opens
/// one frame after the turn saw the end, not two.
#[tokio::test(start_paused = true)]
async fn the_pause_runs_from_the_instant_an_end_was_seen() {
    let (_events, lane, raise) = told(0);
    let quick = Duration::from_millis(5);
    let steps = vec![Step::Goes(quick), Step::Ends(quick)];
    let ran = scripted(&lane, &raise, steps).await;
    let opens = ran.opens.borrow();
    assert_eq!(opens[1].duration_since(opens[0]), quick + lane.peer.connect);
}

/// A call stood when its end was seen a stand or longer after its open,
/// whichever side saw the end: the call's outcome, or a send that found
/// the receiver gone. A millisecond short of a stand is no stand.
#[tokio::test(start_paused = true)]
async fn a_call_stood_when_its_end_was_seen_a_stand_after_its_open() {
    for (age, stood) in [(STAND - Duration::from_millis(1), false), (STAND, true)] {
        let (mut sent, mut lost) = (0, 0);
        let (_raise, mut queue) = mpsc::channel(4);
        let (tx, _rx) = mpsc::channel(OBSERVE_WINDOW);
        let call = async move {
            tokio::time::sleep(age).await;
            Err::<(), _>(tonic::Status::unavailable("the plugin went"))
        };
        let ended = forward(call, tx, &mut queue, &mut sent, &mut lost, RECONNECT_PAUSE)
            .await
            .expect("the plugin is loaded");
        assert_eq!(ended.stood, stood, "an outcome {age:?} after the open");

        let (raise, mut queue) = mpsc::channel(4);
        let (tx, rx) = mpsc::channel(OBSERVE_WINDOW);
        let call = async move {
            tokio::time::sleep(age).await;
            drop(rx);
            raise.try_send(raised()).expect("the queue takes an event");
            std::future::pending::<Result<(), tonic::Status>>().await
        };
        let ended = forward(call, tx, &mut queue, &mut sent, &mut lost, RECONNECT_PAUSE)
            .await
            .expect("the plugin is loaded");
        assert_eq!(ended.stood, stood, "a send {age:?} after the open");
    }
}

/// An `Observe` call's status is the plugin's words: the end carries
/// their start and no more than the bound.
#[tokio::test(start_paused = true)]
async fn an_observe_calls_status_is_cut_at_the_bound() {
    let (_raise, mut queue) = mpsc::channel(4);
    let (tx, _rx) = mpsc::channel(OBSERVE_WINDOW);
    let call = async { Err::<(), _>(tonic::Status::unavailable("x".repeat(10_000))) };
    let (mut sent, mut lost) = (0, 0);
    let ended = forward(call, tx, &mut queue, &mut sent, &mut lost, RECONNECT_PAUSE)
        .await
        .expect("the plugin is loaded");
    let bytes = ended.why.len();
    assert!(bytes <= PEER_TEXT + "...".len(), "{bytes} bytes");
    assert!(ended.why.ends_with("x..."), "{}", ended.why);
}

/// A stream fault made of a text carries the plugin's words (a status
/// at a stream's open): their start and no more than the bound.
#[test]
fn a_stream_faults_text_is_cut_at_the_bound() {
    let said = Fault::from("\u{20ac}".repeat(1_000)).to_string();
    assert!(
        said.len() <= PEER_TEXT + "...".len(),
        "{} bytes",
        said.len()
    );
    assert!(said.starts_with('\u{20ac}'), "{said}");
    assert!(said.ends_with("\u{20ac}..."), "{said}");
}

/// A plugin that cannot be reached says its failure once, not once a
/// connect: the failed connects of one outage are one line. The loop's
/// clock is paused here and moves by its sleeps alone, and the listener
/// records every telling and panics at none.
#[tokio::test(start_paused = true)]
async fn the_failed_connects_of_one_outage_say_one_line() {
    let (commands, taken) = mpsc::unbounded_channel();
    let mut looping = JoinSet::new();
    looping.spawn(run_commands(taken, Arc::new(Witness::new())));
    let (events, lane, _raise) = told(0);
    assert!(commands.send(Command::Observe(Arc::new(lane))).is_ok());
    // The peer's connect frame is one second: five connects fail here.
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(events.told.lock().len(), 1, "one line an outage");
}

/// An answer that lands in its place as the wait runs out is read: the
/// wait's last look is at the place, after the deadline. Here the answer
/// is written with no wake, so the waiting thread finds it by that look
/// alone.
#[test]
fn an_answer_that_lands_as_the_wait_runs_out_is_read() {
    let events = Arc::new(Counted::default());
    let (lane, _rx) = open_lane(&events, 16);
    let id = lane.next_id();
    lane.send(id, proto::PluginDecide::default()).expect("sent");
    let until = Instant::now() + Duration::from_millis(200);
    let waited = std::thread::scope(|scope| {
        let waiting = scope.spawn(|| lane.wait_answer(id, until));
        std::thread::sleep(Duration::from_millis(20));
        if let Some(place) = lane.pending.lock().place(id) {
            place.answer = Some((lane.witness.stamp(), answer(id)));
        }
        waiting.join()
    });
    assert_eq!(waited.ok().and_then(answered), Some(id));
}
