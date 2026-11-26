//! The I/O side of the gRPC plugin: one thread per process that owns every
//! stream, and the lanes through which a deciding thread hands it a frame
//! and waits for the answer. A deciding thread never touches a socket; the
//! I/O thread never waits on a decision.

use std::{
    any::Any,
    borrow::Cow,
    error::Error,
    fmt,
    future::Future,
    io,
    ops::RangeInclusive,
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::sync_channel,
        Arc, Weak,
    },
    task::{Context, Poll},
    thread,
    time::{Duration, Instant},
};

use hyper_util::rt::TokioIo;
use parking_lot::{Condvar, Mutex};
use rustls::crypto::ring;
#[cfg(feature = "test-util")]
use tokio::sync::oneshot;
use tokio::{
    net::UnixStream,
    sync::{mpsc, watch, Notify},
    task::JoinSet,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity, Uri};
use tower::service_fn;

use super::{config::Transport, view};
use crate::{
    plugins::shared::OpenDecisions,
    service::proto::{self, policy_plugin_client::PolicyPluginClient},
};

/// Lifecycle events one `Observe` call holds unsent at most; the plugin's
/// own queue is in front of it.
const OBSERVE_WINDOW: usize = 64;

/// Frames one `Decide` stream holds unanswered at most, the hello's
/// `max_in_flight`: a decision opens one frame per phase per bound set of
/// the plugin and a pair plan opens both sides before either is read, so
/// this is two bound sets on each of the four phases. A frame past it is
/// not sent.
pub(crate) const WINDOW: usize = 8;

/// How often the I/O thread stamps its witness while a frame is in flight.
const WITNESS_TICK: Duration = Duration::from_millis(1);

/// The I/O thread's promise, in microseconds: while a frame is in flight it
/// runs at least this often. A longer gap is a stall, the host's.
pub(crate) const PROMISE_MICROS: u64 = 4_000;

/// How long a lane that failed leaves its plugin alone before a decision
/// asks for the next stream, whatever the breaker is set to: a socket that
/// refuses at once is dialled once a pause, never once a decision.
pub(crate) const RECONNECT_PAUSE: Duration = Duration::from_millis(100);

/// How long an `Observe` call stands before its end shows the plugin was
/// up: the end of such a call starts an outage and a streak anew, so each
/// says its line again. It bounds the lines' rate, one a stand and a pause
/// at worst (a plugin that holds each call just past it); two outages with
/// less than a stand of standing between them are one line.
const STAND: Duration = Duration::from_secs(60);

/// The most bytes of the plugin's own words an error or a log line carries:
/// an ack's reason and a status's message are the peer's, of any length
/// the transport lets through.
const PEER_TEXT: usize = 256;

/// `text` cut to [`PEER_TEXT`] bytes at a character's start; a cut text
/// ends in an ellipsis.
fn bounded(mut text: String) -> String {
    if text.len() > PEER_TEXT {
        let mut end = PEER_TEXT;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str("...");
    }
    text
}

/// The most bytes one message of the plugin's may hold, judged before it
/// is decoded: a fixed part (the ids, the pick, a decline with its detail,
/// an ack with its reason) and what the row lets an answer carry, a
/// surviving position and a score a candidate and an increment a declared
/// counter. A longer message ends the stream.
pub(crate) fn answer_bytes(max_candidates: u32, counters: usize) -> usize {
    const FIXED: usize = 64 * 1024;
    const A_CANDIDATE: usize = 16;
    const A_COUNTER: usize = 32;
    let candidates = usize::try_from(max_candidates).unwrap_or(usize::MAX);
    FIXED
        .saturating_add(A_CANDIDATE.saturating_mul(candidates))
        .saturating_add(A_COUNTER.saturating_mul(counters))
}

/// The I/O thread's witness: when it last ran, and the last gap between two
/// of its runs that was longer than its promise. A remote plugin is judged
/// by what reached the host, and the I/O thread is the host's part of that
/// road: a frame whose answer was owed while the thread did not run is the
/// host's, counted as a stall and left out of the plugin's latency sample.
/// The thread stamps the witness once a tick while any frame is in flight
/// and at every answer. With none in flight nothing ticks: the first frame
/// to take flight stamps it from its own thread and wakes the ticker.
#[derive(Debug)]
pub(crate) struct Witness {
    /// The zero of the instants below.
    origin: Instant,
    /// When the I/O thread last ran, in microseconds since `origin`.
    ran: AtomicU64,
    /// The last gap past the promise, its two ends in the same unit.
    stall: Mutex<(u64, u64)>,
    /// The frames in flight on every lane of the runtime.
    in_flight: AtomicUsize,
    /// Wakes the ticker at the first frame in flight.
    wake: Notify,
}

impl Witness {
    fn new() -> Self {
        Self {
            origin: Instant::now(),
            ran: AtomicU64::new(0),
            stall: Mutex::new((0, 0)),
            in_flight: AtomicUsize::new(0),
            wake: Notify::new(),
        }
    }

    fn micros(&self, instant: Instant) -> u64 {
        let since = instant.saturating_duration_since(self.origin);
        u64::try_from(since.as_micros()).unwrap_or(u64::MAX)
    }

    /// The I/O thread runs, now; the instant it read.
    fn stamp(&self) -> Instant {
        let now = Instant::now();
        self.stamp_at(now);
        now
    }

    /// The I/O thread ran at `now`: a gap past the promise since its last
    /// run is the last stall.
    fn stamp_at(&self, now: Instant) {
        let now = self.micros(now);
        let before = self.ran.swap(now, Ordering::Relaxed);
        if now.saturating_sub(before) > PROMISE_MICROS {
            *self.stall.lock() = (before, now);
        }
    }

    /// A frame takes flight at `at`. Nothing ticked while none was in
    /// flight, so the first one stamps the witness itself and wakes the
    /// ticker: the promise runs from here.
    fn took_flight(&self, at: Instant) {
        if self.in_flight.fetch_add(1, Ordering::Relaxed) == 0 {
            self.ran.store(self.micros(at), Ordering::Relaxed);
            self.wake.notify_one();
        }
    }

    /// A frame in flight landed: answered, missed, broken or forgotten.
    fn landed(&self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    /// Whether the I/O thread stalled while an answer was owed from `from`
    /// to `until`: it has not run for longer than its promise at `until`,
    /// or the last gap it closed crosses the two. The gap is exclusive at
    /// both ends. Its first is the microsecond the thread last ran in: an
    /// answer owed only until then met a running thread. Its second is the
    /// microsecond the thread came back in: a frame that took flight then
    /// waited on it for less than the witness's unit, and a later miss of
    /// it is not the thread's. A frame whose own budget is no longer than
    /// the promise is excused only where another frame's flight already
    /// shows the thread stalled: alone it misses before the thread can be
    /// seen late, and its miss is nobody's to say ([`Self::vouches_for`]).
    pub(crate) fn stalled_between(&self, from: Instant, until: Instant) -> bool {
        let (from, until) = (self.micros(from), self.micros(until));
        if until.saturating_sub(self.ran.load(Ordering::Relaxed)) > PROMISE_MICROS {
            return true;
        }
        let (start, end) = *self.stall.lock();
        start < end && start < until && from < end
    }

    /// Whether the witness can vouch for a flight of `flight_micros` that
    /// no stall crossed. Longer than the promise, the stamp that stands at
    /// its end is the I/O thread's own and was made inside the flight: the
    /// thread was seen running while the answer was owed. A flight no
    /// longer than the promise ends before the thread can be seen late, and
    /// the stamp at its end may be the sender's own: nothing says whether
    /// the thread ran.
    pub(crate) const fn vouches_for(flight_micros: u64) -> bool {
        flight_micros > PROMISE_MICROS
    }
}

/// Stamps the witness once a tick while any frame is in flight; with none
/// it waits for the first.
async fn witness_ticks(witness: Arc<Witness>) {
    loop {
        while witness.in_flight.load(Ordering::Relaxed) == 0 {
            witness.wake.notified().await;
        }
        witness.stamp();
        tokio::time::sleep(WITNESS_TICK).await;
    }
}

/// What the I/O thread needs of a plugin to open its streams: the row's
/// transport and the hello's content.
pub(crate) struct Peer {
    pub(crate) name: String,
    pub(crate) transport: Transport,
    pub(crate) config: Vec<u8>,
    pub(crate) exposure: proto::Exposure,
    pub(crate) counters: Vec<String>,
    /// The connect and the hello run under this.
    pub(crate) connect: Duration,
    /// The most bytes one message of the plugin's holds ([`answer_bytes`]).
    pub(crate) answer_bytes: usize,
    /// The protocol versions a `Decide` hello offers while the plugin's
    /// version is not fixed.
    pub(crate) versions: RangeInclusive<u32>,
    /// The version the plugin speaks, as its `Decide` streams fix it.
    pub(crate) spoken: Spoken,
}

impl Peer {
    /// The hello of a stream that offers the versions from `protocol_min`
    /// to `protocol_max`.
    fn hello(&self, (protocol_min, protocol_max): (u32, u32)) -> proto::PluginHello {
        proto::PluginHello {
            protocol_min,
            protocol_max,
            plugin: self.name.clone(),
            config: self.config.clone(),
            exposure: Some(self.exposure.clone()),
            counters: self.counters.clone(),
            // A stream serves every profile that binds the plugin: any of
            // the three stages may be asked on it.
            stages: view::FILTER | view::SCORE | view::PICK,
            max_in_flight: u32::try_from(WINDOW).unwrap_or(u32::MAX),
            // The spec's generation arrives with the `plugins` row.
            generation: 0,
        }
    }
}

/// The protocol version one plugin speaks, by the contract's rule: the first
/// `Decide` stream to settle fixes it, a stream opened while one stands is
/// offered it alone, and the end of the last one frees it for the next
/// settle.
pub(crate) struct Spoken {
    stand: watch::Sender<Stand>,
}

/// A plugin's version as its `Decide` streams leave it.
#[derive(Clone, Copy, Default)]
struct Stand {
    /// The version last fixed, zero before the first settle.
    fixed: u32,
    /// How many settled streams stand on it: with none the version is free.
    standing: usize,
    /// The settles that fixed a version, the last one with them. A stream
    /// that settles where none stands and ends between two looks of the
    /// events task shows in this count alone.
    settles: u64,
}

impl Default for Spoken {
    fn default() -> Self {
        Self {
            stand: watch::Sender::new(Stand::default()),
        }
    }
}

impl Spoken {
    /// The version and its streams as they stand, in one reading.
    fn stand(&self) -> Stand {
        *self.stand.borrow()
    }

    /// The versions a `Decide` hello offers: the fixed one alone while a
    /// stream stands on it, the host's range otherwise.
    fn offer(&self, versions: &RangeInclusive<u32>) -> (u32, u32) {
        match self.stand() {
            Stand { standing: 0, .. } => (*versions.start(), *versions.end()),
            Stand { fixed, .. } => (fixed, fixed),
        }
    }

    /// A `Decide` stream's ack named `version`: it fixes the plugin's where
    /// no stream stands and joins it where it names the fixed one, and the
    /// stream stands until its [`Standing`] goes. An ack that names another
    /// version comes back with the fixed one: two streams were offered the
    /// range, and this one settled second.
    fn settle(&self, version: u32) -> Result<Standing<'_>, u32> {
        let mut other = None;
        self.stand.send_if_modified(|stand| {
            if stand.standing == 0 {
                // The events task waits for the version to move, and with
                // a word kept for a settle. The count shows this one though
                // its stream ends before the task looks.
                stand.fixed = version;
                stand.standing = 1;
                stand.settles += 1;
                return true;
            }
            if stand.fixed == version {
                stand.standing += 1;
            } else {
                other = Some(stand.fixed);
            }
            false
        });
        match other {
            None => Ok(Standing(self)),
            Some(fixed) => Err(fixed),
        }
    }

    /// Whether a stream stands on a version older than `since`. An event
    /// that version first carries is then written to no stream of the
    /// plugin, so it is not raised: in the queue it would take a place a
    /// heard event needs, and a refusal of it would number a loss the
    /// stream never had. With no stream standing the next settle may fix
    /// any version: the event is raised, and the events task keeps it for
    /// the call that carries it ([`forward`]).
    pub(crate) fn stands_under(&self, since: u32) -> bool {
        let stand = self.stand();
        stand.standing != 0 && stand.fixed < since
    }
}

/// One settled `Decide` stream's stand on its plugin's version, given up
/// when the stream ends.
struct Standing<'a>(&'a Spoken);

impl Drop for Standing<'_> {
    fn drop(&mut self) {
        // No waiter reads a stream's end: the version stays the last one
        // fixed, and the next settle says what it is.
        self.0.stand.send_if_modified(|stand| {
            stand.standing = stand.standing.saturating_sub(1);
            false
        });
    }
}

/// What the plugin learns from its streams, on the I/O thread: counted and
/// judged by the plugin, never by the lane.
pub(crate) trait LaneEvents: Send + Sync {
    /// A `Decide` stream settled (the hello acked).
    fn connected(&self);
    /// A stream failed to open, ended or erred, at the instant its lane
    /// dated the failure by: the one its pause runs from.
    fn stream_failed(&self, why: &str, at: Instant);
    /// An answer for a decision nobody waits for any more.
    fn late_answer(&self);
    /// The `Observe` stream failed to open or ended: told once an outage.
    fn observe_failed(&self, why: &str);
}

/// Why a stream is not open.
#[derive(Clone, Debug)]
pub(crate) enum Fault {
    /// The plugin's `init` failed: the ack's reason.
    Init(String),
    /// The connect, the hello or the stream itself failed.
    Stream(String),
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Init(why) => write!(f, "init failed: {why}"),
            Self::Stream(why) => f.write_str(why),
        }
    }
}

impl From<&str> for Fault {
    fn from(why: &str) -> Self {
        Self::Stream(why.to_owned())
    }
}

impl From<String> for Fault {
    fn from(why: String) -> Self {
        Self::Stream(bounded(why))
    }
}

/// Where a lane stands.
enum LaneState {
    /// No stream asked for yet.
    Cold,
    /// The connect and the hello run on the I/O thread.
    Connecting,
    /// The stream is open and the hello acked.
    Ready {
        tx: mpsc::Sender<proto::PluginRequest>,
        /// The stage bits the plugin implements.
        stages: u32,
    },
    /// The stream failed, at this instant; a decision at least
    /// [`RECONNECT_PAUSE`] later asks for a new one.
    Failed(Fault, Instant),
}

/// One frame in flight: what the I/O thread fills in.
#[derive(Default)]
struct Pending {
    /// The decision id an answer is wanted for; `0` when the place is free.
    id: u64,
    /// The answer and when it arrived.
    answer: Option<(Instant, proto::PluginDecision)>,
    /// The stream went while the frame waited.
    broken: bool,
}

/// The frames of a lane in flight, by decision id (never `0`).
#[derive(Default)]
struct Table([Pending; WINDOW]);

impl Table {
    fn place(&mut self, id: u64) -> Option<&mut Pending> {
        self.0.iter_mut().find(|pending| pending.id == id)
    }

    /// Takes a free place for `id`; `false` when the window is full.
    fn arm(&mut self, id: u64) -> bool {
        let Some(free) = self.place(0) else {
            return false;
        };
        free.id = id;
        true
    }

    /// Frees the place of `id` with whatever it holds; `false` when no
    /// place held it.
    fn free(&mut self, id: u64) -> bool {
        if id == 0 {
            return false;
        }
        let Some(pending) = self.place(id) else {
            return false;
        };
        *pending = Pending::default();
        true
    }
}

/// One plugin's stream for one deciding thread: the thread sends frames and
/// waits on their places in the table; the I/O thread connects, reads and
/// fills them. The lane holds the plugin weakly: a dropped plugin's streams
/// end.
pub(crate) struct Lane {
    peer: Arc<Peer>,
    events: Weak<dyn LaneEvents>,
    state: Mutex<LaneState>,
    ready: Condvar,
    pending: Mutex<Table>,
    answered: Condvar,
    /// The I/O thread's witness: a frame in flight is counted there.
    witness: Arc<Witness>,
    /// The stream's decision ids: its one deciding thread numbers them.
    ids: AtomicU64,
    /// The plugin was dropped: the stream's task ends.
    closed: Notify,
}

impl fmt::Debug for Lane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lane")
            .field("plugin", &self.peer.name)
            .finish_non_exhaustive()
    }
}

/// Whether a lane is open, at the end of a wait for it.
pub(crate) enum Readiness {
    Ready,
    /// Still connecting at the deadline.
    NotYet,
    /// The stream failed.
    Failed,
}

/// Why a frame was not sent.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SendError {
    /// The stream is not open.
    NotReady,
    /// The stream holds its window of frames in flight: the plugin is not
    /// answering or not reading the ones before.
    Full,
}

/// How a wait for an answer ended.
pub(crate) enum Wait {
    /// The answer and when the I/O thread received it.
    Answered(Instant, proto::PluginDecision),
    /// The deadline passed first.
    Missed,
    /// The stream went while the decision waited.
    Broken,
}

impl Lane {
    pub(crate) fn new(
        peer: Arc<Peer>,
        events: Weak<dyn LaneEvents>,
        witness: Arc<Witness>,
    ) -> Self {
        Self {
            peer,
            events,
            state: Mutex::new(LaneState::Cold),
            ready: Condvar::new(),
            pending: Mutex::new(Table::default()),
            answered: Condvar::new(),
            witness,
            ids: AtomicU64::new(0),
            closed: Notify::new(),
        }
    }

    /// The next decision id of this stream; never `0`.
    pub(crate) fn next_id(&self) -> u64 {
        self.ids.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// A decision's ask: the I/O thread is asked for a stream when the lane
    /// is cold, or failed at least [`RECONNECT_PAUSE`] ago.
    pub(crate) fn ensure_connecting(self: &Arc<Self>, runtime: &Runtime) {
        self.ask(runtime, RECONNECT_PAUSE);
    }

    /// The apply's ask, which waits for the stream: asked for when the lane
    /// is cold or failed, however lately.
    pub(crate) fn connect(self: &Arc<Self>, runtime: &Runtime) {
        self.ask(runtime, Duration::ZERO);
    }

    fn ask(self: &Arc<Self>, runtime: &Runtime, pause: Duration) {
        let mut state = self.state.lock();
        let due = match &*state {
            LaneState::Cold => true,
            LaneState::Failed(_, at) => at.elapsed() >= pause,
            LaneState::Connecting | LaneState::Ready { .. } => false,
        };
        if due {
            *state = LaneState::Connecting;
            drop(state);
            if !runtime.send(Command::Connect(Arc::clone(self))) {
                self.fail("the plugin runtime is gone".into());
            }
        }
    }

    /// The stage bits the plugin implements, when the stream is open.
    pub(crate) fn stages(&self) -> Option<u32> {
        match &*self.state.lock() {
            LaneState::Ready { stages, .. } => Some(*stages),
            _ => None,
        }
    }

    /// Waits until the lane is open, failed, or `deadline` passes.
    pub(crate) fn wait_ready(&self, deadline: Instant) -> Readiness {
        let mut state = self.state.lock();
        loop {
            match &*state {
                LaneState::Ready { .. } => return Readiness::Ready,
                LaneState::Failed(..) | LaneState::Cold => return Readiness::Failed,
                LaneState::Connecting => {
                    if self.ready.wait_until(&mut state, deadline).timed_out() {
                        return match &*state {
                            LaneState::Ready { .. } => Readiness::Ready,
                            LaneState::Connecting => Readiness::NotYet,
                            LaneState::Failed(..) | LaneState::Cold => Readiness::Failed,
                        };
                    }
                }
            }
        }
    }

    /// Why the lane failed, when it did.
    pub(crate) fn failure(&self) -> Option<Fault> {
        match &*self.state.lock() {
            LaneState::Failed(why, _) => Some(why.clone()),
            _ => None,
        }
    }

    /// Sends one frame and arms a place in the table for its answer; the
    /// instant it left this thread.
    pub(crate) fn send(
        &self,
        decision_id: u64,
        frame: proto::PluginDecide,
    ) -> Result<Instant, SendError> {
        let state = self.state.lock();
        let LaneState::Ready { tx, .. } = &*state else {
            return Err(SendError::NotReady);
        };
        let at = Instant::now();
        if !self.pending.lock().arm(decision_id) {
            return Err(SendError::Full);
        }
        self.witness.took_flight(at);
        let request = proto::PluginRequest {
            request: Some(proto::plugin_request::Request::Decide(frame)),
        };
        match tx.try_send(request) {
            Ok(()) => Ok(at),
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.free(&mut self.pending.lock(), decision_id);
                Err(SendError::Full)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.free(&mut self.pending.lock(), decision_id);
                Err(SendError::NotReady)
            }
        }
    }

    /// Waits for the answer of `decision_id` until `deadline`; its place is
    /// free again when this returns, so an answer after a miss is late. An
    /// answer already in its place is returned whenever it arrived: a
    /// thread never throws away an answer it need not wait for, and its
    /// lateness is read from the arrival it comes back with.
    pub(crate) fn wait_answer(&self, decision_id: u64, deadline: Instant) -> Wait {
        let mut table = self.pending.lock();
        loop {
            let Some(pending) = table.place(decision_id) else {
                return Wait::Broken;
            };
            if let Some((arrived, answer)) = pending.answer.take() {
                self.free(&mut table, decision_id);
                return Wait::Answered(arrived, answer);
            }
            if pending.broken {
                self.free(&mut table, decision_id);
                return Wait::Broken;
            }
            if self.answered.wait_until(&mut table, deadline).timed_out() {
                let answer = table.place(decision_id).and_then(|p| p.answer.take());
                self.free(&mut table, decision_id);
                return match answer {
                    Some((arrived, answer)) => Wait::Answered(arrived, answer),
                    None => Wait::Missed,
                };
            }
        }
    }

    /// Forgets a frame in flight at its decision's close; a later answer is
    /// late.
    pub(crate) fn forget(&self, decision_id: u64) {
        self.free(&mut self.pending.lock(), decision_id);
    }

    /// Frees the place of `decision_id`: a frame that was in flight has
    /// landed.
    fn free(&self, table: &mut Table, decision_id: u64) {
        if table.free(decision_id) {
            self.witness.landed();
        }
    }

    /// Ends the stream: the plugin is gone. The sender goes with the state
    /// and the stream's task stops reading.
    pub(crate) fn close(&self) {
        self.fail("the plugin was dropped".into());
        self.closed.notify_one();
    }

    // --- the I/O thread's side ---

    fn events(&self) -> Option<Arc<dyn LaneEvents>> {
        self.events.upgrade()
    }

    fn set_ready(&self, tx: mpsc::Sender<proto::PluginRequest>, stages: u32) {
        *self.state.lock() = LaneState::Ready { tx, stages };
        self.ready.notify_all();
    }

    fn fail(&self, why: Fault) -> Instant {
        let at = Instant::now();
        *self.state.lock() = LaneState::Failed(why, at);
        self.ready.notify_all();
        // Every frame in flight went with the stream.
        for pending in &mut self.pending.lock().0 {
            pending.broken = pending.id != 0;
        }
        self.answered.notify_all();
        at
    }

    /// Fails the lane and tells its plugin, the state first: the telling
    /// reads the host's clock, and whatever that does, the lane reads as
    /// failed and is asked for again. The telling carries the instant the
    /// failure was dated by, the one the lane's pause runs from, so the count
    /// on the breaker reads no instant a held I/O thread moved.
    fn fail_and_tell(&self, why: Fault) {
        let said = why.to_string();
        let at = self.fail(why);
        if let Some(events) = self.events() {
            events.stream_failed(&said, at);
        }
    }

    fn deliver(&self, decision: proto::PluginDecision) {
        // The I/O thread ran: an answer is stamped before it is handed on,
        // so a gap it arrives across is on the witness when it is read.
        let arrived = self.witness.stamp();
        let mut table = self.pending.lock();
        // An id of `0` names no frame: the free places carry it.
        let place = match decision.decision_id {
            0 => None,
            id => table.place(id),
        };
        if let Some(pending) = place {
            pending.answer = Some((arrived, decision));
            drop(table);
            self.answered.notify_all();
        } else {
            drop(table);
            if let Some(events) = self.events() {
                events.late_answer();
            }
        }
    }
}

impl Drop for Lane {
    fn drop(&mut self) {
        // A frame still in its place when the lane goes never lands.
        for pending in &self.pending.get_mut().0 {
            if pending.id != 0 {
                self.witness.landed();
            }
        }
    }
}

/// The plugin's `Observe` side: the events every thread sends into one
/// queue, forwarded by the I/O thread over a call it reopens when it ends.
pub(crate) struct ObserveLane {
    peer: Arc<Peer>,
    events: Weak<dyn LaneEvents>,
    queue: Mutex<Option<mpsc::Receiver<proto::ObserveRequest>>>,
}

impl ObserveLane {
    pub(crate) fn new(
        peer: Arc<Peer>,
        events: Weak<dyn LaneEvents>,
        queue: mpsc::Receiver<proto::ObserveRequest>,
    ) -> Self {
        Self {
            peer,
            events,
            queue: Mutex::new(Some(queue)),
        }
    }

    fn failed(&self, why: &str) {
        if let Some(events) = self.events.upgrade() {
            events.observe_failed(why);
        }
    }
}

pub(crate) enum Command {
    Connect(Arc<Lane>),
    Observe(Arc<ObserveLane>),
    /// A case's aimed hold: the thread says it took the command, then
    /// stays off its work this long.
    #[cfg(feature = "test-util")]
    Hold(Duration, oneshot::Sender<()>),
}

/// The I/O thread every gRPC plugin of a process shares: a current-thread
/// runtime that owns the streams, and the clock the breakers read. It is
/// built over the deciding host's [`OpenDecisions`], as every plugin kind's
/// runtime is: a decision that opens plugins of two kinds is one decision
/// to them all, with one remainder.
pub struct Runtime {
    /// The decision open on each thread and its budget: shared with every
    /// runtime built over the same ones.
    decisions: Arc<OpenDecisions>,
    commands: mpsc::UnboundedSender<Command>,
    clock: Box<dyn Fn() -> u64 + Send + Sync>,
    /// When the I/O thread last ran: read where a frame is judged.
    witness: Arc<Witness>,
    io: Option<thread::JoinHandle<()>>,
}

impl fmt::Debug for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runtime").finish_non_exhaustive()
    }
}

impl Runtime {
    /// The runtime over `decisions`, on the process's monotonic clock.
    pub fn new(decisions: &Arc<OpenDecisions>) -> Result<Arc<Self>, io::Error> {
        let start = Instant::now();
        Self::with_clock(decisions, move || {
            u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
        })
    }

    /// The runtime over `decisions`, on the given clock (microseconds; the
    /// breakers' windows and cool-downs read it).
    pub fn with_clock(
        decisions: &Arc<OpenDecisions>,
        clock: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Result<Arc<Self>, io::Error> {
        let (commands, rx) = mpsc::unbounded_channel();
        let witness = Arc::new(Witness::new());
        // The I/O thread builds its event loop and says how that went before
        // the constructor returns: a host that cannot build one reads the
        // error where it asked. The thread's closure owns no loop, so a
        // thread that cannot start drops nothing that blocks on the caller.
        let (built, building) = sync_channel(1);
        let io = thread::Builder::new()
            .name("smg-policy-grpc".to_owned())
            .spawn({
                let witness = Arc::clone(&witness);
                move || {
                    let io_loop = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(io_loop) => io_loop,
                        Err(error) => {
                            let _ = built.send(Err(error));
                            return;
                        }
                    };
                    let _ = built.send(Ok(()));
                    // Every stream's task is owned by the loop's set and goes
                    // with it.
                    io_loop.block_on(async move { drop(run_commands(rx, witness).await) });
                }
            })?;
        match building.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                // The thread ends once it has said so.
                let _ = io.join();
                return Err(error);
            }
            Err(_) => {
                let _ = io.join();
                return Err(io::Error::other(
                    "the I/O thread ended before its event loop was built",
                ));
            }
        }
        Ok(Arc::new(Self {
            decisions: Arc::clone(decisions),
            commands,
            clock: Box::new(clock),
            witness,
            io: Some(io),
        }))
    }

    /// Microseconds on the runtime's clock.
    pub fn now_micros(&self) -> u64 {
        (self.clock)()
    }

    /// The open decisions the runtime was built over.
    pub(crate) fn decisions(&self) -> &OpenDecisions {
        &self.decisions
    }

    /// The I/O thread's witness.
    pub(crate) fn witness(&self) -> &Arc<Witness> {
        &self.witness
    }

    pub(crate) fn send(&self, command: Command) -> bool {
        self.commands.send(command).is_ok()
    }

    /// Holds the I/O thread off its work for `hold` and returns once the
    /// hold began: a case's stand-in for a host that takes the thread's
    /// core. The package's own tests build it (`test-util`), and the
    /// gateway's `test-util` turns it on for the gateway's tests. A build
    /// of the gateway's package alone leaves it off.
    #[cfg(feature = "test-util")]
    #[doc(hidden)]
    pub fn hold_io(&self, hold: Duration) {
        let (held, began) = oneshot::channel();
        if self.send(Command::Hold(hold, held)) {
            let _ = began.blocking_recv();
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // The sender goes first: the I/O loop ends and its runtime drops
        // every stream. The thread is joined, unless the last plugin went
        // on that thread itself (a stream's task held it for an event).
        let (closed, _) = mpsc::unbounded_channel();
        drop(std::mem::replace(&mut self.commands, closed));
        if let Some(io) = self.io.take() {
            if io.thread().id() != thread::current().id() {
                let _ = io.join();
            }
        }
    }
}

/// The I/O thread's loop: a task for each stream it is asked for, and each
/// ended task taken out of the set, so a stream that ends and reopens for
/// the life of a process leaves no task behind. It ends when the runtime
/// that sends the commands is gone, and hands back the tasks still in its
/// set.
async fn run_commands(
    mut commands: mpsc::UnboundedReceiver<Command>,
    witness: Arc<Witness>,
) -> JoinSet<()> {
    let mut streams = JoinSet::new();
    streams.spawn(witness_ticks(witness));
    loop {
        tokio::select! {
            command = commands.recv() => match command {
                Some(Command::Connect(lane)) => {
                    streams.spawn(run_lane(lane));
                }
                Some(Command::Observe(lane)) => {
                    streams.spawn(run_observe(lane));
                }
                #[cfg(feature = "test-util")]
                Some(Command::Hold(hold, held)) => {
                    let _ = held.send(());
                    thread::sleep(hold);
                }
                None => break,
            },
            Some(ended) = streams.join_next() => {
                // A stream's task ends by returning. One that panicked did
                // so in the host's own code (the clock a build handed the
                // runtime); its lane's state had moved, so the stream is
                // asked for again, and this line says what happened.
                if let Some(error) = ended.err().filter(tokio::task::JoinError::is_panic) {
                    tracing::error!(%error, "a plugin stream's task panicked");
                }
            }
        }
    }
    streams
}

/// A future whose panic is its output: the task that awaits it goes on.
struct Caught<F>(Pin<Box<F>>);

impl<F: Future> Future for Caught<F> {
    type Output = thread::Result<F::Output>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let turn = self.0.as_mut();
        match panic::catch_unwind(AssertUnwindSafe(|| turn.poll(cx))) {
            Ok(Poll::Pending) => Poll::Pending,
            Ok(Poll::Ready(output)) => Poll::Ready(Ok(output)),
            Err(panicked) => Poll::Ready(Err(panicked)),
        }
    }
}

/// The words a panic carried, when it carried words.
fn panic_text(panicked: &(dyn Any + Send)) -> &str {
    panicked
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panicked.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a panic with no words")
}

/// An error with its sources: a transport error names its cause there.
fn chain(error: &dyn Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// Connects to the plugin's socket and checks who answers there: the peer's
/// uid must be this process's own, or the one the row names.
async fn connect_unix(
    path: PathBuf,
    peer_uid: Option<u32>,
) -> Result<TokioIo<UnixStream>, io::Error> {
    let stream = UnixStream::connect(&path).await?;
    let expected = peer_uid.unwrap_or_else(|| rustix::process::geteuid().as_raw());
    let peer = stream.peer_cred()?.uid();
    if peer != expected {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("the plugin's peer runs as uid {peer}, not {expected}"),
        ));
    }
    Ok(TokioIo::new(stream))
}

async fn channel(transport: &Transport, connect: Duration) -> Result<Channel, String> {
    match transport {
        Transport::Unix { path, peer_uid } => {
            let path = path.clone();
            let peer_uid = *peer_uid;
            // The authority is a placeholder: the connector dials the path.
            Endpoint::from_static("http://plugin.invalid")
                .connect_timeout(connect)
                .connect_with_connector(service_fn(move |_: Uri| {
                    connect_unix(path.clone(), peer_uid)
                }))
                .await
                .map_err(|e| chain(&e))
        }
        Transport::Tls { address, tls } => {
            // The process TLS crypto provider, installed once before the
            // first stream unless the host installed its own.
            let _ = ring::default_provider().install_default();
            Endpoint::from_shared(format!("https://{address}"))
                .map_err(|e| chain(&e))?
                .tls_config(
                    ClientTlsConfig::new()
                        .ca_certificate(Certificate::from_pem(&tls.ca_pem))
                        .identity(Identity::from_pem(&tls.cert_pem, &tls.key_pem))
                        .domain_name(tls.server_name.clone()),
                )
                .map_err(|e| chain(&e))?
                .connect_timeout(connect)
                .connect()
                .await
                .map_err(|e| chain(&e))
        }
    }
}

type Inbound = tonic::Streaming<proto::PluginResponse>;

/// A `Decide` stream as it settled: its request side, its answers, the stage
/// bits its ack named and its stand on the version that ack named.
type Settled<'a> = (
    mpsc::Sender<proto::PluginRequest>,
    Inbound,
    u32,
    Standing<'a>,
);

/// Opens a `Decide` stream: the connect, the hello and its ack, each under
/// the connect frame.
async fn open_decide(peer: &Peer) -> Result<Settled<'_>, Fault> {
    let connect = peer.connect;
    let channel = tokio::time::timeout(connect, channel(&peer.transport, connect))
        .await
        .map_err(|_| "the connect did not settle within its frame")??;
    let mut client = PolicyPluginClient::new(channel).max_decoding_message_size(peer.answer_bytes);
    // The hello and a window of frames. The versions offered are read as
    // the hello is written: the fixed one alone while another stream of the
    // plugin stands.
    let (tx, rx) = mpsc::channel(WINDOW + 1);
    let (min, max) = peer.spoken.offer(&peer.versions);
    tx.try_send(proto::PluginRequest {
        request: Some(proto::plugin_request::Request::Hello(
            peer.hello((min, max)),
        )),
    })
    .map_err(|_| "the hello did not enter the stream")?;
    let response = tokio::time::timeout(
        connect,
        client.decide(tonic::Request::new(ReceiverStream::new(rx))),
    )
    .await
    .map_err(|_| "the stream did not open within the connect frame")?
    .map_err(|status| status.to_string())?;
    let mut inbound = response.into_inner();
    let first = tokio::time::timeout(connect, inbound.message())
        .await
        .map_err(|_| "the hello was not acked within the connect frame")?
        .map_err(|status| status.to_string())?;
    match first {
        Some(proto::PluginResponse {
            response: Some(proto::plugin_response::Response::Hello(ack)),
        }) => {
            if !ack.ok {
                return Err(Fault::Init(bounded(ack.error)));
            }
            if !(min..=max).contains(&ack.protocol) {
                let offered = if min == max {
                    min.to_string()
                } else {
                    format!("{min} to {max}")
                };
                return Err(format!(
                    "the plugin chose protocol {}, outside the offered {offered}",
                    ack.protocol
                )
                .into());
            }
            // An ack that lost the race to fix the plugin's version ends
            // its stream here: the error cancels it, and names both.
            match peer.spoken.settle(ack.protocol) {
                Ok(standing) => Ok((tx, inbound, ack.stages, standing)),
                Err(fixed) => Err(format!(
                    "the plugin chose protocol {} on this stream and {fixed} on the one that settled first",
                    ack.protocol
                )
                .into()),
            }
        }
        Some(_) => Err("the first frame was not the hello's ack".into()),
        None => Err("the stream ended before the hello's ack".into()),
    }
}

async fn run_lane(lane: Arc<Lane>) {
    let (tx, mut inbound, stages, standing) = match open_decide(&lane.peer).await {
        Ok(opened) => opened,
        Err(why) => {
            lane.fail_and_tell(why);
            return;
        }
    };
    lane.set_ready(tx, stages);
    if let Some(events) = lane.events() {
        events.connected();
    }
    let why = loop {
        tokio::select! {
            () = lane.closed.notified() => break "the plugin was dropped".to_owned(),
            message = inbound.message() => match message {
                Ok(Some(proto::PluginResponse {
                    response: Some(proto::plugin_response::Response::Decision(decision)),
                })) => lane.deliver(decision),
                Ok(Some(_)) => {}
                Ok(None) => break "the plugin ended the stream".to_owned(),
                Err(status) => break status.to_string(),
            },
        }
    };
    // The stream's stand on its plugin's version goes first, however the
    // stream ended: a stream opened on the word that this one failed is
    // offered what the plugin's other streams leave it.
    drop(standing);
    // A status's text is the plugin's: cut.
    lane.fail_and_tell(Fault::Stream(bounded(why)));
}

async fn run_observe(lane: Arc<ObserveLane>) {
    let peer = &*lane.peer;
    observe(&lane, |version| open_observe(peer, version)).await;
}

/// The lines one plugin's events task has said and not armed again: the
/// outage's (a call that failed) and the streak's (a turn that panicked).
/// One edge arms both, the end of a call that stood: evidence only the
/// plugin makes. An event sent or carried arms neither: a send enters the
/// host's own channel before the plugin has answered anything.
#[derive(Default)]
struct Said {
    outage: bool,
    streak: bool,
}

impl Said {
    /// A call ended, or none opened (`stood` is false): whether the failure
    /// says its line, once an outage.
    fn ended(&mut self, stood: bool) -> bool {
        if stood {
            *self = Self::default();
        }
        !std::mem::replace(&mut self.outage, true)
    }

    /// A turn panicked: whether it says its line, once a streak.
    fn panicked(&mut self) -> bool {
        !std::mem::replace(&mut self.streak, true)
    }
}

/// One `Observe` call as it is opened: the call and the sender of its
/// request stream, the hello already in it.
type Opened<C> = (C, mpsc::Sender<proto::ObserveRequest>);

/// The events task of one plugin over the calls `open` makes, each under
/// the protocol version it is handed, one turn after another until the
/// plugin is dropped.
async fn observe<O, F, C, T>(lane: &ObserveLane, mut open: O)
where
    O: FnMut(u32) -> F,
    F: Future<Output = Result<Opened<C>, Cow<'static, str>>>,
    C: Future<Output = Result<T, tonic::Status>>,
{
    let Some(mut queue) = lane.queue.lock().take() else {
        return;
    };
    // The events sent, one count for this load of the plugin. An event's
    // number is that count plus the highest count of refusals an event has
    // carried: both only grow, so a number never goes down, on one stream or
    // across two, and a gap is what the plugin lost.
    let mut sent = 0_u64;
    // The refusals the events carried, the highest so far.
    let mut lost = 0_u64;
    // A turn that panicked did so in the host's own code (the plugin's
    // telling of a failure). The queue, the counts and the lines said are
    // held here, outside the turn: the events raised meanwhile wait in the
    // queue, the numbers go on, a line said stays said, and the stream is
    // asked for again a pause later. No decision asks for this stream, as
    // one does for a `Decide` stream.
    let mut said = Said::default();
    // The event a turn took and could not write: the next turn to find room
    // writes it first, in the same call or the next.
    let mut kept = None;
    while let Err(panicked) = Caught(Box::pin(observe_streams(
        lane, &mut queue, &mut kept, &mut sent, &mut lost, &mut said, &mut open,
    )))
    .await
    {
        if said.panicked() {
            tracing::error!(
                plugin = lane.peer.name,
                error = panic_text(&*panicked),
                "a plugin's observe task panicked"
            );
        }
        tokio::time::sleep(RECONNECT_PAUSE).await;
    }
}

/// The number of the next event sent: the events sent, this one with them,
/// plus the events lost ahead of it. An event comes with the refusals its
/// thread had seen as it entered the queue, and the highest carried so far
/// is what was lost ahead. A count carried late, by a thread that read it
/// before another's refusal, moves no number back.
fn number(sent: &mut u64, lost: &mut u64, carried: u64) -> u64 {
    *lost = (*lost).max(carried);
    *sent += 1;
    *sent + *lost
}

/// Opens one `Observe` call: the channel within the connect frame, then the
/// call, its hello first on the request stream. Nothing answers a hello
/// here, so it names `version` alone: the one the plugin's `Decide` streams
/// fixed.
async fn open_observe(
    peer: &Peer,
    version: u32,
) -> Result<
    Opened<impl Future<Output = Result<tonic::Response<proto::Ack>, tonic::Status>>>,
    Cow<'static, str>,
> {
    let dialled = tokio::time::timeout(peer.connect, channel(&peer.transport, peer.connect));
    let channel = match dialled.await {
        Ok(Ok(channel)) => channel,
        Ok(Err(why)) => return Err(Cow::Owned(why)),
        Err(_) => return Err(Cow::Borrowed("the connect did not settle within its frame")),
    };
    let mut client = PolicyPluginClient::new(channel).max_decoding_message_size(peer.answer_bytes);
    let (tx, rx) = mpsc::channel::<proto::ObserveRequest>(OBSERVE_WINDOW);
    let hello = proto::ObserveRequest {
        request: Some(proto::observe_request::Request::Hello(
            peer.hello((version, version)),
        )),
    };
    tx.try_send(hello)
        .map_err(|_| "the hello did not enter the stream")?;
    let call = async move {
        client
            .observe(tonic::Request::new(ReceiverStream::new(rx)))
            .await
    };
    Ok((call, tx))
}

/// The `Observe` calls of one plugin, each opened when the one before it
/// ended, until the plugin is dropped. A call names the version the
/// plugin's `Decide` streams last fixed: none opens before one has settled
/// (the events wait in the queue), and a call lives under the version it
/// named until a settle fixes another, when it is ended and the next names
/// that one. With no stream standing the last version holds, so a call that
/// ends is opened again under it. A word that version has no kind for ends
/// the call that takes it then: the word is kept, the events behind it
/// wait in the queue, and the next call opens once a stream stands, the
/// version fixed carries the word, or a stream has settled since the word
/// was kept, though it has ended.
async fn observe_streams<O, F, C, T>(
    lane: &ObserveLane,
    queue: &mut mpsc::Receiver<proto::ObserveRequest>,
    kept: &mut Option<Kept>,
    sent: &mut u64,
    lost: &mut u64,
    said: &mut Said,
    open: &mut O,
) where
    O: FnMut(u32) -> F,
    F: Future<Output = Result<Opened<C>, Cow<'static, str>>>,
    C: Future<Output = Result<T, tonic::Status>>,
{
    let frame = lane.peer.connect;
    let mut spoken = lane.peer.spoken.stand.subscribe();
    loop {
        if lane.events.strong_count() == 0 {
            return;
        }
        // The wait for the first settle is cut each frame: the plugin may
        // have been dropped meanwhile. A word kept waits the same way, for
        // a stream that stands, a version fixed that carries it, or a
        // settle since it was kept: a stream that settled and ended says
        // what becomes of the word as one that stands does.
        let (since, under) = kept
            .as_ref()
            .map_or((0, 0), |kept| (carried_since(&kept.word), kept.settles));
        let settled = spoken.wait_for(|stand| {
            stand.fixed != 0
                && (stand.standing != 0 || stand.fixed >= since || stand.settles != under)
        });
        let version = match tokio::time::timeout(frame, settled).await {
            Ok(Ok(stand)) => stand.fixed,
            Ok(Err(_)) => return,
            Err(_) => continue,
        };
        let (call, tx) = match open(version).await {
            Ok(opened) => opened,
            Err(why) => {
                // The state first, the telling after: a telling that panics
                // is told once an outage.
                if said.ended(false) {
                    lane.failed(&why);
                }
                tokio::time::sleep(frame).await;
                continue;
            }
        };
        let moved = async {
            if spoken
                .wait_for(|stand| stand.fixed != version)
                .await
                .is_err()
            {
                std::future::pending::<()>().await;
            }
        };
        let requests = Requests {
            tx,
            version,
            spoken: &lane.peer.spoken,
            kept: &mut *kept,
        };
        let ended = match forward(call, requests, queue, sent, lost, frame, moved).await {
            Turn::Dropped => return,
            // No call failed and no pause is owed: the next call opens as
            // soon as a stream says its version. A call that stood arms
            // the lines again, as one that ended by itself does.
            Turn::Moved { stood } => {
                if stood {
                    *said = Said::default();
                }
                continue;
            }
            Turn::Ended(ended) => ended,
        };
        if said.ended(ended.stood) {
            lane.failed(&ended.why);
        }
        // The pause runs from the instant the end was seen: a turn's wait
        // for the call's outcome is inside it.
        tokio::time::sleep_until(ended.at + frame).await;
    }
}

/// The first protocol version whose streams carry `request`'s event: `0`
/// for a request that is no event.
fn carried_since(request: &proto::ObserveRequest) -> u32 {
    match &request.request {
        Some(proto::observe_request::Request::Event(numbered)) => {
            numbered.event.as_ref().map_or(0, view::since)
        }
        _ => 0,
    }
}

/// The request side of one `Observe` call: where its events are written,
/// and the protocol version its hello named, which says what it carries.
/// A word it does not carry is judged by the plugin's version as it stands
/// when the word is taken.
struct Requests<'a> {
    tx: mpsc::Sender<proto::ObserveRequest>,
    version: u32,
    /// The plugin's version, as its `Decide` streams stand on it.
    spoken: &'a Spoken,
    /// The event a turn took and could not write, in this call or the one
    /// before: the next turn to find room writes it first, or keeps it in
    /// turn.
    kept: &'a mut Option<Kept>,
}

/// An event an events call took and could not write: a word its version
/// has no kind for, kept for the call that opens next, or any event that
/// found the call's window full or its receiver gone, which the next turn
/// to find room writes first, in the same call or the next. It holds the
/// settles its plugin had seen as it was kept: a later count says a stream
/// settled since, though none may stand.
struct Kept {
    word: proto::ObserveRequest,
    settles: u64,
}

/// How one turn of the events task ended.
enum Turn {
    /// The plugin was dropped: its queue closed.
    Dropped,
    /// The host ended the call for the next one: a settle fixed another
    /// version than it named, or it took a word it cannot carry while no
    /// stream stood on its version. `stood` says whether it had stood
    /// [`STAND`] or longer.
    Moved { stood: bool },
    /// The call ended with its plugin loaded.
    Ended(Ended),
}

/// How one `Observe` call ended, its plugin still loaded.
struct Ended {
    /// Why, in the words of the host's line.
    why: Cow<'static, str>,
    /// Whether the call stood: [`STAND`] or longer from its open to the
    /// instant its end was seen.
    stood: bool,
    /// When the end was seen, by the call's outcome or by a turn.
    at: tokio::time::Instant,
}

/// What one turn of a call's select found.
enum Found {
    /// An event was written or let go, or the call found room for the one
    /// it kept: the next turn takes the next.
    Next,
    /// The call yields to the next one: a settle fixed another version, or
    /// the word taken is kept for a call that carries it.
    Yields,
    /// The call's receiver is gone: its stream has no room to give.
    Gone,
    /// The plugin was dropped: its queue closed.
    Closed,
}

/// One `Observe` call until it ends, the queue's events forwarded into it,
/// the word kept from the call before ahead of them. The host ends it for
/// the next call in two cases: `moved`, a settle fixed another version than
/// the call named; or the call took a word it cannot carry while no stream
/// stood on its version, and the word is kept.
///
/// A call's end shows on two sides, and the select takes either first: the
/// call's own outcome, or a turn that finds the call's receiver gone as it
/// asks for the stream's room. A turn that saw the end first waits for the
/// outcome, no longer than `frame`, so the same end is said from either
/// side; the end is dated where it was first seen.
async fn forward<C, T, M>(
    call: C,
    requests: Requests<'_>,
    queue: &mut mpsc::Receiver<proto::ObserveRequest>,
    sent: &mut u64,
    lost: &mut u64,
    frame: Duration,
    moved: M,
) -> Turn
where
    C: Future<Output = Result<T, tonic::Status>>,
    M: Future<Output = ()>,
{
    let Requests {
        tx,
        version,
        spoken,
        kept,
    } = requests;
    tokio::pin!(call);
    tokio::pin!(moved);
    let opened = tokio::time::Instant::now();
    let (outcome, at) = loop {
        // One event: the word kept first, then the queue's next. It is
        // judged as it is taken, room or none: a word the call's version
        // has no kind for does not wait behind a full window. Any other
        // event needs the stream's room. With room at hand it is written
        // at once. With none it rests in the task's keep, unnumbered, so a
        // turn cut short there, by a move or by the call's end, leaves it
        // for the next call. The wait for room is cut each frame, as the
        // wait for a settle is: a plugin dropped while its call's window
        // is full is read at the cut, and its task ends with it.
        let send = async {
            let (mut event, under) = match kept.take() {
                Some(kept) => (kept.word, Some(kept.settles)),
                None => match queue.recv().await {
                    Some(event) => (event, None),
                    None => return Found::Closed,
                },
            };
            if let Some(proto::observe_request::Request::Event(numbered)) = event.request.as_ref() {
                let newer = numbered
                    .event
                    .as_ref()
                    .is_some_and(|event| view::since(event) > version);
                // A word the call's version has no kind for is not written
                // and takes no number. While a stream stands on that
                // version no stream of the plugin hears the word: it is let
                // go, and the refusals it carried still count. So is a word
                // kept since before a settle that fixed that version, though
                // the stream has ended: the settle said what becomes of it.
                // Otherwise the version is free or has moved: the word is
                // kept, and the call that opens next writes it first. One
                // reading judges the word and dates its keeping.
                if newer {
                    let stand = spoken.stand();
                    let fixed = stand.fixed == version;
                    let settled = under.is_some_and(|under| stand.settles != under);
                    if fixed && (stand.standing != 0 || settled) {
                        *lost = (*lost).max(numbered.seq);
                        return Found::Next;
                    }
                    *kept = Some(Kept {
                        word: event,
                        settles: stand.settles,
                    });
                    return Found::Yields;
                }
            }
            let room = match tx.try_reserve() {
                Ok(room) => room,
                Err(refused) => {
                    let settles = under.unwrap_or_else(|| spoken.stand().settles);
                    *kept = Some(Kept {
                        word: event,
                        settles,
                    });
                    if matches!(refused, mpsc::error::TrySendError::Closed(())) {
                        return Found::Gone;
                    }
                    // The room this wait finds is given back: the turn that
                    // follows takes the kept event and asks again.
                    loop {
                        match tokio::time::timeout(frame, tx.reserve()).await {
                            Ok(Ok(_)) => return Found::Next,
                            Ok(Err(_)) => return Found::Gone,
                            Err(_) if queue.is_closed() => return Found::Closed,
                            Err(_) => {}
                        }
                    }
                }
            };
            // Numbered here, where it enters the stream: one task numbers
            // every event, in the order it sends.
            if let Some(proto::observe_request::Request::Event(numbered)) = event.request.as_mut() {
                numbered.seq = number(sent, lost, numbered.seq);
            }
            room.send(event);
            Found::Next
        };
        // One event a turn, and the move ahead of it: a call whose version
        // a settle has left takes no further event.
        let found = tokio::select! {
            biased;
            () = &mut moved => Found::Yields,
            outcome = &mut call => break (Some(outcome), tokio::time::Instant::now()),
            found = send => found,
        };
        match found {
            Found::Next => {}
            Found::Closed => {
                // The request stream ends with its sender; the plugin
                // answers its one Ack, awaited no longer than a connect.
                drop(tx);
                let _ = tokio::time::timeout(frame, &mut call).await;
                return Turn::Dropped;
            }
            Found::Yields => {
                // Ended as for a dropped plugin: the request stream closes
                // and the plugin's one Ack is awaited no longer than a
                // connect.
                let stood = opened.elapsed() >= STAND;
                drop(tx);
                let _ = tokio::time::timeout(frame, &mut call).await;
                return Turn::Moved { stood };
            }
            Found::Gone => {
                let at = tokio::time::Instant::now();
                break (tokio::time::timeout(frame, &mut call).await.ok(), at);
            }
        }
    };
    let why = match outcome {
        Some(Ok(_)) => Cow::Borrowed("the plugin ended the observe stream"),
        Some(Err(status)) => Cow::Owned(bounded(status.to_string())),
        None => Cow::Borrowed("the observe stream took no more events"),
    };
    Turn::Ended(Ended {
        why,
        stood: at.duration_since(opened) >= STAND,
        at,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/grpc/runtime/tests.rs"]
mod tests;
