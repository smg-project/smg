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
    /// core. The package's own tests build it (`test-util`); no build of a
    /// product has it.
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
mod tests {
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
        number, observe, run_commands, Command, Ended, Fault, Lane, LaneEvents, LaneState,
        ObserveLane, Opened, Peer, Requests, Runtime, Said, SendError, Spoken, Stand, Turn, Wait,
        Witness, OBSERVE_WINDOW, PEER_TEXT, PROMISE_MICROS, RECONNECT_PAUSE, STAND, WINDOW,
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
                !lane.witness.stalled_between(after, after),
                "the delivery was a run of the thread"
            );
            let soon = Instant::now() + Duration::from_secs(5);
            let Wait::Answered(arrived, _) = lane.wait_answer(first, soon) else {
                panic!("the first frame was answered");
            };
            assert!(
                lane.witness.stalled_between(sent, arrived),
                "the answer came across a gap past the promise"
            );
            // The gap ends at the arrival, exclusive: a frame that took
            // flight in its closing microsecond is not crossed, and one a
            // microsecond earlier is.
            assert!(
                !lane.witness.stalled_between(arrived, arrived),
                "a frame in the gap's closing microsecond"
            );
            assert!(
                lane.witness
                    .stalled_between(arrived - Duration::from_micros(1), arrived),
                "a frame a microsecond before the gap closed"
            );
            // The gap began in the microsecond the first frame took
            // flight: an answer owed only until then is not crossed, and
            // one owed a microsecond longer is.
            assert!(
                !lane.witness.stalled_between(sent, sent),
                "an answer owed until the gap's first microsecond"
            );
            assert!(
                lane.witness
                    .stalled_between(sent, sent + Duration::from_micros(1)),
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
        let witness = Arc::clone(runtime.witness());
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
}
