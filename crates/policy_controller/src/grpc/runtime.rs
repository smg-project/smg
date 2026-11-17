//! The I/O side of the gRPC plugin: one thread per process that owns every
//! stream, and the lanes through which a deciding thread hands it a frame
//! and waits for the answer. A deciding thread never touches a socket; the
//! I/O thread never waits on a decision.

use std::{
    any::Any,
    error::Error,
    fmt,
    future::Future,
    io,
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
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
use tokio::{
    net::UnixStream,
    sync::{mpsc, oneshot, Notify},
    task::JoinSet,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity, Uri};
use tower::service_fn;

use super::{config::Transport, view, PROTOCOL};
use crate::{
    plugins::OpenDecisions,
    proto::{self, policy_plugin_client::PolicyPluginClient},
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
    /// or the last gap it closed crosses the two. A frame whose own budget
    /// is no longer than the promise is excused only where another frame's
    /// flight already shows the thread stalled: alone it misses before the
    /// thread can be seen late, and its miss is nobody's to say
    /// ([`Self::vouches_for`]).
    pub(crate) fn stalled_between(&self, from: Instant, until: Instant) -> bool {
        let (from, until) = (self.micros(from), self.micros(until));
        if until.saturating_sub(self.ran.load(Ordering::Relaxed)) > PROMISE_MICROS {
            return true;
        }
        let (start, end) = *self.stall.lock();
        start < end && start <= until && from <= end
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
}

impl Peer {
    fn hello(&self) -> proto::PluginHello {
        proto::PluginHello {
            protocol_min: PROTOCOL,
            protocol_max: PROTOCOL,
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
    /// The `Observe` stream failed to open or ended.
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
    /// core.
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

/// Opens a `Decide` stream: the connect, the hello and its ack, each under
/// the connect frame. The ack's stage bits come back with the stream.
async fn open_decide(
    peer: &Peer,
) -> Result<(mpsc::Sender<proto::PluginRequest>, Inbound, u32), Fault> {
    let connect = peer.connect;
    let channel = tokio::time::timeout(connect, channel(&peer.transport, connect))
        .await
        .map_err(|_| "the connect did not settle within its frame")??;
    let mut client = PolicyPluginClient::new(channel).max_decoding_message_size(peer.answer_bytes);
    // The hello and a window of frames.
    let (tx, rx) = mpsc::channel(WINDOW + 1);
    tx.try_send(proto::PluginRequest {
        request: Some(proto::plugin_request::Request::Hello(peer.hello())),
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
                Err(Fault::Init(bounded(ack.error)))
            } else if ack.protocol != PROTOCOL {
                Err(format!(
                    "the plugin chose protocol {}, outside the offered {PROTOCOL}",
                    ack.protocol
                )
                .into())
            } else {
                Ok((tx, inbound, ack.stages))
            }
        }
        Some(_) => Err("the first frame was not the hello's ack".into()),
        None => Err("the stream ended before the hello's ack".into()),
    }
}

async fn run_lane(lane: Arc<Lane>) {
    let (tx, mut inbound, stages) = match open_decide(&lane.peer).await {
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
    // A status's text is the plugin's: cut.
    lane.fail_and_tell(Fault::Stream(bounded(why)));
}

async fn run_observe(lane: Arc<ObserveLane>) {
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
    // telling of a failure). The queue and the counts are held here, outside
    // the turn: the events raised meanwhile wait in the queue, the numbers
    // go on, and the stream is asked for again a pause later. No decision
    // asks for this stream, as one does for a `Decide` stream. A panic at
    // every start says its line once: the count of events sent at the last
    // line said.
    let mut said_at = None;
    while let Err(panicked) = Caught(Box::pin(observe_streams(
        &lane, &mut queue, &mut sent, &mut lost,
    )))
    .await
    {
        if streak_says(&mut said_at, sent) {
            tracing::error!(
                plugin = lane.peer.name,
                error = panic_text(&*panicked),
                "a plugin's observe task panicked"
            );
        }
        tokio::time::sleep(RECONNECT_PAUSE).await;
    }
}

/// Whether a panicked turn says its line: once a streak. A streak is the
/// panicked turns that sent no event between them, so a panic at every start
/// says itself once, and a panic after an event was sent says itself again.
fn streak_says(said_at: &mut Option<u64>, sent: u64) -> bool {
    said_at.replace(sent) != Some(sent)
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

/// The `Observe` calls of one plugin, each opened when the one before it
/// ended, until the plugin is dropped.
async fn observe_streams(
    lane: &ObserveLane,
    queue: &mut mpsc::Receiver<proto::ObserveRequest>,
    sent: &mut u64,
    lost: &mut u64,
) {
    let peer = &lane.peer;
    // One line per outage: the first failure after a call that carried an
    // event.
    let mut reported = false;
    loop {
        if lane.events.strong_count() == 0 {
            return;
        }
        let channel = match tokio::time::timeout(
            peer.connect,
            channel(&peer.transport, peer.connect),
        )
        .await
        {
            Ok(Ok(channel)) => channel,
            outcome => {
                if !reported {
                    reported = true;
                    match outcome {
                        Ok(Err(why)) => lane.failed(&why),
                        _ => lane.failed("the connect did not settle within its frame"),
                    }
                }
                tokio::time::sleep(peer.connect).await;
                continue;
            }
        };
        let mut client =
            PolicyPluginClient::new(channel).max_decoding_message_size(peer.answer_bytes);
        let (tx, rx) = mpsc::channel::<proto::ObserveRequest>(OBSERVE_WINDOW);
        let hello = proto::ObserveRequest {
            request: Some(proto::observe_request::Request::Hello(peer.hello())),
        };
        if tx.try_send(hello).is_err() {
            continue;
        }
        let call = client.observe(tonic::Request::new(ReceiverStream::new(rx)));
        tokio::pin!(call);
        // Forwards until the plugin is dropped (its queue closes: `false`)
        // or the call lets its receiver go (`true`).
        let carried = AtomicBool::new(false);
        let forward = async {
            loop {
                match queue.recv().await {
                    Some(mut event) => {
                        // Numbered here, where it enters the stream: one
                        // task numbers every event, in the order it sends.
                        if let Some(proto::observe_request::Request::Event(numbered)) =
                            event.request.as_mut()
                        {
                            numbered.seq = number(sent, lost, numbered.seq);
                        }
                        if tx.send(event).await.is_err() {
                            return true;
                        }
                        carried.store(true, Ordering::Relaxed);
                    }
                    None => return false,
                }
            }
        };
        let plugin_lives = tokio::select! {
            outcome = &mut call => {
                if carried.load(Ordering::Relaxed) {
                    reported = false;
                }
                if !reported {
                    reported = true;
                    match outcome {
                        Ok(_) => lane.failed("the plugin ended the observe stream"),
                        Err(status) => lane.failed(&bounded(status.to_string())),
                    }
                }
                true
            }
            alive = forward => alive,
        };
        if !plugin_lives {
            // The request stream ends with its sender; the plugin answers
            // its one Ack, awaited no longer than a connect.
            drop(tx);
            let _ = tokio::time::timeout(peer.connect, &mut call).await;
            return;
        }
        tokio::time::sleep(peer.connect).await;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc, Weak,
        },
        time::{Duration, Instant},
    };

    use tokio::{sync::mpsc, task::JoinSet};

    use super::{
        number, run_commands, streak_says, Command, Lane, LaneEvents, LaneState, ObserveLane, Peer,
        Runtime, SendError, Wait, Witness, PROMISE_MICROS, RECONNECT_PAUSE, WINDOW,
    };
    use crate::{grpc::config::Transport, plugins::OpenDecisions, proto};

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

    /// A peer nothing listens behind.
    fn nowhere() -> Arc<Peer> {
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
            witness.stalled_between(back, later),
            "a frame sent as it ended"
        );
        assert!(
            !witness.stalled_between(back + Duration::from_micros(1), later),
            "a frame sent after it"
        );
        assert!(
            !witness.stalled_between(at(&witness, 1), at(&witness, 9_999)),
            "a frame that landed before it"
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
    /// it starts in time. Nothing ticks here: no I/O thread runs, and this
    /// thread stands for it. A try whose runner held the thread past the
    /// promise between the arrival and the next frame says nothing of the
    /// witness and is made again.
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

    /// What an `Observe` lane tells its plugin, with a first telling that
    /// panics.
    #[derive(Default)]
    struct PanicsOnce {
        told: AtomicU64,
    }

    impl LaneEvents for PanicsOnce {
        fn connected(&self) {}
        fn stream_failed(&self, _why: &str, _at: Instant) {}
        fn late_answer(&self) {}
        fn observe_failed(&self, _why: &str) {
            let earlier = self.told.fetch_add(1, Ordering::Relaxed);
            assert_ne!(earlier, 0, "the first telling panics");
        }
    }

    /// A panicked turn says its line once a streak: turns that sent no event
    /// between them are one streak, and a turn that sent one starts another.
    #[test]
    fn a_streak_of_panicked_turns_says_one_line() {
        let mut said_at = None;
        assert!(streak_says(&mut said_at, 0));
        assert!(!streak_says(&mut said_at, 0));
        assert!(!streak_says(&mut said_at, 0));
        assert!(streak_says(&mut said_at, 7));
        assert!(!streak_says(&mut said_at, 7));
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

    /// An `Observe` task that panics is not the end of the plugin's events.
    /// Here the connect finds no socket and the plugin's own telling of that
    /// panics, once. The queue keeps its receiver across the panic, so an
    /// event raised after it is taken, and the stream is asked for again a
    /// pause later: the second telling stands.
    #[tokio::test]
    async fn an_observe_task_that_panics_is_started_again() {
        let (commands, taken) = mpsc::unbounded_channel();
        let mut looping = JoinSet::new();
        looping.spawn(run_commands(taken, Arc::new(Witness::new())));
        let events = Arc::new(PanicsOnce::default());
        let listener: Arc<dyn LaneEvents> = Arc::clone(&events) as _;
        let (queue, raised) = mpsc::channel(4);
        let lane = ObserveLane::new(nowhere(), Arc::downgrade(&listener), raised);
        assert!(commands.send(Command::Observe(Arc::new(lane))).is_ok());
        for telling in 1..=2 {
            let mut waits = 0;
            while events.told.load(Ordering::Relaxed) < telling {
                waits += 1;
                assert!(waits < 10_000, "telling {telling} never came");
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            // The first telling panicked: the queue still has its receiver.
            assert!(
                queue.try_send(proto::ObserveRequest::default()).is_ok(),
                "the queue lost its receiver with the task that panicked"
            );
        }
    }

    /// What an `Observe` lane tells its plugin, with tellings that panic
    /// until the `last`, and the loop's clock read at each.
    struct PanicsUntil {
        last: usize,
        told: parking_lot::Mutex<Vec<tokio::time::Instant>>,
    }

    impl LaneEvents for PanicsUntil {
        fn connected(&self) {}
        fn stream_failed(&self, _why: &str, _at: Instant) {}
        fn late_answer(&self) {}
        fn observe_failed(&self, _why: &str) {
            let tellings = {
                let mut told = self.told.lock();
                told.push(tokio::time::Instant::now());
                told.len()
            };
            assert!(tellings >= self.last, "a telling before the last panics");
        }
    }

    /// A turn that panicked is started again a pause later, never at once:
    /// under a telling that panics every time the I/O thread, which carries
    /// every stream, would spin. The loop's clock is paused here and moves
    /// by its sleeps alone, and each telling reads it.
    #[tokio::test(start_paused = true)]
    async fn a_panicked_turn_is_started_again_a_pause_later() {
        let (commands, taken) = mpsc::unbounded_channel();
        let mut looping = JoinSet::new();
        looping.spawn(run_commands(taken, Arc::new(Witness::new())));
        let events = Arc::new(PanicsUntil {
            last: 4,
            told: parking_lot::Mutex::new(Vec::new()),
        });
        let listener: Arc<dyn LaneEvents> = Arc::clone(&events) as _;
        let (_queue, raised) = mpsc::channel(4);
        let lane = ObserveLane::new(nowhere(), Arc::downgrade(&listener), raised);
        assert!(commands.send(Command::Observe(Arc::new(lane))).is_ok());
        tokio::time::sleep(RECONNECT_PAUSE * 8).await;
        let told = events.told.lock().clone();
        assert!(told.len() >= 4, "{} tellings in eight pauses", told.len());
        for pair in told.windows(2).take(3) {
            assert!(
                pair[1].duration_since(pair[0]) >= RECONNECT_PAUSE,
                "a panicked turn was started again at once"
            );
        }
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
