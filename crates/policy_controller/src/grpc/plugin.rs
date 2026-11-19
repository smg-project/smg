//! A connected plugin and how the host runs it: one stream per deciding
//! thread, one frame per decision per side, the answer read at the first
//! stage that needs it under the frame's deadline, and the rules for what a
//! miss, a decline, an invalid answer or a lost stream does.

use std::{
    cell::{RefCell, RefMut},
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use thread_local::ThreadLocal;
use tokio::sync::mpsc;

use super::{
    config::{GrpcPluginSpec, Transport},
    runtime::{
        answer_bytes, Command, Fault, Lane, LaneEvents, ObserveLane, Peer, Readiness, Runtime,
        SendError, Wait, Witness,
    },
    unit::Unit,
    view::{self, Answer, FILTER, PICK, SCORE},
};
use crate::{
    decision::{Decision, Target},
    framework::DecisionScope,
    lease::{Lease, Membership, Outcome, ReleaseReason, Side, Signal},
    plugins::{
        config::{EventSet, Exposure, Failure, FrameBudget},
        intern, own_bound, BoundPlugin, Counters, Fallback, HalfOpen, OpenDecision, Stage,
    },
    proto,
    spec::Needs,
    view::{Candidate, Phase, PoolFacts, RequestFacts},
};

/// Lifecycle events waiting for the `Observe` stream at most; past it an
/// event is dropped and counted, never a wait on the thread that raised it.
const OBSERVE_QUEUE: usize = 1_024;

/// Why a plugin did not load.
#[derive(Debug)]
pub enum LoadError {
    /// The row is not a plugin the host can connect to.
    Spec(String),
    /// The connect or the hello did not settle within the connect frame, or
    /// the stream failed before the ack.
    Connect(String),
    /// The plugin's `init` failed: the ack's reason. Fails the apply.
    Init(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spec(why) => write!(f, "the plugin row is invalid: {why}"),
            Self::Connect(why) => write!(f, "the plugin did not connect: {why}"),
            Self::Init(why) => write!(f, "the plugin's init failed: {why}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// The counters of one plugin, read through [`Plugin::stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Frames the plugin answered within the contract.
    pub answered: u64,
    /// Stages the plugin declined (one per declined stage).
    pub declined: u64,
    /// Frames declined for the plugin's own budget: its Busy.
    pub busy: u64,
    /// Frames whose answer broke a validity rule.
    pub invalid: u64,
    /// Decisions whose budget was gone on arrival: no frame, the stages
    /// fell back, the breaker counted a budget hit.
    pub budget_gone: u64,
    /// Decisions the budget's remainder ended short of the plugin's answer:
    /// no frame because an earlier open of the decision spent it, or a
    /// deadline missed that is not the plugin's failure (every miss but
    /// `own_misses`); the stages fell back, nothing counted on the breaker.
    pub budget_spent: u64,
    /// Frames whose deadline passed without an answer; each falls back. A
    /// miss is the plugin's failure (`own_misses`), the host's (`stalls`),
    /// nobody's to say (`unread`) or the decision's budget spent by
    /// others. Every miss but the host's is in the plugin's latency sample
    /// for its decision, at what the frame was given.
    pub missed: u64,
    /// Missed frames that are the plugin's failure, one a decision on the
    /// breaker: the bound was the plugin's own (the frame had its whole
    /// budget, or the plugin's own spend in the decision and this grant
    /// reach the decision's budget) and the I/O thread was seen running
    /// inside a flight longer than its promise. Among `missed`, not among
    /// `budget_spent`.
    pub own_misses: u64,
    /// Missed frames under the plugin's own bound whose flight was no
    /// longer than the I/O thread's promise: it ended before that thread
    /// could be seen late, so nothing says whether the plugin or the host
    /// was late. The decision's budget spent and no failure: among
    /// `missed` and `budget_spent`.
    pub unread: u64,
    /// Frames answered or missed while the I/O thread stalled (it did not
    /// run for longer than its promise while the answer was owed): the
    /// host's, in no latency sample. A missed one is among `missed` and
    /// `budget_spent` too and falls back all the same.
    pub stalls: u64,
    /// Decisions bypassed by the open breaker.
    pub bypassed: u64,
    /// Decisions whose pool exceeded `max_candidates`: no frame.
    pub too_many: u64,
    /// Decisions that found this thread's stream not open (never opened,
    /// or failed and reconnecting): no frame and no wait; the stream opens
    /// behind the decision.
    pub cold: u64,
    /// Frames the stream did not take, its window of frames in flight
    /// full (the plugin was not answering or not reading the ones before):
    /// no frame, the stages fell back; nothing on the breaker, which the
    /// misses before it reach through the latency sample.
    pub refused: u64,
    /// `Decide` streams that settled (the hello acked).
    pub connects: u64,
    /// Streams that failed to open, ended or erred. The breaker hears a
    /// plugin's streams as one: at most one failure a pause.
    pub failed_streams: u64,
    /// Answers for a decision nobody waited for any more.
    pub late_answers: u64,
    /// Lifecycle events handed to the `Observe` stream.
    pub observed: u64,
    /// Lifecycle events refused at a full queue: each leaves a gap of one in
    /// the events' numbers.
    pub observe_dropped: u64,
    /// Times the breaker opened.
    pub breaker_opens: u64,
}

#[derive(Default)]
struct Tally {
    answered: AtomicU64,
    declined: AtomicU64,
    busy: AtomicU64,
    invalid: AtomicU64,
    budget_gone: AtomicU64,
    budget_spent: AtomicU64,
    missed: AtomicU64,
    own_misses: AtomicU64,
    unread: AtomicU64,
    stalls: AtomicU64,
    bypassed: AtomicU64,
    too_many: AtomicU64,
    cold: AtomicU64,
    refused: AtomicU64,
    connects: AtomicU64,
    failed_streams: AtomicU64,
    late_answers: AtomicU64,
    observed: AtomicU64,
    observe_dropped: AtomicU64,
}

fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// What every thread and stream of one plugin shares.
struct Inner {
    me: Weak<Inner>,
    runtime: Arc<Runtime>,
    name: &'static str,
    frame: FrameBudget,
    exposure: Exposure,
    needs: Needs,
    observes: EventSet,
    failure: Failure,
    max_candidates: u32,
    counters: Counters,
    unit: Unit,
    tally: Tally,
    peer: Arc<Peer>,
    observe_tx: mpsc::Sender<proto::ObserveRequest>,
    /// The `Observe` stream's queue until the first `Decide` stream
    /// settles, which opens it. None from the load on where the row lists no
    /// kind of event: no stream is opened, and nothing is told.
    observe_rx: Mutex<Option<mpsc::Receiver<proto::ObserveRequest>>>,
    /// How many counters the row declares: an answer holds no more
    /// increments.
    declared: usize,
}

impl Inner {
    /// The plugin as its streams hold it: weakly, so a dropped plugin's
    /// streams end.
    fn events(&self) -> Weak<dyn LaneEvents> {
        let me: Weak<Self> = self.me.clone();
        me
    }

    fn now(&self) -> u64 {
        self.runtime.now_micros()
    }

    /// An answer outside the contract: the plugin's fault, one failure of
    /// the decision it was read in.
    fn invalid(&self, why: view::Invalid) {
        bump(&self.tally.invalid);
        tracing::warn!(plugin = self.name, %why, "a plugin's answer broke the contract");
        self.unit.fail(self.now());
    }

    fn observe(&self, event: proto::lifecycle_event::Event) {
        // The number is the stream's: the I/O thread writes it where the
        // event enters the stream, so threads that raise events at once
        // cannot send a later number first. A refused event is counted
        // here, and an event carries that count as it enters the queue: the
        // stream's task adds the highest it has seen, so a gap sits behind
        // every event that entered the queue before the loss. An event whose
        // thread stood between this read and its entry when the loss fell
        // can arrive on either side.
        let request = proto::ObserveRequest {
            request: Some(proto::observe_request::Request::Event(
                proto::LifecycleEvent {
                    seq: self.tally.observe_dropped.load(Ordering::Relaxed),
                    event: Some(event),
                },
            )),
        };
        match self.observe_tx.try_send(request) {
            Ok(()) => bump(&self.tally.observed),
            Err(_) => bump(&self.tally.observe_dropped),
        }
    }
}

impl LaneEvents for Inner {
    fn connected(&self) {
        bump(&self.tally.connects);
        if let Some(queue) = self.observe_rx.lock().take() {
            let lane = ObserveLane::new(Arc::clone(&self.peer), self.events(), queue);
            self.runtime.send(Command::Observe(Arc::new(lane)));
        }
    }

    fn stream_failed(&self, why: &str, at: Instant) {
        bump(&self.tally.failed_streams);
        tracing::warn!(plugin = self.name, why, "a plugin stream failed");
        self.unit.stream_failed(at, || self.now());
    }

    fn late_answer(&self) {
        bump(&self.tally.late_answers);
    }

    fn observe_failed(&self, why: &str) {
        tracing::warn!(plugin = self.name, why, "a plugin's observe stream failed");
    }
}

/// A connected plugin, shared by every profile that binds it and every
/// thread that decides.
pub struct Plugin {
    inner: Arc<Inner>,
    /// One stream per deciding thread, by `thread_local`'s thread id (a
    /// thread that exited leaves its lane to the next thread with its id).
    lanes: ThreadLocal<Arc<Lane>>,
}

impl fmt::Debug for Plugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plugin")
            .field("name", &self.inner.name)
            .finish_non_exhaustive()
    }
}

fn check(spec: &GrpcPluginSpec) -> Result<(), LoadError> {
    let refuse = |why: &str| Err(LoadError::Spec(why.to_owned()));
    if spec.name.is_empty() {
        return refuse("the name is empty");
    }
    if spec.max_candidates == 0 {
        return refuse("max_candidates is zero");
    }
    if spec.connect_micros == 0 {
        return refuse("the connect frame is zero");
    }
    match &spec.transport {
        Transport::Unix { path, .. } => {
            if path.as_os_str().is_empty() {
                return refuse("the socket path is empty");
            }
        }
        Transport::Tls { address, tls } => {
            if address.is_empty() || tls.server_name.is_empty() {
                return refuse("the address or the server name is empty");
            }
            if tls.ca_pem.is_empty() || tls.cert_pem.is_empty() || tls.key_pem.is_empty() {
                return refuse("a TLS PEM is empty");
            }
        }
    }
    Ok(())
}

impl Plugin {
    /// Checks `spec`, opens this thread's stream and waits for the hello's
    /// ack within the connect frame: a plugin that cannot be reached, or
    /// whose `init` failed, fails the apply. The loading thread's stream
    /// lives as long as the plugin, whether or not that thread decides.
    pub fn load(runtime: &Arc<Runtime>, spec: GrpcPluginSpec) -> Result<Arc<Self>, LoadError> {
        check(&spec)?;
        let frame = spec.frame();
        let name = intern(&spec.name);
        let peer = Arc::new(Peer {
            name: spec.name,
            transport: spec.transport,
            config: spec.config,
            exposure: view::exposure(&spec.view),
            counters: spec.metrics.clone(),
            connect: Duration::from_micros(u64::from(spec.connect_micros)),
            answer_bytes: answer_bytes(spec.max_candidates, spec.metrics.len()),
        });
        let (observe_tx, observe_rx) = mpsc::channel(OBSERVE_QUEUE);
        let inner = Arc::new_cyclic(|me| Inner {
            me: me.clone(),
            runtime: Arc::clone(runtime),
            name,
            frame,
            exposure: spec.view,
            needs: spec.needs,
            observes: spec.observes,
            failure: spec.failure,
            max_candidates: spec.max_candidates,
            declared: spec.metrics.len(),
            counters: Counters::new(spec.metrics),
            unit: Unit::new(name, spec.breaker),
            tally: Tally::default(),
            peer,
            observe_tx,
            observe_rx: Mutex::new((!spec.observes.is_empty()).then_some(observe_rx)),
        });
        let plugin = Arc::new(Self {
            inner,
            lanes: ThreadLocal::new(),
        });
        plugin.warm()?;
        Ok(plugin)
    }

    fn lane(&self) -> &Arc<Lane> {
        self.lanes.get_or(|| {
            Arc::new(Lane::new(
                Arc::clone(&self.inner.peer),
                self.inner.events(),
                Arc::clone(self.inner.runtime.witness()),
            ))
        })
    }

    /// Opens this thread's stream and waits for the hello's ack within the
    /// connect frame; the apply calls it on every deciding thread, so no
    /// decision meets a stream that was never opened. The I/O thread gives
    /// the connect, the stream's open and the ack a connect frame each, and
    /// this wait is one frame for the three: a slower open is refused here
    /// and settles behind. A failed stream is asked for at once: the pause
    /// between two asks is the decisions' alone.
    pub fn warm(&self) -> Result<(), LoadError> {
        let lane = self.lane();
        lane.connect(&self.inner.runtime);
        match lane.wait_ready(Instant::now() + self.inner.peer.connect) {
            Readiness::Ready => Ok(()),
            Readiness::NotYet => Err(LoadError::Connect(
                "the hello was not acked within the connect frame".to_owned(),
            )),
            Readiness::Failed => Err(match lane.failure() {
                Some(Fault::Init(why)) => LoadError::Init(why),
                Some(Fault::Stream(why)) => LoadError::Connect(why),
                None => LoadError::Connect("the stream is not open".to_owned()),
            }),
        }
    }

    /// Whether this thread's stream is open.
    pub fn is_warm(&self) -> bool {
        self.lanes.get().is_some_and(|lane| lane.stages().is_some())
    }

    pub fn name(&self) -> &'static str {
        self.inner.name
    }

    pub fn failure(&self) -> Failure {
        self.inner.failure
    }

    pub fn needs(&self) -> Needs {
        self.inner.needs
    }

    /// The kinds of lifecycle event the plugin hears, as its row lists
    /// them.
    pub fn observes(&self) -> EventSet {
        self.inner.observes
    }

    /// The counters the plugin adds to.
    pub fn counters(&self) -> &Counters {
        &self.inner.counters
    }

    pub fn stats(&self) -> Stats {
        let t = &self.inner.tally;
        let read = |c: &AtomicU64| c.load(Ordering::Relaxed);
        Stats {
            answered: read(&t.answered),
            declined: read(&t.declined),
            busy: read(&t.busy),
            invalid: read(&t.invalid),
            budget_gone: read(&t.budget_gone),
            budget_spent: read(&t.budget_spent),
            missed: read(&t.missed),
            own_misses: read(&t.own_misses),
            unread: read(&t.unread),
            stalls: read(&t.stalls),
            bypassed: read(&t.bypassed),
            too_many: read(&t.too_many),
            cold: read(&t.cold),
            refused: read(&t.refused),
            connects: read(&t.connects),
            failed_streams: read(&t.failed_streams),
            late_answers: read(&t.late_answers),
            observed: read(&t.observed),
            observe_dropped: read(&t.observe_dropped),
            breaker_opens: self.inner.unit.opens(),
        }
    }

    /// One engine build's hold on the plugin: every profile of the build
    /// that binds the plugin takes its bound set from the one binder, and a
    /// later build takes a binder of its own.
    pub fn binder(self: &Arc<Self>) -> Binder {
        Binder {
            plugin: Arc::clone(self),
            observer_taken: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The bound set of the one profile of a build that binds the plugin
    /// once: the three adapters of the profile share it, and the engine
    /// opens it once per decision. A build that binds the plugin to several
    /// profiles takes each set from one [`Plugin::binder`]; taken here, each
    /// set would forward the lifecycle.
    pub fn bind(self: &Arc<Self>) -> Arc<Bound> {
        self.binder().bind()
    }
}

impl Drop for Plugin {
    fn drop(&mut self) {
        // A closed lane lets its sender go and its task stops reading: the
        // streams end with the plugin.
        for lane in &self.lanes {
            lane.close();
        }
    }
}

/// A frame on its way: the ticket its send returned. The take reads the
/// frame's answer over the same ticket, and the wait between the two is one
/// function (`Bound::wait`): a host that parks a decision on its ticket
/// instead of blocking its thread replaces that function alone.
#[derive(Clone, Copy)]
struct Ticket {
    /// The frame's id on its stream.
    decision_id: u64,
    /// When the frame left the deciding thread.
    at: Instant,
    /// When the wait for its answer ends.
    deadline: Instant,
}

/// What a missed frame says of its plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Miss {
    /// The I/O thread stalled while the answer was owed: the host's, no
    /// sample and no failure.
    Held,
    /// The bound was the plugin's own and the I/O thread was seen running
    /// inside the flight: the plugin's failure.
    Own,
    /// The bound was the plugin's own and the flight too short for the
    /// witness: nothing says whether the plugin or the host was late. The
    /// decision's budget spent.
    Unread,
    /// The decision's remainder clamped the frame: the decision's budget
    /// spent, whoever spent it.
    Clamped,
}

impl Miss {
    /// The one table of a miss, the witness's excuse first: `stalled` is
    /// the witness's word on the flight, `own_bound` whose bound the frame
    /// ran out under, `given_micros` the flight the plugin was given. A
    /// miss is the plugin's failure only where its bound was its own and
    /// the witness vouches for the flight; where nothing says which, it is
    /// the decision's spend.
    fn of(stalled: bool, own_bound: bool, given_micros: u64) -> Self {
        if stalled {
            Self::Held
        } else if !own_bound {
            Self::Clamped
        } else if Witness::vouches_for(given_micros) {
            Self::Own
        } else {
            Self::Unread
        }
    }
}

/// One open decision of a bound set on this thread, per phase.
#[derive(Default)]
struct Open {
    is_open: bool,
    /// Every stage of this decision takes its fallback.
    bypass: bool,
    count: u32,
    decision_id: u64,
    stamp: u64,
    /// The frame's deadline, as sent.
    frame_micros: u32,
    /// The stage bits the frame asks for (the profile's, among the
    /// plugin's); `0` before the frame is sent.
    asked: u32,
    /// The frame named the rows the profile's filters left (a pick-only
    /// ask): a pick outside them is the plugin's fault.
    named_survivors: bool,
    ticket: Option<Ticket>,
    read: bool,
    answer: Option<Answer>,
    /// The frame of a pick-only ask, built at the open (the picker is not
    /// handed the pool) and sent at the pick with the combined scores.
    pending: Option<proto::PluginDecide>,
}

impl Open {
    /// Forgets the decision.
    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// One open decision per phase, in the framework's phase slots: a pair
/// plan opens the prefill side and the decode side of one decision.
#[derive(Default)]
struct Opens([Open; 4]);

/// One engine build's hold on a plugin. The engine hands every lifecycle
/// event to the scorers and the picker of every profile, and a plugin bound
/// to several profiles hears it once: the bound sets a build takes from its
/// one binder share the build's observer, the first scorer or picker adapter
/// made from any of them. A later build over the same connected plugin (a
/// spec applied again) takes a binder of its own, so its adapters forward for
/// it while the earlier build stands and once it is gone.
#[derive(Debug)]
pub struct Binder {
    plugin: Arc<Plugin>,
    observer_taken: Arc<AtomicBool>,
}

impl Binder {
    /// The bound set of one profile of this build: the three adapters of
    /// the profile share it, and the engine opens it once per decision.
    pub fn bind(&self) -> Arc<Bound> {
        Arc::new(Bound {
            plugin: Arc::clone(&self.plugin),
            opens: ThreadLocal::new(),
            stages: AtomicU32::new(0),
            observer_taken: Arc::clone(&self.observer_taken),
            unimplemented_said: AtomicBool::new(false),
        })
    }
}

/// A plugin bound to one profile: its open decisions per thread and the
/// stages the profile asks of it.
pub struct Bound {
    plugin: Arc<Plugin>,
    opens: ThreadLocal<RefCell<Opens>>,
    /// The stage bits the profile binds this plugin as.
    stages: AtomicU32,
    /// The build's observer, shared with the build's other bound sets.
    observer_taken: Arc<AtomicBool>,
    /// The line of a stage bound and not implemented was written.
    unimplemented_said: AtomicBool,
}

impl fmt::Debug for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bound")
            .field("plugin", &self.plugin.inner.name)
            .finish_non_exhaustive()
    }
}

fn phase_slot(phase: Phase) -> usize {
    match phase {
        Phase::Single => 0,
        Phase::Prefill => 1,
        Phase::Decode => 2,
        Phase::Encode => 3,
    }
}

impl Bound {
    pub fn plugin(&self) -> &Arc<Plugin> {
        &self.plugin
    }

    fn state(&self, phase: Phase) -> RefMut<'_, Open> {
        RefMut::map(self.opens.get_or_default().borrow_mut(), |opens| {
            &mut opens.0[phase_slot(phase)]
        })
    }

    /// The first scorer or picker adapter made from any bound set of the
    /// build forwards the lifecycle; the others do not, so an event reaches
    /// the plugin once per build, whichever of its profiles bind the plugin.
    pub(crate) fn take_observer(&self) -> bool {
        !self.observer_taken.swap(true, Ordering::Relaxed)
    }

    /// A stage the profile binds and the plugin's ack leaves out takes its
    /// fallback on every decision, and no counter moves for it: one line a
    /// bound set names the mismatch, in the protocol's stage bits.
    #[cold]
    fn say_unimplemented(&self, bound_stages: u32, implemented_stages: u32) {
        if !self.unimplemented_said.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                plugin = self.plugin.inner.name,
                bound_stages,
                implemented_stages,
                "a plugin does not implement a stage its profile binds: the stage takes its fallback"
            );
        }
    }

    /// Sends the decision's frame for `asked` under what the decision has
    /// left now. What the frame costs is read when its answer is.
    fn send(
        &self,
        open: &mut Open,
        budget: &OpenDecision,
        mut frame: proto::PluginDecide,
        asked: u32,
    ) {
        let inner = &self.plugin.inner;
        let remaining = u32::try_from(budget.remaining_micros).unwrap_or(u32::MAX);
        if remaining == 0 {
            // A pick-only ask whose decision spent its budget since the open.
            bump(&inner.tally.budget_spent);
            open.bypass = true;
            return;
        }
        let frame_micros = inner.frame.frame_micros(remaining);
        view::stamp(&mut frame, asked, frame_micros, remaining);
        let lane = self.plugin.lane();
        match lane.send(open.decision_id, frame) {
            Ok(at) => {
                open.asked = asked;
                open.frame_micros = frame_micros;
                open.ticket = Some(Ticket {
                    decision_id: open.decision_id,
                    at,
                    deadline: at + Duration::from_micros(u64::from(frame_micros)),
                });
            }
            Err(SendError::NotReady) => {
                // The stream went between the open's look and the frame.
                lane.ensure_connecting(&inner.runtime);
                bump(&inner.tally.cold);
                open.bypass = true;
            }
            Err(SendError::Full) => {
                // The stream's window is full: no blame here, the frames
                // that fill it are judged by their own misses.
                bump(&inner.tally.refused);
                open.bypass = true;
            }
        }
    }

    /// Reads the frame's answer at the first stage that needs it: the wait
    /// ([`Self::wait`]) and the take ([`Self::take`]) over the ticket the
    /// send left. `shown` is the rows a pick-only frame named as surviving:
    /// a pick outside them breaks the contract like any other invalid
    /// answer.
    fn ensure_answer(&self, open: &mut Open, shown: Option<&[u32]>) {
        if open.read || open.bypass {
            return;
        }
        if self.plugin.inner.unit.failed_in(open.stamp) {
            // The plugin failed this decision after this frame was sent (the
            // other side of a pair answered outside the contract): its
            // answer is not read, so nothing counts twice, and the close
            // forgets the frame.
            open.bypass = true;
            return;
        }
        open.read = true;
        let Some(ticket) = open.ticket else {
            open.bypass = true;
            return;
        };
        let mut budget = self.plugin.inner.runtime.decisions().on_this_thread();
        let (started, wait) = self.wait(ticket, budget.remaining_micros);
        self.take(open, ticket, started, wait, &mut budget, shown);
    }

    /// The wait between a frame's send and its take: this thread blocks
    /// until the answer, the frame's deadline or the end of what the
    /// decision has left, whichever is first. Says when the wait began
    /// and how it ended.
    fn wait(&self, ticket: Ticket, remaining_micros: u64) -> (Instant, Wait) {
        let started = Instant::now();
        let until = ticket
            .deadline
            .min(started + Duration::from_micros(remaining_micros));
        let wait = self.plugin.lane().wait_answer(ticket.decision_id, until);
        (started, wait)
    }

    /// Takes what the wait over `ticket` ended with: the time this thread
    /// blocked is drawn from the decision's remainder (a missed frame
    /// draws its whole grant), and the round trip is the plugin's charge
    /// in the decision's latency sample. What a missed frame says of its
    /// plugin is one table, [`Miss::of`]. A late answer the thread finds
    /// without waiting is used and sampled at its whole flight: a miss is a
    /// frame the thread stopped waiting for.
    fn take(
        &self,
        open: &mut Open,
        ticket: Ticket,
        started: Instant,
        wait: Wait,
        budget: &mut OpenDecision,
        shown: Option<&[u32]>,
    ) {
        let inner = &self.plugin.inner;
        match wait {
            Wait::Answered(arrived, decision) => {
                budget.remaining_micros = budget
                    .remaining_micros
                    .saturating_sub(micros(arrived.saturating_duration_since(started)));
                if inner.runtime.witness().stalled_between(ticket.at, arrived) {
                    // The I/O thread stalled while this answer was owed:
                    // the round trip holds the host's time and is no
                    // sample. The answer is read all the same.
                    bump(&inner.tally.stalls);
                } else {
                    inner.unit.charge(
                        open.stamp,
                        micros(arrived.saturating_duration_since(ticket.at)).max(1),
                    );
                }
                let read = view::read(decision, open.asked, open.count, inner.declared).and_then(
                    |answer| match (shown, answer.pick) {
                        (Some(shown), Some(pick)) if !shown.contains(&pick) => {
                            Err(view::Invalid::Pick)
                        }
                        _ => Ok(answer),
                    },
                );
                match read {
                    Ok(answer) => {
                        for delta in &answer.counters {
                            inner.counters.add(delta.id, delta.by);
                        }
                        if answer.busy {
                            bump(&inner.tally.busy);
                        }
                        inner
                            .tally
                            .declined
                            .fetch_add(u64::from(answer.declined.count_ones()), Ordering::Relaxed);
                        if answer.declined != open.asked {
                            bump(&inner.tally.answered);
                        }
                        open.answer = Some(answer);
                    }
                    Err(why) => {
                        inner.invalid(why);
                        open.bypass = true;
                    }
                }
            }
            Wait::Missed => {
                // What the plugin was given: the frame as sent, or less
                // where the decision's remainder ended the wait before the
                // frame's own deadline.
                let before_the_wait = micros(started.saturating_duration_since(ticket.at));
                let given = u64::from(open.frame_micros)
                    .min(before_the_wait.saturating_add(budget.remaining_micros));
                // The frame's whole grant is drawn, whatever the verdict.
                budget.remaining_micros = budget
                    .remaining_micros
                    .saturating_sub(u64::from(open.frame_micros));
                bump(&inner.tally.missed);
                open.bypass = true;
                let stalled = inner
                    .runtime
                    .witness()
                    .stalled_between(ticket.at, Instant::now());
                // Whose bound the frame ran out under, asked where every
                // kind asks: the plugin's own where the frame had its whole
                // budget, or where the plugin's spend in the decision so
                // far and this grant reach the decision's budget.
                let seed = u32::try_from(budget.seed_micros).unwrap_or(u32::MAX);
                let whole = given == u64::from(inner.frame.frame_micros(seed));
                let spent = inner.unit.charged(open.stamp);
                let own = own_bound(whole, spent, given, budget.seed_micros);
                let miss = Miss::of(stalled, own, given);
                if miss != Miss::Held {
                    // The missed frame's part of the decision's sample is
                    // what it was given and used up: never the wait this
                    // thread measured, whose part past the deadline is the
                    // thread's own lateness.
                    inner.unit.charge(open.stamp, given.max(1));
                }
                match miss {
                    Miss::Own => {
                        // One failure of the decision, like a fault: its
                        // later opens send no frame.
                        bump(&inner.tally.own_misses);
                        inner.unit.fail(inner.now());
                    }
                    Miss::Held => {
                        // The host's: the frame falls back all the same.
                        bump(&inner.tally.stalls);
                        bump(&inner.tally.budget_spent);
                    }
                    Miss::Unread => {
                        bump(&inner.tally.unread);
                        bump(&inner.tally.budget_spent);
                    }
                    Miss::Clamped => bump(&inner.tally.budget_spent),
                }
            }
            Wait::Broken => {
                // Counted by the stream's own failure; this decision falls
                // back, and a later one asks for the stream again. The time
                // this thread blocked is drawn like an answered wait's: a
                // plugin asked later in the decision is asked under the
                // rest.
                budget.remaining_micros = budget
                    .remaining_micros
                    .saturating_sub(micros(started.elapsed()));
                open.bypass = true;
            }
        }
    }

    pub(crate) fn open(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        pool: &PoolFacts<'_>,
    ) {
        let mut open = self.state(facts.phase);
        if open.is_open {
            return;
        }
        open.reset();
        open.is_open = true;
        open.count = u32::try_from(candidates.len()).unwrap_or(u32::MAX);
        let inner = &self.plugin.inner;
        // The decision's budget is one a thread: its first open seeds it,
        // every open of the decision (each side, each plugin of every
        // runtime built over the same open decisions) draws on it.
        let mut budget = inner.runtime.decisions().on_this_thread();
        open.stamp = budget.open(facts.budget_micros);
        if inner.unit.enter(open.stamp) {
            // The plugin already failed this decision (the other side of a
            // pair, another binding): its part in the decision is over, with
            // no frame and no second count.
            open.bypass = true;
            return;
        }
        let now = inner.now();
        if inner.unit.is_open(now) {
            bump(&inner.tally.bypassed);
            open.bypass = true;
            return;
        }
        if facts.budget_micros == 0 {
            // The budget is gone on arrival: no frame, the breaker counts a
            // budget hit.
            bump(&inner.tally.budget_gone);
            inner.unit.fail(now);
            open.bypass = true;
            return;
        }
        if budget.remaining_micros == 0 {
            // An earlier open of this decision spent the remainder: no frame,
            // the stages fall back, the breaker is untouched.
            bump(&inner.tally.budget_spent);
            open.bypass = true;
            return;
        }
        if open.count > inner.max_candidates {
            // Whole view or none: the plugin is not asked.
            bump(&inner.tally.too_many);
            open.bypass = true;
            return;
        }
        let lane = self.plugin.lane();
        let Some(implemented) = lane.stages() else {
            // No decision waits for a connect: the stream opens on the I/O
            // thread behind this one, which takes its fallbacks at once.
            lane.ensure_connecting(&inner.runtime);
            bump(&inner.tally.cold);
            open.bypass = true;
            return;
        };
        let asked = self.stages.load(Ordering::Relaxed);
        let bound = asked & implemented;
        if bound != asked && !self.unimplemented_said.load(Ordering::Relaxed) {
            self.say_unimplemented(asked, implemented);
        }
        if bound == 0 {
            // The plugin implements none of the stages this profile binds.
            open.bypass = true;
            return;
        }
        open.decision_id = lane.next_id();
        let frame = view::decide(
            open.decision_id,
            facts,
            candidates,
            pool,
            &inner.exposure,
            inner.needs,
        );
        if bound & (FILTER | SCORE) == 0 {
            // The pick alone: the frame waits for the combined scores.
            open.pending = Some(frame);
        } else {
            // A filter or a scorer is asked at the open, so the native stages
            // run while the plugin computes; the pick rides the same frame.
            self.send(&mut open, &budget, frame, bound);
        }
    }

    pub(crate) fn close(&self) {
        let inner = &self.plugin.inner;
        let mut opens = self.opens.get_or_default().borrow_mut();
        let mut budget = inner.runtime.decisions().on_this_thread();
        // The scope is closed first: every open phase is forgotten and its
        // place in the thread's decision given back before a call that is
        // the host's own (the lane, the clock a build handed the runtime).
        // Whatever such a call does, no phase reads as open to the thread's
        // next decision, and that decision is seeded as its own.
        let closed = opens.0.each_mut().map(|open| {
            open.is_open.then(|| {
                // Sent and never read (the decision ended before a stage of
                // the plugin's ran): the answer, if it comes, is late.
                let unread = open.ticket.filter(|_| !open.read);
                let stamp = open.stamp;
                open.reset();
                budget.close();
                (unread, stamp)
            })
        });
        for (unread, stamp) in closed.into_iter().flatten() {
            if let Some(ticket) = unread {
                self.plugin.lane().forget(ticket.decision_id);
            }
            // The plugin's last close of the decision takes its sample: its
            // round trips in it, over the decision's budget.
            inner.unit.leave(stamp, budget.seed_micros, || inner.now());
        }
    }

    fn answer<'a>(&self, open: &'a mut Open, shown: Option<&[u32]>) -> Option<&'a Answer> {
        self.ensure_answer(open, shown);
        if open.bypass {
            return None;
        }
        open.answer.as_ref()
    }

    pub(crate) fn keep(&self, facts: &RequestFacts<'_>, position: u32) -> Result<bool, Fallback> {
        let mut open = self.state(facts.phase);
        if open.asked & FILTER == 0 {
            return Err(Fallback);
        }
        let answer = self.answer(&mut open, None).ok_or(Fallback)?;
        if !answer.mask_set {
            return Err(Fallback);
        }
        Ok(view::bit(&answer.mask, position))
    }

    pub(crate) fn retain(
        &self,
        facts: &RequestFacts<'_>,
        kept: &mut Vec<u32>,
    ) -> Result<(), Fallback> {
        let mut open = self.state(facts.phase);
        if open.asked & FILTER == 0 {
            return Err(Fallback);
        }
        let answer = self.answer(&mut open, None).ok_or(Fallback)?;
        if !answer.mask_set {
            return Err(Fallback);
        }
        kept.retain(|&i| view::bit(&answer.mask, i));
        Ok(())
    }

    pub(crate) fn score(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        out: &mut [f32],
    ) -> Result<(), Fallback> {
        let mut open = self.state(facts.phase);
        if open.asked & SCORE == 0 {
            return Err(Fallback);
        }
        let answer = self.answer(&mut open, None).ok_or(Fallback)?;
        if answer.scores.is_empty() {
            return Err(Fallback);
        }
        for (o, &i) in out.iter_mut().zip(kept) {
            *o = answer.scores.get(i as usize).copied().unwrap_or(0.0);
        }
        Ok(())
    }

    /// The picker's answer over the survivors (`scores[i]` belongs to
    /// `kept[i]`). A plugin asked for the pick alone is asked here, its
    /// frame carrying one combined score per candidate, `-1.0` on a row the
    /// filters dropped: a pick outside the rows it was shown is its fault,
    /// judged when the answer is read. A plugin asked at the open picked
    /// over the whole view: a pick the profile's filters dropped since is no
    /// fault, the picker falls back.
    pub(crate) fn pick(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        scores: &[f32],
    ) -> Result<u32, Fallback> {
        let mut open = self.state(facts.phase);
        if let Some(mut frame) = open.pending.take() {
            if self.plugin.inner.unit.failed_in(open.stamp) {
                // The plugin failed this decision since the open (the other
                // side of a pair answered outside the contract): its part
                // in the decision is over, and this side sends no frame.
                open.bypass = true;
            } else {
                frame.scores_in.resize(open.count as usize, -1.0);
                for (&i, &s) in kept.iter().zip(scores) {
                    if let Some(slot) = frame.scores_in.get_mut(i as usize) {
                        // A surviving row never reads below zero.
                        *slot = s.max(0.0);
                    }
                }
                open.named_survivors = true;
                let budget = self.plugin.inner.runtime.decisions().on_this_thread();
                self.send(&mut open, &budget, frame, PICK);
            }
        }
        if open.asked & PICK == 0 {
            return Err(Fallback);
        }
        let shown = open.named_survivors.then_some(kept);
        let position = self
            .answer(&mut open, shown)
            .ok_or(Fallback)?
            .pick
            .ok_or(Fallback)?;
        if kept.contains(&position) {
            Ok(position)
        } else {
            Err(Fallback)
        }
    }
}

/// This open marks its phase open and counts the decision before it reads
/// the runtime's clock and builds its frame, so it runs under the guard
/// that undoes an open that stops half-way.
impl DecisionScope for Bound {
    fn open(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], pool: &PoolFacts<'_>) {
        let mut half_open = HalfOpen::new(self);
        Bound::open(self, facts, candidates, pool);
        half_open.returned();
    }

    fn close(&self) {
        Bound::close(self);
    }
}

/// The shared stage adapters read the bound decision through this: the gRPC
/// kind's answers, named by the plugin.
impl BoundPlugin for Bound {
    fn name(&self) -> &'static str {
        self.plugin.name()
    }

    fn failure(&self) -> Failure {
        self.plugin.failure()
    }

    fn needs(&self) -> Needs {
        self.plugin.needs()
    }

    fn observes(&self) -> EventSet {
        self.plugin.observes()
    }

    fn take_observer(&self) -> bool {
        Bound::take_observer(self)
    }

    fn bound_to(&self, stage: Stage) {
        let bit = match stage {
            Stage::Filter => FILTER,
            Stage::Score => SCORE,
            Stage::Pick => PICK,
        };
        self.stages.fetch_or(bit, Ordering::Relaxed);
    }

    fn keep(&self, facts: &RequestFacts<'_>, position: u32) -> Result<bool, Fallback> {
        Bound::keep(self, facts, position)
    }

    fn retain(&self, facts: &RequestFacts<'_>, kept: &mut Vec<u32>) -> Result<(), Fallback> {
        Bound::retain(self, facts, kept)
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        out: &mut [f32],
    ) -> Result<(), Fallback> {
        Bound::score(self, facts, kept, out)
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        scores: &[f32],
    ) -> Result<u32, Fallback> {
        Bound::pick(self, facts, kept, scores)
    }

    fn observe_issued(&self, lease: &Lease, decision: &Decision) {
        let inner = &self.plugin.inner;
        inner.observe(view::issued(lease, decision, &inner.exposure));
    }

    fn observe_dispatched(
        &self,
        lease: &Lease,
        _target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.plugin
            .inner
            .observe(view::dispatched(lease, side, since_decision_micros));
    }

    fn observe_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.plugin.inner.observe(view::completed(lease, outcome));
    }

    fn observe_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.plugin.inner.observe(view::released(lease, reason));
    }

    fn observe_membership(&self, change: &Membership<'_>) {
        if let Some(event) = view::membership(change) {
            self.plugin.inner.observe(event);
        }
    }

    fn observe_signal(&self, signal: &Signal) {
        let inner = &self.plugin.inner;
        inner.observe(view::signal(signal, inner.now()));
    }
}

#[cfg(test)]
mod tests {
    use super::Miss;
    use crate::grpc::runtime::PROMISE_MICROS;

    /// The table of a miss, row by row: the witness's excuse first, then
    /// whose bound it was, then whether the witness can vouch for a flight
    /// that long.
    #[test]
    fn the_table_of_a_miss_row_by_row() {
        const LONG: u64 = PROMISE_MICROS + 1;
        // Stalled, the plugin's own bound, the flight given, the verdict.
        let table = [
            (true, true, LONG, Miss::Held),
            (true, false, LONG, Miss::Held),
            (true, true, 1, Miss::Held),
            (false, false, LONG, Miss::Clamped),
            (false, false, 1, Miss::Clamped),
            (false, true, LONG, Miss::Own),
            // No longer than the promise: the thread cannot be seen late.
            (false, true, PROMISE_MICROS, Miss::Unread),
            (false, true, 1, Miss::Unread),
        ];
        for (stalled, own_bound, given, verdict) in table {
            assert_eq!(
                Miss::of(stalled, own_bound, given),
                verdict,
                "stalled {stalled}, own {own_bound}, given {given}"
            );
        }
    }
}
