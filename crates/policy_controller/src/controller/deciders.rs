//! The deciders (design 5.5): the threads the controller's decisions run
//! on, each behind one bounded queue.
//!
//! A model's decisions all run on one decider ([`Deciders::of`]): the model
//! is the key, so what a model's decisions write has one writer, and the
//! requests a stream sends for one model are decided in the order it sent
//! them. No runtime thread decides and none waits for a decision: a
//! stream's reader hands a request over ([`Deciders::submit`]) or is told
//! at once that the queue is full, and the answer comes back on the
//! stream's own queue.
//!
//! One invariant: **an answer leaves under the term it was made under, or
//! it does not leave.** A decider reads the fence twice: before it builds
//! anything, and again where it hands its answer over. A fence that closed
//! in between, or stands open under another term, sends `Busy` in the
//! decision's place.
//!
//! One request, in this order:
//!
//! 1. its decider's queue is full: the request comes back to its submitter,
//!    whose answer is `Busy` for the queue's depth;
//! 2. the fence is closed when the decider takes the request: `Busy`, and
//!    nothing is built. It is read before the budget: a replica that does
//!    not lead says so whatever else holds of the request, since that
//!    reason still holds for the next one;
//! 3. the budget, counted from the frame's receipt, is gone, or under one
//!    microsecond of it is left: `Busy`, and nothing is read for it;
//! 4. one load of the world ([`MemoryStore::deciding`]): the pool's rows as
//!    the asking gateway reaches them, less the members that share no wire
//!    with it where it states its wires, and the engine of the generation
//!    those rows were published under;
//! 5. the engine's decision, told what is left of the budget;
//! 6. the fence again, and the budget at the same read: a decision whose
//!    budget ended while it was made leaves as `Busy` under the reason a
//!    late arrival is told, and under a count of its own ([`Counts`]).
//!
//! A gateway's wires are a fact before the profile, as its vetoes are: no
//! filter of the catalog reads them, and no profile picks a member its
//! gateway cannot dial. The candidates the wires take are counted, and a
//! decision whose pool has rows and whose gateway shares a wire with none
//! of them is shed under [`WIRES`]. One whose gateway hints a subset of 0,
//! a decision over no candidates, is shed under [`SUBSET_HINT`]. A positive
//! hint is read by no stage yet.
//!
//! What this build has no decision for is answered `Busy` with no reason,
//! and its gateway decides it locally at once: a shed is a policy's word
//! and reaches the client, and no policy spoke. Three arms, each counted
//! by its cause ([`Counts`]): a decision that panics, a request for a side
//! of a pair plan (prefill or decode under either pair plan, encode under
//! the plan that has an encode side: the pair door decides those, over
//! the store's pair classes), and a request that carries its prompt as
//! text (the controller is to hash it, and a decision made as if the
//! request carried no prefix would be another request's). A text is said
//! right after the fence and before the budget is read, so the count
//! holds every text a leader took.
//!
//! A decision that panics costs its own request alone. Its decider frees
//! the panic's payload under a catch of its own, since a payload's drop
//! can panic too, takes the next request on new buffers, so that one reads
//! nothing the fault left, and writes one line for a streak of faults,
//! never one a request.
//! A decider's thread that unwinds anywhere else leaves its models with no
//! decider: the deciders tell it ([`Deciders::ended`]), for the controller
//! to stop, since a leader that cannot decide for a model must not keep
//! the lead.
//!
//! What this step leaves to the next ones. No lease is issued: an answer
//! carries the engine's decision and nothing is recorded for it. A view
//! reads no session pin of the store's.

use std::{
    io,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, SyncSender, TrySendError},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use tokio::sync::{mpsc, watch};
use tracing::error;

use super::{
    election::Standing,
    wire::{self, Checked, Lists, Viewed},
};
use crate::{
    decision::{Decision, Shed, ShedReason, Trace},
    elector::Fence,
    framework::{release, Plan, Scratch},
    proto::{self, busy::Reason},
    store::{Ask, GatewayId, MemoryStore, ViewBuffers, World},
    view::{fingerprint, Candidate, Constraints, Phase, PoolFacts, RequestFacts, Role, WireModes},
};

/// What the deciders count on the controller's own side: the requests this
/// build has no decision for, by cause, the requests whose block hashes
/// were cut, the answers no stream took, the candidates a gateway's wires
/// took from its decisions, and the decisions whose budget ended while they
/// were made. The step that serves the call renders them.
#[derive(Debug, Default)]
pub(crate) struct Counts {
    /// Decisions that panicked.
    pub(crate) faults: AtomicU64,
    /// Requests for a side of a pair plan: prefill or decode under either
    /// pair plan, encode under the plan that has an encode side.
    pub(crate) pair_sides: AtomicU64,
    /// Requests that carried their prompt as text.
    pub(crate) texts: AtomicU64,
    /// Requests that carried more block hashes than their generation's
    /// view reads.
    pub(crate) hashes_cut: AtomicU64,
    /// Answers a stream's queue had no place for: its gateway waits its
    /// budget out for each.
    pub(crate) dropped_full: AtomicU64,
    /// Answers whose stream had ended.
    pub(crate) dropped_gone: AtomicU64,
    /// Candidates the wires of their decision's gateway took from its
    /// view, members that share no wire with it: each decider's part, read
    /// whole by [`Counts::undialed`].
    undialed: [Part; PARTS],
    /// Decisions whose budget ended while they were made. Each leaves as
    /// `Busy` under the reason a late arrival is told, and this count is
    /// of them alone.
    pub(crate) outlasted: AtomicU64,
}

/// The parts the wires' count is kept in. A decider adds to the one at its
/// index, and deciders past this many share one.
const PARTS: usize = 16;

/// One decider's part of a count that every decision can move, on cache
/// lines of its own. Where a fleet's gateways speak fewer wires than its
/// members, each decision counts the candidates it lost, and one count for
/// every decider would be a line they all write.
#[derive(Debug, Default)]
#[repr(align(128))]
struct Part(AtomicU64);

impl Counts {
    /// The part decider `index` counts in.
    fn undialed_of(&self, index: usize) -> &AtomicU64 {
        &self.undialed[index % PARTS].0
    }

    /// Candidates the wires took, over every decider.
    pub(crate) fn undialed(&self) -> u64 {
        let parts = self.undialed.iter();
        parts.map(|part| part.0.load(Ordering::Relaxed)).sum()
    }
}

/// The stage a decision is shed under when its pool has rows and its
/// gateway shares a wire with none of them.
pub(crate) const WIRES: &str = "wires";

/// The stage a decision is shed under when its gateway hints a subset of 0
/// of a pool that has rows: a decision over no candidates, as it asked.
pub(crate) const SUBSET_HINT: &str = "subset-hint";

/// The clock a decider reads, twice a decision.
pub(crate) type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// What every decider reads.
#[derive(Clone)]
pub(crate) struct Reads {
    pub(crate) store: Arc<MemoryStore>,
    /// The controller's one fence.
    pub(crate) fence: Arc<Fence>,
    /// The election's last sight of the lease, for the term a `Busy`
    /// carries.
    pub(crate) standing: watch::Receiver<Standing>,
    pub(crate) clock: Clock,
    pub(crate) counts: Arc<Counts>,
}

/// One request on its way to its decider.
pub(crate) struct Job {
    pub(crate) request: Checked,
    /// The gateway the request's stream is: the view is built as that
    /// gateway reaches the pool, under its own vetoes.
    pub(crate) gateway: GatewayId,
    /// When the stream's reader took the frame: the budget counts from it.
    pub(crate) received: Instant,
    /// The stream's own queue of answers. A decider never waits on it: an
    /// answer a full or a closed queue does not take is dropped, and
    /// counted.
    pub(crate) answers: mpsc::Sender<proto::DecideAnswer>,
}

/// Why a request took no place in its decider's queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The queue holds its depth.
    Full,
    /// The decider's thread is gone.
    Gone,
}

/// The deciders. Their drop ends each thread, after the requests its queue
/// still holds.
pub(crate) struct Deciders {
    /// Each decider's queue, at its index.
    queues: Box<[SyncSender<Job>]>,
    /// Shows `true` once a decider's thread unwound.
    told: Arc<watch::Sender<bool>>,
}

/// A decider's thread holds one: it tells when the thread unwinds. A thread
/// that returns, at the queue's end, tells nothing.
struct Unwound(Arc<watch::Sender<bool>>);

impl Drop for Unwound {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.send_replace(true);
        }
    }
}

impl Deciders {
    /// Starts `deciders` threads, each behind a queue of `depth` requests:
    /// at least one thread, and a queue of at least one.
    pub(crate) fn start(deciders: usize, depth: usize, reads: &Reads) -> io::Result<Self> {
        let told = Arc::new(watch::Sender::new(false));
        let queues = (0..deciders.max(1))
            .map(|index| {
                let (queue, jobs) = sync_channel(depth.max(1));
                let decider = Decider {
                    reads: reads.clone(),
                    buffers: Buffers::default(),
                    faulted: false,
                    index,
                };
                watched(index, &told, move || decider.run(&jobs)).map(|_running| queue)
            })
            .collect::<io::Result<_>>()?;
        Ok(Self { queues, told })
    }

    /// Shows `true` once a decider's thread unwound outside a decision: its
    /// models have no decider from then on.
    pub(crate) fn ended(&self) -> watch::Receiver<bool> {
        self.told.subscribe()
    }

    /// What a decider's thread tells at its unwind, told by a case.
    #[cfg(test)]
    pub(crate) fn tell_unwound(&self) {
        self.told.send_replace(true);
    }

    /// The decider `model`'s decisions run on: every request for a model
    /// goes to one, whichever stream carries it.
    pub(crate) fn of(&self, model: &str) -> usize {
        (fingerprint(model.as_bytes()) % self.queues.len() as u64) as usize
    }

    /// Hands `job` to its model's decider, without waiting. `Some` is the
    /// request back with why it took no place: its submitter answers it.
    pub(crate) fn submit(&self, job: Job) -> Option<(Refused, Job)> {
        match self.queues[self.of(job.request.model())].try_send(job) {
            Ok(()) => None,
            Err(TrySendError::Full(job)) => Some((Refused::Full, job)),
            Err(TrySendError::Disconnected(job)) => Some((Refused::Gone, job)),
        }
    }
}

/// What a decider keeps across decisions: the engine's run of a decision
/// allocates only while these grow. An answer is built anew a request: a
/// pick's targets and its trace's top are each a list of its own.
#[derive(Default)]
struct Buffers {
    view: ViewBuffers,
    scratch: Scratch,
    lists: Lists,
}

/// Starts decider `index`'s thread. The thread holds the guard that tells
/// its unwind, whatever it runs.
fn watched(
    index: usize,
    told: &Arc<watch::Sender<bool>>,
    run: impl FnOnce() + Send + 'static,
) -> io::Result<thread::JoinHandle<()>> {
    let unwound = Unwound(Arc::clone(told));
    thread::Builder::new()
        .name(format!("decide-{index}"))
        .spawn(move || {
            let _unwound = unwound;
            run();
        })
}

/// One deciding thread.
struct Decider {
    reads: Reads,
    buffers: Buffers,
    /// The last decision panicked: a streak of them writes one line.
    faulted: bool,
    /// Its place among the deciders, which names its part of a count.
    index: usize,
}

impl Decider {
    /// Decides until the last handle of the queue is dropped.
    fn run(mut self, jobs: &Receiver<Job>) {
        while let Ok(job) = jobs.recv() {
            let Job {
                request,
                gateway,
                received,
                answers,
            } = job;
            let answer = self.answer(request, gateway, received);
            let dropped = match answers.try_send(answer) {
                Ok(()) => continue,
                Err(mpsc::error::TrySendError::Full(_)) => &self.reads.counts.dropped_full,
                Err(mpsc::error::TrySendError::Closed(_)) => &self.reads.counts.dropped_gone,
            };
            dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One request's answer: its decision, or `Busy`.
    fn answer(
        &mut self,
        request: Checked,
        gateway: GatewayId,
        received: Instant,
    ) -> proto::DecideAnswer {
        // What the claim rests on: a decision writes neither the fence nor
        // the store, the engine closes an open scope where it unwinds, and
        // the buffers a fault leaves are dropped whole below; nothing here
        // holds a lock a panic poisons.
        let decided = catch_unwind(AssertUnwindSafe(|| {
            self.decide(&request, gateway, received)
        }));
        let faulted = decided.is_err();
        let answer = match decided {
            Ok(Ok((term, decision))) => wire::decided(request.into_id(), term, &decision),
            Ok(Err(reason)) => wire::busy(request.into_id(), self.known_term(), reason),
            Err(payload) => {
                // A payload's own drop can panic, and that panic let out
                // here would unwind this thread with the request
                // unanswered: the engine's helper frees it under a catch
                // of its own.
                release(payload);
                // The next decision reads nothing of this one: the view,
                // the scratch and the lists stay where the fault found
                // them, and are dropped whole.
                self.buffers = Buffers::default();
                self.reads.counts.faults.fetch_add(1, Ordering::Relaxed);
                if !self.faulted {
                    error!(
                        decider = thread::current().name().unwrap_or_default(),
                        "a decision panicked: its request is answered Busy with no reason, and so is each one that panics after it until one does not"
                    );
                }
                wire::busy(request.into_id(), self.known_term(), Reason::Unspecified)
            }
        };
        self.faulted = faulted;
        answer
    }

    /// One request's decision with the term both reads of the fence showed,
    /// or the reason its answer is `Busy`.
    fn decide(
        &mut self,
        request: &Checked,
        gateway: GatewayId,
        received: Instant,
    ) -> Result<(u64, Decision), Reason> {
        let Reads {
            store,
            fence,
            clock,
            counts,
            ..
        } = &self.reads;
        let now = clock();
        // The fence, read first: a replica that does not lead builds
        // nothing, and says so before any other reason.
        let term = fence.leader_term(now).ok_or(Reason::NotLeader)?;
        if request.has_text() {
            counts.texts.fetch_add(1, Ordering::Relaxed);
            return Err(Reason::Unspecified);
        }
        let taken =
            Taken::at(request, gateway, received, now).ok_or(Reason::BudgetGoneOnArrival)?;
        let (buffers, undialed) = (&mut self.buffers, counts.undialed_of(self.index));
        let decided = store.deciding(|world| decide_in(world, &taken, buffers, counts, undialed));
        let Some(decision) = decided else {
            counts.pair_sides.fetch_add(1, Ordering::Relaxed);
            return Err(Reason::Unspecified);
        };
        // The fence, read again where the answer is handed over: it leaves
        // under the term it was made under, or `Busy` leaves in its place.
        // The budget is judged at that read too, with no clock read more:
        // a decision late by this clock since the receipt is late at its
        // gateway, whose clock started earlier. It leaves as `Busy` under
        // the reason a late arrival is told, and under a count of its own.
        let handed = clock();
        if fence.leader_term(handed) != Some(term) {
            return Err(Reason::NotLeader);
        }
        if taken.outlasted(handed) {
            counts.outlasted.fetch_add(1, Ordering::Relaxed);
            return Err(Reason::BudgetGoneOnArrival);
        }
        Ok((term, decision))
    }

    /// The highest term this replica knows, which a `Busy` carries: no
    /// answer of a replica falls in term, leading or not.
    fn known_term(&self) -> u64 {
        let seen = self.reads.standing.borrow().term;
        seen.max(self.reads.fence.term())
    }
}

/// A request as its decider takes it.
#[derive(Clone, Copy)]
struct Taken<'a> {
    request: &'a Checked,
    gateway: GatewayId,
    /// The decider's first read of its clock: the decision's instant.
    now: Instant,
    /// What is left of the request's budget at that instant.
    left: Duration,
}

impl<'a> Taken<'a> {
    /// `request` as its decider takes it at `now`; `None` when its budget
    /// is gone. The budget counts from the frame's receipt: what the queue
    /// used of it is not the decision's. Under one microsecond left is
    /// gone: the engine is told what is left in whole microseconds, and
    /// reads 0 as none.
    fn at(
        request: &'a Checked,
        gateway: GatewayId,
        received: Instant,
        now: Instant,
    ) -> Option<Self> {
        let left = request
            .budget()
            .saturating_sub(now.saturating_duration_since(received));
        (left.as_micros() > 0).then_some(Self {
            request,
            gateway,
            now,
            left,
        })
    }

    /// Whether the budget is gone at `later`, a read of the clock after the
    /// one the request was taken at: what was left then, less the time
    /// since, under the same rule. One difference of two instants, where
    /// taking the request again would read its budget and build it whole.
    fn outlasted(&self, later: Instant) -> bool {
        let since = later.saturating_duration_since(self.now);
        self.left.saturating_sub(since) < Duration::from_micros(1)
    }
}

/// Whether `phase` is a side of `plan` that the pair door decides: prefill
/// and decode under either pair plan, and encode under the plan that names
/// an encode side. That side rides the pair: the engine's entry keeps the
/// encoders the pair is compatible with, a bit the store sets on no row
/// yet, so it would shed each encode request as if no encoder stood.
fn pair_side(plan: Plan, phase: Phase) -> bool {
    match plan {
        Plan::Single => false,
        Plan::PdPair => matches!(phase, Phase::Prefill | Phase::Decode),
        Plan::Epd { .. } => !matches!(phase, Phase::Single),
    }
}

/// The engine's decision over the world of one load: the rows and the
/// engine are of one publication. `None` for a request for a side of a
/// pair plan ([`pair_side`]): the pair door decides those, and one side's
/// entry of the engine never does.
fn decide_in(
    world: &World,
    taken: &Taken<'_>,
    buffers: &mut Buffers,
    counts: &Counts,
    undialed: &AtomicU64,
) -> Option<Decision> {
    let spec = world.spec();
    let engine = spec.engine();
    let (model, phase) = (taken.request.model(), taken.request.phase());
    // A single request is no plan's side: its binding is looked up once, by
    // the engine.
    if !matches!(phase, Phase::Single)
        && engine
            .binding(model, phase)
            .is_some_and(|binding| pair_side(binding.plan, phase))
    {
        return None;
    }
    if taken.request.hashes() > spec.max_block_hashes() {
        counts.hashes_cut.fetch_add(1, Ordering::Relaxed);
    }
    Some(hand(
        world,
        taken,
        buffers,
        undialed,
        |facts, pool, candidates, constraints, scratch| {
            if candidates.is_empty() && pool.pool_size > 0 {
                // What the gateway states of itself left none of a pool
                // that has rows: said under that constraint's name, where
                // the engine would name no stage.
                let emptied_by = if constraints.subset_hint == Some(0) {
                    SUBSET_HINT
                } else {
                    WIRES
                };
                return shed(
                    ShedReason::NoEndpoints,
                    emptied_by,
                    facts.seed,
                    pool.generation,
                );
            }
            engine.decide(facts, pool, candidates, constraints, scratch)
        },
    ))
}

/// A shed the deciders say in the engine's place, in the shape of the
/// engine's own.
fn shed(reason: ShedReason, emptied_by: &'static str, seed: u64, generation: u64) -> Decision {
    Decision::Shed(Shed {
        reason,
        emptied_by,
        retry_after_secs: 1,
        wait_allowed: false,
        trace: Trace {
            seed,
            generation,
            ..Trace::default()
        },
    })
}

/// Hands `decide` what the engine reads for one request: the request with
/// what is left of its budget, the pool's facts under the generation its
/// rows were published under, and those rows as the asking gateway reaches
/// them at the decision's instant, less what the gateway states of itself
/// takes ([`narrow`]), with the affinity column where the request's
/// profile declares it. The rows its wires take are counted in `undialed`,
/// the decider's own part, and the request's block hashes are cut at that
/// generation's cap.
fn hand<R>(
    world: &World,
    taken: &Taken<'_>,
    buffers: &mut Buffers,
    undialed: &AtomicU64,
    decide: impl FnOnce(
        &RequestFacts<'_>,
        &PoolFacts<'_>,
        &[Candidate],
        &Constraints<'_>,
        &mut Scratch,
    ) -> R,
) -> R {
    let Taken {
        request,
        gateway,
        now,
        left,
    } = *taken;
    let spec = world.spec();
    let (model, phase) = (request.model(), request.phase());
    let ask = Ask {
        gateway,
        now,
        pinned: None,
        affinity: spec.engine().needs_for(model, phase).affinity,
    };
    let role = match phase {
        Phase::Single => Role::Single,
        Phase::Prefill => Role::Prefill,
        Phase::Decode => Role::Decode,
        Phase::Encode => Role::Encode,
    };
    let Buffers {
        view,
        scratch,
        lists,
    } = buffers;
    let (pool, now_micros) = match world.build(model, role, &ask, view) {
        Some(built) => {
            // The pool's facts are of the rows before the request's own
            // narrowing.
            let took = narrow(view, request.wires(), request.subset());
            if took > 0 {
                undialed.fetch_add(took, Ordering::Relaxed);
            }
            (built.pool_facts(), built.now_micros)
        }
        // No member serves the model in the role: the engine answers over
        // no candidate, as it answers any host that hands it none.
        None => (
            PoolFacts {
                generation: spec.generation(),
                ..PoolFacts::default()
            },
            world.micros(now),
        ),
    };
    let viewed = Viewed {
        now_micros,
        budget_micros: u32::try_from(left.as_micros()).unwrap_or(u32::MAX),
        affinity: &view.affinity,
        max_block_hashes: spec.max_block_hashes(),
    };
    request.read(lists, viewed, |facts, constraints| {
        decide(facts, &pool, &view.candidates, constraints, scratch)
    })
}

/// Narrows a built view by what the request's gateway states of itself,
/// and returns how many rows its wires took. A hint of 0 keeps none, and
/// leaves the wires nothing to take. A gateway that states wires keeps the
/// members that share one with it: a member that states none is taken as
/// one that states another is, and a gateway that states none keeps every
/// row. A positive hint cuts nothing. The kept rows keep their order and
/// take their positions, and the affinity column follows them. A request
/// that states neither pays the two compares.
fn narrow(view: &mut ViewBuffers, wires: Option<WireModes>, subset: Option<u32>) -> u64 {
    if subset == Some(0) {
        view.candidates.clear();
        view.affinity.clear();
        return 0;
    }
    let Some(wires) = wires else {
        return 0;
    };
    let rows = view.candidates.len();
    let dialed = |row: &Candidate| row.wire_modes.intersects(wires);
    // The column is by row, where the profile declared one: it is cut by
    // the rows' wires before the rows are.
    let mut beside = view.candidates.iter();
    view.affinity.retain(|_| beside.next().is_some_and(dialed));
    view.candidates.retain(dialed);
    let took = rows - view.candidates.len();
    if took > 0 {
        for (row, position) in view.candidates.iter_mut().zip(0..) {
            row.position = position;
        }
    }
    took as u64
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeSet, VecDeque},
        io, mem,
        panic::resume_unwind,
        sync::{
            atomic::{AtomicU64, Ordering},
            mpsc as std_mpsc, Arc,
        },
        thread,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;
    use tokio::{
        sync::{mpsc, watch},
        time::timeout,
    };

    use super::{
        hand, narrow, watched, Buffers, Clock, Counts, Decider, Deciders, Job, Reads, Refused,
        Taken, PARTS,
    };
    use crate::{
        controller::{
            election::Standing,
            gateways::tests::PATIENT,
            wire::{self, Checked},
        },
        elector::{Elector, Tick},
        framework::Scratch,
        proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome},
        spec::{AppliedSpec, PolicySpec},
        store::{
            Ask, GatewayId, HealthSink, MemoryStore, ViewBuffers, WorkerFacts, WorkerStatus,
            WorkersStore,
        },
        view::{
            AffinityFacts, Candidate, Constraints, PoolFacts, RequestFacts, Role, WireModes,
            WorkerId,
        },
    };

    /// How long a case waits for a decider before it fails.
    const PATIENCE: Duration = Duration::from_secs(20);
    const MODEL: &str = "model-a";
    const GATEWAY: GatewayId = GatewayId(11);
    const OTHER: GatewayId = GatewayId(12);
    /// The budget a case's frame gives its decision.
    const BUDGET: Duration = Duration::from_secs(5);
    /// Past the renew deadline of the bed's election.
    const FENCED: Duration = Duration::from_secs(1_801);

    /// What one read of the case's clock does before it shows the hour.
    enum Step {
        /// Shows the taken instant plus this much, from this read on.
        At(Duration),
        /// Tells the case a decider reached this read, then waits for the
        /// case's word.
        Hold {
            reached: std_mpsc::SyncSender<()>,
            go: std_mpsc::Receiver<()>,
        },
        Panic,
        /// Panics with a payload whose own drop panics.
        Bomb,
    }

    /// A panic's payload whose own drop panics.
    struct Bomb;

    impl Drop for Bomb {
        fn drop(&mut self) {
            panic!("the case's own panic, at a payload's drop");
        }
    }

    /// The clock of a case: each read takes the script's next step, and a
    /// read past the script shows what the last one showed. It keeps the
    /// name of every thread that read it.
    struct Script {
        /// When the bed's candidate took its lease.
        taken: Instant,
        steps: Mutex<VecDeque<Step>>,
        shown: Mutex<Duration>,
        readers: Mutex<Vec<String>>,
    }

    impl Script {
        fn read(&self) -> Instant {
            let reader = thread::current().name().unwrap_or_default().to_owned();
            self.readers.lock().push(reader);
            let step = self.steps.lock().pop_front();
            match step {
                Some(Step::At(offset)) => *self.shown.lock() = offset,
                Some(Step::Hold { reached, go }) => {
                    reached.send(()).expect("the case waits for this read");
                    go.recv_timeout(PATIENCE)
                        .expect("the case lets the read go on");
                }
                Some(Step::Panic) => panic!("the case's own panic, inside a decision"),
                Some(Step::Bomb) => resume_unwind(Box::new(Bomb)),
                None => {}
            }
            self.taken + *self.shown.lock()
        }
    }

    /// A store, a candidate of its election and the deciders over both.
    struct Bed {
        elector: Elector,
        standing: watch::Sender<Standing>,
        store: Arc<MemoryStore>,
        script: Arc<Script>,
        /// What the bed's deciders read, for a decider of the case's own.
        reads: Reads,
        counts: Arc<Counts>,
        deciders: Deciders,
        /// The members serving [`MODEL`] in the single role, ready.
        workers: Vec<WorkerId>,
        sender: mpsc::Sender<proto::DecideAnswer>,
        answers: mpsc::Receiver<proto::DecideAnswer>,
    }

    fn applied(document: &PolicySpec) -> AppliedSpec {
        AppliedSpec::validate(document).expect("the document validates")
    }

    /// A preset of this build as generation `generation`.
    fn preset(name: &str, generation: u64) -> AppliedSpec {
        let mut document = PolicySpec::preset(name).expect("a preset of this build");
        document.generation = generation;
        applied(&document)
    }

    /// The member `name`, serving `model` in `role` and stating `wires`,
    /// ready: written again where it stood, with its status as it was.
    fn ready(
        store: &MemoryStore,
        name: &str,
        model: &str,
        role: Role,
        wires: WireModes,
    ) -> WorkerId {
        let key = format!("http://{name}:8000");
        let facts = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role,
            wire_modes: wires,
            models: vec![model.to_owned()],
            dp_size: 1,
            max_running: Some(4),
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        let row = WorkersStore::get(store, id).expect("the row");
        if row.status != WorkerStatus::Ready {
            store
                .set_status(id, WorkerStatus::Ready, row.revision)
                .expect("the row stands");
        }
        id
    }

    /// Three ready members serving [`MODEL`] in the single role, and one in
    /// each other role beside the phase that role serves. None states a
    /// wire.
    fn fleet(store: &MemoryStore) -> (Vec<WorkerId>, [(proto::Phase, WorkerId); 3]) {
        let member = |name: &str, role: Role| ready(store, name, MODEL, role, WireModes::NONE);
        let singles = ["w0", "w1", "w2"].map(|name| member(name, Role::Single));
        let sides = [
            (proto::Phase::Prefill, member("p0", Role::Prefill)),
            (proto::Phase::Decode, member("d0", Role::Decode)),
            (proto::Phase::Encode, member("e0", Role::Encode)),
        ];
        (singles.to_vec(), sides)
    }

    /// The fleet under `spec` (the unbound generation with none), a
    /// candidate that took its lease at the script's instant under term 1
    /// when it `leads`, and `deciders` deciders behind queues of `depth`
    /// whose clock runs `steps`.
    fn bed(
        spec: Option<AppliedSpec>,
        leads: bool,
        (deciders, depth): (usize, usize),
        steps: impl IntoIterator<Item = Step>,
    ) -> Bed {
        let store = Arc::new(MemoryStore::new());
        let (workers, _sides) = fleet(&store);
        if let Some(spec) = spec {
            store.apply(spec).expect("above the unbound start");
        }
        // The bed's lease is its own, handed to nobody: an elector alone,
        // which leads at its take.
        let mut elector = Elector::alone("controller-0@10.0.0.7:30100", PATIENT)
            .expect("the settings are in order");
        let taken = Instant::now();
        if leads {
            let tick = elector.tick(taken).expect("the take");
            assert_eq!(tick, Tick::Acquired { term: 1 });
        }
        let script = Arc::new(Script {
            taken,
            steps: Mutex::new(steps.into_iter().collect()),
            shown: Mutex::new(Duration::ZERO),
            readers: Mutex::default(),
        });
        let clock: Clock = {
            let script = Arc::clone(&script);
            Arc::new(move || script.read())
        };
        let (standing, stood) = watch::channel(Standing::default());
        let reads = Reads {
            store: Arc::clone(&store),
            fence: elector.fence(),
            standing: stood,
            clock,
            counts: Arc::default(),
        };
        let deciders = Deciders::start(deciders, depth, &reads).expect("the threads start");
        let (sender, answers) = mpsc::channel(64);
        Bed {
            counts: Arc::clone(&reads.counts),
            reads,
            elector,
            standing,
            store,
            script,
            deciders,
            workers,
            sender,
            answers,
        }
    }

    /// The bed most cases stand on: the `random` preset as generation 1, a
    /// leader, two deciders behind queues of eight.
    fn leading(steps: impl IntoIterator<Item = Step>) -> Bed {
        bed(Some(preset("random", 1)), true, (2, 8), steps)
    }

    /// A frame for `model` with the case's budget.
    fn frame_for(model: &str, id: u8, seed: u64) -> proto::DecideRequest {
        proto::DecideRequest {
            request_id: vec![id],
            facts: Some(proto::RequestFacts {
                model: model.to_owned(),
                seed,
                budget_micros: u32::try_from(BUDGET.as_micros()).expect("five seconds"),
                ..proto::RequestFacts::default()
            }),
            constraints: None,
        }
    }

    fn frame(id: u8, seed: u64) -> proto::DecideRequest {
        frame_for(MODEL, id, seed)
    }

    /// `frame` in `phase`.
    fn phased(mut frame: proto::DecideRequest, phase: proto::Phase) -> proto::DecideRequest {
        frame
            .facts
            .as_mut()
            .expect("the case's frame has facts")
            .phase = phase.into();
        frame
    }

    fn checked(frame: proto::DecideRequest) -> Checked {
        Checked::new(frame).expect("a frame the record holds")
    }

    impl Bed {
        /// Hands `frame` over as `gateway`'s, received `received` after the
        /// take; why it took no place, if it took none.
        fn submit_at(
            &self,
            frame: proto::DecideRequest,
            gateway: GatewayId,
            received: Duration,
        ) -> Option<Refused> {
            let job = Job {
                request: checked(frame),
                gateway,
                received: self.script.taken + received,
                answers: self.sender.clone(),
            };
            self.deciders.submit(job).map(|(refused, _job)| refused)
        }

        fn submit(&self, frame: proto::DecideRequest) -> Option<Refused> {
            self.submit_at(frame, GATEWAY, Duration::ZERO)
        }

        /// The next answer, within the case's patience.
        async fn answer(&mut self) -> proto::DecideAnswer {
            timeout(PATIENCE, self.answers.recv())
                .await
                .expect("an answer within the case's patience")
                .expect("the bed holds a sender")
        }

        /// `frame`'s answer: handed over and waited for.
        async fn ask(&mut self, frame: proto::DecideRequest) -> proto::DecideAnswer {
            assert_eq!(self.submit(frame), None);
            self.answer().await
        }

        /// The engine's own decision for `frame`, made in the case's thread
        /// over the store's view at the take, as the wire carries it.
        fn engines(&self, frame: &proto::DecideRequest, term: u64) -> proto::DecideAnswer {
            let wire = frame.facts.as_ref().expect("the case's frame has facts");
            let decision = self.store.deciding(|world| {
                let ask = Ask {
                    gateway: GATEWAY,
                    now: self.script.taken,
                    pinned: None,
                    affinity: false,
                };
                let mut view = ViewBuffers::default();
                let built = world.build(&wire.model, Role::Single, &ask, &mut view);
                let facts = RequestFacts {
                    model: &wire.model,
                    seed: wire.seed,
                    budget_micros: wire.budget_micros,
                    now_micros: world.micros(ask.now),
                    ..RequestFacts::default()
                };
                let pool = built.map_or_else(
                    || PoolFacts {
                        generation: world.spec().generation(),
                        ..PoolFacts::default()
                    },
                    |built| built.pool_facts(),
                );
                world.spec().engine().decide(
                    &facts,
                    &pool,
                    &view.candidates,
                    &Constraints::default(),
                    &mut Scratch::new(),
                )
            });
            wire::decided(frame.request_id.clone(), term, &decision)
        }
    }

    /// The reason of a `Busy` answer.
    fn busy(answer: &proto::DecideAnswer) -> Reason {
        match &answer.answer {
            Some(Answer::Busy(busy)) => busy.reason(),
            other => panic!("a Busy: {other:?}"),
        }
    }

    /// The outcome of a decision's answer.
    fn outcome(answer: &proto::DecideAnswer) -> &Outcome {
        match &answer.answer {
            Some(Answer::Decision(proto::Decision {
                outcome: Some(outcome),
                ..
            })) => outcome,
            other => panic!("a decision: {other:?}"),
        }
    }

    /// The worker a pick names first.
    fn picked(answer: &proto::DecideAnswer) -> WorkerId {
        match outcome(answer) {
            Outcome::Pick(pick) => WorkerId(pick.targets.first().expect("a target").id),
            other => panic!("a pick: {other:?}"),
        }
    }

    /// A shed as the engine declines: its reason beside its stage.
    fn declined(reason: proto::ShedReason, stage: &str) -> Outcome {
        Outcome::Shed(proto::Shed {
            reason: reason.into(),
            emptied_by: stage.to_owned(),
            retry_after_secs: 1,
            wait_allowed: false,
        })
    }

    #[tokio::test]
    async fn a_decision_names_a_registered_worker_under_the_leaders_term() {
        let mut bed = leading([]);
        for (id, seed) in (1..=16).zip(100..) {
            let frame = frame(id, seed);
            let answer = bed.ask(frame.clone()).await;
            assert_eq!((answer.request_id.as_slice(), answer.term), (&[id][..], 1));
            assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
            // The answer is the engine's own decision over the store's
            // view, under the generation that view was published under.
            assert_eq!(answer, bed.engines(&frame, 1));
        }
    }

    #[tokio::test]
    async fn a_request_whose_deciders_queue_is_full_comes_back_at_once() {
        let (reached, at_read) = std_mpsc::sync_channel(1);
        let (release, go) = std_mpsc::sync_channel(1);
        // No decider and no depth asked: one thread behind a queue of one.
        let spec = Some(preset("random", 1));
        let mut bed = bed(spec, true, (0, 0), [Step::Hold { reached, go }]);
        assert_eq!(bed.submit(frame(1, 1)), None);
        at_read
            .recv_timeout(PATIENCE)
            .expect("the decider took the first request");
        assert_eq!(bed.submit(frame(2, 2)), None, "the queue holds one");
        assert_eq!(bed.submit(frame(3, 3)), Some(Refused::Full));
        release.send(()).expect("the decider waits for the case");
        // The two that took a place are decided in the order they were
        // handed over; the third never reached the decider.
        for id in [1, 2] {
            let answer = bed.answer().await;
            assert_eq!(answer.request_id, [id]);
            assert!(bed.workers.contains(&picked(&answer)));
        }
        assert_eq!(bed.ask(frame(4, 4)).await.request_id, [4]);
    }

    #[tokio::test]
    async fn an_answer_its_streams_full_queue_does_not_take_is_dropped_and_the_decider_goes_on() {
        let mut bed = leading([]);
        // A stream whose queue holds two answers and that reads none.
        let (sender, mut answers) = mpsc::channel(2);
        for id in 1..=3 {
            let job = Job {
                request: checked(frame(id, 1)),
                gateway: GATEWAY,
                received: bed.script.taken,
                answers: sender.clone(),
            };
            assert!(bed.deciders.submit(job).is_none());
        }
        // One decider decides a model's requests in order: it is past the
        // three when it answers a fourth, on a queue that is read.
        assert_eq!(bed.ask(frame(4, 1)).await.request_id, [4]);
        for id in [1, 2] {
            let answer = answers.recv().await.expect("the queue held it");
            assert_eq!(answer.request_id, [id]);
        }
        assert!(
            answers.try_recv().is_err(),
            "the third found the queue full"
        );
        let dropped = |counts: &Counts| {
            [&counts.dropped_full, &counts.dropped_gone].map(|count| count.load(Ordering::Relaxed))
        };
        assert_eq!(dropped(&bed.counts), [1, 0]);
        // A stream that ended takes none: its answer is counted as gone.
        drop(answers);
        let job = Job {
            request: checked(frame(5, 1)),
            gateway: GATEWAY,
            received: bed.script.taken,
            answers: sender,
        };
        assert!(bed.deciders.submit(job).is_none());
        assert_eq!(bed.ask(frame(6, 1)).await.request_id, [6]);
        assert_eq!(dropped(&bed.counts), [1, 1]);
    }

    #[tokio::test]
    async fn a_budget_the_queue_used_up_is_busy_and_one_microsecond_of_it_is_a_decision() {
        let short = BUDGET - Duration::from_micros(1);
        // The first request is taken with one microsecond left, the second
        // with none, the third with under one: the engine reads what is
        // left in whole microseconds, and 0 as none.
        let under = BUDGET - Duration::from_nanos(999);
        let steps = [short, short, BUDGET, under].map(Step::At);
        let mut bed = leading(steps);
        let answer = bed.ask(frame(1, 1)).await;
        assert!(bed.workers.contains(&picked(&answer)));
        let answer = bed.ask(frame(2, 2)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!((answer.request_id.as_slice(), answer.term), (&[2][..], 1));
        let answer = bed.ask(frame(3, 3)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        // One read of the clock each: nothing was read for either after
        // the budget.
        assert_eq!(bed.script.readers.lock().len(), 4);
    }

    #[tokio::test]
    async fn a_replica_that_does_not_lead_answers_not_leader_under_the_highest_term_it_knows() {
        // The third request is taken a whole budget after its receipt.
        let steps = [Duration::ZERO, Duration::ZERO, BUDGET].map(Step::At);
        let mut bed = bed(Some(preset("random", 1)), false, (2, 8), steps);
        let answer = bed.ask(frame(1, 1)).await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 0));
        // The election saw another replica hold the lease under term 4.
        bed.standing.send_replace(Standing {
            holder: "controller-1@10.0.0.8:30100".to_owned(),
            term: 4,
            looked: true,
        });
        let answer = bed.ask(frame(2, 2)).await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 4));
        // The fence is read before the budget: the reason that still holds
        // for the next request is the one said.
        let answer = bed.ask(frame(3, 3)).await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 4));
        // One read of the clock each: nothing was built for any.
        assert_eq!(bed.script.readers.lock().len(), 3);
    }

    #[tokio::test]
    async fn a_leader_past_its_renew_deadline_answers_not_leader_under_its_own_term() {
        let mut bed = leading([Step::At(FENCED)]);
        assert_eq!(bed.submit_at(frame(1, 1), GATEWAY, FENCED), None);
        let answer = bed.answer().await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
    }

    #[tokio::test]
    async fn a_fence_that_closes_between_the_build_and_the_hand_over_sends_busy_in_the_decisions_place(
    ) {
        // The first read finds the fence open and the decision is made; the
        // read at the hand-over is past the renew deadline.
        let mut bed = leading([Step::At(Duration::ZERO), Step::At(FENCED)]);
        let answer = bed.ask(frame(1, 1)).await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
        assert_eq!(bed.script.readers.lock().len(), 2);
    }

    #[tokio::test]
    async fn a_fence_open_under_a_later_term_at_the_hand_over_sends_busy_too() {
        let (reached, at_read) = std_mpsc::sync_channel(1);
        let (release, go) = std_mpsc::sync_channel(1);
        let mut bed = leading([Step::At(Duration::ZERO), Step::Hold { reached, go }]);
        assert_eq!(bed.submit(frame(1, 1)), None);
        at_read
            .recv_timeout(PATIENCE)
            .expect("the decider reached the hand-over");
        // The decision was made under term 1. The candidate gives the lease
        // up and takes it again: the fence is open, under term 2.
        assert!(bed.elector.release().expect("the release"));
        let tick = bed.elector.tick(bed.script.taken).expect("the take");
        assert_eq!(tick, Tick::Acquired { term: 2 });
        release.send(()).expect("the decider waits for the case");
        let answer = bed.answer().await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 2));
        // The next request is decided under the term that stands.
        let answer = bed.ask(frame(2, 2)).await;
        assert_eq!(answer.term, 2);
        assert!(bed.workers.contains(&picked(&answer)));
    }

    /// The term a decision leaves under is the fence's, the one both reads
    /// showed. A higher one the election has seen since is a `Busy`'s to
    /// carry.
    #[tokio::test]
    async fn a_decision_leaves_under_the_fences_term_while_the_election_has_seen_a_later_one() {
        let mut bed = leading([]);
        bed.standing.send_replace(Standing {
            holder: "controller-1@10.0.0.8:30100".to_owned(),
            term: 4,
            looked: true,
        });
        let answer = bed.ask(frame(1, 1)).await;
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        assert_eq!(answer.term, 1);
    }

    /// The deciders as they stand once their one thread ended: its queue
    /// has no reader.
    fn gone() -> Deciders {
        let (queue, jobs) = std_mpsc::sync_channel(1);
        drop(jobs);
        Deciders {
            queues: Box::new([queue]),
            told: Arc::new(watch::Sender::new(false)),
        }
    }

    /// A decider's thread that ended takes no request: each one for its
    /// models comes back whole, as gone and never as a full queue.
    #[test]
    fn a_request_for_a_decider_whose_thread_ended_comes_back_as_gone() {
        let deciders = gone();
        let (answers, _unread) = mpsc::channel(1);
        let job = Job {
            request: checked(frame(1, 1)),
            gateway: GATEWAY,
            received: Instant::now(),
            answers,
        };
        let back = deciders
            .submit(job)
            .map(|(refused, job)| (refused, job.request.into_id()));
        assert_eq!(back, Some((Refused::Gone, vec![1])));
    }

    #[tokio::test]
    async fn a_models_requests_are_decided_by_one_decider_in_the_order_they_were_handed_over() {
        let mut bed = bed(Some(preset("random", 1)), true, (4, 64), []);
        for id in 1..=48 {
            assert_eq!(bed.submit(frame(id, u64::from(id))), None);
        }
        for id in 1..=48 {
            assert_eq!(bed.answer().await.request_id, [id]);
        }
        let own = format!("decide-{}", bed.deciders.of(MODEL));
        let readers = mem::take(&mut *bed.script.readers.lock());
        assert_eq!(readers.len(), 96, "two reads a decision");
        assert!(readers.iter().all(|reader| *reader == own), "{readers:?}");
        // A model with another key is another decider's, whoever asks.
        let other = (0..64)
            .map(|index| format!("model-{index}"))
            .find(|model| bed.deciders.of(model) != bed.deciders.of(MODEL))
            .expect("some model of sixty-four has another key");
        for gateway in [GATEWAY, OTHER] {
            let frame = frame_for(&other, 1, 1);
            assert_eq!(bed.submit_at(frame, gateway, Duration::ZERO), None);
            bed.answer().await;
        }
        let theirs = format!("decide-{}", bed.deciders.of(&other));
        let readers = mem::take(&mut *bed.script.readers.lock());
        assert!(
            !readers.is_empty() && readers.iter().all(|reader| *reader == theirs),
            "{readers:?}"
        );
    }

    /// What the engine is handed for one request, copied out of the
    /// hand-over.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Handed {
        now_micros: u64,
        budget_micros: u32,
        /// The affinity column's length.
        column: usize,
        candidates: Vec<WorkerId>,
        stalled: Vec<WorkerId>,
        /// The pool's size and its generation.
        pool: (u32, u64),
    }

    /// The hand-over of `frame` as `gateway`'s at `now`, with 1,234
    /// microseconds of its budget left.
    fn handed(
        store: &MemoryStore,
        frame: proto::DecideRequest,
        gateway: GatewayId,
        now: Instant,
    ) -> Handed {
        let request = checked(frame);
        // Received so long before `now` that this much of the budget is
        // left when the decider takes it.
        let received = now - (BUDGET - Duration::from_micros(1_234));
        let taken = Taken::at(&request, gateway, received, now).expect("budget is left");
        let mut buffers = Buffers::default();
        store.deciding(|world| {
            hand(
                world,
                &taken,
                &mut buffers,
                &AtomicU64::new(0),
                |facts, pool, rows, _, _| Handed {
                    now_micros: facts.now_micros,
                    budget_micros: facts.budget_micros,
                    column: facts.affinity.len(),
                    candidates: rows.iter().map(|row| row.id).collect(),
                    stalled: rows
                        .iter()
                        .filter(|row| row.health.stalled)
                        .map(|row| row.id)
                        .collect(),
                    pool: (pool.pool_size, pool.generation),
                },
            )
        })
    }

    #[test]
    fn the_engine_is_handed_the_asking_gateways_view_at_the_deciders_instant_and_what_is_left_of_the_budget(
    ) {
        let store = MemoryStore::new();
        let (mut singles, sides) = fleet(&store);
        singles.sort_unstable();
        store
            .apply(preset("random", 1))
            .expect("above the unbound start");
        let taken = Instant::now();
        let now = taken + Duration::from_secs(3);
        let whole = Handed {
            now_micros: store.micros(now),
            budget_micros: 1_234,
            column: 0,
            candidates: singles.clone(),
            stalled: Vec::new(),
            pool: (3, 1),
        };
        assert_eq!(handed(&store, frame(1, 1), GATEWAY, now), whole);

        // A veto is its gateway's alone, and it stands at the decision's
        // instant or it masks nothing.
        let vetoed = singles[1];
        store.gateway_veto(GATEWAY, vetoed, Some(taken + Duration::from_secs(60)));
        let masked = Handed {
            stalled: vec![vetoed],
            ..whole.clone()
        };
        assert_eq!(handed(&store, frame(1, 1), GATEWAY, now), masked);
        assert_eq!(handed(&store, frame(1, 1), OTHER, now), whole);
        let ended = taken + Duration::from_secs(61);
        let after = handed(&store, frame(1, 1), GATEWAY, ended);
        assert_eq!(
            (after.stalled.as_slice(), after.now_micros),
            (&[][..], store.micros(ended))
        );

        // A request is decided over the pool of its phase's role.
        for (phase, member) in sides {
            let side = handed(&store, phased(frame(1, 1), phase), GATEWAY, now);
            let alone = Handed {
                candidates: vec![member],
                pool: (1, 1),
                ..whole.clone()
            };
            assert_eq!(side, alone, "{phase:?}");
        }

        // A model no member serves: no candidate, under the generation
        // that stands, at the same instant of the rows' clock.
        let nobody = Handed {
            candidates: Vec::new(),
            pool: (0, 1),
            ..whole.clone()
        };
        let frame_nobody = frame_for("model-nobody-serves", 1, 1);
        assert_eq!(handed(&store, frame_nobody, GATEWAY, now), nobody);

        // A profile that declares the affinity column is handed one entry
        // a candidate, and no other profile is.
        store
            .apply(preset("cache_aware", 2))
            .expect("above the standing one");
        let column = handed(&store, frame(1, 1), OTHER, now);
        assert_eq!((column.column, column.pool), (3, (3, 2)));
    }

    #[tokio::test]
    async fn a_decision_is_made_for_the_gateway_its_request_came_from() {
        let mut bed = leading([]);
        let vetoed = bed.workers[0];
        let until = bed.script.taken + Duration::from_secs(60);
        bed.store.gateway_veto(GATEWAY, vetoed, Some(until));
        let mut theirs = Vec::new();
        for (id, seed) in (1..=48).zip(1..) {
            let own = bed.ask(frame(id, seed)).await;
            assert_ne!(picked(&own), vetoed, "seed {seed}");
            assert_eq!(bed.submit_at(frame(id, seed), OTHER, Duration::ZERO), None);
            theirs.push(picked(&bed.answer().await));
        }
        // The veto is the asking gateway's alone.
        assert!(theirs.contains(&vetoed), "{theirs:?}");
    }

    #[tokio::test]
    async fn a_pair_plans_side_and_a_prompt_as_text_have_no_decision_and_are_counted_by_cause() {
        let document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": { "pick": "random" } },
              "bindings": [
                { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
                  "profiles": { "prefill": "p", "decode": "p" } },
                { "profile": "p" }
              ]
            }"#,
        )
        .expect("a document of this build");
        let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
        // The pair door decides a pair plan's sides, and no step serves it
        // yet: Busy with no reason, never one side's decision and never a
        // shed, which is a policy's word.
        for (id, phase) in [(1, proto::Phase::Prefill), (2, proto::Phase::Decode)] {
            let answer = bed.ask(phased(frame(id, 77), phase)).await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice(), answer.term),
                (Reason::Unspecified, &[id][..], 1),
                "{phase:?}"
            );
        }
        // Text the controller is to hash: never decided as a request with
        // no prefix.
        let mut text = frame(3, 77);
        text.facts
            .as_mut()
            .expect("the case's frame has facts")
            .prompt_text = b"a prompt".to_vec();
        let answer = bed.ask(text).await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice(), answer.term),
            (Reason::Unspecified, &[3][..], 1)
        );
        let counted = |count: &AtomicU64| count.load(Ordering::Relaxed);
        let Counts {
            faults,
            pair_sides,
            texts,
            hashes_cut,
            ..
        } = &*bed.counts;
        assert_eq!(
            (
                counted(faults),
                counted(pair_sides),
                counted(texts),
                counted(hashes_cut)
            ),
            (0, 2, 1, 0)
        );
        // The same document's other phases are decided: the single binding
        // covers them.
        for (id, phase) in [(4, proto::Phase::Single), (5, proto::Phase::Encode)] {
            let answer = bed.ask(phased(frame(id, 77), phase)).await;
            assert!(matches!(outcome(&answer), Outcome::Pick(_)), "{phase:?}");
        }
    }

    /// The encode side rides the plan that names it. The engine's entry
    /// would shed an encode request there as if no encoder stood, while
    /// one stands ready.
    #[tokio::test]
    async fn every_side_of_the_plan_with_an_encode_side_has_no_decision_and_is_counted() {
        let document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": { "pick": "random" } },
              "bindings": [
                { "match": { "phase": ["encode", "prefill", "decode"] }, "plan": "epd",
                  "profiles": { "encode": "p", "prefill": "p", "decode": "p" } },
                { "profile": "p" }
              ]
            }"#,
        )
        .expect("a document of this build");
        let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
        for (id, phase) in [
            (1, proto::Phase::Encode),
            (2, proto::Phase::Prefill),
            (3, proto::Phase::Decode),
        ] {
            let answer = bed.ask(phased(frame(id, 77), phase)).await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice(), answer.term),
                (Reason::Unspecified, &[id][..], 1),
                "{phase:?}"
            );
        }
        assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 3);
        // A single request falls to the single binding and is decided.
        let answer = bed.ask(phased(frame(4, 77), proto::Phase::Single)).await;
        assert!(matches!(outcome(&answer), Outcome::Pick(_)));
    }

    /// The guard names a pair plan's sides and no other phase: a phase
    /// the plan has no side for stays the engine's word, a decline by the
    /// binding.
    #[tokio::test]
    async fn a_phase_a_pair_plan_has_no_side_for_is_declined_by_its_binding() {
        let document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": { "pick": "random" } },
              "bindings": [
                { "plan": "pd-pair", "profiles": { "prefill": "p", "decode": "p" } }
              ]
            }"#,
        )
        .expect("a document of this build");
        let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
        for (id, phase) in [(1, proto::Phase::Encode), (2, proto::Phase::Single)] {
            let answer = bed.ask(phased(frame(id, 77), phase)).await;
            assert_eq!(
                *outcome(&answer),
                declined(proto::ShedReason::PolicyDeclined, "binding"),
                "{phase:?}"
            );
        }
        assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
    }

    /// A replica that does not lead says so for a request it has no
    /// decision for, too.
    #[tokio::test]
    async fn a_replica_that_does_not_lead_says_so_before_it_reads_a_prompt_as_text() {
        let mut bed = bed(Some(preset("random", 1)), false, (2, 8), []);
        let mut text = frame(1, 1);
        text.facts
            .as_mut()
            .expect("the case's frame has facts")
            .prompt_text = b"a prompt".to_vec();
        assert_eq!(busy(&bed.ask(text).await), Reason::NotLeader);
        assert_eq!(bed.counts.texts.load(Ordering::Relaxed), 0);
    }

    /// A text is said before the budget is read: the count holds every
    /// text a leader took, whatever was left of its budget.
    #[tokio::test]
    async fn a_prompt_as_text_whose_budget_is_gone_is_still_counted_as_a_text() {
        let mut bed = leading([Step::At(BUDGET)]);
        let mut text = frame(1, 1);
        text.facts
            .as_mut()
            .expect("the case's frame has facts")
            .prompt_text = b"a prompt".to_vec();
        assert_eq!(busy(&bed.ask(text).await), Reason::Unspecified);
        assert_eq!(bed.counts.texts.load(Ordering::Relaxed), 1);
        // The same instant with no text: the budget's own reason.
        let answer = bed.ask(frame(2, 2)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    }

    /// The lines a case's own subscriber wrote.
    #[derive(Clone, Default)]
    struct Logged(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Logged {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A decider of the case's own, on the case's thread: after a fault it
    /// holds new buffers, and a streak of faults writes one line.
    #[test]
    fn a_fault_leaves_the_next_decision_new_buffers_and_a_streak_of_them_writes_one_line() {
        // The second read of the clock panics: the view is built by then.
        let fault = || [Step::At(Duration::ZERO), Step::Panic];
        let steps = fault().into_iter().chain(fault()).chain(fault());
        let bed = bed(Some(preset("random", 1)), true, (1, 8), steps);
        let mut decider = Decider {
            reads: bed.reads.clone(),
            buffers: Buffers::default(),
            faulted: false,
            index: 0,
        };
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        // A second subscriber stays registered beside it: with one alone a
        // line is on or off by the thread that reaches it first.
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let written = || {
            let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
            text.matches("a decision panicked").count()
        };
        tracing::subscriber::with_default(subscriber, || {
            let ask = |decider: &mut Decider, id| {
                decider.answer(checked(frame(id, 1)), GATEWAY, bed.script.taken)
            };
            for id in 1..=3 {
                assert_eq!(busy(&ask(&mut decider, id)), Reason::Unspecified);
                assert_eq!(
                    decider.buffers.view.candidates.capacity(),
                    0,
                    "the buffers the fault left are gone"
                );
            }
            assert_eq!(written(), 1, "one line for the streak");
            // A decision that does not fault ends the streak and keeps its
            // buffers.
            let answer = ask(&mut decider, 4);
            assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
            assert!(decider.buffers.view.candidates.capacity() > 0);
            bed.script.steps.lock().extend(fault());
            assert_eq!(busy(&ask(&mut decider, 5)), Reason::Unspecified);
            assert_eq!(written(), 2, "a new streak, a new line");
        });
        assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 4);
    }

    /// A decider's thread holds the guard from its start: one that unwinds
    /// is told, to a receiver taken after the word too; one that returns at
    /// its queue's end, and a decision's own panic, are not.
    #[tokio::test]
    async fn a_thread_that_unwinds_is_told_and_one_that_returns_is_not() {
        let told = Arc::new(watch::Sender::new(false));
        let returned = watched(0, &told, || {}).expect("the thread starts").join();
        assert!(returned.is_ok() && !*told.subscribe().borrow());
        // No receiver stands at the unwind: the word is kept for the next.
        let unwound = watched(1, &told, || {
            panic!("the case's own panic, outside a decision")
        })
        .expect("the thread starts")
        .join();
        assert!(unwound.is_err() && *told.subscribe().borrow());

        // A decision that panics is caught: its decider's thread goes on,
        // and the deciders tell nothing.
        let spec = Some(preset("random", 1));
        let mut bed = bed(spec, true, (1, 8), [Step::Panic]);
        assert_eq!(busy(&bed.ask(frame(1, 1)).await), Reason::Unspecified);
        let answer = bed.ask(frame(2, 2)).await;
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        assert!(!*bed.deciders.ended().borrow());
        // The word a thread's unwind says is the one the deciders show.
        bed.deciders.tell_unwound();
        assert!(*bed.deciders.ended().borrow());
    }

    /// The engine is handed its generation's cap of a request's block
    /// hashes, the first ones, and a request that carried more is counted.
    #[tokio::test]
    async fn the_block_hashes_past_the_generations_cap_are_cut_and_the_request_is_counted() {
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = 1;
        document.view.max_block_hashes = 4;
        let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
        let hashed = |id, hashes: u64| {
            let mut frame = frame(id, 1);
            frame
                .facts
                .as_mut()
                .expect("the case's frame has facts")
                .block_hashes = (1..=hashes).collect();
            frame
        };
        let now = bed.script.taken;
        let read = |frame| {
            let request = checked(frame);
            let taken = Taken::at(&request, GATEWAY, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            bed.store.deciding(|world| {
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &AtomicU64::new(0),
                    |facts, _, _, _, _| facts.block_hashes.to_vec(),
                )
            })
        };
        assert_eq!(read(hashed(1, 6)), [1, 2, 3, 4]);
        assert_eq!(read(hashed(2, 4)), [1, 2, 3, 4]);
        assert_eq!(read(hashed(3, 2)), [1, 2]);
        // Never refused: each is decided, and the one that was cut counted.
        for (id, hashes) in [(4, 6), (5, 4), (6, 0)] {
            let answer = bed.ask(hashed(id, hashes)).await;
            assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        }
        assert_eq!(bed.counts.hashes_cut.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn a_decision_that_panics_is_answered_busy_with_no_reason_and_its_decider_decides_the_next(
    ) {
        let spec = Some(preset("random", 1));
        let mut bed = bed(spec, true, (1, 8), [Step::Panic]);
        let answer = bed.ask(frame(1, 1)).await;
        assert_eq!((busy(&answer), answer.term), (Reason::Unspecified, 1));
        assert_eq!(answer.request_id, [1]);
        assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
        let frame = frame(2, 2);
        assert_eq!(bed.ask(frame.clone()).await, bed.engines(&frame, 1));
    }

    /// A panic's payload can panic at its own drop. The decider frees it
    /// under a catch of its own: the request is answered, its thread stands
    /// and the deciders tell nothing.
    #[tokio::test]
    async fn a_panic_whose_payload_panics_at_its_drop_costs_its_own_request_alone() {
        let spec = Some(preset("random", 1));
        let mut bed = bed(spec, true, (1, 8), [Step::Bomb]);
        let ended = bed.deciders.ended();
        let answer = bed.ask(frame(1, 1)).await;
        assert_eq!((busy(&answer), answer.term), (Reason::Unspecified, 1));
        assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
        let frame = frame(2, 2);
        assert_eq!(bed.ask(frame.clone()).await, bed.engines(&frame, 1));
        assert!(!*ended.borrow(), "the thread stands");
    }

    #[tokio::test]
    async fn the_engine_answers_where_no_generation_binds_the_model_and_where_no_member_serves_it()
    {
        // The unbound generation: number 0, declined by the binding.
        let mut unbound = bed(None, true, (2, 8), []);
        let frame = frame(1, 1);
        let answer = unbound.ask(frame.clone()).await;
        assert_eq!(
            *outcome(&answer),
            declined(proto::ShedReason::PolicyDeclined, "binding")
        );
        assert_eq!(answer, unbound.engines(&frame, 1));
        // A model no member serves: the engine's answer over no candidate.
        let mut bed = leading([]);
        let frame = frame_for("model-nobody-serves", 2, 2);
        let answer = bed.ask(frame.clone()).await;
        assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
        assert_eq!(answer, bed.engines(&frame, 1));
    }

    /// Every worker a pick names, its alternates with it.
    fn targets(answer: &proto::DecideAnswer) -> Vec<WorkerId> {
        match outcome(answer) {
            Outcome::Pick(pick) => pick.targets.iter().map(|t| WorkerId(t.id)).collect(),
            other => panic!("a pick: {other:?}"),
        }
    }

    /// `frame` with what its gateway alone knows.
    fn constrained(
        mut frame: proto::DecideRequest,
        wires: u32,
        hint: Option<u32>,
    ) -> proto::DecideRequest {
        frame.constraints = Some(proto::Constraints {
            wire_modes: wires,
            subset_hint: hint,
            ..proto::Constraints::default()
        });
        frame
    }

    /// One ready member that speaks HTTP alone, asked for under each thing
    /// a gateway states of itself. A gateway that states no wire, or one
    /// the member speaks, is answered the member. One whose wires the
    /// member shares none of is shed under the wires' stage, and so is one
    /// that states only a wire this build does not know. A hint of 0 is a
    /// decision over no candidates, shed under the hint's stage whatever
    /// the wires, and a positive hint cuts nothing and names no stage: the
    /// wires that share none are the stage under one. Then a member the
    /// gateway cannot dial and a second one it can, beside the first, which
    /// it just failed to reach: the second is its whole answer.
    #[tokio::test]
    async fn a_gateway_is_answered_no_member_it_cannot_dial_and_none_under_a_hint_of_0() {
        const SPOKEN: &str = "model-http";
        let mut bed = leading([]);
        let store = Arc::clone(&bed.store);
        let first = ready(&store, "h0", SPOKEN, Role::Single, WireModes::HTTP);
        let answer = bed.ask(frame_for(SPOKEN, 100, 100)).await;
        assert_eq!(targets(&answer), [first], "no constraints");
        // The wires a gateway states, its hint, and the stage its decision
        // is shed under where it is shed.
        let cells = [
            (0b001, None, None),
            (0b011, None, None),
            (0b010, None, Some("wires")),
            (0b110, None, Some("wires")),
            (0b1000, None, Some("wires")),
            (0b010, Some(1), Some("wires")),
            (0b001, Some(0), Some("subset-hint")),
            (0b010, Some(0), Some("subset-hint")),
            (0, Some(0), Some("subset-hint")),
            (0b001, Some(1), None),
            (0, Some(7), None),
        ];
        for (id, (wires, hint, stage)) in (1..).zip(cells) {
            let frame = constrained(frame_for(SPOKEN, id, u64::from(id)), wires, hint);
            let answer = bed.ask(frame).await;
            let cell = format!("wires {wires:#b}, a hint of {hint:?}");
            match stage {
                Some(stage) => assert_eq!(
                    outcome(&answer),
                    &declined(proto::ShedReason::NoEndpoints, stage),
                    "{cell}"
                ),
                None => assert_eq!(targets(&answer), [first], "{cell}"),
            }
        }
        // The wires took the member in four decisions: under a hint of 0
        // they had none to take.
        assert_eq!(bed.counts.undialed(), 4);
        ready(&store, "g0", SPOKEN, Role::Single, WireModes::GRPC);
        let second = ready(&store, "h1", SPOKEN, Role::Single, WireModes::HTTP);
        for seed in 0..32_u8 {
            let frame = frame_for(SPOKEN, 40 + seed, u64::from(seed));
            let mut frame = constrained(frame, 0b001, None);
            frame.constraints.get_or_insert_default().excluded = vec![first.0];
            let answer = bed.ask(frame).await;
            assert_eq!(targets(&answer), [second], "seed {seed}");
        }
        assert_eq!(bed.counts.undialed(), 4 + 32);
    }

    /// A gateway that states its wires is answered members that share one
    /// with it, its alternates too, and each one its wires took is counted.
    /// A member that states no wire is no candidate of a gateway that
    /// states some: silence is no waiver. A gateway that states none is not
    /// narrowed, and a positive hint, which no stage reads yet, cuts
    /// nothing.
    #[tokio::test]
    async fn a_gateways_wires_narrow_its_decision_to_the_members_that_share_one() {
        let mut bed = leading([]);
        // The fleet's three members state no wire.
        let answer = bed.ask(constrained(frame(1, 1), 0b111, None)).await;
        let shed = declined(proto::ShedReason::NoEndpoints, "wires");
        assert_eq!(outcome(&answer), &shed);
        assert_eq!(bed.counts.undialed(), 3);
        let store = Arc::clone(&bed.store);
        let both = WireModes::HTTP.union(WireModes::GRPC);
        let [_, grpc, either] = [
            ("w0", WireModes::HTTP),
            ("w1", WireModes::GRPC),
            ("w2", both),
        ]
        .map(|(name, wires)| ready(&store, name, MODEL, Role::Single, wires));
        let mut dialed = [grpc, either];
        dialed.sort_unstable();
        let mut named = BTreeSet::new();
        for (id, seed) in (2..=33).zip(100..) {
            let answer = bed.ask(constrained(frame(id, seed), 0b010, None)).await;
            named.extend(targets(&answer));
        }
        assert_eq!(named.into_iter().collect::<Vec<_>>(), dialed);
        // The member that speaks HTTP alone, in each of the 32 decisions.
        assert_eq!(bed.counts.undialed(), 3 + 32);
        for (id, hint) in (40..).zip([None, Some(1), Some(2), Some(64)]) {
            let answer = bed.ask(constrained(frame(id, 40), 0, hint)).await;
            assert_eq!(targets(&answer).len(), 3, "a hint of {hint:?}");
        }
        assert_eq!(bed.counts.undialed(), 3 + 32);
    }

    /// The budget is judged again where the answer is handed over, at the
    /// read the fence takes there and with no clock read more. A decision
    /// with a microsecond left leaves. One whose budget ended while it was
    /// made leaves as `Busy` under the reason a late arrival is told, and
    /// is counted apart from one.
    #[tokio::test]
    async fn a_decision_whose_budget_ended_while_it_was_made_leaves_as_busy_and_is_counted_apart() {
        let short = BUDGET - Duration::from_micros(1);
        let under = BUDGET - Duration::from_nanos(999);
        let steps = [
            Duration::ZERO,
            short,
            Duration::ZERO,
            under,
            Duration::ZERO,
            BUDGET,
            BUDGET,
        ]
        .map(Step::At);
        let mut bed = leading(steps);
        let answer = bed.ask(frame(1, 1)).await;
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 0);
        // Under one microsecond left is gone, as where a request is taken.
        for (id, count) in [(2, 1), (3, 2)] {
            let answer = bed.ask(frame(id, 2)).await;
            assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
            assert_eq!((answer.request_id.as_slice(), answer.term), (&[id][..], 1));
            assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), count);
        }
        // A request that arrived late is told the same reason at its
        // decider's first read, and is none of the count.
        let answer = bed.ask(frame(4, 4)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 2);
        assert_eq!(bed.script.readers.lock().len(), 7);
    }

    /// The wires' count is kept in parts, a decider's own at its index, so
    /// no line is written by two of them, and it is read whole. A model on
    /// each of the bed's two deciders, each with one member its gateway
    /// cannot dial: a decision counts in its own decider's part, and
    /// deciders past the parts share one.
    #[tokio::test]
    async fn a_decider_counts_the_candidates_the_wires_took_in_its_own_part() {
        let mut bed = leading([]);
        let store = Arc::clone(&bed.store);
        let part = |counts: &Counts, index| counts.undialed_of(index).load(Ordering::Relaxed);
        let mut id = 0;
        for decider in 0..2 {
            let model = (0..)
                .map(|n| format!("model-{n}"))
                .find(|model| bed.deciders.of(model) == decider)
                .expect("a model on each decider");
            ready(
                &store,
                &format!("m{decider}"),
                &model,
                Role::Single,
                WireModes::HTTP,
            );
            // One decision on the first decider, two on the second.
            for seed in 0..=decider {
                id += 1;
                let frame = constrained(frame_for(&model, id, seed as u64), 0b010, None);
                let answer = bed.ask(frame).await;
                let shed = declined(proto::ShedReason::NoEndpoints, "wires");
                assert_eq!(outcome(&answer), &shed);
            }
        }
        assert_eq!((part(&bed.counts, 0), part(&bed.counts, 1)), (1, 2));
        assert_eq!(bed.counts.undialed(), 3);
        let counts = Counts::default();
        for index in 0..PARTS + 2 {
            counts
                .undialed_of(index)
                .fetch_add(1 << index, Ordering::Relaxed);
        }
        assert_eq!(part(&counts, 0), 1 + (1 << PARTS));
        assert_eq!(part(&counts, PARTS + 1), 2 + (2 << PARTS));
        assert_eq!(counts.undialed(), (1 << (PARTS + 2)) - 1);
    }

    /// The pass over a view. The rows a gateway's wires keep hold their
    /// order, take their positions and carry the affinity column beside
    /// them, with a column or without, and the pass says how many rows the
    /// wires took: a row that states no wire is taken as one that states
    /// another is. A gateway that states only wires no row shares takes
    /// every row, and one that states none takes none. A hint of 0 keeps
    /// none and leaves the wires nothing to take; a positive hint cuts
    /// nothing.
    #[test]
    fn a_narrowed_view_keeps_its_rows_order_their_positions_and_the_column_beside_them() {
        let both = WireModes::HTTP.union(WireModes::GRPC);
        let wires = [
            WireModes::HTTP,
            WireModes::GRPC,
            both,
            WireModes::GRPC,
            WireModes::ZMQ,
            WireModes::NONE,
        ];
        let built = |column: bool| {
            let mut view = ViewBuffers::default();
            for (position, wire_modes) in (0_u32..).zip(wires) {
                view.candidates.push(Candidate {
                    id: WorkerId(u64::from(position) + 10),
                    position,
                    wire_modes,
                    ..Candidate::default()
                });
                if column {
                    view.affinity.push(AffinityFacts {
                        admitted_age_ms: position + 100,
                        ..AffinityFacts::default()
                    });
                }
            }
            view
        };
        let read = |view: &ViewBuffers| {
            let rows = view.candidates.iter().map(|row| (row.id.0, row.position));
            let ages = view.affinity.iter().map(|facts| facts.admitted_age_ms);
            (rows.collect::<Vec<_>>(), ages.collect::<Vec<_>>())
        };
        let kept = vec![(11, 0), (12, 1), (13, 2)];
        let mut view = built(true);
        assert_eq!(narrow(&mut view, Some(WireModes::GRPC), None), 3);
        assert_eq!(read(&view), (kept.clone(), vec![101, 102, 103]));
        let mut view = built(false);
        assert_eq!(narrow(&mut view, Some(WireModes::GRPC), Some(2)), 3);
        assert_eq!(read(&view), (kept, vec![]));
        let mut view = built(true);
        assert_eq!(narrow(&mut view, Some(both.union(WireModes::ZMQ)), None), 1);
        assert_eq!(read(&view).1, [100, 101, 102, 103, 104]);
        let mut view = built(true);
        assert_eq!(narrow(&mut view, Some(WireModes::NONE), None), 6);
        assert_eq!(read(&view), (vec![], vec![]));
        let mut view = built(true);
        assert_eq!(narrow(&mut view, None, Some(64)), 0);
        assert_eq!(read(&view), read(&built(true)));
        let mut view = built(true);
        assert_eq!(narrow(&mut view, Some(WireModes::GRPC), Some(0)), 0);
        assert_eq!(read(&view), (vec![], vec![]));
    }
}
