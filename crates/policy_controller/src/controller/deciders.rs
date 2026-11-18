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
//! 6. the lease, for a pick: one a decision, a credit on the pick's first
//!    target, minted and filed by the store under the term the first read
//!    showed, at that read's instant ([`MemoryStore::issue_decided`]). The
//!    engine that decided is told of it, and the answer carries it;
//! 7. the fence again, and the budget at the same read, with the answer's
//!    hand-over alone behind it: a decision whose budget ended while it was
//!    made, its lease's filing and the stages' hearing of it included,
//!    leaves as `Busy` under the reason a late arrival is told, and under
//!    a count of its own ([`Counts`]). So does a pick whose lease has
//!    under a millisecond of its dispatch span left at that read: the span
//!    counts from the first read, and the answer's lease carries what is
//!    left of it. The lease of an answer that does not leave is given up
//!    at once: the store's release door first, then the engine told, as
//!    abandoned under a closed fence and as a late decision past the
//!    budget or the span.
//!
//! A lease is what the next decisions read. The store counts the leases on
//! file by the worker each books, and a decision reads that count as each
//! row's requests in flight ([`crate::RequestFacts::in_flight`]): beside
//! the rows where the store lent them, in the row where they were written
//! for the request. A decision is made over what the decisions before it
//! left on file. The file is the one thing every decider writes, under its
//! turn, held for an apply and for one lease's filing or its giving up,
//! never across a decision. What a gateway's frame says of a lease is
//! handed to the file by its session and applied by a decider at work:
//! before a decision reads its rows' counts, at its issue, and once more
//! before the decider waits on an empty queue. Where no decider is at work
//! the session applies it. A lease leaves the file where its answer does
//! not leave (step 7), at its gateway's word (a `Session`'s `Completed` or
//! `Released` frame, the gateway's own end) or at its dispatch deadline:
//! each issue ends the leases at the head of the store's deadline queue
//! that no gateway named a dispatch for in time, two at most, and their
//! engine is told under the turn where its generation still stands. So a
//! lease whose answer never reached its gateway (an answer in the queue
//! of a stream that then ended, a decision that panicked past its issue)
//! stands until its deadline and the issues after it, and through a lull
//! until the next issue: no timer ends one yet. A decision its stream
//! stops at the send is none of them: the stream hands its lease's
//! release there, as a session hands a frame's ([`super::decide`]). Nor
//! is an answer its stream's queue does not take, full or closed: its
//! decider hands the release where the queue refuses it.
//!
//! What a lease does not do yet. A request sent twice under one id is
//! issued two leases. A stage hears a lease's issue here, its release
//! where its answer does not leave, its dispatch, completion and release
//! at its gateway's word and its end at the dispatch deadline, and not the
//! end of a lease whose gateway was let go: a document that names a stage
//! in need of the calls stays refused.
//!
//! A term mints 2^32 - 1 ids, and under every term but the last a leader
//! leaves before they end. The first answer a term hands over with an id at
//! or past its mark ([`past_mark`], three quarters of the ids) tells the
//! election to hand the lease over ([`Deciders::leave_through`]), once a
//! term whichever decider's it is: the next take brings a term whose ids
//! start again. The counter's end stands as the last guard, where a pick is
//! answered `Busy`: under the last term a replica can take no word is said.
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
//! and reaches the client, and no policy spoke. Four arms, each counted
//! by its cause ([`Counts`]): a decision that panics, a request for a side
//! of a pair plan (prefill or decode under either pair plan, encode under
//! the plan that has an encode side: the pair door decides those, over
//! the store's pair classes), a request that carries its prompt as text
//! (the controller is to hash it, and a decision made as if the request
//! carried no prefix would be another request's), and a pick the store
//! mints no lease for, its term's ids being spent. A text is said
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
//! What this step leaves to the next ones. A view reads no session pin of
//! the store's, and its two session counts read zero.

use std::{
    io,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError},
        Arc, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

use super::{
    election::Standing,
    lifecycle,
    wire::{self, Checked, Lists, Viewed},
};
use crate::{
    decider::Decider,
    decision::{Decision, Shed, ShedReason, Trace},
    elector::{term_said, Fence},
    framework::{release, Plan, Scratch},
    lease::{Lease, LeaseId, LeaseKind, Lifecycle, ReleaseReason, Side},
    proto::{self, busy::Reason},
    sessions::SessionVerdict,
    store::{
        leases::DISPATCH_TTL, Ask, AtWork, Built, GatewayId, LeaseRecord, LeaseRefused, LeaseState,
        MemoryStore, Named, Naming, Picked, Said, SessionsStore, ViewBuffers, World,
    },
    view::{
        fingerprint, AffinityFacts, Candidate, Constraints, Phase, PinSource, PoolFacts,
        RequestFacts, Role, Session, WireModes, WorkerId,
    },
};

/// What the deciders and the streams count on the controller's own side,
/// for its scrape to render: each `Busy` by its reason, the requests this
/// build has no decision for by cause, the requests whose block hashes
/// were cut, the answers no stream took, the candidates a gateway's wires
/// took from its decisions, and the decisions cut at their last read with
/// their budget or their lease's dispatch span all but gone. A request that
/// is decided inside both, with its hashes inside the cap and every
/// candidate one its gateway can dial, moves none of them.
#[derive(Debug, Default)]
pub(crate) struct Counts {
    /// `Busy` answers, each at its reason's number on the wire.
    pub(crate) busy: [AtomicU64; 5],
    /// Decisions that panicked, and the lifecycle calls that did as a
    /// decider left its work.
    pub(crate) faults: AtomicU64,
    /// Requests for a side of a pair plan: prefill or decode under either
    /// pair plan, encode under the plan that has an encode side.
    pub(crate) pair_sides: AtomicU64,
    /// Requests that carried their prompt as text.
    pub(crate) texts: AtomicU64,
    /// Requests whose decider's thread had ended.
    pub(crate) gone: AtomicU64,
    /// Answers a stream's queue had no place for. A stream's window rules
    /// it out: its requests in flight are at most its queue's places.
    pub(crate) dropped_full: AtomicU64,
    /// Answers whose stream had ended.
    pub(crate) dropped_gone: AtomicU64,
    /// What a decision that is made may count, each decider's in a part of
    /// its own: read whole by [`Counts::undialed`] and
    /// [`Counts::hashes_cut`].
    parts: [Part; PARTS],
    /// Decisions cut at their last read, with under a microsecond of the
    /// budget or under a millisecond of the lease's dispatch span left. Each
    /// leaves as `Busy` under the reason a late arrival is told, and this
    /// count is of them alone.
    pub(crate) outlasted: AtomicU64,
    /// Picks the store minted no lease for, its term's ids being spent or
    /// the term past the width an id carries. Each leaves as `Busy` with
    /// no reason: a pick with no lease is no answer.
    pub(crate) unleased: AtomicU64,
}

/// The parts a made decision's counts are kept in. A decider adds to the
/// one at its index, and deciders past this many share one.
const PARTS: usize = 16;

/// One decider's part of the counts that every decision can move, on cache
/// lines of its own. Where a fleet's gateways speak fewer wires than its
/// members, each decision counts the candidates it lost; where a host
/// hashes past what its generation reads, each counts its request. One
/// count for every decider would be a line they all write.
#[derive(Debug, Default)]
#[repr(align(128))]
pub(crate) struct Part {
    /// Candidates the wires of their decision's gateway took from its
    /// view, members that share no wire with it.
    pub(crate) undialed: AtomicU64,
    /// Requests that carried more block hashes than their generation
    /// reads of them.
    pub(crate) hashes_cut: AtomicU64,
}

impl Counts {
    /// The reasons of a `Busy`, in the order of their numbers on the wire.
    pub(crate) const BUSY: [&'static str; 5] = [
        "unspecified",
        "queue_depth",
        "no_credit",
        "budget_gone_on_arrival",
        "not_leader",
    ];

    /// The part decider `index` counts in.
    pub(crate) fn part(&self, index: usize) -> &Part {
        &self.parts[index % PARTS]
    }

    /// Candidates the wires took, over every decider.
    pub(crate) fn undialed(&self) -> u64 {
        self.whole(|part| &part.undialed)
    }

    /// Requests whose block hashes were cut, over every decider.
    pub(crate) fn hashes_cut(&self) -> u64 {
        self.whole(|part| &part.hashes_cut)
    }

    /// One of a part's counts, summed over the parts.
    fn whole(&self, count: impl Fn(&Part) -> &AtomicU64) -> u64 {
        let parts = self.parts.iter();
        parts.map(|part| count(part).load(Ordering::Relaxed)).sum()
    }
}

/// The stage a decision is shed under when its pool has rows and its
/// gateway shares a wire with none of them.
pub(crate) const WIRES: &str = "wires";

/// The stage a decision is shed under when its gateway hints a subset of 0
/// of a pool that has rows: a decision over no candidates, as it asked.
pub(crate) const SUBSET_HINT: &str = "subset-hint";

/// The requests one decider's queue holds: one stream's window. At the
/// design's cost of a decision a full queue is one cross-node budget of
/// waiting, and a request behind it is told at once to decide locally.
pub(crate) const QUEUE_DEPTH: usize = 128;

/// The clock a decider reads, twice a decision.
pub(crate) type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// The counter at which a term's leader is told to leave: three quarters of
/// the ids one term mints. A leader that leaves there takes its next term
/// with the counter at its start, so no term reaches the counter's end; the
/// quarter left is what the term still mints between the word and its
/// fence's close.
pub(crate) const TERM_MARK: u32 = 3 << 30;

/// Whether `id` is at or past its term's mark ([`TERM_MARK`]): the counter
/// is the id's low half, as the contract lays an id out. At or past, and
/// not the mark's own id alone: the mint steps over an id on file, and a
/// term's ids need not all pass through one decider.
fn past_mark(id: LeaseId) -> bool {
    (id.0 & u64::from(u32::MAX)) >= u64::from(TERM_MARK)
}

/// What a decider calls once it handed over a term's first answer at or
/// past the mark: the election's planned hand-over, handed that term.
pub(crate) type Leave = Box<dyn Fn(u64) + Send + Sync>;

/// What the deciders share of a term's mark: the word, and the highest term
/// it was said under, so one decider says it and once.
#[derive(Default)]
struct Mark {
    /// The word ([`Deciders::leave_through`]): none until the election runs.
    leave: OnceLock<Leave>,
    /// The highest term whose word was said, or was found to have none to
    /// say: the last term a replica can take.
    said: AtomicU64,
}

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

impl Reads {
    /// The highest term this replica knows, which a `Busy` and a `Credit`
    /// carry: no answer of a replica falls in term, leading or not.
    pub(crate) fn known_term(&self) -> u64 {
        let seen = self.standing.borrow().term;
        seen.max(self.fence.term())
    }

    /// The `Busy` that answers the request `id` for `reason`, under the
    /// highest term this replica knows, counted at its reason.
    pub(crate) fn busy(&self, id: Vec<u8>, reason: Reason) -> proto::DecideAnswer {
        if let Some(count) = self.counts.busy.get(reason as usize) {
            count.fetch_add(1, Ordering::Relaxed);
        }
        wire::busy(id, self.known_term(), reason)
    }
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
    pub(crate) answers: mpsc::Sender<wire::Decided>,
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
    /// What a stream's reader reads beside its deciders: the clock a
    /// frame's receipt is stamped by and the term the reader's own answers
    /// carry.
    reads: Reads,
    /// A term's mark, as every decider reads it.
    mark: Arc<Mark>,
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
        lifecycle::tells(&reads.store);
        let told = Arc::new(watch::Sender::new(false));
        let mark = Arc::new(Mark::default());
        let queues = (0..deciders.max(1))
            .map(|index| {
                let (queue, jobs) = sync_channel(depth.max(1));
                let decider = Deciding {
                    reads: reads.clone(),
                    buffers: Buffers::default(),
                    faulted: false,
                    index,
                    marked: None,
                    mark: Arc::clone(&mark),
                    working: None,
                    name: u8::try_from(index).ok().map(Decider::new),
                };
                watched(index, &told, move || decider.run(&jobs)).map(|_running| queue)
            })
            .collect::<io::Result<_>>()?;
        Ok(Self {
            queues,
            told,
            reads: reads.clone(),
            mark,
        })
    }

    /// Shows `true` once a decider's thread unwound outside a decision: its
    /// models have no decider from then on. The word is awaited by its
    /// value (`wait_for`), never by a change: a receiver made after the
    /// unwind has seen it.
    pub(crate) fn ended(&self) -> watch::Receiver<bool> {
        self.told.subscribe()
    }

    /// Hands the deciders the word a term's mark says: the decider that
    /// hands over a term's first answer at or past the mark ([`past_mark`])
    /// calls `leave` with that term once that answer is handed over. Set
    /// once, when the election runs: a second call changes nothing.
    pub(crate) fn leave_through(&self, leave: impl Fn(u64) + Send + Sync + 'static) {
        let _ = self.mark.leave.set(Box::new(leave));
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

    /// The deciders' clock. A stream's reader stamps a frame's receipt by
    /// it, so a budget is counted on one clock.
    pub(crate) fn now(&self) -> Instant {
        (self.reads.clock)()
    }

    /// The highest term this replica knows ([`Reads::known_term`]).
    pub(crate) fn known_term(&self) -> u64 {
        self.reads.known_term()
    }

    /// A `Busy` of a stream's reader ([`Reads::busy`]).
    pub(crate) fn busy(&self, id: Vec<u8>, reason: Reason) -> proto::DecideAnswer {
        self.reads.busy(id, reason)
    }

    /// What the deciders and their streams count.
    pub(crate) fn counts(&self) -> &Arc<Counts> {
        &self.reads.counts
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
/// allocates only while these grow. A decision is handed to its stream as
/// it was made ([`wire::Decided`]), and the stream builds its frame: a
/// decider makes no list for an answer.
#[derive(Default)]
struct Buffers {
    view: ViewBuffers,
    scratch: Scratch,
    lists: Lists,
    named: Named,
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
struct Deciding {
    reads: Reads,
    buffers: Buffers,
    /// The last decision panicked: a streak of them writes one line.
    faulted: bool,
    /// Its place among the deciders, which names its part of a count.
    index: usize,
    /// The term of the answer in hand where its lease's id is at or past
    /// the term's mark: the word follows the answer's hand-over.
    marked: Option<u64>,
    /// The mark's word and its latch, shared by the deciders.
    mark: Arc<Mark>,
    /// Its place among the threads at work at the leases' file, while it
    /// has a job at hand ([`MemoryStore::at_work`]).
    working: Option<AtWork>,
    /// Its name among the host's deciders, which its thread acts as while
    /// it decides ([`Decider::enter`]): none past the names a host gives.
    name: Option<Decider>,
}

impl Deciding {
    /// Decides until the last handle of the queue is dropped. With a job
    /// at hand the decider is at work at the leases' file: what the
    /// sessions hand is left to it, applied before each decision reads
    /// its rows' counts, where the file's turn is free, and at each
    /// issue. It leaves that work before it
    /// waits on an empty queue: what is handed meanwhile is applied by
    /// the session that hands it.
    fn run(mut self, jobs: &Receiver<Job>) {
        loop {
            let job = match jobs.try_recv() {
                Ok(job) => job,
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {
                    self.rest();
                    let Ok(job) = jobs.recv() else { break };
                    job
                }
            };
            if self.working.is_none() {
                self.working = Some(self.reads.store.at_work());
            }
            // The thread acts as its decider for this job: a local, so the
            // name is let go where the job ends.
            let _acting = self.name.map(Decider::enter);
            let Job {
                request,
                gateway,
                received,
                answers,
            } = job;
            let answer = self.answer(request, gateway, received);
            // An answer its stream does not take is counted, and the lease
            // it carries is given up: no gateway will name a dispatch for
            // it. The word of the answer's term's mark follows either way.
            if let Err(refused) = answers.try_send(answer) {
                let (dropped, unsent) = match refused {
                    mpsc::error::TrySendError::Full(unsent) => {
                        (&self.reads.counts.dropped_full, unsent)
                    }
                    mpsc::error::TrySendError::Closed(unsent) => {
                        (&self.reads.counts.dropped_gone, unsent)
                    }
                };
                dropped.fetch_add(1, Ordering::Relaxed);
                self.unsent(gateway, unsent);
            }
            if let Some(term) = self.marked.take() {
                self.leave_at(term);
            }
        }
        // The queue's last handle is gone: the work drops here too.
        self.rest();
    }

    /// Leaves the decider's work at the leases' file. What was handed
    /// behind its last take is applied here, and the engine's lifecycle
    /// calls run with it, outside any decision ([`Self::held`]).
    fn rest(&mut self) {
        let Some(leaving) = self.working.take() else {
            return;
        };
        self.held(
            "a lease's lifecycle call panicked as its decider left its work: what it left is applied at the decider's next take",
            |_| drop(leaving),
        );
    }

    /// Gives up the lease of an answer its stream did not take: the queue
    /// was full, or the stream had ended. No gateway will name a dispatch
    /// for it, and it would book its worker to its dispatch deadline. The
    /// decider hands its release to the leases' file as abandoned, as a
    /// stream does for a decision it stops at the send, and whoever takes
    /// the turn next applies it: this decider, before its next decision
    /// reads a count. A full ring makes the hand apply in place, outside
    /// any decision ([`Self::held`]).
    #[cold]
    fn unsent(&mut self, gateway: GatewayId, answer: wire::Decided) {
        let wire::Decided::Decision {
            lease: Some(leased),
            ..
        } = answer
        else {
            return;
        };
        self.held(
            "a lease's lifecycle call panicked as its decider gave up an answer no stream took: what it left is applied at the decider's next take",
            |reads| {
                let hand = lifecycle::open(&reads.store, gateway);
                let abandoned = Said::Released(None, ReleaseReason::Abandoned);
                reads.store.hand(&hand, leased.lease.id, abandoned);
            },
        );
    }

    /// Runs what applies handed frames outside a decision. A lifecycle
    /// call that panics there is held as a decision's is: counted with the
    /// faults and told once a streak, as `told`. What it left in the ring
    /// is the next take's.
    fn held(&mut self, told: &'static str, apply: impl FnOnce(&Reads)) {
        let reads = &self.reads;
        let Err(payload) = catch_unwind(AssertUnwindSafe(|| apply(reads))) else {
            return;
        };
        // A payload's own drop can panic: the engine's helper frees it
        // under a catch of its own.
        release(payload);
        self.reads.counts.faults.fetch_add(1, Ordering::Relaxed);
        if !self.faulted {
            error!(
                decider = thread::current().name().unwrap_or_default(),
                "{told}"
            );
        }
        self.faulted = true;
    }

    /// The word of `term`'s mark: the planned hand-over, so the next take
    /// brings a term whose ids start again. Once a term: every answer past
    /// the mark comes here, and the first alone goes on. Under the last
    /// term below the ceiling no word is said: a hand-over there leaves no
    /// term to take, and the counter runs on to its end, where a pick is
    /// answered `Busy` ([`Counts::unleased`]).
    #[cold]
    fn leave_at(&self, term: u64) {
        // A read first: past the mark every answer comes here until the
        // hand-over lands, and a write each time would pass the line from
        // decider to decider.
        if self.mark.said.load(Ordering::Relaxed) >= term
            || self.mark.said.fetch_max(term, Ordering::Relaxed) >= term
        {
            return;
        }
        if term_said(Some(term)).is_err() {
            warn!(
                term,
                "the last term a replica can take passed the mark of its lease ids: no hand-over is said, and picks are answered Busy once the ids end"
            );
            return;
        }
        let Some(leave) = self.mark.leave.get() else {
            warn!(
                term,
                "a term's lease ids passed their mark and no election listens: no hand-over is said, and picks are answered Busy once the ids end"
            );
            return;
        };
        info!(
            term,
            "the term's lease ids passed their mark: this leader hands the lease over, and the next take mints under a new term"
        );
        leave(term);
    }

    /// One request's answer: its decision, or `Busy`.
    fn answer(&mut self, request: Checked, gateway: GatewayId, received: Instant) -> wire::Decided {
        // What the claim rests on: a decision does not write the fence, and
        // writes the store at its lease's issue and, where its last read
        // fails, at that lease's release, each whole or not at all under a
        // lock no panic poisons; the engine closes an open scope where it
        // unwinds, and the buffers a fault leaves are dropped whole below.
        let decided = catch_unwind(AssertUnwindSafe(|| {
            self.decide(&request, gateway, received)
        }));
        let faulted = decided.is_err();
        let answer = match decided {
            Ok(Ok((term, decision, lease))) => {
                // An id at or past the term's mark: the word follows this
                // answer, where no answer of the term said it yet.
                if lease.is_some_and(|leased| past_mark(leased.lease.id)) {
                    self.marked = Some(term);
                }
                // The decision leaves as it was made: its stream builds
                // the frame where it sends it.
                wire::Decided::Decision {
                    id: request.into_id(),
                    term,
                    decision,
                    lease,
                }
            }
            Ok(Err(reason)) => wire::Decided::Frame(self.reads.busy(request.into_id(), reason)),
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
                wire::Decided::Frame(self.reads.busy(request.into_id(), Reason::Unspecified))
            }
        };
        self.faulted = faulted;
        answer
    }

    /// One request's decision with the term both reads of the fence showed
    /// and, where it is a pick, the lease issued for it with what is left
    /// of its dispatch span, or the reason its answer is `Busy`.
    fn decide(
        &mut self,
        request: &Checked,
        gateway: GatewayId,
        received: Instant,
    ) -> Result<(u64, Decision, Option<wire::Leased>), Reason> {
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
        // The store's pins, lent to the decision: its engine asks them for
        // a key its own table holds no pin of, as after an apply of a new
        // generation.
        let pins = StorePins { store, now };
        let taken = Taken::at(request, gateway, received, now)
            .ok_or(Reason::BudgetGoneOnArrival)?
            .lent(&pins);
        let (buffers, own) = (&mut self.buffers, counts.part(self.index));
        // What the sessions handed is applied before the rows' counts are
        // read, where the turn is free.
        if let Some(at_work) = &self.working {
            at_work.apply();
        }
        // One load of the world for the decision and for its lease: the
        // lease books the count that load laid beside its target's row,
        // and the engine told of it is the one that decided.
        store.deciding(|world| {
            let Some((decision, booking)) = decide_in(world, &taken, buffers, own) else {
                counts.pair_sides.fetch_add(1, Ordering::Relaxed);
                return Err(Reason::Unspecified);
            };
            // The lease, for a pick: filed under the first read's term and
            // at its instant, and the engine told, before the read that
            // lets the answer leave. That read judges the store's write
            // and a stage's call with the build before them.
            let filed = booking
                .map(|booking| issue(store, world, &taken, term, &decision, booking))
                .transpose();
            // The fence, read again where the answer is handed over, with
            // the hand-over alone behind it: the answer leaves under
            // the term it was made under, or `Busy` leaves in its place.
            // The budget is judged at that read too, with no clock read
            // more: a decision late by this clock since the receipt is
            // late at its gateway, whose clock started earlier. It leaves
            // as `Busy` under the reason a late arrival is told, and under
            // a count of its own.
            let handed = clock();
            // The decision's length, one difference of its two reads, is
            // what the budget and a lease's dispatch span are judged by.
            // The span counts from the first read: the answer tells its
            // gateway what is left of it, and a lease with under a
            // millisecond left is none to hand over.
            let since = taken.since(handed);
            let left = DISPATCH_TTL.saturating_sub(since);
            let spent = matches!(filed, Ok(Some(_))) && left < wire::Leased::LEAST;
            let failed = if fence.leader_term(handed) != Some(term) {
                Some((Reason::NotLeader, ReleaseReason::Abandoned))
            } else if taken.outlasted(since) || spent {
                counts.outlasted.fetch_add(1, Ordering::Relaxed);
                Some((Reason::BudgetGoneOnArrival, ReleaseReason::LateDecision))
            } else {
                None
            };
            let Some((busy, ended)) = failed else {
                return match filed {
                    Ok(lease) => Ok((
                        term,
                        decision,
                        lease.map(|lease| wire::Leased {
                            lease,
                            left,
                            due: taken.due(),
                        }),
                    )),
                    // The term mints no more ids. A pick with no lease is
                    // no answer: its gateway decides it locally.
                    Err(LeaseRefused::Unminted { lease, store }) if lease == store => {
                        counts.unleased.fetch_add(1, Ordering::Relaxed);
                        Err(Reason::Unspecified)
                    }
                    // The store stands under another term than the fence
                    // showed: a take or a loss this decision straddled.
                    Err(_) => Err(Reason::NotLeader),
                };
            };
            // The answer does not leave, and the lease filed for it goes
            // with it.
            if let Ok(Some(lease)) = filed {
                give_up(store, world, taken.gateway, &lease, ended);
            }
            Err(busy)
        })
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
    /// The pins of record lent to the decision, where its host keeps them:
    /// its engine asks them for a key its own table holds no pin of.
    pins: Option<&'a dyn PinSource>,
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
            pins: None,
        })
    }

    /// The request with the pins of record lent to its decision.
    fn lent(self, pins: &'a dyn PinSource) -> Self {
        Self {
            pins: Some(pins),
            ..self
        }
    }

    /// The time since the request was taken at `later`, a read of the
    /// clock after that one. One difference of two instants judges the
    /// budget and a lease's dispatch span, where taking the request again
    /// would read its budget and build it whole.
    fn since(&self, later: Instant) -> Duration {
        later.saturating_duration_since(self.now)
    }

    /// When the dispatch span of a lease filed for the request ends: the
    /// span counts from the read the request was taken at.
    fn due(&self) -> Instant {
        self.now + DISPATCH_TTL
    }

    /// Whether the budget is gone `since` the request was taken: what was
    /// left then, less that time, under the same rule.
    fn outlasted(&self, since: Duration) -> bool {
        self.left.saturating_sub(since) < Duration::from_micros(1)
    }
}

/// The store's pins of record as a decision's engine reads them
/// ([`PinSource`]): asked for a key the engine's table holds no pin of,
/// under no lock but the pins' own.
struct StorePins<'a> {
    store: &'a MemoryStore,
    /// The decision's instant: a pin's end and its age are read against
    /// it.
    now: Instant,
}

impl std::fmt::Debug for StorePins<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorePins").finish_non_exhaustive()
    }
}

impl PinSource for StorePins<'_> {
    fn pinned(&self, key: Session, ttl_micros: Option<u64>) -> Option<WorkerId> {
        let pin = SessionsStore::lookup(self.store, key.get())?;
        // A pin at or past its own end is no pin, whether or not the sweep
        // has run: its end is the instant the engine's table forgot the
        // key. A TTL counted from the dispatch ends later than that, and
        // so may the TTL of a rule applied since. The end stands on the
        // decision clock: the ask's instant is turned into it here, at a
        // miss that finds a pin with an end and nowhere else.
        let ended = |ends: u64| ends <= self.store.micros(self.now);
        if pin.ends.is_some_and(ended) {
            return None;
        }
        // The asking rule's TTL holds as well, against the pin's last use:
        // a pin as old as that TTL is no pin, to the microsecond.
        let age = self.now.saturating_duration_since(pin.last_used);
        ttl_micros
            .is_none_or(|ttl| age < Duration::from_micros(ttl))
            .then_some(pin.worker)
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
/// entry of the engine never does. What a decision that is made counts, it
/// counts in `own`, its decider's part.
fn decide_in(
    world: &World,
    taken: &Taken<'_>,
    buffers: &mut Buffers,
    own: &Part,
) -> Option<(Decision, Option<Booking>)> {
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
    if taken.request.hashes() > taken.request.hashes_read(spec) {
        own.hashes_cut.fetch_add(1, Ordering::Relaxed);
    }
    let ((decision, booking), built) = hand(
        world,
        taken,
        buffers,
        &own.undialed,
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
                let none = shed(
                    ShedReason::NoEndpoints,
                    emptied_by,
                    facts.seed,
                    pool.generation,
                );
                return (none, None);
            }
            let decision = engine.decide(facts, pool, candidates, constraints, scratch);
            // What a pick's lease takes of its request, read where the
            // request stands as the engine read it.
            let booking = decision.pick().and_then(|pick| {
                Some(Booking {
                    target: pick.targets.first()?.id,
                    picked: Picked::of(&pick.targets),
                    verdict: pick.session_verdict,
                    generation: pick.trace.generation,
                    session: facts.session(),
                    tokens: facts.prompt_tokens,
                    now_micros: facts.now_micros,
                    built: None,
                })
            });
            (decision, booking)
        },
    );
    Some((
        decision,
        booking.map(|booking| Booking { built, ..booking }),
    ))
}

/// What a pick's lease takes of its decision's hand-over.
#[derive(Clone, Copy)]
struct Booking {
    /// The pick's first target: the worker the lease books.
    target: WorkerId,
    /// The pick's other targets and the ranks, for a `Dispatched` that
    /// names one by its index.
    picked: Picked,
    /// What the decision did with the request's session key.
    verdict: SessionVerdict,
    /// The generation that decided.
    generation: u64,
    /// The request's session; none for a request with no key.
    session: Session,
    /// The token-work the lease books: the prompt, as its host counted it.
    tokens: u32,
    /// The decision's instant in the rows' clock.
    now_micros: u64,
    /// The build the decision read: where its target's count stands.
    built: Option<Built>,
}

/// Issues the lease of a pick, one a decision, ahead of the read that lets
/// its answer leave: a credit on the pick's first target, minted and filed
/// by the store under `term`, the fence's at the decision's first read, at
/// that read's instant, from which [`DISPATCH_TTL`] runs for its gateway
/// to name a dispatch. The lease books the count the decision's own world
/// lays beside its target's row, and the engine that decided is told of
/// it.
///
/// # Errors
///
/// The store's refusal: it stands under another term, or mints no id
/// under this one.
fn issue(
    store: &MemoryStore,
    world: &World,
    taken: &Taken<'_>,
    term: u64,
    decision: &Decision,
    booking: Booking,
) -> Result<Lease, LeaseRefused> {
    let lease = Lease {
        // The store mints the id.
        id: LeaseId::default(),
        kind: LeaseKind::Credit,
        side: Side::Single,
        target_index: 0,
        generation: booking.generation,
        session: booking.session,
        session_verdict: booking.verdict,
        issued_micros: booking.now_micros,
        tokens: booking.tokens,
    };
    let record = LeaseRecord {
        lease,
        gateway: taken.gateway,
        request: fingerprint(taken.request.id()),
        target: booking.target,
        picked: booking.picked,
        decode: None,
        term,
        issued_at: taken.now,
        dispatch_deadline: taken.due(),
        state: LeaseState::Issued,
    };
    let built = booking.built;
    let booked = built.and_then(|built| world.booked(&built, booking.target));
    // Under one hold of the turn the store applies what is handed, files
    // the lease, tells the engine that decided of it, and ends and tells
    // the leases it finds past their dispatch deadline.
    let engine = world.spec().engine();
    let lease = store.issue_decided(record, booked, |lease| {
        engine.on_decision_issued(lease, decision);
    })?;
    Ok(lease)
}

/// Gives up the lease of a decision whose answer does not leave: the
/// store's release door first ([`crate::store::LeasesStore::released`],
/// which applies what is handed ahead of it), then the engine that heard
/// the issue, told once where the store held the lease. A lease the store
/// no longer holds was ended by another hand, and nothing is told of it
/// here.
fn give_up(
    store: &MemoryStore,
    world: &World,
    gateway: GatewayId,
    lease: &Lease,
    reason: ReleaseReason,
) {
    if crate::store::LeasesStore::released(store, gateway, lease.id, None, reason) {
        world.spec().engine().on_released(lease, reason);
    }
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

/// What [`hand`] reads of a pool for one request: the pool's facts, the
/// decision's instant in the rows' clock, the rows, the mask of the asking
/// gateway's vetoes beside lent rows, the affinity column, the leases'
/// counts beside lent rows, and the build the rows were read by.
type Read<'a> = (
    PoolFacts<'a>,
    u64,
    &'a [Candidate],
    &'a [u32],
    &'a [AffinityFacts],
    Option<crate::LentCounts<'a>>,
    Option<Built>,
);

/// Hands `decide` what the engine reads for one request: the request with
/// what is left of its budget, the pool's facts under the generation its
/// rows were published under, and those rows as the asking gateway reaches
/// them at the decision's instant, with the affinity column where the
/// request's profile declares it. A request that states no wires and no
/// hint of 0 reads its pool by the store's one door ([`World::view`]): the
/// rows where their publication left them, with its gateway's standing
/// vetoes and the leases' counts beside them, or rows the door writes
/// where the request's profile shows them to a plugin. One that states
/// wires or a hint of 0 is handed rows written for it ([`World::build`]),
/// what the leases on file book on each row's worker written over them
/// ([`World::count_in_flight`]), less what the gateway states of itself
/// takes ([`narrow`]). The rows its wires take are counted in `undialed`,
/// the decider's own part, and the list they leave takes a name of its
/// own ([`cut_reach`]). The request's block hashes are cut at what that
/// generation reads of them. `decide`'s answer comes back with the build
/// its rows were read by, `None` where no member serves the model.
///
/// A request whose profile reads the affinity column and takes a sample is
/// handed no column: the rows the store names for it stand in the column's
/// place ([`World::name`]), at most that sample of each list
/// ([`Engine::named_for`](crate::framework::Engine::named_for)).
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
) -> (R, Option<Built>) {
    let Taken {
        request,
        gateway,
        now,
        left,
        pins,
    } = *taken;
    let spec = world.spec();
    let (model, phase) = (request.model(), request.phase());
    let engine = spec.engine();
    // What the decision's profile reads, and whether it shows its rows to
    // a plugin.
    let (needs, shown) = engine.reads_for(model, phase);
    // A profile whose picker takes a sample is handed the rows the store
    // names in the column's place, capped by that sample, and no column.
    // The deciding profile answers for itself: the needs are merged over
    // a pair's two phases and a sessions rule, so they say only where to
    // ask, and no other decision pays the second walk.
    let cap = if needs.affinity && needs.sample > 0 {
        engine.named_for(model, phase)
    } else {
        0
    };
    let names = cap > 0;
    let ask = Ask {
        gateway,
        now,
        pinned: None,
        affinity: needs.affinity && !names,
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
        named,
    } = buffers;
    let wires = request.wires();
    // No member serves the model in the role: the engine answers over no
    // candidate, as it answers any host that hands it none.
    let none = || PoolFacts {
        generation: spec.generation(),
        ..PoolFacts::default()
    };
    // Nothing of the request's own is cut out of the rows the door hands:
    // the publication's own, the leases' counts beside them, or a copy the
    // door writes with the counts on it for a plugin, which is shown a
    // row's bits and ages as the row's own fields.
    let by_door = wires.is_none() && request.subset() != Some(0);
    let (pool, now_micros, candidates, stalled, affinity, counts, built): Read<'_> = if by_door {
        match world.view(model, role, &ask, shown, view) {
            Some(read) => (
                read.built.pool_facts(),
                read.built.now_micros,
                read.candidates,
                read.stalled,
                read.affinity,
                read.in_flight,
                Some(read.built),
            ),
            None => (none(), world.micros(now), &[], &[], &[], None, None),
        }
    } else {
        let built = world.build(model, role, &ask, view);
        let (pool, now_micros) = match built {
            Some(built) => {
                // What the leases on file book on each row's worker,
                // written before the request's own narrowing moves a row:
                // rows that are written carry their counts. A request that
                // keeps no row is counted nothing.
                if keeps_a_row(wires, request.subset()) {
                    world.count_in_flight(&built, &ask, &mut view.candidates);
                }
                // The pool's facts are of the rows before the request's own
                // narrowing, all but the list's name: the rows the wires
                // left are another list of the pool.
                let mut pool = built.pool_facts();
                let took = narrow(view, wires, request.subset());
                if took > 0 {
                    undialed.fetch_add(took, Ordering::Relaxed);
                    if let Some(wires) = wires {
                        pool.reach = cut_reach(pool.reach, wires);
                    }
                }
                (pool, built.now_micros)
            }
            None => (none(), world.micros(now)),
        };
        (
            pool,
            now_micros,
            &view.candidates,
            &[],
            &view.affinity,
            None,
            built,
        )
    };
    if names {
        // A row is named by its place among the rows the decision is made
        // over: the lent rows, or the built ones after their narrowing.
        let hashes = request.block_hashes();
        let naming = Naming {
            gateway,
            // The first hashes, as many as the generation reads of them:
            // the engine's request takes the same count.
            block_hashes: hashes.get(..request.hashes_read(spec)).unwrap_or(hashes),
            seed: request.seed(),
            cap,
            wires,
            now_micros,
        };
        world.name(model, role, &naming, candidates, named);
    }
    let viewed = Viewed {
        now_micros,
        budget_micros: u32::try_from(left.as_micros()).unwrap_or(u32::MAX),
        affinity,
        stalled,
        in_flight: counts,
        max_block_hashes: request.hashes_read(spec),
        named: names.then(|| named.affinity()),
        pins,
    };
    let answer = request.read(lists, viewed, |facts, constraints| {
        decide(facts, &pool, candidates, constraints, scratch)
    });
    (answer, built)
}

/// Whether the request's own narrowing can leave a row: a subset of none
/// and a mask of no wire clear every row ([`narrow`]).
fn keeps_a_row(wires: Option<WireModes>, subset: Option<u32>) -> bool {
    subset != Some(0) && wires != Some(WireModes::NONE)
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

/// The name of the list a gateway's wires cut out of a build's
/// ([`PoolFacts::reach`]): the build's word turned by the wires. Each of the
/// eight sets of wires turns it by its own multiple of one odd word, eight
/// words that differ and none of them zero: no cut reads as its build's
/// whole list, and no two cuts of one build read alike.
fn cut_reach(built: u64, wires: WireModes) -> u64 {
    built ^ (u64::from(wires.bits()) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[cfg(test)]
pub(crate) mod tests {
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
        cut_reach, hand, keeps_a_row, narrow, watched, Buffers, Clock, Counts, Deciders, Deciding,
        Job, Part, Reads, Refused, Taken, PARTS, TERM_MARK,
    };
    use crate::{
        builtins::Random,
        catalog::Catalog,
        controller::{
            election::Standing,
            gateways::tests::PATIENT,
            lifecycle,
            wire::{self, Checked},
        },
        decider::Decider,
        decision::{Decision, Targets},
        elector::{Elector, Tick, TERM_CEILING},
        framework::{DecisionScope, Filter, Picker, Scratch, TieBand},
        lease::{Lease, LeaseId, Lifecycle, ReleaseReason},
        proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome},
        spec::{AppliedSpec, PolicySpec, PowerOfTwoMetric},
        store::{
            leases::DISPATCH_TTL, Ask, Dispatch, GatewayId, HealthSink, LeaseState, LeasesStore,
            MemoryStore, Said, ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore,
        },
        view::{
            fingerprint, AffinityFacts, Candidate, Constraints, PoolFacts, RequestFacts, Role,
            WireModes, WorkerId,
        },
    };

    /// How long a case waits for a decider before it fails.
    pub(crate) const PATIENCE: Duration = Duration::from_secs(20);
    pub(crate) const MODEL: &str = "model-a";
    pub(crate) const GATEWAY: GatewayId = GatewayId(11);
    const OTHER: GatewayId = GatewayId(12);
    /// The budget a case's frame gives its decision.
    pub(crate) const BUDGET: Duration = Duration::from_secs(5);
    /// Past the renew deadline of the bed's election.
    const FENCED: Duration = Duration::from_secs(1_801);

    /// What one read of the case's clock does before it shows the hour.
    pub(crate) enum Step {
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
    pub(crate) struct Script {
        /// When the bed's candidate took its lease.
        pub(crate) taken: Instant,
        pub(crate) steps: Mutex<VecDeque<Step>>,
        pub(crate) shown: Mutex<Duration>,
        pub(crate) readers: Mutex<Vec<String>>,
    }

    impl Script {
        pub(crate) fn read(&self) -> Instant {
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
        sender: mpsc::Sender<wire::Decided>,
        answers: mpsc::Receiver<wire::Decided>,
    }

    fn applied(document: &PolicySpec) -> AppliedSpec {
        AppliedSpec::validate(document).expect("the document validates")
    }

    /// A preset of this build as generation `generation`.
    pub(crate) fn preset(name: &str, generation: u64) -> AppliedSpec {
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
    pub(crate) fn fleet(store: &MemoryStore) -> (Vec<WorkerId>, [(proto::Phase, WorkerId); 3]) {
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
            // The election's report tells the store the term it took: a
            // lease is minted under the store's own term.
            store.set_term(1);
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
    pub(crate) fn frame_for(model: &str, id: u8, seed: u64) -> proto::DecideRequest {
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

    pub(crate) fn frame(id: u8, seed: u64) -> proto::DecideRequest {
        frame_for(MODEL, id, seed)
    }

    /// `frame` with `budget` in place of the case's.
    fn budgeted(mut frame: proto::DecideRequest, budget: Duration) -> proto::DecideRequest {
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        facts.budget_micros = u32::try_from(budget.as_micros()).expect("a budget of seconds");
        frame
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

        /// The next answer, within the case's patience, as its stream
        /// would send it.
        async fn answer(&mut self) -> proto::DecideAnswer {
            timeout(PATIENCE, self.answers.recv())
                .await
                .expect("an answer within the case's patience")
                .expect("the bed holds a sender")
                .frame()
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
            wire::decided(frame.request_id.clone(), term, &decision, None)
        }
    }

    /// The reason of a `Busy` answer.
    pub(crate) fn busy(answer: &proto::DecideAnswer) -> Reason {
        match &answer.answer {
            Some(Answer::Busy(busy)) => busy.reason(),
            other => panic!("a Busy: {other:?}"),
        }
    }

    /// The outcome of a decision's answer.
    pub(crate) fn outcome(answer: &proto::DecideAnswer) -> &Outcome {
        match &answer.answer {
            Some(Answer::Decision(proto::Decision {
                outcome: Some(outcome),
                ..
            })) => outcome,
            other => panic!("a decision: {other:?}"),
        }
    }

    /// The worker a pick names first.
    pub(crate) fn picked(answer: &proto::DecideAnswer) -> WorkerId {
        match outcome(answer) {
            Outcome::Pick(pick) => WorkerId(pick.targets.first().expect("a target").id),
            other => panic!("a pick: {other:?}"),
        }
    }

    /// The one lease a pick's answer carries.
    pub(crate) fn lease_of(answer: &proto::DecideAnswer) -> &proto::Lease {
        match &answer.answer {
            Some(Answer::Decision(decision)) => match decision.leases.as_slice() {
                [lease] => lease,
                other => panic!("one lease: {other:?}"),
            },
            other => panic!("a decision: {other:?}"),
        }
    }

    /// `answer` less the lease it carries: what the engine alone decided.
    fn unleased(mut answer: proto::DecideAnswer) -> proto::DecideAnswer {
        if let Some(Answer::Decision(decision)) = &mut answer.answer {
            decision.leases.clear();
        }
        answer
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
            // view, under the generation that view was published under,
            // with its lease.
            assert_eq!(unleased(answer), bed.engines(&frame, 1));
        }
    }

    /// A decision waits in its stream's queue as it was made, with the
    /// lease issued for its pick and what is left of that lease's span: its
    /// frame is its stream's to build. A `Busy` is handed over as its
    /// frame.
    #[tokio::test]
    async fn a_decision_is_handed_to_its_stream_as_it_was_made_and_a_busy_as_its_frame() {
        let steps = [Duration::ZERO, Duration::ZERO, BUDGET].map(Step::At);
        let mut bed = leading(steps);
        assert_eq!(bed.submit(frame(1, 1)), None);
        let handed = timeout(PATIENCE, bed.answers.recv()).await;
        let Ok(Some(wire::Decided::Decision {
            id,
            term,
            decision,
            lease: Some(leased),
        })) = handed
        else {
            panic!("a decision as it was made, with its lease: {handed:?}");
        };
        assert_eq!((id.as_slice(), term), (&[1][..], 1));
        let first = decision.pick().and_then(|pick| pick.targets.first());
        let first = first.expect("a pick names a target");
        assert!(bed.workers.contains(&first.id), "{decision:?}");
        assert_eq!(leased.lease.id, LeaseId(1 << 32 | 1));
        assert_eq!(leased.left, DISPATCH_TTL);
        // A request taken with its budget gone is answered where it is
        // taken, and the answer carries no list.
        assert_eq!(bed.submit(frame(2, 2)), None);
        let handed = timeout(PATIENCE, bed.answers.recv()).await;
        let Ok(Some(wire::Decided::Frame(told))) = handed else {
            panic!("a Busy as its frame: {handed:?}");
        };
        assert_eq!(busy(&told), Reason::BudgetGoneOnArrival);
        assert_eq!(told.request_id, [2]);
    }

    /// A member's requests in flight are the leases on file that book it:
    /// a decision by the fewest requests goes where the decisions before
    /// it left the fewest, and a lease that ends gives its worker the next.
    #[tokio::test]
    async fn a_decision_by_the_fewest_requests_reads_the_leases_the_decisions_before_it_left() {
        let mut document = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
            .expect("a form of this build's preset");
        document.generation = 1;
        let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
        // Nine decisions and no word from their gateway: each round of
        // three names each of the three members once.
        let mut held = Vec::new();
        for round in 0..3u8 {
            let mut named = BTreeSet::new();
            for turn in 1..=3u8 {
                let id = round * 3 + turn;
                let answer = bed.ask(frame(id, u64::from(id))).await;
                named.insert(picked(&answer));
                held.push((picked(&answer), LeaseId(lease_of(&answer).id)));
            }
            assert_eq!(named.len(), 3, "round {round} named {named:?}");
        }
        // One member's three leases end at their gateway's word: the next
        // three decisions are that member's.
        let (freed, _) = held[0];
        for &(_, lease) in held.iter().filter(|(worker, _)| *worker == freed) {
            let ended = bed
                .store
                .released(GATEWAY, lease, None, ReleaseReason::ClientGone);
            assert!(ended, "{lease:?} was on file");
        }
        for id in 10..=12u8 {
            let answer = bed.ask(frame(id, u64::from(id))).await;
            assert_eq!(picked(&answer), freed, "request {id}");
        }
    }

    #[tokio::test]
    async fn a_picks_answer_carries_the_lease_the_store_filed_under_the_fences_term() {
        let mut bed = leading([]);
        let answer = bed.ask(frame(7, 7)).await;
        let lease = *lease_of(&answer);
        assert_eq!(lease.id, 1 << 32 | 1, "the term over its first count");
        assert_eq!(
            (lease.term, lease.origin(), lease.kind(), lease.side()),
            (
                answer.term,
                proto::LeaseOrigin::Controller,
                proto::LeaseKind::Credit,
                proto::Side::Single
            )
        );
        assert_eq!(
            (lease.target_index, lease.generation, lease.session_key),
            (0, 1, 0)
        );
        assert_eq!(u128::from(lease.dispatch_ttl_ms), DISPATCH_TTL.as_millis());
        let store: &dyn LeasesStore = bed.store.as_ref();
        let filed = store
            .get(GATEWAY, LeaseId(lease.id))
            .expect("the lease is on file");
        assert_eq!(
            (filed.target, filed.term, filed.state, filed.request),
            (picked(&answer), 1, LeaseState::Issued, fingerprint(&[7]))
        );
        assert_eq!(filed.dispatch_deadline, filed.issued_at + DISPATCH_TTL);
        // The next decision's id is the term's next count, its lease
        // carries its request's prompt tokens, and both leases stand for
        // their gateway alone.
        assert_eq!(lease.tokens, 0, "a request that states no prompt");
        let mut prompted = frame(8, 8);
        let facts = prompted.facts.as_mut();
        let facts = facts.expect("the case's frame has facts");
        facts.prompt_tokens = 48;
        facts.session_key = Some(proto::Identity {
            form: Some(proto::identity::Form::Hash(17)),
        });
        let next = bed.ask(prompted).await;
        assert_eq!(
            (lease_of(&next).id, lease_of(&next).tokens),
            (1 << 32 | 2, 48)
        );
        let store: &dyn LeasesStore = bed.store.as_ref();
        assert_eq!((store.live(GATEWAY), store.live(OTHER)), (2, 0));
        // The lease carries its request's session, on the wire as on file:
        // never zero for a request that states one.
        let session = lease_of(&next).session_key;
        let filed = store.get(GATEWAY, LeaseId(1 << 32 | 2));
        let filed = filed.expect("the lease is on file").lease;
        assert_ne!(session, 0);
        assert_eq!(filed.session.get(), session);
    }

    #[tokio::test]
    async fn an_answer_that_names_no_worker_or_does_not_leave_files_no_lease() {
        // A shed: no member serves the model.
        let mut bed = leading([]);
        let answer = bed.ask(frame_for("model-none", 1, 1)).await;
        assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
        assert_eq!(unleased(answer.clone()), answer, "a shed carries no lease");
        assert_eq!(bed.store.live(GATEWAY), 0);
        // A decision whose fence closed at its hand-over, and one whose
        // budget ended while it was made: `Busy`, and nothing on file.
        for (second_read, reason) in [
            (FENCED, Reason::NotLeader),
            (BUDGET, Reason::BudgetGoneOnArrival),
        ] {
            let mut bed = leading([Step::At(Duration::ZERO), Step::At(second_read)]);
            assert_eq!(busy(&bed.ask(frame(2, 2)).await), reason);
            assert_eq!(bed.store.live(GATEWAY), 0, "{reason:?}");
        }
    }

    #[tokio::test]
    async fn a_pick_the_store_mints_no_lease_for_leaves_as_busy() {
        // The term's last id, then none: `Busy` with no reason, counted.
        let mut bed = leading([]);
        bed.store.spend_ids(1);
        let last = bed.ask(frame(1, 1)).await;
        assert_eq!(lease_of(&last).id, 1 << 32 | u64::from(u32::MAX));
        for id in 2..=3 {
            assert_eq!(busy(&bed.ask(frame(id, 1)).await), Reason::Unspecified);
        }
        assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 2);
        assert_eq!(bed.store.live(GATEWAY), 1, "the last id's lease alone");
        // A store that stands under a later term than the fence showed:
        // not this leader's to answer, and no lease.
        let mut bed = leading([]);
        bed.store.set_term(2);
        assert_eq!(busy(&bed.ask(frame(4, 1)).await), Reason::NotLeader);
        assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 0);
        assert_eq!(bed.store.live(GATEWAY), 0);
    }

    /// Sixteen requests of one stream in flight at once, and their answers:
    /// a pick's lease id, or the reason of a `Busy`.
    pub(crate) async fn burst(
        deciders: &Deciders,
        sender: &mpsc::Sender<wire::Decided>,
        answers: &mut mpsc::Receiver<wire::Decided>,
    ) -> Vec<Result<u64, Reason>> {
        for id in 0..16 {
            let job = Job {
                request: checked(frame(id, u64::from(id))),
                gateway: GATEWAY,
                received: Instant::now(),
                answers: sender.clone(),
            };
            assert!(deciders.submit(job).is_none(), "the queue takes sixteen");
        }
        let mut read = Vec::new();
        for _ in 0..16 {
            let answer = timeout(PATIENCE, answers.recv())
                .await
                .expect("an answer within the case's patience")
                .expect("the case holds a sender")
                .frame();
            read.push(match &answer.answer {
                Some(Answer::Decision(decision)) => match decision.leases.as_slice() {
                    [lease] => Ok(lease.id),
                    other => panic!("one lease a pick: {other:?}"),
                },
                _ => Err(busy(&answer)),
            });
        }
        read
    }

    /// The word a term's mark says, as a case hears it: each call tells the
    /// case, then waits for its go.
    fn heard(bed: &Bed) -> (std_mpsc::Receiver<()>, std_mpsc::SyncSender<()>) {
        let (reached, words) = std_mpsc::sync_channel(8);
        let (go, held) = std_mpsc::sync_channel::<()>(8);
        let held = Mutex::new(held);
        bed.deciders.leave_through(move |_term| {
            reached.send(()).expect("the case hears the word");
            held.lock().recv().expect("the case lets the decider go");
        });
        (words, go)
    }

    /// The answer that carries the id at a term's mark says the word to
    /// leave, only once it is handed over, and no id past it says another.
    #[tokio::test]
    async fn the_id_at_a_terms_mark_says_the_word_once_its_answer_is_handed_over() {
        let mut bed = leading([]);
        let (words, go) = heard(&bed);
        // The mark is three quarters of a term's ids. The id under it, the
        // mark's, and three past it.
        assert_eq!(u64::from(TERM_MARK) * 4, 3 << 32);
        bed.store.spend_ids(u32::MAX - TERM_MARK + 2);
        let under = bed.ask(frame(1, 1)).await;
        assert_eq!(lease_of(&under).id, 1 << 32 | u64::from(TERM_MARK - 1));
        // The decider is held inside the word: an answer that reaches the
        // case was handed over before the word was said.
        let at = bed.ask(frame(2, 1)).await;
        assert_eq!(lease_of(&at).id, 1 << 32 | u64::from(TERM_MARK));
        words
            .recv_timeout(PATIENCE)
            .expect("the mark's decider says the word");
        // A later word finds its go waiting. The ids past the mark say
        // none: the answer after each shows its decider said nothing.
        for _ in 0..8 {
            go.send(()).expect("the decider holds the other end");
        }
        for (id, past) in [(3, 1), (4, 2), (5, 3)] {
            let answer = bed.ask(frame(id, 1)).await;
            assert_eq!(lease_of(&answer).id, 1 << 32 | u64::from(TERM_MARK + past));
        }
        assert!(words.try_recv().is_err(), "one answer a term says the word");
        assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 0);
    }

    /// The mark's own id need not be one an answer carries: the mint steps
    /// over an id on file. The first answer past the mark says the word
    /// then, and once.
    #[tokio::test]
    async fn an_id_past_a_terms_mark_says_the_word_the_marks_own_never_said() {
        let mut bed = leading([]);
        let (words, go) = heard(&bed);
        for _ in 0..8 {
            go.send(()).expect("the decider holds the other end");
        }
        // The term's counter stands two ids past its mark.
        bed.store.spend_ids(u32::MAX - TERM_MARK - 2);
        let first = bed.ask(frame(1, 1)).await;
        assert_eq!(lease_of(&first).id, 1 << 32 | u64::from(TERM_MARK + 3));
        words
            .recv_timeout(PATIENCE)
            .expect("the first answer past the mark says the word");
        // The answer after each shows its decider said nothing.
        for (id, past) in [(2, 4), (3, 5)] {
            let answer = bed.ask(frame(id, 1)).await;
            assert_eq!(lease_of(&answer).id, 1 << 32 | u64::from(TERM_MARK + past));
        }
        assert!(words.try_recv().is_err(), "one answer a term says the word");
    }

    /// Each term says its own word: a leader past its mark under one term,
    /// then under the next, hands the lease over twice.
    #[tokio::test]
    async fn two_terms_in_a_row_each_say_the_word_at_their_mark() {
        let mut bed = leading([]);
        let (words, go) = heard(&bed);
        let taken = bed.script.taken;
        for term in 1..=2 {
            go.send(()).expect("the decider holds the other end");
            if term > 1 {
                // The bed's candidate, alone over its lease, leaves and
                // takes again at one instant, as its election would.
                let left = bed.elector.step_down(None, taken);
                assert!(left.expect("a planned hand-over"));
                let took = bed.elector.tick(taken).expect("the take");
                assert_eq!(took, Tick::Acquired { term });
                bed.store.set_term(term);
            }
            bed.store.spend_ids(u32::MAX - TERM_MARK + 1);
            let at = bed.ask(frame(1, 1)).await;
            assert_eq!(lease_of(&at).id, term << 32 | u64::from(TERM_MARK));
            // The answer after the mark's shows its decider said its word.
            let past = bed.ask(frame(2, 1)).await;
            assert_eq!(lease_of(&past).id, term << 32 | u64::from(TERM_MARK + 1));
            assert!(words.try_recv().is_ok(), "the word under term {term}");
            assert!(words.try_recv().is_err(), "one word under term {term}");
        }
    }

    /// Under the last term a replica can take, the mark says no word: a
    /// hand-over there leaves no term to take. The term before it says its
    /// own.
    #[tokio::test]
    async fn the_mark_of_the_last_term_says_no_word() {
        let mut bed = leading([]);
        let (words, go) = heard(&bed);
        go.send(()).expect("the decider holds the other end");
        // The bed's candidate, alone over its lease, leaves and takes again
        // at one instant, twice: above a term said, the term before the
        // last; then at a planned hand-over, the last under the ceiling.
        let (last, taken) = (TERM_CEILING - 1, bed.script.taken);
        for (term, above, says) in [(last - 1, Some(last - 2), true), (last, None, false)] {
            let left = bed.elector.step_down(above, taken);
            assert!(left.expect("a term that leaves one to take"));
            let took = bed.elector.tick(taken).expect("the take");
            assert_eq!(took, Tick::Acquired { term });
            bed.store.set_term(term);
            bed.store.spend_ids(u32::MAX - TERM_MARK + 1);
            let at = bed.ask(frame(1, 1)).await;
            assert_eq!(lease_of(&at).id, term << 32 | u64::from(TERM_MARK));
            // The answer after the mark's shows its decider said what it
            // had to say.
            let past = bed.ask(frame(2, 1)).await;
            assert_eq!(lease_of(&past).id, term << 32 | u64::from(TERM_MARK + 1));
            assert_eq!(words.try_recv().is_ok(), says, "under term {term}");
        }
    }

    /// What the engine told a stage of leases: an issue by its id, a release
    /// by its id and reason.
    pub(crate) type Calls = Vec<(u64, Option<ReleaseReason>)>;

    /// The `random` picker, keeping what the engine tells it of leases.
    #[derive(Debug)]
    struct Told(Arc<Mutex<Calls>>);

    impl Lifecycle for Told {
        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            self.0.lock().push((lease.id.0, None));
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            self.0.lock().push((lease.id.0, Some(reason)));
        }
    }

    impl Picker for Told {
        fn name(&self) -> &'static str {
            "told"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// The `random` preset as `generation`, picking through [`Told`]: what
    /// that generation's engine tells its stage of leases is kept in
    /// `told`.
    pub(crate) fn hearing(told: &Arc<Mutex<Calls>>, generation: u64) -> AppliedSpec {
        let kept = Arc::clone(told);
        let mut catalog = Catalog::builtin();
        catalog
            .picker("told", move || Box::new(Told(Arc::clone(&kept))))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = generation;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "told".into();
        let digest = document.digest().expect("the document digests");
        AppliedSpec::validate_digested(&document, digest, &catalog)
            .expect("the catalog holds the stage")
    }

    /// [`Told`], whose issue call waits for its case: what falls while a
    /// stage's own call runs is the case's to say.
    #[derive(Debug)]
    struct Waits {
        told: Told,
        reached: std_mpsc::SyncSender<()>,
        go: Arc<Mutex<std_mpsc::Receiver<()>>>,
        /// Its release call panics.
        bursts: bool,
    }

    impl Lifecycle for Waits {
        fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
            self.told.on_decision_issued(lease, decision);
            self.reached.send(()).expect("the case waits for this call");
            let word = self.go.lock().recv_timeout(PATIENCE);
            word.expect("the case lets the call go on");
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            assert!(!self.bursts, "the case's stage panics at a release");
            self.told.on_released(lease, reason);
        }
    }

    impl Picker for Waits {
        fn name(&self) -> &'static str {
            "waits"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A leader whose one picker waits inside its issue call: the bed, what
    /// the engine told the stage of leases, and the case's two ends of the
    /// wait, the call reached and the word that lets it go on.
    fn waiting() -> (
        Bed,
        Arc<Mutex<Calls>>,
        std_mpsc::Receiver<()>,
        std_mpsc::SyncSender<()>,
    ) {
        waits(false)
    }

    /// [`waiting`]; where it `bursts`, the picker's release call panics.
    fn waits(
        bursts: bool,
    ) -> (
        Bed,
        Arc<Mutex<Calls>>,
        std_mpsc::Receiver<()>,
        std_mpsc::SyncSender<()>,
    ) {
        let told = Arc::new(Mutex::new(Vec::new()));
        let (reached, at_issue) = std_mpsc::sync_channel(1);
        let (word, go) = std_mpsc::sync_channel(1);
        let (kept, go) = (Arc::clone(&told), Arc::new(Mutex::new(go)));
        let mut catalog = Catalog::builtin();
        catalog
            .picker("waits", move || {
                Box::new(Waits {
                    told: Told(Arc::clone(&kept)),
                    reached: reached.clone(),
                    go: Arc::clone(&go),
                    bursts,
                })
            })
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "waits".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog)
            .expect("the catalog holds the stage");
        (bed(Some(spec), true, (2, 8), []), told, at_issue, word)
    }

    /// The budget is judged behind the lease's issue: what a stage's issue
    /// call takes is the decision's time too. A decision that call outlasts
    /// leaves as `Busy`, its lease is off the file, and the engine that
    /// heard the issue hears the release.
    #[tokio::test]
    async fn a_decision_its_stages_issue_call_outlasts_leaves_as_busy_and_gives_its_lease_up() {
        let (mut bed, told, at_issue, word) = waiting();
        assert_eq!(bed.submit(frame(1, 1)), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        // The call runs under the turn of the leases' file, on the lease
        // filed ahead of it: no door of the leases answers until it ends.
        // The call took the budget: the next read of the clock, the
        // decision's last, shows the hour past it.
        bed.script.steps.lock().push_back(Step::At(BUDGET));
        word.send(()).expect("the stage waits for the case");
        let answer = bed.answer().await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
        assert_eq!(bed.store.live(GATEWAY), 0);
        let (id, late) = (1 << 32 | 1, Some(ReleaseReason::LateDecision));
        assert_eq!(*told.lock(), [(id, None), (id, late)]);
        // Two reads of the clock, as for a decision that leaves.
        assert_eq!(bed.script.readers.lock().len(), 2);
    }

    /// The fence is judged behind the lease's issue too: a word that closes
    /// it while a stage's issue call runs stops the answer, and the lease
    /// filed for it is given up as abandoned.
    #[tokio::test]
    async fn a_fence_closed_by_a_word_inside_a_stages_issue_call_stops_the_answer_and_its_lease() {
        let (mut bed, told, at_issue, word) = waiting();
        assert_eq!(bed.submit(frame(1, 1)), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        // The candidate gives the lead up while the call runs. A successor
        // takes at once on that word: no answer made under the term leaves
        // after it.
        assert!(bed.elector.release().expect("the release"));
        word.send(()).expect("the stage waits for the case");
        let answer = bed.answer().await;
        assert_eq!(busy(&answer), Reason::NotLeader);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 0);
        assert_eq!(bed.store.live(GATEWAY), 0);
        let (id, abandoned) = (1 << 32 | 1, Some(ReleaseReason::Abandoned));
        assert_eq!(*told.lock(), [(id, None), (id, abandoned)]);
    }

    /// A lease another hand ended while its decision ran is not the
    /// decision's to tell of. A release its gateway hands while the stage's
    /// issue call runs waits for the turn that call runs under: the
    /// give-up's door applies it first and finds none on file, and the
    /// engine hears the issue and the gateway's release alone.
    #[tokio::test]
    async fn a_lease_another_hand_ended_is_told_nothing_at_the_give_up() {
        let (mut bed, told, at_issue, word) = waiting();
        assert_eq!(bed.submit(frame(1, 1)), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        // Its gateway gives the lease up while the stage's call runs.
        let id = LeaseId(1 << 32 | 1);
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = ReleaseReason::ClientGone;
        bed.store.hand(&hand, id, Said::Released(None, gone));
        bed.script.steps.lock().push_back(Step::At(BUDGET));
        word.send(()).expect("the stage waits for the case");
        let answer = bed.answer().await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(*told.lock(), [(id.0, None), (id.0, Some(gone))]);
        assert_eq!(bed.store.live(GATEWAY), 0);
    }

    /// A decider that leaves its work on an empty queue applies what was
    /// handed behind its last take, and the engine's lifecycle calls run
    /// with it, outside any decision. A stage's call that panics there is
    /// a fault as a decision's is: counted, and the decider stands for its
    /// next request.
    #[tokio::test]
    async fn a_stages_call_that_panics_as_a_decider_leaves_its_work_is_a_fault_it_outlives() {
        let (mut bed, told, at_issue, word) = waits(true);
        let ended = bed.deciders.ended();
        assert_eq!(bed.submit(frame(1, 1)), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        // Its gateway gives the lease up while the stage's call runs: the
        // frame is left to the decider at work, which applies it as it
        // leaves that work.
        let id = LeaseId(1 << 32 | 1);
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        bed.store.hand(&hand, id, gone);
        word.send(()).expect("the stage waits for the case");
        picked(&bed.answer().await);
        let (faults, waited) = (&bed.counts.faults, Instant::now());
        while faults.load(Ordering::Relaxed) == 0 && !*ended.borrow() {
            assert!(
                waited.elapsed() < PATIENCE,
                "the decider never left its work"
            );
            thread::sleep(Duration::from_millis(1));
        }
        assert!(!*ended.borrow(), "the decider's thread unwound");
        assert_eq!(faults.load(Ordering::Relaxed), 1);
        // The lease left the file ahead of the call that panicked.
        assert_eq!(bed.store.live(GATEWAY), 0);
        assert_eq!(*told.lock(), [(id.0, None)]);
        // The decider stands: its next request is decided.
        assert_eq!(bed.submit(frame(2, 1)), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        word.send(()).expect("the stage waits for the case");
        picked(&bed.answer().await);
    }

    /// An answer its stream's queue does not take carries a lease no
    /// gateway will name a dispatch for. Its decider hands the release
    /// where the queue refuses the answer, full or closed: the lease is
    /// off the file and its stage told, abandoned, before the next
    /// decision reads a count. An answer the queue took keeps its lease.
    #[tokio::test]
    async fn a_lease_whose_answer_no_stream_took_is_given_up_where_it_is_dropped() {
        let (mut bed, told, at_issue, word) = waiting();
        let passed = || {
            at_issue
                .recv_timeout(PATIENCE)
                .expect("the stage's issue call");
            word.send(()).expect("the stage waits for the case");
        };
        let job = |bed: &Bed, id: u8, answers: &mpsc::Sender<wire::Decided>| Job {
            request: checked(frame(id, 1)),
            gateway: GATEWAY,
            received: bed.script.taken,
            answers: answers.clone(),
        };
        // A stream whose queue holds one answer and that reads none: the
        // first answer waits in it, and the second finds it full.
        let (sender, answers) = mpsc::channel(1);
        for id in [1, 2] {
            assert!(bed.deciders.submit(job(&bed, id, &sender)).is_none());
            passed();
        }
        // One decider decides a model's requests in order: it is past the
        // two when it answers a third, on a queue that is read.
        assert_eq!(bed.submit(frame(3, 1)), None);
        passed();
        picked(&bed.answer().await);
        let id = |count: u64| 1 << 32 | count;
        let abandoned = Some(ReleaseReason::Abandoned);
        let mut heard = vec![
            (id(1), None),
            (id(2), None),
            (id(2), abandoned),
            (id(3), None),
        ];
        assert_eq!(*told.lock(), heard);
        assert_eq!(bed.store.live(GATEWAY), 2);
        // The stream ends, and the answer in its queue with it: a fourth
        // answer finds the queue closed.
        drop(answers);
        assert!(bed.deciders.submit(job(&bed, 4, &sender)).is_none());
        passed();
        assert_eq!(bed.submit(frame(5, 1)), None);
        passed();
        picked(&bed.answer().await);
        heard.extend([(id(4), None), (id(4), abandoned), (id(5), None)]);
        assert_eq!(*told.lock(), heard);
        assert_eq!(bed.store.live(GATEWAY), 3);
        let dropped = [&bed.counts.dropped_full, &bed.counts.dropped_gone];
        assert_eq!(dropped.map(|count| count.load(Ordering::Relaxed)), [1, 1]);
    }

    /// The hand of an unsent answer applies in place where the ring is
    /// full, outside any decision. A stage's call that panics there is a
    /// fault as a decision's is: counted, and the decider stands for its
    /// next request.
    #[tokio::test]
    async fn a_stages_panic_at_an_unsent_answers_give_up_is_a_fault_its_decider_outlives() {
        let (mut bed, told, at_issue, word) = waits(true);
        let ended = bed.deciders.ended();
        // A stream that ended: its answer finds the queue closed.
        let (sender, answers) = mpsc::channel(1);
        drop(answers);
        let job = Job {
            request: checked(frame(1, 1)),
            gateway: GATEWAY,
            received: bed.script.taken,
            answers: sender,
        };
        assert!(bed.deciders.submit(job).is_none());
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        // While the stage's call runs, its gateway gives the lease up and
        // fills the ring behind that frame: the decider's own hand finds
        // the ring full and applies in place, where the stage's release
        // call panics.
        let id = LeaseId(1 << 32 | 1);
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = || Said::Released(None, ReleaseReason::ClientGone);
        bed.store.hand(&hand, id, gone());
        for _ in 1..MemoryStore::HANDOVER {
            bed.store.hand(&hand, LeaseId(0), gone());
        }
        word.send(()).expect("the stage waits for the case");
        let (faults, waited) = (&bed.counts.faults, Instant::now());
        while faults.load(Ordering::Relaxed) == 0 && !*ended.borrow() {
            assert!(
                waited.elapsed() < PATIENCE,
                "the decider never gave the lease up"
            );
            thread::sleep(Duration::from_millis(1));
        }
        assert!(!*ended.borrow(), "the decider's thread unwound");
        assert_eq!(faults.load(Ordering::Relaxed), 1);
        // The lease left the file ahead of the call that panicked.
        assert_eq!(bed.store.live(GATEWAY), 0);
        assert_eq!(*told.lock(), [(id.0, None)]);
        // The decider stands: its next request is decided.
        assert_eq!(bed.submit(frame(2, 1)), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        word.send(()).expect("the stage waits for the case");
        picked(&bed.answer().await);
    }

    /// A lease's dispatch span counts from its decision's first read, and
    /// its answer says what is left of the span at the last. A decision
    /// inside its budget that leaves under a millisecond of the span has no
    /// lease to hand over: it leaves as `Busy`, late, its lease given up.
    /// One with no lease has no span to end, and leaves.
    #[tokio::test]
    async fn a_slow_decisions_lease_says_what_is_left_of_its_span_and_none_left_is_no_answer() {
        // A budget of twice the span: the span ends first.
        let slow = |id: u8| budgeted(frame(id, 1), DISPATCH_TTL * 2);
        let (mut bed, told, at_issue, word) = waiting();
        let id = |count: u64| 1 << 32 | count;
        // The first decision's stage leaves a millisecond of the span.
        let first = DISPATCH_TTL - Duration::from_millis(1);
        assert_eq!(bed.submit(slow(1)), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        bed.script.steps.lock().push_back(Step::At(first));
        word.send(()).expect("the stage waits for the case");
        let answer = bed.answer().await;
        assert_eq!(lease_of(&answer).dispatch_ttl_ms, 1);
        let store: &dyn LeasesStore = bed.store.as_ref();
        let filed = store.get(GATEWAY, LeaseId(id(1)));
        let filed = filed.expect("the lease is on file");
        assert_eq!(filed.dispatch_deadline, bed.script.taken + DISPATCH_TTL);
        // The second is taken where the first left the clock, and its
        // stage leaves under a millisecond of the span, half its budget.
        let second = first + DISPATCH_TTL - Duration::from_micros(999);
        assert_eq!(bed.submit_at(slow(2), GATEWAY, first), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        bed.script.steps.lock().push_back(Step::At(second));
        word.send(()).expect("the stage waits for the case");
        let answer = bed.answer().await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
        // A third of the same length, which the wires shed: no lease, so
        // no span to end, and it leaves as its shed.
        let third = second + DISPATCH_TTL - Duration::from_micros(999);
        let reads = [second, third].map(Step::At);
        bed.script.steps.lock().extend(reads);
        let wireless = constrained(slow(3), 0b010, None);
        assert_eq!(bed.submit_at(wireless, GATEWAY, second), None);
        let shed = declined(proto::ShedReason::NoEndpoints, "wires");
        assert_eq!(outcome(&bed.answer().await), &shed);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
        let late = (id(2), Some(ReleaseReason::LateDecision));
        assert_eq!(*told.lock(), [(id(1), None), (id(2), None), late]);
        assert_eq!(bed.store.live(GATEWAY), 1, "the first decision's lease");
    }

    #[tokio::test]
    async fn the_engine_that_decided_is_told_of_a_lease_and_of_its_end_at_the_dispatch_deadline() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
        let id = |count: u64| 1 << 32 | count;
        // Two leases at the take, the second dispatched.
        for request in 1..=2 {
            bed.ask(frame(request, 1)).await;
        }
        let store: &dyn LeasesStore = bed.store.as_ref();
        assert!(store.dispatched(
            GATEWAY,
            LeaseId(id(2)),
            Dispatch::default(),
            bed.script.taken
        ));
        assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
        // A decision past the dispatch deadline ends the first as
        // abandoned, and its engine is told; the dispatched lease stands.
        let late = DISPATCH_TTL + Duration::from_secs(1);
        bed.script.steps.lock().push_back(Step::At(late));
        assert_eq!(bed.submit_at(frame(3, 1), GATEWAY, late), None);
        bed.answer().await;
        let abandoned = (id(1), Some(ReleaseReason::Abandoned));
        assert_eq!(told.lock()[2..], [(id(3), None), abandoned]);
        let store: &dyn LeasesStore = bed.store.as_ref();
        assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
        assert_eq!(store.live(GATEWAY), 2);
    }

    /// A decider with a job at hand is at work at the leases' file: what a
    /// session hands meanwhile is left to it, and it applies it before its
    /// decision reads the rows' counts, ahead of its own issue. A frame
    /// handed while no decider has a job is applied all the same.
    #[tokio::test]
    async fn a_frame_handed_while_a_decider_decides_is_applied_by_that_decider() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&told);
        let mut catalog = Catalog::builtin();
        catalog
            .picker("told", move || Box::new(Told(Arc::clone(&kept))))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "told".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog)
            .expect("the catalog holds the stage");
        let mut bed = bed(Some(spec), true, (1, 8), []);
        let id = |count: u64| 1 << 32 | count;
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        bed.ask(frame(1, 1)).await;
        // The second decision is held at its first read of the clock: its
        // decider has the job at hand.
        let (reached, held) = std_mpsc::sync_channel(1);
        let (go, word) = std_mpsc::sync_channel(1);
        let hold = Step::Hold { reached, go: word };
        bed.script.steps.lock().push_back(hold);
        assert_eq!(bed.submit(frame(2, 1)), None);
        held.recv_timeout(PATIENCE)
            .expect("the decider reads the clock");
        let hand = lifecycle::open(&bed.store, GATEWAY);
        bed.store.hand(&hand, LeaseId(id(1)), gone);
        assert_eq!(*told.lock(), [(id(1), None)], "left to the decider at work");
        go.send(()).expect("the decider holds the other end");
        bed.answer().await;
        let released = |count| (id(count), Some(ReleaseReason::ClientGone));
        assert_eq!(*told.lock(), [(id(1), None), released(1), (id(2), None)]);
        // The decider waits on its empty queue, or is about to: it applies
        // as it leaves its work, or the hand does.
        bed.store.hand(&hand, LeaseId(id(2)), gone);
        let since = Instant::now();
        while told.lock().len() < 4 {
            assert!(
                since.elapsed() < PATIENCE,
                "a handed frame waits on no lull"
            );
            tokio::task::yield_now().await;
        }
        assert_eq!(told.lock()[3], released(2));
    }

    /// The `random` picker, keeping the decider its thread acts as at each
    /// call of a lease's issue and release.
    #[derive(Debug)]
    struct Named(Arc<Mutex<Vec<Option<Decider>>>>);

    impl Lifecycle for Named {
        fn on_decision_issued(&self, _: &Lease, _: &Decision) {
            self.0.lock().push(Decider::acting());
        }

        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            self.0.lock().push(Decider::acting());
        }
    }

    impl Picker for Named {
        fn name(&self) -> &'static str {
            "named"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A decider's thread acts as its decider while it decides, and a
    /// lease's event is told under the name of the decider that issued it,
    /// whichever thread applies the frame.
    #[tokio::test]
    async fn a_decision_and_its_leases_events_act_as_the_leases_decider() {
        let named = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&named);
        let mut catalog = Catalog::builtin();
        catalog
            .picker("named", move || Box::new(Named(Arc::clone(&kept))))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "named".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog)
            .expect("the catalog holds the stage");
        let mut bed = bed(Some(spec), true, (1, 8), []);
        let answer = bed.ask(frame(1, 1)).await;
        // The case's thread names no decider: its hand applies the release,
        // or the decider does as it leaves its work.
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        bed.store.hand(&hand, LeaseId(lease_of(&answer).id), gone);
        let since = Instant::now();
        while named.lock().len() < 2 {
            assert!(
                since.elapsed() < PATIENCE,
                "a handed frame waits on no lull"
            );
            tokio::task::yield_now().await;
        }
        assert_eq!(*named.lock(), [Some(Decider::new(0)); 2]);
        assert_eq!(Decider::acting(), None);
    }

    /// An abandoned lease is told to the engine whose generation issued it
    /// and to no later one: a later generation's stages never saw the
    /// issue. The lease is ended all the same.
    #[tokio::test]
    async fn a_later_generations_engine_is_not_told_of_a_lease_it_did_not_issue() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let spec = |generation: u64| hearing(&told, generation);
        let mut bed = bed(Some(spec(1)), true, (2, 8), []);
        let id = |count: u64| 1 << 32 | count;
        bed.ask(frame(1, 1)).await;
        // The next generation stands when the first lease's deadline is
        // past: its engine is told of its own issue and of nothing else.
        bed.store.apply(spec(2)).expect("a later generation");
        let late = DISPATCH_TTL + Duration::from_secs(1);
        bed.script.steps.lock().push_back(Step::At(late));
        assert_eq!(bed.submit_at(frame(2, 1), GATEWAY, late), None);
        bed.answer().await;
        assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
        let store: &dyn LeasesStore = bed.store.as_ref();
        assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
        assert_eq!(store.live(GATEWAY), 1);
    }

    /// What one decision was handed of each row's requests in flight: the
    /// count the request's facts read, and the row's own field.
    type SeenRows = Vec<(u32, u32)>;

    /// The `random` picker, keeping the reads of every decision.
    #[derive(Debug)]
    struct Seen {
        rows: Arc<Mutex<Vec<SeenRows>>>,
    }

    impl Lifecycle for Seen {}

    impl Picker for Seen {
        fn name(&self) -> &'static str {
            "seen"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            let counts = candidates
                .iter()
                .map(|row| (facts.in_flight(row), row.in_flight));
            self.rows.lock().push(counts.collect());
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A decision reads what the leases on file book, whatever its profile
    /// and with nothing declared: its facts read the three leases the
    /// decisions before it left, beside lent rows, no row of the
    /// publication carrying a count.
    #[tokio::test]
    async fn a_decision_reads_the_leases_count_beside_the_rows_its_store_lends() {
        let rows = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&rows);
        let mut catalog = Catalog::builtin();
        let seen = move || -> Box<dyn Picker> {
            let rows = Arc::clone(&kept);
            Box::new(Seen { rows })
        };
        catalog.picker("seen", seen).expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "seen".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog)
            .expect("the catalog holds the stage");
        let mut bed = bed(Some(spec), true, (2, 8), []);
        // Three decisions and no word from their gateway: three leases
        // stand on file when the fourth is made.
        for request in 1..=4u8 {
            bed.ask(frame(request, u64::from(request))).await;
        }
        // What the fourth decision was handed: its facts read the three
        // leases, beside its rows.
        let rows = rows.lock();
        let (read, own): (Vec<u32>, Vec<u32>) = rows[3].iter().copied().unzip();
        assert_eq!(read.iter().sum::<u32>(), 3, "{rows:?}");
        assert_eq!(own.iter().sum::<u32>(), 0, "{rows:?}");
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
            assert_eq!(answer.frame().request_id, [id]);
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

    /// Each `Busy` a decider answers is counted at its reason, and a
    /// decision moves no count.
    #[tokio::test]
    async fn each_busy_is_counted_at_its_reason_and_a_decision_moves_no_count() {
        // A decision; a request taken with its budget gone; one whose
        // decision panics; two taken past the renew deadline.
        let steps = [
            Step::At(Duration::ZERO),
            Step::At(Duration::ZERO),
            Step::At(BUDGET),
            Step::Panic,
            Step::At(FENCED),
        ];
        let mut bed = bed(Some(preset("random", 1)), true, (1, 8), steps);
        let counted = |counts: &Counts| {
            counts
                .busy
                .each_ref()
                .map(|count| count.load(Ordering::Relaxed))
        };
        let answer = bed.ask(frame(1, 1)).await;
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        assert_eq!(counted(&bed.counts), [0; 5]);
        assert_eq!(
            busy(&bed.ask(frame(2, 2)).await),
            Reason::BudgetGoneOnArrival
        );
        assert_eq!(busy(&bed.ask(frame(3, 3)).await), Reason::Unspecified);
        for id in [4, 5] {
            assert_eq!(busy(&bed.ask(frame(id, 1)).await), Reason::NotLeader);
        }
        // In the order of the reasons' numbers on the wire.
        assert_eq!(Counts::BUSY[3..], ["budget_gone_on_arrival", "not_leader"]);
        assert_eq!(counted(&bed.counts), [1, 0, 0, 1, 2]);
        // The counts are sized by hand beside the wire's enum: every reason
        // the contract names has a place, and the number past them is no
        // reason. A reason added to the contract fails here, not silently.
        let places = i32::try_from(Counts::BUSY.len()).expect("a handful of reasons");
        assert!((0..places).all(|number| Reason::try_from(number).is_ok()));
        assert!(Reason::try_from(places).is_err());
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
        // The election's report tells the store the term of each take.
        bed.store.set_term(2);
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
    pub(crate) fn gone(reads: &Reads) -> Deciders {
        let (queue, jobs) = std_mpsc::sync_channel(1);
        drop(jobs);
        Deciders {
            queues: Box::new([queue]),
            told: Arc::new(watch::Sender::new(false)),
            reads: reads.clone(),
            mark: Arc::default(),
        }
    }

    /// A decider's thread that ended takes no request: each one for its
    /// models comes back whole, as gone and never as a full queue.
    #[test]
    fn a_request_for_a_decider_whose_thread_ended_comes_back_as_gone() {
        let deciders = gone(&leading([]).reads);
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

    /// A request that states nothing of its gateway reads the rows where
    /// their publication left them, and the decider's buffers take no row.
    /// One that states wires, or a hint of 0, is handed rows written into
    /// those buffers, which its narrowing cuts. No decider's facts pin a
    /// row.
    #[test]
    fn a_request_that_states_nothing_of_its_gateway_reads_the_published_rows() {
        const SPOKEN: &str = "model-http";
        let store = MemoryStore::new();
        ready(&store, "h0", SPOKEN, Role::Single, WireModes::HTTP);
        ready(&store, "g0", SPOKEN, Role::Single, WireModes::GRPC);
        // The rows the engine is handed, and the rows written into the
        // decider's buffers for them.
        let read = |frame: proto::DecideRequest| {
            let request = checked(frame);
            let now = Instant::now();
            let taken = Taken::at(&request, GATEWAY, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            let handed = store.deciding(|world| {
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &AtomicU64::new(0),
                    |facts, _, rows, _, _| {
                        assert_eq!(facts.pinned_at, 0);
                        rows.len()
                    },
                )
                .0
            });
            (handed, buffers.view.candidates.len())
        };
        let plain = frame_for(SPOKEN, 1, 1);
        assert_eq!(read(plain), (2, 0), "no constraints: lent");
        let hinted = constrained(frame_for(SPOKEN, 2, 2), 0, Some(7));
        assert_eq!(read(hinted), (2, 0), "a positive hint: lent");
        let wired = constrained(frame_for(SPOKEN, 3, 3), 0b001, None);
        assert_eq!(read(wired), (1, 1), "wires: written and cut");
        let both = constrained(frame_for(SPOKEN, 4, 4), 0b011, None);
        assert_eq!(read(both), (2, 2), "wires that cut none: written");
        let none = constrained(frame_for(SPOKEN, 5, 5), 0, Some(0));
        assert_eq!(read(none), (0, 0), "a hint of 0");
    }

    /// A profile that shows its rows to a plugin is handed rows the door
    /// wrote for it, under a veto of its gateway as under none: the vetoed
    /// row carries its own bit, no mask stands beside the rows, and the
    /// engine decides where it declines lent rows.
    #[test]
    fn a_profile_that_shows_its_rows_is_handed_written_rows_and_decides_under_a_veto() {
        #[derive(Debug)]
        struct Shown;

        impl Lifecycle for Shown {}

        impl DecisionScope for Shown {
            fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

            fn close(&self) {}
        }

        impl Filter for Shown {
            fn name(&self) -> &'static str {
                "shown"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn scope(&self) -> Option<&dyn DecisionScope> {
                Some(self)
            }
        }

        let mut catalog = Catalog::builtin();
        catalog
            .filter("shown", || Box::new(Shown))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = 1;
        for profile in document.profiles.values_mut() {
            profile.filter.push("shown".to_string());
        }
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog)
            .expect("the catalog holds every stage the document names");
        let store = MemoryStore::new();
        let (mut singles, _) = fleet(&store);
        singles.sort_unstable();
        store.apply(spec).expect("above the unbound start");
        let taken = Instant::now();
        let now = taken + Duration::from_secs(3);
        // The rows whose own bit says stalled, the mask's length, whether
        // the engine picked, and the rows written into the decider's
        // buffers.
        let decided = || {
            let request = checked(frame(1, 1));
            let taken = Taken::at(&request, GATEWAY, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            let (stalled, masked, picked) = store.deciding(|world| {
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &AtomicU64::new(0),
                    |facts, pool, rows, cx, scratch| {
                        let stalled: Vec<WorkerId> = rows
                            .iter()
                            .filter(|row| row.health.stalled)
                            .map(|row| row.id)
                            .collect();
                        let engine = world.spec().engine();
                        let decision = engine.decide(facts, pool, rows, cx, scratch);
                        (stalled, cx.stalled.len(), decision.pick().is_some())
                    },
                )
                .0
            });
            (stalled, masked, picked, buffers.view.candidates.len())
        };
        assert_eq!(decided(), (Vec::new(), 0, true, 3), "no veto");
        let vetoed = singles[1];
        store.gateway_veto(GATEWAY, vetoed, Some(taken + Duration::from_secs(60)));
        assert_eq!(decided(), (vec![vetoed], 0, true, 3), "a veto");
    }

    /// A key under a sessions rule with an in-flight cap leaves an owner
    /// over its thresholds in a decision this service makes. The store
    /// writes no count of the key's requests into a decision's rows, so the
    /// cap could not hand the key on: the veto stands for the key, and it
    /// goes to the ring's next worker.
    #[test]
    fn a_key_under_a_capped_rule_leaves_an_owner_over_its_thresholds() {
        let mut document =
            PolicySpec::preset("consistent_hashing").expect("a preset of this build");
        let manual = PolicySpec::preset("manual").expect("a preset of this build");
        let mut rule = manual
            .bindings
            .first()
            .and_then(|binding| binding.sessions.clone())
            .expect("the manual preset's rule");
        rule.assignment = crate::spec::Assignment::Delegate;
        rule.inflight_cap = 2;
        document.generation = 1;
        document.view.headers = manual.view.headers.clone();
        let binding = document.bindings.first_mut().expect("the preset's binding");
        binding.sessions = Some(rule);
        binding.overload = Some(crate::spec::OverloadSpec {
            waiting_requests: Some(1),
            ..crate::spec::OverloadSpec::default()
        });
        let store = MemoryStore::with_publish_window(Duration::ZERO);
        let (singles, _) = fleet(&store);
        store
            .apply(applied(&document))
            .expect("above the unbound start");
        let now = Instant::now();
        // The worker a request of the key is sent to, and the workers of
        // its pool that read overloaded.
        let decided = |id: u8| {
            let mut keyed = frame(id, u64::from(id));
            keyed
                .facts
                .as_mut()
                .expect("the case's frame has facts")
                .session_key = Some(proto::Identity {
                form: Some(proto::identity::Form::Hash(17)),
            });
            let request = checked(keyed);
            let taken = Taken::at(&request, GATEWAY, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            store.deciding(|world| {
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &AtomicU64::new(0),
                    |facts, pool, rows, cx, scratch| {
                        let over: Vec<WorkerId> = rows
                            .iter()
                            .filter(|row| row.health.overloaded)
                            .map(|row| row.id)
                            .collect();
                        let engine = world.spec().engine();
                        let decision = engine.decide(facts, pool, rows, cx, scratch);
                        let picked = decision
                            .pick()
                            .and_then(|pick| pick.targets.first())
                            .map(|target| target.id);
                        (picked, over)
                    },
                )
                .0
            })
        };
        let (owner, over) = decided(1);
        let owner = owner.expect("a pick over a free pool");
        assert!(singles.contains(&owner), "{owner:?} of {singles:?}");
        assert_eq!(over, Vec::new(), "no worker is over its thresholds");
        crate::store::LoadSink::record(
            &store,
            owner,
            0,
            crate::LoadRow {
                queued: 1,
                ..crate::LoadRow::default()
            },
            now,
            crate::store::LoadSource::Stream,
        );
        let (next, over) = decided(2);
        assert_eq!(over, vec![owner], "the key's owner reads overloaded, alone");
        let next = next.expect("a pick beside two workers under their thresholds");
        assert_ne!(
            next, owner,
            "the cap's count is not written, so the veto stands for the key"
        );
    }

    /// One lease on file reads as its worker's requests in flight on both
    /// forms of a decision's rows: in the row where the rows were written
    /// for a request that states its wires, beside the row where they were
    /// lent. The request's facts read it either way, under a profile that
    /// reads no count too.
    #[test]
    fn a_lease_on_file_is_read_in_a_written_row_and_beside_a_lent_one() {
        const SPOKEN: &str = "model-http";
        for (stated, own) in [(true, 1), (false, 0)] {
            let store = MemoryStore::new();
            store
                .apply(preset("random", 1))
                .expect("above the unbound start");
            let member = ready(&store, "h0", SPOKEN, Role::Single, WireModes::HTTP);
            let leases: &dyn LeasesStore = &store;
            leases.set_term(1);
            let now = Instant::now();
            let filed = leases.issue(crate::store::LeaseRecord {
                lease: Lease {
                    id: LeaseId(1 << 63 | 1),
                    ..Lease::default()
                },
                gateway: GATEWAY,
                request: 1,
                target: member,
                picked: crate::store::Picked::default(),
                decode: None,
                term: 1,
                issued_at: now,
                dispatch_deadline: now + DISPATCH_TTL,
                state: LeaseState::Issued,
            });
            assert_eq!(filed, Ok(()));
            let frame = frame_for(SPOKEN, 1, 1);
            let frame = if stated {
                constrained(frame, 0b001, None)
            } else {
                frame
            };
            let request = checked(frame);
            let taken = Taken::at(&request, GATEWAY, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            let counts = store.deciding(|world| {
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &AtomicU64::new(0),
                    |facts, _, rows, _, _| {
                        let read = |row: &Candidate| (facts.in_flight(row), row.in_flight);
                        rows.iter().map(read).collect::<Vec<_>>()
                    },
                )
                .0
            });
            assert_eq!(counts, [(1, own)], "wires stated: {stated}");
        }
    }

    /// The model of the cases that book a fleet's leases themselves.
    const BOOKED: &str = "model-http";

    /// Files one issued lease of `GATEWAY` on `target`, under `id`.
    fn file(store: &MemoryStore, id: u64, target: WorkerId, now: Instant) {
        let leases: &dyn LeasesStore = store;
        let filed = leases.issue(crate::store::LeaseRecord {
            lease: Lease {
                id: LeaseId(1 << 63 | id),
                ..Lease::default()
            },
            gateway: GATEWAY,
            request: id,
            target,
            picked: crate::store::Picked::default(),
            decode: None,
            term: 1,
            issued_at: now,
            dispatch_deadline: now + DISPATCH_TTL,
            state: LeaseState::Issued,
        });
        assert_eq!(filed, Ok(()));
    }

    /// A store under `spec` with one member of `BOOKED` a count of `held`,
    /// each booked by its count of leases on file: the store, the members
    /// in the counts' order and the instant the leases were issued at.
    fn booked(spec: AppliedSpec, held: &[u64]) -> (MemoryStore, Vec<WorkerId>, Instant) {
        let store = MemoryStore::new();
        store.apply(spec).expect("above the unbound start");
        let members: Vec<WorkerId> = (0..held.len())
            .map(|i| {
                ready(
                    &store,
                    &format!("h{i}"),
                    BOOKED,
                    Role::Single,
                    WireModes::HTTP,
                )
            })
            .collect();
        let leases: &dyn LeasesStore = &store;
        leases.set_term(1);
        let now = Instant::now();
        let targets = members.iter().zip(held);
        let targets = targets.flat_map(|(&member, &held)| (0..held).map(move |_| member));
        for (target, id) in targets.zip(1..) {
            file(&store, id, target, now);
        }
        (store, members, now)
    }

    /// One decision of `frame` over `store` by the decider's own hand: each
    /// row's requests in flight as the request's facts first read them, by
    /// worker, and the first target. `late` is filed on its workers after
    /// those reads and before the engine is asked, and the facts then read
    /// every row as they first read it.
    fn decided(
        store: &MemoryStore,
        frame: proto::DecideRequest,
        now: Instant,
        late: &[WorkerId],
    ) -> (Vec<(WorkerId, u32)>, Option<WorkerId>) {
        let request = checked(frame);
        let taken = Taken::at(&request, GATEWAY, now, now).expect("budget is left");
        let mut buffers = Buffers::default();
        store.deciding(|world| {
            hand(
                world,
                &taken,
                &mut buffers,
                &AtomicU64::new(0),
                |facts, pool, rows, cx, scratch| {
                    let read = || -> Vec<(WorkerId, u32)> {
                        let rows = rows.iter();
                        rows.map(|row| (row.id, facts.in_flight(row))).collect()
                    };
                    let first = read();
                    for (&target, id) in late.iter().zip(1_000..) {
                        file(store, id, target, now);
                    }
                    assert_eq!(read(), first, "a decision reads one count a row");
                    let decision = world.spec().engine().decide(facts, pool, rows, cx, scratch);
                    let picked = decision.pick();
                    let target = picked.and_then(|pick| pick.targets.as_slice().first());
                    (first, target.map(|target| target.id))
                },
            )
            .0
        })
    }

    /// `held` by the worker of `members` each count is of, in the rows'
    /// order: by worker id.
    fn by_id(members: &[WorkerId], held: &[u32]) -> Vec<(WorkerId, u32)> {
        let mut held: Vec<(WorkerId, u32)> = members.iter().copied().zip(held.to_vec()).collect();
        held.sort_unstable();
        held
    }

    /// One reader, two feeds, by the decision: under each preset that reads
    /// a worker's requests in flight, a request that states its wires (rows
    /// written for it, the count on each) and one that states none (lent
    /// rows, the leases' table beside them) read the same counts and are
    /// answered by the same pick.
    #[test]
    fn a_count_reading_preset_picks_alike_over_written_and_lent_rows() {
        for name in ["least_load", "power_of_two", "prefix_hash", "cache_aware"] {
            // The i-th worker books i leases: the counts part every row.
            let (store, members, now) = booked(preset(name, 1), &[0, 1, 2, 3, 4, 5]);
            let held = by_id(&members, &[0, 1, 2, 3, 4, 5]);
            for seed in 1..=32 {
                let plain = frame_for(BOOKED, 1, seed);
                let stated = constrained(plain.clone(), 0b001, None);
                let lent = decided(&store, plain, now, &[]);
                assert_eq!(lent.0, held, "{name}, seed {seed}");
                assert_eq!(
                    decided(&store, stated, now, &[]),
                    lent,
                    "{name}, seed {seed}"
                );
            }
        }
    }

    /// A decision reads one count a row: a row's first read is kept, and a
    /// lease filed after it moves nothing the decision's stages read. Two
    /// leases are filed on the row that held the fewest, after the facts'
    /// first reads and before the `count-first` picker's narrowing and
    /// pick: the picker still names that row. The next decision reads the
    /// two.
    #[test]
    fn a_lease_filed_after_a_rows_first_read_moves_nothing_the_decision_reads() {
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "count-first".into();
        let (store, members, now) = booked(applied(&document), &[1, 0, 2]);
        let fewest = members[1];
        let first = decided(&store, frame_for(BOOKED, 1, 1), now, &[fewest, fewest]);
        assert_eq!(first, (by_id(&members, &[1, 0, 2]), Some(fewest)));
        let next = decided(&store, frame_for(BOOKED, 2, 2), now, &[]);
        assert_eq!(next, (by_id(&members, &[1, 2, 2]), Some(members[0])));
    }

    /// The engine's own placement by the fewest requests reads the leases
    /// on file as a stage does, with nothing declared: a profile whose
    /// picker leaves the placement to its rule, under no rule and under a
    /// rule that delegates, and a sessions rule that assigns `min_load`.
    /// Each places a request on the worker that holds the fewest leases,
    /// whatever the request's seed.
    #[test]
    fn a_placement_by_the_fewest_requests_reads_the_leases_on_file() {
        let ruled = |assignment: crate::Assignment, pick: &str| {
            let mut document = PolicySpec::preset("manual").expect("a preset of this build");
            let rule = document.bindings[0].sessions.as_mut();
            rule.expect("the preset's rule").assignment = assignment;
            let profile = document.profiles.get_mut("manual");
            profile.expect("the preset's profile").pick = pick.into();
            document
        };
        let unruled = r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"manual"}}}"#;
        let mut unruled: PolicySpec = serde_json::from_str(unruled).expect("a document");
        unruled.generation = 1;
        let (delegate, min_load) = (crate::Assignment::Delegate, crate::Assignment::MinLoad);
        let cases = [
            ("no rule", unruled, false),
            ("a rule that delegates", ruled(delegate, "manual"), false),
            (
                "a rule that delegates, a key",
                ruled(delegate, "manual"),
                true,
            ),
            ("min_load", ruled(min_load, "random"), true),
            (
                "min_load, the placement its rule's",
                ruled(min_load, "manual"),
                true,
            ),
        ];
        for (name, document, keyed) in cases {
            let (store, members, now) = booked(applied(&document), &[2, 0, 3]);
            for seed in 1..=16u8 {
                let mut frame = frame_for(BOOKED, seed, u64::from(seed));
                if keyed {
                    let facts = frame.facts.as_mut().expect("the case's frame has facts");
                    facts.session_key = Some(proto::Identity {
                        form: Some(proto::identity::Form::Hash(u64::from(seed))),
                    });
                }
                let (_, placed) = decided(&store, frame, now, &[]);
                assert_eq!(placed, Some(members[1]), "{name}, seed {seed}");
            }
        }
    }

    /// What the engine is handed for one request, copied out of the
    /// hand-over.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Handed {
        now_micros: u64,
        budget_micros: u32,
        /// The affinity column's length.
        column: usize,
        /// The rows named in the column's place: how many holders and how
        /// many warming rows, where the request carries any list.
        named: Option<(usize, usize)>,
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
                |facts, pool, rows, cx, _| Handed {
                    now_micros: facts.now_micros,
                    budget_micros: facts.budget_micros,
                    column: facts.affinity.len(),
                    named: facts
                        .named
                        .map(|named| (named.holders.len(), named.warming.len())),
                    candidates: rows.iter().map(|row| row.id).collect(),
                    // By the row's own bit where the rows were written, and
                    // by the mask beside them where they were lent.
                    stalled: rows
                        .iter()
                        .zip(0..)
                        .filter(|(row, at)| row.health.stalled || cx.masks(*at))
                        .map(|(row, _)| row.id)
                        .collect(),
                    pool: (pool.pool_size, pool.generation),
                },
            )
            .0
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
            named: None,
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

        // A profile whose picker takes a sample is handed the rows the
        // store names in the column's place, and no column. Here no row
        // holds a block and the three, admitted together, are all young:
        // nobody is named.
        store
            .apply(preset("cache_aware", 2))
            .expect("above the standing one");
        let named = handed(&store, frame(1, 1), OTHER, now);
        assert_eq!(
            (named.column, named.named, named.pool),
            (0, Some((0, 0)), (3, 2))
        );
        // A profile that declares the column and takes no sample is handed
        // one entry a candidate, and no other profile is.
        let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
        document.generation = 3;
        for profile in document.profiles.values_mut() {
            profile.pick = "best".into();
        }
        store
            .apply(applied(&document))
            .expect("above the standing one");
        let column = handed(&store, frame(1, 1), OTHER, now);
        assert_eq!(
            (column.column, column.named, column.pool),
            (3, None, (3, 3))
        );
    }

    /// A member of `role` whose signals ask for events, holding the chain
    /// of `contents`.
    fn holding(store: &MemoryStore, name: &str, role: Role, wires: WireModes, contents: &[u64]) {
        let facts = WorkerFacts {
            role,
            wire_modes: wires,
            ..WorkerFacts::default()
        };
        held_by(store, name, facts, contents);
    }

    /// A member of the role, the wires and the reach of `facts` whose
    /// signals ask for events, holding the chain of `contents`: its id.
    fn held_by(store: &MemoryStore, name: &str, facts: WorkerFacts, contents: &[u64]) -> WorkerId {
        use kv_index::chain_prefix_hash;

        use crate::store::{
            BatchFacts, BatchOutcome, ContentHash, KvEvent, KvEventSink, SequenceHash, Signals,
            StoredBlock,
        };

        let facts = WorkerFacts {
            key: name.to_owned(),
            engine: name.to_owned(),
            incarnation: 1,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            signals: Signals {
                events: true,
                ..Signals::default()
            },
            ..facts
        };
        let id = store.upsert(facts).id();
        let mut blocks: Vec<StoredBlock> = Vec::with_capacity(contents.len());
        for &content in contents {
            let seq_hash = match blocks.last() {
                Some(parent) => chain_prefix_hash(parent.seq_hash, ContentHash(content)),
                None => SequenceHash(content),
            };
            blocks.push(StoredBlock {
                seq_hash,
                content_hash: ContentHash(content),
            });
        }
        let batch = BatchFacts {
            sequence: 1,
            ..BatchFacts::default()
        };
        let events = [KvEvent::Stored {
            parent: None,
            blocks,
        }];
        assert_eq!(store.admit(id, 0, batch, &events), BatchOutcome::Applied);
        id
    }

    /// The hand names for the gateway its request came from: a holder
    /// behind the asking gateway's own socket is named, and one behind
    /// another gateway's is no row of its view.
    #[test]
    fn the_hand_names_for_the_gateway_its_request_came_from() {
        use crate::store::Reachability;

        let store = MemoryStore::new();
        store
            .apply(preset("cache_aware", 1))
            .expect("above the unbound start");
        let member = |name: &str, reach: Reachability, contents: &[u64]| {
            let facts = WorkerFacts {
                reachable_from: reach,
                ..WorkerFacts::default()
            };
            held_by(&store, name, facts, contents)
        };
        // Three members hold the request's block: one every gateway dials
        // and one behind each of two gateways' sockets. A fourth holds
        // another block, so the head is not the whole pool's.
        let any = member("any", Reachability::Any, &[1]);
        let mine = member("mine", Reachability::Gateway(GATEWAY), &[1]);
        let theirs = member("theirs", Reachability::Gateway(OTHER), &[1]);
        member("cold", Reachability::Any, &[9]);
        let holders = |gateway: GatewayId| {
            let mut frame = frame(1, 7);
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.block_hashes = vec![1];
            facts.hash_kind = proto::HashKind::Text.into();
            let request = checked(frame);
            let now = Instant::now() + Duration::from_secs(3);
            let taken = Taken::at(&request, gateway, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            store.deciding(|world| {
                let none = AtomicU64::new(0);
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &none,
                    |facts, _, rows, _, _| {
                        let named = facts.named.expect("a sampling profile names its rows");
                        let holders = named.holders.iter();
                        let mut ids: Vec<WorkerId> =
                            holders.map(|holder| rows[holder.row as usize].id).collect();
                        ids.sort_unstable();
                        ids
                    },
                )
                .0
            })
        };
        let sorted = |mut ids: Vec<WorkerId>| {
            ids.sort_unstable();
            ids
        };
        assert_eq!(holders(GATEWAY), sorted(vec![any, mine]));
        assert_eq!(holders(OTHER), sorted(vec![any, theirs]));
    }

    /// The rows a decider's hand names are the store's own naming of the
    /// request: by the frame's first hashes up to the generation's cap, its
    /// seed, the sample of its profile's picker as the cap, its gateway's
    /// wires, and the instant its view was built at. Each is read off a
    /// fleet where it moves the rows named.
    #[test]
    fn the_hand_names_a_requests_rows_by_the_frames_own_facts() {
        use std::collections::BTreeSet;

        use crate::{
            store::{Named, Naming},
            view::NamedAffinity,
        };

        /// The lists by worker: the holders with their depths, the warming
        /// rows, the bit.
        fn lists(
            named: &NamedAffinity<'_>,
            rows: &[Candidate],
        ) -> (Vec<(WorkerId, u32)>, Vec<WorkerId>, bool) {
            let id = |row: u32| rows[row as usize].id;
            let holders = named.holders.iter();
            (
                holders
                    .map(|holder| (id(holder.row), holder.overlap_blocks))
                    .collect(),
                named.warming.iter().map(|&row| id(row)).collect(),
                named.thin,
            )
        }

        let store = MemoryStore::new();
        // The sampling preset, its view cut at two block hashes.
        let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
        document.generation = 1;
        document.view.max_block_hashes = 2;
        store
            .apply(applied(&document))
            .expect("above the unbound start");
        let member = |name: &str, wires: WireModes, contents: &[u64]| {
            holding(&store, name, Role::Single, wires, contents);
        };
        // One member past its first blocks, so the fleet is not all young;
        // seventeen more that speak gRPC and two that speak HTTP alone, each
        // holding the request's three blocks and young.
        let both = WireModes::HTTP.union(WireModes::GRPC);
        let filled: Vec<u64> = (1..=1_100).collect();
        member("big", both, &filled);
        for n in 0..17 {
            member(&format!("g{n:02}"), both, &[1, 2, 3]);
        }
        for n in 0..2 {
            member(&format!("h{n}"), WireModes::HTTP, &[1, 2, 3]);
        }
        // What the decider's hand gives the engine for a frame of three
        // block hashes from a gateway that speaks gRPC alone.
        let by_hand = |seed: u64, at: Instant| {
            let mut frame = frame(1, seed);
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.block_hashes = vec![1, 2, 3];
            facts.hash_kind = proto::HashKind::Text.into();
            let frame = constrained(frame, u32::from(WireModes::GRPC.bits()), None);
            let request = checked(frame);
            let taken = Taken::at(&request, GATEWAY, at, at).expect("budget is left");
            let mut buffers = Buffers::default();
            store.deciding(|world| {
                let none = AtomicU64::new(0);
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &none,
                    |facts, _, rows, _, _| facts.named.map(|named| lists(&named, rows)),
                )
                .0
            })
        };
        // What the store names for that request by its own door: the view
        // built and narrowed as the decider narrows it, then named.
        let by_store = |seed: u64, at: Instant| {
            let mut view = ViewBuffers::default();
            let mut named = Named::default();
            store.deciding(|world| {
                let ask = Ask {
                    gateway: GATEWAY,
                    now: at,
                    pinned: None,
                    affinity: false,
                };
                let built = world
                    .build(MODEL, Role::Single, &ask, &mut view)
                    .expect("the pool");
                view.candidates
                    .retain(|row| row.wire_modes.intersects(WireModes::GRPC));
                let naming = Naming {
                    gateway: GATEWAY,
                    block_hashes: &[1, 2],
                    seed,
                    cap: 16,
                    wires: Some(WireModes::GRPC),
                    now_micros: built.now_micros,
                };
                world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
            });
            lists(&named.affinity(), &view.candidates)
        };
        let taken = Instant::now();
        let now = taken + Duration::from_secs(3);
        // Eighteen holders the gateway dials under a cap of sixteen: the
        // draw is the seed's, and the depth the generation's cap of hashes.
        let mut drawn = BTreeSet::new();
        for seed in 1..=8 {
            let named = by_hand(seed, now).expect("a sampling profile names its rows");
            assert_eq!(named, by_store(seed, now), "seed {seed}");
            assert_eq!(named.0.len(), 16, "seed {seed}");
            assert!(named.0.iter().all(|holder| holder.1 == 2), "seed {seed}");
            assert!(!named.1.is_empty(), "the young rows warm: seed {seed}");
            drawn.insert(named.0);
        }
        assert!(drawn.len() > 1, "each seed draws its own sixteen");
        // Sixty-one seconds on, with no window published between, the young
        // rows' namings are over.
        let later = taken + Duration::from_secs(61);
        let named = by_hand(1, later).expect("a sampling profile names its rows");
        assert_eq!(named, by_store(1, later));
        assert!(named.1.is_empty());
    }

    /// A decode decision of a model that also pairs is decided by its own
    /// profile, and the hand reads that profile's sample: it caps the rows
    /// it names by it, and a profile that takes none keeps its column. So
    /// does one whose own stages read no column, and one whose warm-up
    /// stages rule by different numbers. The needs of its workers' stream
    /// carry the pair's decode side, which reads the column and samples
    /// sixteen here.
    #[test]
    fn the_hand_caps_the_rows_it_names_by_the_profile_that_decides() {
        use crate::Phase;

        // What the hand gives a decode request asked alone under `pick`:
        // the holders named with the rows handed, and the column's length.
        let handed = |score: &str, pick: &str, (own, named, ruled): (u32, u32, bool)| {
            let document: PolicySpec = serde_json::from_str(&format!(
                r#"{{
                  "apiVersion": "smg.policy/v1",
                  "generation": 1,
                  "profiles": {{
                    "alone": {{ "score": [{score}], "pick": {pick} }},
                    "wide": {{ "score": ["prefix-affinity"], "pick": {{ "best": {{ "sample": 16 }} }} }}
                  }},
                  "bindings": [
                    {{ "match": {{ "phase": ["decode"] }}, "profile": "alone" }},
                    {{ "plan": "pd-pair", "profiles": {{ "prefill": "wide", "decode": "wide" }} }}
                  ]
                }}"#
            ))
            .expect("a document of this build");
            let store = MemoryStore::new();
            store
                .apply(applied(&document))
                .expect("above the unbound start");
            store.deciding(|world| {
                let engine = world.spec().engine();
                assert_eq!(engine.needs_for(MODEL, Phase::Decode).sample, 16);
                assert_eq!(engine.sample_for(MODEL, Phase::Decode), own);
                assert_eq!(engine.named_for(MODEL, Phase::Decode), named);
                let rule = engine.warm_rule_for(MODEL, Phase::Decode);
                assert_eq!(rule.is_some(), ruled);
            });
            // Twenty decode members hold the request's one block.
            for n in 0..20 {
                let name = format!("d{n:02}");
                holding(&store, &name, Role::Decode, WireModes::NONE, &[1]);
            }
            let mut frame = phased(frame(1, 7), proto::Phase::Decode);
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.block_hashes = vec![1];
            facts.hash_kind = proto::HashKind::Text.into();
            let request = checked(frame);
            let now = Instant::now() + Duration::from_secs(3);
            let taken = Taken::at(&request, GATEWAY, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            store.deciding(|world| {
                let none = AtomicU64::new(0);
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &none,
                    |facts, _, rows, _, _| {
                        let named = facts.named.map(|named| (named.holders.len(), rows.len()));
                        (named, facts.affinity.len())
                    },
                )
                .0
            })
        };
        // A sample of four names four of the twenty holders, and no column.
        let affinity = r#""prefix-affinity""#;
        let four = r#"{ "best": { "sample": 4 } }"#;
        assert_eq!(handed(affinity, four, (4, 4, false)), (Some((4, 20)), 0));
        // No sample: the column, an entry a row, and no row named. Lists
        // capped at the pair's sample would hide holders from its pass.
        assert_eq!(handed(affinity, r#""best""#, (0, 0, false)), (None, 20));
        // A sample and no stage of its own that reads the column: the
        // pair's needs alone ask for it, and the profile is named no row.
        assert_eq!(handed(r#""least-load""#, four, (4, 0, false)), (None, 20));
        // One warm-up ruling names. Two that rule by different numbers do
        // not: a pool's warming rows are named by one ruling, which is none
        // for such a profile, and over the column each stage rules for itself.
        let one = r#"{ "warmup-slice": { "secs": 30 } }, "prefix-affinity""#;
        assert_eq!(handed(one, four, (4, 4, true)), (Some((4, 20)), 0));
        let two = r#"{ "warmup-slice": { "secs": 30 } }, { "warmup-slice": { "secs": 90 } }"#;
        assert_eq!(handed(two, four, (4, 0, false)), (None, 20));
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
            busy: busies,
            faults,
            pair_sides,
            texts,
            ..
        } = &*bed.counts;
        assert_eq!(
            (
                counted(faults),
                counted(pair_sides),
                counted(texts),
                bed.counts.hashes_cut()
            ),
            (0, 2, 1, 0)
        );
        // Each of the three is a `Busy` with no reason, and counted as one.
        assert_eq!(busies.each_ref().map(counted), [3, 0, 0, 0, 0]);
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
        let mut decider = Deciding {
            reads: bed.reads.clone(),
            buffers: Buffers::default(),
            faulted: false,
            working: None,
            name: None,
            index: 0,
            marked: None,
            mark: Arc::default(),
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
            let ask = |decider: &mut Deciding, id| {
                let handed = decider.answer(checked(frame(id, 1)), GATEWAY, bed.script.taken);
                handed.frame()
            };
            for id in 1..=3 {
                // The buffers hold something when the fault comes: a lent
                // view writes nothing into them.
                decider.buffers.view.candidates.reserve(1);
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
            decider.buffers.view.candidates.reserve(1);
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
        // The word a thread's unwind says is the one the deciders show, and
        // a receiver made after it reads it by its value.
        bed.deciders.tell_unwound();
        assert!(*bed.deciders.ended().borrow());
        let mut late = bed.deciders.ended();
        assert!(late.wait_for(|ended| *ended).await.is_ok());
    }

    /// A frame that carries `hashes` block hashes, `1` the first.
    fn hashed(id: u8, hashes: u64) -> proto::DecideRequest {
        let mut frame = frame(id, 1);
        frame
            .facts
            .as_mut()
            .expect("the case's frame has facts")
            .block_hashes = (1..=hashes).collect();
        frame
    }

    /// What the engine is handed of each frame's block hashes with
    /// `applied` as the standing generation.
    fn handed_hashes(applied: AppliedSpec, frames: Vec<proto::DecideRequest>) -> Vec<Vec<u64>> {
        let bed = bed(Some(applied), true, (2, 8), []);
        let now = bed.script.taken;
        frames
            .into_iter()
            .map(|frame| {
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
                    .0
                })
            })
            .collect()
    }

    /// The engine is handed as many of a request's block hashes as its
    /// generation reads of them, the first ones, and a request that carried
    /// more is counted.
    #[tokio::test]
    async fn the_block_hashes_past_what_the_generation_reads_are_cut_and_the_request_is_counted() {
        let first = |count: u64| (1..=count).collect::<Vec<u64>>();
        // A prefix key 256 tokens deep reads 16 blocks of 16 tokens. The
        // blocks are the request's own: 32 of a request counted by 8, 4 of
        // one counted by 64, and of a text 64 blocks of 16 characters.
        let sized = |id, hashes, block_size| {
            let mut frame = hashed(id, hashes);
            frame
                .facts
                .as_mut()
                .expect("the case's frame has facts")
                .block_size = block_size;
            frame
        };
        let mut text = hashed(6, 80);
        text.facts
            .as_mut()
            .expect("the case's frame has facts")
            .hash_kind = proto::HashKind::Text.into();
        assert_eq!(
            handed_hashes(
                preset("prefix_hash", 1),
                vec![
                    hashed(1, 20),
                    hashed(2, 16),
                    hashed(3, 2),
                    sized(4, 40, 8),
                    sized(5, 10, 64),
                    text,
                ],
            ),
            [
                first(16),
                first(16),
                first(2),
                first(32),
                first(4),
                first(64)
            ]
        );
        // A stage with no depth of its own reads to its view's cap.
        let mut held = PolicySpec::preset("cache_aware").expect("a preset of this build");
        held.generation = 1;
        held.view.max_block_hashes = 4;
        assert_eq!(
            handed_hashes(
                applied(&held),
                vec![hashed(1, 6), hashed(2, 4), hashed(3, 2)]
            ),
            [first(4), first(4), first(2)]
        );
        // No stage of the random preset reads the hashes, and none are
        // handed on.
        assert_eq!(
            handed_hashes(preset("random", 1), vec![hashed(1, 6)]),
            [first(0)]
        );
        // Never refused: each request is decided, and each that carried
        // more than its generation reads is counted.
        let mut unread = leading([]);
        for (id, hashes) in [(4, 6), (5, 1), (6, 0)] {
            let answer = unread.ask(hashed(id, hashes)).await;
            assert!(unread.workers.contains(&picked(&answer)), "{answer:?}");
        }
        assert_eq!(unread.counts.hashes_cut(), 2);
        let mut keyed = bed(Some(preset("prefix_hash", 1)), true, (2, 8), []);
        for (id, hashes) in [(7, 20), (8, 16)] {
            let answer = keyed.ask(hashed(id, hashes)).await;
            assert!(keyed.workers.contains(&picked(&answer)), "{answer:?}");
        }
        assert_eq!(keyed.counts.hashes_cut(), 1);
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
        assert_eq!(
            unleased(bed.ask(frame.clone()).await),
            bed.engines(&frame, 1)
        );
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
        assert_eq!(
            unleased(bed.ask(frame.clone()).await),
            bed.engines(&frame, 1)
        );
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

    /// How many times each worker of `picked` stands in it, in the
    /// workers' order.
    fn turns(mut picked: Vec<WorkerId>) -> Vec<usize> {
        picked.sort_unstable();
        picked
            .chunk_by(|one, next| one == next)
            .map(<[WorkerId]>::len)
            .collect()
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

    /// The rows a gateway's wires leave are a list of their own and turn a
    /// rotation of their own. Sixteen members speak HTTP alone and sixteen
    /// gRPC alone under `round_robin`, and the asks alternate between the
    /// two wires: each list is sixteen rows cut out of one build, and every
    /// member is the pick once in its list's sixteen asks. Under one
    /// rotation between the two each list was drawn at every other row.
    /// Wires that take no row leave the build's own list and its rotation:
    /// asks that state the one wire a pool speaks, in turn with asks that
    /// state none, walk its sixteen rows once.
    #[tokio::test]
    async fn the_rows_a_gateways_wires_leave_turn_a_rotation_of_their_own() {
        const TWO: &str = "model-two-wires";
        const ONE: &str = "model-one-wire";
        let mut bed = bed(Some(preset("round_robin", 1)), true, (2, 8), []);
        let store = Arc::clone(&bed.store);
        for row in 0..16 {
            ready(
                &store,
                &format!("h{row}"),
                TWO,
                Role::Single,
                WireModes::HTTP,
            );
            ready(
                &store,
                &format!("g{row}"),
                TWO,
                Role::Single,
                WireModes::GRPC,
            );
            ready(
                &store,
                &format!("o{row}"),
                ONE,
                Role::Single,
                WireModes::HTTP,
            );
        }
        let mut picked = Vec::new();
        for ask in 0..64_u8 {
            let wires = if ask.is_multiple_of(2) { 0b001 } else { 0b010 };
            let frame = constrained(frame_for(TWO, ask, u64::from(ask)), wires, None);
            picked.push(targets(&bed.ask(frame).await)[0]);
        }
        assert_eq!(turns(picked), [2; 32], "two wires over one pool");
        let mut picked = Vec::new();
        for ask in 64..80_u8 {
            let frame = frame_for(ONE, ask, u64::from(ask));
            let frame = if ask.is_multiple_of(2) {
                constrained(frame, 0b001, None)
            } else {
                frame
            };
            picked.push(targets(&bed.ask(frame).await)[0]);
        }
        assert_eq!(turns(picked), [1; 16], "wires that take no row");
    }

    /// The budget is judged again where the answer is handed over, at the
    /// read the fence takes there and with no clock read more. A decision
    /// with a microsecond left leaves. One whose budget ended while it was
    /// made leaves as `Busy` under the reason a late arrival is told, and
    /// is counted apart from one. The case's budget is half the dispatch
    /// span, so no lease's span ends where its decision's budget does.
    #[tokio::test]
    async fn a_decision_whose_budget_ended_while_it_was_made_leaves_as_busy_and_is_counted_apart() {
        let budget = DISPATCH_TTL / 2;
        let halved = |id, seed| budgeted(frame(id, seed), budget);
        let short = budget - Duration::from_micros(1);
        let under = budget - Duration::from_nanos(999);
        let steps = [
            Duration::ZERO,
            short,
            Duration::ZERO,
            under,
            Duration::ZERO,
            budget,
            budget,
        ]
        .map(Step::At);
        let mut bed = leading(steps);
        let answer = bed.ask(halved(1, 1)).await;
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 0);
        // Under one microsecond left is gone, as where a request is taken.
        for (id, count) in [(2, 1), (3, 2)] {
            let answer = bed.ask(halved(id, 2)).await;
            assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
            assert_eq!((answer.request_id.as_slice(), answer.term), (&[id][..], 1));
            assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), count);
        }
        // A request that arrived late is told the same reason at its
        // decider's first read, and is none of the count.
        let answer = bed.ask(halved(4, 4)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 2);
        assert_eq!(bed.script.readers.lock().len(), 7);
    }

    /// The counts a made decision can move are kept in parts, a decider's
    /// own at its index, so no line is written by two of them, and each is
    /// read whole. A model on each of the bed's two deciders, each with one
    /// member its gateway cannot dial, and each decider's first request
    /// with one block hash more than the view reads: a decision counts in
    /// its own decider's part, and deciders past the parts share one.
    #[tokio::test]
    async fn a_decider_counts_what_its_decisions_move_in_its_own_part() {
        let mut bed = leading([]);
        let store = Arc::clone(&bed.store);
        let part = |counts: &Counts, index| {
            let part = counts.part(index);
            let read = |count: &AtomicU64| count.load(Ordering::Relaxed);
            (read(&part.undialed), read(&part.hashes_cut))
        };
        let cap = store.deciding(|world| world.spec().max_block_hashes());
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
            // One decision on the first decider, two on the second, and
            // the first of each with its hashes past the cap.
            for seed in 0..=decider {
                id += 1;
                let mut frame = constrained(frame_for(&model, id, seed as u64), 0b010, None);
                if seed == 0 {
                    frame
                        .facts
                        .as_mut()
                        .expect("the case's frame has facts")
                        .block_hashes = (0..=cap as u64).collect();
                }
                let answer = bed.ask(frame).await;
                let shed = declined(proto::ShedReason::NoEndpoints, "wires");
                assert_eq!(outcome(&answer), &shed);
            }
        }
        assert_eq!(
            (part(&bed.counts, 0), part(&bed.counts, 1)),
            ((1, 1), (2, 1))
        );
        assert_eq!((bed.counts.undialed(), bed.counts.hashes_cut()), (3, 2));
        // A part stands on cache lines of its own, so two deciders' parts
        // share none.
        assert_eq!((size_of::<Part>(), align_of::<Part>()), (128, 128));
        let counts = Counts::default();
        for index in 0..PARTS + 2 {
            let part = counts.part(index);
            part.undialed.fetch_add(1 << index, Ordering::Relaxed);
            part.hashes_cut.fetch_add(3 << index, Ordering::Relaxed);
        }
        let whole = (1 << (PARTS + 2)) - 1;
        assert_eq!(part(&counts, 0), (1 + (1 << PARTS), 3 + (3 << PARTS)));
        assert_eq!(
            part(&counts, PARTS + 1),
            (2 + (2 << PARTS), 6 + (6 << PARTS))
        );
        assert_eq!((counts.undialed(), counts.hashes_cut()), (whole, 3 * whole));
    }

    /// The name of a cut list. Over three builds' words, the pool's whole
    /// list and two gateways' own, cut by each of the eight sets of wires:
    /// no two cuts read alike, and none reads as a build's word.
    #[test]
    fn no_two_cuts_read_alike_and_none_as_a_builds_word() {
        let builds = [0, GATEWAY.0, OTHER.0];
        let mut words = BTreeSet::from(builds);
        for built in builds {
            for bits in 0..8 {
                let cut = cut_reach(built, WireModes::from_bits(bits));
                assert!(words.insert(cut), "{built} cut by {bits:#b}");
            }
        }
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

    /// The bed's frame under a session key.
    fn keyed(key: u64, seed: u64) -> proto::DecideRequest {
        let mut frame = frame(u8::try_from(seed % 251).expect("under 251"), seed);
        frame
            .facts
            .as_mut()
            .expect("the case's frame has facts")
            .session_key = Some(proto::Identity {
            form: Some(proto::identity::Form::Hash(key)),
        });
        frame
    }

    /// A keyed session keeps its worker across an apply of a new
    /// generation. The new engine's table of sessions starts empty: a
    /// decision that finds no pin there reads the store's, which the
    /// session's `Dispatched` frame wrote, and is made on it. Inside a
    /// generation a decision of a dispatched key is made on its pin too.
    #[tokio::test]
    async fn a_keyed_session_keeps_its_worker_across_an_apply() {
        use crate::{
            controller::lifecycle::{self, tests::a_dispatched},
            sessions::SessionVerdict,
        };
        const KEYS: u64 = 24;
        let mut bed = bed(Some(preset("manual", 1)), true, (1, 8), []);
        let at = bed.script.taken;
        let verdict = |bed: &Bed, answer: &proto::DecideAnswer| {
            LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
                .expect("the lease is on file")
                .lease
                .session_verdict
        };
        // The key's value as its decision made it, read off the lease.
        let session = |bed: &Bed, answer: &proto::DecideAnswer| {
            LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
                .expect("the lease is on file")
                .lease
                .session
        };
        let pins = |bed: &Bed, session: crate::view::Session| {
            bed.store
                .deciding(|world| world.spec().engine().sessions().pins(session))
        };
        // Each key's first decision claims a worker, and its gateway's
        // Dispatched frame commits it: the engine's pin and the store's.
        let mut first = Vec::new();
        for key in 1..=KEYS {
            let answer = bed.ask(keyed(key, 1_000 + key)).await;
            let worker = picked(&answer);
            assert_eq!(verdict(&bed, &answer), SessionVerdict::Claimed(worker));
            let hand = lifecycle::open(&bed.store, GATEWAY);
            lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&answer).id), at)
                .expect("the frame reads");
            first.push((key, worker, session(&bed, &answer)));
        }
        // The keys stand on more than one worker: a bed that kept every
        // key on one would hold nothing.
        let stood = bed
            .workers
            .iter()
            .filter(|member| first.iter().any(|&(_, worker, _)| worker == **member))
            .count();
        assert!(stood > 1, "the keys stand on {stood} worker");
        for &(key, worker, _) in &first {
            let answer = bed.ask(keyed(key, 2_000 + key)).await;
            assert_eq!(picked(&answer), worker);
            assert_eq!(verdict(&bed, &answer), SessionVerdict::Pinned);
        }
        // A new generation: its engine's table holds no pin of any key.
        assert!(pins(&bed, first[0].2).is_some());
        bed.store
            .apply(preset("manual", 2))
            .expect("a later generation");
        assert_eq!(pins(&bed, first[0].2), None);
        for &(key, worker, session) in &first {
            let answer = bed.ask(keyed(key, 3_000 + key)).await;
            assert_eq!(
                picked(&answer),
                worker,
                "key {key} left its worker at the apply"
            );
            assert_eq!(verdict(&bed, &answer), SessionVerdict::Pinned);
            assert_eq!(pins(&bed, session), Some([worker, WorkerId(0)]));
        }
    }

    /// The store's pin is the key's worker before its own end and for less
    /// than the rule's TTL since its last use: the engine's table forgets
    /// a key at its TTL to the microsecond, and the pin of record is no
    /// older a pin.
    #[test]
    fn a_stored_pin_as_old_as_the_rules_ttl_is_no_pin() {
        use super::StorePins;
        use crate::{
            store::SessionsStore,
            view::{PinSource, Session},
        };
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let key = Session::from_raw(41);
        SessionsStore::pin(&store, key.get(), WorkerId(7), t0, None);
        let at = |micros: u64| StorePins {
            store: &store,
            now: t0 + Duration::from_micros(micros),
        };
        assert_eq!(at(999).pinned(key, Some(1_000)), Some(WorkerId(7)));
        assert_eq!(at(1_000).pinned(key, Some(1_000)), None);
        assert_eq!(
            at(1_000_000).pinned(key, None),
            Some(WorkerId(7)),
            "a rule with no TTL"
        );
        assert_eq!(at(0).pinned(Session::from_raw(42), None), None);
        // A pin at its own end is no pin, whatever TTL the asking rule
        // has: the engine's table forgot the key at that instant.
        let ended = Session::from_raw(43);
        let end = store.micros(t0) + 500;
        SessionsStore::pin(&store, ended.get(), WorkerId(8), t0, Some(end));
        assert_eq!(at(499).pinned(ended, Some(1_000)), Some(WorkerId(8)));
        assert_eq!(at(500).pinned(ended, Some(1_000)), None);
        assert_eq!(at(500).pinned(ended, None), None, "a rule with no TTL");
        // A decision's clock behind the pin's last use reads no age.
        let behind = StorePins {
            store: &store,
            now: t0.checked_sub(Duration::from_micros(5)).unwrap_or(t0),
        };
        assert_eq!(behind.pinned(key, Some(1)), Some(WorkerId(7)));
    }

    /// The count pass is skipped for the two requests whose own words keep
    /// no row, a subset of none and a mask of no wire, and nowhere else: a
    /// mask that no row of the pool speaks is still counted.
    #[test]
    fn a_request_is_counted_nothing_only_where_its_narrowing_keeps_no_row() {
        let narrowed = |wires: Option<WireModes>, subset: Option<u32>| {
            let mut view = ViewBuffers::default();
            let row = Candidate {
                wire_modes: WireModes::HTTP,
                ..Candidate::default()
            };
            view.candidates.extend([row; 3]);
            narrow(&mut view, wires, subset);
            view.candidates.len()
        };
        let none = [
            (None, Some(0)),
            (Some(WireModes::NONE), None),
            (Some(WireModes::NONE), Some(2)),
            (Some(WireModes::HTTP), Some(0)),
        ];
        for (wires, subset) in none {
            assert!(!keeps_a_row(wires, subset), "{wires:?} {subset:?}");
            assert_eq!(narrowed(wires, subset), 0, "{wires:?} {subset:?}");
        }
        let some = [(None, None), (None, Some(2)), (Some(WireModes::HTTP), None)];
        for (wires, subset) in some {
            assert!(keeps_a_row(wires, subset), "{wires:?} {subset:?}");
            assert_eq!(narrowed(wires, subset), 3, "{wires:?} {subset:?}");
        }
        // A mask that no row of the pool speaks keeps no row and is still
        // counted: the skip reads the request's own words, never the pool.
        assert!(keeps_a_row(Some(WireModes::GRPC), None));
        assert_eq!(narrowed(Some(WireModes::GRPC), None), 0);
    }
}
