//! The decision slots (design 5.5): the threads the controller's decisions
//! run on, each behind one bounded queue.
//!
//! A model's decisions all run on one decider ([`DecisionSlots::of`]): the
//! model is the key, so what a model's decisions write has one writer, and the
//! requests a stream sends for one model are decided in the order it sent them.
//! No runtime thread decides and none waits for a decision: a stream's reader
//! hands a request over ([`DecisionSlots::submit`]) or is told at once that the
//! queue is full, and the answer comes back on the stream's own queue.
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
//! 4. one load of the world ([`MemoryStore::deciding`]): the document's
//!    engine, the list of the model's single workers as the replica's
//!    worker objects lend it, and the places of that list the asking
//!    gateway may take by its reach, its wires and its standing vetoes
//!    ([`frame_decision::decide`]). A pair at its prefill is read its
//!    model's two lists, its prefills and its decodes, each with its own
//!    places ([`frame_decision::decide_pair`]). A prefill or a decode frame
//!    that stands alone is read the list of its own phase's role
//!    ([`frame_decision::decide_alone`]);
//! 5. the engine's decision over that list, at its entry that issues no
//!    lease, told what is left of the budget;
//! 6. the lease, for a pick: one a decision, a credit on the pick's first
//!    target, minted and filed by the store under the term the first read
//!    showed, at that read's instant ([`MemoryStore::issue_decided`]). The
//!    engine that decided is told of it, and the answer carries it. A
//!    pair's pick is filed a lease a side under one id
//!    ([`MemoryStore::issue_decided_pair`]), and its answer carries both,
//!    or the prefill side's alone where its frame stated the later decode;
//! 7. the fence again, and the budget at the same read, before sampling
//!    and handing the answer over: a decision whose budget ended while it was
//!    made, its lease's filing and the stages' hearing of it included,
//!    leaves as `Busy` under the reason a late arrival is told, and under
//!    a count of its own ([`DecideStats`]). So does a pick whose lease has
//!    under a millisecond of its dispatch span left at that read: the span
//!    counts from the first read, and the answer's lease carries what is
//!    left of it. The lease of an answer that does not leave is given up
//!    at once: the store's release door first, then the engine told, as
//!    abandoned under a closed fence and as a late decision past the
//!    budget or the span.
//!
//! A pick or shed past the last read can be sampled for Explain. Each
//! decider counts its own answers and takes its ring's lock only to keep
//! a sample. A reader takes the lock for one scan.
//!
//! A lease is what the next decisions read. The store counts the leases on
//! file by the worker each books, and steps the count of the worker's
//! object with each booking and freeing ([`crate::tracking::HostCount`]):
//! a decision over worker lists reads a worker's requests in flight there.
//! A decision is made over what the decisions before it
//! left on file. Each decider issues into a file of leases of its own,
//! under that file's turn, held for an apply and for one lease's filing or
//! its giving up, never across a decision. A worker's count is one atomic
//! a row, whichever file books it. What a gateway's frame says of a lease
//! is handed by its session to the file the lease's id names and applied
//! by that file's decider at work: as it takes up its work, by each
//! request it takes whatever its answer, ahead of the rows' counts a
//! decision reads, at its issue, and once more before the decider
//! waits on an empty queue. So a decision reads its own file
//! behind everything handed to it, and a row another file books lags by
//! that file's next take. Where a file's decider waits for a job and
//! its file holds no lease, the session applies what it hands. While a
//! lease stands there the decider keeps its watch and applies what is
//! handed at its next look ([`Deciding::wait`]). A controller's only
//! decider keeps none. A lease leaves its file where its
//! answer does not leave (step 7), at its gateway's word (a `Session`'s
//! `Completed` or `Released` frame, the gateway's own end) or at its
//! dispatch deadline: each issue ends the leases at the head of its
//! file's deadline queue that no gateway named a dispatch for in time,
//! two at most, and the gateways' sweep ends them in a file whose decider
//! waits for a job. Their engine is told under the turn where its
//! generation still stands. So a lease whose answer never reached its
//! gateway (an answer lost behind its stream's send, a decision that
//! panicked past its issue) stands until its gateway asks the request
//! again, or else until its deadline and then the next issue of its file
//! or the sweep's next tick. A decision its stream stops at the send is
//! none of them, nor one still in the queue of a stream that ends: the
//! stream hands its lease's release there, as a session hands a frame's
//! ([`super::decide`]). Nor is an answer its stream's queue does not
//! take, full or closed: its decider hands the release where the queue
//! refuses it and takes its file's turn behind the hand, so its next
//! decision reads no count of that lease.
//!
//! A single request, a pair at its prefill and a decode call behind a
//! pair's held prefill are decided here. So is a frame that stands alone
//! at a side's phase: a prefill frame, or a decode frame that names no
//! lease, whose plan pairs no side of its phase. The binding of its own
//! phase decides it over that phase's list, and its pick is filed a
//! single's lease. An encode request that carries its items is decided
//! here too, as its pair's one request: for both sides and for each
//! item's encoder, with its pair's leases and none for an encoder. A frame
//! of the encode phase that does not state `DecideRequest.encode` has no
//! decision over worker lists, and a decode frame that names no lease
//! under a plan that pairs it has none at all: each is answered `Busy`
//! with no reason, and its gateway decides it. A decode call its lease
//! refuses is answered `Busy` under that refusal's own reason: at the
//! call's first test of the lease, ahead of every stage its decision runs,
//! or at its issue, where the lease moved under the decision.
//!
//! One lease stands for a request its gateway asks again. A gateway whose
//! stream ended with a request unanswered sends it again and says so
//! (`DecideRequest.again`). The request's decider then ends the lease the
//! earlier ask was issued, where it still waits for a dispatch, as the
//! dispatch deadline would end it ([`MemoryStore::asked_again`]), and
//! decides the request as any other. The earlier lease is on that
//! decider's file: a model's requests go to one decider for the process's
//! life ([`DecisionSlots::of`]). A request sent twice under one id without the
//! word is two requests and is issued two leases: a lease is keyed by its
//! gateway and the id the store mints, and no two asks are joined by the
//! id their asker states. A repeat its stream answers `Busy` before a
//! decider takes it (no credit, a full queue, a budget of 0) ends
//! nothing: the earlier lease waits out its deadline. The word is for a
//! stream that ended: an answer that waits in an open stream's queue is
//! not held back, and it leaves with the lease the word ended.
//!
//! What a lease does not do yet. A stage hears a lease's issue here, its
//! release where its decider's last read or its stream's send stops its
//! answer, where its stream's queue refuses it and where that stream ends
//! with it still in the queue, its dispatch, completion and release at its
//! gateway's word and its end at the dispatch deadline or where its
//! request is asked again, and not the end of a lease whose gateway was
//! let go: a document that names a stage in need of the calls stays
//! refused.
//!
//! A term mints 2^32 - 1 ids, and under every term but the last a leader
//! leaves before they end. The first answer a term hands over with an id at
//! or past its mark ([`past_mark`], three quarters of the ids) tells the
//! election to hand the lease over ([`DecisionSlots::leave_through`]), once a
//! term whichever decider's it is: the next take brings a term whose ids
//! start again. The counter's end stands as the last guard, where a pick is
//! answered `Busy`: under the last term a replica can take no word is said.
//!
//! A gateway's wires are a fact before the profile, as its vetoes are: no
//! filter of the catalog reads them, and no profile picks a member its
//! gateway cannot dial. The candidates the wires take are counted, and a
//! decision whose pool has rows and whose gateway shares a wire with none
//! of them is shed under [`WIRES`](crate::engine::decision::WIRES).
//! One whose gateway hints a subset of 0, a decision over no candidates, is
//! shed under [`SUBSET_HINT`](crate::engine::decision::SUBSET_HINT).
//! A positive hint is read by no stage yet.
//!
//! What this build has no decision for is answered `Busy` with no reason,
//! and its gateway decides it locally at once: a shed is a policy's word
//! and reaches the client, and no policy spoke. Four arms, each counted
//! by its cause ([`DecideStats`]): a decision that panics, an encode
//! request or a decode frame that names no lease under a plan that pairs
//! it, a request that carries its prompt as text
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
//! never one a request. A lease's lifecycle call that panics is held
//! where the engine is told ([`lifecycle`]), whoever applies the lease's
//! event: it costs no request its decision and ends no decider.
//! A decider's thread that unwinds anywhere else leaves its models with no
//! decider: the deciders tell it ([`DecisionSlots::ended`]), for the controller
//! to stop, since a leader that cannot decide for a model must not keep
//! the lead.
//!
//! The engine holds a panic of its own decision under its own catch and
//! refuses the request ([`frame_decision::decide`]). Its decider answers
//! and counts that request as one its own catch held.
//!
//! What this step leaves to the next ones. A decision reads no load report
//! under a document with a stage that keeps reports. An encode request has
//! no decision, and neither has a decode frame that names no lease under a
//! plan that pairs the decode phase.

use std::{
    io, mem,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError},
        Arc, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::load::state::{LoadReceiver, LoadSnapshot};
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

use super::{
    election::ElectionObservation,
    frame_decision::{self, FrameBuffers},
    frame_loads::FrameLoads,
    lease_words::issuing,
    lifecycle,
    wire::{self, Checked},
};
use crate::{
    engine::{
        decision::Decision,
        decision_engine::{DecisionRefused, DecodeRefused},
        decision_slot::DecisionSlot,
        explain::{DecisionRecord, SampleCount, Samples},
        framework::{release, Engine},
        view::{fingerprint, Phase},
    },
    service::{
        elector::{term_said, Fence},
        proto::{self, busy::Reason},
    },
    stages::lifecycle::Lifecycle,
    store::{
        AtWork, DecodeIssue, GatewayId, Hand, LeaseRefused, LookOutcome, MemoryStore, StoreSnapshot,
    },
    tracking::lease::{Lease, LeaseId, ReleaseReason, DISPATCH_TTL},
    worker_host::{Members, Objects},
};

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

/// One of the parts' counts, summed over `parts`.
pub(crate) fn summed(parts: &[Part], count: impl Fn(&Part) -> &AtomicU64) -> u64 {
    let parts = parts.iter();
    parts.map(|part| count(part).load(Ordering::Relaxed)).sum()
}

/// A request as its decision slot takes it.
#[derive(Clone, Copy)]
pub(crate) struct DecisionCall<'a, A> {
    pub(crate) request: &'a A,
    pub(crate) gateway: GatewayId,
    /// The decider's first read of its clock: the decision's instant.
    pub(crate) now: Instant,
    /// What is left of the request's budget at that instant.
    pub(crate) left: Duration,
    /// The place of the decider that decides, at its file of leases: a
    /// decision whose profile holds a stage that may refuse by a report
    /// of leases applies what waits at the files of leases through it.
    /// None for a caller with no place, where no file's frames wait for a
    /// look.
    pub(crate) place: Option<&'a AtWork>,
}

impl<'a, A> DecisionCall<'a, A> {
    /// When the dispatch span of a lease filed for the request ends: the
    /// span counts from the read the request was taken at.
    pub(crate) fn due(&self) -> Instant {
        self.now + DISPATCH_TTL
    }
}

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
pub(crate) fn past_mark(id: LeaseId) -> bool {
    (id.0 & u64::from(u32::MAX)) >= u64::from(TERM_MARK)
}

/// The decider a model's decisions run on among `deciders` of them: the
/// model is the key, so what a model's decisions write has one writer.
pub(crate) fn decider_of(model: &str, deciders: usize) -> usize {
    let deciders = u64::try_from(deciders.max(1)).unwrap_or(1);
    usize::try_from(fingerprint(model.as_bytes()) % deciders).unwrap_or(0)
}

/// What the deciders and the streams count on the controller's own side,
/// for its scrape to render: each `Busy` by its reason, the requests this
/// build has no decision for by cause, the requests whose block hashes
/// were cut, the answers no stream took, the candidates a gateway's wires
/// took from its decisions, and the decisions cut at their last read with
/// their budget or their lease's dispatch span all but gone. A request that
/// is decided inside both, with its hashes inside the cap and every
/// candidate one its gateway can dial, moves none of them.
#[derive(Debug, Default)]
pub(crate) struct DecideStats {
    /// `Busy` answers, each at its reason's number on the wire.
    pub(crate) busy: [AtomicU64; 9],
    /// Decisions that panicked.
    pub(crate) faults: AtomicU64,
    /// Requests that have no decision over worker lists: an encode
    /// request, which has none yet, and a decode frame that names no
    /// lease under a plan that pairs the decode phase.
    pub(crate) pair_sides: AtomicU64,
    /// Requests that carried their prompt as text.
    pub(crate) texts: AtomicU64,
    /// Requests whose decider's thread had ended.
    pub(crate) gone: AtomicU64,
    /// Answers a stream's queue had no place for. A stream's window rules
    /// it out: its requests in flight are at most its queue's places.
    pub(crate) dropped_full: AtomicU64,
    /// Answers whose stream had ended: at their decider's send, or with
    /// them still in its queue.
    pub(crate) dropped_gone: AtomicU64,
    /// What a decision that is made may count, each decider's in a part of
    /// its own: read whole by [`DecideStats::undialed`] and
    /// [`DecideStats::hashes_cut`].
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
    /// Requests no pick was made of while the start's document of a
    /// replica under a shared lease waited for its first lead. Each leaves
    /// as `Busy` with no reason: no decision is made under no document.
    pub(crate) awaited: AtomicU64,
    /// Requests their gateway said it asked again, holding no answer to
    /// the first ask: at 0 those that found no lease of the request
    /// waiting for a dispatch, at 1 those whose earlier lease was ended.
    pub(crate) again: [AtomicU64; 2],
}

/// The parts a made decision's counts are kept in. A decider adds to the
/// one at its index, and deciders past this many share one.
const PARTS: usize = 16;

impl DecideStats {
    /// The reasons of a `Busy`, in the order of their numbers on the wire.
    pub(crate) const BUSY: [&'static str; 9] = [
        "unspecified",
        "queue_depth",
        "no_credit",
        "budget_gone_on_arrival",
        "not_leader",
        "decode_no_lease",
        "decode_not_chosen",
        "decode_prefill_not_held",
        "decode_stands",
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
        summed(&self.parts, count)
    }
}

/// The requests one decider's queue holds: one stream's window. At the
/// design's cost of a decision a full queue is one cross-node budget of
/// waiting, and a request behind it is told at once to decide locally.
pub(crate) const QUEUE_DEPTH: usize = 128;

/// The shortest wait of a decider with no job at hand before it looks at
/// its file of leases again, while a lease stands on the file
/// ([`Deciding::wait`]): its wait behind a job and behind a look that
/// applied a frame. The lag it buys: while a file's frames flow, a frame
/// a session hands while the file's decider waits for a job is applied
/// within one such wait and its thread's wake, and another decider's
/// decision reads what its apply writes that much later.
const LOOK_FLOOR: Duration = Duration::from_millis(1);

/// The longest such wait: the wait doubles behind each look that applied
/// no frame, up to this. The lag it buys: the first frame a session hands
/// to a file that was quiet is applied within this wait and its thread's
/// wake, and another decider's decision reads what its apply writes that
/// much later. For that a decider with a lease on a quiet file and no
/// job wakes some sixty times a second, where the floor alone would wake
/// it a thousand times; one whose file holds no lease wakes for a job
/// alone.
const LOOK_CAP: Duration = Duration::from_millis(16);

/// The clock a decider reads, twice a decision.
pub(crate) type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// What a decider calls once it handed over a term's first answer at or
/// past the mark: the election's planned hand-over, handed that term.
pub(crate) type Leave = Box<dyn Fn(u64) + Send + Sync>;

/// What the deciders share of a term's mark: the word, and the highest term
/// it was said under, so one decider says it and once.
#[derive(Default)]
struct Mark {
    /// The word ([`DecisionSlots::leave_through`]): none until the election
    /// runs.
    leave: OnceLock<Leave>,
    /// The highest term whose word was said, or was found to have none to
    /// say: the last term a replica can take.
    said: AtomicU64,
}

#[derive(Default)]
struct Shared {
    mark: Mark,
    /// One lock per decider, laid out on the first kept sample.
    samples: OnceLock<Box<[Mutex<Samples>]>>,
}

type Answered = Result<(u64, Decision, Option<wire::Leased>), Reason>;

impl Shared {
    #[cold]
    #[inline(never)]
    fn sampled(
        &self,
        (index, deciders): (usize, usize),
        count: &mut SampleCount,
        engine: &Engine,
        taken: &DecisionCall<'_, Checked>,
        order: impl FnOnce() -> Answered,
    ) -> Answered {
        let answer = order()?;
        if count.keeps(answer.0, engine, &answer.1) {
            let seats = self
                .samples
                .get_or_init(|| (0..deciders).map(|_| Mutex::default()).collect());
            let mut samples = seats[index].lock();
            // The record names the binding its decision ran under, as the
            // in-process door's does: a decode call's is its pair's,
            // whatever phase its frame states.
            let request = &taken.request;
            let binding = match request.decode_of() {
                Some(_) => engine.pair_position(request.model()),
                None => engine.binding_position(request.model(), request.phase()),
            };
            let asked = (taken.now, taken.gateway, request.id());
            samples.push(answer.0, binding, asked, &answer.1, None, [None; 2]);
        }
        Ok(answer)
    }
}

/// What every decider reads.
#[derive(Clone)]
pub(crate) struct Reads {
    pub(crate) store: Arc<MemoryStore>,
    /// The replica's worker objects: a decision is made over the list
    /// their registry holds for its model.
    pub(crate) objects: Arc<Objects<Members>>,
    /// The channel of the replica's load monitor: a decision reads the
    /// reports of the snapshot that stands on it.
    pub(crate) loads: LoadReceiver,
    /// The controller's one fence.
    pub(crate) fence: Arc<Fence>,
    /// The election's last sight of the lease, for the term a `Busy`
    /// carries.
    pub(crate) standing: watch::Receiver<ElectionObservation>,
    pub(crate) clock: Clock,
    pub(crate) counts: Arc<DecideStats>,
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

/// The decision slots. Their drop ends each thread, after the requests its
/// queue still holds.
pub(crate) struct DecisionSlots {
    /// Each decider's queue, at its index.
    queues: Box<[SyncSender<Job>]>,
    /// Shows `true` once a decider's thread unwound.
    told: Arc<watch::Sender<bool>>,
    /// What a stream's reader reads beside its deciders: the clock a
    /// frame's receipt is stamped by and the term the reader's own answers
    /// carry.
    reads: Reads,
    shared: Arc<Shared>,
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

impl DecisionSlots {
    /// Starts `deciders` threads, each behind a queue of `depth` requests:
    /// at least one thread and at most `DecisionSlot::COUNT`, and a queue of at
    /// least one. Refused over a store whose files were laid for another
    /// count of deciders.
    pub(crate) fn start(deciders: usize, depth: usize, reads: &Reads) -> io::Result<Self> {
        // A decider issues into a file of its own: the store was made for
        // this count of deciders, or its one file would have two writers.
        let deciders = deciders.clamp(1, DecisionSlot::COUNT);
        if reads.store.decision_slots() != deciders {
            let files = "the store's files of leases are not one a decider";
            return Err(io::Error::new(io::ErrorKind::InvalidInput, files));
        }
        // A replica's shared state gave the store its word already, and the
        // first stays: this gives it for deciders started with no such state.
        lifecycle::tells(&reads.store, &reads.objects);
        let told = Arc::new(watch::Sender::new(false));
        let shared = Arc::new(Shared::default());
        let queues = (0..deciders)
            .map(|index| {
                let (queue, jobs) = sync_channel(depth.max(1));
                let decider = Deciding {
                    reads: reads.clone(),
                    frames: FrameBuffers::default(),
                    loads: FrameLoads::of(reads.loads.clone()),
                    sample_count: SampleCount::default(),
                    faulted: false,
                    caught: false,
                    index,
                    marked: None,
                    shared: Arc::clone(&shared),
                    working: None,
                    name: u8::try_from(index).ok().map(DecisionSlot::new),
                };
                watched(index, &told, move || decider.run(&jobs)).map(|_running| queue)
            })
            .collect::<io::Result<_>>()?;
        Ok(Self {
            queues,
            told,
            reads: reads.clone(),
            shared,
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
        let _ = self.shared.mark.leave.set(Box::new(leave));
    }

    /// Scans one seat at a time for a record kept under the term read.
    pub(crate) fn explain(
        &self,
        term: u64,
        gateway: GatewayId,
        request_id: &[u8],
    ) -> Option<DecisionRecord> {
        let seats = self.shared.samples.get()?;
        let key = fingerprint(request_id);
        seats
            .iter()
            .filter_map(|seat| seat.lock().find(term, gateway, key))
            .max_by_key(|record| record.at)
    }

    /// What a decider's thread tells at its unwind, told by a case.
    #[cfg(test)]
    pub(crate) fn tell_unwound(&self) {
        self.told.send_replace(true);
    }

    /// The decider `model`'s decisions run on: every request for a model
    /// goes to one, whichever stream carries it.
    pub(crate) fn of(&self, model: &str) -> usize {
        decider_of(model, self.queues.len())
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
    pub(crate) fn counts(&self) -> &Arc<DecideStats> {
        &self.reads.counts
    }

    /// Hands `job` to its decider, without waiting: its model's, or for a
    /// decode call the one whose file holds the call's lease
    /// ([`MemoryStore::file_of`]). A file is its decider's own, so a
    /// decode call is decided, and its issue taken back, by the thread at
    /// work at its lease's file, whatever model the call names and
    /// whichever term minted the lease. A file past the queues, which a
    /// store built with a file a decider never names, falls to the
    /// model's decider. `Some` is the request back with why it took no
    /// place: its submitter answers it.
    pub(crate) fn submit(&self, job: Job) -> Option<(Refused, Job)> {
        let home = job.request.decode_of().and_then(|lease| {
            let file = self.reads.store.file_of(lease);
            self.queues.get(file)
        });
        let queue = home.unwrap_or_else(|| &self.queues[self.of(job.request.model())]);
        match queue.try_send(job) {
            Ok(()) => None,
            Err(TrySendError::Full(job)) => Some((Refused::Full, job)),
            Err(TrySendError::Disconnected(job)) => Some((Refused::Gone, job)),
        }
    }
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
    frames: FrameBuffers,
    /// The newest snapshot of the load monitor, held between frames.
    loads: FrameLoads,
    sample_count: SampleCount,
    /// The last decision panicked: a streak of them writes one line.
    faulted: bool,
    /// The engine caught a panic of the decision in hand and said so
    /// ([`frame_decision::decide`]): the decider counts it as one it
    /// caught itself.
    caught: bool,
    /// Its place among the deciders, which names its part of a count.
    index: usize,
    /// The term of the answer in hand where its lease's id is at or past
    /// the term's mark: the word follows the answer's hand-over.
    marked: Option<u64>,
    shared: Arc<Shared>,
    /// Its seat at its own file of leases, taken at its first job
    /// ([`MemoryStore::at_work`]): at work while it has a job at hand, and
    /// on its watch while a lease stands on the file.
    working: Option<AtWork>,
    /// Its name among the host's deciders, which its thread acts as while
    /// it decides ([`DecisionSlot::enter`]): none past the names a host gives.
    name: Option<DecisionSlot>,
}

impl Deciding {
    /// Decides until the last handle of the queue is dropped. With a job
    /// at hand the decider is at work at its own file of leases: what the
    /// sessions hand to that file is left to it, applied as it takes up
    /// its work, by each request it takes, refused or decided, where the
    /// file's turn is free, at each issue, and behind the give-up of an
    /// answer its stream refused. It leaves that work before it waits on
    /// an empty queue, and keeps its watch over the file while a lease
    /// stands on it ([`Self::wait`]): what is handed meanwhile is still
    /// its own to apply. A controller's only decider keeps no watch: what
    /// is handed while it waits is applied by its hand.
    fn run(mut self, jobs: &Receiver<Job>) {
        loop {
            let job = match jobs.try_recv() {
                Ok(job) => job,
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {
                    let Some(job) = self.wait(jobs) else { break };
                    job
                }
            };
            // The thread acts as its decider for this job: a local, so the
            // name is let go where the job ends.
            let _acting = self.name.map(DecisionSlot::enter);
            // Its seat is at the file its name finds. It says it works and
            // takes its turn before its first decision.
            match &mut self.working {
                Some(seat) => seat.work(),
                None => self.working = Some(self.reads.store.at_work()),
            }
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

    /// Waits for the next job on an empty queue; `None` once the queue's
    /// last handle is dropped. The decider leaves its work at its file
    /// first ([`Self::look`]). While a lease stands on the file it keeps
    /// its watch: it waits a period at most and looks again, so a frame
    /// handed on a lull is applied on this thread, and the session that
    /// hands it passes it on unapplied. The period is [`LOOK_FLOOR`]
    /// behind a job and behind a look that applied a frame, since a file
    /// whose frames flow is not quiet, and twice the last behind a look
    /// that applied none, up to [`LOOK_CAP`]: a quiet file costs its
    /// decider few wakes. With no lease on file it waits with no bound: a
    /// frame handed there is applied by its hand.
    ///
    /// A controller's only decider keeps no watch: it rests and waits
    /// with no bound. A session applies in place only while a decider
    /// has no job. With one decider the watch would move no apply off a
    /// thread that has work, and a wait under a period costs a decision
    /// more than a wait with no bound. The count read here does not
    /// change under a decider: a store's files are laid where it is
    /// built ([`MemoryStore::with_decision_slots`]), before a thread shares it.
    fn wait(&mut self, jobs: &Receiver<Job>) -> Option<Job> {
        if self.reads.store.decision_slots() == 1 {
            self.rest();
            return jobs.recv().ok();
        }
        self.wait_by(|within| match within {
            Some(period) => jobs.recv_timeout(period),
            None => jobs.recv().map_err(RecvTimeoutError::from),
        })
    }

    /// [`Self::wait`] over `receive`, its queue's receive: within the
    /// period it is handed, and with no bound where it is handed none.
    fn wait_by(
        &mut self,
        mut receive: impl FnMut(Option<Duration>) -> Result<Job, RecvTimeoutError>,
    ) -> Option<Job> {
        let mut found = self.look();
        let mut period = LOOK_FLOOR;
        loop {
            let within = (found != LookOutcome::Ended).then_some(period);
            match receive(within) {
                Ok(job) => return Some(job),
                Err(RecvTimeoutError::Disconnected) => return None,
                Err(RecvTimeoutError::Timeout) => {
                    found = self.look();
                    period = match found {
                        LookOutcome::Fed => LOOK_FLOOR,
                        LookOutcome::Quiet | LookOutcome::Ended => (2 * period).min(LOOK_CAP),
                    };
                }
            }
        }
    }

    /// A look at the decider's file with no job at hand
    /// ([`AtWork::look`]): it leaves its work there and says what stands
    /// of its watch, none before its first job. What was handed behind
    /// its last take is applied here, and the engine's lifecycle calls
    /// run with it, outside any decision. A call that panics is held
    /// where it is made ([`lifecycle::tells`]'s word to the engine): it
    /// costs no request and ends no decider.
    fn look(&mut self) -> LookOutcome {
        self.working
            .as_mut()
            .map_or(LookOutcome::Ended, AtWork::look)
    }

    /// Leaves the decider's work at the leases' file and ends its watch:
    /// the thread owes the file no further look. What was handed behind
    /// its last take is applied here, as at a look.
    fn rest(&mut self) {
        if let Some(seat) = &mut self.working {
            seat.rest();
        }
    }

    /// Gives up the lease of an answer its stream did not take: the queue
    /// was full, or the stream had ended. No gateway will name a dispatch
    /// for it, and it would book its worker to its dispatch deadline. The
    /// decider hands the give-up's word to its own file as abandoned
    /// ([`wire::Leased::given_up`]: a decode call's answer takes its one
    /// issue back, any other lease is released whole), as a
    /// stream does for a decision it stops at the send, and gives the hand
    /// back as a stream does at its end ([`MemoryStore::hand_back`]): it
    /// takes its file's turn, whoever holds it, and the release is applied
    /// before the give-up returns. Its look ahead of a decision passes a
    /// held turn ([`AtWork::apply`]): a release left to that look could
    /// book its worker through the next pick. The wait is the turn's,
    /// outside any decision, as is a full ring's, where the hand applies
    /// in place. A stage's call that panics at either take is held where
    /// the engine is told, as every call of a lease's event is: the hand
    /// is through, and the file holds what the word left.
    #[cold]
    fn unsent(&self, gateway: GatewayId, answer: wire::Decided) {
        let wire::Decided::Decision {
            lease: Some(leased),
            ..
        } = answer
        else {
            return;
        };
        let hand = Hand::new(gateway);
        let abandoned = ReleaseReason::Abandoned;
        leased.given_up(&self.reads.store, &hand, abandoned);
        // The decider is at work at that file, so the hand left the frame
        // to it: it takes the turn here.
        self.reads.store.hand_back(hand);
    }

    /// The word of `term`'s mark: the planned hand-over, so the next take
    /// brings a term whose ids start again. Once a term: every answer past
    /// the mark comes here, and the first alone goes on. Under the last
    /// term below the ceiling no word is said: a hand-over there leaves no
    /// term to take, and the counter runs on to its end, where a pick is
    /// answered `Busy` ([`DecideStats::unleased`]).
    #[cold]
    fn leave_at(&self, term: u64) {
        // A read first: past the mark every answer comes here until the
        // hand-over lands, and a write each time would pass the line from
        // decider to decider.
        let mark = &self.shared.mark;
        if mark.said.load(Ordering::Relaxed) >= term
            || mark.said.fetch_max(term, Ordering::Relaxed) >= term
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
        let Some(leave) = mark.leave.get() else {
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
        let caught = mem::take(&mut self.caught);
        let faulted = decided.is_err() || caught;
        let answer = match decided {
            Ok(Ok((term, decision, lease))) => {
                // An id at or past the term's mark: the word follows this
                // answer, where no answer of the term said it yet. A decode
                // call's answer minted none: its lease is its pair's, of
                // whatever term minted it, and says no word.
                if lease.is_some_and(|leased| past_mark(leased.lease.id) && leased.minted()) {
                    self.marked = Some(term);
                }
                // An encode request's pick leaves with where its items go,
                // as its decision left it on this thread.
                let (id, encode) = if request.carries_items() {
                    self.frames.answered(request, &decision)
                } else {
                    (request.into_id(), None)
                };
                // The decision leaves as it was made: its stream builds
                // the frame where it sends it.
                wire::Decided::Decision {
                    id,
                    term,
                    decision,
                    lease,
                    encode,
                }
            }
            Ok(Err(reason)) => {
                if caught {
                    self.fault_said();
                }
                wire::Decided::Frame(self.reads.busy(request.into_id(), reason))
            }
            Err(payload) => {
                // A payload's own drop can panic, and that panic let out
                // here would unwind this thread with the request
                // unanswered: the engine's helper frees it under a catch
                // of its own.
                release(payload);
                // The next decision reads nothing of this one: the lists
                // and the places stay where the fault found them, and are
                // dropped whole.
                self.frames = FrameBuffers::default();
                self.fault_said();
                wire::Decided::Frame(self.reads.busy(request.into_id(), Reason::Unspecified))
            }
        };
        self.faulted = faulted;
        answer
    }

    /// Counts a decision that panicked, and writes one line for a streak
    /// of them: the decider's own catch and the engine's alike.
    fn fault_said(&self) {
        self.reads.counts.faults.fetch_add(1, Ordering::Relaxed);
        if !self.faulted {
            error!(
                decider = thread::current().name().unwrap_or_default(),
                "a decision panicked: its request is answered Busy with no reason, and so is each one that panics after it until one does not"
            );
        }
    }

    /// What a request is refused for before anything is built, or the term
    /// the fence shows at `now` with the request as taken then. The fence
    /// is read first: a replica that does not lead builds nothing, and
    /// says so before any other reason. A text is said next, ahead of the
    /// budget, so its count holds every text a leader took. A request
    /// asked again ends its earlier lease there too, whatever becomes of
    /// its own decision. The two are words few requests say, behind one
    /// test.
    fn judged<'a>(
        &self,
        request: &'a Checked,
        gateway: GatewayId,
        received: Instant,
        now: Instant,
    ) -> Result<(u64, DecisionCall<'a, Checked>), Reason> {
        let Reads { fence, counts, .. } = &self.reads;
        let term = fence.leader_term(now).ok_or(Reason::NotLeader)?;
        if request.says_more() {
            if request.has_text() {
                counts.texts.fetch_add(1, Ordering::Relaxed);
                return Err(Reason::Unspecified);
            }
            if request.again() {
                // Beside a lease's decode side the word reaches that side
                // alone: no search by the id's fingerprint, which could
                // end another request's lease.
                match request.decode_of() {
                    Some(lease) => self.decode_asked_again(gateway, lease),
                    None => self.asked_again(request, gateway),
                }
            }
        }
        let taken = DecisionCall::at(request, gateway, received, now);
        Ok((term, taken.ok_or(Reason::BudgetGoneOnArrival)?))
    }

    /// A request its gateway asks again, holding no answer to the first
    /// ask: the lease that ask was issued ends here, where it still waits
    /// for a dispatch, as its dispatch deadline would end it
    /// ([`MemoryStore::asked_again`]). It ends ahead of the request's own
    /// decision and whatever that decision's answer, a `Busy` for a spent
    /// budget included: no dispatch will name it. Counted by whether a
    /// lease was ended.
    #[cold]
    fn asked_again(&self, request: &Checked, gateway: GatewayId) {
        let Reads { store, counts, .. } = &self.reads;
        let ended = store.asked_again(gateway, fingerprint(request.id()));
        counts.again[usize::from(ended)].fetch_add(1, Ordering::Relaxed);
    }

    /// A decode call its gateway asks again, holding no answer to the
    /// first: the side that call issued returns to chosen here, where no
    /// dispatch named it ([`MemoryStore::take_back_decode`]). It returns
    /// ahead of the repeat's own decision, so whatever the repeat answers
    /// the first pick's dispatch books the side, and a pick of the repeat
    /// issues it anew. Counted as a request asked again, by whether an
    /// issue was taken back.
    #[cold]
    fn decode_asked_again(&self, gateway: GatewayId, lease: LeaseId) {
        let Reads { store, counts, .. } = &self.reads;
        let ended = store.take_back_decode(gateway, lease, ReleaseReason::Abandoned);
        counts.again[usize::from(ended)].fetch_add(1, Ordering::Relaxed);
    }

    /// One request's decision with the term both reads of the fence showed
    /// and, where it is a pick, the lease issued for it with what is left
    /// of its dispatch span, or the reason its answer is `Busy`.
    fn decide(&mut self, request: &Checked, gateway: GatewayId, received: Instant) -> Answered {
        let Reads {
            store,
            objects,
            fence,
            clock,
            counts,
            ..
        } = &self.reads;
        let now = clock();
        let judged = self.judged(request, gateway, received, now);
        // What the sessions handed is applied where the turn is free,
        // whatever was judged: before the leases' counts are read, and by
        // a request refused above too, so a hand at a full ring waits for
        // the first request of this decider that finds the turn free.
        if let Some(at_work) = &self.working {
            at_work.apply();
        }
        let (term, taken) = judged?;
        // A single request, a pair at its prefill, a decode call behind a
        // pair's held prefill and a prefill or a decode frame that stands
        // alone have a decision over worker lists. So has an encode
        // request that carries its items: it is read at the prefill phase,
        // as its pair's one request. A frame of the encode phase that
        // does not state them has no decision: it is answered `Busy` with
        // no reason before anything is read for it, and its gateway
        // decides it locally. A frame that names a lease is read at the
        // decode phase, so a single request and a prefill are asked for
        // none.
        let entry = match request.phase() {
            Phase::Single => Entry::Single,
            Phase::Prefill => Entry::Prefill,
            Phase::Decode => match request.decode_of() {
                Some(lease) => Entry::Decode(lease),
                None => Entry::DecodeAlone,
            },
            Phase::Encode => {
                counts.pair_sides.fetch_add(1, Ordering::Relaxed);
                return Err(Reason::Unspecified);
            }
        };
        let taken = taken.placed(self.working.as_ref());
        let (frames, own) = (&mut self.frames, counts.part(self.index));
        let caught = &mut self.caught;
        // The load reports the decision reads: the monitor's newest, taken
        // once a frame.
        let loads = self.loads.newest();
        // One load of the world for the decision and for its lease: the
        // engine that decides is the one whose stages the store tells of
        // the lease.
        let mut order = |world: &StoreSnapshot| {
            // The lease, for a pick: filed under the first read's term and
            // at its instant, and the stages told, before the read that
            // lets the answer leave. That read judges the store's write
            // and a stage's call with the decision before them. A pair's
            // pick is filed its decode side's lease beside it, and a decode
            // call's pick is filed under the pair's own lease. A frame
            // that stands alone is filed as a single request's is.
            let (decision, filed, beside) = match entry {
                Entry::Single => {
                    let decided =
                        frame_decision::decide(world, objects, &taken, frames, loads, own);
                    let (decision, booked) = made(decided, caught)?;
                    let filed =
                        frame_decision::issue(store, world, &taken, term, &decision, booked);
                    (decision, filed, FiledBeside::Alone)
                }
                Entry::Prefill => {
                    let theirs = (&mut *frames, loads, own);
                    prefill(store, world, objects, &taken, term, theirs, caught)?
                }
                Entry::Decode(lease) => {
                    let theirs = (&mut *frames, loads, own);
                    decode_call(store, world, objects, &taken, lease, theirs, caught)?
                }
                Entry::DecodeAlone => {
                    let theirs = (&mut *frames, loads, own, counts.as_ref());
                    decode_alone(store, world, objects, &taken, term, theirs, caught)?
                }
            };
            // No pick, while the start's document of a replica under a
            // shared lease waits for its first lead: a shed would say that
            // a document refuses the request, and no document has landed.
            // That is no decision. The mark is read behind a decision that
            // booked nothing, and there alone: a pick pays nothing for it.
            if matches!(filed, Ok(None)) && world.spec().awaits_document() {
                counts.awaited.fetch_add(1, Ordering::Relaxed);
                return Err(Reason::Unspecified);
            }
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
                            beside: match beside {
                                FiledBeside::Alone => wire::Beside::Alone,
                                // On the wire a pair's decode side differs
                                // from its prefill's by the side and the
                                // key, at its issue.
                                FiledBeside::Decode(decode) => wire::Beside::Decode(decode.session),
                                FiledBeside::Issue(issue) => wire::Beside::Issue(issue),
                            },
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
                let gateway = taken.gateway;
                match beside {
                    FiledBeside::Alone => give_up(store, world, gateway, &lease, None, ended),
                    FiledBeside::Decode(decode) => {
                        give_up(store, world, gateway, &lease, Some(&decode), ended);
                    }
                    FiledBeside::Issue(issue) => {
                        take_back(store, world, gateway, &lease, issue, ended);
                    }
                }
            }
            Err(busy)
        };
        store.deciding(|world| {
            if world.samples_decisions() {
                self.shared.sampled(
                    (self.index, store.decision_slots()),
                    &mut self.sample_count,
                    world.spec().engine(),
                    &taken,
                    || order(world),
                )
            } else {
                order(world)
            }
        })
    }
}

impl<'a> DecisionCall<'a, Checked> {
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
            place: None,
        })
    }

    /// The request with its decider's place at the leases' files.
    fn placed(self, place: Option<&'a AtWork>) -> Self {
        Self { place, ..self }
    }

    /// The time since the request was taken at `later`, a read of the
    /// clock after that one. One difference of two instants judges the
    /// budget and a lease's dispatch span, where taking the request again
    /// would read its budget and build it whole.
    fn since(&self, later: Instant) -> Duration {
        later.saturating_duration_since(self.now)
    }

    /// Whether the budget is gone `since` the request was taken: what was
    /// left then, less that time, under the same rule.
    fn outlasted(&self, since: Duration) -> bool {
        self.left.saturating_sub(since) < Duration::from_micros(1)
    }
}

/// A decision as its deciding thread takes it from the engine, or the
/// reason its request is answered `Busy`. A panic the engine caught and
/// counted is written into `caught`: the request is answered as one the
/// decider caught. No other refusal reaches a single request or a pair at
/// its prefill: one that did would be no panic, and is counted as none.
///
/// Inline in each caller: a single request's decision stays the code it was
/// before a pair's took the same step.
#[inline(always)]
fn made<T>(decided: Result<T, DecisionRefused>, caught: &mut bool) -> Result<T, Reason> {
    match decided {
        Ok(decided) => Ok(decided),
        Err(DecisionRefused::Faulted) => {
            *caught = true;
            Err(Reason::Unspecified)
        }
        Err(
            refused @ (DecisionRefused::PairSide
            | DecisionRefused::Unleased
            | DecisionRefused::Decode(_)),
        ) => {
            debug_assert!(false, "a decision over worker lists refused: {refused:?}");
            Err(Reason::Unspecified)
        }
    }
}

/// The path a request takes to its decision.
#[derive(Clone, Copy)]
enum Entry {
    /// A single request.
    Single,
    /// A prefill frame: a pair's one request where its plan pairs, and a
    /// frame that stands alone where it pairs no side. An encode request
    /// that carries its items is read as one, and is its pair's one
    /// request under any plan.
    Prefill,
    /// The decode call that names this lease.
    Decode(LeaseId),
    /// A decode frame that names no lease: one that stands alone where
    /// its plan pairs no side, and one with no decision where it does.
    DecodeAlone,
}

/// What a deciding thread files for a pick: the store's answer for the
/// lease its answer carries first.
type Filed = Result<Option<Lease>, LeaseRefused>;

/// What a decision's lease was filed with, beside it: the thread's form of
/// [`wire::Beside`], with a pair's decode side's whole lease, which a
/// give-up tells of.
#[derive(Clone, Copy)]
enum FiledBeside {
    /// A single's lease, a pair's prefill side's whose decode comes later,
    /// and no lease at all: nothing.
    Alone,
    /// The decode side's lease of a pair's pick, issued with its prefill.
    Decode(Lease),
    /// The issue of a decode call's pick: an answer that does not leave
    /// takes this one issue back.
    Issue(DecodeIssue),
}

/// A prefill frame's decision and its leases. A frame that enters a pair
/// plan at its prefill is its pair's one request: decided for both sides
/// ([`frame_decision::decide_pair`]) and, for a pick, issued a lease a side
/// ([`frame_decision::issue_pair`]). The prefill side's lease is the one
/// filed, as a single's is, and the decode side's comes beside it where it
/// was issued. A decode side its frame asked for later is chosen and not
/// issued: none comes, and the answer carries the prefill's lease alone.
///
/// A prefill whose model's plan pairs no side stands alone
/// ([`stands_alone`]).
///
/// An encode request that carries its items is such a frame under any
/// plan: decided for its pair and for each item's encoder
/// ([`frame_decision::decide_encode`]) and issued as a pair is, with no
/// lease for an encoder. It never stands alone, so no frame with items is
/// decided without them: under a plan with no encode side the engine
/// declines it.
///
/// Out of line: a single's decision carries no part of a pair's.
#[inline(never)]
fn prefill(
    store: &MemoryStore,
    world: &StoreSnapshot,
    objects: &Objects<Members>,
    taken: &DecisionCall<'_, Checked>,
    term: u64,
    (frames, loads, own): (&mut FrameBuffers, &Arc<LoadSnapshot>, &Part),
    caught: &mut bool,
) -> Result<(Decision, Filed, FiledBeside), Reason> {
    let encode = taken.request.carries_items();
    if !encode
        && !world
            .spec()
            .engine()
            .pairs(taken.request.model(), Phase::Prefill)
    {
        return stands_alone(
            store,
            world,
            objects,
            taken,
            term,
            (frames, loads, own),
            caught,
        );
    }
    let decided = if encode {
        frame_decision::decide_encode(world, objects, taken, frames, loads, own)
    } else {
        frame_decision::decide_pair(world, objects, taken, frames, loads, own)
    };
    let (decision, booked) = made(decided, caught)?;
    let (filed, beside) =
        match frame_decision::issue_pair(store, world, taken, term, &decision, booked) {
            Ok(Some((prefill, Some(decode)))) => (Ok(Some(prefill)), FiledBeside::Decode(decode)),
            Ok(Some((prefill, None))) => (Ok(Some(prefill)), FiledBeside::Alone),
            Ok(None) => (Ok(None), FiledBeside::Alone),
            Err(refused) => (Err(refused), FiledBeside::Alone),
        };
    Ok((decision, filed, beside))
}

/// The decision and the lease of a frame that stands alone at a side's
/// phase: a prefill frame, or a decode frame that names no lease, whose
/// plan pairs no side of its phase. The binding of its own phase decides
/// it over that phase's list ([`frame_decision::decide_alone`]), and a
/// pick is filed one lease, as a single request's is
/// ([`frame_decision::issue`]).
fn stands_alone(
    store: &MemoryStore,
    world: &StoreSnapshot,
    objects: &Objects<Members>,
    taken: &DecisionCall<'_, Checked>,
    term: u64,
    (frames, loads, own): (&mut FrameBuffers, &Arc<LoadSnapshot>, &Part),
    caught: &mut bool,
) -> Result<(Decision, Filed, FiledBeside), Reason> {
    let decided = frame_decision::decide_alone(world, objects, taken, frames, loads, own);
    let (decision, booked) = made(decided, caught)?;
    let filed = frame_decision::issue(store, world, taken, term, &decision, booked);
    Ok((decision, filed, FiledBeside::Alone))
}

/// The decision of a decode frame that names no lease. Where the decode
/// phase's own lookup resolves to a plan that pairs it, the frame is a
/// pair's side with no lease to decide behind: it has no decision, is
/// answered `Busy` with no reason and counted with the sides, and its
/// gateway decides it locally. Under any other plan it stands alone
/// ([`stands_alone`]): a binding of one profile listed ahead of the pair's
/// for the decode phase so decides it.
///
/// Out of line: no other request carries a part of it.
#[inline(never)]
fn decode_alone(
    store: &MemoryStore,
    world: &StoreSnapshot,
    objects: &Objects<Members>,
    taken: &DecisionCall<'_, Checked>,
    term: u64,
    (frames, loads, own, counts): (&mut FrameBuffers, &Arc<LoadSnapshot>, &Part, &DecideStats),
    caught: &mut bool,
) -> Result<(Decision, Filed, FiledBeside), Reason> {
    if world
        .spec()
        .engine()
        .pairs(taken.request.model(), Phase::Decode)
    {
        counts.pair_sides.fetch_add(1, Ordering::Relaxed);
        return Err(Reason::Unspecified);
    }
    stands_alone(
        store,
        world,
        objects,
        taken,
        term,
        (frames, loads, own),
        caught,
    )
}

/// A decode call's decision and its side's issue, for the frame that names
/// the decode side of `lease` (`DecideRequest.decode_of`). The store's own
/// test of the lease comes first ([`MemoryStore::held_pair`]), so a call it
/// refuses runs no stage. A call whose frame names its worker where the
/// decode's side reads the number is refused next, as one whose decode
/// never stood chosen ([`frame_decision::names_decode`]): the decode its
/// request named stands, and no stage runs. Then the decode is decided
/// again behind the held prefill ([`frame_decision::decide_decode`]) and,
/// for a pick, issued under the lease's id
/// ([`frame_decision::issue_decode`]). The lease filed is the decode
/// side's, and its issue's count comes beside it. The issue tests the
/// lease again: one that moved under the decision is refused there, behind
/// the stages.
///
/// Out of line: no other request carries a part of it.
///
/// # Errors
///
/// The reason the call's answer is `Busy` under: one of its own for each
/// refusal of the store ([`busy_of`]), at either test, and the one of a
/// decode never chosen for a call that names its decode.
#[inline(never)]
fn decode_call(
    store: &MemoryStore,
    world: &StoreSnapshot,
    objects: &Objects<Members>,
    taken: &DecisionCall<'_, Checked>,
    lease: LeaseId,
    (frames, loads, own): (&mut FrameBuffers, &Arc<LoadSnapshot>, &Part),
    caught: &mut bool,
) -> Result<(Decision, Filed, FiledBeside), Reason> {
    let held = store.held_pair(taken.gateway, lease).map_err(busy_of)?;
    if frame_decision::names_decode(world.spec(), taken.request) {
        return Err(busy_of(DecodeRefused::NotChosen));
    }
    let decided = frame_decision::decide_decode(world, objects, taken, held, frames, loads, own);
    let (decision, booked) = made(decided, caught)?;
    let issued = frame_decision::issue_decode(store, world, taken, lease, &decision, booked);
    Ok(match issued.map_err(busy_of)? {
        Some((side, issue)) => (decision, Ok(Some(side)), FiledBeside::Issue(issue)),
        None => (decision, Ok(None), FiledBeside::Alone),
    })
}

/// The `Busy` a refused decode call leaves under: a reason of its own for
/// each refusal of the store, never a decision that sheds. No policy spoke:
/// the lease's own state refused the call, and the reason names which. A
/// call that names its decode leaves under the reason of a side never
/// chosen ([`decode_call`]): behind either its gateway sends the decode
/// the first pick named.
const fn busy_of(refused: DecodeRefused) -> Reason {
    match refused {
        DecodeRefused::NoLease => Reason::DecodeNoLease,
        DecodeRefused::NotChosen => Reason::DecodeNotChosen,
        DecodeRefused::PrefillNotHeld => Reason::DecodePrefillNotHeld,
        DecodeRefused::DecodeStands => Reason::DecodeStands,
    }
}

/// Takes back the issue of a decode call whose answer does not leave: the
/// side stands as the call found it and its booking is freed
/// ([`MemoryStore::unissue_decode`]), then the engine that heard the issue
/// is told the side's release, once: the engine of the lease's generation,
/// or what a later one kept of its stages ([`issuing`]). A side that moved
/// since was ended or issued anew by another hand, and nothing is told of
/// it here.
fn take_back(
    store: &MemoryStore,
    world: &StoreSnapshot,
    gateway: GatewayId,
    lease: &Lease,
    issue: DecodeIssue,
    reason: ReleaseReason,
) {
    if store.unissue_decode(gateway, lease.id, issue) {
        if let Some(engine) = issuing(world, lease) {
            engine.on_released(lease, reason);
        }
    }
}

/// Gives up the lease of a decision whose answer does not leave: the
/// store's release door first ([`crate::store::LeasesStore::released`],
/// which applies what is handed ahead of it), then the engine that heard
/// the issue, told once where the store held the lease: once a side of a
/// pair's that was issued, which is given up whole. A lease the store no
/// longer holds was ended by another hand, and nothing is told of it here.
/// A pair's lease that hand ended one side of still stands, and each of
/// its issued sides is told here: no gateway names a lease ahead of its
/// answer.
fn give_up(
    store: &MemoryStore,
    world: &StoreSnapshot,
    gateway: GatewayId,
    lease: &Lease,
    decode: Option<&Lease>,
    reason: ReleaseReason,
) {
    if crate::store::LeasesStore::released(store, gateway, lease.id, None, reason) {
        let engine = world.spec().engine();
        for side in std::iter::once(lease).chain(decode) {
            engine.on_released(side, reason);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/decision_slots/tests/mod.rs"]
pub(crate) mod tests;
