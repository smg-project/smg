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
//! 7. the fence again, and the budget at the same read, before sampling
//!    and handing the answer over: a decision whose budget ended while it was
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
//! A pick or shed past the last read can be sampled for Explain. Each
//! decider counts its own answers and takes its ring's lock only to keep
//! a sample. A reader takes the lock for one scan.
//!
//! A lease is what the next decisions read. The store counts the leases on
//! file by the worker each books, and a decision reads that count as each
//! row's requests in flight ([`crate::engine::RequestFacts::in_flight`]): beside
//! the rows where the store lent them, in the row where they were written
//! for the request. A decision is made over what the decisions before it
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
//! A pair is one request, one answer, one id and two sides. The request
//! that enters a pair plan at its prefill is decided once, by the engine's
//! pair entry over both pools of one publication
//! ([`crate::engine::store_backed_engine::pair_in`]), and its pick is issued a
//! lease a side under one id ([`issue`]): one record on file and one write,
//! each side booking its own worker's count and living by its own frames. Both
//! leases are credits: nothing reserves a decode room yet.
//!
//! One lease stands for a request its gateway asks again. A gateway whose
//! stream ended with a request unanswered sends it again and says so
//! (`DecideRequest.again`). The request's decider then ends the lease the
//! earlier ask was issued, where it still waits for a dispatch, as the
//! dispatch deadline would end it ([`MemoryStore::asked_again`]), and
//! decides the request as any other. The earlier lease is on that
//! decider's file: a model's requests go to one decider for the process's
//! life ([`Deciders::of`]). A request sent twice under one id without the
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
//! election to hand the lease over ([`Deciders::leave_through`]), once a
//! term whichever decider's it is: the next take brings a term whose ids
//! start again. The counter's end stands as the last guard, where a pick is
//! answered `Busy`: under the last term a replica can take no word is said.
//!
//! A gateway's wires are a fact before the profile, as its vetoes are: no
//! filter of the catalog reads them, and no profile picks a member its
//! gateway cannot dial. The candidates the wires take are counted, and a
//! decision whose pool has rows and whose gateway shares a wire with none
//! of them is shed under [`WIRES`](crate::engine::store_backed_engine::WIRES).
//! One whose gateway hints a subset of 0, a decision over no candidates, is
//! shed under [`SUBSET_HINT`](crate::engine::store_backed_engine::SUBSET_HINT).
//! A positive hint is read by no stage yet.
//!
//! What this build has no decision for is answered `Busy` with no reason,
//! and its gateway decides it locally at once: a shed is a policy's word
//! and reaches the client, and no policy spoke. Four arms, each counted
//! by its cause ([`Counts`]): a decision that panics, a request for a side
//! of a pair plan that is not the pair's one request (decode under either
//! pair plan, encode under the plan that has an encode side: a pair is
//! decided at its prefill, and its alternates ride that answer), a
//! request that carries its prompt as text
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
        mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError},
        Arc, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

use super::{
    election::ElectionObservation,
    lifecycle,
    wire::{self, Checked, Lists},
};
use crate::{
    engine::{
        decision::Decision,
        decision_slot::DecisionSlot,
        framework::{release, Engine},
        store_backed_engine::{
            decide_in, decider_of, decode_in, entry, issue, issue_decode, issue_pair, issuing,
            pair_in, past_mark, summed, Buffers, DecisionCall, DecisionRecord, Entry, Part,
            RequestForm, SampleCount, Samples, StorePins,
        },
        view::fingerprint,
    },
    service::{
        elector::{term_said, Fence},
        proto::{self, busy::Reason},
    },
    stages::lifecycle::Lifecycle,
    store::{
        leases::DISPATCH_TTL, AtWork, DecodeIssue, DecodeRefused, GatewayId, LeaseRefused,
        MemoryStore, StoreSnapshot, Watch,
    },
    tracking::lease::{Lease, LeaseId, ReleaseReason},
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
    pub(crate) busy: [AtomicU64; 9],
    /// Decisions that panicked.
    pub(crate) faults: AtomicU64,
    /// Requests for a side of a pair plan that is not the pair's one
    /// request: decode under either pair plan, encode under the plan that
    /// has an encode side.
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

impl Counts {
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
    /// The word ([`Deciders::leave_through`]): none until the election runs.
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
            samples.push(answer.0, binding, taken, &answer.1, None, [None; 2]);
        }
        Ok(answer)
    }
}

/// What every decider reads.
#[derive(Clone)]
pub(crate) struct Reads {
    pub(crate) store: Arc<MemoryStore>,
    /// The controller's one fence.
    pub(crate) fence: Arc<Fence>,
    /// The election's last sight of the lease, for the term a `Busy`
    /// carries.
    pub(crate) standing: watch::Receiver<ElectionObservation>,
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

impl Deciders {
    /// Starts `deciders` threads, each behind a queue of `depth` requests:
    /// at least one thread and at most `DecisionSlot::COUNT`, and a queue of at
    /// least one. Refused over a store whose files were laid for another
    /// count of deciders.
    pub(crate) fn start(deciders: usize, depth: usize, reads: &Reads) -> io::Result<Self> {
        // A decider issues into a file of its own: the store was made for
        // this count of deciders, or its one file would have two writers.
        let deciders = deciders.clamp(1, DecisionSlot::COUNT);
        if reads.store.deciders() != deciders {
            let files = "the store's files of leases are not one a decider";
            return Err(io::Error::new(io::ErrorKind::InvalidInput, files));
        }
        lifecycle::tells(&reads.store);
        let told = Arc::new(watch::Sender::new(false));
        let shared = Arc::new(Shared::default());
        let queues = (0..deciders)
            .map(|index| {
                let (queue, jobs) = sync_channel(depth.max(1));
                let decider = Deciding {
                    reads: reads.clone(),
                    buffers: Buffers::default(),
                    sample_count: SampleCount::default(),
                    faulted: false,
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
    pub(crate) fn counts(&self) -> &Arc<Counts> {
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
    buffers: Buffers<Lists>,
    sample_count: SampleCount,
    /// The last decision panicked: a streak of them writes one line.
    faulted: bool,
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
    /// built ([`MemoryStore::with_deciders`]), before a thread shares it.
    fn wait(&mut self, jobs: &Receiver<Job>) -> Option<Job> {
        if self.reads.store.deciders() == 1 {
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
            let within = (found != Watch::Ended).then_some(period);
            match receive(within) {
                Ok(job) => return Some(job),
                Err(RecvTimeoutError::Disconnected) => return None,
                Err(RecvTimeoutError::Timeout) => {
                    found = self.look();
                    period = match found {
                        Watch::Fed => LOOK_FLOOR,
                        Watch::Quiet | Watch::Ended => (2 * period).min(LOOK_CAP),
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
    fn look(&mut self) -> Watch {
        self.working.as_mut().map_or(Watch::Ended, AtWork::look)
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
        let hand = lifecycle::open(&self.reads.store, gateway);
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
    /// answered `Busy` ([`Counts::unleased`]).
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
            fence,
            clock,
            counts,
            ..
        } = &self.reads;
        let now = clock();
        let judged = self.judged(request, gateway, received, now);
        // What the sessions handed is applied where the turn is free,
        // whatever was judged: before the rows' counts are read, and by a
        // request refused above too, so a hand at a full ring waits for
        // the first request of this decider that finds the turn free.
        if let Some(at_work) = &self.working {
            at_work.apply();
        }
        let (term, taken) = judged?;
        // The store's pins, lent to the decision: its engine asks them for
        // a key its own table holds no pin of, as after an apply of a new
        // generation.
        let pins = StorePins { store, now };
        let taken = taken.lent(&pins).placed(self.working.as_ref());
        let (buffers, own) = (&mut self.buffers, counts.part(self.index));
        // One load of the world for the decision and for its lease: the
        // lease books the count that load laid beside its target's row,
        // and the engine told of it is the one that decided.
        let mut order = |world: &StoreSnapshot| {
            // The lease, for a pick: filed under the first read's term and
            // at its instant, and the engine told, before the read that
            // lets the answer leave. That read judges the store's write
            // and a stage's call with the build before them. A pair's
            // pick is filed its decode side's lease beside it.
            let mut decode = None;
            let mut undo = None;
            let (decision, filed) = match entry(world, &taken) {
                Entry::Single => {
                    let (decision, booking) = decide_in(world, &taken, buffers, own);
                    let named = &buffers.named;
                    let engine = world.spec().engine();
                    let filed = booking
                        .map(|booking| {
                            issue(store, world, &taken, term, booking, named, |lease| {
                                engine.on_decision_issued(lease, &decision);
                            })
                        })
                        .transpose();
                    (decision, filed)
                }
                Entry::Pair => pair(store, world, &taken, term, buffers, own, &mut decode),
                Entry::Side => {
                    // A side has one decision of its own: the decode of a
                    // lease its request names. It is read here alone, so a
                    // single and a pair pay nothing for it.
                    let Some(lease) = taken.request.decode_of() else {
                        counts.pair_sides.fetch_add(1, Ordering::Relaxed);
                        return Err(Reason::Unspecified);
                    };
                    match decode_side(store, world, &taken, lease, buffers, own, &mut undo) {
                        Ok(decided) => decided,
                        Err(refused) => return Err(busy_of(refused)),
                    }
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
                            beside: match (decode, undo) {
                                (Some(decode), _) => {
                                    // At its issue a pair's decode side
                                    // differs from its prefill's by the
                                    // side and the key.
                                    debug_assert_eq!(
                                        (decode.tokens, decode.target_index),
                                        (lease.tokens, lease.target_index)
                                    );
                                    wire::Beside::Decode(decode.session)
                                }
                                (None, Some(issue)) => wire::Beside::Issue(issue),
                                (None, None) => wire::Beside::Alone,
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
                match undo {
                    Some(issue) => take_back(store, world, taken.gateway, &lease, issue, ended),
                    None => give_up(store, world, taken.gateway, &lease, decode.as_ref(), ended),
                }
            }
            Err(busy)
        };
        store.deciding(|world| {
            if world.samples_decisions() {
                self.shared.sampled(
                    (self.index, store.deciders()),
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

/// A pair's decision and its leases, for the one request that enters a
/// pair plan at its prefill ([`Entry::Pair`]): decided for both sides
/// ([`pair_in`]) and, for a pick, issued a lease a side
/// ([`issue_pair`]). The prefill side's lease is the one filed, as a
/// single's is, and the decode side's is laid in `decode`. A decode side
/// its request asked for later is chosen and not issued: none is laid, and
/// the answer carries the prefill's lease alone. Out of line: a single's
/// decision carries no part of a pair's.
#[inline(never)]
fn pair(
    store: &MemoryStore,
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, Checked>,
    term: u64,
    buffers: &mut Buffers<<Checked as RequestForm>::Lists>,
    own: &Part,
    decode: &mut Option<Lease>,
) -> (Decision, Result<Option<Lease>, LeaseRefused>) {
    let (decision, booking) = pair_in(world, taken, buffers, own);
    let filed = booking
        .map(|booking| issue_pair(store, world, taken, term, &decision, booking))
        .transpose();
    let later = taken.request.decode_later();
    let filed = filed.map(|leases| {
        leases.map(|(prefill, side)| {
            *decode = (!later).then_some(side);
            prefill
        })
    });
    (decision, filed)
}

/// A decode call's decision and its side's issue, for the one request that
/// names the decode side of `lease` (`DecideRequest.decode_of`): decided over
/// the held prefill's partners ([`decode_in`]) and, for a pick, issued
/// under the lease's id ([`issue_decode`]). The lease filed is the decode
/// side's, and its issue's count is laid in `undo`. Out of line: no other
/// request carries a part of it.
///
/// # Errors
///
/// [`DecodeRefused`]: why the lease takes no decode side. Nothing is filed.
#[inline(never)]
fn decode_side(
    store: &MemoryStore,
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, Checked>,
    lease: LeaseId,
    buffers: &mut Buffers<<Checked as RequestForm>::Lists>,
    own: &Part,
    undo: &mut Option<DecodeIssue>,
) -> Result<(Decision, Result<Option<Lease>, LeaseRefused>), DecodeRefused> {
    let (decision, booking) = decode_in(world, store, taken, lease, buffers, own)?;
    let filed = booking
        .map(|booking| issue_decode(store, world, taken, lease, &decision, booking))
        .transpose()?;
    let lease = filed.map(|(lease, issue)| {
        *undo = Some(issue);
        lease
    });
    Ok((decision, Ok(lease)))
}

/// The `Busy` a refused decode call leaves under: a reason of its own for
/// each refusal of the store, never a decision that sheds. A gateway that
/// read a shed would refuse a request whose prefill is running.
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
/// is told the side's release, once: the engine of the lease's generation
/// ([`issuing`]), where it still stands. A side that moved since was ended
/// or issued anew by another hand, and nothing is told of it here.
fn take_back(
    store: &MemoryStore,
    world: &StoreSnapshot,
    gateway: GatewayId,
    lease: &Lease,
    issue: DecodeIssue,
    reason: ReleaseReason,
) {
    if store.unissue_decode(gateway, lease.id, issue) {
        if let Some(engine) = issuing(world.spec(), lease) {
            engine.on_released(lease, reason);
        }
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
            pins: None,
            place: None,
        })
    }

    /// The request with the store's sessions lent to its decision.
    fn lent(self, pins: &'a StorePins<'a>) -> Self {
        Self {
            pins: Some(pins),
            ..self
        }
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

/// Gives up the lease of a decision whose answer does not leave: the
/// store's release door first ([`crate::store::LeasesStore::released`],
/// which applies what is handed ahead of it), then the engine that heard
/// the issue, told once where the store held the lease: once a side of a
/// pair's, which is given up whole. A lease the store no longer holds was
/// ended by another hand, and nothing is told of it here.
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
        decode_side, watched, Buffers, Clock, Counts, Deciders, Deciding, DecisionCall, Job, Part,
        Reads, Refused, LOOK_CAP, LOOK_FLOOR, PARTS,
    };
    use crate::{
        document::spec::{self, AppliedSpec, PolicySpec},
        engine::{
            decision::{Decision, Overlap, OverlapSource, Pick, Targets},
            decision_slot::DecisionSlot,
            framework::{DecisionScope, Filter, Picker, Scratch, TieBand},
            store_backed_engine::{
                self, cut_reach, keeps_a_row, narrow, pair_in, tests::pair_counts, Request,
                RequestForm, StoreBackedEngine, TERM_MARK,
            },
            view::{
                self, fingerprint, AffinityFacts, Candidate, Constraints, PoolFacts, RequestFacts,
                Role, WireModes, WorkerId,
            },
        },
        policies::{power_of_two::PowerOfTwoMetric, random::Random},
        service::{
            controller::{
                election::ElectionObservation,
                gateways::{
                    tests::{GRACE, PATIENT},
                    Gateways, Vetoed,
                },
                lifecycle,
                wire::{self, Checked},
            },
            elector::{Elector, Tick, TERM_CEILING},
            proto::{self, busy::Reason, decide_answer::Answer, decision::Outcome},
        },
        stages::{lifecycle::Lifecycle, pickers::Best, registry::StageRegistry},
        store::{
            leases::DISPATCH_TTL, Ask, Built, DecodeIssue, Dispatch, GatewayId, HealthSink,
            LeaseState, LeasesStore, MemoryStore, Said, StoreSnapshot, ViewBuffers, WorkerFacts,
            WorkerStatus, WorkersStore,
        },
        tracking::lease::{Lease, LeaseId, ReleaseReason, Side},
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

    /// The door's hand for a decider of one form: a case's function of the
    /// view decides alike over rows the store named and over any other.
    fn hand<A: RequestForm, R>(
        world: &StoreSnapshot,
        taken: &DecisionCall<'_, A>,
        buffers: &mut Buffers<A::Lists>,
        undialed: &AtomicU64,
        decide: impl FnOnce(
            &RequestFacts<'_>,
            &PoolFacts<'_>,
            &[Candidate],
            &Constraints<'_>,
            &mut Scratch,
        ) -> R,
    ) -> (R, Option<Built>) {
        store_backed_engine::hand(world, taken, buffers, undialed, decide)
    }

    /// Waits until what a handed frame changes shows, [`PATIENCE`] at
    /// most. The frame's file applies it by its decider, as that leaves
    /// its work or at its next look, or by the hand: a case that reads at
    /// once reads ahead of a decider still at work, or of its look. A
    /// case waits by it too for what a decider shows behind its answer:
    /// its watch, or that it left its work.
    pub(crate) async fn settled(mut shows: impl FnMut() -> bool) {
        let since = Instant::now();
        while !shows() {
            assert!(
                since.elapsed() < PATIENCE,
                "what the case waits for shows within a decider's look"
            );
            tokio::task::yield_now().await;
        }
    }

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
        standing: watch::Sender<ElectionObservation>,
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

    /// The `random` preset as generation 1 with the case's picker, laid
    /// under `name`, in its profile's pick.
    fn picking(
        name: &'static str,
        picker: impl Fn() -> Box<dyn Picker> + Send + Sync + 'static,
    ) -> AppliedSpec {
        let mut catalog = StageRegistry::builtin();
        catalog.picker(name, picker).expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = name.into();
        let digest = document.digest().expect("the document digests");
        AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the catalog holds the stage")
    }

    /// The member `name`, serving `model` in `role` and stating `wires`,
    /// ready: written again where it stood, with its status as it was.
    pub(crate) fn ready(
        store: &MemoryStore,
        name: &str,
        model: &str,
        role: Role,
        wires: WireModes,
    ) -> WorkerId {
        serving(store, name, &[model], role, wires)
    }

    /// [`ready`] for a member that serves each of `models`.
    fn serving(
        store: &MemoryStore,
        name: &str,
        models: &[&str],
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
            models: models.iter().map(|&model| model.to_owned()).collect(),
            dp_size: 1,
            max_running: Some(4),
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        store.publish_now();
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
        let store = Arc::new(MemoryStore::new().with_deciders(deciders));
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
        let (standing, stood) = watch::channel(ElectionObservation::default());
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

    /// A decider of the case's own over the bed's reads, for the case's
    /// thread. It is not at work until its case seats it.
    fn deciding(bed: &Bed) -> Deciding {
        Deciding {
            reads: bed.reads.clone(),
            buffers: Buffers::default(),
            sample_count: Default::default(),
            faulted: false,
            working: None,
            name: None,
            index: 0,
            marked: None,
            shared: Arc::default(),
        }
    }

    /// A decider issues into a file of leases of its own: the deciders do
    /// not start over a store made for another count of them.
    #[tokio::test]
    async fn the_deciders_do_not_start_over_a_store_made_for_another_count() {
        let bed = leading([]);
        assert_eq!(bed.store.deciders(), 2);
        let refused = Deciders::start(3, 8, &bed.reads).err();
        let refused = refused.expect("two files of leases, three deciders");
        assert_eq!(refused.kind(), io::ErrorKind::InvalidInput);
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
            again: false,
            ..Default::default()
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
                    signal_cells: false,
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

    /// Same facts, same answer: over two stores laid alike, the served
    /// path and the door in process pick the same targets for a request,
    /// and file its lease under the same id on the same first target.
    #[tokio::test]
    async fn the_door_and_the_served_path_decide_alike() {
        let mut bed = bed(Some(preset("random", 1)), true, (1, 8), []);
        let beside = Arc::new(MemoryStore::new().with_deciders(1));
        fleet(&beside);
        let applied = beside.apply(preset("random", 1));
        applied.expect("above the unbound start");
        let door = StoreBackedEngine::new(Arc::clone(&beside));
        let first_on_file = |store: &MemoryStore, lease: u64| {
            let record = LeasesStore::get(store, GATEWAY, LeaseId(lease));
            record.map(|record| record.target)
        };
        for (id, seed) in (1..=9).zip(100..) {
            let served = bed.ask(frame(id, seed)).await;
            let asked = Request {
                id: &[id],
                facts: RequestFacts {
                    model: MODEL,
                    seed,
                    budget_micros: u32::try_from(BUDGET.as_micros()).expect("five seconds"),
                    ..RequestFacts::default()
                },
                constraints: Constraints::default(),
            };
            let decided = door.decide(GATEWAY, &asked, bed.script.taken);
            let decided = decided.expect("the door decides");
            let pick = decided.decision.pick().expect("a pick over the fleet");
            let Outcome::Pick(sent) = outcome(&served) else {
                panic!("a pick: {served:?}");
            };
            let sent = sent.targets.iter().map(|target| WorkerId(target.id));
            let sent: Vec<_> = sent.collect();
            let made: Vec<_> = pick.targets.iter().map(|target| target.id).collect();
            assert_eq!(sent, made, "request {id}: the same targets, in order");
            let lease = decided.lease.expect("a pick's lease").lease.id.0;
            assert_eq!(lease_of(&served).id, lease, "request {id}: the same id");
            let booked = first_on_file(&beside, lease);
            assert_eq!(first_on_file(&bed.store, lease), booked, "request {id}");
            assert_eq!(booked, made.first().copied(), "request {id}");
        }
    }

    /// A served decision writes the record of sent prefixes at its lease's
    /// issue, as the door does in process: on workers that publish no cache
    /// events a request lands where its prefix was sent.
    #[tokio::test]
    async fn a_served_request_lands_where_its_prefix_was_sent() {
        let mut bed = bed(Some(preset("cache_aware", 1)), true, (1, 8), []);
        let asked = |id: u8, seed: u64| {
            let mut frame = frame(id, seed);
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.block_hashes = (1..=8).collect();
            facts.prompt_tokens = 8 * 16;
            frame
        };
        let sent = picked(&bed.ask(asked(1, 100)).await);
        // Each lease on file books its worker: load alone would send the
        // next request to another.
        for (id, seed) in (2..=4).zip(101..) {
            let returned = picked(&bed.ask(asked(id, seed)).await);
            assert_eq!(returned, sent, "request {id}");
        }
    }

    /// An entry of the record of sent prefixes is a prediction made at the
    /// decision. A served decision whose last read of the clock finds its
    /// budget gone leaves as `Busy` and gives its lease up. Its entry
    /// stays, and the next request with that prefix follows it.
    #[tokio::test]
    async fn a_decision_that_leaves_as_busy_keeps_its_prefix_for_the_next_request() {
        let asked = |id: u8, seed: u64| {
            let mut frame = frame(id, seed);
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.block_hashes = (1..=8).collect();
            facts.prompt_tokens = 8 * 16;
            frame
        };
        let fresh = |steps: Vec<Step>| bed(Some(preset("cache_aware", 1)), true, (1, 8), steps);
        // Where the late request lands, and a second request that lands
        // elsewhere as the first of its prefix, each on a bed of its own.
        let late = picked(&fresh(Vec::new()).ask(asked(1, 100)).await);
        let mut elsewhere = None;
        for seed in 101..=132 {
            if picked(&fresh(Vec::new()).ask(asked(2, seed)).await) != late {
                elsewhere = Some(seed);
                break;
            }
        }
        let seed = elsewhere.expect("a seed that lands on another worker");
        // The same two requests on one bed. The first decision's last read
        // of the clock shows its budget gone, and the second is taken at
        // the first's instant again.
        let steps = [Duration::ZERO, BUDGET, Duration::ZERO].map(Step::At);
        let mut bed = fresh(steps.into());
        let answer = bed.ask(asked(1, 100)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(bed.store.live(GATEWAY), 0);
        let next = picked(&bed.ask(asked(2, seed)).await);
        assert_eq!(next, late, "the late decision's entry is followed");
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

    /// A decision reads its rows behind what was handed while its decider
    /// worked: a release handed while the decider is held at a request is
    /// applied by that request ahead of its reads. Under the fewest
    /// requests the request then picks as the first one did under its
    /// seed: that worker stands at no request again.
    #[tokio::test]
    async fn a_decision_reads_its_rows_behind_what_was_handed_ahead_of_it() {
        let mut document = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
            .expect("a form of this build's preset");
        document.generation = 1;
        let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
        let first = bed.ask(frame(1, 7)).await;
        // The second request is held at its read of the clock: its decider
        // is at work, and what is handed is left to it.
        let (reached, held) = std_mpsc::sync_channel(1);
        let (go, word) = std_mpsc::sync_channel(1);
        let hold = Step::Hold { reached, go: word };
        bed.script.steps.lock().push_back(hold);
        assert_eq!(bed.submit(frame(2, 7)), None);
        held.recv_timeout(PATIENCE)
            .expect("the decider reads the clock");
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        let hand = lifecycle::open(&bed.store, GATEWAY);
        bed.store.hand(&hand, LeaseId(lease_of(&first).id), gone);
        let live = LeasesStore::live(&*bed.store, GATEWAY);
        assert_eq!(live, 1, "the hand leaves the release to the decider");
        go.send(()).expect("the decider holds the other end");
        let second = bed.answer().await;
        assert_eq!(picked(&second), picked(&first), "no request stands there");
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
            (picked(&answer), 1, LeaseState::ISSUED, fingerprint(&[7]))
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

    /// A frame's client key places its request on the ring: one client's
    /// requests take one worker whatever their seeds, and clients spread
    /// over the workers.
    #[tokio::test]
    async fn a_frames_client_key_places_its_request_on_the_ring() {
        let mut bed = bed(Some(preset("consistent_hashing", 1)), true, (2, 8), []);
        let from = |id: u8, client: u64| {
            let mut frame = frame(id, u64::from(id));
            frame
                .facts
                .as_mut()
                .expect("the frame has facts")
                .client_key = client;
            frame
        };
        let mut of_one = Vec::new();
        for id in 1..=12u8 {
            of_one.push(picked(&bed.ask(from(id, 0x9e37_79b9_7f4a_7c15)).await));
        }
        assert!(
            of_one.windows(2).all(|pair| pair[0] == pair[1]),
            "{of_one:?}"
        );
        let mut of_many = Vec::new();
        for id in 13..=28u8 {
            let client = u64::from(id).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            of_many.push(picked(&bed.ask(from(id, client)).await));
        }
        assert!(
            of_many.windows(2).any(|pair| pair[0] != pair[1]),
            "{of_many:?}"
        );
    }

    /// Requests that state no prompt size spread over the members: each
    /// dispatch is its worker's credit at the load stage's mean until that
    /// worker's next report, and each lease states no size.
    #[tokio::test]
    async fn a_dispatched_request_that_stated_no_size_is_priced_until_its_workers_next_report() {
        use crate::{
            engine::view::LoadRow,
            store::{LoadSink, LoadSource, PUBLISH_WINDOW},
        };
        let mut bed = bed(Some(preset("least_load", 1)), true, (2, 8), []);
        // Each member reports at the take, at one rate: the first holds no
        // work and the other two a quarter of a mean prompt each. The
        // window the first record opens ends, and one world shows the
        // three.
        let (sink, at): (&dyn LoadSink, _) = (bed.store.as_ref(), bed.script.taken);
        for (turn, &id) in bed.workers.iter().enumerate() {
            let row = LoadRow {
                waiting_uncached_tokens: if turn == 0 { 0 } else { 256 },
                live_rate: 1_000,
                ..LoadRow::default()
            };
            sink.record(id, 0, row, at, LoadSource::Stream);
        }
        sink.publish(at + PUBLISH_WINDOW);
        // Three requests that state no prompt size, each dispatched before
        // the next is decided. A dispatch is its worker's credit until that
        // worker's next report, and the load stage credits one of no stated
        // size at its mean: at no tokens it would cost nothing there, and
        // each request would read the first member as free.
        let mut named = Vec::new();
        for id in 1..=3u8 {
            let answer = bed.ask(frame(id, u64::from(id))).await;
            assert_eq!(
                lease_of(&answer).tokens,
                0,
                "the lease states what the request did"
            );
            named.push(picked(&answer));
            let frame = proto::Dispatched {
                lease_id: lease_of(&answer).id,
                target_index: 0,
                side: proto::Side::Single.into(),
                since_decision_micros: 250,
            };
            let hand = lifecycle::open(&bed.store, GATEWAY);
            lifecycle::dispatched(&bed.store, &hand, &frame, bed.script.taken)
                .expect("the frame reads");
        }
        assert_eq!(named[0], bed.workers[0], "the member with no work");
        let members: BTreeSet<WorkerId> = named.iter().copied().collect();
        assert_eq!(members.len(), 3, "each member took one request: {named:?}");
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
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("told", move || Box::new(Told(Arc::clone(&kept))))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = generation;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "told".into();
        let digest = document.digest().expect("the document digests");
        AppliedSpec::validate_digested(&document, digest, &catalog, &[])
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
        let mut catalog = StageRegistry::builtin();
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
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
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
    /// held as a decision's fault is, counted apart from the decisions',
    /// and the decider stands for its next request.
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
        let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
        while faults() == 0 && !*ended.borrow() {
            assert!(
                waited.elapsed() < PATIENCE,
                "the decider never left its work"
            );
            thread::sleep(Duration::from_millis(1));
        }
        assert!(!*ended.borrow(), "the decider's thread unwound");
        // The call's fault is its own count: no request went unanswered.
        assert_eq!(faults(), 1);
        assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
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
    /// full, outside any decision. A stage's call that panics there is
    /// held as a decision's fault is, counted apart from the decisions',
    /// and the decider stands for its next request.
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
        let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
        while faults() == 0 && !*ended.borrow() {
            assert!(
                waited.elapsed() < PATIENCE,
                "the decider never gave the lease up"
            );
            thread::sleep(Duration::from_millis(1));
        }
        assert!(!*ended.borrow(), "the decider's thread unwound");
        // The call's fault is its own count: no decision panicked.
        assert_eq!(faults(), 1);
        assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
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

    /// A give-up whose hand applies in place and meets another lease's
    /// panicking call there is through: the call is held where the engine
    /// is told, and the hand writes its frame behind it. The unsent
    /// answer's lease leaves the file at the take behind its hand.
    #[tokio::test]
    async fn a_give_up_behind_another_leases_panic_at_a_full_ring_is_still_handed() {
        let (mut bed, told, at_issue, word) = waits(true);
        let ended = bed.deciders.ended();
        // The first lease: its answer is taken and the lease stays on file.
        assert_eq!(bed.submit(frame(1, 1)), None);
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        word.send(()).expect("the stage waits for the case");
        picked(&bed.answer().await);
        assert_eq!(bed.store.live(GATEWAY), 1);
        // The second: a stream that ended, so its answer finds the queue
        // closed.
        let (sender, answers) = mpsc::channel(1);
        drop(answers);
        let job = Job {
            request: checked(frame(2, 1)),
            gateway: GATEWAY,
            received: bed.script.taken,
            answers: sender,
        };
        assert!(bed.deciders.submit(job).is_none());
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        // While the stage's call runs, the gateway gives the first lease up
        // and fills the ring behind that frame.
        let first = LeaseId(1 << 32 | 1);
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = || Said::Released(None, ReleaseReason::ClientGone);
        bed.store.hand(&hand, first, gone());
        for _ in 1..MemoryStore::HANDOVER {
            bed.store.hand(&hand, LeaseId(0), gone());
        }
        word.send(()).expect("the stage waits for the case");
        // Two held calls: the first lease's release, where the give-up's
        // hand applied in place, and the second lease's own, where the
        // decider takes its turn behind the hand and applies its frame.
        let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
        while faults() < 2 && !*ended.borrow() {
            assert!(
                waited.elapsed() < PATIENCE,
                "the give-up was not handed behind the call that panicked"
            );
            thread::sleep(Duration::from_millis(1));
        }
        assert!(!*ended.borrow(), "the decider's thread unwound");
        // No decision panicked: the calls' faults are their own count.
        assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
        assert_eq!(bed.store.live(GATEWAY), 0);
        assert_eq!(*told.lock(), [(first.0, None), (first.0 + 1, None)]);
    }

    /// The lease of an answer its stream's queue refuses is off its file
    /// when its decider's give-up returns. The decider is at work there,
    /// so the hand leaves the frame to it, and its look ahead of a
    /// decision passes a held turn: the give-up takes the turn itself.
    /// The decider is the case's own, at work on the case's thread. A
    /// second thread, which may not apply at that file, reads it.
    #[test]
    fn a_refused_answers_lease_is_off_its_file_when_its_give_up_returns() {
        let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
        let mut decider = deciding(&bed);
        decider.working = Some(bed.store.at_work());
        let answer = decider.answer(checked(frame(1, 1)), GATEWAY, bed.script.taken);
        let leased = matches!(answer, wire::Decided::Decision { lease: Some(_), .. });
        assert!(leased, "the decision carries a lease");
        let live = || {
            thread::scope(|scope| {
                let read = scope.spawn(|| bed.store.live(GATEWAY));
                read.join().expect("the reading thread")
            })
        };
        assert_eq!(live(), 1);
        // Its stream refuses the answer.
        decider.unsent(GATEWAY, answer);
        assert_eq!(live(), 0, "the release waited for the decider's next take");
        assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
    }

    /// A request refused before anything is built applies what is handed
    /// too, so a hand at a full ring waits for one request of the file's
    /// decider, whatever its answer. The decider here is a former leader
    /// behind its closed fence whose gateway still asks it: it refuses each
    /// request and never rests, and the hand is through all the same.
    #[tokio::test]
    async fn a_refused_request_applies_what_a_full_ring_holds() {
        let mut bed = bed(Some(preset("random", 1)), true, (1, 8), []);
        bed.ask(frame(1, 1)).await;
        let hold = || {
            let (reached, held) = std_mpsc::sync_channel(1);
            let (go, word) = std_mpsc::sync_channel(1);
            (Step::Hold { reached, go: word }, held, go)
        };
        // The second request is held at its read of the clock: its decider
        // is at work, and what is handed is left to it.
        let (first, held, go) = hold();
        bed.script.steps.lock().push_back(first);
        assert_eq!(bed.submit(frame(2, 1)), None);
        held.recv_timeout(PATIENCE)
            .expect("the decider reads the clock");
        // The ring fills, and one hand more waits at it.
        let gone = || Said::Released(None, ReleaseReason::ClientGone);
        let hand = lifecycle::open(&bed.store, GATEWAY);
        for _ in 0..MemoryStore::HANDOVER {
            bed.store.hand(&hand, LeaseId(0), gone());
        }
        let (through, handed) = std_mpsc::channel();
        let store = Arc::clone(&bed.store);
        let waiting = thread::spawn(move || {
            store.hand(&lifecycle::open(&store, GATEWAY), LeaseId(0), gone());
            through.send(()).expect("the case waits for the hand");
        });
        // The lead is an hour gone at every read from here on. Two more
        // requests wait behind the held one, and the last is held at its
        // read of the clock: by then the decider has refused two requests
        // and has not rested.
        let hour = Duration::from_secs(3600);
        let (last, held, last_go) = hold();
        *bed.script.shown.lock() = hour;
        bed.script.steps.lock().extend([Step::At(hour), last]);
        assert_eq!(bed.submit(frame(3, 1)), None);
        assert_eq!(bed.submit(frame(4, 1)), None);
        go.send(()).expect("the decider holds the other end");
        held.recv_timeout(PATIENCE)
            .expect("the decider reads the clock for the last request");
        let handed = handed.recv_timeout(PATIENCE);
        last_go.send(()).expect("the decider holds the other end");
        handed.expect("a refused request applies what the ring holds");
        waiting.join().expect("the hand's thread");
        for id in 2..=4 {
            let answer = bed.answer().await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice()),
                (Reason::NotLeader, &[id][..])
            );
        }
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

    /// `frame` as its gateway sends it again, holding no answer to the
    /// first ask.
    fn again(mut frame: proto::DecideRequest) -> proto::DecideRequest {
        frame.again = true;
        frame
    }

    /// The requests the bed's deciders counted as asked again: those that
    /// found no lease waiting, and those whose earlier lease was ended.
    fn asked_again(bed: &Bed) -> [u64; 2] {
        let again = bed.counts.again.each_ref();
        again.map(|count| count.load(Ordering::Relaxed))
    }

    /// One lease stands for a request its gateway asks again. The lease
    /// the first ask was issued ends as abandoned where it waits, and the
    /// engine that issued it is told, ahead of the repeat's own lease. A
    /// lease of another request stands.
    #[tokio::test]
    async fn a_request_asked_again_ends_the_lease_its_first_ask_was_issued() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
        let id = |count: u64| 1 << 32 | count;
        for request in 1..=2 {
            bed.ask(frame(request, 1)).await;
        }
        // The answer to the first never arrived: its gateway asks again.
        let answer = bed.ask(again(frame(1, 1))).await;
        assert_eq!(lease_of(&answer).id, id(3));
        let abandoned = (id(1), Some(ReleaseReason::Abandoned));
        assert_eq!(
            *told.lock(),
            [(id(1), None), (id(2), None), abandoned, (id(3), None)]
        );
        let store: &dyn LeasesStore = bed.store.as_ref();
        assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
        assert_eq!(store.live(GATEWAY), 2, "another request's and the repeat's");
        assert_eq!(asked_again(&bed), [0, 1]);
    }

    /// A request sent twice under one id without the word is two requests:
    /// each is issued a lease and neither ends the other's.
    #[tokio::test]
    async fn a_request_sent_twice_without_the_word_is_issued_two_leases() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
        let id = |count: u64| 1 << 32 | count;
        for _ in 0..2 {
            bed.ask(frame(1, 1)).await;
        }
        assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
        assert_eq!(bed.store.live(GATEWAY), 2);
        assert_eq!(asked_again(&bed), [0, 0]);
    }

    /// A repeat ends a lease that waits and no other: a lease its gateway
    /// named a dispatch for stands, and a request no decider took has
    /// none. Each such repeat is decided as a first ask.
    #[tokio::test]
    async fn a_request_asked_again_ends_no_lease_that_was_dispatched() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
        let id = |count: u64| 1 << 32 | count;
        bed.ask(frame(1, 1)).await;
        let store: &dyn LeasesStore = bed.store.as_ref();
        let first = LeaseId(id(1));
        assert!(store.dispatched(GATEWAY, first, Dispatch::default(), bed.script.taken));
        // The gateway holds the first answer and asks again all the same,
        // and it asks again for a request no decider ever took.
        for request in [1, 9] {
            bed.ask(again(frame(request, 1))).await;
        }
        assert_eq!(*told.lock(), [(id(1), None), (id(2), None), (id(3), None)]);
        assert_eq!(bed.store.live(GATEWAY), 3);
        assert_eq!(asked_again(&bed), [2, 0]);
    }

    /// The word reaches the asking gateway's own leases alone: a lease
    /// another gateway holds under the same id stands.
    #[tokio::test]
    async fn a_request_asked_again_ends_no_lease_of_another_gateway() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
        let id = |count: u64| 1 << 32 | count;
        bed.ask(frame(1, 1)).await;
        let repeat = again(frame(1, 1));
        assert_eq!(bed.submit_at(repeat, OTHER, Duration::ZERO), None);
        bed.answer().await;
        assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
        assert_eq!((bed.store.live(GATEWAY), bed.store.live(OTHER)), (1, 1));
        assert_eq!(asked_again(&bed), [1, 0]);
    }

    /// A repeat ends its earlier lease whatever becomes of its own
    /// decision: one answered `Busy` for a budget that is gone leaves no
    /// lease of its request on file.
    #[tokio::test]
    async fn a_request_asked_again_past_its_budget_ends_its_earlier_lease_all_the_same() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
        let id = |count: u64| 1 << 32 | count;
        bed.ask(frame(1, 1)).await;
        // The repeat's budget is gone at its decider's read of the clock,
        // well inside the first lease's dispatch span.
        let (budget, read) = (Duration::from_millis(1), Duration::from_millis(2));
        bed.script.steps.lock().push_back(Step::At(read));
        let answer = bed.ask(budgeted(again(frame(1, 1)), budget)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        let abandoned = (id(1), Some(ReleaseReason::Abandoned));
        assert_eq!(*told.lock(), [(id(1), None), abandoned]);
        assert_eq!(bed.store.live(GATEWAY), 0);
        assert_eq!(asked_again(&bed), [0, 1]);
    }

    /// A dispatch handed ahead of the repeat stands: what was handed to
    /// the file is applied before the repeat looks for a lease that waits.
    #[tokio::test]
    async fn a_dispatch_handed_ahead_of_a_request_asked_again_is_applied_first() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(hearing(&told, 1)), true, (1, 8), []);
        let id = |count: u64| 1 << 32 | count;
        bed.ask(frame(1, 1)).await;
        // The repeat's decider is held at its first read of the clock: it
        // has the job at hand, so a frame handed now is left to it.
        let (reached, held) = std_mpsc::sync_channel(1);
        let (go, word) = std_mpsc::sync_channel(1);
        let hold = Step::Hold { reached, go: word };
        bed.script.steps.lock().push_back(hold);
        assert_eq!(bed.submit(again(frame(1, 1))), None);
        held.recv_timeout(PATIENCE)
            .expect("the decider reads the clock");
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let dispatched = Said::Dispatched(Dispatch::default(), bed.script.taken);
        bed.store.hand(&hand, LeaseId(id(1)), dispatched);
        go.send(()).expect("the decider holds the other end");
        bed.answer().await;
        assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
        assert_eq!(
            bed.store.live(GATEWAY),
            2,
            "the dispatched lease and the repeat's"
        );
        assert_eq!(asked_again(&bed), [1, 0]);
    }

    /// A decider with a job at hand is at work at the leases' file: what a
    /// session hands meanwhile is left to it, and it applies it before its
    /// decision reads the rows' counts, ahead of its own issue. A frame
    /// handed while no decider has a job is applied all the same.
    #[tokio::test]
    async fn a_frame_handed_while_a_decider_decides_is_applied_by_that_decider() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&told);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("told", move || Box::new(Told(Arc::clone(&kept))))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "told".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
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
        settled(|| told.lock().len() >= 4).await;
        assert_eq!(told.lock()[3], released(2));
    }

    /// The `random` picker, keeping the decider its thread acts as at each
    /// call of a lease's issue and release.
    #[derive(Debug)]
    struct Named(Arc<Mutex<Vec<Option<DecisionSlot>>>>);

    impl Lifecycle for Named {
        fn on_decision_issued(&self, _: &Lease, _: &Decision) {
            self.0.lock().push(DecisionSlot::acting());
        }

        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            self.0.lock().push(DecisionSlot::acting());
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
    /// lease's event is told under the name of the decider whose file holds
    /// the lease, whichever thread applies the frame.
    #[tokio::test]
    async fn a_decision_and_its_leases_events_act_as_the_leases_decider() {
        let named = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&named);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("named", move || Box::new(Named(Arc::clone(&kept))))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "named".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the catalog holds the stage");
        let mut bed = bed(Some(spec), true, (1, 8), []);
        let answer = bed.ask(frame(1, 1)).await;
        // The case's thread names no decider: its hand applies the release,
        // or the decider does as it leaves its work.
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        bed.store.hand(&hand, LeaseId(lease_of(&answer).id), gone);
        settled(|| named.lock().len() >= 2).await;
        assert_eq!(*named.lock(), [Some(DecisionSlot::new(0)); 2]);
        assert_eq!(DecisionSlot::acting(), None);
    }

    /// The `random` picker, keeping the name of the thread each release of
    /// a lease is told on.
    #[derive(Debug)]
    struct Threads(Arc<Mutex<Vec<Option<String>>>>);

    impl Lifecycle for Threads {
        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            let name = thread::current().name().map(str::to_owned);
            self.0.lock().push(name);
        }
    }

    impl Picker for Threads {
        fn name(&self) -> &'static str {
            "threads"
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

    /// A decider that waits for a job with a lease on its file keeps its
    /// watch: a frame handed on the lull is left in the ring, and the
    /// decider's own thread applies it within its look. The hand's thread
    /// applies none. Once no lease stands on the file the watch ends, and
    /// the decider wakes for a job alone.
    #[tokio::test]
    async fn a_frame_handed_on_a_lull_is_applied_by_its_leases_decider() {
        let threads = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&threads);
        let spec = picking("threads", move || Box::new(Threads(Arc::clone(&kept))));
        let mut bed = bed(Some(spec), true, (2, 8), []);
        let taker = bed.deciders.of(MODEL);
        let answers = [bed.ask(frame(1, 1)).await, bed.ask(frame(2, 1)).await];
        // Two leases stand on its file: its look behind the answers
        // starts its watch, and it waits for a job from there.
        watching(&bed, 1).await;
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        for answer in &answers {
            bed.store.hand(&hand, LeaseId(lease_of(answer).id), gone);
        }
        settled(|| threads.lock().len() >= 2).await;
        let decider = || Some(format!("decide-{taker}"));
        assert_eq!(*threads.lock(), [decider(), decider()]);
        // No lease stands on the file: the decider's next look ends its
        // watch.
        watching(&bed, 0).await;
    }

    /// A controller's only decider keeps no watch. With leases on its
    /// file and no job it rests, so a frame handed on the lull is applied
    /// before its hand returns: by the hand in place, or by the decider's
    /// last take where the hand meets its rest.
    #[tokio::test]
    async fn a_controllers_only_decider_keeps_no_watch() {
        let threads = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&threads);
        let spec = picking("threads", move || Box::new(Threads(Arc::clone(&kept))));
        let mut bed = bed(Some(spec), true, (1, 8), []);
        let answers = [bed.ask(frame(1, 1)).await, bed.ask(frame(2, 1)).await];
        // The decider leaves its work behind its last answer: the case
        // waits until the store shows no thread at work.
        settled(|| bed.store.working() == 0).await;
        assert_eq!(bed.store.watches(), 0, "two leases stand on its file");
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        for (applied, answer) in answers.iter().enumerate() {
            bed.store.hand(&hand, LeaseId(lease_of(answer).id), gone);
            let told = threads.lock().len();
            assert_eq!(told, applied + 1, "applied before its hand returned");
        }
        assert_eq!(bed.store.watches(), 0, "no look is owed");
    }

    /// Waits until `files` of the bed's files show a watch. A decider
    /// starts its watch at its look, behind its answer, and ends it at a
    /// look: a read right behind an answer may come first.
    async fn watching(bed: &Bed, files: usize) {
        settled(|| bed.store.watches() == files).await;
    }

    /// A decider that waits with a lease on file waits its floor behind a
    /// job, twice as long behind each look that applied no frame, up to
    /// its cap, and its floor again behind a look that applied one: a
    /// file whose frames flow is not quiet. Once no lease stands on the
    /// file it waits with no bound. The decider is the case's own, and
    /// the case's receive answers each wait as a queue with no job would.
    #[test]
    fn a_watching_deciders_wait_doubles_over_a_quiet_file_and_falls_at_a_frame() {
        let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
        let mut decider = deciding(&bed);
        decider.working = Some(bed.store.at_work());
        let answer = decider.answer(checked(frame(1, 1)), GATEWAY, bed.script.taken);
        let lease = LeaseId(lease_of(&answer.frame()).id);
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let mut waits = Vec::new();
        let job = decider.wait_by(|within| {
            waits.push(within.map(|period| period.as_millis()));
            match waits.len() {
                // Handed in the seventh wait: the look behind it applies
                // a frame.
                7 => {
                    let dispatch = Said::Dispatched(Dispatch::default(), bed.script.taken);
                    bed.store.hand(&hand, lease, dispatch);
                }
                // Handed in the ninth: the look behind it applies the
                // release of the file's last lease.
                9 => {
                    let gone = Said::Released(None, ReleaseReason::ClientGone);
                    bed.store.hand(&hand, lease, gone);
                }
                10 => return Err(std_mpsc::RecvTimeoutError::Disconnected),
                _ => {}
            }
            Err(std_mpsc::RecvTimeoutError::Timeout)
        });
        assert!(job.is_none(), "the case's queue ends with no job");
        // Seven waits over the quiet file, two behind the frame, and no
        // bound behind the release.
        let mut wanted: Vec<_> = [1, 2, 4, 8, 16, 16, 16, 1, 2].map(Some).into();
        wanted.push(None);
        assert_eq!(waits, wanted);
        assert_eq!((LOOK_FLOOR.as_millis(), LOOK_CAP.as_millis()), (1, 16));
    }

    /// A decider with no place yet keeps no watch: before its first job
    /// its wait is handed no period, as over a file that holds no lease.
    #[test]
    fn a_decider_with_no_place_yet_waits_with_no_bound() {
        let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
        let mut decider = deciding(&bed);
        let mut waits = Vec::new();
        let job = decider.wait_by(|within| {
            waits.push(within);
            Err(std_mpsc::RecvTimeoutError::Disconnected)
        });
        assert!(job.is_none(), "the case's queue ends with no job");
        assert_eq!(waits, [None], "no place, so no watch and no period");
        assert_eq!(bed.store.watches(), 0);
    }

    /// The `random` picker, keeping the instant each release of a lease is
    /// told at.
    #[derive(Debug)]
    struct Heard(Arc<Mutex<Vec<Instant>>>);

    impl Lifecycle for Heard {
        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            self.0.lock().push(Instant::now());
        }
    }

    impl Picker for Heard {
        fn name(&self) -> &'static str {
            "heard"
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

    /// [`Seen`], as a stage that says it may refuse by a report of
    /// leases.
    #[derive(Debug)]
    struct Counted(Seen);

    impl Lifecycle for Counted {}

    impl Picker for Counted {
        fn name(&self) -> &'static str {
            self.0.name()
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
            self.0.pick(facts, candidates, kept, scores, band, out);
        }

        fn refuses_by_reports(&self) -> bool {
            true
        }
    }

    /// The `random` preset as generation 1 with the case's picker in its
    /// profile's pick: a picker that keeps what each decision reads in
    /// `rows`, made by `stage`.
    fn seeing<P: Picker + 'static>(
        rows: &Arc<Mutex<Vec<SeenRows>>>,
        stage: impl Fn(Seen) -> P + Send + Sync + 'static,
    ) -> AppliedSpec {
        let kept = Arc::clone(rows);
        picking("seen", move || {
            let rows = Arc::clone(&kept);
            Box::new(stage(Seen { rows }))
        })
    }

    /// A decision under a stage that may refuse by a report of leases
    /// reads a completion its gateway reported of a lease on another
    /// decider's file. That decider waits for a job with its watch over
    /// the file, so the frames were left in its ring; the deciding thread
    /// applies them before it reads its rows. The two models are taken by
    /// two deciders and served by the same members.
    #[tokio::test]
    async fn a_decision_whose_stage_may_refuse_by_reports_applies_what_waits_at_the_other_files() {
        use crate::service::controller::lifecycle::tests::{a_completed, a_dispatched};

        let rows = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(seeing(&rows, Counted)), true, (2, 8), []);
        let models = a_model_each(&bed.deciders, 2);
        let served: Vec<&str> = models.iter().map(String::as_str).collect();
        for name in ["s0", "s1", "s2"] {
            serving(&bed.store, name, &served, Role::Single, WireModes::NONE);
        }
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let answer = bed.ask(frame_for(&models[0], 1, 1)).await;
        let lease = lease_of(&answer).id;
        // The first model's decider waits for a job on its watch over the
        // lease.
        watching(&bed, 1).await;
        let at = Instant::now();
        lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease), at)
            .expect("the frame reads");
        lifecycle::completed(&bed.store, &hand, &a_completed(lease, picked(&answer)), at)
            .expect("the frame reads");
        // The other model's decision, at the other file, right behind the
        // completion: it reads no request in flight.
        bed.ask(frame_for(&models[1], 2, 1)).await;
        let rows = rows.lock();
        let read: u32 = rows[1].iter().map(|(count, _)| count).sum();
        assert_eq!(read, 0, "{rows:?}");
    }

    /// Two leases on the second file of `store`, booked on `member`, under
    /// the watch of that file's place, and the release of the first handed
    /// and left in the file's ring. The place is the case's own, so no
    /// decider's look applies the frame: its next look says whether the
    /// frame still waited ([`Watch::Fed`]) or was applied meanwhile
    /// ([`Watch::Quiet`]).
    fn a_frame_under_the_second_files_watch(
        store: &Arc<MemoryStore>,
        member: WorkerId,
    ) -> crate::store::AtWork {
        use crate::{
            engine::decision_slot::DecisionSlot,
            store::{Hand, Said, Watch},
            tracking::lease::ReleaseReason,
        };

        let now = Instant::now();
        let _acting = DecisionSlot::new(1).enter();
        let mut place = store.at_work();
        let issue = |request| {
            let record = crate::store::LeaseRecord {
                lease: Lease::default(),
                gateway: GATEWAY,
                request,
                target: member,
                picked: crate::store::Picked::default(),
                decode: None,
                term: 1,
                issued_at: now,
                dispatch_deadline: now + DISPATCH_TTL,
                state: LeaseState::ISSUED,
            };
            let booked = store.snapshot().booked_of(member);
            let issued = store.issue_decided(record, booked, |_| ());
            issued.expect("minted under the store's term").id
        };
        let first = issue(1);
        issue(2);
        assert_eq!(place.look(), Watch::Quiet, "two leases stand on its file");
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        store.hand(&Hand::new(GATEWAY), first, gone);
        place
    }

    /// A decision whose stages refuse by no report reads no place, though
    /// its decider lent one: a frame left under another file's watch
    /// stays there for that file's thread, and the decision takes no
    /// turn at it.
    #[test]
    fn a_decision_whose_stages_refuse_by_no_report_applies_nothing_at_the_other_files() {
        use crate::{engine::decision_slot::DecisionSlot, store::Watch};

        const SPOKEN: &str = "model-http";
        let store = Arc::new(MemoryStore::new().with_deciders(2));
        store
            .apply(preset("random", 1))
            .expect("above the unbound start");
        let member = ready(&store, "h0", SPOKEN, Role::Single, WireModes::HTTP);
        LeasesStore::set_term(&*store, 1);
        let mut watch = a_frame_under_the_second_files_watch(&store, member);
        // The first file's decision, its place lent, under the preset's
        // stages: each says it refuses by no report.
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let request = checked(frame_for(SPOKEN, 1, 1));
        let now = Instant::now();
        let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
        let taken = taken.placed(Some(&place));
        let mut buffers = Buffers::default();
        let handed = store.deciding(|world| {
            hand(
                world,
                &taken,
                &mut buffers,
                &AtomicU64::new(0),
                |_, _, rows, _, _| rows.len(),
            )
            .0
        });
        assert_eq!(handed, 1, "the decision was handed its row");
        assert_eq!(
            watch.look(),
            Watch::Fed,
            "the frame waited for its file's look"
        );
    }

    /// A pair's one decision applies what waits at the other files where
    /// a stage of either side may refuse by a report, as a single's does:
    /// the pair builds its own rows, so the apply is its own. Under the
    /// same plan with stages that refuse by no report it reads no place,
    /// and the frame waits for its file's look.
    #[test]
    fn a_pairs_decision_applies_what_waits_where_a_stage_of_either_side_may_refuse() {
        use crate::{engine::decision_slot::DecisionSlot, store::Watch};

        let waited = |prefill: &str, decode: &str| {
            let rows = Arc::new(Mutex::new(Vec::new()));
            let mut catalog = StageRegistry::builtin();
            let counted = move || -> Box<dyn Picker> {
                let rows = Arc::clone(&rows);
                Box::new(Counted(Seen { rows }))
            };
            catalog.picker("seen", counted).expect("a free name");
            let document = r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": {
                "p": { "pick": "PREFILL" },
                "d": { "pick": "DECODE" }
              },
              "bindings": [
                { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
                  "profiles": { "prefill": "p", "decode": "d" } },
                { "profile": "p" }
              ]
            }"#;
            let document = document
                .replace("PREFILL", prefill)
                .replace("DECODE", decode);
            let document: PolicySpec =
                serde_json::from_str(&document).expect("a document of this build");
            let digest = document.digest().expect("the document digests");
            let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
            let store = Arc::new(MemoryStore::new().with_deciders(2));
            let (singles, _) = fleet(&store);
            store
                .apply(spec.expect("the catalog holds the stage"))
                .expect("above the unbound start");
            LeasesStore::set_term(&*store, 1);
            let mut watch = a_frame_under_the_second_files_watch(&store, singles[0]);
            let _acting = DecisionSlot::new(0).enter();
            let place = store.at_work();
            let request = checked(phased(frame(1, 7), proto::Phase::Prefill));
            let now = Instant::now() + Duration::from_secs(3);
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
            let taken = taken.placed(Some(&place));
            let mut buffers = Buffers::default();
            let own = Part::default();
            let decision = store.deciding(|world| pair_in(world, &taken, &mut buffers, &own).0);
            assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
            watch.look()
        };
        assert_eq!(waited("random", "random"), Watch::Fed, "no such stage");
        assert_eq!(waited("random", "seen"), Watch::Quiet, "the decode side's");
        assert_eq!(waited("seen", "random"), Watch::Quiet, "the prefill side's");
    }

    /// One model a decider, in the deciders' order: the first names of an
    /// endless run whose requests each of `count` deciders takes.
    fn a_model_each(deciders: &Deciders, count: usize) -> Vec<String> {
        let mut models: Vec<Option<String>> = vec![None; count];
        for name in (0_u32..).map(|at| format!("model-{at}")) {
            if models.iter().all(Option::is_some) {
                break;
            }
            let taker = &mut models[deciders.of(&name)];
            if taker.is_none() {
                *taker = Some(name);
            }
        }
        models.into_iter().flatten().collect()
    }

    /// The idle cost of a watch, for its measurement at `deciders`
    /// deciders: one dispatched lease on each decider's file and no job,
    /// for five seconds between two printed lines. A reader of the
    /// threads named `decide-*` counts their wakes and their time on a
    /// core over the pause. The tenth of a second before the first line
    /// is not counted: each decider's wait grows to its cap in it.
    async fn idle(deciders: usize) {
        use std::io::Write as _;

        use crate::service::controller::lifecycle::tests::a_dispatched;

        let mut out = io::stdout();
        let mut bed = bed(Some(preset("random", 1)), true, (deciders, 8), []);
        let models = a_model_each(&bed.deciders, deciders);
        let served: Vec<&str> = models.iter().map(String::as_str).collect();
        for name in ["s0", "s1", "s2"] {
            serving(&bed.store, name, &served, Role::Single, WireModes::NONE);
        }
        let hand = lifecycle::open(&bed.store, GATEWAY);
        for model in &models {
            let answer = bed.ask(frame_for(model, 1, 1)).await;
            let dispatched = a_dispatched(lease_of(&answer).id);
            lifecycle::dispatched(&bed.store, &hand, &dispatched, Instant::now())
                .expect("the frame reads");
        }
        thread::sleep(Duration::from_millis(100));
        writeln!(
            out,
            "the idle pause begins: a lease on each of {deciders} files, no job"
        )
        .expect("stdout");
        thread::sleep(Duration::from_secs(5));
        writeln!(out, "the idle pause ends").expect("stdout");
    }

    /// The idle cost of one decider's watch ([`idle`]).
    #[tokio::test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    async fn one_idle_decider_measured() {
        idle(1).await;
    }

    /// The idle cost of eight deciders' watches ([`idle`]).
    #[tokio::test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    async fn eight_idle_deciders_measured() {
        idle(8).await;
    }

    /// The lag of a frame handed on a lull, measured at one of two
    /// deciders: from its hand to the decider's tell, over `FRAMES`
    /// frames a cell. At
    /// the floor each frame is handed within one shortest wait of its
    /// decision. At the cap it is handed three longest waits behind its
    /// decision and within one more: the decider's wait has grown to its
    /// cap by then. Each is handed at a phase that sweeps that wait, so
    /// no frame's instant is locked to a look.
    #[tokio::test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    async fn a_lulls_lag_measured() {
        use std::io::Write as _;

        const FRAMES: usize = 500;
        let mut out = io::stdout();
        let heard = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&heard);
        let spec = picking("heard", move || Box::new(Heard(Arc::clone(&kept))));
        let mut bed = bed(Some(spec), true, (2, 8), []);
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        let parts = u32::try_from(FRAMES).expect("five hundred");
        let cells = [
            ("at the floor", Duration::ZERO, LOOK_FLOOR),
            ("at the cap", 3 * LOOK_CAP, LOOK_CAP),
        ];
        let mut told = 0;
        for (cell, quiet, wait) in cells {
            let mut lags = Vec::with_capacity(FRAMES);
            for sent in 0..FRAMES {
                let answer = bed.ask(frame(1, 1)).await;
                // A step coprime to the sweep: each part of the wait once.
                let part = u32::try_from(sent * 7_919 % FRAMES).expect("under five hundred");
                thread::sleep(quiet + wait * part / parts);
                let handed = Instant::now();
                bed.store.hand(&hand, LeaseId(lease_of(&answer).id), gone);
                let at = loop {
                    if let Some(at) = heard.lock().get(told) {
                        break *at;
                    }
                    assert!(
                        handed.elapsed() < PATIENCE,
                        "a handed frame waits one look at most"
                    );
                    std::hint::spin_loop();
                };
                told += 1;
                lags.push(at.saturating_duration_since(handed));
            }
            lags.sort_unstable();
            let at = |part: usize| lags[(FRAMES * part / 100).min(FRAMES - 1)].as_micros();
            writeln!(
                out,
                "the lag of a frame handed on a lull {cell}, {FRAMES} frames: p50 {} us, p99 {} us, the longest {} us",
                at(50),
                at(99),
                at(100)
            )
            .expect("stdout");
        }
    }

    /// One closed-loop client, measured: the share of its decisions that
    /// read a count from before its last completion. The client asks,
    /// names its dispatch, reports the completion `served` later and asks
    /// again `think` later, so no lease of its own stands at a decision:
    /// one that reads a request in flight read a count from before the
    /// completion. Its requests are for one model, or for two in turn
    /// that two deciders take and the same members serve: each decision
    /// of the second form is made at another file than the completion
    /// before it.
    /// The stage that reads the counts is made by `stage`, and `under`
    /// names it in the cell's lines.
    async fn closed_loop<P: Picker + 'static>(
        stage: impl Fn(Seen) -> P + Send + Sync + 'static,
        under: &str,
    ) {
        use std::io::Write as _;

        use crate::service::controller::lifecycle::tests::{a_completed, a_dispatched};

        const ASKED: usize = 200;
        let mut out = io::stdout();
        let rows = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(seeing(&rows, stage)), true, (2, 8), []);
        let models = a_model_each(&bed.deciders, 2);
        let served: Vec<&str> = models.iter().map(String::as_str).collect();
        for name in ["s0", "s1", "s2"] {
            serving(&bed.store, name, &served, Role::Single, WireModes::NONE);
        }
        let hand = lifecycle::open(&bed.store, GATEWAY);
        let pause = |long: Duration| {
            if !long.is_zero() {
                thread::sleep(long);
            }
        };
        let millis = Duration::from_millis;
        let paces = [
            (millis(0), millis(0)),
            (millis(0), millis(2)),
            (millis(48), millis(0)),
            (millis(48), millis(20)),
        ];
        for (form, turn) in [("one model", [0, 0]), ("two models in turn", [0, 1])] {
            for (served, think) in paces {
                rows.lock().clear();
                for asked in 0..ASKED {
                    let answer = bed.ask(frame_for(&models[turn[asked % 2]], 1, 1)).await;
                    let lease = lease_of(&answer).id;
                    lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease), Instant::now())
                        .expect("the frame reads");
                    pause(served);
                    let completed = a_completed(lease, picked(&answer));
                    lifecycle::completed(&bed.store, &hand, &completed, Instant::now())
                        .expect("the frame reads");
                    pause(think);
                }
                // The first decision of a cell stands behind the cell
                // before it: it is not counted.
                let rows = rows.lock();
                let behind = rows.iter().skip(1);
                let late = behind.filter(|read| read.iter().any(|(count, _)| *count > 0));
                writeln!(
                    out,
                    "a closed loop of {form}{under}, served {} ms, think {} ms: {} of {} decisions read a count from before the last completion",
                    served.as_millis(),
                    think.as_millis(),
                    late.count(),
                    ASKED - 1
                )
                .expect("stdout");
            }
        }
    }

    /// [`closed_loop`] under a stage that only picks by its counts.
    #[tokio::test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    async fn a_closed_loop_measured() {
        closed_loop(|seen| seen, "").await;
    }

    /// [`closed_loop`] under a stage that says it may refuse by a report
    /// of leases.
    #[tokio::test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    async fn a_closed_loop_under_a_counting_stage_measured() {
        closed_loop(Counted, " under a stage that may refuse by a report").await;
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

        // Keeps what a decision reads and picks as the random picker.
        fn refuses_by_reports(&self) -> bool {
            false
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
        let mut catalog = StageRegistry::builtin();
        let seen = move || -> Box<dyn Picker> {
            let rows = Arc::clone(&kept);
            Box::new(Seen { rows })
        };
        catalog.picker("seen", seen).expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "seen".into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
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
        assert_eq!(counted(&bed.counts), [0; 9]);
        assert_eq!(
            busy(&bed.ask(frame(2, 2)).await),
            Reason::BudgetGoneOnArrival
        );
        assert_eq!(busy(&bed.ask(frame(3, 3)).await), Reason::Unspecified);
        for id in [4, 5] {
            assert_eq!(busy(&bed.ask(frame(id, 1)).await), Reason::NotLeader);
        }
        // In the order of the reasons' numbers on the wire.
        assert_eq!(Counts::BUSY[3..5], ["budget_gone_on_arrival", "not_leader"]);
        assert_eq!(counted(&bed.counts), [1, 0, 0, 1, 2, 0, 0, 0, 0]);
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
        bed.standing.send_replace(ElectionObservation {
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
        let mut document = PolicySpec::preset("random").expect("a preset");
        document.generation = 2;
        document.explain.sample = 1;
        bed.store
            .apply(applied(&document))
            .expect("a later generation");
        let answer = bed.ask(frame(1, 1)).await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
        assert_eq!(bed.script.readers.lock().len(), 2);
        assert!(bed.deciders.explain(1, GATEWAY, &[1]).is_none());
        assert!(bed.deciders.shared.samples.get().is_none());
    }

    #[tokio::test]
    async fn an_unsampled_served_decision_lays_out_no_rings() {
        let mut bed = leading([]);
        let answer = bed.ask(frame(1, 1)).await;
        assert!(matches!(outcome(&answer), Outcome::Pick(_)));
        assert!(bed.deciders.explain(1, GATEWAY, &[1]).is_none());
        assert!(bed.deciders.shared.samples.get().is_none());
        assert_eq!(bed.script.readers.lock().len(), 2);
    }

    #[test]
    fn a_served_decision_that_keeps_no_record_answers_during_a_read() {
        let mut document = PolicySpec::preset("random").expect("a preset");
        document.generation = 1;
        document.explain.sample = 2;
        let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("the case's runtime");
        for id in 1..=2 {
            runtime.block_on(bed.ask(frame(id, 1)));
        }

        let shared = Arc::clone(&bed.deciders.shared);
        let rings = shared.samples.get().expect("the ring of the first sample");
        let _read = rings[0].lock();
        let answer = runtime.block_on(bed.ask(frame(3, 1)));
        assert!(matches!(outcome(&answer), Outcome::Pick(_)));
    }

    #[tokio::test]
    async fn a_new_term_restarts_the_served_sample_count() {
        let mut document = PolicySpec::preset("random").expect("a preset");
        document.generation = 1;
        document.explain.sample = 2;
        let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
        let first = bed.ask(frame(1, 1)).await;
        assert!(matches!(outcome(&first), Outcome::Pick(_)));

        assert!(bed.elector.release().expect("the release"));
        assert_eq!(
            bed.elector.tick(bed.script.taken).expect("the take"),
            Tick::Acquired { term: 2 }
        );
        bed.store.set_term(2);
        let answer = bed.ask(frame(2, 1)).await;
        assert_eq!(answer.term, 2);
        assert!(matches!(outcome(&answer), Outcome::Pick(_)));
        assert!(bed.deciders.explain(2, GATEWAY, &[2]).is_none());
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
        bed.standing.send_replace(ElectionObservation {
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
            shared: Arc::default(),
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
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
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

    /// A decision over the publication's own rows is handed the pool's
    /// engines beside them: the ranks of one data-parallel engine as one
    /// engine of the list, and a worker alone as another. So is a decision
    /// over rows written for its request while no row is cut from them. A
    /// row the request's narrowing takes moves the rows behind it from the
    /// places the list names, and those rows are handed none.
    #[test]
    fn a_decision_over_the_published_rows_is_handed_the_pools_engines() {
        const SPOKEN: &str = "model-ranks";
        let store = MemoryStore::new();
        for rank in 0..2 {
            store.upsert(WorkerFacts {
                key: format!("grpc://e:9000#{rank}"),
                engine: "grpc://e:9000".to_owned(),
                incarnation: 1,
                wire_modes: WireModes::GRPC,
                models: vec![SPOKEN.to_owned()],
                dp_rank: rank,
                dp_size: 2,
                ..WorkerFacts::default()
            });
        }
        ready(&store, "h0", SPOKEN, Role::Single, WireModes::HTTP);
        // The engines the engine is handed, each as its rows' ranks.
        let engines = |frame: proto::DecideRequest| {
            let request = checked(frame);
            let now = Instant::now();
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
            let (mut buffers, undialed) = (Buffers::default(), AtomicU64::new(0));
            store.deciding(|world| {
                let handed = hand(
                    world,
                    &taken,
                    &mut buffers,
                    &undialed,
                    |_, pool, rows, _, _| {
                        let ranks = |engine| {
                            let listed = pool.engines.rows(engine).iter();
                            listed.map(|&row| rows[row as usize].dp_rank).collect()
                        };
                        let mut engines: Vec<Vec<u16>> =
                            (0..pool.engines.len()).map(ranks).collect();
                        engines.sort();
                        engines
                    },
                );
                handed.0
            })
        };
        let lent = engines(frame_for(SPOKEN, 1, 1));
        assert_eq!(lent, [vec![0], vec![0, 1]], "lent: one engine of two ranks");
        let wired = constrained(frame_for(SPOKEN, 2, 2), 0b011, None);
        assert_eq!(engines(wired), lent, "written and whole: the same list");
        let cut = constrained(frame_for(SPOKEN, 3, 3), 0b001, None);
        assert_eq!(engines(cut), Vec::<Vec<u16>>::new(), "a row cut: no list");
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

        let mut catalog = StageRegistry::builtin();
        catalog
            .filter("shown", || Box::new(Shown))
            .expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = 1;
        for profile in document.profiles.values_mut() {
            profile.filter.push("shown".into());
        }
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the catalog holds every stage the document names");
        let store = MemoryStore::new();
        let (mut singles, _) = fleet(&store);
        singles.sort_unstable();
        store.apply(spec).expect("above the unbound start");
        let taken = Instant::now();
        let now = taken + Duration::from_secs(3);
        // The rows whose own bit says a veto, the mask's length, whether
        // the engine picked, and the rows written into the decider's
        // buffers.
        let decided = || {
            let request = checked(frame(1, 1));
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
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
                            .filter(|row| row.health.breaker_open)
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
        rule.assignment = spec::Assignment::Delegate;
        rule.inflight_cap = 2;
        document.generation = 1;
        document.view.headers = manual.view.headers.clone();
        let binding = document.bindings.first_mut().expect("the preset's binding");
        // The rule names the key's header, so the ring's own sources go: a
        // binding names them once.
        binding.routing_key = None;
        binding.sessions = Some(rule);
        binding.overload = Some(spec::OverloadSpec {
            waiting_requests: Some(1),
            ..spec::OverloadSpec::default()
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
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
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
            crate::engine::LoadRow {
                queued: 1,
                ..crate::engine::LoadRow::default()
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
    /// reads no count too. A hint that keeps rows beside the wires leaves
    /// the written row its count.
    #[test]
    fn a_lease_on_file_is_read_in_a_written_row_and_beside_a_lent_one() {
        const SPOKEN: &str = "model-http";
        for (stated, hint, own) in [(true, None, 1), (true, Some(7), 1), (false, None, 0)] {
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
                state: LeaseState::ISSUED,
            });
            assert_eq!(filed, Ok(()));
            let frame = frame_for(SPOKEN, 1, 1);
            let frame = if stated {
                constrained(frame, 0b001, hint)
            } else {
                frame
            };
            let request = checked(frame);
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
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
            assert_eq!(counts, [(1, own)], "wires stated: {stated}, {hint:?}");
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
            state: LeaseState::ISSUED,
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
        let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
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
        let ruled = |assignment: crate::document::Assignment, pick: &str| {
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
        let (delegate, min_load) = (
            crate::document::Assignment::Delegate,
            crate::document::Assignment::MinLoad,
        );
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
        let taken = DecisionCall::at(&request, gateway, received, now).expect("budget is left");
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
                        .filter(|(row, at)| row.health.breaker_open || cx.masks(*at))
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
        // A profile whose own stages read the column and whose picker takes
        // no sample is named too, under no cap: no column, and here nobody
        // to name.
        let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
        document.generation = 3;
        for profile in document.profiles.values_mut() {
            profile.pick = "best".into();
        }
        store
            .apply(applied(&document))
            .expect("above the standing one");
        let whole = handed(&store, frame(1, 1), OTHER, now);
        assert_eq!(
            (whole.column, whole.named, whole.pool),
            (0, Some((0, 0)), (3, 3))
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
        // The build this registration owes, asked: the cases read the row
        // at once.
        store.publish_now();
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
            facts.hash_kind = proto::HashKind::Token.into();
            let request = checked(frame);
            let now = Instant::now() + Duration::from_secs(3);
            let taken = DecisionCall::at(&request, gateway, now, now).expect("budget is left");
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
            engine::view::NamedAffinity,
            store::{Named, Naming},
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
            facts.hash_kind = proto::HashKind::Token.into();
            let frame = constrained(frame, u32::from(WireModes::GRPC.bits()), None);
            let request = checked(frame);
            let taken = DecisionCall::at(&request, GATEWAY, at, at).expect("budget is left");
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
                    signal_cells: false,
                };
                let built = world
                    .build(MODEL, Role::Single, &ask, &mut view)
                    .expect("the pool");
                view.candidates
                    .retain(|row| row.wire_modes.intersects(WireModes::GRPC));
                let naming = Naming {
                    gateway: GATEWAY,
                    block_hashes: &[1, 2],
                    namespace_seed: 0,
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
    /// it names by it, and a profile that takes none is named every holder
    /// and every warming row. One whose own stages read no column keeps
    /// its column, and so does one whose warm-up stages rule by different
    /// numbers. The needs of its workers' stream carry the pair's decode
    /// side, which reads the column and samples sixteen here.
    #[test]
    fn the_hand_caps_the_rows_it_names_by_the_profile_that_decides() {
        use crate::engine::Phase;

        // What the hand gives a decode request asked alone under `pick`:
        // the holders named with the rows handed, and the column's length.
        // The pair's profile is the document's first, so the cap of the
        // profile that decides is not the first the engine keeps.
        let handed = |score: &str, pick: &str, (own, named, ruled): (u32, u32, bool)| {
            let document: PolicySpec = serde_json::from_str(&format!(
                r#"{{
                  "apiVersion": "smg.policy/v1",
                  "generation": 1,
                  "profiles": {{
                    "pair": {{ "score": ["prefix-affinity"], "pick": {{ "best": {{ "sample": 16 }} }} }},
                    "solo": {{ "score": [{score}], "pick": {pick} }}
                  }},
                  "bindings": [
                    {{ "match": {{ "phase": ["decode"] }}, "profile": "solo" }},
                    {{ "plan": "pd-pair", "profiles": {{ "prefill": "pair", "decode": "pair" }} }}
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
                // No binding of this document covers an encode item: no row
                // is named for one, whatever its profiles cap.
                assert_eq!(engine.named_for(MODEL, Phase::Encode), 0);
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
            facts.hash_kind = proto::HashKind::Token.into();
            let request = checked(frame);
            let now = Instant::now() + Duration::from_secs(3);
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
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
        // No sample: every one of the twenty holders is named, and no
        // column. Lists capped at the pair's sample would hide holders from
        // its pass.
        let all = (0, u32::MAX, false);
        assert_eq!(handed(affinity, r#""best""#, all), (Some((20, 20)), 0));
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

    /// A stage that reads holders is fed them served, whatever its picker's
    /// sample: under a profile that ranks by the prefix and takes no sample
    /// a request goes to the one member that holds its block, among
    /// members that hold other blocks and nothing in flight. A request
    /// for a block nobody holds reads no holder, and its seeds spread it.
    #[test]
    fn a_profile_that_takes_no_sample_reads_its_requests_holder_served() {
        let document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": { "score": ["prefix-affinity"], "pick": "best" } },
              "bindings": [{ "profile": "p" }]
            }"#,
        )
        .expect("a document of this build");
        let store = MemoryStore::new();
        store
            .apply(applied(&document))
            .expect("above the unbound start");
        // Eight ready members, each holding a block of its own: the sixth
        // holds the request's.
        let members: Vec<WorkerId> = (0..8u64)
            .map(|n| {
                let facts = WorkerFacts {
                    wire_modes: WireModes::HTTP,
                    ..WorkerFacts::default()
                };
                let block = if n == 5 { 1 } else { 100 + n };
                let id = held_by(&store, &format!("w{n}"), facts, &[block]);
                let row = WorkersStore::get(&store, id).expect("the row");
                store
                    .set_status(id, WorkerStatus::Ready, row.revision)
                    .expect("the row stands");
                id
            })
            .collect();
        store.publish_now();
        let now = Instant::now() + Duration::from_secs(3);
        let asked = |block: u64, seed: u64| {
            let mut frame = frame(1, seed);
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.block_hashes = vec![block];
            facts.hash_kind = proto::HashKind::Token.into();
            decided(&store, frame, now, &[]).1
        };
        let held = (1..=32).filter(|&seed| asked(1, seed) == Some(members[5]));
        assert_eq!(
            held.count(),
            32,
            "decisions that go to the block's one holder"
        );
        let spread: BTreeSet<Option<WorkerId>> = (1..=32).map(|seed| asked(999, seed)).collect();
        assert!(spread.len() > 1, "a block nobody holds: {spread:?}");
    }

    /// A decision over rows the store named states what its first target
    /// holds of the request: the holder's block of the request's, read
    /// from the index, and nothing held where the block is nobody's. The
    /// engine's plain entry states nothing over the same rows, and under a
    /// profile with no prefix stage a pick states nothing.
    #[test]
    fn a_decision_states_what_its_first_target_holds_of_the_request() {
        let stated = |score: &str, block: u64, view: &str| {
            let document: PolicySpec = serde_json::from_str(&format!(
                r#"{{
                  "apiVersion": "smg.policy/v1",
                  "generation": 1,
                  "view": {view},
                  "profiles": {{ "p": {{ "score": [{score}], "pick": "best" }} }},
                  "bindings": [{{ "profile": "p" }}]
                }}"#
            ))
            .expect("a document of this build");
            let store = MemoryStore::new();
            store
                .apply(applied(&document))
                .expect("above the unbound start");
            // Eight ready members, each holding a block of its own: the
            // sixth holds block 1.
            let members: Vec<WorkerId> = (0..8u64)
                .map(|n| {
                    let facts = WorkerFacts {
                        wire_modes: WireModes::HTTP,
                        ..WorkerFacts::default()
                    };
                    let held = if n == 5 { 1 } else { 100 + n };
                    let id = held_by(&store, &format!("w{n}"), facts, &[held]);
                    let row = WorkersStore::get(&store, id).expect("the row");
                    store
                        .set_status(id, WorkerStatus::Ready, row.revision)
                        .expect("the row stands");
                    id
                })
                .collect();
            store.publish_now();
            let now = Instant::now() + Duration::from_secs(3);
            let mut frame = frame(1, 7);
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.block_hashes = vec![block];
            facts.hash_kind = proto::HashKind::Token.into();
            let request = checked(frame);
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            let (decision, _) = store.deciding(|world| {
                store_backed_engine::decide_in(world, &taken, &mut buffers, &Part::default())
            });
            let pick = *decision.pick().expect("a pick over eight ready members");
            // The same rows at the engine's plain entry: no stage is asked.
            let plain = store.deciding(|world| {
                let engine = world.spec().engine();
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &AtomicU64::new(0),
                    |facts, pool, rows, cx, scratch| engine.decide(facts, pool, rows, cx, scratch),
                )
                .0
            });
            assert_eq!(plain.pick().map(Pick::overlap), Some(Overlap::NONE));
            let first = pick.targets.first().map(|target| target.id);
            (first == Some(members[5]), pick.overlap())
        };
        let (holder, held) = stated(r#""prefix-affinity""#, 1, "{}");
        assert!(holder, "the block's one holder is picked");
        assert_eq!(
            (held.blocks, held.source, held.margin_held),
            (1, OverlapSource::Index, false)
        );
        assert!(held.of >= 1, "the request's blocks: {held:?}");
        // Under a limit of the reports' age the stage stands behind the
        // limit's reader of loads, and says the same.
        let aged = stated(r#""prefix-affinity""#, 1, r#"{"max_report_age_ms": 1000}"#);
        assert_eq!(aged, (true, held));
        let (_, missed) = stated(r#""prefix-affinity""#, 999, "{}");
        let nothing = Overlap {
            of: held.of,
            ..Overlap::default()
        };
        assert_eq!(missed, nothing, "a block nobody holds");
        let (_, unsaid) = stated(r#""least-load""#, 1, "{}");
        assert_eq!(unsaid, Overlap::default(), "no prefix stage");
    }

    /// One document, one pool, one answer on both feeds: served, the rows
    /// named under no cap decide as a column a host fills from them. Twenty
    /// of twenty-eight members hold the request's first block, more than
    /// any sample names, and eight of those its second: every one is named
    /// with its depth, and each seed's pick is one of the deepest holders
    /// on either feed.
    #[test]
    fn rows_named_under_no_cap_decide_as_the_column_filled_from_them() {
        let document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": { "score": ["prefix-affinity"], "pick": "best" } },
              "bindings": [{ "profile": "p" }]
            }"#,
        )
        .expect("a document of this build");
        let store = MemoryStore::new();
        store
            .apply(applied(&document))
            .expect("above the unbound start");
        let members: Vec<WorkerId> = (0..28u64)
            .map(|n| {
                let facts = WorkerFacts {
                    wire_modes: WireModes::HTTP,
                    ..WorkerFacts::default()
                };
                let held: &[u64] = match n {
                    0..=7 => &[1, 2],
                    8..=19 => &[1],
                    _ => &[77],
                };
                let id = held_by(&store, &format!("w{n:02}"), facts, held);
                let row = WorkersStore::get(&store, id).expect("the row");
                store
                    .set_status(id, WorkerStatus::Ready, row.revision)
                    .expect("the row stands");
                id
            })
            .collect();
        store.publish_now();
        let deepest: BTreeSet<WorkerId> = members[..8].iter().copied().collect();
        let now = Instant::now() + Duration::from_secs(3);
        let mut picked = BTreeSet::new();
        for seed in 1..=64 {
            let mut frame = frame(1, seed);
            let facts = frame.facts.as_mut().expect("the case's frame has facts");
            facts.block_hashes = vec![1, 2];
            facts.hash_kind = proto::HashKind::Token.into();
            let request = checked(frame);
            let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
            let mut buffers = Buffers::default();
            let (named, by_named, by_column) = store.deciding(|world| {
                let engine = world.spec().engine();
                hand(
                    world,
                    &taken,
                    &mut buffers,
                    &AtomicU64::new(0),
                    |facts, pool, rows, cx, scratch| {
                        let named = facts.named.expect("a profile that reads holders is named");
                        assert!(facts.affinity.is_empty(), "no column beside the rows named");
                        // The column a host fills from the same holders.
                        let mut column = vec![AffinityFacts::default(); rows.len()];
                        for holder in named.holders {
                            column[holder.row as usize].overlap_blocks = holder.overlap_blocks;
                        }
                        let over_column = RequestFacts {
                            affinity: &column,
                            named: None,
                            ..*facts
                        };
                        let mut first = |facts: &RequestFacts<'_>| {
                            let decision = engine.decide(facts, pool, rows, cx, scratch);
                            let pick = decision.pick();
                            pick.and_then(|pick| pick.targets.as_slice().first().map(|to| to.id))
                        };
                        let depths: Vec<u32> =
                            named.holders.iter().map(|at| at.overlap_blocks).collect();
                        (depths, first(facts), first(&over_column))
                    },
                )
                .0
            });
            assert_eq!(named.len(), 20, "every holder is named: seed {seed}");
            assert_eq!(named.iter().filter(|&&depth| depth == 2).count(), 8);
            assert_eq!(by_named, by_column, "seed {seed}");
            let to = by_named.expect("a pick");
            assert!(deepest.contains(&to), "seed {seed} went to {to:?}");
            picked.insert(to);
        }
        assert!(picked.len() > 1, "the deepest holders tie: {picked:?}");
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

    /// The kinds of a gateway's vetoes reach its decisions. A pool it doubts
    /// for liveness on every worker is steered: a doubt is not an outage. A
    /// breaker's open window on each worker refuses the pool, as the gateway
    /// refuses it. One doubted worker among held ones takes every request,
    /// a kind this build has no name for holds as a breaker does, and
    /// another gateway reads none of it.
    #[tokio::test]
    async fn a_pool_behind_the_gateways_open_breakers_is_refused_and_one_it_doubts_is_steered() {
        const BREAKER: i32 = proto::gateway_veto::Kind::Breaker as i32;
        const LIVENESS: i32 = proto::gateway_veto::Kind::Liveness as i32;
        let mut bed = leading([]);
        let now = bed.script.taken;
        let until = now + Duration::from_secs(60);
        let workers = bed.workers.clone();
        let fence = bed.elector.fence();
        let gateways = Arc::new(Gateways::new(GRACE, 1_024, fence, Arc::clone(&bed.store)));
        let seat = gateways
            .welcome(GATEWAY, "gateway-a", 1, now)
            .expect("the table takes the Hello");
        let veto = |worker: WorkerId, kind: i32, until: Option<Instant>| {
            assert_eq!(seat.veto(worker, kind, until, now), Vetoed::Shown);
        };
        let shed_at = |answer: &proto::DecideAnswer| match outcome(answer) {
            Outcome::Shed(shed) => (shed.reason(), shed.emptied_by.clone()),
            other => panic!("a shed: {other:?}"),
        };
        let at_health = (proto::ShedReason::NoEndpoints, "healthy".to_owned());
        for &worker in &workers {
            veto(worker, LIVENESS, Some(until));
        }
        let steered = bed.ask(frame(1, 1)).await;
        assert!(workers.contains(&picked(&steered)));
        for &worker in &workers {
            veto(worker, BREAKER, Some(until));
        }
        assert_eq!(shed_at(&bed.ask(frame(2, 2)).await), at_health);
        // One breaker closes: its worker is doubted alone.
        let doubted = workers[1];
        veto(doubted, BREAKER, None);
        for (id, seed) in (3..=10).zip(3..) {
            let answer = bed.ask(frame(id, seed)).await;
            assert_eq!(picked(&answer), doubted, "seed {seed}");
        }
        veto(doubted, 7, Some(until));
        assert_eq!(shed_at(&bed.ask(frame(11, 11)).await), at_health);
        assert_eq!(bed.submit_at(frame(12, 12), OTHER, Duration::ZERO), None);
        assert!(workers.contains(&picked(&bed.answer().await)));
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
        // A pair enters at its prefill. A request for its decode side has
        // no decision of its own: Busy with no reason, never one side's
        // decision and never a shed, which is a policy's word.
        for id in [1, 2] {
            let answer = bed.ask(phased(frame(id, 77), proto::Phase::Decode)).await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice(), answer.term),
                (Reason::Unspecified, &[id][..], 1)
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
        assert_eq!(busies.each_ref().map(counted), [3, 0, 0, 0, 0, 0, 0, 0, 0]);
        // The same document's other phases are decided: the single binding
        // covers them.
        for (id, phase) in [(4, proto::Phase::Single), (5, proto::Phase::Encode)] {
            let answer = bed.ask(phased(frame(id, 77), phase)).await;
            assert!(matches!(outcome(&answer), Outcome::Pick(_)), "{phase:?}");
        }
    }

    /// The worker and the decode of each target of a pick's answer, and
    /// each lease's id, side and target index.
    type PairAnswer = (Vec<(u64, Option<u64>)>, Vec<(u64, proto::Side, u32)>);

    fn pair_of(answer: &proto::DecideAnswer) -> PairAnswer {
        let Outcome::Pick(pick) = outcome(answer) else {
            panic!("a pick: {answer:?}");
        };
        let decode = |target: &proto::Target| target.pair.as_ref().map(|pair| pair.decode_id);
        let targets = pick
            .targets
            .iter()
            .map(|target| (target.id, decode(target)));
        (targets.collect(), sides_of(answer))
    }

    fn sides_of(answer: &proto::DecideAnswer) -> Vec<(u64, proto::Side, u32)> {
        let Some(Answer::Decision(decision)) = &answer.answer else {
            panic!("a decision: {answer:?}");
        };
        let side = |lease: &proto::Lease| (lease.id, lease.side(), lease.target_index);
        decision.leases.iter().map(side).collect()
    }

    #[tokio::test]
    async fn a_pair_s_reply_names_the_decode_side_s_own_session() {
        use crate::engine::view::{Identity, Phase};

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
        let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
        let key = 17;
        let session = |phase| {
            RequestFacts {
                model: MODEL,
                phase,
                session_key: Identity::hashed(key),
                ..RequestFacts::default()
            }
            .session()
            .get()
        };
        let frame = keyed(session(Phase::Prefill), 77);
        let answer = bed.ask(phased(frame, proto::Phase::Prefill)).await;
        let Some(Answer::Decision(decision)) = answer.answer else {
            panic!("a decision: {answer:?}");
        };
        let leases = decision
            .leases
            .iter()
            .map(|lease| (lease.side(), lease.session_key));
        assert_eq!(
            leases.collect::<Vec<_>>(),
            [
                (proto::Side::Prefill, session(Phase::Prefill)),
                (proto::Side::Decode, session(Phase::Decode)),
            ]
        );
    }

    /// A pair's ask reads what both sides' profiles read. The prefill
    /// lookup answers for the prefill's profile alone, so a column the
    /// decode profile alone declares is written by the decode lookup's
    /// word: one entry a decode row.
    #[test]
    fn a_pairs_rows_carry_the_column_its_decode_profile_alone_reads() {
        let document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": {
                "p": { "pick": "random" },
                "d": { "score": ["prefix-affinity"], "pick": "best" }
              },
              "bindings": [
                { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
                  "profiles": { "prefill": "p", "decode": "d" } },
                { "profile": "p" }
              ]
            }"#,
        )
        .expect("a document of this build");
        let store = MemoryStore::new();
        fleet(&store);
        store
            .apply(applied(&document))
            .expect("above the unbound start");
        let request = checked(phased(frame(1, 7), proto::Phase::Prefill));
        let now = Instant::now() + Duration::from_secs(3);
        let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
        let mut buffers = Buffers::default();
        let own = Part::default();
        let decision = store.deciding(|world| pair_in(world, &taken, &mut buffers, &own).0);
        assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
        let pools = buffers.pair.pools();
        assert_eq!(
            (pools.decodes.len(), pools.decode_affinity.len()),
            (1, 1),
            "the decode side's column"
        );
    }

    /// The pair door: one request at a pair plan's prefill is one answer
    /// that names both workers under two leases of one id, and one record
    /// on file whose two sides each book their own worker.
    #[tokio::test]
    async fn a_pair_plans_prefill_request_is_one_answer_of_both_workers_and_two_leases_of_one_id() {
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
        let member = |name: &str| {
            let row = WorkersStore::by_key(&*bed.store, &format!("http://{name}:8000"));
            row.expect("a member of the fleet").id
        };
        let (prefill, decode) = (member("p0"), member("d0"));
        let answer = bed.ask(phased(frame(1, 77), proto::Phase::Prefill)).await;
        let (targets, sides) = pair_of(&answer);
        assert_eq!(targets, [(prefill.0, Some(decode.0))]);
        let id = sides.first().map_or(0, |side| side.0);
        assert_ne!(id, 0);
        assert_eq!(
            sides,
            [(id, proto::Side::Prefill, 0), (id, proto::Side::Decode, 0)]
        );
        assert_eq!(LeasesStore::live(&*bed.store, GATEWAY), 1, "one record");
        assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
        // Each side's lease books its own worker: the next decision's rows
        // read one request on the prefill and one on the decode.
        let mut rows = crate::store::PairBuffers::default();
        bed.store.deciding(|world| {
            let ask = Ask {
                gateway: GATEWAY,
                now: Instant::now(),
                pinned: None,
                affinity: false,
                signal_cells: false,
            };
            let built = world
                .build_pair(MODEL, &ask, &mut rows)
                .expect("both sides stand");
            world.count_pair_in_flight(&built, &ask, &mut rows);
        });
        let pools = rows.pools();
        let booked = |rows: &[Candidate]| rows.iter().map(|row| row.in_flight).collect::<Vec<_>>();
        assert_eq!(
            (booked(pools.prefills), booked(pools.decodes)),
            (vec![1], vec![1])
        );
        // The next pair's two sides share an id of their own.
        let answer = bed.ask(phased(frame(2, 78), proto::Phase::Prefill)).await;
        let (_, sides) = pair_of(&answer);
        assert_eq!(sides.len(), 2);
        assert_ne!(sides[0].0, id, "one id a decision");
    }

    /// The next pair reads what the first left on file on each side: under
    /// a profile that scores by load, the same request asked again takes
    /// the workers the first pair did not book.
    #[tokio::test]
    async fn a_pairs_next_decision_reads_the_counts_its_first_left_on_each_side() {
        let document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "side": { "score": ["least-load"], "pick": "best" } },
              "bindings": [
                { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
                  "profiles": { "prefill": "side", "decode": "side" } },
                { "profile": "side" }
              ]
            }"#,
        )
        .expect("a document of this build");
        let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
        // A second worker a side beside the bed's own.
        for (name, role) in [("p1", Role::Prefill), ("d1", Role::Decode)] {
            ready(&bed.store, name, MODEL, role, WireModes::NONE);
        }
        // One seed, the same rows: the first pair's leases alone tell the
        // two decisions apart, one request on each worker it took.
        let first = bed.ask(phased(frame(1, 77), proto::Phase::Prefill)).await;
        let second = bed.ask(phased(frame(2, 77), proto::Phase::Prefill)).await;
        let (first, second) = (pair_of(&first).0, pair_of(&second).0);
        let (first, second) = (first[0], second[0]);
        assert_ne!(first.0, second.0, "the prefill the first pair booked");
        assert_ne!(first.1, second.1, "the decode the first pair booked");
    }

    /// A pair's answer that does not leave gives the lease up whole: both
    /// sides leave the file at once, and the engine that heard each side's
    /// issue hears each side's end.
    #[tokio::test]
    async fn a_pairs_answer_that_does_not_leave_gives_both_sides_up() {
        let told = Arc::new(Mutex::new(Vec::new()));
        // The decision's last read of the clock shows its budget gone.
        let steps = [Step::At(Duration::ZERO), Step::At(BUDGET)];
        let mut bed = bed(Some(hearing_pairs(&told)), true, (1, 8), steps);
        let answer = bed.ask(phased(frame(1, 77), proto::Phase::Prefill)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
        assert_eq!(bed.store.live(GATEWAY), 0);
        // One id, and each call once a side.
        let (id, late) = (1 << 32 | 1, Some(ReleaseReason::LateDecision));
        assert_eq!(
            *told.lock(),
            [(id, None), (id, None), (id, late), (id, late)]
        );
    }

    /// A pair plan for the prefill and decode phases, picking through
    /// [`Told`]: what its engine tells its stage of leases is kept in
    /// `told`.
    fn hearing_pairs(told: &Arc<Mutex<Calls>>) -> AppliedSpec {
        let kept = Arc::clone(told);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("told", move || Box::new(Told(Arc::clone(&kept))))
            .expect("a free name");
        let document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": { "pick": "told" } },
              "bindings": [
                { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
                  "profiles": { "prefill": "p", "decode": "p" } },
                { "profile": "p" }
              ]
            }"#,
        )
        .expect("a document of this build");
        let digest = document.digest().expect("the document digests");
        AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the catalog holds the stage")
    }

    /// A pair asked again ends each side of its lease that waits: one id,
    /// and the engine that heard each side's issue hears each side's end,
    /// ahead of the two sides the repeat is issued.
    #[tokio::test]
    async fn a_pair_asked_again_ends_each_side_that_waits() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let mut bed = bed(Some(hearing_pairs(&told)), true, (1, 8), []);
        let pair = || phased(frame(1, 77), proto::Phase::Prefill);
        bed.ask(pair()).await;
        bed.ask(again(pair())).await;
        let (first, second) = (1 << 32 | 1, 1 << 32 | 2);
        let abandoned = Some(ReleaseReason::Abandoned);
        let sides = |id, word| [(id, word), (id, word)];
        let heard = [
            sides(first, None),
            sides(first, abandoned),
            sides(second, None),
        ];
        assert_eq!(*told.lock(), heard.concat());
        let store: &dyn LeasesStore = bed.store.as_ref();
        assert_eq!(store.get(GATEWAY, LeaseId(first)), None);
        assert_eq!(asked_again(&bed), [0, 1]);
    }

    /// The encode side rides the plan that names it. The engine's entry
    /// would shed an encode request there as if no encoder stood, while
    /// one stands ready.
    #[tokio::test]
    async fn the_plan_with_an_encode_side_decides_its_pair_and_no_other_side() {
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
        for (id, phase) in [(1, proto::Phase::Encode), (3, proto::Phase::Decode)] {
            let answer = bed.ask(phased(frame(id, 77), phase)).await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice(), answer.term),
                (Reason::Unspecified, &[id][..], 1),
                "{phase:?}"
            );
        }
        assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 2);
        // The plan's pair half is decided at its prefill, both sides in
        // one answer.
        let answer = bed.ask(phased(frame(2, 77), proto::Phase::Prefill)).await;
        assert_eq!(sides_of(&answer).len(), 2, "{answer:?}");
        assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 2);
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
        let mut decider = deciding(&bed);
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
                let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
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
        // one counted by 64, and of a text 64 windows of 16 bytes, which a
        // text request states.
        let sized = |id, hashes, block_size| {
            let mut frame = hashed(id, hashes);
            frame
                .facts
                .as_mut()
                .expect("the case's frame has facts")
                .block_size = block_size;
            frame
        };
        let mut text = sized(6, 80, 16);
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
        assert_eq!(unbound.counts.awaited.load(Ordering::Relaxed), 0);
        // A model no member serves: the engine's answer over no candidate.
        let mut bed = leading([]);
        let frame = frame_for("model-nobody-serves", 2, 2);
        let answer = bed.ask(frame.clone()).await;
        assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
        assert_eq!(answer, bed.engines(&frame, 1));
    }

    /// While the start's document of a replica under a shared lease waits
    /// for its first lead, a request is no decision: `Busy` with no
    /// reason, so with no lease and no trace, under a count of its own. The
    /// first request behind the document's landing is decided under it.
    #[tokio::test]
    async fn a_request_ahead_of_the_starts_document_is_no_decision() {
        let mut bed = bed(None, true, (2, 8), []);
        bed.store.snapshot().spec().mark_document_awaited();
        let answer = bed.ask(frame(1, 1)).await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice(), answer.term),
            (Reason::Unspecified, &[1][..], 1)
        );
        let counted = |count: &AtomicU64| count.load(Ordering::Relaxed);
        assert_eq!(
            (
                counted(&bed.counts.awaited),
                counted(&bed.counts.pair_sides)
            ),
            (1, 0)
        );
        // The document lands, and the mark goes with the unbound start.
        bed.store
            .apply(preset("random", 1))
            .expect("above the unbound start");
        let answer = bed.ask(frame(2, 1)).await;
        assert!(matches!(outcome(&answer), Outcome::Pick(_)), "{answer:?}");
        assert_eq!(counted(&bed.counts.awaited), 1);
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
            engine::sticky_sessions::SessionVerdict,
            service::controller::lifecycle::{self, tests::a_dispatched},
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
        let pins = |bed: &Bed, session: view::Session| {
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

    /// The controller's own service counts a key's requests. Under a rule
    /// with an in-flight cap the request that finds the cap met on its
    /// key's worker is placed again and lands on a worker with fewer
    /// requests, and the key's next request returns once one of them
    /// ended.
    #[tokio::test]
    async fn the_service_hands_a_key_on_at_its_rules_cap() {
        use crate::{
            engine::sticky_sessions::SessionVerdict,
            service::controller::lifecycle::{self, tests::a_dispatched},
        };
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        rule.inflight_cap = 2;
        // The rule places a key on the worker with the fewest requests: a
        // request placed again at the cap leaves the key's worker.
        rule.assignment = crate::document::Assignment::MinLoad;
        let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
        let at = bed.script.taken;
        let verdict = |bed: &Bed, answer: &proto::DecideAnswer| {
            LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(answer).id))
                .expect("the lease is on file")
                .lease
                .session_verdict
        };
        // The key's first request claims a worker, and its gateway's
        // Dispatched frame pins the key there.
        let first = bed.ask(keyed(1, 1_001)).await;
        let owner = picked(&first);
        assert_eq!(verdict(&bed, &first), SessionVerdict::Claimed(owner));
        let hand = lifecycle::open(&bed.store, GATEWAY);
        lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&first).id), at)
            .expect("the frame reads");
        // One request of the key in flight: the second stays.
        let second = bed.ask(keyed(1, 1_002)).await;
        assert_eq!(picked(&second), owner);
        assert_eq!(verdict(&bed, &second), SessionVerdict::Pinned);
        // Two in flight on the key's worker: the third is placed again,
        // on a worker with fewer requests.
        let third = bed.ask(keyed(1, 1_003)).await;
        assert_ne!(picked(&third), owner);
        assert_eq!(verdict(&bed, &third), SessionVerdict::Respill);
        // One of the two ended: the key's next request is under the cap.
        let ended = LeaseId(lease_of(&second).id);
        let gone = ReleaseReason::ClientGone;
        assert!(LeasesStore::released(
            &*bed.store,
            GATEWAY,
            ended,
            None,
            gone
        ));
        let fourth = bed.ask(keyed(1, 1_004)).await;
        assert_eq!(picked(&fourth), owner);
        assert_eq!(verdict(&bed, &fourth), SessionVerdict::Pinned);
    }

    /// A keyed lease in flight at an apply tells no engine of its dispatch.
    /// Its record on file carries the TTL of the rule its decision was made
    /// under, and its pin of record ends by it, counted from the issue:
    /// the generation that stands keeps a key an hour and ends no such pin.
    #[tokio::test]
    async fn a_lease_in_flight_at_an_apply_ends_its_pin_by_its_own_rule() {
        use crate::{
            service::controller::lifecycle::{self, tests::a_dispatched},
            store::SessionsStore,
        };
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        let kept = u64::from(rule.ttl_s) * 1_000_000;
        // The generation that stands at the dispatch keeps a key an hour.
        rule.ttl_s = 3_600;
        document.generation = 2;
        let later = applied(&document);
        let mut bed = bed(Some(preset("manual", 1)), true, (1, 8), []);
        let at = bed.script.taken;
        let answer = bed.ask(keyed(1, 1_001)).await;
        let record = LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(&answer).id))
            .expect("the lease is on file");
        assert_eq!(record.state.session_ttl_micros(), Some(kept));
        // An unkeyed decision behind it, on the scratch the keyed one left
        // its TTL on: no rule governed its pick, and its record carries
        // none.
        let answer = bed.ask(frame(2, 1_002)).await;
        let unkeyed = LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(&answer).id))
            .expect("the lease is on file");
        assert_eq!(unkeyed.state.session_ttl_micros(), None);
        bed.store.apply(later).expect("a later generation");
        let lease = record.lease;
        let hand = lifecycle::open(&bed.store, GATEWAY);
        lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease.id.0), at)
            .expect("the frame reads");
        // The decider waits on its empty queue, or is about to: it applies
        // the frame as it leaves its work, or the hand did.
        let read = || SessionsStore::lookup(&*bed.store, lease.session.get());
        settled(|| read().is_some()).await;
        let pin = read().expect("a pin");
        assert_eq!(pin.ends, Some(lease.issued_micros + kept));
    }

    /// The store's pin is the key's worker before its own end and for less
    /// than the rule's TTL since its last use: the engine's table forgets
    /// a key at its TTL to the microsecond, and the pin of record is no
    /// older a pin.
    #[test]
    fn a_stored_pin_as_old_as_the_rules_ttl_is_no_pin() {
        use super::StorePins;
        use crate::{
            engine::view::{PinSource, Session},
            store::SessionsStore,
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

    /// What the engine told the stage of the decode-later bed of leases:
    /// the side and the worker of an issue, and of a release with its
    /// reason.
    pub(crate) type SideCalls = Vec<(Side, WorkerId, Option<ReleaseReason>)>;

    /// The `best` picker, keeping what the engine tells it of leases by
    /// their side.
    #[derive(Debug)]
    struct ToldSides(Arc<Mutex<SideCalls>>);

    impl Lifecycle for ToldSides {
        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            self.0.lock().push((lease.side, lease.booked, None));
        }

        fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
            self.0.lock().push((lease.side, lease.booked, Some(reason)));
        }
    }

    impl Picker for ToldSides {
        fn name(&self) -> &'static str {
            "told-sides"
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
            Best.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// The bed of the decode-later cases: a pair plan whose sides score by
    /// the least load, the decode's picking through [`ToldSides`], over the
    /// fleet's prefill and its decode beside one more decode. With the bed:
    /// what the engine told that stage, and the prefill.
    fn later_bed() -> (Bed, Arc<Mutex<SideCalls>>, WorkerId) {
        later_bed_of(1)
    }

    /// [`later_bed`] behind `deciders` deciders.
    fn later_bed_of(deciders: usize) -> (Bed, Arc<Mutex<SideCalls>>, WorkerId) {
        let told = Arc::new(Mutex::new(Vec::new()));
        let bed = bed(Some(later_spec(1, &told)), true, (deciders, 8), []);
        ready(&bed.store, "d1", MODEL, Role::Decode, WireModes::NONE);
        let prefill = WorkersStore::by_key(&*bed.store, "http://p0:8000");
        let prefill = prefill.expect("the fleet's prefill").id;
        (bed, told, prefill)
    }

    /// The document of [`later_bed`] as `generation`, whose decode side
    /// keeps what its engine tells it in `told`.
    pub(crate) fn later_spec(generation: u64, told: &Arc<Mutex<SideCalls>>) -> AppliedSpec {
        later_spec_with(generation, told, |_| {})
    }

    /// [`later_spec`] with `edit` made to its document ahead of its check.
    fn later_spec_with(
        generation: u64,
        told: &Arc<Mutex<SideCalls>>,
        edit: impl FnOnce(&mut PolicySpec),
    ) -> AppliedSpec {
        let kept = Arc::clone(told);
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker("told-sides", move || Box::new(ToldSides(Arc::clone(&kept))))
            .expect("a free name");
        let mut document: PolicySpec = serde_json::from_str(
            r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": {
                "side": { "score": ["least-load"], "pick": "best" },
                "told": { "score": ["least-load"], "pick": "told-sides" }
              },
              "bindings": [
                { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
                  "profiles": { "prefill": "side", "decode": "told" } },
                { "profile": "side" }
              ]
            }"#,
        )
        .expect("a document of this build");
        document.generation = generation;
        edit(&mut document);
        let digest = document.digest().expect("the document digests");
        AppliedSpec::validate_digested(&document, digest, &catalog, &[])
            .expect("the catalog holds the stage")
    }

    /// A pair's one request under `id`, whose decode is asked for after
    /// its prefill where `later`.
    pub(crate) fn pair_frame(id: u8, later: bool) -> proto::DecideRequest {
        let mut frame = phased(frame(id, 77), proto::Phase::Prefill);
        frame.decode_later = later;
        frame
    }

    /// The decode call of the pair's lease `lease`, for the request `id`.
    pub(crate) fn call(id: u8, lease: u64) -> proto::DecideRequest {
        let mut frame = pair_frame(id, false);
        frame.decode_of = Some(lease);
        frame
    }

    /// A dispatch of `side` to the pick's first target.
    pub(crate) fn sent(side: Side) -> Dispatch {
        Dispatch {
            side,
            target_index: 0,
            since_decision_micros: 0,
        }
    }

    /// A pair asked with its decode later, and its prefill dispatched: the
    /// lease's id and the first pick's two decodes, the chosen one first.
    /// A second pair asked whole then books the chosen decode, so the
    /// least load sends a call to the other: that pair's lease comes last.
    async fn held(bed: &mut Bed) -> (u64, [WorkerId; 2], u64) {
        let answer = bed.ask(pair_frame(1, true)).await;
        let (targets, sides) = pair_of(&answer);
        let id = sides.first().expect("the prefill's lease").0;
        let decodes = [0, 1].map(|at| WorkerId(targets[at].1.expect("a pair's target")));
        // One seed, the same decode rows: the whole pair takes the same.
        let whole = bed.ask(pair_frame(2, false)).await;
        let (targets, sides) = pair_of(&whole);
        assert_eq!(targets[0].1, Some(decodes[0].0));
        let whole = sides.first().expect("the whole pair's lease").0;
        let store: &dyn LeasesStore = bed.store.as_ref();
        let at = bed.script.taken;
        assert!(store.dispatched(GATEWAY, LeaseId(id), sent(Side::Prefill), at));
        (id, decodes, whole)
    }

    /// The decode side of `lease` as the file holds it.
    fn decode_on_file(bed: &Bed, lease: u64) -> (LeaseState, WorkerId) {
        let sides = bed.store.sides(GATEWAY, LeaseId(lease));
        let decode = sides.and_then(|sides| sides.decode);
        let decode = decode.expect("the decode side on file");
        (decode.state, decode.target)
    }

    /// A served pair asked with its decode later is issued its prefill
    /// alone: the answer carries one lease, and the pick's decode stands
    /// chosen on file and books nobody. The same frame without the word
    /// carries a lease a side, as before.
    #[tokio::test]
    async fn a_served_pair_asked_with_its_decode_later_issues_the_prefill_alone() {
        let (mut bed, told, prefill) = later_bed();
        let answer = bed.ask(pair_frame(1, true)).await;
        let (targets, sides) = pair_of(&answer);
        let id = sides.first().map_or(0, |side| side.0);
        assert_eq!(sides, [(id, proto::Side::Prefill, 0)]);
        let decode = WorkerId(targets[0].1.expect("a pair's target"));
        assert_eq!(targets[0].0, prefill.0);
        assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, decode));
        assert_eq!(pair_counts(&bed.store, [prefill, decode]), [1, 0]);
        assert_eq!(*told.lock(), [(Side::Prefill, prefill, None)]);
        // Without the word: two leases of one id, and the decode booked.
        let answer = bed.ask(pair_frame(2, false)).await;
        let (targets, sides) = pair_of(&answer);
        let whole = sides.first().map_or(0, |side| side.0);
        let both = [proto::Side::Prefill, proto::Side::Decode];
        assert_eq!(sides, both.map(|side| (whole, side, 0)));
        assert_eq!(targets[0], (prefill.0, Some(decode.0)));
        assert_eq!(pair_counts(&bed.store, [prefill, decode]), [2, 1]);
    }

    /// The decode call answers a decode pick under the pair's own lease:
    /// decided on the counts of its instant, its targets name the held
    /// prefill, and the new pick's decode is booked in the first's place.
    #[tokio::test]
    async fn a_served_decode_call_answers_a_decode_pick_under_the_same_lease() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        assert_eq!(pair_counts(&bed.store, [prefill, chosen, other]), [2, 1, 0]);
        let answer = bed.ask(call(1, id)).await;
        let (targets, sides) = pair_of(&answer);
        assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
        // The chosen decode took a lease since: the call picks the other.
        assert_eq!(targets[0], (prefill.0, Some(other.0)));
        assert!(targets.iter().all(|target| target.0 == prefill.0));
        assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, other));
        assert_eq!(pair_counts(&bed.store, [prefill, chosen, other]), [2, 1, 1]);
        assert_eq!(told.lock().last(), Some(&(Side::Decode, other, None)));
        assert_eq!(asked_again(&bed), [0, 0]);
    }

    /// A served decode call's kept record names the pair binding the call
    /// decided under, whatever phase its frame states. A binding of the
    /// decode phase stands ahead of the pair's here: it decides a decode
    /// asked alone, and no pair's.
    #[tokio::test]
    async fn a_served_decode_calls_record_names_its_pairs_binding() {
        use crate::{
            document::spec::{BindingSpec, Match},
            engine::view::Phase,
        };

        let (mut bed, told, _) = later_bed();
        let sampled = later_spec_with(2, &told, |document| {
            let ahead = BindingSpec {
                matcher: Match {
                    phase: vec![Phase::Decode],
                    ..Match::default()
                },
                profile: Some("side".into()),
                ..BindingSpec::default()
            };
            document.bindings.insert(0, ahead);
            document.explain.sample = 1;
        });
        bed.store.apply(sampled).expect("a later generation");
        let (id, _, _) = held(&mut bed).await;
        let answer = bed.ask(phased(call(1, id), proto::Phase::Decode)).await;
        assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
        let record = bed.deciders.explain(answer.term, GATEWAY, &[1]);
        // The pair's binding is the document's second.
        assert_eq!(record.expect("the sampled call").binding, Some(1));
    }

    /// Each refusal of the call leaves as a `Busy` of its own reason,
    /// never as a decision, and moves nothing: no lease under the id, a
    /// pair filed whole, a prefill not yet sent, and a side a call issued.
    #[tokio::test]
    async fn each_refusal_of_the_call_is_a_busy_of_its_own_reason_and_moves_nothing() {
        let (mut bed, told, prefill) = later_bed();
        let lease = |answer: &proto::DecideAnswer| sides_of(answer)[0].0;
        let later = lease(&bed.ask(pair_frame(1, true)).await);
        let whole = lease(&bed.ask(pair_frame(2, false)).await);
        let store = Arc::clone(&bed.store);
        let decodes = ["d0", "d1"].map(|name| {
            let row = WorkersStore::by_key(&*store, &format!("http://{name}:8000"));
            row.expect("a decode of the fleet").id
        });
        let file = || {
            let sides = [later, whole].map(|id| store.sides(GATEWAY, LeaseId(id)));
            let counts = pair_counts(&store, [prefill, decodes[0], decodes[1]]);
            (sides, counts, told.lock().len())
        };
        let rows = [
            (later + 100, Reason::DecodeNoLease),
            (whole, Reason::DecodeNotChosen),
            (later, Reason::DecodePrefillNotHeld),
        ];
        for (id, reason) in rows {
            let before = file();
            assert_eq!(busy(&bed.ask(call(1, id)).await), reason);
            assert_eq!(file(), before, "{reason:?}");
        }
        // The fourth: a side a first call issued stands.
        let leases: &dyn LeasesStore = store.as_ref();
        let at = bed.script.taken;
        assert!(leases.dispatched(GATEWAY, LeaseId(later), sent(Side::Prefill), at));
        let issued = bed.ask(call(1, later)).await;
        assert!(matches!(outcome(&issued), Outcome::Pick(_)));
        let before = file();
        assert_eq!(busy(&bed.ask(call(1, later)).await), Reason::DecodeStands);
        assert_eq!(file(), before);
        // Each is counted at its own reason, once.
        let counted = |reason: Reason| bed.counts.busy[reason as usize].load(Ordering::Relaxed);
        let reasons = [
            Reason::DecodeNoLease,
            Reason::DecodeNotChosen,
            Reason::DecodePrefillNotHeld,
            Reason::DecodeStands,
        ];
        assert_eq!(reasons.map(counted), [1; 4]);
    }

    /// A call that sheds is a decision, as any shed is: nothing is issued
    /// and the side stays chosen with the first pick's targets. A dispatch
    /// then books the first pick's decode.
    #[tokio::test]
    async fn a_shed_at_the_call_is_a_decision_and_leaves_the_side_chosen() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let before = store.sides(GATEWAY, LeaseId(id));
        let heard = told.lock().len();
        let answer = bed.ask(constrained(call(1, id), 0, Some(0))).await;
        let Outcome::Shed(shed) = outcome(&answer) else {
            panic!("a shed: {answer:?}");
        };
        assert_eq!(shed.emptied_by, "subset-hint");
        assert!(sides_of(&answer).is_empty());
        assert_eq!(store.sides(GATEWAY, LeaseId(id)), before);
        assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 0]);
        assert_eq!(told.lock().len(), heard);
        let leases: &dyn LeasesStore = store.as_ref();
        let at = bed.script.taken;
        assert!(leases.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 2, 0]);
    }

    /// A decode issued under a term that ended returns to chosen. The
    /// answer leaves as `Busy`, the lease stands as the call found it with
    /// the first pick's targets, the new pick's booking is freed and the
    /// engine hears that issue's release once. A dispatch then books the
    /// first pick's decode.
    #[tokio::test]
    async fn a_decode_issued_under_a_term_that_ended_returns_to_chosen() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let before = store.sides(GATEWAY, LeaseId(id));
        // The call's first read finds the fence open; its read at the
        // hand-over is past the renew deadline.
        let steps = [Step::At(Duration::ZERO), Step::At(FENCED)];
        bed.script.steps.lock().extend(steps);
        let answer = bed.ask(call(1, id)).await;
        assert_eq!(busy(&answer), Reason::NotLeader);
        assert_eq!(store.sides(GATEWAY, LeaseId(id)), before);
        assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 0]);
        let abandoned = Some(ReleaseReason::Abandoned);
        let heard = told.lock().clone();
        let issue = [
            (Side::Decode, other, None),
            (Side::Decode, other, abandoned),
        ];
        assert_eq!(heard[heard.len() - 2..], issue);
        let leases: &dyn LeasesStore = store.as_ref();
        let at = bed.script.taken;
        assert!(leases.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 2, 0]);
    }

    /// A frame that says neither word decides as before: a single is a
    /// pick under one lease, and a pair one answer under two leases of one
    /// id with both sides issued and booked.
    #[tokio::test]
    async fn a_frame_with_neither_field_decides_as_before() {
        let (mut bed, _, prefill) = later_bed();
        let plain = frame(1, 77);
        assert_eq!((plain.decode_later, plain.decode_of), (false, None));
        let answer = bed.ask(plain).await;
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        assert_eq!(sides_of(&answer).len(), 1);
        let answer = bed.ask(pair_frame(2, false)).await;
        let (targets, sides) = pair_of(&answer);
        let id = sides.first().map_or(0, |side| side.0);
        let both = [proto::Side::Prefill, proto::Side::Decode];
        assert_eq!(sides, both.map(|side| (id, side, 0)));
        let decode = WorkerId(targets[0].1.expect("a pair's target"));
        assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, decode));
        assert_eq!(pair_counts(&bed.store, [prefill, decode]), [1, 1]);
        assert_eq!(asked_again(&bed), [0, 0]);
    }

    /// A repeat of the call frees the lost issue and issues the side anew
    /// under the same lease id. The word takes back the first call's
    /// issue, ahead of the repeat's own decision, and the engine hears
    /// that issue's release as abandoned.
    #[tokio::test]
    async fn a_repeat_of_the_call_frees_the_lost_issue_and_issues_the_side_anew() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let first = bed.ask(call(1, id)).await;
        assert_eq!(pair_of(&first).0[0].1, Some(other.0));
        // The loads move while the answer is lost: two more requests on
        // the decode the first call issued.
        for extra in [1, 2] {
            file(&store, extra, other, bed.script.taken);
        }
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 3]);
        let repeat = bed.ask(again(call(1, id))).await;
        let (targets, sides) = pair_of(&repeat);
        assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
        assert_eq!(targets[0], (prefill.0, Some(chosen.0)));
        assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, chosen));
        // The first call's decode gave its booking back, and the repeat's
        // took one.
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 2, 2]);
        let abandoned = Some(ReleaseReason::Abandoned);
        let heard = told.lock().clone();
        let words = [
            (Side::Decode, other, abandoned),
            (Side::Decode, chosen, None),
        ];
        assert_eq!(heard[heard.len() - 2..], words);
        assert_eq!(asked_again(&bed), [0, 1]);
    }

    /// A repeat that sheds leaves the side chosen, as its first decision
    /// filed it: the word took the lost issue back ahead of the repeat's
    /// decision, whatever its answer. The gateway then sends the decode its
    /// first pick named, and that dispatch books the side.
    #[tokio::test]
    async fn a_repeat_that_sheds_leaves_the_side_chosen_for_the_first_picks_dispatch() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let lease = LeaseId(id);
        let filed = store.sides(GATEWAY, lease);
        bed.ask(call(1, id)).await;
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 1]);
        assert_ne!(store.sides(GATEWAY, lease), filed);
        let repeat = again(constrained(call(1, id), 0, Some(0)));
        let answer = bed.ask(repeat).await;
        assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
        // The lease stands as its first decision filed it, both its parts:
        // the first pick's targets and the lease's dispatch deadline.
        assert_eq!(store.sides(GATEWAY, lease), filed);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 0]);
        let abandoned = Some(ReleaseReason::Abandoned);
        assert_eq!(told.lock().last(), Some(&(Side::Decode, other, abandoned)));
        assert_eq!(asked_again(&bed), [0, 1]);
        // The first pick's decode goes out: its dispatch books the side.
        let leases: &dyn LeasesStore = store.as_ref();
        let at = bed.script.taken;
        assert!(leases.dispatched(GATEWAY, lease, sent(Side::Decode), at));
        let sent_to = (LeaseState::Dispatched { at }, chosen);
        assert_eq!(decode_on_file(&bed, id), sent_to);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 2, 0]);
    }

    /// A repeat on a side still chosen is a first call: no call issued the
    /// side, and the word ends nothing.
    #[tokio::test]
    async fn a_repeat_on_a_chosen_side_is_a_first_call() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let heard = told.lock().len();
        let answer = bed.ask(again(call(1, id))).await;
        let (targets, sides) = pair_of(&answer);
        assert_eq!(sides, [(id, proto::Side::Decode, 0)]);
        assert_eq!(targets[0].1, Some(other.0));
        assert_eq!(pair_counts(&bed.store, [prefill, chosen, other]), [2, 1, 1]);
        assert_eq!(told.lock()[heard..], [(Side::Decode, other, None)]);
        assert_eq!(asked_again(&bed), [1, 0]);
    }

    /// A repeat on a side its gateway named a dispatch for is refused as
    /// standing: the word ends no side that was sent.
    #[tokio::test]
    async fn a_repeat_on_a_dispatched_side_is_refused_as_standing() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        bed.ask(call(1, id)).await;
        let leases: &dyn LeasesStore = store.as_ref();
        let at = bed.script.taken;
        assert!(leases.dispatched(GATEWAY, LeaseId(id), sent(Side::Decode), at));
        let file = || (store.sides(GATEWAY, LeaseId(id)), told.lock().len());
        let before = file();
        let answer = bed.ask(again(call(1, id))).await;
        assert_eq!(busy(&answer), Reason::DecodeStands);
        assert_eq!(file(), before);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 1]);
        assert_eq!(asked_again(&bed), [1, 0]);
    }

    /// A decode frame runs no search by its id's fingerprint, with the
    /// word or without: another request of the gateway under the same id
    /// keeps the lease it waits under.
    #[tokio::test]
    async fn a_decode_frame_runs_no_search_by_its_ids_fingerprint() {
        let (mut bed, _, _) = later_bed();
        let (id, _, _) = held(&mut bed).await;
        // A single of the gateway under the id the call's frame carries:
        // its lease waits for a dispatch.
        let single = bed.ask(frame(9, 77)).await;
        let waits = LeaseId(lease_of(&single).id);
        let answer = bed.ask(again(call(9, id))).await;
        assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
        let leases: &dyn LeasesStore = bed.store.as_ref();
        assert!(leases.get(GATEWAY, waits).is_some(), "the single's lease");
        assert_eq!(asked_again(&bed), [1, 0]);
    }

    /// The issue that stands of `lease`'s decode side, taken back by the
    /// case's own hand once the bed's decider left its work: a thread acts
    /// at a file where nobody is at work, and beside nobody.
    async fn taken_back_by_hand(store: &MemoryStore, lease: LeaseId) {
        let abandoned = ReleaseReason::Abandoned;
        settled(|| store.take_back_decode(GATEWAY, lease, abandoned)).await;
    }

    /// One decode call's order by the case's own hand, at the bed's first
    /// instant: the issue its decider would undo.
    fn issued_by_hand(bed: &Bed, frame: proto::DecideRequest) -> DecodeIssue {
        let request = checked(frame);
        let now = bed.script.taken;
        let taken = DecisionCall::at(&request, GATEWAY, now, now).expect("budget is left");
        let (mut buffers, mut undo) = (Buffers::default(), None);
        let part = Part::default();
        let lease = request.decode_of().expect("a decode call's frame");
        let decided = bed.store.deciding(|world| {
            decode_side(
                &bed.store,
                world,
                &taken,
                lease,
                &mut buffers,
                &part,
                &mut undo,
            )
        });
        assert!(decided.is_ok(), "a decode pick");
        undo.expect("a pick's issue")
    }

    /// An undo undoes its own issue alone, under one reading of the clock.
    /// Where the word took that issue back, and where a repeat issued the
    /// side since, the first call's undo moves nothing and frees nothing:
    /// the place counts the side's issues, and an undo names its own.
    #[tokio::test]
    async fn an_undo_behind_a_repeats_issue_moves_nothing() {
        let (mut bed, _, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let lease = LeaseId(id);
        let file = || {
            let counts = pair_counts(&store, [prefill, chosen, other]);
            (store.sides(GATEWAY, lease), counts)
        };
        let first = issued_by_hand(&bed, call(1, id));
        taken_back_by_hand(&store, lease).await;
        let ended = file();
        assert!(!store.unissue_decode(GATEWAY, lease, first));
        assert_eq!(file(), ended);
        issued_by_hand(&bed, again(call(1, id)));
        let issued = file();
        assert_eq!(issued.1, [2, 1, 1]);
        assert!(!store.unissue_decode(GATEWAY, lease, first));
        assert_eq!(file(), issued);
    }

    /// An undo behind the prefill's end takes the lease off the file: the
    /// side would stand chosen beside a prefill that ended, and no later
    /// event ends a side that is not live.
    #[tokio::test]
    async fn an_undo_behind_its_prefills_end_takes_the_lease_off_the_file() {
        let (mut bed, _, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let lease = LeaseId(id);
        let issue = issued_by_hand(&bed, call(1, id));
        let leases: &dyn LeasesStore = store.as_ref();
        let gone = ReleaseReason::ClientGone;
        assert!(leases.released(GATEWAY, lease, Some(Side::Prefill), gone));
        assert!(store.sides(GATEWAY, lease).is_some(), "the issued decode");
        assert!(store.unissue_decode(GATEWAY, lease, issue));
        assert_eq!(store.sides(GATEWAY, lease), None);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [1, 1, 0]);
    }

    /// A decode call's answer that its stream does not take gives its one
    /// issue back, and no more of the lease: the prefill that runs under
    /// the same id keeps its booking. The side stands chosen with the
    /// first pick's targets, the engine hears the issue's release as
    /// abandoned, and the first pick's dispatch then books the side.
    #[tokio::test]
    async fn a_decode_calls_answer_its_stream_does_not_take_gives_its_issue_back() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let lease = LeaseId(id);
        let filed = store.sides(GATEWAY, lease);
        // A stream that ended takes no answer.
        let (sender, answers) = mpsc::channel(1);
        drop(answers);
        let job = Job {
            request: checked(call(1, id)),
            gateway: GATEWAY,
            received: bed.script.taken,
            answers: sender,
        };
        assert!(bed.deciders.submit(job).is_none());
        // One decider decides a model's requests in order: it is past the
        // call, and past its answer's give-up, when it answers the next.
        bed.ask(frame(9, 9)).await;
        assert_eq!(store.sides(GATEWAY, lease), filed);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 0]);
        let abandoned = (Side::Decode, other, Some(ReleaseReason::Abandoned));
        assert!(told.lock().contains(&abandoned), "{:?}", told.lock());
        let leases: &dyn LeasesStore = store.as_ref();
        let at = bed.script.taken;
        assert!(leases.dispatched(GATEWAY, lease, sent(Side::Decode), at));
        let sent_to = (LeaseState::Dispatched { at }, chosen);
        assert_eq!(decode_on_file(&bed, id), sent_to);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 2, 0]);
    }

    /// A decode side's answer of `lease` as its stream's queue holds it,
    /// handed over with `beside` at `at`.
    fn queued(lease: LeaseId, beside: wire::Beside, at: Instant) -> wire::Leased {
        wire::Leased {
            lease: Lease {
                id: lease,
                side: Side::Decode,
                ..Lease::default()
            },
            beside,
            left: DISPATCH_TTL,
            due: at + DISPATCH_TTL,
        }
    }

    /// A side takes as many issues as its count holds and no more. The
    /// call behind the last is refused as a decode that stands, with the
    /// word or without, and nothing moves: no count repeats, so an undo
    /// names one issue for the lease's life. The first pick's dispatch
    /// still books the side.
    #[tokio::test]
    async fn a_side_whose_count_of_issues_is_full_takes_no_more_calls() {
        let (mut bed, _, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let lease = LeaseId(id);
        let file = || {
            let counts = pair_counts(&store, [prefill, chosen, other]);
            (store.sides(GATEWAY, lease), counts)
        };
        let filed = file();
        for _ in 0..u8::MAX {
            issued_by_hand(&bed, call(1, id));
            taken_back_by_hand(&store, lease).await;
        }
        assert_eq!(file(), filed);
        for full in [call(1, id), again(call(1, id))] {
            assert_eq!(busy(&bed.ask(full).await), Reason::DecodeStands);
        }
        assert_eq!(file(), filed);
        let leases: &dyn LeasesStore = store.as_ref();
        let at = bed.script.taken;
        assert!(leases.dispatched(GATEWAY, lease, sent(Side::Decode), at));
        let sent_to = (LeaseState::Dispatched { at }, chosen);
        assert_eq!(decode_on_file(&bed, id), sent_to);
    }

    /// A decode call's answer given up while a decider is at work at the
    /// lease's file moves nothing and tells nothing on the thread that
    /// gives it up: the word waits in the file's ring. The decider applies
    /// it at its next take, ahead of its own decision: the issue goes back
    /// and the engine hears its release there, so no stage hears a word of
    /// a lease beside its decider at work.
    #[tokio::test]
    async fn an_answer_given_up_beside_a_decider_at_work_is_left_to_that_decider() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let lease = LeaseId(id);
        let filed = store.sides(GATEWAY, lease);
        let issue = issued_by_hand(&bed, call(1, id));
        let answer = queued(lease, wire::Beside::Issue(issue), bed.script.taken);
        let heard = told.lock().len();
        // The next decision is held at its first read of the clock: its
        // decider has the job at hand, at the lease's file.
        let (reached, held) = std_mpsc::sync_channel(1);
        let (go, word) = std_mpsc::sync_channel(1);
        let hold = Step::Hold { reached, go: word };
        bed.script.steps.lock().push_back(hold);
        assert_eq!(bed.submit(frame(9, 9)), None);
        held.recv_timeout(PATIENCE)
            .expect("the decider reads the clock");
        let hand = lifecycle::open(&store, GATEWAY);
        let late = ReleaseReason::LateDecision;
        // A take-back in place is refused beside the decider at work.
        assert!(!store.take_back_decode(GATEWAY, lease, late));
        answer.given_up(&store, &hand, late);
        assert_eq!(told.lock().len(), heard, "left to the decider at work");
        go.send(()).expect("the decider holds the other end");
        bed.answer().await;
        assert_eq!(store.sides(GATEWAY, lease), filed);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 0]);
        let released = (Side::Decode, other, Some(late));
        assert_eq!(told.lock().get(heard), Some(&released));
        store.hand_back(hand);
    }

    /// A decode call is decided at its lease's file, whatever model it
    /// names: its job goes to the decider that issued the lease, not to
    /// the decider of the model the call names. A repeat under such a
    /// name takes the call's issue back there, and the first pick's decode
    /// stands chosen where the named model has no pair.
    #[tokio::test]
    async fn a_decode_call_is_decided_by_its_leases_decider_whatever_model_it_names() {
        let (mut bed, _, prefill) = later_bed_of(4);
        let (id, [chosen, other], _) = held(&mut bed).await;
        let home = format!("decide-{}", bed.deciders.of(MODEL));
        let elsewhere = (0..64)
            .map(|index| format!("model-{index}"))
            .find(|model| bed.deciders.of(model) != bed.deciders.of(MODEL))
            .expect("some model of sixty-four has another key");
        let answer = bed.ask(call(1, id)).await;
        assert_eq!(sides_of(&answer), [(id, proto::Side::Decode, 0)]);
        assert_eq!(decode_on_file(&bed, id), (LeaseState::ISSUED, other));
        bed.script.readers.lock().clear();
        let mut repeat = again(call(1, id));
        let facts = repeat.facts.as_mut().expect("the case's frame has facts");
        facts.model = elsewhere;
        bed.ask(repeat).await;
        let readers = mem::take(&mut *bed.script.readers.lock());
        assert!(
            !readers.is_empty() && readers.iter().all(|reader| *reader == home),
            "{readers:?}"
        );
        assert_eq!(asked_again(&bed), [0, 1]);
        assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
        assert_eq!(pair_counts(&bed.store, [prefill, chosen, other]), [2, 1, 0]);
    }

    /// What an answer that does not leave gives up, by what its lease is
    /// handed over with. A decode call's answer takes back its own issue
    /// alone, by its count: behind a repeat's issue it moves nothing. Any
    /// other answer's lease is released whole, both its sides.
    #[tokio::test]
    async fn an_answer_given_up_takes_back_its_own_issue_or_releases_its_lease_whole() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let lease = LeaseId(id);
        let filed = store.sides(GATEWAY, lease);
        let file = || {
            let counts = pair_counts(&store, [prefill, chosen, other]);
            (store.sides(GATEWAY, lease), counts)
        };
        let at = bed.script.taken;
        let answer = |beside| queued(lease, beside, at);
        let hand = lifecycle::open(&store, GATEWAY);
        let late = ReleaseReason::LateDecision;
        // The first call's answer, given up behind a repeat's issue.
        let first = answer(wire::Beside::Issue(issued_by_hand(&bed, call(1, id))));
        taken_back_by_hand(&store, lease).await;
        let repeat = answer(wire::Beside::Issue(issued_by_hand(&bed, call(1, id))));
        let issued = file();
        assert_eq!(issued.1, [2, 1, 1]);
        first.given_up(&store, &hand, late);
        assert_eq!(file(), issued);
        // The repeat's answer, given up: its own issue goes back. Its word
        // is applied by the hand, or by a decider that leaves its work.
        repeat.given_up(&store, &hand, late);
        settled(|| file() == (filed, [2, 1, 0])).await;
        assert_eq!(told.lock().last(), Some(&(Side::Decode, other, Some(late))));
        // A pair's first answer, given up: the lease leaves whole.
        let whole = answer(wire::Beside::Decode(view::Session::NONE));
        whole.given_up(&store, &hand, late);
        store.hand_back(hand);
        assert_eq!(file(), (None, [1, 1, 0]));
    }

    /// A give-up that names a chosen decode side ends it: a lost call that
    /// runs late behind the give-up is refused as standing, with the word
    /// or without, and books nobody.
    #[tokio::test]
    async fn a_give_up_that_names_a_chosen_side_ends_it() {
        let (mut bed, _, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let leases: &dyn LeasesStore = store.as_ref();
        let abandoned = ReleaseReason::Abandoned;
        assert!(leases.released(GATEWAY, LeaseId(id), Some(Side::Decode), abandoned));
        let ended = (LeaseState::Released(abandoned), chosen);
        assert_eq!(decode_on_file(&bed, id), ended);
        for late in [call(1, id), again(call(1, id))] {
            assert_eq!(busy(&bed.ask(late).await), Reason::DecodeStands);
        }
        assert_eq!(decode_on_file(&bed, id), ended);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 0]);
    }

    /// A decode call whose model a reload bound to no pair plan is a
    /// side's request still: it is not decided as a single. Its decision
    /// sheds, nothing is issued and the side stays chosen.
    #[tokio::test]
    async fn a_decode_call_behind_a_reload_to_no_pair_plan_is_no_single() {
        let (mut bed, _, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let reloaded = store.apply(preset("random", 2));
        reloaded.expect("a later generation");
        let live = LeasesStore::live(&*store, GATEWAY);
        let answer = bed.ask(call(1, id)).await;
        assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
        assert!(sides_of(&answer).is_empty());
        assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
        assert_eq!(LeasesStore::live(&*store, GATEWAY), live);
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 0]);
    }

    /// A call's issue taken back behind a reload is told to no engine. The
    /// engine of the call's instant heard no issue of the lease, whose
    /// generation is the first decision's: it hears no release of it.
    #[tokio::test]
    async fn a_decode_taken_back_behind_a_reload_is_told_to_no_engine() {
        let (mut bed, told, prefill) = later_bed();
        let (id, [chosen, other], _) = held(&mut bed).await;
        let store = Arc::clone(&bed.store);
        let before = told.lock().len();
        let reloaded = Arc::new(Mutex::new(Vec::new()));
        let applied = store.apply(later_spec(2, &reloaded));
        applied.expect("a later generation");
        let steps = [Step::At(Duration::ZERO), Step::At(FENCED)];
        bed.script.steps.lock().extend(steps);
        let answer = bed.ask(call(1, id)).await;
        assert_eq!(busy(&answer), Reason::NotLeader);
        assert_eq!(decode_on_file(&bed, id), (LeaseState::Chosen, chosen));
        assert_eq!(pair_counts(&store, [prefill, chosen, other]), [2, 1, 0]);
        assert_eq!(*reloaded.lock(), []);
        assert_eq!(told.lock().len(), before);
    }
}
