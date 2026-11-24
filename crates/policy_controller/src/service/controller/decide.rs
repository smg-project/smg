//! The `Decide` call (design 5.2): one stream a gateway runtime thread, a
//! `DecideRequest` in, a decision, a `Busy` or a `Credit` out, the term on
//! every answer.
//!
//! One invariant: **a stream holds nothing a reconnect must find.** Its
//! credit, its queue of answers and its place among its gateway's streams
//! end with it, and so does the lease of each decision still in that
//! queue: no gateway will read one, so the stream gives its lease up at
//! its end, as it does for a decision it stops at the send. Its hand goes
//! back to the store there, so what it handed is applied as it leaves.
//!
//! Where the replica leads, a stream is a present gateway's
//! ([`Gateways`](super::gateways::Gateways)): it is refused at its open
//! when no session of the gateway stands or stood within the grace period,
//! and it ends, in the same words, when the sweep lets the gateway go. A
//! session that ends changes nothing for a stream inside the grace period.
//!
//! A stream ends with the listeners: at the drain's word it ends
//! `UNAVAILABLE`, so a gateway that holds its streams open does not hold a
//! drain to its grace period.
//!
//! Both ends reach a stream from outside it, and a poll reads each by one
//! load. What wakes a sleeping stream at them is polled only to hold its
//! waker: once a stream, once more for each place it takes, and again
//! under another waker. Those polls run outside the runtime's budget, so
//! each one registers the waker.
//!
//! A stream's place in that table is of one term. A replica that does not
//! lead has no session and no table to ask: a stream opens there, and each
//! request on it is answered `Busy` for the leader that is elsewhere. The
//! first request the replica would decide, once it leads, asks the table
//! as an open does, and so does the first one after the term under a
//! stream changed. No decision asks otherwise.
//!
//! A decision leaves under the fence and inside its lease's dispatch span
//! (design 5.3): it waits in its stream's queue for as long as the gateway
//! reads nothing, so the stream reads both where it sends one, at one read
//! of the clock. Past the term a decision was made under, `Busy` for the
//! leader that is elsewhere leaves in its place, and with under a
//! millisecond of the span left, the `Busy` a late decision is told. The
//! lease of a decision stopped there does not wait for its dispatch
//! deadline: the stream gives it up, by a release handed to the leases'
//! file as a `Released` frame's is ([`MemoryStore::hand`]). A decision
//! waits in the queue as it was made ([`wire::Decided`]), and its frame is
//! built at the send, on the thread that encodes it, with what the span
//! has left there.
//!
//! The reader is the stream itself and never waits for a decision: a frame
//! is stamped with its receipt, checked ([`Checked`]) and handed to its
//! model's decider ([`DecisionSlots`](super::decision_slots::DecisionSlots)),
//! or answered here with `Busy`: for no credit, for a replica that does not
//! lead, for a budget of 0, for the queue's depth, or, with no reason, where
//! the decider's thread ended. Every `Busy` is counted at its reason
//! ([`DecideStats`](super::decision_slots::DecideStats)).
//!
//! Credits ride the stream: it starts with [`WINDOW`], a request spends
//! one, and a `Credit` frame grants half a window each time that many
//! answers left. Credit is never taken back. A request sent with none is
//! answered `Busy` at once and spends none.
//!
//! What a stream holds is bounded by its window: its deciders' answers
//! wait in a queue of [`WINDOW`] places, which a request with no credit
//! never reaches, and the reader holds one answer of its own at most,
//! reading no further frame until that one left. A decision made leaves
//! before the reader's own answer. A place holds a frame of
//! [`wire::MAX_FRAME`] bytes as received at most, which the transport
//! holds it to, and as read no more than those and a record for each of
//! its headers, which the check holds to a view's 256. An answer carries
//! its request's id back, which the check holds to
//! [`wire::MAX_REQUEST_ID`]: a frame is checked before any `Busy` is said
//! of it. A gateway holds [`MAX_STREAMS`] streams at most.

use std::{
    future::{self, Future},
    mem,
    pin::Pin,
    sync::{atomic::Ordering, Arc},
    task::{Context, Poll, Waker},
};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tokio::{
    sync::{mpsc, watch},
    task::coop::unconstrained,
};
use tonic::{codegen::tokio_stream::Stream, Status, Streaming};

use super::{
    decision_slots::{Job, Refused},
    gateways::Presence,
    principal::Principal,
    service::Shared,
    session,
    wire::{self, Checked, Refusal},
};
use crate::{
    service::proto::{self, busy::Reason},
    store::{GatewayId, Hand, MemoryStore},
    tracking::lease::ReleaseReason,
};

/// The requests a decide stream may have unanswered: its credit at the
/// open, as `Welcome.decide_window` tells a gateway.
pub(crate) const WINDOW: u32 = 128;

/// The decide streams one gateway holds open at most, as
/// `Welcome.max_decide_streams` tells it.
pub(crate) const MAX_STREAMS: u32 = 256;

/// The answer to a decide stream of a gateway that is not present.
pub(crate) const NO_SESSION: &str = "no session of this gateway stands on this controller: a decide stream is a present gateway's (say Hello on a Session stream first)";

/// The last word of a decide stream that is open when the listeners close.
pub(crate) const CLOSING: &str =
    "this controller closes its listeners: open the decide stream on the leader (ask Leader)";

type Out = Result<proto::DecideAnswer, Status>;

/// Wakes a sleeping stream at an end that reaches it from outside: its
/// presence's, or the listeners' close. Resolved, it is the end itself.
///
/// It waits outside the runtime's budget. A stream holds the waker it
/// polled its two waits under, and a wait the budget refuses returns
/// pending before it registers one: a stream whose task spent its budget
/// on answers would sleep through both ends.
type Wake = Pin<Box<dyn Future<Output = ()> + Send>>;

/// What wakes a stream at the end of `presence`.
fn ended(mut presence: Presence) -> Wake {
    Box::pin(unconstrained(async move { presence.ended().await }))
}

/// The decide streams each gateway holds open, against one cap a gateway.
pub(crate) struct Streams {
    cap: u32,
    open: Mutex<FxHashMap<GatewayId, u32>>,
}

impl Streams {
    pub(crate) fn new(cap: u32) -> Self {
        Self {
            cap,
            open: Mutex::default(),
        }
    }

    /// A place for one more stream of `gateway`; `None` at the cap.
    fn place(self: &Arc<Self>, gateway: GatewayId) -> Option<Place> {
        let mut open = self.open.lock();
        let held = open.get(&gateway).copied().unwrap_or(0);
        if held >= self.cap {
            return None;
        }
        open.insert(gateway, held + 1);
        Some(Place {
            streams: Arc::clone(self),
            gateway,
        })
    }
}

/// One open stream's place among its gateway's, given back at its drop.
struct Place {
    streams: Arc<Streams>,
    gateway: GatewayId,
}

impl Drop for Place {
    fn drop(&mut self) {
        let mut open = self.streams.open.lock();
        match open.get_mut(&self.gateway) {
            Some(held) if *held > 1 => *held -= 1,
            _ => {
                open.remove(&self.gateway);
            }
        }
    }
}

/// The stream a `Decide` call answers on, and the call's reader: it is
/// polled with the answers, so no task outlives the stream.
pub(crate) struct Answers<S = Streaming<proto::DecideRequest>> {
    shared: Arc<Shared>,
    gateway: GatewayId,
    /// The stream's hand at the leases' file: a decision stopped at the
    /// send gives its lease up through it, and so does each decision still
    /// in the queue at the stream's end, where the hand is given back.
    hand: Hand,
    /// The gateway's frames; `None` once it closed its side.
    inbound: Option<S>,
    /// The deciders' answers, and the handle each request carries to its
    /// decider.
    decided: mpsc::Receiver<wire::Decided>,
    sender: mpsc::Sender<wire::Decided>,
    /// The term the stream's place in the presence table was asked under;
    /// `None` while it asked nothing, opened where the replica did not
    /// lead.
    placed: Option<u64>,
    /// The presence the stream holds, `None` while it holds none: its end
    /// is the stream's.
    presence: Option<Presence>,
    /// The listeners' close ([`Shared::stop`]): the run loop says the word
    /// once, so a change of it is the stream's end.
    stop: watch::Receiver<bool>,
    /// What wakes a sleeping stream at either end, and the waker both
    /// hold: none before the stream first sleeps, and none again once it
    /// takes a place.
    gone: Wake,
    closing: Wake,
    armed: Option<Waker>,
    /// The requests the gateway may still send.
    credit: u32,
    /// The answers that left since the last `Credit` frame, each to a
    /// request that spent a credit.
    left: u32,
    /// The requests a decider holds or whose answer waits in the queue.
    deciding: u32,
    /// The reader's own answer, beside whether its request spent a credit.
    own: Option<(proto::DecideAnswer, bool)>,
    ended: bool,
    _place: Place,
}

/// Serves one `Decide` stream of `principal`, or refuses its open.
pub(crate) fn serve<S>(
    shared: Arc<Shared>,
    principal: &Principal,
    inbound: S,
) -> Result<Answers<S>, Status> {
    let gateway = principal.gateway();
    // A receiver made after the word has seen it: a stream that opens
    // after the listeners' close is refused in an open one's last words.
    let stop = shared.stop.subscribe();
    if *stop.borrow() {
        return Err(Status::unavailable(CLOSING));
    }
    let now = shared.deciders.now();
    // The table is asked where this replica leads. One that does not lead
    // has none to ask: the stream opens, and its requests are told.
    let (placed, presence) = match shared.fence.leader_term(now) {
        Some(term) => {
            let presence = shared
                .gateways
                .presence(gateway, now)
                .ok_or_else(|| Status::failed_precondition(NO_SESSION))?;
            (Some(term), Some(presence))
        }
        None => (None, None),
    };
    let gone = presence
        .clone()
        .map_or_else(|| Box::pin(future::pending()) as Wake, ended);
    let place = shared.streams.place(gateway).ok_or_else(|| {
        Status::resource_exhausted(format!(
            "the gateway holds its cap of {} decide streams on this controller",
            shared.streams.cap
        ))
    })?;
    let mut word = stop.clone();
    let closing = Box::pin(unconstrained(async move {
        let _ = word.wait_for(|stop| *stop).await;
    }));
    let (sender, decided) = mpsc::channel(WINDOW as usize);
    let hand = Hand::new(gateway);
    Ok(Answers {
        shared,
        gateway,
        hand,
        inbound: Some(inbound),
        decided,
        sender,
        placed,
        presence,
        stop,
        gone,
        closing,
        armed: None,
        credit: WINDOW,
        left: 0,
        deciding: 0,
        own: None,
        ended: false,
        _place: place,
    })
}

/// The status a refused frame ends its stream with: the contract's code
/// for a field out of its range.
fn refused(refusal: Refusal) -> Status {
    Status::invalid_argument(refusal.to_string())
}

impl<S> Answers<S> {
    /// The stream's last word.
    fn end(&mut self, status: Status) -> Poll<Option<Out>> {
        self.ended = true;
        Poll::Ready(Some(Err(status)))
    }

    /// One answer leaves; `spent` says its request spent a credit, which a
    /// `Credit` frame then owes back.
    fn leave(&mut self, answer: proto::DecideAnswer, spent: bool) -> Poll<Option<Out>> {
        self.left += u32::from(spent);
        Poll::Ready(Some(Ok(answer)))
    }

    /// A decider's answer as it is sent: a decision leaves under the fence
    /// and inside its lease's dispatch span (design 5.3). It waited in the
    /// stream's queue for as long as its gateway read nothing, so both are
    /// read here, behind the decider's reads and at one read of the clock,
    /// and its frame is built here, on the thread that encodes it. Past
    /// the term it was made under, or with under a millisecond of the
    /// span left, `Busy` leaves in its place and the lease is given up:
    /// each under the reasons a decider gives where its own last read
    /// finds the same.
    fn fenced(&self, decided: wire::Decided) -> proto::DecideAnswer {
        let (id, term, decision, lease) = match decided {
            wire::Decided::Decision {
                id,
                term,
                decision,
                lease,
            } => (id, term, decision, lease),
            wire::Decided::Frame(frame) => return frame,
        };
        let deciders = &self.shared.deciders;
        let now = deciders.now();
        let lease = lease.map(|leased| leased.sent_at(now));
        let stopped = if self.shared.fence.leader_term(now) != Some(term) {
            Some((Reason::NotLeader, ReleaseReason::Abandoned))
        } else if lease.as_ref().is_some_and(wire::Leased::spent) {
            Some((Reason::BudgetGoneOnArrival, ReleaseReason::LateDecision))
        } else {
            None
        };
        let Some((reason, release)) = stopped else {
            return wire::decided(id, term, &decision, lease.as_ref());
        };
        if let Some(leased) = lease {
            leased.given_up(&self.shared.store, &self.hand, release);
        }
        deciders.busy(id, reason)
    }

    /// One frame of the gateway: handed to its model's decider, or answered
    /// here. `Err` ends the stream.
    fn take(&mut self, frame: proto::DecideRequest) -> Result<(), Status> {
        let deciders = &self.shared.deciders;
        let received = deciders.now();
        // A frame is checked before anything is said of it: an answer
        // carries its request's id back, and the check bounds the id.
        let request = Checked::new(frame).map_err(refused)?;
        if self.credit == 0 {
            let busy = deciders.busy(request.into_id(), Reason::NoCredit);
            self.own = Some((busy, false));
            return Ok(());
        }
        self.credit -= 1;
        // The stream's place in the presence table is of one term: the
        // first request this replica would decide under a term asks the
        // table, as an open does.
        let Some(term) = self.shared.fence.leader_term(received) else {
            let busy = deciders.busy(request.into_id(), Reason::NotLeader);
            self.own = Some((busy, true));
            return Ok(());
        };
        if self.placed != Some(term) {
            let presence = self
                .shared
                .gateways
                .presence(self.gateway, received)
                .ok_or_else(|| Status::failed_precondition(NO_SESSION))?;
            // The place has no waker yet: the stream arms it before it
            // sleeps, in the poll that took it.
            self.gone = ended(presence.clone());
            self.presence = Some(presence);
            self.armed = None;
            self.placed = Some(term);
        }
        // A budget of 0 is gone at its receipt, and so is the one a frame
        // that states none carries: told here, it takes no place of its
        // decider's queue.
        if request.budget().is_zero() {
            let busy = deciders.busy(request.into_id(), Reason::BudgetGoneOnArrival);
            self.own = Some((busy, true));
            return Ok(());
        }
        let job = Job {
            request,
            gateway: self.gateway,
            received,
            answers: self.sender.clone(),
        };
        match deciders.submit(job) {
            None => self.deciding += 1,
            Some((Refused::Full, job)) => {
                let busy = deciders.busy(job.request.into_id(), Reason::QueueDepth);
                self.own = Some((busy, true));
            }
            // The controller stops for a decider that ended; until it
            // has, the model's requests are the gateway's to decide.
            Some((Refused::Gone, job)) => {
                deciders.counts().gone.fetch_add(1, Ordering::Relaxed);
                let busy = deciders.busy(job.request.into_id(), Reason::Unspecified);
                self.own = Some((busy, true));
            }
        }
        Ok(())
    }
}

impl<S> Drop for Answers<S> {
    /// The stream's end, however it came: the lease of each decision
    /// still in its queue is given up. No gateway will read such a
    /// decision, and its lease would book its worker until the dispatch
    /// deadline. Each is released as abandoned by a word handed to the
    /// leases' file, as a decision stopped at the send is. The queue is
    /// closed first: an answer a decider makes from here on is refused
    /// there, and its decider gives that lease up. Each answer found is
    /// counted as that decider counts one: dropped, its stream gone.
    ///
    /// The hand then goes back to the store, as a session's does at its
    /// end ([`MemoryStore::hand_back`]). Where a file's decider keeps its
    /// watch a hand leaves its frame for that decider's next look. This
    /// last look applies it in place where no decider is at work: what
    /// the stream handed here, and at each send it stopped.
    fn drop(&mut self) {
        self.decided.close();
        let abandoned = ReleaseReason::Abandoned;
        let mut dropped = 0;
        while let Ok(decided) = self.decided.try_recv() {
            dropped += 1;
            if let wire::Decided::Decision {
                lease: Some(leased),
                ..
            } = decided
            {
                leased.given_up(&self.shared.store, &self.hand, abandoned);
            }
        }
        if dropped != 0 {
            let counts = self.shared.deciders.counts();
            counts.dropped_gone.fetch_add(dropped, Ordering::Relaxed);
        }
        let hand = mem::replace(&mut self.hand, Hand::new(self.gateway));
        let store: &MemoryStore = &self.shared.store;
        store.hand_back(hand);
    }
}

impl<S> Stream for Answers<S>
where
    S: Stream<Item = Result<proto::DecideRequest, Status>> + Unpin,
{
    type Item = Out;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Out>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        // The two ends from outside, one load each.
        if this.presence.as_ref().is_some_and(Presence::is_ended) {
            return this.end(Status::failed_precondition(NO_SESSION));
        }
        if this.stop.has_changed().unwrap_or(true) {
            return this.end(Status::unavailable(CLOSING));
        }
        // Half a window of answers left: the credit their requests spent
        // goes back in one frame.
        let half = (WINDOW / 2).max(1);
        if this.left >= half {
            let grant = this.left;
            this.left = 0;
            this.credit += grant;
            let term = this.shared.deciders.known_term();
            return Poll::Ready(Some(Ok(wire::credit(term, grant))));
        }
        // The gateway's frames, without waiting, until one is the reader's
        // own to answer. A place one of them takes is polled for its end
        // before the stream sleeps.
        while this.own.is_none() {
            let Some(inbound) = this.inbound.as_mut() else {
                break;
            };
            match Pin::new(inbound).poll_next(cx) {
                Poll::Pending => break,
                Poll::Ready(None) => this.inbound = None,
                Poll::Ready(Some(Err(status))) => {
                    return this.end(session::unread("a DecideRequest", status));
                }
                Poll::Ready(Some(Ok(frame))) => {
                    if let Err(status) = this.take(frame) {
                        return this.end(status);
                    }
                }
            }
        }
        // A decision made is older work than an answer of the reader's
        // own, and its budget runs: it leaves first. The reader's answer
        // waits, and no further frame is read while it does, so a run of
        // requests answered here never holds a decision back.
        if let Poll::Ready(Some(decided)) = this.decided.poll_recv(cx) {
            this.deciding -= 1;
            let answer = this.fenced(decided);
            return this.leave(answer, true);
        }
        if let Some((answer, spent)) = this.own.take() {
            return this.leave(answer, spent);
        }
        // The gateway closed its side and every request it sent is
        // answered: the stream's clean end.
        if this.inbound.is_none() && this.deciding == 0 {
            this.ended = true;
            return Poll::Ready(None);
        }
        // The stream sleeps, and either end from outside wakes it. Both
        // hold the waker they were last polled under: they are polled
        // again only under another one, or after a place taken in this
        // poll, so the sweep's word wakes a stream whose request is still
        // with its decider.
        let armed = this.armed.as_ref();
        if !armed.is_some_and(|armed| armed.will_wake(cx.waker())) {
            if this.gone.as_mut().poll(cx).is_ready() {
                return this.end(Status::failed_precondition(NO_SESSION));
            }
            if this.closing.as_mut().poll(cx).is_ready() {
                return this.end(Status::unavailable(CLOSING));
            }
            this.armed = Some(cx.waker().clone());
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::{poll_fn, Future},
        net::SocketAddr,
        pin::{pin, Pin},
        sync::{atomic::Ordering, mpsc as std_mpsc, Arc},
        task::{Context, Poll, Waker},
        thread,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;
    use tokio::{
        net::TcpListener,
        sync::{mpsc, watch},
        task::{
            coop::{consume_budget, has_budget_remaining},
            spawn_blocking, yield_now, JoinSet,
        },
        time::{sleep, timeout},
    };
    use tonic::{
        client::Grpc,
        codegen::{
            http::uri::PathAndQuery,
            tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt},
        },
        transport::{server::TcpIncoming, Channel, Error, Server},
        Code, Request, Status, Streaming,
    };
    use tonic_prost::ProstCodec;

    use super::{serve, Answers, Streams, CLOSING, MAX_STREAMS, NO_SESSION, WINDOW};
    use crate::{
        engine::view::WorkerId,
        service::{
            controller::{
                config::Scope,
                decision_slots::{
                    tests::{
                        busy, fleet, followed, frame, frame_for, gone, hearing, holder, lease_of,
                        picked, preset, settled, Script, Step, BUDGET, PATIENCE,
                    },
                    Clock, DecideStats, DecisionSlots, Job, Reads,
                },
                election::{ElectionObservation, LeasePlace},
                frame_loads,
                gateways::{tests::PATIENT, SessionPlace},
                principal::{tests::told, Layer, ENTRY},
                replication::journaled_for,
                service::{Service, Shared},
                session::GRACE_PERIOD,
                watch_workers::tests::Woken,
                wire::{Checked, Decided, MAX_FRAME},
            },
            elector::{Elector, TickOutcome},
            proto::{self, busy::Reason, decide_answer::Answer, identity::Form},
        },
        store::{HealthSink, LeasesStore},
        tracking::lease::{LeaseId, ReleaseReason, DISPATCH_TTL},
    };

    const ID: &str = "controller-0";
    const ADDRESS: &str = "10.0.0.7:30100";
    const IDENTITY: &str = "controller-0@10.0.0.7:30100";
    const NAME: &str = "gateway-a";
    const OTHER: &str = "gateway-b";
    /// The answers a `Credit` frame follows.
    const HALF: u32 = WINDOW / 2;
    const MS: Duration = Duration::from_millis(1);

    /// The inbound side of a stream a case polls by hand.
    type Frames = mpsc::Sender<Result<proto::DecideRequest, Status>>;

    /// A stream a case polls by hand, over frames of the case's own.
    type ByHand = Answers<ReceiverStream<Result<proto::DecideRequest, Status>>>;

    /// A gateway listener that takes identities as told, over a candidate
    /// that leads under term 1, the deciders' fleet under the `random`
    /// preset as generation 1, and deciders on the case's clock.
    struct Bed {
        elector: Elector,
        standing: watch::Sender<ElectionObservation>,
        shared: Arc<Shared>,
        script: Arc<Script>,
        /// The members serving the cases' model in the single role, ready.
        workers: Vec<WorkerId>,
        addr: SocketAddr,
        /// Aborted at its drop, at the case's end.
        _serving: JoinSet<Result<(), Error>>,
    }

    /// The bed most cases stand on: two deciders behind queues of eight,
    /// the build's cap of streams, a clock that stands still.
    async fn bed() -> Bed {
        bed_with((2, 8), None, []).await
    }

    /// `deciders` deciders behind queues of `depth`, `cap` streams a
    /// gateway (the build's with none), and a clock whose reads by a
    /// decider run `steps`: a stream's own reads show the hour the script
    /// shows and take no step.
    async fn bed_with(
        (deciders, depth): (usize, usize),
        cap: Option<u32>,
        steps: impl IntoIterator<Item = Step>,
    ) -> Bed {
        let start = |reads: &Reads| {
            DecisionSlots::start(deciders, depth, reads).expect("the deciders start")
        };
        bed_over(start, deciders, cap, steps).await
    }

    /// [`bed_with`] over the deciders `start` makes of what they read.
    async fn bed_over(
        start: impl FnOnce(&Reads) -> DecisionSlots,
        deciders: usize,
        cap: Option<u32>,
        steps: impl IntoIterator<Item = Step>,
    ) -> Bed {
        let (store, feed) = journaled_for(&LeasePlace::Process, 0, deciders);
        let (workers, _sides) = fleet(&store);
        store
            .apply(preset("random", 1))
            .expect("above the unbound start");
        let mut elector = Elector::alone(IDENTITY, PATIENT).expect("the settings are in order");
        let taken = Instant::now();
        assert_eq!(
            elector.tick(taken).expect("the take"),
            TickOutcome::Acquired { term: 1 }
        );
        // The election's report tells the store the term of each take: a
        // lease is minted under the store's own term.
        store.set_term(1);
        let (standing, stood) = watch::channel(ElectionObservation {
            holder: IDENTITY.to_owned(),
            term: 1,
            looked: true,
        });
        let script = Arc::new(Script {
            taken,
            steps: Mutex::new(steps.into_iter().collect()),
            shown: Mutex::new(Duration::ZERO),
            readers: Mutex::default(),
        });
        let clock: Clock = {
            let script = Arc::clone(&script);
            Arc::new(move || {
                let decider = thread::current()
                    .name()
                    .is_some_and(|name| name.starts_with("decide-"));
                if decider {
                    script.read()
                } else {
                    script.taken + *script.shown.lock()
                }
            })
        };
        let mut shared = Shared::new(
            ID.to_owned(),
            ADDRESS.to_owned(),
            &elector,
            stood.clone(),
            followed((Arc::clone(&store), feed)),
            Arc::new(crate::stages::StageRegistry::builtin()),
            deciders,
        )
        .expect("the deciders start");
        let reads = Reads {
            objects: holder(&store),
            loads: frame_loads::silent(),
            store,
            fence: elector.fence(),
            standing: stood,
            clock,
            counts: Arc::default(),
        };
        shared.deciders = start(&reads);
        if let Some(cap) = cap {
            shared.streams = Arc::new(Streams::new(cap));
        }
        let shared = Arc::new(shared);
        let service = Service::new(vec![Scope::Gateway], Arc::clone(&shared));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a loopback port");
        let addr = listener.local_addr().expect("the bound address");
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(service.routes(Layer::AsTold))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        Bed {
            elector,
            standing,
            shared,
            script,
            workers,
            addr,
            _serving: serving,
        }
    }

    impl Bed {
        /// How many `Busy` answers the controller counted for `reason`.
        fn busied(&self, reason: Reason) -> u64 {
            self.counts().busy[reason as usize].load(Ordering::Relaxed)
        }

        fn counts(&self) -> &DecideStats {
            self.shared.deciders.counts()
        }

        /// The hour the case's clock shows.
        fn now(&self) -> Instant {
            self.script.taken + *self.script.shown.lock()
        }

        /// The case's clock shows the take plus `offset` from here on.
        fn at(&self, offset: Duration) {
            *self.script.shown.lock() = offset;
        }

        /// A session of the gateway `name` takes its seat at the clock's
        /// hour, under `term`.
        fn welcomed(&self, name: &str, term: u64) -> SessionPlace {
            self.shared
                .gateways
                .welcome(told(name).gateway(), name, term, self.now())
                .expect("the table takes the Hello")
        }

        /// The candidate takes its lease again at the clock's hour, under
        /// `term`, and the election publishes the look and tells the store
        /// the term.
        fn takes(&mut self, term: u64) {
            let tick = self.elector.tick(self.now()).expect("the take");
            assert_eq!(tick, TickOutcome::Acquired { term });
            self.shared.store.set_term(term);
            self.standing.send_replace(ElectionObservation {
                holder: IDENTITY.to_owned(),
                term,
                looked: true,
            });
        }

        /// One `Decide` stream of the gateway `name` that sends frames of
        /// the case's kind, or the status its open was refused with.
        async fn open_with<F>(&self, name: &'static str) -> Result<Opened<F>, Status>
        where
            F: prost::Message + Send + 'static,
        {
            let channel = Channel::from_shared(format!("http://{}", self.addr))
                .expect("a uri")
                .connect()
                .await
                .expect("the listener accepts");
            let (frames, sent) = mpsc::channel(4 * WINDOW as usize);
            let mut request = Request::new(ReceiverStream::new(sent));
            request
                .metadata_mut()
                .insert(ENTRY, name.parse().expect("an ASCII value"));
            let mut call = Grpc::new(channel);
            call.ready().await.expect("the channel is ready");
            let answers = call
                .streaming(
                    request,
                    PathAndQuery::from_static("/smg.policy.v1.PolicyController/Decide"),
                    ProstCodec::<F, proto::DecideAnswer>::default(),
                )
                .await?
                .into_inner();
            Ok(Opened {
                frames: Some(frames),
                answers,
            })
        }

        async fn open(&self, name: &'static str) -> Result<Opened, Status> {
            self.open_with(name).await
        }

        /// The stream a present gateway opens.
        async fn opened(&self, name: &'static str) -> Opened {
            match self.open(name).await {
                Ok(stream) => stream,
                Err(status) => panic!("a present gateway's stream opens: {status:?}"),
            }
        }

        /// The status the open of a stream of `name` is refused with, as
        /// its code and its words.
        async fn refused(&self, name: &'static str) -> (Code, String) {
            match self.open(name).await {
                Ok(_) => panic!("the stream's open is refused"),
                Err(status) => (status.code(), status.message().to_owned()),
            }
        }

        /// `stream` takes one request and answers it with a member of the
        /// fleet under `term`.
        async fn decides(&self, stream: &mut Opened, id: u8, term: u64) {
            stream.send(frame(id, u64::from(id))).await;
            let answer = stream.answer().await;
            assert_eq!(
                (answer.request_id.as_slice(), answer.term),
                (&[id][..], term)
            );
            assert!(self.workers.contains(&picked(&answer)), "{answer:?}");
        }
    }

    /// One `Decide` stream, from the gateway's side.
    struct Opened<F = proto::DecideRequest> {
        /// `None` once the gateway closed its side.
        frames: Option<mpsc::Sender<F>>,
        answers: Streaming<proto::DecideAnswer>,
    }

    impl<F> Opened<F> {
        async fn send(&self, frame: F) {
            let frames = self.frames.as_ref().expect("the gateway's side is open");
            assert!(frames.send(frame).await.is_ok(), "the stream takes a frame");
        }

        /// The gateway closes its side.
        fn close(&mut self) {
            self.frames = None;
        }

        /// The stream's next within the case's patience: an answer, `None`
        /// at its clean end, or the status it ended with.
        async fn next(&mut self) -> Result<Option<proto::DecideAnswer>, Status> {
            timeout(PATIENCE, self.answers.message())
                .await
                .expect("the stream's next within the case's patience")
        }

        async fn answer(&mut self) -> proto::DecideAnswer {
            self.next()
                .await
                .expect("an answer, not a status")
                .expect("an answer, not the stream's end")
        }

        /// The status the stream ended with, as its code and its words.
        async fn status(&mut self) -> (Code, String) {
            let status = self.next().await.expect_err("the stream's last status");
            (status.code(), status.message().to_owned())
        }
    }

    /// A frame of the case's own bytes: `facts` is the bytes a request's
    /// `RequestFacts` is decoded from.
    #[derive(Clone, PartialEq, prost::Message)]
    struct Odd {
        #[prost(string, tag = "2")]
        facts: String,
    }

    /// The grant of a `Credit` frame, which carries no request's id.
    fn granted(answer: &proto::DecideAnswer) -> u32 {
        assert!(answer.request_id.is_empty(), "{answer:?}");
        match &answer.answer {
            Some(Answer::Credit(credit)) => credit.window,
            other => panic!("a Credit: {other:?}"),
        }
    }

    /// A frame of `size` bytes as encoded: block hashes by eights, and a
    /// tenant's key for the rest.
    fn sized(id: u8, size: usize) -> proto::DecideRequest {
        use prost::Message as _;

        let key = |len: usize| proto::Identity {
            form: Some(Form::Plain("t".repeat(len))),
        };
        let mut frame = frame(id, u64::from(id));
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        facts.block_hashes = vec![7; (size - 1024) / 8];
        facts.tenant = Some(key(200));
        let short = size - frame.encoded_len();
        let facts = frame.facts.as_mut().expect("the case's frame has facts");
        facts.tenant = Some(key(200 + short));
        assert_eq!(frame.encoded_len(), size);
        frame
    }

    /// A request's id out of a count.
    fn nth(count: u32) -> u8 {
        u8::try_from(count).expect("an id of one byte")
    }

    /// A hold for a decider's read of the clock: the step, the end the
    /// case hears the decider on, and the end it lets the read go on by.
    fn hold() -> (Step, std_mpsc::Receiver<()>, std_mpsc::SyncSender<()>) {
        let (reached, at_read) = std_mpsc::sync_channel(1);
        let (release, go) = std_mpsc::sync_channel(1);
        (Step::Hold { reached, go }, at_read, release)
    }

    /// Waits, off the runtime's thread, for a decider to reach its hold.
    async fn held(at_read: std_mpsc::Receiver<()>) {
        spawn_blocking(move || at_read.recv_timeout(PATIENCE))
            .await
            .expect("the wait ends")
            .expect("a decider reached its hold");
    }

    #[tokio::test]
    async fn a_present_gateways_requests_are_decided_in_their_order_under_the_leaders_term() {
        // A queue that holds the case's burst whole.
        let bed = bed_with((2, 16), None, []).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        // Sixteen requests of one model before the first answer is read.
        for (id, seed) in (1..=16).zip(100..) {
            stream.send(frame(id, seed)).await;
        }
        for id in 1..=16 {
            let answer = stream.answer().await;
            assert_eq!((answer.request_id.as_slice(), answer.term), (&[id][..], 1));
            assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        }
    }

    #[tokio::test]
    async fn a_stream_is_refused_at_its_open_where_no_session_of_its_gateway_stands() {
        let bed = bed().await;
        // The cap the controller starts with is the one its Welcome tells.
        assert_eq!(bed.shared.streams.cap, MAX_STREAMS);
        // Another gateway's session is not this one's.
        let _other = bed.welcomed(OTHER, 1);
        assert_eq!(
            bed.refused(NAME).await,
            (
                Code::FailedPrecondition,
                "no session of this gateway stands on this controller: a decide stream is a present gateway's (say Hello on a Session stream first)"
                    .to_owned()
            )
        );
        let mut theirs = bed.opened(OTHER).await;
        bed.decides(&mut theirs, 1, 1).await;
    }

    #[tokio::test]
    async fn a_session_that_ends_and_one_that_follows_inside_the_grace_period_change_nothing_for_a_stream(
    ) {
        let bed = bed().await;
        let seat = bed.welcomed(NAME, 1);
        let mut early = bed.opened(NAME).await;
        // The session ends; a millisecond of its gateway's period is left.
        seat.end(bed.now());
        bed.at(GRACE_PERIOD - MS);
        let mut late = bed.opened(NAME).await;
        bed.decides(&mut early, 1, 1).await;
        bed.decides(&mut late, 2, 1).await;
        // A new session takes its seat inside the period, and the clock
        // runs far past the first one's end: the presence never ended.
        let _seat = bed.welcomed(NAME, 1);
        bed.at(GRACE_PERIOD * 3);
        assert_eq!(bed.shared.gateways.sweep(bed.now()), 0);
        bed.decides(&mut early, 3, 1).await;
        bed.decides(&mut late, 4, 1).await;
    }

    #[tokio::test]
    async fn no_stream_opens_past_the_grace_period_and_an_open_one_ends_at_the_sweep() {
        let bed = bed().await;
        let seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        seat.end(bed.now());
        bed.at(GRACE_PERIOD);
        // The period is over with no sweep passed: presence is read from
        // the clock.
        let no_session = (Code::FailedPrecondition, NO_SESSION.to_owned());
        assert_eq!(bed.refused(NAME).await, no_session);
        // No decision asks the table: the open stream decides until the
        // sweep lets its gateway go, and ends in the same words.
        bed.decides(&mut stream, 1, 1).await;
        assert_eq!(bed.shared.gateways.sweep(bed.now()), 1);
        assert_eq!(stream.status().await, no_session);
        // A new Hello is a new presence.
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        bed.decides(&mut stream, 2, 1).await;
    }

    /// The listeners' close ends an open stream: a gateway that holds its
    /// streams open does not hold a drain to its grace period.
    #[tokio::test]
    async fn an_open_stream_ends_at_the_listeners_close() {
        let closing = (Code::Unavailable, CLOSING.to_owned());
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut asked = bed.opened(NAME).await;
        let mut idle = bed.opened(NAME).await;
        bed.decides(&mut asked, 1, 1).await;
        // The word wakes a stream that waits for its next frame and one
        // that read none.
        bed.shared.stop.send_replace(true);
        assert_eq!(asked.status().await, closing);
        assert_eq!(idle.status().await, closing);
        // A stream that opens after the word is refused in the same words.
        assert_eq!(bed.refused(NAME).await, closing);
    }

    /// Under no lead a stream holds no place, and ends at the listeners'
    /// close the same.
    #[tokio::test]
    async fn a_stream_under_no_lead_ends_at_the_listeners_close() {
        let mut bed = bed().await;
        assert!(bed.elector.release().expect("the release"));
        let mut stream = bed.opened(NAME).await;
        stream.send(frame(1, 1)).await;
        assert_eq!(busy(&stream.answer().await), Reason::NotLeader);
        bed.shared.stop.send_replace(true);
        let closing = (Code::Unavailable, CLOSING.to_owned());
        assert_eq!(stream.status().await, closing);
    }

    /// At a take every gone gateway's period starts again, and a stream's
    /// open may be the table's first read under the new term.
    #[tokio::test]
    async fn a_stream_opened_right_after_a_take_finds_its_gateway_present() {
        let mut bed = bed().await;
        let seat = bed.welcomed(NAME, 1);
        let mut open = bed.opened(NAME).await;
        seat.end(bed.now());
        // The replica gives its lease up, and the gateway's period passes
        // while it does not lead: an open stream is told, and stands.
        assert!(bed.elector.release().expect("the release"));
        bed.at(GRACE_PERIOD + Duration::from_secs(1));
        open.send(frame(1, 1)).await;
        let answer = open.answer().await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
        // It takes the lease again: no sweep and no Hello read the table
        // since.
        bed.takes(2);
        let mut new = bed.opened(NAME).await;
        bed.decides(&mut open, 2, 2).await;
        bed.decides(&mut new, 3, 2).await;
        // The open stream asked the table at its first request under the
        // new term and asks no more: the period passes again with no
        // sweep, a new stream is refused, and the open one decides.
        bed.at(GRACE_PERIOD * 2 + Duration::from_secs(2));
        assert_eq!(bed.refused(NAME).await.0, Code::FailedPrecondition);
        bed.decides(&mut open, 4, 2).await;
    }

    #[tokio::test]
    async fn a_credit_frame_gives_back_half_a_window_each_time_that_many_answers_left() {
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        // Three half windows: half a window past what the stream started
        // with.
        for half in 0..3 {
            for count in 1..=HALF {
                bed.decides(&mut stream, nth(half * HALF + count), 1).await;
            }
            let credit = stream.answer().await;
            assert_eq!((granted(&credit), credit.term), (HALF, 1));
        }
        bed.decides(&mut stream, 255, 1).await;
    }

    /// The window never leaks: a stream flooded a window past its credit is
    /// answered for every request, and has its whole window after.
    #[tokio::test]
    async fn a_stream_flooded_past_its_credit_is_told_at_once_and_has_its_whole_window_after() {
        // One decider behind a queue that holds a window, held at its first
        // read of the clock.
        let (step, at_read, release) = hold();
        let bed = bed_with((1, WINDOW as usize), None, [step]).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        // Two windows at once: the decider holds the first request, its
        // queue the first window's rest, and the second window finds no
        // credit.
        for count in 0..2 * WINDOW {
            stream.send(frame(nth(count), 1)).await;
        }
        held(at_read).await;
        // The second window is answered while the decider is held, in the
        // order it was sent, and spends nothing.
        for count in WINDOW..2 * WINDOW {
            let answer = stream.answer().await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice(), answer.term),
                (Reason::NoCredit, &[nth(count)][..], 1)
            );
        }
        assert_eq!(bed.busied(Reason::NoCredit), u64::from(WINDOW));
        release.send(()).expect("the decider waits for the case");
        // The first window's answers in their order, a Credit frame behind
        // each half of them: the refused window gave no credit back.
        for count in 0..WINDOW {
            let answer = stream.answer().await;
            assert_eq!(answer.request_id, [nth(count)]);
            assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
            if (count + 1) % HALF == 0 {
                let credit = stream.answer().await;
                assert_eq!((granted(&credit), credit.term), (HALF, 1));
            }
        }
        // The window is whole: another one, sent at once, is decided to its
        // last request.
        for count in 0..WINDOW {
            stream.send(frame(nth(count), 2)).await;
        }
        let mut decided = 0;
        while decided < WINDOW {
            let answer = stream.answer().await;
            if answer.request_id.is_empty() {
                assert_eq!(granted(&answer), HALF);
                continue;
            }
            assert_eq!(answer.request_id, [nth(decided)]);
            assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
            decided += 1;
        }
    }

    /// A replica that does not lead has no table to ask: a stream opens
    /// there, is told on each request, and takes its place in the table at
    /// the first request the replica would decide.
    #[tokio::test]
    async fn a_stream_opened_where_the_replica_does_not_lead_takes_its_place_once_it_leads() {
        let mut bed = bed().await;
        assert!(bed.elector.release().expect("the release"));
        // No session stands, and none could: both streams open.
        let mut absent = bed.opened(NAME).await;
        let mut present = bed.opened(OTHER).await;
        for stream in [&mut absent, &mut present] {
            stream.send(frame(1, 1)).await;
            let answer = stream.answer().await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice(), answer.term),
                (Reason::NotLeader, &[1][..], 1)
            );
        }
        // The replica takes the lead, and one of the two gateways says its
        // Hello.
        bed.takes(2);
        let seat = bed.welcomed(OTHER, 2);
        bed.decides(&mut present, 2, 2).await;
        let no_session = (Code::FailedPrecondition, NO_SESSION.to_owned());
        absent.send(frame(2, 2)).await;
        assert_eq!(absent.status().await, no_session);
        // The place a stream took is its gateway's presence, and ends with
        // it.
        seat.end(bed.now());
        bed.at(GRACE_PERIOD);
        assert_eq!(bed.shared.gateways.sweep(bed.now()), 1);
        assert_eq!(present.status().await, no_session);
    }

    /// A gateway let go while its stream's first request under a term is
    /// with its decider: the stream ends at the sweep, not at that answer.
    #[tokio::test]
    async fn a_place_taken_at_a_request_ends_its_stream_while_the_request_is_decided() {
        let (step, at_read, release) = hold();
        let mut bed = bed_with((1, 8), None, [step]).await;
        assert!(bed.elector.release().expect("the release"));
        // Opened under no lead, the stream holds no place.
        let mut stream = bed.opened(NAME).await;
        bed.takes(2);
        let seat = bed.welcomed(NAME, 2);
        // Its first request takes the place and waits at its decider.
        stream.send(frame(1, 1)).await;
        held(at_read).await;
        seat.end(bed.now());
        bed.at(GRACE_PERIOD);
        assert_eq!(bed.shared.gateways.sweep(bed.now()), 1);
        let no_session = (Code::FailedPrecondition, NO_SESSION.to_owned());
        assert_eq!(stream.status().await, no_session);
        release.send(()).expect("the decider waits for the case");
    }

    /// A stream of the gateway `NAME` that a case polls by hand, beside its
    /// inbound side, which the case keeps open.
    fn by_hand(bed: &Bed) -> (Frames, impl Stream<Item = super::Out> + Unpin) {
        let (frames, inbound) = mpsc::channel(1);
        match serve(
            Arc::clone(&bed.shared),
            &told(NAME),
            ReceiverStream::new(inbound),
        ) {
            Ok(stream) => (frames, stream),
            Err(status) => panic!("a present gateway's stream opens: {status:?}"),
        }
    }

    /// Polls `stream`, which sleeps, under one waker and then under
    /// another: how often the second was woken is read at `Woken`.
    fn slept_under_two<S: Stream + Unpin>(stream: &mut S) -> Arc<Woken> {
        let wakers = [Woken::waker(), Woken::waker()];
        for (_, waker) in &wakers {
            let mut cx = Context::from_waker(waker);
            assert!(Pin::new(&mut *stream).poll_next(&mut cx).is_pending());
        }
        let [_, (second, _)] = wakers;
        second
    }

    /// Either end wakes a sleeping stream under the waker it was last
    /// polled under: a stream polled under another one holds that one.
    #[tokio::test]
    async fn an_end_wakes_a_stream_under_the_waker_it_was_last_polled_under() {
        let bed = bed().await;
        // The sweep lets the gateway go.
        let seat = bed.welcomed(NAME, 1);
        let (_frames, mut stream) = by_hand(&bed);
        let second = slept_under_two(&mut stream);
        seat.end(bed.now());
        bed.at(GRACE_PERIOD);
        assert_eq!(bed.shared.gateways.sweep(bed.now()), 1);
        assert_eq!(second.wakes(), 1);
        // The listeners close, under a new presence.
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, mut stream) = by_hand(&bed);
        let second = slept_under_two(&mut stream);
        bed.shared.stop.send_replace(true);
        assert_eq!(second.wakes(), 1);
    }

    /// Polls `stream`, which sleeps, in a poll of the case's task that spent
    /// its budget first, then in the task's next poll under the same
    /// waker, as the runtime polls a task it made yield.
    async fn slept_with_no_budget<S: Stream + Unpin>(stream: &mut S) -> Arc<Woken> {
        let (woken, waker) = Woken::waker();
        poll_fn(|_| {
            // More units than a task's budget holds.
            let mut spent = pin!(async {
                for _ in 0..1_024 {
                    consume_budget().await;
                }
            });
            let mut idle = Context::from_waker(Waker::noop());
            assert!(spent.as_mut().poll(&mut idle).is_pending());
            assert!(!has_budget_remaining());
            let mut cx = Context::from_waker(&waker);
            assert!(Pin::new(&mut *stream).poll_next(&mut cx).is_pending());
            Poll::Ready(())
        })
        .await;
        yield_now().await;
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut *stream).poll_next(&mut cx).is_pending());
        woken
    }

    /// Either end wakes a stream that slept with its task's budget spent:
    /// the runtime refuses a budgeted wait before it registers a waker, and
    /// a stream's two waits run outside the budget.
    #[tokio::test]
    async fn an_end_wakes_a_stream_that_slept_with_its_tasks_budget_spent() {
        let bed = bed().await;
        // The sweep lets the gateway go. The budget's own wakes came first.
        let seat = bed.welcomed(NAME, 1);
        let (_frames, mut stream) = by_hand(&bed);
        let woken = slept_with_no_budget(&mut stream).await;
        let before = woken.wakes();
        seat.end(bed.now());
        bed.at(GRACE_PERIOD);
        assert_eq!(bed.shared.gateways.sweep(bed.now()), 1);
        assert_eq!(woken.wakes(), before + 1);
        // The listeners close, under a new presence.
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, mut stream) = by_hand(&bed);
        let woken = slept_with_no_budget(&mut stream).await;
        let before = woken.wakes();
        bed.shared.stop.send_replace(true);
        assert_eq!(woken.wakes(), before + 1);
    }

    /// Under no lead a request spends a credit as any other, and its
    /// answer gives it back.
    #[tokio::test]
    async fn a_stream_under_no_lead_is_told_past_its_window_and_keeps_its_credit() {
        let mut bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        assert!(bed.elector.release().expect("the release"));
        for half in 0..3 {
            for count in 1..=HALF {
                let id = nth(half * HALF + count);
                stream.send(frame(id, 1)).await;
                let answer = stream.answer().await;
                assert_eq!(
                    (busy(&answer), answer.request_id.as_slice()),
                    (Reason::NotLeader, &[id][..])
                );
            }
            assert_eq!(granted(&stream.answer().await), HALF);
        }
    }

    #[tokio::test]
    async fn a_request_its_deciders_full_queue_refuses_is_answered_busy_at_once_and_its_credit_comes_back(
    ) {
        // One decider behind a queue of one, held at its first read.
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 1), None, [step]).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        stream.send(frame(1, 1)).await;
        held(at_read).await;
        // The decider holds the first; the queue takes the second and has
        // no place for the third.
        stream.send(frame(2, 2)).await;
        stream.send(frame(3, 3)).await;
        let answer = stream.answer().await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice(), answer.term),
            (Reason::QueueDepth, &[3][..], 1)
        );
        assert_eq!(bed.busied(Reason::QueueDepth), 1);
        release.send(()).expect("the decider waits for the case");
        for id in [1, 2] {
            let answer = stream.answer().await;
            assert_eq!(answer.request_id, [id]);
            assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        }
        // The refused request spent a credit as the two decided did: the
        // Credit frame follows the half window's last answer, three of
        // which left above.
        for count in 4..=HALF {
            bed.decides(&mut stream, nth(count), 1).await;
        }
        assert_eq!(granted(&stream.answer().await), HALF);
    }

    /// Where the replica does not lead, the stream tells a request itself
    /// and hands it to no decider: it is told while the one decider is
    /// held, and the full queue behind that one is never asked.
    #[tokio::test]
    async fn a_request_read_under_no_lead_is_told_by_the_stream_and_reaches_no_decider() {
        // One decider behind a queue of one, held at its first read.
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 1), None, [step]).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        stream.send(frame(1, 1)).await;
        held(at_read).await;
        // The queue takes the second. A request with no budget is the
        // stream's own to answer: its answer says the second was read.
        stream.send(frame(2, 2)).await;
        let mut none = frame(3, 3);
        none.facts
            .as_mut()
            .expect("the case's frame has facts")
            .budget_micros = 0;
        stream.send(none).await;
        assert_eq!(busy(&stream.answer().await), Reason::BudgetGoneOnArrival);
        // The clock passes the renew deadline. A request handed over now
        // would come back for the queue's depth.
        bed.at(PATIENT.renew_deadline + Duration::from_secs(1));
        stream.send(frame(4, 4)).await;
        let answer = stream.answer().await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice(), answer.term),
            (Reason::NotLeader, &[4][..], 1)
        );
        assert_eq!(bed.busied(Reason::QueueDepth), 0);
        // Released, the decider finds the fence closed for the two it held.
        release.send(()).expect("the decider waits for the case");
        for id in [1, 2] {
            let answer = stream.answer().await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice()),
                (Reason::NotLeader, &[id][..])
            );
        }
        assert_eq!(bed.busied(Reason::NotLeader), 3);
    }

    #[tokio::test]
    async fn a_budget_that_is_gone_and_a_replica_that_does_not_lead_are_told_on_the_stream_and_it_stands(
    ) {
        // The decider's first read of the clock is a whole budget past the
        // first frame's receipt.
        let mut bed = bed_with((2, 8), None, [Step::At(BUDGET)]).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        stream.send(frame(1, 1)).await;
        let answer = stream.answer().await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice(), answer.term),
            (Reason::BudgetGoneOnArrival, &[1][..], 1)
        );
        // A request that brings no budget has none left at its receipt.
        let mut none = frame(2, 2);
        none.facts
            .as_mut()
            .expect("the case's frame has facts")
            .budget_micros = 0;
        stream.send(none).await;
        assert_eq!(busy(&stream.answer().await), Reason::BudgetGoneOnArrival);
        // Told by the stream itself: the request took no place of a queue,
        // and no decider read its clock for it.
        assert_eq!(bed.script.readers.lock().len(), 1);
        bed.decides(&mut stream, 3, 1).await;

        // The replica gives its lease up: Busy under the term it led under,
        // then under the term the election saw another replica hold.
        assert!(bed.elector.release().expect("the release"));
        stream.send(frame(4, 4)).await;
        let answer = stream.answer().await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
        bed.standing.send_replace(ElectionObservation {
            holder: "controller-1@10.0.0.8:30100".to_owned(),
            term: 4,
            looked: true,
        });
        stream.send(frame(5, 5)).await;
        let answer = stream.answer().await;
        assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 4));
        // Each was counted at its reason, by the decider or by the stream.
        assert_eq!(bed.busied(Reason::BudgetGoneOnArrival), 2);
        assert_eq!(bed.busied(Reason::NotLeader), 2);
    }

    /// A decision made is older work than an answer of the stream's own:
    /// it leaves first.
    #[tokio::test]
    async fn a_decision_made_leaves_before_an_answer_of_the_streams_own() {
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 8), None, [step]).await;
        let _seat = bed.welcomed(NAME, 1);
        let gateway = told(NAME);
        let (frames, inbound) = mpsc::channel::<Result<proto::DecideRequest, Status>>(4);
        let mut stream = match serve(
            Arc::clone(&bed.shared),
            &gateway,
            ReceiverStream::new(inbound),
        ) {
            Ok(stream) => stream,
            Err(status) => panic!("a present gateway's stream opens: {status:?}"),
        };
        // One poll hands the first request to its decider, which the case
        // holds at its first read: no answer is there yet.
        frames
            .send(Ok(frame(1, 1)))
            .await
            .expect("the stream's side is open");
        let mut cx = Context::from_waker(Waker::noop());
        assert!(Pin::new(&mut stream).poll_next(&mut cx).is_pending());
        held(at_read).await;
        release.send(()).expect("the decider waits for the case");
        // The one decider answers a request of the case's own after it:
        // the first one's answer waits in the stream's queue by then.
        let (sender, mut answers) = mpsc::channel(1);
        let job = Job {
            request: Checked::new(frame(2, 2)).expect("a frame the record holds"),
            gateway: gateway.gateway(),
            received: bed.now(),
            answers: sender,
        };
        assert!(bed.shared.deciders.submit(job).is_none());
        let other = timeout(PATIENCE, answers.recv()).await;
        assert!(matches!(other, Ok(Some(_))), "{other:?}");
        // A request that brings no budget is the stream's own to answer,
        // and it is read in the same poll that finds the decision.
        let mut none = frame(3, 3);
        none.facts
            .as_mut()
            .expect("the case's frame has facts")
            .budget_micros = 0;
        frames
            .send(Ok(none))
            .await
            .expect("the stream's side is open");
        let mut next = async || match timeout(PATIENCE, stream.next()).await {
            Ok(Some(Ok(answer))) => answer,
            other => panic!("an answer within the case's patience: {other:?}"),
        };
        let first = next().await;
        assert_eq!(first.request_id, [1]);
        assert!(bed.workers.contains(&picked(&first)), "{first:?}");
        let second = next().await;
        assert_eq!(
            (busy(&second), second.request_id.as_slice()),
            (Reason::BudgetGoneOnArrival, &[3][..])
        );
    }

    /// A stream of the gateway `NAME` that a case polls by hand, with the
    /// decisions of its requests 1 and 2 waiting in its queue. One poll
    /// hands both to the bed's one decider, held at its first read. Let
    /// go, it decides both under the open fence of term 1 and answers a
    /// request of the case's own behind them: both wait in the queue by
    /// then. The inbound side comes back with the stream, and the case
    /// keeps it open.
    async fn two_queued(
        bed: &Bed,
        at_read: std_mpsc::Receiver<()>,
        release: &std_mpsc::SyncSender<()>,
    ) -> (Frames, ByHand) {
        let gateway = told(NAME);
        let (frames, inbound) = mpsc::channel(4);
        let mut stream = match serve(
            Arc::clone(&bed.shared),
            &gateway,
            ReceiverStream::new(inbound),
        ) {
            Ok(stream) => stream,
            Err(status) => panic!("a present gateway's stream opens: {status:?}"),
        };
        for id in [1, 2] {
            frames
                .send(Ok(frame(id, 1)))
                .await
                .expect("the stream's side is open");
        }
        let mut cx = Context::from_waker(Waker::noop());
        assert!(Pin::new(&mut stream).poll_next(&mut cx).is_pending());
        held(at_read).await;
        release.send(()).expect("the decider waits for the case");
        let (sender, mut answers) = mpsc::channel(1);
        let job = Job {
            request: Checked::new(frame(3, 3)).expect("a frame the record holds"),
            gateway: gateway.gateway(),
            received: bed.now(),
            answers: sender,
        };
        assert!(bed.shared.deciders.submit(job).is_none());
        let other = timeout(PATIENCE, answers.recv()).await;
        assert!(matches!(other, Ok(Some(_))), "{other:?}");
        (frames, stream)
    }

    /// A decision waits in its stream's queue for as long as its gateway
    /// reads nothing, so the fence is read where one is sent, and the term
    /// it must show is the decision's own. One made inside the renew
    /// deadline and sent past it leaves as `Busy` for the leader that is
    /// elsewhere, and so does one sent once the replica leads under the
    /// next term. Each is counted at that reason.
    #[tokio::test]
    async fn a_decision_leaves_as_busy_once_the_fence_no_longer_shows_its_term() {
        let (step, at_read, release) = hold();
        let mut bed = bed_with((1, 8), None, [step]).await;
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, mut stream) = two_queued(&bed, at_read, &release).await;
        let mut next = async || match timeout(PATIENCE, stream.next()).await {
            Ok(Some(Ok(answer))) => answer,
            other => panic!("an answer within the case's patience: {other:?}"),
        };
        // The clock passes the renew deadline before the stream is polled
        // again: the fence shows no term.
        bed.at(PATIENT.renew_deadline + Duration::from_secs(1));
        let first = next().await;
        assert_eq!(
            (busy(&first), first.request_id.as_slice(), first.term),
            (Reason::NotLeader, &[1][..], 1)
        );
        // The replica gives its lease up and takes it again under term 2:
        // the fence shows a term, and not the second decision's.
        assert!(bed.elector.release().expect("the release"));
        bed.takes(2);
        let second = next().await;
        assert_eq!(
            (busy(&second), second.request_id.as_slice(), second.term),
            (Reason::NotLeader, &[2][..], 2)
        );
        assert_eq!(bed.busied(Reason::NotLeader), 2);
    }

    /// A decision its stream stops for the fence keeps no lease. A lease
    /// stands on file while its decision waits in the queue, its engine
    /// told of the issue, and a decision that leaves keeps its own. The
    /// stream that stops one hands its lease's release where it would send
    /// it. With no decider at work the lease is off the file at that send,
    /// the engine that issued it told once, as of a lease whose answer
    /// never left.
    #[tokio::test]
    async fn a_decision_stopped_for_the_fence_gives_its_lease_up_at_the_send() {
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 8), None, [step]).await;
        let heard = Arc::new(Mutex::new(Vec::new()));
        let later = bed.shared.store.apply(hearing(&heard, 2));
        later.expect("a later generation");
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, mut stream) = two_queued(&bed, at_read, &release).await;
        let gateway = told(NAME).gateway();
        let id = |count: u64| 1 << 32 | count;
        let store: &dyn LeasesStore = bed.shared.store.as_ref();
        // Two leases of the stream's decisions and the case's own.
        assert_eq!(store.live(gateway), 3);
        assert_eq!(*heard.lock(), [(id(1), None), (id(2), None), (id(3), None)]);
        let mut next = async || match timeout(PATIENCE, stream.next()).await {
            Ok(Some(Ok(answer))) => answer,
            other => panic!("an answer within the case's patience: {other:?}"),
        };
        // The first leaves under the open fence, its lease under its term,
        // and the lease stands.
        let first = next().await;
        let lease = lease_of(&first);
        assert_eq!((first.term, lease.term, lease.id), (1, 1, id(1)));
        assert_eq!(store.live(gateway), 3);
        // The clock passes the renew deadline before the second is sent:
        // the fence shows no term.
        bed.at(PATIENT.renew_deadline + Duration::from_secs(1));
        let second = next().await;
        assert_eq!(
            (busy(&second), second.request_id.as_slice(), second.term),
            (Reason::NotLeader, &[2][..], 1)
        );
        // The lease's release was handed at the send: its file's decider
        // applies it as it leaves its work, or this read does.
        settled(|| store.get(gateway, LeaseId(id(2))).is_none()).await;
        assert_eq!(store.live(gateway), 2);
        let abandoned = (id(2), Some(ReleaseReason::Abandoned));
        assert_eq!(heard.lock()[3..], [abandoned]);
    }

    /// The word is for a request whose stream ended (the contract's field).
    /// Said while the earlier answer waits in an open stream's queue, it
    /// ends the lease all the same, and nothing holds that answer back: it
    /// leaves with the ended lease, which no dispatch can name. The send's
    /// fence reads the term and the span alone, and no read there could
    /// call back an answer that left before the repeat was read.
    #[tokio::test]
    async fn an_answer_queued_before_its_request_is_asked_again_leaves_with_the_ended_lease() {
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 8), None, [step]).await;
        let heard = Arc::new(Mutex::new(Vec::new()));
        let later = bed.shared.store.apply(hearing(&heard, 2));
        later.expect("a later generation");
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, mut stream) = two_queued(&bed, at_read, &release).await;
        let gateway = told(NAME).gateway();
        let id = |count: u64| 1 << 32 | count;
        let store: &dyn LeasesStore = bed.shared.store.as_ref();
        assert_eq!(store.live(gateway), 3);
        // Request 1 again, as on a second stream of its gateway, while the
        // answer to its first ask waits in the open stream's queue.
        let mut repeat = frame(1, 1);
        repeat.again = true;
        let (sender, mut answers) = mpsc::channel(1);
        let job = Job {
            request: Checked::new(repeat).expect("a frame the record holds"),
            gateway,
            received: bed.now(),
            answers: sender,
        };
        assert!(bed.shared.deciders.submit(job).is_none());
        let anew = timeout(PATIENCE, answers.recv()).await;
        let leased = matches!(anew, Ok(Some(Decided::Decision { lease: Some(_), .. })));
        assert!(leased, "{anew:?}");
        // The word ended the first ask's lease, its engine told ahead of
        // the repeat's own lease: one lease of the request is on file.
        let ended = (id(1), Some(ReleaseReason::Abandoned));
        assert_eq!(heard.lock()[3..], [ended, (id(4), None)]);
        assert!(store.get(gateway, LeaseId(id(1))).is_none());
        assert_eq!(store.live(gateway), 3);
        // The queued answer leaves all the same, with the ended lease.
        let first = match timeout(PATIENCE, stream.next()).await {
            Ok(Some(Ok(answer))) => answer,
            other => panic!("an answer within the case's patience: {other:?}"),
        };
        let lease = lease_of(&first);
        assert_eq!((first.request_id.as_slice(), lease.id), (&[1][..], id(1)));
        assert!(store.get(gateway, LeaseId(id(1))).is_none());
        assert_eq!(store.live(gateway), 3);
    }

    /// A stream that ends gives up the lease of each decision still in
    /// its queue: no gateway will read one. Both leave the file at the
    /// stream's drop, and the engine that issued them is told once of
    /// each, as of a lease whose answer never left. A lease that another
    /// stream's decision holds stands.
    #[tokio::test]
    async fn a_stream_that_ends_gives_up_the_leases_of_the_decisions_in_its_queue() {
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 8), None, [step]).await;
        let heard = Arc::new(Mutex::new(Vec::new()));
        let later = bed.shared.store.apply(hearing(&heard, 2));
        later.expect("a later generation");
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, stream) = two_queued(&bed, at_read, &release).await;
        let gateway = told(NAME).gateway();
        let id = |count: u64| 1 << 32 | count;
        let store: &dyn LeasesStore = bed.shared.store.as_ref();
        // Two leases of the stream's decisions and the case's own.
        assert_eq!(store.live(gateway), 3);
        drop(stream);
        // Their releases were handed at the drop: the file's decider
        // applies them as it leaves its work, or this read does.
        settled(|| store.live(gateway) == 1).await;
        let abandoned = |count| (id(count), Some(ReleaseReason::Abandoned));
        assert_eq!(heard.lock()[3..], [abandoned(1), abandoned(2)]);
        assert!(store.get(gateway, LeaseId(id(3))).is_some());
        // Both answers are counted as dropped for a stream that is gone.
        assert_eq!(bed.counts().dropped_gone.load(Ordering::Relaxed), 2);
    }

    /// A request asked again behind its stream's end finds the lease of
    /// its queued answer given up already. The stream's word is applied
    /// ahead of the search: the engine hears of that lease once, the
    /// repeat ends none, and one lease stands for the request.
    #[tokio::test]
    async fn a_request_asked_again_behind_its_streams_end_finds_no_lease_to_end() {
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 8), None, [step]).await;
        let heard = Arc::new(Mutex::new(Vec::new()));
        let later = bed.shared.store.apply(hearing(&heard, 2));
        later.expect("a later generation");
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, stream) = two_queued(&bed, at_read, &release).await;
        let gateway = told(NAME).gateway();
        let id = |count: u64| 1 << 32 | count;
        drop(stream);
        let mut repeat = frame(1, 1);
        repeat.again = true;
        let (sender, mut answers) = mpsc::channel(1);
        let job = Job {
            request: Checked::new(repeat).expect("a frame the record holds"),
            gateway,
            received: bed.now(),
            answers: sender,
        };
        assert!(bed.shared.deciders.submit(job).is_none());
        let anew = timeout(PATIENCE, answers.recv()).await;
        assert!(matches!(anew, Ok(Some(_))), "{anew:?}");
        let abandoned = |count| (id(count), Some(ReleaseReason::Abandoned));
        let told = [abandoned(1), abandoned(2), (id(4), None)];
        assert_eq!(heard.lock()[3..], told);
        // The repeat found no lease that waits: `none` is counted, not `ended`.
        let again = bed.counts().again.each_ref();
        assert_eq!(again.map(|count| count.load(Ordering::Relaxed)), [1, 0]);
        let store: &dyn LeasesStore = bed.shared.store.as_ref();
        assert_eq!(store.live(gateway), 2);
    }

    /// A stream's end gives its hand back, as a session's end does. With
    /// more than one decider a file that holds a lease keeps its decider's
    /// watch, and a hand leaves its frame there for that decider's next
    /// look. The stream's last look applies it in place: the engine has
    /// heard of both leases by the time the stream is gone.
    #[tokio::test]
    async fn a_stream_that_ends_at_a_watched_file_applies_its_releases_as_it_leaves() {
        let (step, at_read, release) = hold();
        let bed = bed_with((2, 8), None, [step]).await;
        let heard = Arc::new(Mutex::new(Vec::new()));
        let later = bed.shared.store.apply(hearing(&heard, 2));
        later.expect("a later generation");
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, stream) = two_queued(&bed, at_read, &release).await;
        // Three leases stand on the decider's file, so it keeps its watch
        // there. The watch can start ahead of the case's own request: the
        // case goes on once the decider has left that work too.
        let store = Arc::clone(&bed.shared.store);
        settled(|| store.watches() == 1 && store.working() == 0).await;
        let issued: Vec<u64> = heard.lock().iter().map(|&(id, _)| id).collect();
        assert_eq!(issued.len(), 3);
        drop(stream);
        let abandoned = Some(ReleaseReason::Abandoned);
        let told = [(issued[0], abandoned), (issued[1], abandoned)];
        assert_eq!(heard.lock()[3..], told);
    }

    /// A lease tells what its dispatch span has left where its stream
    /// sends it, and under a millisecond is none to hand over. A decision
    /// sent with a millisecond of the span left leaves with its lease,
    /// which tells that millisecond. One sent a microsecond later leaves
    /// as the `Busy` a late decision is told, its lease given up as a late
    /// decision's. The count of decisions that ended while they were made
    /// stands: this one was made in time.
    #[tokio::test]
    async fn a_decision_sent_past_its_leases_span_leaves_as_busy_and_gives_the_lease_up() {
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 8), None, [step]).await;
        let heard = Arc::new(Mutex::new(Vec::new()));
        let later = bed.shared.store.apply(hearing(&heard, 2));
        later.expect("a later generation");
        let _seat = bed.welcomed(NAME, 1);
        let (_frames, mut stream) = two_queued(&bed, at_read, &release).await;
        let gateway = told(NAME).gateway();
        let id = |count: u64| 1 << 32 | count;
        let mut next = async || match timeout(PATIENCE, stream.next()).await {
            Ok(Some(Ok(answer))) => answer,
            other => panic!("an answer within the case's patience: {other:?}"),
        };
        bed.at(DISPATCH_TTL - MS);
        let first = next().await;
        let lease = lease_of(&first);
        assert_eq!((lease.id, lease.dispatch_ttl_ms), (id(1), 1));
        bed.at(DISPATCH_TTL - MS + Duration::from_micros(1));
        let second = next().await;
        assert_eq!(
            (busy(&second), second.request_id.as_slice(), second.term),
            (Reason::BudgetGoneOnArrival, &[2][..], 1)
        );
        let store: &dyn LeasesStore = bed.shared.store.as_ref();
        // The lease's release was handed at the send: its file's decider
        // applies it as it leaves its work, or this read does.
        settled(|| store.get(gateway, LeaseId(id(2))).is_none()).await;
        assert_eq!(store.live(gateway), 2);
        let late = (id(2), Some(ReleaseReason::LateDecision));
        assert_eq!(heard.lock()[3..], [late]);
        assert_eq!(bed.busied(Reason::BudgetGoneOnArrival), 1);
        assert_eq!(bed.counts().outlasted.load(Ordering::Relaxed), 0);
    }

    /// The controller stops for a deciding thread that ended. Until it has,
    /// a request for that thread's models is the gateway's to decide: told
    /// at once with no reason, counted by its cause, on a stream that
    /// stands.
    #[tokio::test]
    async fn a_request_whose_deciders_thread_ended_is_answered_busy_with_no_reason_and_counted() {
        let bed = bed_over(gone, 1, None, []).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        for id in 1..=HALF {
            stream.send(frame(nth(id), 1)).await;
            let answer = stream.answer().await;
            assert_eq!(
                (busy(&answer), answer.request_id.as_slice(), answer.term),
                (Reason::Unspecified, &[nth(id)][..], 1)
            );
        }
        // Each spent a credit, and the half window's Credit frame follows.
        assert_eq!(granted(&stream.answer().await), HALF);
        assert_eq!(bed.counts().gone.load(Ordering::Relaxed), u64::from(HALF));
        assert_eq!(bed.busied(Reason::Unspecified), u64::from(HALF));
    }

    #[tokio::test]
    async fn a_gateway_holds_its_cap_of_streams_and_a_stream_that_is_gone_gives_its_place_back() {
        let bed = bed_with((2, 8), Some(2), []).await;
        let _seat = bed.welcomed(NAME, 1);
        let _other = bed.welcomed(OTHER, 1);
        let first = bed.opened(NAME).await;
        let mut second = bed.opened(NAME).await;
        let past_the_cap = (
            Code::ResourceExhausted,
            "the gateway holds its cap of 2 decide streams on this controller".to_owned(),
        );
        assert_eq!(bed.refused(NAME).await, past_the_cap);
        // The cap is each gateway's own.
        let mut theirs = bed.opened(OTHER).await;
        // The controller gives the place back when it sees the stream gone:
        // within two seconds here, and every try before is refused the same.
        drop(first);
        let mut again = None;
        for _try in 0..400 {
            match bed.open(NAME).await {
                Ok(stream) => {
                    again = Some(stream);
                    break;
                }
                Err(status) => assert_eq!(status.code(), Code::ResourceExhausted),
            }
            sleep(Duration::from_millis(5)).await;
        }
        let mut again = again.expect("the place of the stream that is gone");
        assert_eq!(bed.refused(NAME).await, past_the_cap);
        bed.decides(&mut second, 1, 1).await;
        bed.decides(&mut theirs, 2, 1).await;
        bed.decides(&mut again, 3, 1).await;
    }

    /// A gateway's last stream gives its place back too: its row leaves the
    /// table of open streams, and the next stream takes the place.
    #[tokio::test]
    async fn a_gateways_last_stream_gives_its_place_back_and_its_row_leaves_the_table() {
        let bed = bed_with((2, 8), Some(1), []).await;
        let _seat = bed.welcomed(NAME, 1);
        let first = bed.opened(NAME).await;
        assert_eq!(bed.shared.streams.open.lock().len(), 1);
        // The controller sees the stream gone within two seconds here, and
        // every try before is refused for the cap.
        drop(first);
        let mut again = None;
        for _try in 0..400 {
            match bed.open(NAME).await {
                Ok(stream) => {
                    again = Some(stream);
                    break;
                }
                Err(status) => assert_eq!(status.code(), Code::ResourceExhausted),
            }
            sleep(Duration::from_millis(5)).await;
        }
        drop(again.expect("the place of the gateway's one stream"));
        let mut rows = usize::MAX;
        for _try in 0..400 {
            rows = bed.shared.streams.open.lock().len();
            if rows == 0 {
                break;
            }
            sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(rows, 0, "a gateway with no stream open keeps no row");
    }

    /// A request told its budget is gone spent a credit and gets it back: a
    /// stream of such requests is never told it has no credit.
    #[tokio::test]
    async fn a_request_told_its_budget_is_gone_gets_its_credit_back() {
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        let mut back = 0;
        for count in 1..=3 * HALF {
            let mut frame = frame(nth(count), 1);
            frame
                .facts
                .as_mut()
                .expect("the case's frame has facts")
                .budget_micros = 0;
            stream.send(frame).await;
            let answer = loop {
                let answer = stream.answer().await;
                match &answer.answer {
                    Some(Answer::Credit(credit)) => back += credit.window,
                    _ => break answer,
                }
            };
            assert_eq!(
                busy(&answer),
                Reason::BudgetGoneOnArrival,
                "request {count}"
            );
        }
        // Half a window came back after each half but the last, whose grant
        // follows the last answer.
        assert_eq!(back, WINDOW);
        assert_eq!(bed.busied(Reason::BudgetGoneOnArrival), u64::from(3 * HALF));
    }

    /// A stream placed under one term asks the table again at its first
    /// request under the next: where an open is refused there, that request
    /// ends the stream in the open's words.
    #[tokio::test]
    async fn a_placed_stream_asks_the_table_again_at_its_first_request_under_the_next_term() {
        let mut bed = bed().await;
        let seat = bed.welcomed(NAME, 1);
        let mut open = bed.opened(NAME).await;
        bed.decides(&mut open, 1, 1).await;
        seat.end(bed.now());
        // The replica gives its lease up and takes it again under term 2.
        // The gone gateway's period starts again at the table's first read
        // under that term, a pass of the sweep here, which lets nobody go.
        assert!(bed.elector.release().expect("the release"));
        bed.at(GRACE_PERIOD + Duration::from_secs(1));
        bed.takes(2);
        assert_eq!(bed.shared.gateways.sweep(bed.now()), 0);
        // That period passes too, with no sweep and no Hello: an open is
        // refused, and the placed stream's first request under term 2 asks
        // the table as that open did.
        bed.at(GRACE_PERIOD * 2 + Duration::from_secs(2));
        let refused = bed.refused(NAME).await;
        assert_eq!(refused.0, Code::FailedPrecondition);
        open.send(frame(2, 2)).await;
        assert_eq!(open.status().await, refused);
    }

    #[tokio::test]
    async fn a_request_out_of_its_range_ends_the_stream_by_its_fields_name_and_a_prompt_as_text_does_not(
    ) {
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        let mut odd = frame(1, 1);
        odd.facts
            .as_mut()
            .expect("the case's frame has facts")
            .phase = 99;
        stream.send(odd).await;
        assert_eq!(
            stream.status().await,
            (
                Code::InvalidArgument,
                "facts.phase is out of its range in this protocol version".to_owned()
            )
        );
        // A prompt as text is no field out of its range: the build has no
        // decision for it yet, its request is answered Busy with no reason,
        // and the stream stands.
        let mut stream = bed.opened(NAME).await;
        let mut text = frame(2, 2);
        text.facts
            .as_mut()
            .expect("the case's frame has facts")
            .prompt_text = b"a prompt".to_vec();
        stream.send(text).await;
        let answer = stream.answer().await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice()),
            (Reason::Unspecified, &[2][..])
        );
        bed.decides(&mut stream, 3, 1).await;
    }

    /// An answer carries its request's id back, and the id is its request's
    /// key: the contract holds it to 1 to 256 bytes. One at the bound is
    /// answered like any other; one byte more, and an empty one, end the
    /// stream by the field's name. A frame is checked before the reader says
    /// anything of its own: a stream with no credit left is refused the same
    /// way, and no Busy carries the long id back.
    #[tokio::test]
    async fn a_request_id_out_of_its_range_ends_the_stream_and_no_answer_carries_it() {
        let refused = (
            Code::InvalidArgument,
            "request_id is out of its range in this protocol version".to_owned(),
        );
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        // The range by its numbers, never by the constant: 256 bytes are
        // carried back and 257 end the stream.
        let mut longest = frame(1, 1);
        longest.request_id = vec![7; 256];
        stream.send(longest).await;
        let answer = stream.answer().await;
        assert_eq!(answer.request_id, vec![7; 256]);
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        let mut past = frame(2, 2);
        past.request_id = vec![7; 257];
        stream.send(past.clone()).await;
        assert_eq!(stream.status().await, refused);
        let mut stream = bed.opened(NAME).await;
        let mut keyless = frame(3, 3);
        keyless.request_id = Vec::new();
        stream.send(keyless).await;
        assert_eq!(stream.status().await, refused);
        // No credit left: one decider held at its first read of the clock,
        // a window in flight behind it, then the long id.
        let (step, at_read, release) = hold();
        let bed = bed_with((1, WINDOW as usize), None, [step]).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        for count in 0..WINDOW {
            stream.send(frame(nth(count), 1)).await;
        }
        held(at_read).await;
        stream.send(past).await;
        assert_eq!(stream.status().await, refused);
        assert_eq!(bed.busied(Reason::NoCredit), 0);
        release.send(()).expect("the decider waits for the case");
    }

    /// A `DecideRequest` is at most [`MAX_FRAME`] bytes as received: one at
    /// the bound is decided, and one byte more ends the stream before it
    /// is read, by the transport's limit on this call. A window and a
    /// decider's queue hold frames by the place, so this is what bounds
    /// their bytes.
    #[tokio::test]
    async fn a_frame_past_the_calls_bound_ends_the_stream_and_one_at_the_bound_is_decided() {
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        // The bound by its numbers, never by the constant: 65,536 bytes
        // are decided and one byte more ends the stream.
        stream.send(sized(1, 65_536)).await;
        let answer = stream.answer().await;
        assert_eq!(answer.request_id, [1]);
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        stream.send(sized(2, 65_537)).await;
        let (code, message) = stream.status().await;
        assert_eq!(code, Code::ResourceExhausted, "{message}");
        assert!(
            message.starts_with("a DecideRequest is over the transport's limit: ")
                && message.contains("65537")
                && message.contains("65536"),
            "{message}"
        );
    }

    /// The bound is the frame's as received. A frame padded with a field
    /// this build does not know reads to a few bytes, and is held to the
    /// bound all the same: at it the frame is answered, and one byte more
    /// ends the stream.
    #[tokio::test]
    async fn a_frame_padded_with_a_field_this_build_does_not_know_is_held_to_the_bound() {
        /// A `DecideRequest`'s id, and a field no `DecideRequest` names.
        #[derive(Clone, PartialEq, prost::Message)]
        struct Padded {
            #[prost(bytes = "vec", tag = "1")]
            request_id: Vec<u8>,
            #[prost(bytes = "vec", tag = "1000")]
            pad: Vec<u8>,
        }
        // Three bytes of the id, two of the pad's tag, three of its length.
        let padded = |id: u8, size: usize| {
            let frame = Padded {
                request_id: vec![id],
                pad: vec![0; size - 8],
            };
            assert_eq!(prost::Message::encoded_len(&frame), size);
            frame
        };
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = match bed.open_with::<Padded>(NAME).await {
            Ok(stream) => stream,
            Err(status) => panic!("a present gateway's stream opens: {status:?}"),
        };
        // A frame that states no budget has none: answered at the stream.
        stream.send(padded(1, 65_536)).await;
        let answer = stream.answer().await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice()),
            (Reason::BudgetGoneOnArrival, &[1][..])
        );
        stream.send(padded(2, 65_537)).await;
        let (code, message) = stream.status().await;
        assert_eq!(code, Code::ResourceExhausted, "{message}");
        assert!(
            message.starts_with("a DecideRequest is over the transport's limit: ")
                && message.contains("65537"),
            "{message}"
        );
    }

    /// A frame's headers are as many as a view names at most, 256: an empty
    /// header is two bytes as encoded and a record of its own as read, so
    /// their count is what bounds them in a place. 256 are decided, and one
    /// more ends the stream by the list's path.
    #[tokio::test]
    async fn a_frame_of_more_headers_than_a_view_names_ends_the_stream() {
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        let headers = |id, count: usize| {
            let mut frame = frame(id, 1);
            frame
                .facts
                .as_mut()
                .expect("the case's frame has facts")
                .headers = vec![proto::Header::default(); count];
            frame
        };
        stream.send(headers(1, 256)).await;
        let answer = stream.answer().await;
        assert_eq!(answer.request_id, [1]);
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        stream.send(headers(2, 257)).await;
        assert_eq!(
            stream.status().await,
            (
                Code::InvalidArgument,
                "facts.headers is out of its range in this protocol version".to_owned()
            )
        );
    }

    /// What a window of the largest frames holds resident: one decider held
    /// at its first read of the clock, the window behind it, and one frame
    /// more, whose `Busy` for no credit says the window was read. A
    /// measurement for the Numbers table, run by name with `--ignored`: it
    /// prints, and asserts only that the window is decided once the decider
    /// goes on.
    #[tokio::test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    async fn rss_with_a_window_of_the_largest_frames_behind_a_held_decider() {
        use std::io::Write as _;

        /// `VmRSS` of this process, in KiB; `0` where `/proc` has no answer.
        fn vm_rss_kib() -> u64 {
            std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|status| {
                    status.lines().find_map(|line| {
                        line.strip_prefix("VmRSS:")
                            .and_then(|rest| rest.trim().split(' ').next())
                            .and_then(|kib| kib.parse().ok())
                    })
                })
                .unwrap_or(0)
        }

        let (step, at_read, release) = hold();
        let bed = bed_with((1, WINDOW as usize), None, [step]).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        let before = vm_rss_kib();
        for count in 0..WINDOW {
            stream.send(sized(nth(count), MAX_FRAME)).await;
        }
        held(at_read).await;
        stream.send(frame(nth(WINDOW), 1)).await;
        assert_eq!(busy(&stream.answer().await), Reason::NoCredit);
        let with_window = vm_rss_kib();
        let mut out = std::io::stdout();
        writeln!(out, "RSS_BEFORE_KIB {before}").expect("stdout");
        writeln!(out, "RSS_WINDOW_HELD_KIB {with_window}").expect("stdout");
        release.send(()).expect("the decider waits for the case");
        let mut decided = 0;
        while decided < WINDOW {
            let answer = stream.answer().await;
            if !answer.request_id.is_empty() {
                assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
                decided += 1;
            }
        }
    }

    /// A frame the stream cannot read ends it with a status: one over the
    /// transport's limit, and one no request decodes from.
    #[tokio::test]
    async fn a_frame_a_decide_stream_cannot_read_ends_it_with_a_status() {
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        // Five MiB: far over the call's bound, refused at its length as a
        // frame one byte over is.
        let mut stream = bed.opened(NAME).await;
        let model = "m".repeat(5 * 1024 * 1024);
        stream.send(frame_for(&model, 1, 1)).await;
        let (code, message) = stream.status().await;
        assert_eq!(code, Code::ResourceExhausted, "{message}");
        assert!(
            message.starts_with("a DecideRequest is over the transport's limit: ")
                && message.contains("65536"),
            "{message}"
        );
        // No `RequestFacts` decodes from this byte: the transport's own
        // status, which is INTERNAL for a message its codec could not
        // decode.
        let mut stream = match bed.open_with::<Odd>(NAME).await {
            Ok(stream) => stream,
            Err(status) => panic!("a present gateway's stream opens: {status:?}"),
        };
        stream
            .send(Odd {
                facts: "\u{f}".to_owned(),
            })
            .await;
        let (code, message) = stream.status().await;
        assert_eq!(code, Code::Internal, "{message}");
        assert!(!message.is_empty());
    }

    #[tokio::test]
    async fn a_stream_ends_cleanly_once_its_gateway_closed_its_side_and_every_request_is_answered()
    {
        // One decider, held at its first read.
        let (step, at_read, release) = hold();
        let bed = bed_with((1, 8), None, [step]).await;
        let _seat = bed.welcomed(NAME, 1);
        let mut stream = bed.opened(NAME).await;
        for id in 1..=3 {
            stream.send(frame(id, 1)).await;
        }
        held(at_read).await;
        // The close reaches the controller while the decider is held. No
        // line below waits on it: the pause only lets an end that came
        // ahead of the answers show.
        stream.close();
        sleep(Duration::from_millis(50)).await;
        release.send(()).expect("the decider waits for the case");
        for id in 1..=3 {
            let answer = stream.answer().await;
            assert_eq!(answer.request_id, [id]);
            assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        }
        assert!(matches!(stream.next().await, Ok(None)));
    }

    #[tokio::test]
    async fn a_stream_decides_for_the_gateway_its_principal_names() {
        let bed = bed().await;
        let _seat = bed.welcomed(NAME, 1);
        let _other = bed.welcomed(OTHER, 1);
        let vetoed = bed.workers[0];
        let until = bed.now() + Duration::from_secs(60);
        bed.shared
            .store
            .gateway_veto(told(NAME).gateway(), vetoed, Some(until));
        let mut own = bed.opened(NAME).await;
        let mut theirs = bed.opened(OTHER).await;
        let mut picks = Vec::new();
        for (id, seed) in (1..=48).zip(1..) {
            own.send(frame(id, seed)).await;
            assert_ne!(picked(&own.answer().await), vetoed, "seed {seed}");
            theirs.send(frame(id, seed)).await;
            picks.push(picked(&theirs.answer().await));
        }
        // The veto is the stating gateway's alone.
        assert!(picks.contains(&vetoed), "{picks:?}");
    }
}
