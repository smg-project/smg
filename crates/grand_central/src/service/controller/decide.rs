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
    // The word is said once, and read here: a stream that opens after the
    // listeners' close is refused in an open one's last words.
    let stop = shared.stop.clone();
    if *stop.borrow() {
        return Err(Status::unavailable(CLOSING));
    }
    let now = shared.deciders.now();
    // The table is asked where this replica leads. One that does not lead
    // has none to ask: the stream opens, and its requests are told.
    let (placed, presence) = match shared.leading(now) {
        Some(term) => {
            let presence = shared.present(gateway, now)?;
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
        let (id, term, decision, lease, encode) = match decided {
            wire::Decided::Decision {
                id,
                term,
                decision,
                lease,
                encode,
            } => (id, term, decision, lease, encode),
            wire::Decided::Frame(frame) => return frame,
        };
        let deciders = &self.shared.deciders;
        let now = deciders.now();
        let lease = lease.map(|leased| leased.sent_at(now));
        let stopped = if self.shared.leading(now) != Some(term) {
            Some((Reason::NotLeader, ReleaseReason::Abandoned))
        } else if lease.as_ref().is_some_and(wire::Leased::spent) {
            Some((Reason::BudgetGoneOnArrival, ReleaseReason::LateDecision))
        } else {
            None
        };
        let Some((reason, release)) = stopped else {
            return wire::decided(id, term, &decision, lease.as_ref(), encode);
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
        // The stream's place in the presence table is of one term: a
        // request that spends a credit under a term the stream holds no
        // place under asks the table, as an open does, whatever budget it
        // states.
        let Some(term) = self.shared.leading(received) else {
            let busy = deciders.busy(request.into_id(), Reason::NotLeader);
            self.own = Some((busy, true));
            return Ok(());
        };
        if self.placed != Some(term) {
            let presence = self.shared.present(self.gateway, received)?;
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
#[path = "../../../tests/unit/service/controller/decide/tests.rs"]
mod tests;
