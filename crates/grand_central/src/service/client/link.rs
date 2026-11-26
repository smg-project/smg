//! A host's link to its controller. The session takes lifecycle reports and
//! one decide stream carries the host's concurrent asks. A watch opens on
//! the same channel in a stream of its own.

use std::{collections::VecDeque, mem, pin::pin, sync::Arc, time::Duration};

use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHashMap;
use tokio::{
    sync::{mpsc, oneshot, watch, Mutex as AsyncMutex},
    task::JoinHandle,
    time::{timeout_at, Instant},
};
use tokio_stream::{
    once,
    wrappers::{ReceiverStream, UnboundedReceiverStream},
    Stream, StreamExt,
};
use tonic::{transport::Endpoint, Status};

use super::{
    dial_leader, fit, hello, numbered, Answer, Decide, Error, Session, Stub, Watch, ANSWER_WAIT,
    KEEPALIVE,
};
use crate::{
    engine::view::Phase,
    service::{
        controller::wire::{self, Refusal},
        proto::{
            busy, controller_frame, decide_answer, gateway_frame, released, Busy, ControllerFrame,
            DecideAnswer, DecideRequest, Decision, EncodePlaced, GatewayFrame, Released, Side,
            SpecAck, SpecPush, WatchWorkersRequest, Welcome,
        },
    },
};

/// About six seconds at ten thousand reports a second.
const FRAME_QUEUE: usize = 65_536;

/// The reports that wait for a session at most: half of what a session's
/// queue holds, so the next session takes the line whole and keeps room
/// for the reports made while it sends it.
const LINE: usize = FRAME_QUEUE / 2;

/// A stream's clean end, as the message of the `OK` status the link keeps
/// of it ([`Link::ended`]).
const SESSION_ENDED: &str = "the session ended";
const DECIDE_ENDED: &str = "the decide stream ended";

/// A host's session and one decide stream, shared by its requests.
///
/// A link is dialed again in place ([`Link::redial`]): its next session
/// opens in the state its clones share. A request in flight holds a clone,
/// so what it reports rides the session that stands when it reports,
/// whichever session its decision came over. A report made while no
/// session stands waits for the next one ([`Link::queue_frame`]).
///
/// A controller can push its policy document over the session
/// ([`Link::spec_pushes`]). The link's holder applies a push to its own
/// engine and answers it ([`Link::spec_ack`]). Every hello of the link
/// states the generation the holder took last.
#[derive(Clone, Debug)]
pub struct Link {
    shared: Arc<Shared>,
}

/// Why an ask has no usable decision from the controller. The host can
/// decide the request itself.
#[derive(Debug)]
#[non_exhaustive]
pub enum DecisionUnavailable {
    /// The stream holds no credit. Nothing was sent.
    NoCredit,
    /// The controller's reason for leaving the request to the host and the
    /// wait it names.
    Busy(Busy),
    /// No answer inside the budget. The link releases a decision that arrives
    /// later, and the ask's id stays taken until then.
    Late,
    /// An ask under this id is in flight already, or one that left has its
    /// answer still due. Nothing was sent.
    InFlight,
    /// No session stands, or the request queue closed. [`Link::ended`]
    /// tells why the last session ended.
    Ended,
    /// The request could not be sent, or a handle could not read the
    /// decision it got back. An unread decision's leases are released as late.
    Unsent(Error),
}

/// The documents a controller pushes over a link, as the link's holder
/// reads them ([`Link::spec_pushes`]).
#[derive(Debug)]
pub struct SpecPushes {
    last: watch::Receiver<Option<Arc<SpecPush>>>,
}

impl SpecPushes {
    /// Waits for a push this reader has not read, and answers the last one
    /// the link read: a later push replaces an earlier one unread. None
    /// once the link's last clone is gone.
    pub async fn next(&mut self) -> Option<Arc<SpecPush>> {
        loop {
            self.last.changed().await.ok()?;
            if let Some(push) = self.last.borrow_and_update().clone() {
                return Some(push);
            }
        }
    }
}

impl Link {
    /// Dials the endpoint, asks who leads, opens the session and one decide
    /// stream, and starts their readers on the calling runtime.
    ///
    /// The link sets its own keepalive on the endpoint, over any the caller
    /// set: a ping every [`KEEPALIVE`], idle or not, and [`ANSWER_WAIT`]
    /// for its answer. The link's readers wait with no bound of their own,
    /// as a watch does, so the keepalive is what ends a session whose
    /// controller went silent.
    pub async fn connect(endpoint: Endpoint, gateway_id: &str) -> Result<Self, Error> {
        let link = Self {
            shared: Shared::new(),
        };
        link.redial(endpoint, gateway_id).await?;
        Ok(link)
    }

    /// Opens the link's next session at `endpoint`, as [`Self::connect`]
    /// opened its first, in the state the link's clones share. The reports
    /// that waited for a session are sent first, in their order
    /// ([`Self::queue_frame`]).
    ///
    /// A controller keeps a gateway's leases while it holds the gateway
    /// present: through its sessions, and one grace period past the last
    /// one's end as the controller read it. The gateway's next session
    /// works on them. So a request that outlasts its session still reports
    /// its dispatch and its end where the controller reads them.
    ///
    /// A session that still stands is let go once the dial and its `Leader`
    /// ask have answered, ahead of the next session's welcome
    /// ([`Error::Redialed`]): the controller lets it go at the next
    /// session's hello. Two dials at once open their sessions one after
    /// the other, so the session the controller keeps is the one the link
    /// keeps. A dial dropped ahead of its session's start lets that
    /// session go.
    ///
    /// # Errors
    ///
    /// What the dial, the `Leader` ask, the session's open or the decide
    /// stream's open failed with. The link holds no new session then, and
    /// the reports that waited still wait.
    pub async fn redial(&self, endpoint: Endpoint, gateway_id: &str) -> Result<(), Error> {
        let shared = &self.shared;
        // One open at a time: a later dial's hello leaves after an earlier
        // one's welcome, so the controller and the link keep the same
        // session.
        let _turn = shared.dialing.lock().await;
        let endpoint = endpoint
            .http2_keep_alive_interval(KEEPALIVE)
            .keep_alive_timeout(ANSWER_WAIT)
            .keep_alive_while_idle(true);
        let (mut stub, floor) = dial_leader(endpoint.connect(), gateway_id, ANSWER_WAIT).await?;
        let (report, reports) = mpsc::channel(FRAME_QUEUE);
        let (request, requests) = mpsc::unbounded_channel();
        // Dropped ahead of the session's start, by an open that failed or
        // by a caller that left, the seat lets its session go.
        let (seat, floor) = shared.seat(stub.clone(), floor, report, request);
        let session = seat.session;
        let frames = shared.session_frames(session, shared.hello(gateway_id, floor), reports);
        let (controller, welcome, term) =
            Session::welcomed(&mut stub, frames, floor, ANSWER_WAIT).await?;
        let requests = shared.decide_requests(session, requests);
        let decisions = Decide::opened(&mut stub, requests, ANSWER_WAIT).await?;
        match shared.start(session, welcome, term, controller, decisions) {
            Some(_) => Ok(()),
            None => Err(Error::Redialed),
        }
    }

    /// The welcome of the session that stands, or of the last one.
    pub fn welcome(&self) -> Arc<Welcome> {
        Arc::clone(&self.shared.state.lock().welcome)
    }

    /// Whether a session and its decide stream stand.
    pub fn stands(&self) -> bool {
        self.shared.stands()
    }

    /// Why the last session ended, and none while one stands. The first
    /// cause is kept, including the whole status of the session or decide
    /// stream. A clean end reads as an `OK` status that names the stream.
    /// A frame below the term floor, a replica that stopped leading, an
    /// empty answer, a full report queue or the link's next dial
    /// ([`Error::Redialed`]) can also end a session.
    pub fn ended(&self) -> Option<Arc<Error>> {
        self.shared.state.lock().ended.clone()
    }

    /// Waits for the end of the session that stands, or of one whose
    /// welcome has not come. Returns at once where there is neither.
    pub async fn closed(&self) {
        let mut stopped = self.shared.state.lock().stopped.clone();
        let _ = stopped.changed().await;
    }

    /// The reports the line dropped at its bound, each the oldest of a
    /// full line ([`Self::queue_frame`]).
    pub fn frames_dropped(&self) -> usize {
        self.shared.state.lock().dropped
    }

    /// The documents the controller pushes over this link, for its holder.
    /// The reader's first read answers the last push the link read, where
    /// one came ahead of it. The welcome states the generation that stands
    /// at the controller ([`Self::welcome`]).
    pub fn spec_pushes(&self) -> SpecPushes {
        let mut last = self.shared.spec.subscribe();
        if last.borrow().is_some() {
            last.mark_changed();
        }
        SpecPushes { last }
    }

    /// Answers the push of `generation`: taken, or refused with why. A
    /// generation the holder took is the one every later hello of the link
    /// states. A refused one changes nothing.
    ///
    /// The answer rides the session that stands. With no session it is not
    /// sent and does not wait, also while a dial waits for its welcome: the
    /// next hello built states what the holder runs, and the controller
    /// pushes again where that is not its own. A hello that left ahead of
    /// the answer states the generation before it, so the holder can read a
    /// push of the generation it already runs. An answer that meets a full
    /// queue ends the session, as a report does.
    pub fn spec_ack(&self, generation: u64, taken: Result<(), String>) {
        self.shared.spec_ack(generation, taken);
    }

    /// Opens a watch of the controller's workers on the link's channel,
    /// within [`ANSWER_WAIT`]. The watch has its own stream, so the session
    /// and asks can proceed while it waits.
    ///
    /// The watch can outlive the session. Wait for [`Self::closed`]
    /// alongside [`Watch::next`] to stop following it when the session
    /// ends.
    ///
    /// # Errors
    ///
    /// [`Error::LinkEnded`] where no session stands: a session lets its
    /// channel go at its end, and nothing is dialed. Otherwise what the
    /// open answers.
    pub async fn watch_workers(&self, request: WatchWorkersRequest) -> Result<Watch, Error> {
        let stub = self
            .shared
            .state
            .lock()
            .open
            .as_ref()
            .map(|open| open.stub.clone());
        let Some(mut stub) = stub else {
            return Err(Error::LinkEnded);
        };
        Watch::open(&mut stub, request, ANSWER_WAIT).await
    }

    /// The highest term an answer carried.
    pub fn term_floor(&self) -> u64 {
        self.shared.state.lock().floor
    }

    /// The block hashes the link cut from the tails of the asks it queued,
    /// each cut to fit its ask in a frame: the count the contract has a
    /// sender keep on its own side.
    pub fn hashes_cut(&self) -> usize {
        self.shared.state.lock().cut
    }

    /// Sends one ask if the stream holds credit and waits up to `budget`
    /// for its answer. A budget past the clock's range waits with no bound.
    /// The answer is the controller's decision and, beside a pick of a
    /// request that stated encode items (`DecideRequest.encode`), where
    /// its items go (`DecideAnswer.encode`).
    ///
    /// An id stays taken until its ask's answer arrives, also where the
    /// asker left or its budget ended. An ask under it in that time is
    /// [`DecisionUnavailable::InFlight`], so no ask takes an answer made for an
    /// earlier one.
    ///
    /// A request the controller's check refuses is refused here, by the
    /// same check, and nothing is sent: a frame out of its ranges ends the
    /// stream it is sent on, and the link's stream is every caller's.
    pub async fn ask(
        &self,
        request: DecideRequest,
        budget: Duration,
    ) -> Result<(Decision, Option<EncodePlaced>), DecisionUnavailable> {
        self.ask_cut(request, 0, budget).await
    }

    /// [`Self::ask`] for a request whose maker left `left_out` block hashes
    /// out of it already. They are counted with the frame's own cut once
    /// the ask is queued ([`Self::hashes_cut`]).
    pub(crate) async fn ask_cut(
        &self,
        mut request: DecideRequest,
        left_out: usize,
        budget: Duration,
    ) -> Result<(Decision, Option<EncodePlaced>), DecisionUnavailable> {
        let cut = match fit(&mut request).and_then(|cut| in_range(&request).map(|()| cut)) {
            Ok(cut) => cut,
            Err(_) if !self.stands() => return Err(DecisionUnavailable::Ended),
            Err(error) => return Err(DecisionUnavailable::Unsent(error)),
        };
        let deadline = Instant::now().checked_add(budget);
        let mut pending = self.shared.begin(request, deadline, cut + left_out)?;
        if budget.is_zero() {
            return Err(DecisionUnavailable::Late);
        }
        let answer = match deadline {
            Some(at) => timeout_at(at, &mut pending.answer)
                .await
                .map_err(|_| DecisionUnavailable::Late)?,
            None => (&mut pending.answer).await,
        };
        match answer.map_err(|_| DecisionUnavailable::Ended)? {
            Answer::Decision(decision, encode) => Ok((decision, encode)),
            Answer::Busy(busy) => Err(DecisionUnavailable::Busy(busy)),
        }
    }

    /// Queues a report for the session that stands, without waiting for
    /// room. With no session to take it the report waits in the link's
    /// line, and the next session sends the line first, in its order.
    ///
    /// The line is bounded by its length alone: half of what a session's
    /// queue holds. Past that it drops its oldest report for a new one
    /// and counts it ([`Self::frames_dropped`]). The link drops no report
    /// by its age. Whether the controller still holds a lease is the
    /// controller's to know, and a report of a lease it let go is no error
    /// there.
    ///
    /// A completion that waited past the grace period the last welcome
    /// told is said as a release with the reason `EXPIRED`. Its lease
    /// ends, and its outcome is not said: the controller would read it as
    /// one of the instant it arrives.
    ///
    /// A full queue ends the session: it took no report for as long as its
    /// queue covers. The report that met it waits for the next session,
    /// and the reports the queue held are lost with it.
    pub fn queue_frame(&self, frame: gateway_frame::Frame) {
        self.shared.report(frame);
    }

    /// Gives back the leases of a decision its host could not read.
    pub(crate) fn release_late(&self, decision: Decision) {
        self.shared.release(decision);
    }
}

/// The ranges the controller's check reads in a request, by that check's
/// own functions: the lease a decode call names, then the facts, then the
/// items of an encode request. A request with no facts reads as one that
/// leaves every field of them unset, a single's.
fn in_range(request: &DecideRequest) -> Result<(), Error> {
    let checked = wire::asked_decode(request.decode_of).and_then(|decode_of| {
        let phase = match &request.facts {
            Some(facts) => wire::ranges(facts)?.0,
            None => Phase::Single,
        };
        wire::stated_items(request.encode.as_ref(), phase, decode_of)
    });
    match checked {
        Ok(()) => Ok(()),
        Err(Refusal::OutOfRange(field)) => Err(Error::OutOfRange(field)),
    }
}

#[derive(Debug)]
struct Shared {
    state: Mutex<State>,
    // Held over a session's open: one dial at a time.
    dialing: AsyncMutex<()>,
    // The document the controller pushed last, for the link's holder.
    spec: watch::Sender<Option<Arc<SpecPush>>>,
}

#[derive(Debug)]
struct State {
    floor: u64,
    // The session that stands: its welcome came, and neither of its
    // streams has ended.
    open: Option<Open>,
    // The session whose open has not ended. A report waits in the line
    // until it has, so an open that fails takes no report with it.
    seated: Option<Open>,
    // Why the last session ended: the first cause, written as the queues
    // are taken. None while a session stands.
    ended: Option<Arc<Error>>,
    // The block hashes cut from the asks the link queued.
    cut: usize,
    // The number of the session seated last. A stream, a reader, an end and
    // a waiting ask act on their own session alone: what an ended session
    // left behind reaches nothing of the next.
    session: u64,
    // The welcome of the session that stands, or of the last one.
    welcome: Arc<Welcome>,
    // Closed at the end of the session seated last, for who waits on it.
    stopped: watch::Receiver<()>,
    // The reports made with no session to take them, in their order, each
    // with the instant it was made.
    line: VecDeque<(Instant, gateway_frame::Frame)>,
    // The reports the line dropped at its bound.
    dropped: usize,
    // The controller's generation whose document the holder took last.
    // Every hello states it.
    generation: u64,
}

impl State {
    fn observe(&mut self, term: u64) -> Option<&mut Open> {
        let open = self.open.as_mut()?;
        if term < self.floor {
            return None;
        }
        self.floor = term;
        Some(open)
    }

    /// Whether the session `session` is seated or stands.
    fn holds(&self, session: u64) -> bool {
        self.session == session && (self.open.is_some() || self.seated.is_some())
    }

    /// Takes the queues and waiters of the session that stands, and writes
    /// why it ended. The caller drops them outside the lock.
    fn close(&mut self, why: Error) -> Option<Open> {
        let open = self.open.take()?;
        self.ended = Some(Arc::new(why));
        Some(open)
    }

    /// A report with no session to take it joins the line for the next
    /// one. A line at its bound drops its oldest report for it. No report
    /// is dropped by its age: only the controller knows whether it still
    /// holds the report's lease.
    fn wait(&mut self, frame: gateway_frame::Frame) {
        if self.line.len() == LINE {
            self.line.pop_front();
            self.dropped += 1;
        }
        self.line.push_back((Instant::now(), frame));
    }
}

#[derive(Debug)]
struct Open {
    // The session's channel, kept for a watch opened on it. It leaves with
    // the queues, so an ended session holds no connection open.
    stub: Stub,
    reports: mpsc::Sender<gateway_frame::Frame>,
    // Every queued request has spent credit. A grant can exceed the
    // welcome's window, so the queue has no separate limit.
    requests: mpsc::UnboundedSender<DecideRequest>,
    credit: u32,
    pending: FxHashMap<Arc<[u8]>, Waiting>,
    // Dropped with the queues and waiters to wake both stream readers.
    _stop: watch::Sender<()>,
}

/// A session's place until its start. Dropped ahead of that, by an open
/// that failed or by a caller that left, it lets the session go.
#[derive(Debug)]
struct SessionPlace<'a> {
    shared: &'a Shared,
    session: u64,
}

impl Drop for SessionPlace<'_> {
    fn drop(&mut self) {
        self.shared.unseat(self.session);
    }
}

/// A report as its session says it after a wait in the line. A completion
/// that waited past the grace period goes as an expired release: its lease
/// ends, and the controller reads no outcome as one of the instant it
/// arrives. Every other report goes as it was made.
fn waited(frame: gateway_frame::Frame, late: bool) -> gateway_frame::Frame {
    match frame {
        gateway_frame::Frame::Completed(completed) if late => {
            gateway_frame::Frame::Released(Released {
                what: Some(released::What::LeaseId(completed.lease_id)),
                reason: released::Reason::Expired.into(),
                side: completed.side,
            })
        }
        frame => frame,
    }
}

#[derive(Debug)]
struct Waiting {
    deadline: Option<Instant>,
    // `None` once the asker took the answer, or left ahead of it.
    answer: Option<oneshot::Sender<Answer>>,
    // Whether the answer arrived. The id stays in the table until it has
    // and the asker left: an asker that leaves first keeps the id taken,
    // so no later ask under it takes the answer still due.
    answered: bool,
}

impl Waiting {
    /// The asker's end for the answer that arrived, where the asker
    /// still waits inside its budget.
    fn take(&mut self) -> Option<oneshot::Sender<Answer>> {
        self.answered = true;
        if self.deadline.is_some_and(|at| Instant::now() >= at) {
            return None;
        }
        self.answer.take()
    }

    /// Whether the asker left ahead of its answer.
    fn left(&self) -> bool {
        !self.answered && self.answer.is_none()
    }
}

impl Shared {
    /// A link's state ahead of its first session.
    fn new() -> Arc<Self> {
        // No session is seated: who waits for one's end returns at once.
        let (_, stopped) = watch::channel(());
        Arc::new(Self {
            state: Mutex::new(State {
                floor: 0,
                open: None,
                seated: None,
                ended: None,
                cut: 0,
                session: 0,
                welcome: Arc::default(),
                stopped,
                line: VecDeque::new(),
                dropped: 0,
                generation: 0,
            }),
            dialing: AsyncMutex::new(()),
            spec: watch::Sender::new(None),
        })
    }

    /// Seats the link's next session and lets go the one before it. Answers
    /// the session's seat, and the floor its hello states: the link's
    /// floor never falls. The session stands once its welcome comes
    /// ([`Self::start`]).
    fn seat(
        &self,
        stub: Stub,
        floor: u64,
        reports: mpsc::Sender<gateway_frame::Frame>,
        requests: mpsc::UnboundedSender<DecideRequest>,
    ) -> (SessionPlace<'_>, u64) {
        let (stop, stopped) = watch::channel(());
        let mut state = self.state.lock();
        let before = (state.close(Error::Redialed), state.seated.take());
        state.floor = state.floor.max(floor);
        state.session += 1;
        state.stopped = stopped;
        state.seated = Some(Open {
            stub,
            reports,
            requests,
            credit: 0,
            pending: FxHashMap::default(),
            _stop: stop,
        });
        let (session, floor) = (state.session, state.floor);
        drop(state);
        drop(before);
        let seat = SessionPlace {
            shared: self,
            session,
        };
        (seat, floor)
    }

    /// Lets go the session `session` where it has not started. The line
    /// stands as it was.
    fn unseat(&self, session: u64) {
        let mut state = self.state.lock();
        let seated = if state.session == session {
            state.seated.take()
        } else {
            None
        };
        drop(state);
        drop(seated);
    }

    fn session_frames(
        self: &Arc<Self>,
        session: u64,
        first: GatewayFrame,
        queued: mpsc::Receiver<gateway_frame::Frame>,
    ) -> impl Stream<Item = GatewayFrame> + use<> {
        let shared = Arc::downgrade(self);
        let mut seq = first.seq;
        once(first).chain(ReceiverStream::new(queued).map_while(move |frame| {
            let shared = shared.upgrade()?;
            let state = shared.state.lock();
            state.holds(session).then_some(())?;
            seq += 1;
            Some(numbered(seq, state.floor, frame))
        }))
    }

    fn decide_requests(
        self: &Arc<Self>,
        session: u64,
        queued: mpsc::UnboundedReceiver<DecideRequest>,
    ) -> impl Stream<Item = DecideRequest> + use<> {
        let shared = Arc::downgrade(self);
        UnboundedReceiverStream::new(queued).map_while(move |request| {
            let held = shared.upgrade()?.state.lock().holds(session);
            held.then_some(request)
        })
    }

    /// The welcome of the session `session` came: the session stands from
    /// here with its credit, and its two readers start. The reports that
    /// waited are queued first, in their order. A queue whose stream is
    /// gone takes none of them: they keep their place for the session
    /// after. None where the session was let go meanwhile.
    fn start(
        self: &Arc<Self>,
        session: u64,
        welcome: Welcome,
        term: u64,
        controller: impl Stream<Item = Result<ControllerFrame, Status>> + Send + 'static,
        decisions: impl Stream<Item = Result<DecideAnswer, Status>> + Send + 'static,
    ) -> Option<[JoinHandle<()>; 2]> {
        let (stopped, line) = {
            let mut state = self.state.lock();
            if state.session != session {
                return None;
            }
            let mut open = state.seated.take()?;
            // The grace period the waiting reports' leases lived under.
            let grace = Duration::from_millis(u64::from(state.welcome.grace_period_ms));
            let mut line = mem::take(&mut state.line);
            while let Some((made, frame)) = line.pop_front() {
                let late = made.elapsed() >= grace;
                // The line is no longer than half the session's queue, so
                // only a queue whose stream is gone refuses. What it
                // refuses waits for the next session with the rest, as a
                // report made on this one does.
                if let Err(refused) = open.reports.try_send(waited(frame, late)) {
                    line.push_front((made, refused.into_inner()));
                    state.line = mem::take(&mut line);
                    break;
                }
            }
            if term >= state.floor {
                state.floor = term;
                open.credit = welcome.decide_window;
            }
            state.welcome = Arc::new(welcome);
            state.open = Some(open);
            state.ended = None;
            (state.stopped.clone(), line)
        };
        // The line's buffer is freed outside the lock.
        drop(line);
        Some([
            self.reader(
                session,
                stopped.clone(),
                controller,
                Self::read_session,
                SESSION_ENDED,
            ),
            self.reader(session, stopped, decisions, Self::read_decide, DECIDE_ENDED),
        ])
    }

    fn stands(&self) -> bool {
        self.state.lock().open.is_some()
    }

    /// Takes the queues and waiters of the session `session` together,
    /// then drops them outside the lock. Neither stream can send new
    /// frames past this point. The end that takes them writes why: a later
    /// end finds none, and its cause is dropped. So is the end of a
    /// session that is not the one seated last.
    fn end(mut state: MutexGuard<'_, State>, session: u64, why: Error) {
        let open = if state.session == session {
            state.close(why)
        } else {
            None
        };
        drop(state);
        drop(open);
    }

    /// The hello of the link's next session. It states the generation the
    /// holder took last.
    fn hello(&self, gateway_id: &str, floor: u64) -> GatewayFrame {
        hello(gateway_id, floor, self.state.lock().generation)
    }

    fn spec_ack(&self, generation: u64, taken: Result<(), String>) {
        let mut state = self.state.lock();
        if taken.is_ok() {
            state.generation = generation;
        }
        let frame = gateway_frame::Frame::SpecAck(SpecAck {
            generation,
            accepted: taken.is_ok(),
            error: taken.err().unwrap_or_default(),
        });
        // An answer waits for no session: the next hello states what the
        // holder runs.
        let Some(open) = state.open.as_ref() else {
            return;
        };
        // A full queue ends the session, as it does under a report.
        if let Err(mpsc::error::TrySendError::Full(_)) = open.reports.try_send(frame) {
            let full = state.close(Error::FrameQueueFull);
            drop(state);
            drop(full);
        }
    }

    fn report(&self, frame: gateway_frame::Frame) {
        let mut state = self.state.lock();
        let Some(open) = state.open.as_ref() else {
            state.wait(frame);
            return;
        };
        match open.reports.try_send(frame) {
            Ok(()) => {}
            // The session took no report for as long as its queue covers:
            // it ends, and this report waits for the next one.
            Err(mpsc::error::TrySendError::Full(frame)) => {
                let full = state.close(Error::FrameQueueFull);
                state.wait(frame);
                drop(state);
                drop(full);
            }
            // The session's stream is gone, and its reader has yet to say
            // so.
            Err(mpsc::error::TrySendError::Closed(frame)) => state.wait(frame),
        }
    }

    fn begin(
        &self,
        request: DecideRequest,
        deadline: Option<Instant>,
        cut: usize,
    ) -> Result<Pending<'_>, DecisionUnavailable> {
        let mut state = self.state.lock();
        let open = state.open.as_mut().ok_or(DecisionUnavailable::Ended)?;
        if open.pending.contains_key(request.request_id.as_slice()) {
            return Err(DecisionUnavailable::InFlight);
        }
        if open.credit == 0 {
            return Err(DecisionUnavailable::NoCredit);
        }
        let id = Arc::<[u8]>::from(request.request_id.as_slice());
        if open.requests.send(request).is_err() {
            return Err(DecisionUnavailable::Ended);
        }
        open.credit -= 1;
        // A reader takes this same lock before it can deliver an answer.
        let (sender, answer) = oneshot::channel();
        open.pending.insert(
            Arc::clone(&id),
            Waiting {
                deadline,
                answer: Some(sender),
                answered: false,
            },
        );
        // The cut of an ask is counted once the ask is queued.
        state.cut += cut;
        Ok(Pending {
            shared: self,
            session: state.session,
            id,
            answer,
        })
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "dropping the link closes its streams and wakes both readers"
    )]
    fn reader<T>(
        self: &Arc<Self>,
        session: u64,
        mut stopped: watch::Receiver<()>,
        frames: impl Stream<Item = Result<T, Status>> + Send + 'static,
        read: fn(&Self, u64, T),
        clean_end: &'static str,
    ) -> JoinHandle<()>
    where
        T: Send + 'static,
    {
        // The readers hold the link only while handling a frame. Dropping
        // the last handle drops the queues and wakes both readers.
        let shared = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut frames = pin!(frames);
            loop {
                let next = tokio::select! {
                    biased;
                    _ = stopped.changed() => break,
                    next = frames.next() => next,
                };
                let Some(shared) = shared.upgrade() else {
                    break;
                };
                let status = match next {
                    Some(Ok(frame)) => {
                        read(&shared, session, frame);
                        continue;
                    }
                    Some(Err(status)) => status,
                    None => Status::ok(clean_end),
                };
                Self::end(shared.state.lock(), session, status.into());
                break;
            }
        })
    }

    fn read_session(&self, session: u64, frame: ControllerFrame) {
        let mut state = self.state.lock();
        // A frame its reader held across its session's end.
        if state.session != session {
            return;
        }
        let (term, floor) = (frame.term, state.floor);
        if state.observe(term).is_none() {
            Self::end(state, session, Error::BelowFloor { term, floor });
            return;
        }
        match frame.frame {
            Some(controller_frame::Frame::NotLeader(told)) => {
                Self::end(state, session, Error::NotLeader(told.leader_address));
            }
            // The holder reads the last push: a later one replaces it. The
            // one it replaced is freed outside the lock.
            Some(controller_frame::Frame::Spec(push)) => {
                let before = self.spec.send_replace(Some(Arc::new(push)));
                drop(state);
                drop(before);
            }
            _ => {}
        }
    }

    fn read_decide(&self, session: u64, frame: DecideAnswer) {
        let mut state = self.state.lock();
        // A frame its reader held across its session's end. The next
        // session's credit and asks are none of its.
        if state.session != session {
            return;
        }
        let (term, floor) = (frame.term, state.floor);
        let Some(open) = state.observe(term) else {
            Self::end(state, session, Error::BelowFloor { term, floor });
            return;
        };
        let answer = match frame.answer {
            Some(decide_answer::Answer::Credit(credit)) => {
                open.credit = open.credit.saturating_add(credit.window);
                return;
            }
            Some(decide_answer::Answer::Decision(decision)) => {
                Answer::Decision(decision, frame.encode)
            }
            Some(decide_answer::Answer::Busy(busy)) if busy.reason() != busy::Reason::NotLeader => {
                Answer::Busy(busy)
            }
            Some(decide_answer::Answer::Busy(_)) => {
                Self::end(state, session, Error::NotLeader(String::new()));
                return;
            }
            None => {
                let nothing = "an answer of the decide stream carries nothing";
                Self::end(state, session, Error::NoDecision(nothing.to_owned()));
                return;
            }
        };
        let id = frame.request_id.as_slice();
        let waiting = open.pending.get_mut(id);
        // An asker that left ahead of this answer kept its id for it.
        let left = waiting.as_ref().is_some_and(|waiting| waiting.left());
        let undelivered = match waiting.and_then(Waiting::take) {
            Some(sender) => sender.send(answer).err(),
            None => Some(answer),
        };
        if left {
            open.pending.remove(id);
        }
        drop(state);
        if let Some(Answer::Decision(decision, _)) = undelivered {
            self.release(decision);
        }
    }

    fn release(&self, decision: Decision) {
        let mut leases = decision.leases.iter().peekable();
        while let Some(lease) = leases.next() {
            // A pair's two sides come under one id. One frame that names no
            // side gives back both bookings, whatever label each came
            // under.
            let pair = leases.next_if(|next| next.id == lease.id).is_some();
            let side = if pair {
                Side::Unspecified
            } else {
                lease.side()
            };
            let frame = gateway_frame::Frame::Released(Released {
                what: Some(released::What::LeaseId(lease.id)),
                reason: released::Reason::LateDecision.into(),
                // An unknown side gives back the whole lease.
                side: side.into(),
            });
            self.report(frame);
        }
    }
}

struct Pending<'a> {
    shared: &'a Shared,
    // The session the ask went out on.
    session: u64,
    id: Arc<[u8]>,
    answer: oneshot::Receiver<Answer>,
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        self.answer.close();
        let mut state = self.shared.state.lock();
        // A later session's table holds nothing of this ask.
        let own = state.session == self.session;
        if let Some(open) = state.open.as_mut().filter(|_| own) {
            match open.pending.get_mut(self.id.as_ref()) {
                // The asker leaves ahead of its answer. The id stays taken
                // until that answer arrives: an ask under it in between
                // would take an answer made for this one.
                Some(waiting) if !waiting.answered => waiting.answer = None,
                _ => {
                    open.pending.remove(self.id.as_ref());
                }
            }
        }
        drop(state);
        // A send that won the race with close may still be in the channel.
        if let Ok(Answer::Decision(decision, _)) = self.answer.try_recv() {
            self.shared.release(decision);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/client/link/tests.rs"]
pub(super) mod tests;
