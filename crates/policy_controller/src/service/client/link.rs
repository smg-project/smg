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
use crate::service::{
    controller::wire::{self, Refusal},
    proto::{
        busy, controller_frame, decide_answer, gateway_frame, released, Busy, ControllerFrame,
        DecideAnswer, DecideRequest, Decision, GatewayFrame, Released, Side, SpecAck, SpecPush,
        WatchWorkersRequest, Welcome,
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
    ) -> Result<Decision, DecisionUnavailable> {
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
    ) -> Result<Decision, DecisionUnavailable> {
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
            Answer::Decision(decision) => Ok(decision),
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
/// own functions: the lease a decode call names, then the facts.
fn in_range(request: &DecideRequest) -> Result<(), Error> {
    let checked = wire::asked_decode(request.decode_of).and_then(|_| match &request.facts {
        Some(facts) => wire::ranges(facts).map(|_| ()),
        None => Ok(()),
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
            Some(decide_answer::Answer::Decision(decision)) => Answer::Decision(decision),
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
        if let Some(Answer::Decision(decision)) = undelivered {
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
        if let Ok(Answer::Decision(decision)) = self.answer.try_recv() {
            self.shared.release(decision);
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Waker},
    };

    use tokio::time::{advance, timeout};

    use super::*;
    use crate::{
        document::spec::MAX_HEADERS,
        service::{
            client::{PolicyControllerClient, Principal},
            proto::{
                busy, controller_frame, decide_answer, released, Completed, Credit, Header, Lease,
                NotLeader, Released, RequestFacts, Side,
            },
        },
    };

    const WAIT: Duration = Duration::from_secs(5);
    const LEADER_TERM: u64 = 3;
    const WELCOME_TERM: u64 = 4;
    const ASKS: u8 = 64;
    /// The grace period a patient peer's welcome tells.
    const GRACE: Duration = Duration::from_secs(10);
    const GATEWAY: &str = "gateway";
    /// An address no test dials.
    const NOWHERE: &str = "http://127.0.0.1:1";

    /// One change of a request's facts.
    type Edit = fn(&mut RequestFacts);

    pub(crate) struct Peer {
        pub(crate) link: Link,
        pub(crate) reports: Pin<Box<dyn Stream<Item = GatewayFrame>>>,
        pub(crate) requests: Pin<Box<dyn Stream<Item = DecideRequest>>>,
        controller: Option<mpsc::UnboundedSender<Result<ControllerFrame, Status>>>,
        decide: Option<mpsc::UnboundedSender<Result<DecideAnswer, Status>>>,
        readers: [JoinHandle<()>; 2],
        // The floor the session's hello states.
        floor: u64,
    }

    impl Peer {
        pub(crate) fn new(window: u32) -> Self {
            Self::over(window, Endpoint::from_static(NOWHERE))
        }

        /// A peer whose welcome tells a grace period.
        pub(crate) fn patient(window: u32) -> Self {
            let link = Link {
                shared: Shared::new(),
            };
            Self::session(link, Endpoint::from_static(NOWHERE), welcome(window, GRACE))
        }

        fn over(window: u32, endpoint: Endpoint) -> Self {
            let link = Link {
                shared: Shared::new(),
            };
            Self::session(link, endpoint, welcome(window, Duration::ZERO))
        }

        /// Opens the next session of `link` as a redial does past its
        /// dial, and holds the controller's ends of it.
        fn session(link: Link, endpoint: Endpoint, told: Welcome) -> Self {
            let (report, reports) = mpsc::channel(FRAME_QUEUE);
            let (request, requests) = mpsc::unbounded_channel();
            let (controller, session) = mpsc::unbounded_channel();
            let (decide, decisions) = mpsc::unbounded_channel();
            let shared = &link.shared;
            let (seat, floor) = shared.seat(stub(endpoint), LEADER_TERM, report, request);
            let number = seat.session;
            let first = shared.hello(GATEWAY, floor);
            let reports = Box::pin(shared.session_frames(number, first, reports));
            let requests = Box::pin(shared.decide_requests(number, requests));
            let readers = shared
                .start(
                    number,
                    told,
                    WELCOME_TERM.max(floor),
                    UnboundedReceiverStream::new(session),
                    UnboundedReceiverStream::new(decisions),
                )
                .expect("the seated session starts");
            drop(seat);
            Self {
                link,
                reports,
                requests,
                controller: Some(controller),
                decide: Some(decide),
                readers,
                floor,
            }
        }

        /// The link's next session, under a patient welcome. Answers the
        /// peer of the session before it.
        pub(crate) fn redial(&mut self, window: u32) -> Self {
            let link = self.link.clone();
            let next = Self::session(link, Endpoint::from_static(NOWHERE), welcome(window, GRACE));
            mem::replace(self, next)
        }

        /// Ends the session by its stream's clean end, and waits for the
        /// link to read it.
        pub(crate) async fn end_session(&mut self) {
            let link = self.link.clone();
            closes(&link, || drop(self.controller.take())).await;
        }

        pub(crate) fn controller(&self, term: u64, frame: Option<controller_frame::Frame>) {
            self.controller
                .as_ref()
                .expect("the controller stream is open")
                .send(Ok(ControllerFrame {
                    term,
                    frame,
                    ..ControllerFrame::default()
                }))
                .expect("the session reader takes the frame");
        }

        pub(crate) fn answer(&self, id: u8, term: u64, answer: decide_answer::Answer) {
            self.decide
                .as_ref()
                .expect("the decide stream is open")
                .send(Ok(DecideAnswer {
                    request_id: if id == 0 { Vec::new() } else { vec![id] },
                    term,
                    answer: Some(answer),
                }))
                .expect("the decide reader takes the frame");
        }

        fn credit(&self, term: u64, window: u32) {
            self.answer(0, term, decide_answer::Answer::Credit(Credit { window }));
        }

        async fn at_term(&self, term: u64) {
            within(async {
                while self.link.term_floor() < term {
                    tokio::task::yield_now().await;
                }
                assert_eq!(self.link.term_floor(), term);
            })
            .await;
        }

        pub(crate) async fn read_hello(&mut self) {
            let first = within(self.reports.next()).await.expect("the hello");
            assert_eq!((first.seq, first.term_floor), (1, self.floor));
            assert!(matches!(first.frame, Some(gateway_frame::Frame::Hello(_))));
        }

        async fn readers_end(readers: [JoinHandle<()>; 2]) {
            for reader in readers {
                within(reader).await.expect("the reader ends cleanly");
            }
        }
    }

    async fn within<T>(future: impl Future<Output = T>) -> T {
        timeout(WAIT, future).await.expect("the wait ends in time")
    }

    pub(crate) fn poll<T>(future: Pin<&mut impl Future<Output = T>>) -> Poll<T> {
        future.poll(&mut Context::from_waker(Waker::noop()))
    }

    async fn closes(link: &Link, end: impl FnOnce()) {
        let mut closed = pin!(link.closed());
        assert!(poll(closed.as_mut()).is_pending());
        tokio::join!(biased; within(closed), async { end() });
        assert!(poll(pin!(link.closed())).is_ready());
    }

    /// A welcome that grants `window` asks and tells `grace` as its grace
    /// period.
    fn welcome(window: u32, grace: Duration) -> Welcome {
        Welcome {
            decide_window: window,
            grace_period_ms: u32::try_from(grace.as_millis()).expect("a grace period in range"),
            ..Welcome::default()
        }
    }

    fn stub(endpoint: Endpoint) -> Stub {
        let principal = Principal::of(GATEWAY).expect("a gateway id");
        PolicyControllerClient::with_interceptor(endpoint.connect_lazy(), principal)
    }

    /// Seats a session on `link` whose welcome has not come.
    fn seated(link: &Link) -> SessionPlace<'_> {
        let (report, _) = mpsc::channel(FRAME_QUEUE);
        let (request, _) = mpsc::unbounded_channel();
        let endpoint = Endpoint::from_static(NOWHERE);
        let shared = &link.shared;
        let (seat, _) = shared.seat(stub(endpoint), LEADER_TERM, report, request);
        seat
    }

    fn gone(lease: u64) -> gateway_frame::Frame {
        release(lease, Side::Single, released::Reason::ClientGone)
    }

    fn completed(lease: u64) -> gateway_frame::Frame {
        gateway_frame::Frame::Completed(Completed {
            lease_id: lease,
            side: Side::Single.into(),
            ..Completed::default()
        })
    }

    fn request(id: u8) -> DecideRequest {
        DecideRequest {
            request_id: vec![id],
            ..DecideRequest::default()
        }
    }

    fn release(lease: u64, side: Side, reason: released::Reason) -> gateway_frame::Frame {
        gateway_frame::Frame::Released(Released {
            what: Some(released::What::LeaseId(lease)),
            reason: reason.into(),
            side: side.into(),
        })
    }

    fn decision(lease: u64, sides: &[Side]) -> decide_answer::Answer {
        decide_answer::Answer::Decision(Decision {
            leases: sides
                .iter()
                .map(|&side| Lease {
                    id: lease,
                    side: side.into(),
                    ..Lease::default()
                })
                .collect(),
            ..Decision::default()
        })
    }

    #[tokio::test]
    async fn reports_are_numbered_and_stamped_as_they_leave_the_queue() {
        let mut peer = Peer::new(1);
        peer.read_hello().await;
        assert_eq!(peer.link.term_floor(), WELCOME_TERM);
        for id in [7, 3] {
            let link = peer.link.clone();
            std::thread::spawn(move || {
                link.queue_frame(release(id, Side::Single, released::Reason::ClientGone));
            })
            .join()
            .expect("the caller returns");
        }

        peer.controller(5, None);
        peer.at_term(5).await;
        let first = within(peer.reports.next()).await.expect("the first report");
        assert_eq!((first.seq, first.term_floor), (2, 5));
        assert_eq!(
            first.frame,
            Some(release(7, Side::Single, released::Reason::ClientGone))
        );

        peer.credit(6, 1);
        peer.at_term(6).await;
        let second = within(peer.reports.next())
            .await
            .expect("the second report");
        assert_eq!((second.seq, second.term_floor), (3, 6));
        assert_eq!(
            second.frame,
            Some(release(3, Side::Single, released::Reason::ClientGone))
        );
    }

    #[tokio::test]
    async fn a_full_report_queue_ends_the_session() {
        let mut peer = Peer::new(1);
        peer.read_hello().await;
        let mut asked = Box::pin(peer.link.ask(request(1), WAIT));
        assert!(poll(asked.as_mut()).is_pending());
        let report = release(7, Side::Single, released::Reason::ClientGone);
        for _ in 0..65_536 {
            peer.link.queue_frame(report.clone());
        }
        assert!(peer.link.stands());
        closes(&peer.link, || {
            peer.link.queue_frame(report);
            assert!(!peer.link.stands());
            assert!(matches!(
                peer.link.ended().as_deref(),
                Some(Error::FrameQueueFull)
            ));
        })
        .await;
        assert!(matches!(
            within(asked).await,
            Err(DecisionUnavailable::Ended)
        ));
        assert_eq!(within(peer.reports.next()).await, None);
        assert_eq!(within(peer.requests.next()).await, None);
        Peer::readers_end(peer.readers).await;
    }

    #[tokio::test]
    async fn asks_spend_credit_and_credit_frames_add_to_it() {
        let mut peer = Peer::new(1);
        assert_eq!(peer.link.welcome().decide_window, 1);
        assert!(matches!(
            peer.link.ask(DecideRequest::default(), WAIT).await,
            Err(DecisionUnavailable::Unsent(Error::RequestId(0)))
        ));
        let mut too_large = request(4);
        too_large.facts = Some(RequestFacts {
            model: "m".repeat(64 * 1024),
            ..RequestFacts::default()
        });
        assert!(matches!(
            peer.link.ask(too_large, WAIT).await,
            Err(DecisionUnavailable::Unsent(Error::TooLarge(_)))
        ));
        assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());

        let mut first = Box::pin(peer.link.ask(request(1), WAIT));
        assert!(poll(first.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
        assert!(matches!(
            poll(Box::pin(peer.link.ask(request(2), WAIT)).as_mut()),
            Poll::Ready(Err(DecisionUnavailable::NoCredit))
        ));
        assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());

        let busy = Busy {
            reason: busy::Reason::QueueDepth.into(),
            retry_after_micros: 125,
        };
        peer.answer(1, 4, decide_answer::Answer::Busy(busy));
        assert!(
            matches!(within(first).await, Err(DecisionUnavailable::Busy(told)) if told == busy)
        );
        assert!(matches!(
            peer.link.ask(request(2), WAIT).await,
            Err(DecisionUnavailable::NoCredit)
        ));
        peer.credit(5, 1);
        peer.credit(6, 1);
        peer.at_term(6).await;

        assert_eq!(peer.link.hashes_cut(), 0, "no cut frame was queued");
        let mut large = request(2);
        let blocks: Vec<_> = (0..8192).collect();
        large.facts = Some(RequestFacts {
            block_hashes: blocks.clone(),
            ..RequestFacts::default()
        });
        let mut second = Box::pin(peer.link.ask(large, WAIT));
        let mut third = Box::pin(peer.link.ask(request(3), WAIT));
        assert!(poll(second.as_mut()).is_pending());
        assert!(poll(third.as_mut()).is_pending());
        let fitted = within(peer.requests.next()).await.expect("a fitted ask");
        assert_eq!(fitted.request_id, [2]);
        let fitted = fitted.facts.expect("the fitted ask's facts").block_hashes;
        assert!(!fitted.is_empty() && fitted.len() < blocks.len() && blocks.starts_with(&fitted));
        assert_eq!(peer.link.hashes_cut(), blocks.len() - fitted.len());
        assert_eq!(within(peer.requests.next()).await, Some(request(3)));
        assert!(matches!(
            peer.link.ask(request(4), WAIT).await,
            Err(DecisionUnavailable::NoCredit)
        ));
        assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());
    }

    /// A frame out of the ranges the controller's check reads would end the
    /// stream for every caller: the link's door refuses it by the field,
    /// sends nothing, spends no credit and stands.
    #[tokio::test]
    async fn an_ask_out_of_the_calls_ranges_sends_nothing_and_the_link_stands() {
        let mut peer = Peer::new(1);
        let edits: [(&str, Edit); 4] = [
            ("facts.phase", |facts| facts.phase = 99),
            ("facts.hash_kind", |facts| facts.hash_kind = 99),
            ("facts.headers", |facts| {
                facts.headers = vec![Header::default(); MAX_HEADERS + 1];
            }),
            ("facts.headers.name", |facts| {
                facts.headers = vec![Header {
                    name: 256,
                    ..Header::default()
                }];
            }),
        ];
        for (field, edit) in edits {
            let mut wrong = request(1);
            let mut facts = RequestFacts::default();
            edit(&mut facts);
            wrong.facts = Some(facts);
            let refused = peer.link.ask(wrong, WAIT).await;
            assert!(
                matches!(refused, Err(DecisionUnavailable::Unsent(Error::OutOfRange(named))) if named == field),
                "{field}: {refused:?}"
            );
        }
        // A decode call that names lease 0 is refused here too, by its field.
        let wrong = DecideRequest {
            decode_of: Some(0),
            ..request(1)
        };
        let refused = peer.link.ask(wrong, WAIT).await;
        assert!(
            matches!(
                refused,
                Err(DecisionUnavailable::Unsent(Error::OutOfRange("decode_of")))
            ),
            "{refused:?}"
        );
        assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());
        assert!(peer.link.stands() && peer.link.ended().is_none());
        // The window of one is whole: the next ask is sent.
        let mut asked = Box::pin(peer.link.ask(request(1), WAIT));
        assert!(poll(asked.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
    }

    #[tokio::test]
    async fn sixty_four_asks_take_their_own_answers_in_reverse_order() {
        let mut peer = Peer::new(u32::from(ASKS));
        let mut asks: Vec<_> = (1..=ASKS)
            .map(|id| Box::pin(peer.link.ask(request(id), WAIT)))
            .collect();
        for ask in &mut asks {
            assert!(poll(ask.as_mut()).is_pending());
        }
        for id in 1..=ASKS {
            assert_eq!(within(peer.requests.next()).await, Some(request(id)));
        }
        for id in (1..=ASKS).rev() {
            peer.answer(id, 4, decision(u64::from(id), &[Side::Single]));
        }
        for (id, ask) in (1..=ASKS).zip(asks) {
            let answer = within(ask).await.expect("the ask gets a decision");
            assert_eq!(answer.leases[0].id, u64::from(id));
        }
    }

    #[tokio::test]
    async fn a_concurrent_ask_with_the_same_id_sends_nothing() {
        let mut peer = Peer::new(1);
        let mut first = Box::pin(peer.link.ask(request(1), WAIT));
        assert!(poll(first.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
        for credit in [0, 1] {
            peer.credit(u64::from(5 + credit), credit);
            peer.at_term(u64::from(5 + credit)).await;
            assert!(matches!(
                poll(Box::pin(peer.link.ask(request(1), WAIT)).as_mut()),
                Poll::Ready(Err(DecisionUnavailable::InFlight))
            ));
            assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());
        }
        peer.answer(1, 6, decision(11, &[Side::Single]));
        let first = within(first).await.expect("the original ask is answered");
        assert_eq!(first.leases[0].id, 11);
        let mut again = Box::pin(peer.link.ask(request(1), WAIT));
        assert!(poll(again.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
    }

    #[tokio::test(start_paused = true)]
    async fn the_budget_ends_the_wait_and_the_id_stays_taken_until_its_answer() {
        let mut peer = Peer::new(3);
        peer.read_hello().await;
        let budget = Duration::from_secs(1);
        let step = Duration::from_millis(1);
        let mut first = Box::pin(peer.link.ask(request(1), budget));
        assert!(poll(first.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
        advance(budget - step).await;
        assert!(poll(first.as_mut()).is_pending());
        advance(2 * step).await;
        assert!(matches!(
            within(first).await,
            Err(DecisionUnavailable::Late)
        ));

        // The first ask's answer is still due. A second ask under its id
        // would take that answer for its own: it is refused, unsent.
        assert!(matches!(
            poll(Box::pin(peer.link.ask(request(1), budget)).as_mut()),
            Poll::Ready(Err(DecisionUnavailable::InFlight))
        ));
        assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());
        peer.answer(1, 5, decision(7, &[Side::Single]));
        assert_eq!(
            within(peer.reports.next()).await.expect("a release").frame,
            Some(release(7, Side::Single, released::Reason::LateDecision))
        );
        // The answer freed the id: the next ask under it goes out, and
        // the answer it gets is its own.
        let mut again = Box::pin(peer.link.ask(request(1), budget));
        assert!(poll(again.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
        peer.answer(1, 5, decision(8, &[Side::Single]));
        let again = within(again).await.expect("the second ask's own answer");
        assert_eq!(again.leases[0].id, 8);

        assert!(matches!(
            poll(Box::pin(peer.link.ask(request(2), Duration::ZERO)).as_mut()),
            Poll::Ready(Err(DecisionUnavailable::Late))
        ));
        assert_eq!(within(peer.requests.next()).await, Some(request(2)));
    }

    #[tokio::test]
    async fn answers_without_an_ask_release_each_lease_and_drop_busy() {
        let mut peer = Peer::new(4);
        peer.read_hello().await;

        assert!(matches!(
            peer.link.ask(request(1), Duration::ZERO).await,
            Err(DecisionUnavailable::Late)
        ));
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
        // A pair is given back whole, in one frame that names no side.
        peer.answer(1, 4, decision(7, &[Side::Prefill, Side::Decode]));
        assert_eq!(
            within(peer.reports.next()).await.expect("a release").frame,
            Some(release(
                7,
                Side::Unspecified,
                released::Reason::LateDecision
            ))
        );

        let mut dropped = Box::pin(peer.link.ask(request(2), WAIT));
        assert!(poll(dropped.as_mut()).is_pending());
        drop(dropped);
        assert_eq!(within(peer.requests.next()).await, Some(request(2)));
        // The dropped ask's answer is still due: its id stays taken.
        assert!(matches!(
            poll(Box::pin(peer.link.ask(request(2), WAIT)).as_mut()),
            Poll::Ready(Err(DecisionUnavailable::InFlight))
        ));
        peer.answer(2, 4, decision(8, &[Side::Single]));

        let mut closed = peer
            .link
            .shared
            .begin(request(3), None, 0)
            .expect("the ask is sent");
        closed.answer.close();
        assert_eq!(within(peer.requests.next()).await, Some(request(3)));
        peer.answer(3, 4, decision(9, &[Side::Single]));

        let mut queued = Box::pin(peer.link.ask(request(4), WAIT));
        assert!(poll(queued.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(4)));
        peer.answer(4, 4, decision(10, &[Side::Single]));
        peer.answer(2, 5, decide_answer::Answer::Busy(Busy::default()));
        peer.at_term(5).await;
        drop(queued);
        for id in [8, 9, 10] {
            assert_eq!(
                within(peer.reports.next()).await.expect("a release").frame,
                Some(release(id, Side::Single, released::Reason::LateDecision))
            );
        }
        assert!(poll(Box::pin(peer.reports.next()).as_mut()).is_pending());
        assert!(peer.link.stands());
    }

    #[tokio::test]
    async fn both_streams_raise_the_floor_and_are_fenced_below_it() {
        for stale_session in [true, false] {
            let peer = Peer::new(1);
            peer.controller(5, None);
            peer.at_term(5).await;
            peer.answer(1, 6, decide_answer::Answer::Busy(Busy::default()));
            peer.at_term(6).await;
            let mut asked = Box::pin(peer.link.ask(request(1), WAIT));
            assert!(poll(asked.as_mut()).is_pending());
            closes(&peer.link, || {
                if stale_session {
                    peer.controller(5, None);
                } else {
                    peer.answer(1, 5, decision(7, &[Side::Single]));
                }
            })
            .await;
            assert!(matches!(
                within(asked).await,
                Err(DecisionUnavailable::Ended)
            ));
            assert!(!peer.link.stands());
            assert_eq!(peer.link.term_floor(), 6);
            assert!(matches!(
                peer.link.ended().as_deref(),
                Some(Error::BelowFloor { term: 5, floor: 6 })
            ));
        }
    }

    #[tokio::test]
    async fn either_stream_end_ends_every_waiter_and_refuses_new_work() {
        for session in [true, false] {
            for clean in [true, false] {
                check_end(session, Some(clean)).await;
            }
            check_end(session, None).await;
        }
    }

    #[tokio::test]
    async fn a_closed_queue_leaves_the_end_to_its_reader() {
        for session in [false, true] {
            let peer = Peer::new(2);
            let mut asked = pin!(peer.link.ask(request(1), WAIT));
            assert!(poll(asked.as_mut()).is_pending());
            if session {
                drop(peer.reports);
                // The report waits for the link's next session.
                peer.link.queue_frame(gone(7));
            } else {
                drop(peer.requests);
                for _ in 0..2 {
                    assert!(matches!(
                        peer.link.ask(request(2), WAIT).await,
                        Err(DecisionUnavailable::Ended)
                    ));
                }
            }
            assert!(peer.link.stands(), "the reader has seen no end");
            assert!(peer.link.ended().is_none());
            assert!(poll(asked.as_mut()).is_pending());

            let status = Status::unavailable("the stream was reset");
            if session {
                peer.controller
                    .expect("the controller stream is open")
                    .send(Err(status))
                    .expect("the session reader takes the status");
            } else {
                peer.decide
                    .expect("the decide stream is open")
                    .send(Err(status))
                    .expect("the decide reader takes the status");
            }
            assert!(matches!(
                within(asked).await,
                Err(DecisionUnavailable::Ended)
            ));
            Peer::readers_end(peer.readers).await;
            assert!(matches!(
                peer.link.ended().as_deref(),
                Some(Error::Status(status))
                    if status.code() == tonic::Code::Unavailable
                        && status.message() == "the stream was reset"
            ));
        }
    }

    /// Ends a link by one stream's clean end, by its status or by the
    /// controller's word that it leads no more, and reads what the waiters
    /// and the link say of it.
    async fn check_end(session: bool, closed: Option<bool>) {
        let mut peer = Peer::new(2);
        let link = peer.link.clone();
        let mut first = Box::pin(link.ask(request(1), WAIT));
        let mut second = Box::pin(link.ask(request(2), WAIT));
        assert!(poll(first.as_mut()).is_pending());
        assert!(poll(second.as_mut()).is_pending());
        let end = || match (session, closed) {
            (true, Some(true)) => drop(peer.controller.take()),
            (false, Some(true)) => drop(peer.decide.take()),
            (true, Some(false)) => peer
                .controller
                .as_ref()
                .expect("the controller stream is open")
                .send(Err(Status::unavailable("session ended")))
                .expect("the session reader takes the status"),
            (false, Some(false)) => peer
                .decide
                .as_ref()
                .expect("the decide stream is open")
                .send(Err(Status::unavailable("decide ended")))
                .expect("the decide reader takes the status"),
            (true, None) => peer.controller(
                5,
                Some(controller_frame::Frame::NotLeader(NotLeader {
                    leader_address: "replica-b:9000".to_owned(),
                    ..NotLeader::default()
                })),
            ),
            (false, None) => peer.answer(
                1,
                5,
                decide_answer::Answer::Busy(Busy {
                    reason: busy::Reason::NotLeader.into(),
                    ..Busy::default()
                }),
            ),
        };
        closes(&link, end).await;
        assert!(matches!(
            within(first).await,
            Err(DecisionUnavailable::Ended)
        ));
        assert!(matches!(
            within(second).await,
            Err(DecisionUnavailable::Ended)
        ));
        Peer::readers_end(peer.readers).await;
        assert!(!peer.link.stands());
        // The link keeps why it ended: a stream's status whole, and the
        // leader's address where the session named one.
        let ended = peer.link.ended().expect("an ended link says why");
        let said = match (session, closed, &*ended) {
            (_, Some(clean), Error::Status(status)) => {
                let (code, message) = match (clean, session) {
                    (true, true) => (tonic::Code::Ok, SESSION_ENDED),
                    (true, false) => (tonic::Code::Ok, DECIDE_ENDED),
                    (false, true) => (tonic::Code::Unavailable, "session ended"),
                    (false, false) => (tonic::Code::Unavailable, "decide ended"),
                };
                status.code() == code && status.message() == message
            }
            (true, None, Error::NotLeader(address)) => address == "replica-b:9000",
            (false, None, Error::NotLeader(address)) => address.is_empty(),
            _ => false,
        };
        assert!(said, "session {session}, closed {closed:?}: {ended:?}");
        assert!(matches!(
            poll(Box::pin(peer.link.ask(request(3), WAIT)).as_mut()),
            Poll::Ready(Err(DecisionUnavailable::Ended))
        ));
        assert!(matches!(
            peer.link.ask(DecideRequest::default(), WAIT).await,
            Err(DecisionUnavailable::Ended)
        ));
        // A report made now waits for the link's next session.
        peer.link.queue_frame(gone(7));
        assert_eq!(peer.link.frames_dropped(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn a_link_watch_open_has_the_answer_bound() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a listener that answers nothing");
        let address = listener.local_addr().expect("the listener's address");
        let endpoint = Endpoint::from_shared(format!("http://{address}")).expect("a URI");
        let peer = Peer::over(1, endpoint);
        let open = peer.link.watch_workers(WatchWorkersRequest::default());
        let opened = timeout(3 * WAIT, open)
            .await
            .expect("the watch's open ends inside its room");
        assert!(
            matches!(
                opened,
                Err(Error::NoAnswer {
                    what: "the watch's open",
                    wait: WAIT
                })
            ),
            "{opened:?}"
        );
        assert!(peer.link.stands());
    }

    /// An ended link holds no channel. The connection its channel made
    /// closes at the link's end though a clone of the link still stands,
    /// and a watch asked of the ended link is refused with no dial.
    #[tokio::test]
    async fn an_ended_link_lets_its_channel_go_and_refuses_a_watch() {
        use tokio::io::AsyncReadExt as _;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a listener that answers nothing");
        let address = listener.local_addr().expect("the listener's address");
        let endpoint = Endpoint::from_shared(format!("http://{address}")).expect("a URI");
        let peer = Peer::over(1, endpoint);
        let link = peer.link.clone();
        // The first call on the channel dials it. The listener takes the
        // connection and answers nothing, so the call is left once the
        // connection stands.
        let (accepted, unanswered) = {
            let open = pin!(link.watch_workers(WatchWorkersRequest::default()));
            tokio::join!(
                timeout(WAIT, listener.accept()),
                timeout(Duration::from_millis(200), open)
            )
        };
        assert!(unanswered.is_err(), "the listener answers nothing");
        let (mut connection, _) = accepted
            .expect("the channel dials inside the wait")
            .expect("the channel dialed the listener");

        closes(&link, || peer.controller(LEADER_TERM - 1, None)).await;
        let mut sent = Vec::new();
        let closed = timeout(WAIT, connection.read_to_end(&mut sent)).await;
        assert!(
            closed.is_ok(),
            "the connection stays open behind an ended link"
        );
        let refused = link.watch_workers(WatchWorkersRequest::default()).await;
        assert!(matches!(refused, Err(Error::LinkEnded)), "{refused:?}");
    }

    #[tokio::test]
    async fn dropping_the_last_clone_closes_both_streams_and_stops_both_readers() {
        let mut peer = Peer::new(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        drop(peer.link);
        assert!(link.stands());
        link.queue_frame(gone(7));
        assert!(matches!(
            link.ask(request(1), Duration::ZERO).await,
            Err(DecisionUnavailable::Late)
        ));
        drop(link);

        assert_eq!(within(peer.reports.next()).await, None);
        assert_eq!(within(peer.requests.next()).await, None);
        Peer::readers_end(peer.readers).await;
    }

    /// A request in flight holds a clone of the link. What it reports
    /// while no session stands waits, and rides the next session ahead of
    /// what is reported after.
    #[tokio::test]
    async fn reports_made_with_no_session_ride_the_next_one_in_their_order() {
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let held = peer.link.clone();
        peer.end_session().await;
        assert!(!held.stands());
        held.queue_frame(gone(7));
        held.queue_frame(gone(3));

        drop(peer.redial(1));
        assert!(held.stands() && held.ended().is_none());
        peer.read_hello().await;
        held.queue_frame(gone(9));
        for (seq, lease) in [(2, 7), (3, 3), (4, 9)] {
            let said = within(peer.reports.next()).await.expect("a report");
            assert_eq!((said.seq, said.frame), (seq, Some(gone(lease))));
        }
        assert_eq!(held.frames_dropped(), 0);
    }

    /// The link drops no report by its age: a line that waited well past
    /// the grace period is sent whole. The controller would read a
    /// completion as an outcome of the instant it arrives, so one that
    /// waited past the grace period goes as an expired release.
    #[tokio::test(start_paused = true)]
    async fn a_completion_that_waited_past_the_grace_period_is_said_as_an_expired_release() {
        let step = Duration::from_millis(1);
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        peer.end_session().await;
        link.queue_frame(completed(1));
        link.queue_frame(gone(2));
        advance(step).await;
        link.queue_frame(completed(3));
        advance(GRACE - step).await;

        drop(peer.redial(1));
        peer.read_hello().await;
        let expired = release(1, Side::Single, released::Reason::Expired);
        for said in [expired, gone(2), completed(3)] {
            let report = within(peer.reports.next()).await.expect("a report");
            assert_eq!(report.frame, Some(said));
        }
        assert_eq!(link.frames_dropped(), 0);
    }

    /// A line holds half of a session's queue. Past that it drops its
    /// oldest report, and the session that takes a full line has room for
    /// the reports made while it sends it.
    #[tokio::test]
    async fn a_line_past_its_bound_drops_its_oldest_and_counts_it() {
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        peer.end_session().await;
        let bound = u64::try_from(LINE).expect("the bound fits");
        for lease in 0..=bound {
            link.queue_frame(gone(lease));
        }
        assert_eq!(link.frames_dropped(), 1);

        drop(peer.redial(1));
        // No frame is read yet: the session's queue holds the whole line.
        link.queue_frame(gone(bound + 1));
        assert!(link.stands());
        peer.read_hello().await;
        let oldest = within(peer.reports.next()).await.expect("a report");
        assert_eq!(oldest.frame, Some(gone(1)));
        assert_eq!(link.frames_dropped(), 1);
    }

    /// A full queue ends the session, and the report that met it is the
    /// next session's first.
    #[tokio::test]
    async fn a_report_that_meets_a_full_queue_waits_for_the_next_session() {
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        for _ in 0..FRAME_QUEUE {
            link.queue_frame(gone(7));
        }
        closes(&link, || link.queue_frame(gone(8))).await;
        assert!(matches!(
            link.ended().as_deref(),
            Some(Error::FrameQueueFull)
        ));

        drop(peer.redial(1));
        peer.read_hello().await;
        let first = within(peer.reports.next()).await.expect("a report");
        assert_eq!(first.frame, Some(gone(8)));
        assert_eq!(link.frames_dropped(), 0);
    }

    /// A session whose open fails, or whose dial is dropped, takes no
    /// report with it and holds no wait: the line stands for the session
    /// after.
    #[tokio::test]
    async fn a_session_that_fails_to_open_takes_no_report_from_the_line() {
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        peer.end_session().await;
        link.queue_frame(gone(7));

        // The dial reached a leader, and it is dropped ahead of a welcome.
        let failed = seated(&link);
        assert!(!link.stands());
        assert!(poll(pin!(link.closed())).is_pending());
        link.queue_frame(gone(3));
        drop(failed);
        assert!(poll(pin!(link.closed())).is_ready());
        link.queue_frame(gone(9));

        drop(peer.redial(1));
        peer.read_hello().await;
        for lease in [7, 3, 9] {
            let said = within(peer.reports.next()).await.expect("a report");
            assert_eq!(said.frame, Some(gone(lease)));
        }
        assert_eq!(link.frames_dropped(), 0);
    }

    /// A session can start with its frame stream gone: the call was reset
    /// behind its welcome. Its queue refuses the waiting reports, and they
    /// keep their place in the line for the session after.
    #[tokio::test]
    async fn a_session_whose_stream_is_gone_at_its_start_leaves_the_line_standing() {
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        peer.end_session().await;
        link.queue_frame(gone(7));
        link.queue_frame(gone(3));

        // The welcome comes for a session whose report queue has no reader.
        let seat = seated(&link);
        let (controller, session) = mpsc::unbounded_channel();
        let (decide, decisions) = mpsc::unbounded_channel();
        let readers = link
            .shared
            .start(
                seat.session,
                welcome(1, GRACE),
                WELCOME_TERM,
                UnboundedReceiverStream::new(session),
                UnboundedReceiverStream::new(decisions),
            )
            .expect("the seated session starts");
        drop(seat);
        assert!(link.stands());
        // A report made on that session waits behind them.
        link.queue_frame(gone(9));

        drop(peer.redial(1));
        Peer::readers_end(readers).await;
        drop((controller, decide));
        peer.read_hello().await;
        for lease in [7, 3, 9] {
            let said = within(peer.reports.next()).await.expect("a report");
            assert_eq!(said.frame, Some(gone(lease)));
        }
        assert_eq!(link.frames_dropped(), 0);
    }

    /// A redial over a session that stands lets it go, as the controller
    /// does at the next hello: its waiters end, and its streams send no
    /// more.
    #[tokio::test]
    async fn a_session_that_stands_is_let_go_by_the_next_one() {
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        let mut asked = Box::pin(link.ask(request(1), WAIT));
        assert!(poll(asked.as_mut()).is_pending());
        link.queue_frame(gone(7));

        let mut closed = pin!(link.closed());
        assert!(poll(closed.as_mut()).is_pending());
        let mut first = peer.redial(1);
        within(closed).await;
        assert!(matches!(
            within(asked).await,
            Err(DecisionUnavailable::Ended)
        ));
        assert!(link.stands() && link.ended().is_none());
        // The ask and the report sat in the first session's queues: its
        // streams end ahead of them.
        assert_eq!(within(first.reports.next()).await, None);
        assert_eq!(within(first.requests.next()).await, None);
        Peer::readers_end(first.readers).await;
        peer.read_hello().await;
        link.queue_frame(gone(9));
        let said = within(peer.reports.next()).await.expect("a report");
        assert_eq!((said.seq, said.frame), (2, Some(gone(9))));

        // Until the next welcome the link says why the session ended, and
        // its floor does not fall to the dialed leader's.
        let next = seated(&link);
        assert!(matches!(link.ended().as_deref(), Some(Error::Redialed)));
        assert_eq!(link.term_floor(), WELCOME_TERM);
        drop(next);
        assert!(!link.stands());
    }

    /// An asker and a reader of a session can outlast it. What they do
    /// then reaches nothing of the session after: not an ask under the
    /// same id, not its credit, not its floor, not its end.
    #[tokio::test]
    async fn what_an_ended_session_left_behind_reaches_nothing_of_the_next() {
        let mut peer = Peer::patient(1);
        let link = peer.link.clone();
        let mut stale = Box::pin(link.ask(request(1), WAIT));
        assert!(poll(stale.as_mut()).is_pending());
        let first = link.shared.state.lock().session;
        peer.end_session().await;

        drop(peer.redial(1));
        let mut asked = Box::pin(link.ask(request(1), WAIT));
        assert!(poll(asked.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));

        // The first session's asker leaves, and its readers read on.
        drop(stale);
        let late = |answer| DecideAnswer {
            request_id: vec![1],
            term: WELCOME_TERM,
            answer: Some(answer),
        };
        let shared = &link.shared;
        shared.read_decide(first, late(decision(7, &[Side::Single])));
        let credit = decide_answer::Answer::Credit(Credit { window: 8 });
        shared.read_decide(first, late(credit));
        let above = ControllerFrame {
            term: WELCOME_TERM + 1,
            ..ControllerFrame::default()
        };
        shared.read_session(first, above);
        Shared::end(shared.state.lock(), first, Error::LinkEnded);
        assert!(link.stands() && link.ended().is_none());
        assert_eq!(link.term_floor(), WELCOME_TERM);
        assert!(poll(asked.as_mut()).is_pending());
        assert!(matches!(
            poll(Box::pin(link.ask(request(2), WAIT)).as_mut()),
            Poll::Ready(Err(DecisionUnavailable::NoCredit))
        ));

        peer.answer(1, WELCOME_TERM, decision(8, &[Side::Single]));
        let own = within(asked).await.expect("the ask's own answer");
        assert_eq!(own.leases[0].id, 8);
    }

    /// Two dials at once open their sessions one after the other. A later
    /// dial's hello then leaves after the earlier one's welcome, and the
    /// controller keeps the session the link keeps.
    #[tokio::test(start_paused = true)]
    async fn two_dials_at_once_take_their_turns() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a listener that answers nothing");
        let address = listener.local_addr().expect("the listener's address");
        let endpoint = Endpoint::from_shared(format!("http://{address}")).expect("a URI");
        let peer = Peer::new(1);
        let link = peer.link.clone();
        let started = Instant::now();
        let (first, second) = tokio::join!(
            link.redial(endpoint.clone(), GATEWAY),
            link.redial(endpoint, GATEWAY)
        );
        // Each dial waits out its own answer bound: side by side they
        // would end together.
        assert!(matches!(first, Err(Error::NoAnswer { .. })), "{first:?}");
        assert!(matches!(second, Err(Error::NoAnswer { .. })), "{second:?}");
        assert_eq!(started.elapsed(), 2 * ANSWER_WAIT);
        // A dial that reached no leader let no session go.
        assert!(link.stands());
    }

    fn pushed(generation: u64, document: &str) -> controller_frame::Frame {
        controller_frame::Frame::Spec(SpecPush {
            generation,
            document: document.as_bytes().to_vec(),
            format: "yaml".to_owned(),
            ..SpecPush::default()
        })
    }

    /// The holder's answer to the push of `generation`: taken where
    /// `refusal` is empty.
    fn acked(generation: u64, refusal: &str) -> gateway_frame::Frame {
        gateway_frame::Frame::SpecAck(SpecAck {
            generation,
            accepted: refusal.is_empty(),
            error: refusal.to_owned(),
        })
    }

    /// The generation a session's hello states, read off its first frame.
    async fn stated(peer: &mut Peer) -> u64 {
        let first = within(peer.reports.next()).await.expect("the hello");
        match first.frame {
            Some(gateway_frame::Frame::Hello(hello)) => Some(hello.generation),
            _ => None,
        }
        .expect("the first frame is a hello")
    }

    /// A controller pushes its document over the session. The holder reads
    /// the last push, and its answer rides the session that stands.
    #[tokio::test]
    async fn a_pushed_document_reaches_the_holder_and_its_answer_rides_the_session() {
        let mut peer = Peer::new(1);
        peer.read_hello().await;
        let mut pushes = peer.link.spec_pushes();
        assert!(poll(pin!(pushes.next())).is_pending());

        peer.controller(WELCOME_TERM, Some(pushed(7, "first")));
        let push = within(pushes.next()).await.expect("a push");
        assert_eq!(
            (push.generation, push.document.as_slice()),
            (7, &b"first"[..])
        );
        peer.link.spec_ack(7, Ok(()));
        let said = within(peer.reports.next()).await.expect("the answer");
        assert_eq!(said.frame, Some(acked(7, "")));

        // Two pushes between two reads: the holder reads the later one.
        peer.controller(WELCOME_TERM, Some(pushed(8, "second")));
        peer.controller(WELCOME_TERM + 1, Some(pushed(9, "third")));
        peer.at_term(WELCOME_TERM + 1).await;
        let push = within(pushes.next()).await.expect("a push");
        assert_eq!(push.generation, 9);
        assert!(poll(pin!(pushes.next())).is_pending());
        peer.link.spec_ack(9, Err("no such stage".to_owned()));
        let said = within(peer.reports.next()).await.expect("the answer");
        assert_eq!(said.frame, Some(acked(9, "no such stage")));

        // A reader that comes late reads the last push first.
        let mut late = peer.link.spec_pushes();
        let push = within(late.next()).await.expect("the last push");
        assert_eq!(push.generation, 9);
    }

    /// Every hello of the link states the generation its holder took last.
    /// A refused push moves nothing.
    #[tokio::test]
    async fn a_hello_states_the_generation_the_holder_took_last() {
        let mut peer = Peer::patient(1);
        assert_eq!(stated(&mut peer).await, 0);
        peer.link.spec_ack(7, Ok(()));
        peer.link.spec_ack(8, Err("no such stage".to_owned()));

        drop(peer.redial(1));
        assert_eq!(stated(&mut peer).await, 7);
    }

    /// An answer made with no session standing is not sent and does not
    /// wait: the next hello states what the holder runs.
    #[tokio::test]
    async fn an_answer_with_no_session_does_not_wait_for_the_next_one() {
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        peer.end_session().await;
        link.spec_ack(7, Ok(()));
        link.queue_frame(gone(3));

        drop(peer.redial(1));
        assert_eq!(stated(&mut peer).await, 7);
        let said = within(peer.reports.next()).await.expect("a report");
        assert_eq!(said.frame, Some(gone(3)));
        assert_eq!(link.frames_dropped(), 0);
    }

    /// An answer that meets a full queue ends the session, as a report
    /// does. The next hello states the generation.
    #[tokio::test]
    async fn an_answer_that_meets_a_full_queue_ends_the_session() {
        let mut peer = Peer::patient(1);
        peer.read_hello().await;
        let link = peer.link.clone();
        for _ in 0..FRAME_QUEUE {
            link.queue_frame(gone(7));
        }
        closes(&link, || link.spec_ack(7, Ok(()))).await;
        assert!(matches!(
            link.ended().as_deref(),
            Some(Error::FrameQueueFull)
        ));

        drop(peer.redial(1));
        assert_eq!(stated(&mut peer).await, 7);
    }
}
