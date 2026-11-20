//! A host's link to its controller. The session takes lifecycle reports and
//! one decide stream carries the host's concurrent asks. A watch opens on
//! the same channel in a stream of its own.

use std::{pin::pin, sync::Arc, time::Duration};

use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHashMap;
use tokio::{
    sync::{mpsc, oneshot, watch},
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
    controller::wire::{self, Refusal},
    proto::{
        busy, controller_frame, decide_answer, gateway_frame, released, Busy, ControllerFrame,
        DecideAnswer, DecideRequest, Decision, GatewayFrame, Released, Side, WatchWorkersRequest,
        Welcome,
    },
};

/// About six seconds at ten thousand reports a second.
const REPORT_QUEUE: usize = 65_536;

/// A stream's clean end, as the message of the `OK` status the link keeps
/// of it ([`Link::ended`]).
const SESSION_ENDED: &str = "the session ended";
const DECIDE_ENDED: &str = "the decide stream ended";

/// A host's session and one decide stream, shared by its requests.
#[derive(Clone, Debug)]
pub struct Link {
    welcome: Arc<Welcome>,
    shared: Arc<Shared>,
}

/// Why an ask has no usable decision from the controller. The host can
/// decide the request itself.
#[derive(Debug)]
pub enum Unanswered {
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
    /// The link ended or the request queue closed. [`Link::ended`] tells
    /// the cause once the link has ended.
    Ended,
    /// The request could not be sent, or a handle could not read the
    /// decision it got back. An unread decision's leases are released as late.
    Unsent(Error),
}

/// The report queue was full or closed.
#[derive(Debug)]
pub struct Full;

impl Link {
    /// Dials the endpoint, asks who leads, opens the session and one decide
    /// stream, and starts their readers on the calling runtime.
    ///
    /// The link sets its own keepalive on the endpoint, over any the caller
    /// set: a ping every [`KEEPALIVE`], idle or not, and [`ANSWER_WAIT`]
    /// for its answer. The link's readers wait with no bound of their own,
    /// as a watch does, so the keepalive is what ends a link whose
    /// controller went silent.
    pub async fn connect(endpoint: Endpoint, gateway_id: &str) -> Result<Self, Error> {
        let endpoint = endpoint
            .http2_keep_alive_interval(KEEPALIVE)
            .keep_alive_timeout(ANSWER_WAIT)
            .keep_alive_while_idle(true);
        let (mut stub, floor) = dial_leader(endpoint.connect(), gateway_id, ANSWER_WAIT).await?;
        let (report, reports) = mpsc::channel(REPORT_QUEUE);
        let (request, requests) = mpsc::unbounded_channel();
        let shared = Shared::new(stub.clone(), floor, report, request);
        let frames = shared.session_frames(hello(gateway_id, floor), reports);
        let (controller, welcome, term) =
            Session::welcomed(&mut stub, frames, floor, ANSWER_WAIT).await?;
        let decisions =
            Decide::opened(&mut stub, shared.decide_requests(requests), ANSWER_WAIT).await?;
        let (link, _) = shared.start(welcome, term, controller, decisions);
        Ok(link)
    }

    /// The welcome that answered the session's hello.
    pub fn welcome(&self) -> &Welcome {
        &self.welcome
    }

    /// Whether the session and decide stream still stand.
    pub fn stands(&self) -> bool {
        self.shared.stands()
    }

    /// Why the link ended, and none while it stands. The first cause is
    /// kept, including the whole status of the session or decide stream.
    /// A clean end reads as an `OK` status that names the stream. A frame
    /// below the term floor, a replica that stopped leading, an empty
    /// answer or a full report queue can also end the link.
    pub fn ended(&self) -> Option<Arc<Error>> {
        self.shared.state.lock().ended.clone()
    }

    /// Waits for the link to end, returning at once if it already has.
    pub async fn closed(&self) {
        let _ = self.shared.stopped.clone().changed().await;
    }

    /// Opens a watch of the controller's workers on the link's channel,
    /// within [`ANSWER_WAIT`]. The watch has its own stream, so the session
    /// and asks can proceed while it waits.
    ///
    /// The watch can outlive the link. Wait for [`Self::closed`] alongside
    /// [`Watch::next`] to stop following it when the link ends.
    ///
    /// # Errors
    ///
    /// [`Error::LinkEnded`] where the link had ended: it lets its channel
    /// go at its end, and nothing is dialed. Otherwise what the open
    /// answers.
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
    /// [`Unanswered::InFlight`], so no ask takes an answer made for an
    /// earlier one.
    ///
    /// A request the controller's check refuses is refused here, by the
    /// same check, and nothing is sent: a frame out of its ranges ends the
    /// stream it is sent on, and the link's stream is every caller's.
    pub async fn ask(
        &self,
        request: DecideRequest,
        budget: Duration,
    ) -> Result<Decision, Unanswered> {
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
    ) -> Result<Decision, Unanswered> {
        let cut = match fit(&mut request).and_then(|cut| in_range(&request).map(|()| cut)) {
            Ok(cut) => cut,
            Err(_) if !self.stands() => return Err(Unanswered::Ended),
            Err(error) => return Err(Unanswered::Unsent(error)),
        };
        let deadline = Instant::now().checked_add(budget);
        let mut pending = self.shared.begin(request, deadline, cut + left_out)?;
        if budget.is_zero() {
            return Err(Unanswered::Late);
        }
        let answer = match deadline {
            Some(at) => timeout_at(at, &mut pending.answer)
                .await
                .map_err(|_| Unanswered::Late)?,
            None => (&mut pending.answer).await,
        };
        match answer.map_err(|_| Unanswered::Ended)? {
            Answer::Decision(decision) => Ok(decision),
            Answer::Busy(busy) => Err(Unanswered::Busy(busy)),
        }
    }

    /// Queues a report for the session without waiting for room. A full
    /// queue ends the link.
    pub fn report(&self, frame: gateway_frame::Frame) -> Result<(), Full> {
        self.shared.report(frame)
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
    stopped: watch::Receiver<()>,
}

#[derive(Debug)]
struct State {
    floor: u64,
    open: Option<Open>,
    // Why the link ended: the first cause, written as the queues are taken.
    ended: Option<Arc<Error>>,
    // The block hashes cut from the asks the link queued.
    cut: usize,
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
}

#[derive(Debug)]
struct Open {
    // The link's channel, kept for a watch opened on it. It leaves with
    // the queues, so an ended link holds no connection open.
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
    fn new(
        stub: Stub,
        floor: u64,
        reports: mpsc::Sender<gateway_frame::Frame>,
        requests: mpsc::UnboundedSender<DecideRequest>,
    ) -> Arc<Self> {
        let (stop, stopped) = watch::channel(());
        Arc::new(Self {
            state: Mutex::new(State {
                floor,
                open: Some(Open {
                    stub,
                    reports,
                    requests,
                    credit: 0,
                    pending: FxHashMap::default(),
                    _stop: stop,
                }),
                ended: None,
                cut: 0,
            }),
            stopped,
        })
    }

    fn session_frames(
        self: &Arc<Self>,
        first: GatewayFrame,
        queued: mpsc::Receiver<gateway_frame::Frame>,
    ) -> impl Stream<Item = GatewayFrame> + use<> {
        let shared = Arc::downgrade(self);
        let mut seq = first.seq;
        once(first).chain(ReceiverStream::new(queued).map_while(move |frame| {
            let shared = shared.upgrade()?;
            let state = shared.state.lock();
            state.open.as_ref()?;
            seq += 1;
            Some(numbered(seq, state.floor, frame))
        }))
    }

    fn decide_requests(
        self: &Arc<Self>,
        queued: mpsc::UnboundedReceiver<DecideRequest>,
    ) -> impl Stream<Item = DecideRequest> + use<> {
        let shared = Arc::downgrade(self);
        UnboundedReceiverStream::new(queued)
            .map_while(move |request| shared.upgrade()?.stands().then_some(request))
    }

    fn start(
        self: Arc<Self>,
        welcome: Welcome,
        term: u64,
        controller: impl Stream<Item = Result<ControllerFrame, Status>> + Send + 'static,
        decisions: impl Stream<Item = Result<DecideAnswer, Status>> + Send + 'static,
    ) -> (Link, [JoinHandle<()>; 2]) {
        {
            let mut state = self.state.lock();
            if let Some(open) = state.observe(term) {
                open.credit = welcome.decide_window;
            }
        }
        let readers = [
            self.reader(controller, Self::read_session, SESSION_ENDED),
            self.reader(decisions, Self::read_decide, DECIDE_ENDED),
        ];
        let link = Link {
            welcome: Arc::new(welcome),
            shared: self,
        };
        (link, readers)
    }

    fn stands(&self) -> bool {
        self.state.lock().open.is_some()
    }

    /// Takes the queues and waiters together, then drops them outside the
    /// lock. Neither stream can send new frames past this point. The end
    /// that takes them writes why: a later end finds none, and its cause
    /// is dropped.
    fn end(mut state: MutexGuard<'_, State>, why: Error) {
        let open = state.open.take();
        if open.is_some() {
            state.ended = Some(Arc::new(why));
        }
        drop(state);
        drop(open);
    }

    fn report(&self, frame: gateway_frame::Frame) -> Result<(), Full> {
        let state = self.state.lock();
        let open = state.open.as_ref().ok_or(Full)?;
        match open.reports.try_send(frame) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                Self::end(state, Error::ReportsFull);
                Err(Full)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(Full),
        }
    }

    fn begin(
        &self,
        request: DecideRequest,
        deadline: Option<Instant>,
        cut: usize,
    ) -> Result<Pending<'_>, Unanswered> {
        let mut state = self.state.lock();
        let open = state.open.as_mut().ok_or(Unanswered::Ended)?;
        if open.pending.contains_key(request.request_id.as_slice()) {
            return Err(Unanswered::InFlight);
        }
        if open.credit == 0 {
            return Err(Unanswered::NoCredit);
        }
        let id = Arc::<[u8]>::from(request.request_id.as_slice());
        if open.requests.send(request).is_err() {
            return Err(Unanswered::Ended);
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
        frames: impl Stream<Item = Result<T, Status>> + Send + 'static,
        read: fn(&Self, T),
        clean_end: &'static str,
    ) -> JoinHandle<()>
    where
        T: Send + 'static,
    {
        // The readers hold the link only while handling a frame. Dropping
        // the last handle drops the queues and wakes both readers.
        let shared = Arc::downgrade(self);
        let mut stopped = self.stopped.clone();
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
                        read(&shared, frame);
                        continue;
                    }
                    Some(Err(status)) => status,
                    None => Status::ok(clean_end),
                };
                Self::end(shared.state.lock(), status.into());
                break;
            }
        })
    }

    fn read_session(&self, frame: ControllerFrame) {
        let mut state = self.state.lock();
        let (term, floor) = (frame.term, state.floor);
        if state.observe(term).is_none() {
            Self::end(state, Error::BelowFloor { term, floor });
        } else if let Some(controller_frame::Frame::NotLeader(told)) = frame.frame {
            Self::end(state, Error::NotLeader(told.leader_address));
        }
    }

    fn read_decide(&self, frame: DecideAnswer) {
        let mut state = self.state.lock();
        let (term, floor) = (frame.term, state.floor);
        let Some(open) = state.observe(term) else {
            Self::end(state, Error::BelowFloor { term, floor });
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
                Self::end(state, Error::NotLeader(String::new()));
                return;
            }
            None => {
                let nothing = "an answer of the decide stream carries nothing";
                Self::end(state, Error::NoDecision(nothing.to_owned()));
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
            if self.report(frame).is_err() {
                // The controller's session grace covers leases still on file.
                break;
            }
        }
    }
}

struct Pending<'a> {
    shared: &'a Shared,
    id: Arc<[u8]>,
    answer: oneshot::Receiver<Answer>,
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        self.answer.close();
        if let Some(open) = self.shared.state.lock().open.as_mut() {
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
        client::{PolicyControllerClient, Principal},
        proto::{
            busy, controller_frame, decide_answer, released, Credit, Header, Lease, NotLeader,
            Released, RequestFacts, Side,
        },
        spec::MAX_HEADERS,
    };

    const WAIT: Duration = Duration::from_secs(5);
    const LEADER_TERM: u64 = 3;
    const WELCOME_TERM: u64 = 4;
    const ASKS: u8 = 64;

    /// One change of a request's facts.
    type Edit = fn(&mut RequestFacts);

    pub(crate) struct Peer {
        pub(crate) link: Link,
        pub(crate) reports: Pin<Box<dyn Stream<Item = GatewayFrame>>>,
        pub(crate) requests: Pin<Box<dyn Stream<Item = DecideRequest>>>,
        controller: Option<mpsc::UnboundedSender<Result<ControllerFrame, Status>>>,
        decide: Option<mpsc::UnboundedSender<Result<DecideAnswer, Status>>>,
        readers: [JoinHandle<()>; 2],
    }

    impl Peer {
        pub(crate) fn new(window: u32) -> Self {
            Self::over(window, Endpoint::from_static("http://127.0.0.1:1"))
        }

        fn over(window: u32, endpoint: Endpoint) -> Self {
            const GATEWAY: &str = "gateway";
            let principal = Principal::of(GATEWAY).expect("a gateway id");
            let stub = PolicyControllerClient::with_interceptor(endpoint.connect_lazy(), principal);
            let (report, reports) = mpsc::channel(REPORT_QUEUE);
            let (request, requests) = mpsc::unbounded_channel();
            let (controller, session) = mpsc::unbounded_channel();
            let (decide, decisions) = mpsc::unbounded_channel();
            let shared = Shared::new(stub, LEADER_TERM, report, request);
            let reports = Box::pin(shared.session_frames(hello(GATEWAY, LEADER_TERM), reports));
            let requests = Box::pin(shared.decide_requests(requests));
            let welcome = Welcome {
                decide_window: window,
                ..Welcome::default()
            };
            let (link, readers) = shared.start(
                welcome,
                WELCOME_TERM,
                UnboundedReceiverStream::new(session),
                UnboundedReceiverStream::new(decisions),
            );
            Self {
                link,
                reports,
                requests,
                controller: Some(controller),
                decide: Some(decide),
                readers,
            }
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
            assert_eq!((first.seq, first.term_floor), (1, LEADER_TERM));
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
                link.report(release(id, Side::Single, released::Reason::ClientGone))
            })
            .join()
            .expect("the caller returns")
            .expect("the link takes the report");
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
    async fn a_full_report_queue_ends_the_link() {
        let mut peer = Peer::new(1);
        peer.read_hello().await;
        let mut asked = Box::pin(peer.link.ask(request(1), WAIT));
        assert!(poll(asked.as_mut()).is_pending());
        let report = release(7, Side::Single, released::Reason::ClientGone);
        for _ in 0..65_536 {
            assert!(peer.link.report(report.clone()).is_ok());
        }
        assert!(peer.link.stands());
        closes(&peer.link, || {
            assert!(matches!(peer.link.report(report.clone()), Err(Full)));
            assert!(!peer.link.stands());
            assert!(matches!(
                peer.link.ended().as_deref(),
                Some(Error::ReportsFull)
            ));
        })
        .await;
        assert!(matches!(peer.link.report(report), Err(Full)));
        assert!(matches!(within(asked).await, Err(Unanswered::Ended)));
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
            Err(Unanswered::Unsent(Error::RequestId(0)))
        ));
        let mut too_large = request(4);
        too_large.facts = Some(RequestFacts {
            model: "m".repeat(64 * 1024),
            ..RequestFacts::default()
        });
        assert!(matches!(
            peer.link.ask(too_large, WAIT).await,
            Err(Unanswered::Unsent(Error::TooLarge(_)))
        ));
        assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());

        let mut first = Box::pin(peer.link.ask(request(1), WAIT));
        assert!(poll(first.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
        assert!(matches!(
            poll(Box::pin(peer.link.ask(request(2), WAIT)).as_mut()),
            Poll::Ready(Err(Unanswered::NoCredit))
        ));
        assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());

        let busy = Busy {
            reason: busy::Reason::QueueDepth.into(),
            retry_after_micros: 125,
        };
        peer.answer(1, 4, decide_answer::Answer::Busy(busy));
        assert!(matches!(within(first).await, Err(Unanswered::Busy(told)) if told == busy));
        assert!(matches!(
            peer.link.ask(request(2), WAIT).await,
            Err(Unanswered::NoCredit)
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
            Err(Unanswered::NoCredit)
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
                matches!(refused, Err(Unanswered::Unsent(Error::OutOfRange(named))) if named == field),
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
                Err(Unanswered::Unsent(Error::OutOfRange("decode_of")))
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
                Poll::Ready(Err(Unanswered::InFlight))
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
        assert!(matches!(within(first).await, Err(Unanswered::Late)));

        // The first ask's answer is still due. A second ask under its id
        // would take that answer for its own: it is refused, unsent.
        assert!(matches!(
            poll(Box::pin(peer.link.ask(request(1), budget)).as_mut()),
            Poll::Ready(Err(Unanswered::InFlight))
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
            Poll::Ready(Err(Unanswered::Late))
        ));
        assert_eq!(within(peer.requests.next()).await, Some(request(2)));
    }

    #[tokio::test]
    async fn answers_without_an_ask_release_each_lease_and_drop_busy() {
        let mut peer = Peer::new(4);
        peer.read_hello().await;

        assert!(matches!(
            peer.link.ask(request(1), Duration::ZERO).await,
            Err(Unanswered::Late)
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
            Poll::Ready(Err(Unanswered::InFlight))
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
            assert!(matches!(within(asked).await, Err(Unanswered::Ended)));
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
                let report = release(7, Side::Single, released::Reason::ClientGone);
                assert!(matches!(peer.link.report(report), Err(Full)));
            } else {
                drop(peer.requests);
                for _ in 0..2 {
                    assert!(matches!(
                        peer.link.ask(request(2), WAIT).await,
                        Err(Unanswered::Ended)
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
            assert!(matches!(within(asked).await, Err(Unanswered::Ended)));
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
        assert!(matches!(within(first).await, Err(Unanswered::Ended)));
        assert!(matches!(within(second).await, Err(Unanswered::Ended)));
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
            Poll::Ready(Err(Unanswered::Ended))
        ));
        assert!(matches!(
            peer.link.ask(DecideRequest::default(), WAIT).await,
            Err(Unanswered::Ended)
        ));
        assert!(matches!(
            peer.link
                .report(release(7, Side::Single, released::Reason::ClientGone)),
            Err(Full)
        ));
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
        assert!(link
            .report(release(7, Side::Single, released::Reason::ClientGone))
            .is_ok());
        assert!(matches!(
            link.ask(request(1), Duration::ZERO).await,
            Err(Unanswered::Late)
        ));
        drop(link);

        assert_eq!(within(peer.reports.next()).await, None);
        assert_eq!(within(peer.requests.next()).await, None);
        Peer::readers_end(peer.readers).await;
    }
}
