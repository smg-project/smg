//! A host's link to its controller. The session takes lifecycle reports and
//! one decide stream carries the host's concurrent asks.

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
    dial_leader, fit, hello, numbered, Answer, Decide, Error, Session, ANSWER_WAIT, KEEPALIVE,
};
use crate::proto::{
    busy, controller_frame, decide_answer, gateway_frame, released, Busy, ControllerFrame,
    DecideAnswer, DecideRequest, Decision, GatewayFrame, Released, Welcome,
};

/// About six seconds at ten thousand reports a second.
const REPORT_QUEUE: usize = 65_536;

/// A host's session and one decide stream, shared by its requests.
#[derive(Clone, Debug)]
pub struct Link {
    welcome: Arc<Welcome>,
    shared: Arc<Shared>,
}

/// Why an ask has no decision from the controller. The host decides the
/// request itself.
#[derive(Debug)]
pub enum Unanswered {
    /// The stream holds no credit. Nothing was sent.
    NoCredit,
    /// The controller's reason for leaving the request to the host and the
    /// wait it names.
    Busy(Busy),
    /// No answer inside the budget. The link releases a decision that arrives
    /// later.
    Late,
    /// An ask under this id is in flight already. Nothing was sent.
    InFlight,
    /// The link has ended.
    Ended,
    /// The request's id or size prevents sending it.
    Unsent(Error),
}

/// The report queue was full or the link had ended.
#[derive(Debug)]
pub struct Full;

impl Link {
    /// Dials the endpoint, asks who leads, opens the session and one decide
    /// stream, and starts their readers on the calling runtime.
    pub async fn connect(endpoint: Endpoint, gateway_id: &str) -> Result<Self, Error> {
        // The link's readers wait with no bound, as a watch does. Its channel
        // is kept alive, so a controller gone silent ends the link.
        let endpoint = endpoint
            .http2_keep_alive_interval(KEEPALIVE)
            .keep_alive_timeout(ANSWER_WAIT)
            .keep_alive_while_idle(true);
        let (mut stub, floor) = dial_leader(endpoint.connect(), gateway_id, ANSWER_WAIT).await?;
        let (report, reports) = mpsc::channel(REPORT_QUEUE);
        let (request, requests) = mpsc::unbounded_channel();
        let shared = Shared::new(floor, report, request);
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

    /// Whether both streams still stand.
    pub fn stands(&self) -> bool {
        self.shared.stands()
    }

    /// The highest term an answer carried.
    pub fn term_floor(&self) -> u64 {
        self.shared.state.lock().floor
    }

    /// Sends one ask if the stream holds credit and waits up to `budget`
    /// for its answer. Use a distinct id for each request that may still
    /// have an answer on the wire.
    pub async fn ask(
        &self,
        mut request: DecideRequest,
        budget: Duration,
    ) -> Result<Decision, Unanswered> {
        if let Err(error) = fit(&mut request) {
            return Err(if self.stands() {
                Unanswered::Unsent(error)
            } else {
                Unanswered::Ended
            });
        }
        let deadline = Instant::now().checked_add(budget);
        let mut pending = self.shared.begin(request, deadline)?;
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
    // The id stays in the table until its caller takes the answer or leaves.
    answer: Option<oneshot::Sender<Answer>>,
}

impl Waiting {
    fn take(&mut self) -> Option<oneshot::Sender<Answer>> {
        if self.deadline.is_some_and(|at| Instant::now() >= at) {
            return None;
        }
        self.answer.take()
    }
}

impl Shared {
    fn new(
        floor: u64,
        reports: mpsc::Sender<gateway_frame::Frame>,
        requests: mpsc::UnboundedSender<DecideRequest>,
    ) -> Arc<Self> {
        let (stop, stopped) = watch::channel(());
        Arc::new(Self {
            state: Mutex::new(State {
                floor,
                open: Some(Open {
                    reports,
                    requests,
                    credit: 0,
                    pending: FxHashMap::default(),
                    _stop: stop,
                }),
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
            self.reader(controller, Self::read_session),
            self.reader(decisions, Self::read_decide),
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
    /// lock. Neither stream can send new frames past this point.
    fn end(mut state: MutexGuard<'_, State>) {
        let open = state.open.take();
        drop(state);
        drop(open);
    }

    fn report(&self, frame: gateway_frame::Frame) -> Result<(), Full> {
        let state = self.state.lock();
        let open = state.open.as_ref().ok_or(Full)?;
        if open.reports.try_send(frame).is_err() {
            Self::end(state);
            return Err(Full);
        }
        Ok(())
    }

    fn begin(
        &self,
        request: DecideRequest,
        deadline: Option<Instant>,
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
        open.credit -= 1;
        if open.requests.send(request).is_err() {
            Self::end(state);
            return Err(Unanswered::Ended);
        }
        // A reader takes this same lock before it can deliver an answer.
        let (sender, answer) = oneshot::channel();
        open.pending.insert(
            Arc::clone(&id),
            Waiting {
                deadline,
                answer: Some(sender),
            },
        );
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
                match next {
                    Some(Ok(frame)) => read(&shared, frame),
                    Some(Err(_)) | None => {
                        Self::end(shared.state.lock());
                        break;
                    }
                }
            }
        })
    }

    fn read_session(&self, frame: ControllerFrame) {
        let mut state = self.state.lock();
        if state.observe(frame.term).is_none()
            || matches!(frame.frame, Some(controller_frame::Frame::NotLeader(_)))
        {
            Self::end(state);
        }
    }

    fn read_decide(&self, frame: DecideAnswer) {
        let mut state = self.state.lock();
        let Some(open) = state.observe(frame.term) else {
            Self::end(state);
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
            Some(decide_answer::Answer::Busy(_)) | None => {
                Self::end(state);
                return;
            }
        };
        let waiting = open.pending.get_mut(frame.request_id.as_slice());
        let undelivered = match waiting.and_then(Waiting::take) {
            Some(sender) => sender.send(answer).err(),
            None => Some(answer),
        };
        drop(state);
        if let Some(Answer::Decision(decision)) = undelivered {
            self.release(decision);
        }
    }

    fn release(&self, decision: Decision) {
        for lease in decision.leases {
            let frame = gateway_frame::Frame::Released(Released {
                what: Some(released::What::LeaseId(lease.id)),
                reason: released::Reason::LateDecision.into(),
                side: lease.side,
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
            open.pending.remove(self.id.as_ref());
        }
        // A send that won the race with close may still be in the channel.
        if let Ok(Answer::Decision(decision)) = self.answer.try_recv() {
            self.shared.release(decision);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Waker},
    };

    use tokio::time::{advance, timeout};

    use super::*;
    use crate::proto::{
        busy, controller_frame, decide_answer, released, Credit, Lease, NotLeader, Released,
        RequestFacts, Side,
    };

    const WAIT: Duration = Duration::from_secs(5);
    const LEADER_TERM: u64 = 3;
    const WELCOME_TERM: u64 = 4;
    const ASKS: u8 = 64;

    struct Peer {
        link: Link,
        reports: Pin<Box<dyn Stream<Item = GatewayFrame>>>,
        requests: Pin<Box<dyn Stream<Item = DecideRequest>>>,
        controller: Option<mpsc::UnboundedSender<Result<ControllerFrame, Status>>>,
        decide: Option<mpsc::UnboundedSender<Result<DecideAnswer, Status>>>,
        readers: [JoinHandle<()>; 2],
    }

    impl Peer {
        fn new(window: u32) -> Self {
            let (report, reports) = mpsc::channel(REPORT_QUEUE);
            let (request, requests) = mpsc::unbounded_channel();
            let (controller, session) = mpsc::unbounded_channel();
            let (decide, decisions) = mpsc::unbounded_channel();
            let shared = Shared::new(LEADER_TERM, report, request);
            let reports = Box::pin(shared.session_frames(hello("gateway", LEADER_TERM), reports));
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

        fn controller(&self, term: u64, frame: Option<controller_frame::Frame>) {
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

        fn answer(&self, id: u8, term: u64, answer: decide_answer::Answer) {
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

        async fn read_hello(&mut self) {
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

    fn poll<T>(future: Pin<&mut impl Future<Output = T>>) -> Poll<T> {
        future.poll(&mut Context::from_waker(Waker::noop()))
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
        assert!(matches!(peer.link.report(report.clone()), Err(Full)));
        assert!(!peer.link.stands());
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
        assert_eq!(within(peer.requests.next()).await, Some(request(3)));
        assert!(matches!(
            peer.link.ask(request(4), WAIT).await,
            Err(Unanswered::NoCredit)
        ));
        assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());
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
    async fn the_budget_ends_the_wait_and_frees_the_id() {
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

        let mut again = Box::pin(peer.link.ask(request(1), budget));
        assert!(poll(again.as_mut()).is_pending());
        assert_eq!(within(peer.requests.next()).await, Some(request(1)));
        advance(budget).await;
        peer.answer(1, 5, decision(7, &[Side::Single]));
        assert_eq!(
            within(peer.reports.next()).await.expect("a release").frame,
            Some(release(7, Side::Single, released::Reason::LateDecision))
        );
        assert!(matches!(within(again).await, Err(Unanswered::Late)));

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
        peer.answer(1, 4, decision(7, &[Side::Prefill, Side::Decode]));
        for side in [Side::Prefill, Side::Decode] {
            assert_eq!(
                within(peer.reports.next()).await.expect("a release").frame,
                Some(release(7, side, released::Reason::LateDecision))
            );
        }

        let mut dropped = Box::pin(peer.link.ask(request(2), WAIT));
        assert!(poll(dropped.as_mut()).is_pending());
        drop(dropped);
        assert_eq!(within(peer.requests.next()).await, Some(request(2)));
        peer.answer(2, 4, decision(8, &[Side::Single]));

        let mut closed = peer
            .link
            .shared
            .begin(request(3), None)
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
            if stale_session {
                peer.controller(5, None);
            } else {
                peer.answer(1, 5, decision(7, &[Side::Single]));
            }
            assert!(matches!(within(asked).await, Err(Unanswered::Ended)));
            assert!(!peer.link.stands());
            assert_eq!(peer.link.term_floor(), 6);
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

        let peer = Peer::new(1);
        drop(peer.requests);
        assert!(matches!(
            peer.link.ask(request(1), WAIT).await,
            Err(Unanswered::Ended)
        ));
        assert!(!peer.link.stands());
        Peer::readers_end(peer.readers).await;

        let peer = Peer::new(1);
        drop(peer.reports);
        assert!(matches!(
            peer.link
                .report(release(7, Side::Single, released::Reason::ClientGone)),
            Err(Full)
        ));
        assert!(!peer.link.stands());
        Peer::readers_end(peer.readers).await;
    }

    async fn check_end(session: bool, closed: Option<bool>) {
        let mut peer = Peer::new(2);
        let mut first = Box::pin(peer.link.ask(request(1), WAIT));
        let mut second = Box::pin(peer.link.ask(request(2), WAIT));
        assert!(poll(first.as_mut()).is_pending());
        assert!(poll(second.as_mut()).is_pending());
        match (session, closed) {
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
                Some(controller_frame::Frame::NotLeader(NotLeader::default())),
            ),
            (false, None) => peer.answer(
                1,
                5,
                decide_answer::Answer::Busy(Busy {
                    reason: busy::Reason::NotLeader.into(),
                    ..Busy::default()
                }),
            ),
        }
        assert!(matches!(within(first).await, Err(Unanswered::Ended)));
        assert!(matches!(within(second).await, Err(Unanswered::Ended)));
        Peer::readers_end(peer.readers).await;
        assert!(!peer.link.stands());
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
