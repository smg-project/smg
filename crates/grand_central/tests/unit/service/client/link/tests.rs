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
        client::{GrandCentralClient, Principal},
        proto::{
            busy, controller_frame, decide_answer, released, Completed, Credit, EncodeItems,
            Header, Identity, Lease, NotLeader, Phase as FramePhase, Released, RequestFacts, Side,
            Target,
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
        self.answer_beside(id, term, answer, None);
    }

    /// [`Self::answer`] with where an encode request's items go beside
    /// the answer (`DecideAnswer.encode`).
    pub(crate) fn answer_beside(
        &self,
        id: u8,
        term: u64,
        answer: decide_answer::Answer,
        encode: Option<EncodePlaced>,
    ) {
        self.decide
            .as_ref()
            .expect("the decide stream is open")
            .send(Ok(DecideAnswer {
                request_id: if id == 0 { Vec::new() } else { vec![id] },
                term,
                answer: Some(answer),
                encode,
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
    GrandCentralClient::with_interceptor(endpoint.connect_lazy(), principal)
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
    assert!(matches!(within(first).await, Err(DecisionUnavailable::Busy(told)) if told == busy));
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

/// An ask under the id 1 that states `keys` items on a frame of
/// `phase`.
fn with_items(phase: FramePhase, keys: usize) -> DecideRequest {
    DecideRequest {
        facts: Some(RequestFacts {
            phase: phase.into(),
            ..RequestFacts::default()
        }),
        encode: Some(EncodeItems {
            keys: vec![Identity::default(); keys],
            ..EncodeItems::default()
        }),
        ..request(1)
    }
}

/// The items of an encode request have ranges the controller's check
/// reads too: more keys than the contract bounds, and items on a frame
/// of another phase, on one that states no facts, or beside a lease's
/// decode side. The link's door refuses each by its field, sends
/// nothing, spends no credit and stands. A frame at the bound is sent
/// as it is.
#[tokio::test]
async fn an_ask_out_of_the_items_ranges_sends_nothing_and_the_link_stands() {
    let mut peer = Peer::new(1);
    let wrong = [
        (
            "encode.keys",
            with_items(FramePhase::Encode, wire::MAX_ITEMS + 1),
        ),
        ("encode", with_items(FramePhase::Prefill, 1)),
        ("encode", with_items(FramePhase::Unspecified, 0)),
        (
            "encode",
            DecideRequest {
                facts: None,
                ..with_items(FramePhase::Encode, 1)
            },
        ),
        (
            "encode",
            DecideRequest {
                decode_of: Some(7),
                ..with_items(FramePhase::Encode, 1)
            },
        ),
    ];
    for (field, wrong) in wrong {
        let refused = peer.link.ask(wrong, WAIT).await;
        assert!(
            matches!(refused, Err(DecisionUnavailable::Unsent(Error::OutOfRange(named))) if named == field),
            "{field}: {refused:?}"
        );
    }
    assert!(poll(Box::pin(peer.requests.next()).as_mut()).is_pending());
    assert!(peer.link.stands() && peer.link.ended().is_none());
    // The window of one is whole: a frame at the bound is sent.
    let whole = with_items(FramePhase::Encode, wire::MAX_ITEMS);
    let mut asked = Box::pin(peer.link.ask(whole.clone(), WAIT));
    assert!(poll(asked.as_mut()).is_pending());
    assert_eq!(within(peer.requests.next()).await, Some(whole));
}

/// An encode request's answer says where its items go beside its
/// decision: the ask hands both, the encoders in their order, and an
/// ask of any other request hands nothing beside its decision. Such an
/// answer that comes behind its asker gives back its pair's leases, as
/// any late decision does.
#[tokio::test(start_paused = true)]
async fn an_ask_hands_where_an_encode_requests_items_go() {
    let mut peer = Peer::new(3);
    peer.read_hello().await;
    let encoder = |id| Target {
        id,
        ..Target::default()
    };
    let placed = EncodePlaced {
        encoders: vec![encoder(9), encoder(4)],
        one: false,
    };
    let pair = [Side::Prefill, Side::Decode];
    let mut asked = Box::pin(peer.link.ask(request(1), WAIT));
    assert!(poll(asked.as_mut()).is_pending());
    peer.answer_beside(1, WELCOME_TERM, decision(8, &pair), Some(placed.clone()));
    let (told, beside) = within(asked).await.expect("the ask's own answer");
    assert_eq!(told.leases.len(), 2);
    assert_eq!(beside, Some(placed.clone()));

    let mut plain = Box::pin(peer.link.ask(request(2), WAIT));
    assert!(poll(plain.as_mut()).is_pending());
    peer.answer(2, WELCOME_TERM, decision(9, &[Side::Single]));
    assert!(matches!(within(plain).await, Ok((_, None))));

    assert!(matches!(
        peer.link.ask(request(3), Duration::ZERO).await,
        Err(DecisionUnavailable::Late)
    ));
    peer.answer_beside(3, WELCOME_TERM, decision(7, &pair), Some(placed));
    assert_eq!(
        within(peer.reports.next()).await.expect("a release").frame,
        Some(release(
            7,
            Side::Unspecified,
            released::Reason::LateDecision
        ))
    );
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
        let (answer, _) = within(ask).await.expect("the ask gets a decision");
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
    let (first, _) = within(first).await.expect("the original ask is answered");
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
    let (again, _) = within(again).await.expect("the second ask's own answer");
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
        encode: None,
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
    let (own, _) = within(asked).await.expect("the ask's own answer");
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
