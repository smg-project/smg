//! The gateway's side of a row: the streams' clients, closed loop and
//! open, and the session that tells each answered request to its end. A
//! child of the bench's root: it reads the root's private `Shape` and
//! `model`.

use std::{
    mem,
    net::SocketAddr,
    sync::{atomic::Ordering, mpsc as std_mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use smg_grand_central::{
    service::proto::{
        completed::Class, controller_frame, decide_answer::Answer, decision::Outcome,
        gateway_frame::Frame, grand_central_client::GrandCentralClient, Completed, ControllerFrame,
        DecideRequest, Decision, Dispatched, GatewayFrame, Hello, Phase as WirePhase,
        RequestFacts as WireFacts, Side,
    },
    store::{GatewayId, Hand, MemoryStore},
    tracking::{LeaseId, ReleaseReason, Said},
};
use tokio::{runtime, sync::mpsc};
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream, metadata::AsciiMetadataValue,
    transport::Channel, Request, Streaming,
};

use super::{
    allocator::{counting, CALLS, COUNTING},
    kernel, model,
    threads::{Spent, SESSION},
    Shape,
};

const GATEWAY: &str = "bench-gateway";

/// The slots a stream keeps its requests' send instants in, each beside
/// its request's id: above every row's count in flight. An answer whose
/// slot a later request took is counted, not timed.
const RING: u64 = 1_024;
/// The budget every request brings.
const BUDGET_MICROS: u32 = 5_000_000;
/// A stream's window of credit, as the `Welcome` states it.
const WINDOW: u64 = 128;
/// An offered request handed over this long after it was due is counted
/// late: the client's own lag.
const LATE: Duration = Duration::from_micros(100);

/// The reports a stream hands the session together: a hand-over wakes the
/// session's thread, and one a request would time the wake.
pub(super) const REPORTS: usize = 16;

/// A call that states the bench's gateway on the listener that takes
/// identities as told.
fn stated<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .metadata_mut()
        .insert("x-smg-principal", AsciiMetadataValue::from_static(GATEWAY));
    request
}

async fn connect(addr: SocketAddr) -> GrandCentralClient<Channel> {
    GrandCentralClient::connect(format!("http://{addr}"))
        .await
        .expect("the listener accepts")
}

/// The gateway's session: its `Hello` is welcomed, and the stream stands
/// for as long as the returned ends are held.
async fn hello(addr: SocketAddr) -> (mpsc::Sender<GatewayFrame>, Streaming<ControllerFrame>) {
    let mut client = connect(addr).await;
    // Room for the frames of one hand-over of reports: they leave together.
    let (frames, sent) = mpsc::channel(2 * REPORTS);
    let mut answers = client
        .session(stated(ReceiverStream::new(sent)))
        .await
        .expect("the stream opens")
        .into_inner();
    let hello = Hello {
        gateway_id: GATEWAY.to_owned(),
        protocol_min: 1,
        protocol_max: 1,
        ..Hello::default()
    };
    frames
        .send(GatewayFrame {
            seq: 0,
            term_floor: 0,
            frame: Some(Frame::Hello(hello)),
        })
        .await
        .expect("the stream takes the Hello");
    let answer = answers
        .message()
        .await
        .expect("a frame, not a status")
        .expect("a frame, not the stream's end");
    let window = match &answer.frame {
        Some(controller_frame::Frame::Welcome(welcome)) => u64::from(welcome.decide_window),
        _ => 0,
    };
    assert_eq!(
        window, WINDOW,
        "a Welcome answers a Hello, with the window the open loop counts by: {answer:?}"
    );
    (frames, answers)
}

/// What one stream measured.
pub(super) struct Tally {
    /// Each measured decision's round trip, in nanoseconds; none of a
    /// counted run's.
    pub(super) nanos: Vec<u64>,
    /// The measured answers, whether their round trips were kept.
    pub(super) answered: u64,
    pub(super) busy: u64,
    /// The first measured answer's receipt, and the last one's.
    pub(super) opened: Instant,
    pub(super) closed: Instant,
    /// The allocator's calls over the measured decisions, where it counted.
    pub(super) calls: u64,
    /// The measured requests an open loop handed over late.
    pub(super) late: u64,
    /// The measured requests an open loop decided itself: due with no
    /// credit.
    pub(super) local: u64,
    /// The measured answers a closed loop could not time: a later request
    /// took their slot of the ring.
    pub(super) overtaken: u64,
    /// The measured decisions by how their three best rows scored: each
    /// alone, the best alone and the next two alike, the best two alike,
    /// all three alike.
    pub(super) ties: [u64; 4],
    /// The measured picks whose trace held fewer than three rows.
    pub(super) short: u64,
    /// The measured picks whose answer carried a lease.
    pub(super) leases: u64,
}

impl Tally {
    /// Counts a measured pick: under its place of [`Self::ties`], and its
    /// lease where its answer carried one.
    fn count(&mut self, pick: Option<&Answered>) {
        let Some(pick) = pick else {
            return;
        };
        if let Some(count) = self.ties.get_mut(pick.place) {
            *count += 1;
        }
        self.short += u64::from(pick.short);
        self.leases += u64::from(pick.lease.is_some());
    }
}

/// The bench's gateway as the store keys its leases: by the fingerprint of
/// the subject its calls state.
fn gateway() -> GatewayId {
    GatewayId(smg_grand_central::engine::fingerprint(GATEWAY.as_bytes()))
}

/// What a stream reads of a pick's answer.
struct Answered {
    /// Its place in the stream's tally of ties ([`Tally::ties`]).
    place: usize,
    /// Its trace held fewer than the three rows a place reads.
    short: bool,
    /// The lease the answer carried.
    lease: Option<u64>,
    /// The pick's first target: the member a gateway dispatches to.
    served: u64,
}

/// A request that is over, as its stream tells the gateway's session: the
/// lease its answer carried, zero where it carried none (a lease the store
/// does not hold is no error to a session), and the member that served it.
#[derive(Clone, Copy)]
pub(super) struct Over {
    lease: u64,
    served: u64,
}

/// How a stream reports an answered request as over.
pub(super) enum Ends {
    /// Through the stream's own hand at the leases' file, in process: the
    /// hand is made at the stream's first report.
    Door(Arc<MemoryStore>, Option<Hand>),
    /// As a gateway does: a `Dispatched` and a `Completed` on its session,
    /// whose thread takes the stream's reports [`REPORTS`] at a time.
    Session {
        reports: mpsc::Sender<Vec<Over>>,
        held: Vec<Over>,
    },
}

impl Clone for Ends {
    /// A stream's own end: a door's hand is not shared.
    fn clone(&self) -> Self {
        match self {
            Self::Door(store, _) => Self::Door(Arc::clone(store), None),
            Self::Session { reports, held } => Self::Session {
                reports: reports.clone(),
                held: held.clone(),
            },
        }
    }
}

impl Ends {
    /// Reports a picked request as over: its lease, where its answer
    /// carried one, ends at the store's door, or the gateway's session is
    /// told. A row holds as many leases as it has decisions in flight and
    /// frames on their way.
    async fn over(&mut self, gateway: GatewayId, pick: &Answered) {
        match self {
            Self::Door(store, hand) => {
                if let Some(lease) = pick.lease {
                    let hand = hand.get_or_insert_with(|| Hand::new(gateway));
                    let ended = Said::Released(None, ReleaseReason::ClientGone);
                    store.hand(hand, LeaseId(lease), ended);
                }
            }
            Self::Session { reports, held } => {
                held.push(Over {
                    lease: pick.lease.unwrap_or(0),
                    served: pick.served,
                });
                if held.len() >= REPORTS {
                    let together = mem::replace(held, Vec::with_capacity(REPORTS));
                    reports
                        .send(together)
                        .await
                        .expect("the session takes a stream's reports");
                }
            }
        }
    }

    /// Hands the session the reports the stream still holds, as it ends.
    async fn rest(&mut self) {
        if let Self::Session { reports, held } = self {
            if !held.is_empty() {
                reports
                    .send(mem::take(held))
                    .await
                    .expect("the session takes a stream's last reports");
            }
        }
    }
}

/// Every decision of a row is a pick over its fleet. Another outcome means
/// the setup drifted, and the row would print another path's numbers.
/// Reads the decision's place in a stream's tally of ties, by its trace's
/// three best rows (two score alike where their scores are one value, bit
/// for bit), the lease its answer carried and its first target.
fn answered(decision: &Decision) -> Answered {
    let first = match &decision.outcome {
        Some(Outcome::Pick(pick)) => pick.targets.first(),
        _ => None,
    };
    assert!(first.is_some(), "a pick over the fleet: {decision:?}");
    let top = decision.trace.as_ref().map_or(&[][..], |trace| &trace.top);
    let place = match top {
        [first, second, third, ..] => {
            let alike = |one: f32, other: f32| usize::from(one.to_bits() == other.to_bits());
            2 * alike(first.score, second.score) + alike(second.score, third.score)
        }
        _ => 0,
    };
    Answered {
        place,
        short: top.len() < 3,
        lease: decision.leases.first().map(|lease| lease.id),
        served: first.map_or(0, |target| target.id),
    }
}

fn slot(id: u64) -> usize {
    usize::try_from(id % RING).expect("a slot of the ring")
}

/// Request `id` of a row, for one of its models.
fn request(id: u64, models: &[String], pair: bool) -> DecideRequest {
    let name = models
        .get(slot(id) % models.len().max(1))
        .expect("a model of the row");
    let phase = if pair {
        WirePhase::Prefill
    } else {
        WirePhase::default()
    };
    DecideRequest {
        request_id: id.to_be_bytes().to_vec(),
        facts: Some(WireFacts {
            model: name.clone(),
            seed: id,
            budget_micros: BUDGET_MICROS,
            phase: phase.into(),
            ..WireFacts::default()
        }),
        constraints: None,
        again: false,
        ..Default::default()
    }
}

/// Hands the stream its next request and keeps the instant.
async fn hand(
    requests: &mpsc::Sender<DecideRequest>,
    sent_at: &mut [(u64, Instant)],
    next: &mut u64,
    models: &[String],
    pair: bool,
) {
    let id = *next;
    *next += 1;
    let frame = request(id, models, pair);
    if let Some(at) = sent_at.get_mut(slot(id)) {
        *at = (id, Instant::now());
    }
    requests
        .send(frame)
        .await
        .expect("the stream takes the request");
}

/// One stream of the shape: `in_flight` requests kept in flight until the
/// warm and the measured decisions are answered.
pub(super) async fn ask(addr: SocketAddr, shape: Shape, ends: &mut Ends) -> Tally {
    let gateway = gateway();
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let mut client = connect(addr).await;
    let (requests, sent) = mpsc::channel(256);
    let mut answers = client
        .decide(stated(ReceiverStream::new(sent)))
        .await
        .expect("a present gateway's stream opens")
        .into_inner();
    let mut sent_at = vec![(u64::MAX, Instant::now()); slot(RING - 1) + 1];
    let mut next = 0;
    let mut tally = Tally {
        nanos: Vec::with_capacity(usize::try_from(shape.measured).expect("a count that fits")),
        answered: 0,
        busy: 0,
        opened: Instant::now(),
        closed: Instant::now(),
        calls: 0,
        late: 0,
        local: 0,
        overtaken: 0,
        ties: [0; 4],
        short: 0,
        leases: 0,
    };
    if shape.counted && shape.warm == 0 {
        counting();
    }
    for _ in 0..shape.in_flight {
        hand(&requests, &mut sent_at, &mut next, &models, shape.pair).await;
    }
    let mut done = 0;
    loop {
        let answer = answers
            .message()
            .await
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        let received = Instant::now();
        let pick = match &answer.answer {
            Some(Answer::Decision(decision)) => Some(answered(decision)),
            Some(Answer::Busy(_)) => {
                tally.busy += 1;
                None
            }
            Some(Answer::Credit(_)) | None => continue,
        };
        if let Some(pick) = &pick {
            ends.over(gateway, pick).await;
        }
        if done == shape.warm {
            tally.opened = received;
        }
        if done >= shape.warm {
            tally.answered += 1;
            tally.count(pick.as_ref());
        }
        // A counted run keeps no round trip: a sort of them would stay in
        // the difference of two runs.
        if done >= shape.warm && !shape.counted {
            let id: [u8; 8] = answer
                .request_id
                .as_slice()
                .try_into()
                .expect("the request's eight bytes");
            let id = u64::from_be_bytes(id);
            match sent_at.get(slot(id)) {
                Some((held, at)) if *held == id => {
                    let took = received.saturating_duration_since(*at);
                    tally.nanos.push(
                        u64::try_from(took.as_nanos()).expect("a round trip under centuries"),
                    );
                }
                // A later request took the slot while this one waited:
                // its instant is not this request's.
                _ => tally.overtaken += 1,
            }
        }
        done += 1;
        // No request follows the last measured answer, and a counted run
        // opens its window before the first measured request is handed
        // over: the window holds whole exchanges.
        if done == shape.warm + shape.measured {
            break;
        }
        if shape.counted && done == shape.warm {
            counting();
        }
        hand(&requests, &mut sent_at, &mut next, &models, shape.pair).await;
    }
    if shape.counted {
        COUNTING.store(false, Ordering::Relaxed);
        tally.calls = CALLS.load(Ordering::Relaxed);
    }
    tally.closed = Instant::now();
    ends.rest().await;
    tally
}

/// One stream of the shape under an open loop, as the design's gateway
/// runs one: `shape.offered` requests a second, whatever the answers do.
/// Request `i` is due at the start plus `i` over the rate. Due while the
/// stream has credit, it is handed over, and its round trip is read from
/// the due instant; due with no credit, it is the gateway's own decision
/// at once, counted and never sent. The client spins between instants,
/// since a timer's millisecond is coarser than the gaps, and yields to its
/// runtime each turn so the connection is read and written.
pub(super) async fn offer(addr: SocketAddr, shape: Shape, ends: &mut Ends) -> Tally {
    let gateway = gateway();
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let mut client = connect(addr).await;
    let (requests, sent) = mpsc::channel(256);
    let mut answers = client
        .decide(stated(ReceiverStream::new(sent)))
        .await
        .expect("a present gateway's stream opens")
        .into_inner();
    let total = shape.warm + shape.measured;
    let gap = Duration::from_nanos(1_000_000_000 / shape.offered.max(1));
    let mut tally = Tally {
        nanos: Vec::with_capacity(usize::try_from(shape.measured).expect("a count that fits")),
        answered: 0,
        busy: 0,
        opened: Instant::now(),
        closed: Instant::now(),
        calls: 0,
        late: 0,
        local: 0,
        overtaken: 0,
        ties: [0; 4],
        short: 0,
        leases: 0,
    };
    let start = Instant::now();
    let due = |id: u64| start + gap.saturating_mul(u32::try_from(id).unwrap_or(u32::MAX));
    let (mut next, mut done, mut credit) = (0, 0, WINDOW);
    while done < total {
        let now = Instant::now();
        while next < total && due(next) <= now {
            if credit == 0 {
                if next >= shape.warm {
                    tally.local += 1;
                }
                done += 1;
            } else {
                if next >= shape.warm && now.saturating_duration_since(due(next)) > LATE {
                    tally.late += 1;
                }
                requests
                    .try_send(request(next, &models, shape.pair))
                    .expect("a window of requests fits the stream's channel");
                credit -= 1;
            }
            next += 1;
        }
        if done >= total {
            break;
        }
        let answer = tokio::select! {
            biased;
            answer = answers.message() => answer,
            () = tokio::task::yield_now() => continue,
        };
        let answer = answer
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        let received = Instant::now();
        let pick = match &answer.answer {
            Some(Answer::Decision(decision)) => Some(answered(decision)),
            Some(Answer::Busy(_)) => {
                tally.busy += 1;
                None
            }
            Some(Answer::Credit(grant)) => {
                credit += u64::from(grant.window);
                continue;
            }
            None => continue,
        };
        if let Some(pick) = &pick {
            ends.over(gateway, pick).await;
        }
        let id: [u8; 8] = answer
            .request_id
            .as_slice()
            .try_into()
            .expect("the request's eight bytes");
        let id = u64::from_be_bytes(id);
        if id >= shape.warm {
            if tally.answered == 0 {
                tally.opened = received;
            }
            tally.answered += 1;
            tally.count(pick.as_ref());
            let took = received.saturating_duration_since(due(id));
            tally
                .nanos
                .push(u64::try_from(took.as_nanos()).expect("a round trip under centuries"));
        }
        done += 1;
    }
    tally.closed = Instant::now();
    ends.rest().await;
    tally
}

/// What the session's thread returns: the frames it sent and what it
/// spent.
type Told = (u64, Option<Spent>);

/// The gateway's session on a thread of its own: its `Hello` is welcomed,
/// then each request a stream reports as over is told as a gateway tells
/// one served to its end, a `Dispatched` and a `Completed`. Once every
/// reporting end is dropped the thread ends its side of the stream and
/// reads the controller's to its end: every frame was applied by then.
/// Returns the frames it sent and what its thread spent.
pub(super) fn session(
    addr: SocketAddr,
    core: Option<usize>,
) -> (mpsc::Sender<Vec<Over>>, thread::JoinHandle<Told>) {
    let (ready, opened) = std_mpsc::channel();
    let thread = thread::Builder::new()
        .name(SESSION.to_owned())
        .spawn(move || {
            if let Some(core) = core {
                assert!(kernel::pin(0, core), "the session sits on core {core}");
            }
            let sent = current_thread().block_on(async move {
                let (frames, mut answers) = hello(addr).await;
                let (reports, mut over) = mpsc::channel::<Vec<Over>>(64);
                ready.send(reports).expect("the row waits for the session");
                let mut seq = 0;
                while let Some(together) = over.recv().await {
                    for Over { lease, served } in together {
                        let dispatched = Dispatched {
                            lease_id: lease,
                            target_index: 0,
                            side: Side::Single.into(),
                            since_decision_micros: 0,
                        };
                        let completed = Completed {
                            lease_id: lease,
                            side: Side::Single.into(),
                            served,
                            status: 200,
                            class: Class::Ok.into(),
                            ..Completed::default()
                        };
                        for frame in [Frame::Dispatched(dispatched), Frame::Completed(completed)] {
                            seq += 1;
                            let taken = frames
                                .send(GatewayFrame {
                                    seq,
                                    term_floor: 0,
                                    frame: Some(frame),
                                })
                                .await;
                            // A session the controller ended takes no
                            // frame, and its status says why.
                            let ended = if taken.is_err() {
                                answers.message().await.err()
                            } else {
                                None
                            };
                            assert!(taken.is_ok(), "the controller ended the session: {ended:?}");
                        }
                    }
                }
                drop(frames);
                // The session ends as its frames do; a status in place
                // of that end says the last frames were not applied.
                while answers
                    .message()
                    .await
                    .expect("the controller ended the session")
                    .is_some()
                {}
                seq
            });
            (sent, Spent::of("thread-self"))
        })
        .expect("the session's thread");
    (opened.recv().expect("the session opens"), thread)
}

pub(super) fn current_thread() -> runtime::Runtime {
    runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a client's runtime")
}
