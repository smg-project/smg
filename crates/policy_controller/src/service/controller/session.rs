//! The `Session` call (design 5.2 and 5.3): one stream a gateway. `Hello`
//! comes first and `Welcome` answers it; from the `Welcome` the gateway is
//! present ([`Gateways`](super::gateways::Gateways)) and stays so for the
//! grace period after the stream's end, one period more at most while a
//! session a newer one replaced still runs.
//!
//! A stream says `Hello` within [`HELLO_DEADLINE`] of its open or is ended,
//! and while it waits it reads the election's steps: a replica that stops
//! leading ends a stream that said nothing as it ends a `Hello`.
//!
//! The fence judges the `Hello` and every frame after it. One read of the
//! fence gives the verdict and the term together, and a frame out carries
//! the term that read showed: an answer made under one term never leaves
//! under a later one.
//!
//! A session lives under one term, its `Welcome`'s. A fence that closed
//! ends it with a `NotLeader` naming the leader `Leader` names; a fence
//! open under a later term ends it too, so the gateway says `Hello` again
//! under the term that stands. An open session with no frame in flight
//! learns either at the election's next step, as readiness does. The
//! `NotLeader` rides the stream ahead of the status that ends it: on a
//! stream the frame is the status's detail.
//!
//! A frame the stream cannot read ends the session with a status, before
//! the `Welcome` and after it: only the gateway's own close ends a stream
//! with none.
//!
//! One arm a kind of frame. `Hello` and `GatewayVeto` are served here, and
//! the lifecycle's three frames in [`lifecycle`]. The other kinds are
//! refused by name until the step that serves each adds its arm, as the
//! service refuses an unserved call.
//!
//! A gateway's vetoes ride its `Hello`, whole, and are held before the
//! `Welcome` leaves: a new leader masks for the gateway from its first
//! decision. After it a `GatewayVeto` sets or lifts one kind's veto.
//!
//! The session's loop is polled with the stream it answers on, so no task
//! outlives its stream: a gateway that stops reading its stream stops its
//! own frames, and a stream that is dropped ends its session.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use tokio::sync::mpsc;
use tonic::{codegen::tokio_stream::Stream, Code, Status, Streaming};
use tracing::{debug, info, warn};

use super::{
    decide,
    floor::{unbelieved, Told, FLOORS_SPENT_MAX, FLOOR_REACH},
    gateways::{over_kinds, Refused, SessionPlace, Vetoed, KINDS_MAX, VETOES_MAX},
    lifecycle,
    principal::Principal,
    service::Shared,
    wire::millis,
    PROTOCOL_VERSION,
};
use crate::{
    engine::view::WorkerId,
    service::proto::{self, controller_frame, gateway_frame::Frame},
    store::Hand,
    tracking::lease::DISPATCH_TTL,
};

/// The stream rule's grace period (design 4.3): how long a gateway stays
/// present after its session ended with no new `Hello`. A session a newer
/// one replaced that still runs holds it one period more at most.
pub(crate) const GRACE_PERIOD: Duration = Duration::from_secs(10);

/// How many gateways are present at most, each with a standing session or
/// inside its grace period: a `Hello` that would add one past it is refused.
pub(crate) const PRESENT_MAX: usize = 4_096;

/// The longest release string of a peer that the log reads.
const RELEASE_MAX: usize = 64;

/// How long a stream may stand with no `Hello`: one that says none is
/// ended, so a silent stream holds no listener past it.
pub(crate) const HELLO_DEADLINE: Duration = Duration::from_secs(5);

/// The refusal of a `Hello` on a replica that does not lead.
const NOT_LEADING: &str = "this controller does not lead: find the leader through Leader";

/// How many vetoes a `Hello` lists at most: what a gateway holds.
const LISTED_MAX: usize = VETOES_MAX;

/// How often a gateway polls `Leader` for its term floor (design 5.3).
const LEADER_POLL: Duration = Duration::from_secs(1);

/// The frames a session sends before its loop reads again: a `Welcome` or
/// a `NotLeader`, and the status the stream ends with.
const OUT_QUEUE: usize = 4;

type Out = Result<proto::ControllerFrame, Status>;

/// The stream a `Session` call answers on: the session's frames, and the
/// session's loop polled with them.
pub(crate) struct Frames {
    /// `None` once the session ended.
    session: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
    out: mpsc::Receiver<Out>,
}

impl Stream for Frames {
    type Item = Out;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Out>> {
        let this = self.get_mut();
        if let Some(session) = this.session.as_mut() {
            if session.as_mut().poll(cx).is_ready() {
                // Its sender goes with it: the queue drains, then ends.
                this.session = None;
            }
        }
        this.out.poll_recv(cx)
    }
}

/// Serves one `Session` stream of `principal`.
pub(crate) fn serve(
    shared: Arc<Shared>,
    principal: Principal,
    inbound: Streaming<proto::GatewayFrame>,
) -> Frames {
    let (sender, out) = mpsc::channel(OUT_QUEUE);
    Frames {
        session: Some(Box::pin(async move {
            let mut answers = Answers { sender, seq: 0 };
            if let Some(status) = session(&shared, &principal, inbound, &mut answers).await {
                answers.end(status).await;
            }
        })),
        out,
    }
}

/// The session's side of its stream. A frame is numbered here and carries
/// the term its caller read from the fence.
struct Answers {
    sender: mpsc::Sender<Out>,
    seq: u64,
}

impl Answers {
    /// Sends one frame under `term`. A stream that is gone takes nothing:
    /// the loop then ends by its reader.
    async fn frame(&mut self, term: u64, frame: controller_frame::Frame) {
        self.seq += 1;
        let frame = proto::ControllerFrame {
            seq: self.seq,
            term,
            frame: Some(frame),
        };
        let _ = self.sender.send(Ok(frame)).await;
    }

    /// The `NotLeader` of a replica whose fence read closed: the leader
    /// `Leader` names, under the highest term this replica knows.
    async fn not_leader(&mut self, shared: &Shared) {
        let leader = shared.leader(None);
        let frame = controller_frame::Frame::NotLeader(proto::NotLeader {
            leader_address: leader.address,
            successor_id: String::new(),
        });
        self.frame(leader.term, frame).await;
    }

    async fn end(&mut self, status: Status) {
        let _ = self.sender.send(Err(status)).await;
    }
}

/// What a `Welcome` tells of the applied spec. No generation is applied
/// yet: generation `0`, no needs and no hashing rule. The holder of the
/// applied generation fills this one function.
fn applied() -> (u64, Option<proto::NeedsSummary>, Vec<proto::ModelHashing>) {
    (0, None, Vec::new())
}

/// The status of a `Hello` the presence table refused.
fn refusal(refused: Refused) -> Status {
    match refused {
        Refused::AnotherSubject => Status::already_exists(
            "Hello's gateway id collides with another present gateway's in the controller's key: use another id",
        ),
        Refused::Exhausted => {
            Status::resource_exhausted("the controller cannot start another session")
        }
        Refused::Full { ceiling } => Status::resource_exhausted(format!(
            "the controller holds its ceiling of {ceiling} present gateways: say Hello again later"
        )),
        // The election stepped between the session's read of the fence and
        // the table's: the answer is a fenced session's.
        Refused::Stepped { shown: None, .. } => Status::failed_precondition(NOT_LEADING),
        Refused::Stepped {
            read,
            shown: Some(later),
        } => Status::aborted(format!(
            "this controller leads under term {later} now and read this Hello under term {read}: say Hello again"
        )),
    }
}

/// The status of a frame a stream could not read, `frame` naming the
/// stream's kind of frame. The transport's own stands, but for a message
/// over the transport's limit: the contract never raises the transport's
/// code for that one, and the code alone tells the case.
pub(crate) fn unread(frame: &str, status: Status) -> Status {
    if status.code() == Code::OutOfRange {
        Status::resource_exhausted(format!(
            "{frame} is over the transport's limit: {}",
            status.message()
        ))
    } else {
        status
    }
}

/// The answer to a `Hello` on a replica that does not lead: the
/// `NotLeader` goes out, and the status follows it.
async fn not_leading(shared: &Shared, answers: &mut Answers) -> Status {
    answers.not_leader(shared).await;
    Status::failed_precondition(NOT_LEADING)
}

/// A peer's release string as the log reads it: at most [`RELEASE_MAX`]
/// bytes of visible ASCII, else a fixed word. A peer writes no line of its
/// own making into the controller's log.
fn release(version: &str) -> &str {
    if version.len() <= RELEASE_MAX && version.bytes().all(|byte| byte.is_ascii_graphic()) {
        version
    } else {
        "(unreadable)"
    }
}

/// A frame's kind as the contract names it.
const fn kind(frame: &Frame) -> &'static str {
    match frame {
        Frame::Hello(_) => "Hello",
        Frame::HelloPage(_) => "HelloPage",
        Frame::Dispatched(_) => "Dispatched",
        Frame::Completed(_) => "Completed",
        Frame::Released(_) => "Released",
        Frame::Veto(_) => "GatewayVeto",
        Frame::SpecAck(_) => "SpecAck",
    }
}

/// One session, from its first frame to its end. `Some` is the status the
/// stream ends with; `None` is a stream the gateway closed.
async fn session(
    shared: &Shared,
    principal: &Principal,
    mut inbound: Streaming<proto::GatewayFrame>,
    answers: &mut Answers,
) -> Option<Status> {
    // The election's steps from the stream's open on: the receiver is
    // marked seen here, so the wait for the Hello hears the steps after
    // the open and the loop after the Welcome hears on. Every mark of it
    // is made before the session's read of the fence, so no step after
    // that read is missed.
    let mut steps = shared.standing.clone();
    steps.mark_unchanged();
    let mut election = true;
    let first = {
        // The deadline's timer lives as long as the wait: the block's end
        // drops it at the Hello.
        let deadline = tokio::time::sleep(shared.hello_deadline);
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                first = inbound.message() => match first {
                    Ok(Some(first)) => break first,
                    Ok(None) => return None,
                    Err(status) => return Some(unread("a Session frame", status)),
                },
                () = &mut deadline => {
                    return Some(Status::deadline_exceeded(format!(
                        "no Hello came within {:?} of the stream's open",
                        shared.hello_deadline
                    )));
                }
                stepped = steps.changed(), if election => {
                    if stepped.is_err() {
                        // The election is gone with the run loop: the deadline
                        // bounds the wait.
                        election = false;
                    } else if shared.fence.leader_term(Instant::now()).is_none() {
                        return Some(not_leading(shared, answers).await);
                    }
                }
            }
        }
    };
    let now = Instant::now();
    let hello = match first.frame {
        Some(Frame::Hello(hello)) => hello,
        other => {
            let came = other.as_ref().map_or("a frame of no kind", kind);
            return Some(Status::failed_precondition(format!(
                "no session stands on this stream: Hello is the first frame, {came} came first"
            )));
        }
    };
    if hello.gateway_id != principal.subject {
        return Some(Status::unauthenticated(
            "Hello names a gateway id that is not its call's principal",
        ));
    }
    if !(hello.protocol_min..=hello.protocol_max).contains(&PROTOCOL_VERSION) {
        return Some(Status::failed_precondition(format!(
            "no protocol version in common: the gateway speaks {} to {}, this controller {PROTOCOL_VERSION}",
            hello.protocol_min, hello.protocol_max
        )));
    }
    if let Some(status) = listed(&hello) {
        return Some(status);
    }
    // One read: the verdict and the term the Welcome carries.
    let Some(term) = shared.fence.leader_term(now) else {
        return Some(not_leading(shared, answers).await);
    };
    if let Some(status) = floor(shared, &principal.subject, first.term_floor, term) {
        return Some(status);
    }
    let gateway = principal.gateway();
    let seat = match shared
        .gateways
        .welcome(gateway, &principal.subject, term, now)
    {
        Ok(seat) => seat,
        // The fence closed between this session's read and the table's.
        Err(Refused::Stepped { shown: None, .. }) => {
            return Some(not_leading(shared, answers).await);
        }
        Err(refused) => return Some(refusal(refused)),
    };
    // The gateway's vetoes, whole, held before the Welcome leaves.
    let vetoes: Vec<(WorkerId, i32, Instant)> = hello
        .vetoes
        .iter()
        .map(|veto| {
            let remaining = Duration::from_micros(veto.remaining_micros.into());
            (WorkerId(veto.worker_id), veto.kind, now + remaining)
        })
        .collect();
    let vetoes_restored = seat.restore(&vetoes, now);
    let (generation, needs, hashing) = applied();
    let welcome = proto::Welcome {
        protocol: PROTOCOL_VERSION,
        generation,
        controller_id: shared.controller_id.clone(),
        principal: Some(proto::Principal {
            subject: principal.subject.clone(),
            scope: proto::principal::Scope::Gateway.into(),
            method: principal.method.into(),
            role: String::new(),
        }),
        needs,
        hashing,
        // A decide stream's credit at its open, the streams a gateway holds
        // at most, and the time it has to name the dispatch of a lease.
        decide_window: decide::WINDOW,
        max_decide_streams: decide::MAX_STREAMS,
        dispatch_ttl_ms: millis(DISPATCH_TTL),
        grace_period_ms: millis(shared.gateways.grace()),
        leader_poll_ms: millis(LEADER_POLL),
        // The design sends no page: no lease and no pin is restored.
        leases_restored: 0,
        pins_restored: 0,
        vetoes_restored: u32::try_from(vetoes_restored).unwrap_or(u32::MAX),
    };
    answers
        .frame(term, controller_frame::Frame::Welcome(welcome))
        .await;
    info!(
        gateway = %principal.subject,
        version = %release(&hello.version),
        term,
        vetoes = hello.vetoes.len(),
        vetoes_restored,
        "a gateway is welcomed"
    );

    // The session's hand-over at the leases' file: what its frames say of
    // a lease is handed there, and applied before the session leaves.
    let hand = Hand::new(gateway);
    let status = loop {
        tokio::select! {
            frame = inbound.message() => {
                let frame = match frame {
                    Ok(Some(frame)) => frame,
                    Ok(None) => break None,
                    Err(status) => break Some(unread("a Session frame", status)),
                };
                if let Some(status) = judge(shared, &principal.subject, &hand, term, &seat, &frame, answers).await {
                    break Some(status);
                }
            }
            stepped = steps.changed(), if election => {
                if stepped.is_err() {
                    // The election is gone with the run loop: the drain
                    // closes the stream.
                    election = false;
                } else if let Some(status) = fenced(shared, term, Instant::now(), answers).await {
                    break Some(status);
                }
            }
            () = seat.superseded() => break Some(replaced()),
        }
    };
    debug!(gateway = %principal.subject, term, ended = ?status.as_ref().map(Status::code), "a gateway's session ended");
    shared.store.hand_back(hand);
    // The seat's drop starts the gateway's grace period.
    drop(seat);
    status
}

/// The fence's verdict at `now` on a session welcomed under `term`:
/// `None` while the fence shows that term.
async fn fenced(shared: &Shared, term: u64, now: Instant, answers: &mut Answers) -> Option<Status> {
    match shared.fence.leader_term(now) {
        Some(shown) if shown == term => None,
        Some(later) => Some(Status::aborted(format!(
            "this controller leads under term {later} now and welcomed the session under term {term}: say Hello again"
        ))),
        None => {
            answers.not_leader(shared).await;
            Some(Status::aborted(
                "this controller no longer leads: find the leader through Leader",
            ))
        }
    }
}

/// A floor above the term this replica leads under: it has not noticed it
/// lost, or its lease was wiped beside a gateway that remembers. The frame
/// is refused, and the floor is said to the election once a rising floor
/// (the shared state's `floors`): the leader gives the lease up and the
/// next take is above the floor. A floor that is not believed
/// ([`unbelieved`]) is refused by its field's name and says nothing. Past
/// the budget of terms one process moves on such floors the frame is
/// refused in the same words and nothing is said, and so is a floor whose
/// take would land past the reach of every process's floors.
fn floor(shared: &Shared, subject: &str, floor: u64, term: u64) -> Option<Status> {
    if floor <= term {
        return None;
    }
    if let Some(why) = unbelieved(floor, term) {
        return Some(Status::invalid_argument(why));
    }
    match shared.floors.say(floor, term) {
        Told::Said => warn!(
            gateway = %subject,
            floor,
            term,
            "a gateway's term floor is above this controller's term; the election is told to leave the lease"
        ),
        Told::Spent { first: true } => warn!(
            gateway = %subject,
            floor,
            term,
            budget = FLOORS_SPENT_MAX,
            "this process has too few terms left for a gateway's floor; a floor past what is left is refused and moves nothing until the process starts again"
        ),
        Told::Past { first: true } => warn!(
            gateway = %subject,
            floor,
            term,
            reach = FLOOR_REACH,
            "a gateway's term floor is past the terms a floor may move the lease to; it is refused and moves nothing, also after a restart"
        ),
        Told::Kept | Told::Spent { first: false } | Told::Past { first: false } => {}
    }
    Some(Status::failed_precondition(format!(
        "the gateway's term floor {floor} is above this controller's term {term}"
    )))
}

/// The end of a session a newer one of its gateway replaced.
fn replaced() -> Status {
    Status::aborted("a newer session of this gateway took this one's place")
}

/// The form of a `Hello`'s veto list, judged before anything is applied and
/// before a presence is made: its count is the list's length, it holds
/// [`LISTED_MAX`] vetoes at most, no entry is a veto of no kind, and no
/// worker holds more than [`KINDS_MAX`] kinds in it.
fn listed(hello: &proto::Hello) -> Option<Status> {
    if usize::try_from(hello.veto_count).ok() != Some(hello.vetoes.len()) {
        return Some(Status::invalid_argument(format!(
            "Hello counts {} vetoes and lists {}",
            hello.veto_count,
            hello.vetoes.len()
        )));
    }
    if hello.vetoes.len() > LISTED_MAX {
        return Some(Status::resource_exhausted(format!(
            "Hello lists {} vetoes: a list holds {LISTED_MAX} at most",
            hello.vetoes.len()
        )));
    }
    if hello.vetoes.iter().any(|veto| veto.kind == 0) {
        return Some(Status::invalid_argument(
            "Hello lists a veto of no kind: KIND_UNSPECIFIED is not a veto",
        ));
    }
    // An entry with nothing remaining is no veto: it takes no place among
    // a worker's kinds.
    let kinds = hello
        .vetoes
        .iter()
        .filter(|veto| veto.remaining_micros > 0)
        .map(|veto| (WorkerId(veto.worker_id), veto.kind));
    over_kinds(kinds).then(|| {
        Status::resource_exhausted(format!(
            "Hello lists more than {KINDS_MAX} kinds of veto on one worker"
        ))
    })
}

/// The `GatewayVeto` arm: one kind's veto set from the frame's instant, or
/// lifted by `clear`. A duration of zero ends at that instant, which the
/// table reads as no veto: it lifts as `clear` does. `Some` ends the
/// session: a veto of no kind is neither kept nor dropped in silence.
fn vetoed(seat: &SessionPlace, veto: &proto::GatewayVeto, now: Instant) -> Option<Status> {
    if veto.kind == 0 {
        return Some(Status::invalid_argument(
            "GatewayVeto names no kind: KIND_UNSPECIFIED is not a veto",
        ));
    }
    let until = (!veto.clear).then(|| now + Duration::from_micros(veto.for_micros.into()));
    match seat.veto(WorkerId(veto.worker_id), veto.kind, until, now) {
        Vetoed::Shown => None,
        // The table's lock asked the seat's question again and a newer
        // session stands: the same end.
        Vetoed::Replaced => Some(replaced()),
        Vetoed::NoWorker => {
            debug!(
                worker = veto.worker_id,
                kind = veto.kind,
                "a veto on a worker with no row is dropped"
            );
            None
        }
        // A veto frame has no answer in the stream, and one dropped in
        // silence would leave the gateway believing a veto stands that the
        // store does not hold: the session ends by name.
        Vetoed::Full => Some(Status::resource_exhausted(format!(
            "GatewayVeto would stand one kind more than the {KINDS_MAX} a worker holds for one gateway"
        ))),
        Vetoed::Over => Some(Status::resource_exhausted(format!(
            "GatewayVeto would stand one veto more than the {VETOES_MAX} a gateway holds"
        ))),
    }
}

/// One frame after the `Welcome`: the seat, the fence, the floor, then its
/// kind's arm. `Some` ends the session; the frame is then not applied.
async fn judge(
    shared: &Shared,
    subject: &str,
    hand: &Hand,
    term: u64,
    seat: &SessionPlace,
    frame: &proto::GatewayFrame,
    answers: &mut Answers,
) -> Option<Status> {
    // A session that was replaced applies no frame: the gateway's frames
    // are the newer session's from the instant this one is told.
    if seat.replaced() {
        return Some(replaced());
    }
    // One read of the clock a frame: the fence's verdict and the instant
    // an arm writes under are the same.
    let now = Instant::now();
    if let Some(status) = fenced(shared, term, now, answers).await {
        return Some(status);
    }
    if let Some(status) = floor(shared, subject, frame.term_floor, term) {
        return Some(status);
    }
    match &frame.frame {
        // A frame of no known kind tells its floor and nothing else.
        None => None,
        Some(Frame::Hello(_)) => Some(Status::failed_precondition(
            "a session stands on this stream: a second Hello needs a new stream",
        )),
        Some(Frame::Dispatched(dispatched)) => {
            lifecycle::dispatched(&shared.store, hand, dispatched, now).err()
        }
        Some(Frame::Completed(completed)) => {
            lifecycle::completed(&shared.store, hand, completed, now).err()
        }
        Some(Frame::Released(released)) => lifecycle::released(&shared.store, hand, released).err(),
        Some(Frame::Veto(veto)) => vetoed(seat, veto, now),
        Some(unserved @ (Frame::HelloPage(_) | Frame::SpecAck(_))) => Some(Status::unimplemented(
            format!("{} is not served by this build", kind(unserved)),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        net::SocketAddr,
        sync::Arc,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;
    use tokio::{
        net::TcpListener,
        sync::{mpsc, watch},
        task::JoinSet,
        time::timeout,
    };
    use tonic::{
        client::Grpc,
        codegen::{http::uri::PathAndQuery, tokio_stream::wrappers::ReceiverStream},
        transport::{server::TcpIncoming, Channel, Error, Server},
        Code, Request, Status, Streaming,
    };
    use tonic_prost::ProstCodec;

    use super::{
        floor, judge, refusal, release, Answers, GRACE_PERIOD, HELLO_DEADLINE, LEADER_POLL,
        LISTED_MAX, OUT_QUEUE, PRESENT_MAX, RELEASE_MAX,
    };
    use crate::{
        engine::view::WorkerId,
        service::{
            controller::{
                config::Scope,
                decision_slots::tests::followed,
                election::{ElectionObservation, LeasePlace, Said},
                floor::{FLOORS_SPENT_MAX, FLOOR_BELIEVED, FLOOR_REACH},
                gateways::{
                    tests::{lease, PATIENT},
                    Gateways, Refused, KINDS_MAX, VETOES_MAX,
                },
                lifecycle::tests::{a_completed, a_dispatched, a_released, worker},
                principal::{tests::told, Layer, ENTRY},
                replication::journaled_for,
                service::{Service, Shared},
                PROTOCOL_VERSION,
            },
            elector::{ElectionSettings, Elector, TickOutcome},
            proto::{
                self, controller_frame, gateway_frame::Frame,
                policy_controller_client::PolicyControllerClient,
            },
        },
        store::{HealthSink, LeaseState, LeasesStore},
        tracking::lease::LeaseId,
    };

    const ID: &str = "controller-0";
    const ADDRESS: &str = "10.0.0.7:30100";
    const IDENTITY: &str = "controller-0@10.0.0.7:30100";
    const GATEWAY: &str = "gateway-a";

    /// A gateway listener that takes identities as told, served over one
    /// candidate's fence, with the election's writer in the case's hand.
    struct Bed {
        elector: Elector,
        standing: watch::Sender<ElectionObservation>,
        shared: Arc<Shared>,
        addr: SocketAddr,
        /// Aborted at its drop, at the case's end.
        _serving: JoinSet<Result<(), Error>>,
    }

    async fn bed(settings: ElectionSettings) -> Bed {
        bed_with(settings, None, None).await
    }

    /// The same; with a ceiling, the presence table holds that many rows
    /// in place of the shared state's stock number.
    async fn bed_with(
        settings: ElectionSettings,
        ceiling: Option<usize>,
        hello_deadline: Option<Duration>,
    ) -> Bed {
        let elector = Elector::alone(IDENTITY, settings).expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let mut shared = Shared::new(
            ID.to_owned(),
            ADDRESS.to_owned(),
            &elector,
            stood,
            followed(journaled_for(&LeasePlace::Process, 0, 2)),
            Arc::new(crate::stages::StageRegistry::builtin()),
            2,
        )
        .expect("the deciders start");
        if let Some(ceiling) = ceiling {
            shared.gateways = Arc::new(Gateways::new(
                GRACE_PERIOD,
                ceiling,
                elector.fence(),
                Arc::clone(&shared.store),
            ));
        }
        if let Some(hello_deadline) = hello_deadline {
            shared.hello_deadline = hello_deadline;
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
            addr,
            _serving: serving,
        }
    }

    impl Bed {
        /// The candidate leads now under `term`, by a take or a renewal, and
        /// the election publishes the look. The bed's candidate is alone
        /// over its own lease: its first take leads at once.
        fn leads(&mut self, term: u64) {
            let now = Instant::now();
            if self.elector.is_holder() || self.elector.observed().is_some() {
                let tick = self.elector.tick(now).expect("a step");
                assert!(
                    tick == TickOutcome::Acquired { term } || tick == TickOutcome::Renewed { term },
                    "{tick:?}"
                );
            } else {
                self.elector.lead_at(now);
                assert_eq!(self.elector.fence().leader_term(now), Some(term));
            }
            self.step(IDENTITY, term);
        }

        /// One step of the election as a session sees it: the look is
        /// published.
        fn step(&self, holder: &str, term: u64) {
            self.standing.send_replace(ElectionObservation {
                holder: holder.to_owned(),
                term,
                looked: true,
            });
        }

        /// Whether the gateway is present at `now`.
        fn present(&self, now: Instant) -> bool {
            let gateway = told(GATEWAY).gateway();
            self.shared.gateways.presence(gateway, now).is_some()
        }
    }

    /// One `Session` stream of [`GATEWAY`], from the gateway's side.
    struct Opened<F = proto::GatewayFrame> {
        frames: mpsc::Sender<F>,
        answers: Streaming<proto::ControllerFrame>,
    }

    /// A frame of the case's own bytes. A `Hello` rides it as it rides a
    /// frame, and `dispatched` is the bytes the frame's `Dispatched` is
    /// decoded from.
    #[derive(Clone, PartialEq, prost::Message)]
    struct Odd {
        #[prost(message, optional, tag = "3")]
        hello: Option<proto::Hello>,
        #[prost(string, tag = "5")]
        dispatched: String,
    }

    /// One `Session` stream of [`GATEWAY`] that sends frames of the case's
    /// own bytes.
    async fn open_odd(addr: SocketAddr) -> Opened<Odd> {
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("a uri")
            .connect()
            .await
            .expect("the listener accepts");
        let (frames, sent) = mpsc::channel(8);
        let mut request = Request::new(ReceiverStream::new(sent));
        request
            .metadata_mut()
            .insert(ENTRY, GATEWAY.parse().expect("an ASCII value"));
        let mut call = Grpc::new(channel);
        call.ready().await.expect("the channel is ready");
        let answers = call
            .streaming(
                request,
                PathAndQuery::from_static("/smg.policy.v1.PolicyController/Session"),
                ProstCodec::<Odd, proto::ControllerFrame>::default(),
            )
            .await
            .expect("the stream opens")
            .into_inner();
        Opened { frames, answers }
    }

    async fn open(addr: SocketAddr) -> Opened {
        open_as(addr, GATEWAY).await
    }

    /// One `Session` stream of the gateway `name`.
    async fn open_as(addr: SocketAddr, name: &'static str) -> Opened {
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("a uri")
            .connect()
            .await
            .expect("the listener accepts");
        let (frames, sent) = mpsc::channel(8);
        let mut request = Request::new(ReceiverStream::new(sent));
        request
            .metadata_mut()
            .insert(ENTRY, name.parse().expect("an ASCII value"));
        let answers = PolicyControllerClient::new(channel)
            .session(request)
            .await
            .expect("the stream opens")
            .into_inner();
        Opened { frames, answers }
    }

    impl<F> Opened<F> {
        async fn send(&self, frame: F) {
            self.frames
                .send(frame)
                .await
                .expect("the stream takes the frame");
        }

        /// The next answer or the stream's end, bounded.
        async fn next(&mut self) -> Result<Option<proto::ControllerFrame>, Status> {
            timeout(Duration::from_secs(20), self.answers.message())
                .await
                .expect("an answer or the stream's end within 20 s")
        }

        async fn frame(&mut self) -> proto::ControllerFrame {
            self.next()
                .await
                .expect("a frame, not a status")
                .expect("a frame, not the stream's end")
        }

        /// The status the stream ends with, as its code and message.
        async fn status(&mut self) -> (Code, String) {
            let status = self
                .next()
                .await
                .expect_err("the stream ends with a status");
            (status.code(), status.message().to_owned())
        }
    }

    impl Opened {
        /// Says `Hello` and takes the `Welcome`, which carries `term`.
        async fn welcomed(&mut self, term: u64) -> proto::Welcome {
            self.send(hello(GATEWAY)).await;
            let answer = self.frame().await;
            assert_eq!(answer.term, term, "{answer:?}");
            match answer.frame {
                Some(controller_frame::Frame::Welcome(welcome)) => welcome,
                other => panic!("a Welcome answers a Hello: {other:?}"),
            }
        }
    }

    fn frame(term_floor: u64, kind: Option<Frame>) -> proto::GatewayFrame {
        proto::GatewayFrame {
            seq: 0,
            term_floor,
            frame: kind,
        }
    }

    fn hello(gateway_id: &str) -> proto::GatewayFrame {
        frame(
            0,
            Some(Frame::Hello(proto::Hello {
                gateway_id: gateway_id.to_owned(),
                protocol_min: PROTOCOL_VERSION,
                protocol_max: PROTOCOL_VERSION,
                version: "1.0.0-test".to_owned(),
                ..proto::Hello::default()
            })),
        )
    }

    fn not_leader(seq: u64, term: u64, leader_address: &str) -> proto::ControllerFrame {
        proto::ControllerFrame {
            seq,
            term,
            frame: Some(controller_frame::Frame::NotLeader(proto::NotLeader {
                leader_address: leader_address.to_owned(),
                successor_id: String::new(),
            })),
        }
    }

    #[tokio::test]
    async fn a_hello_is_welcomed_under_the_fences_term_and_the_gateway_is_present() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        assert!(!bed.present(Instant::now()));

        let mut session = open(bed.addr).await;
        session.send(hello(GATEWAY)).await;
        assert_eq!(
            session.frame().await,
            proto::ControllerFrame {
                seq: 1,
                term: 1,
                frame: Some(controller_frame::Frame::Welcome(proto::Welcome {
                    protocol: PROTOCOL_VERSION,
                    generation: 0,
                    controller_id: ID.to_owned(),
                    principal: Some(proto::Principal {
                        subject: GATEWAY.to_owned(),
                        scope: proto::principal::Scope::Gateway.into(),
                        method: proto::principal::Method::AsTold.into(),
                        role: String::new(),
                    }),
                    needs: None,
                    hashing: Vec::new(),
                    decide_window: 128,
                    max_decide_streams: 256,
                    grace_period_ms: 10_000,
                    dispatch_ttl_ms: 5_000,
                    leader_poll_ms: 1_000,
                    leases_restored: 0,
                    pins_restored: 0,
                    vetoes_restored: 0,
                })),
            }
        );
        assert_eq!(GRACE_PERIOD, Duration::from_secs(10));
        assert_eq!(LEADER_POLL, Duration::from_secs(1));
        // Present from the Welcome, however long the session stands.
        assert!(bed.present(Instant::now()));
        assert!(bed.present(Instant::now() + 100 * GRACE_PERIOD));
    }

    /// Three streams no session stands on: none makes the gateway present.
    #[tokio::test]
    async fn a_stream_begins_with_a_hello_of_its_own_gateway_in_a_version_this_build_speaks() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);

        let mut session = open(bed.addr).await;
        session
            .send(frame(
                0,
                Some(Frame::Dispatched(proto::Dispatched::default())),
            ))
            .await;
        assert_eq!(
            session.status().await,
            (
                Code::FailedPrecondition,
                "no session stands on this stream: Hello is the first frame, Dispatched came first"
                    .to_owned()
            )
        );

        let mut session = open(bed.addr).await;
        session.send(hello("gateway-b")).await;
        assert_eq!(
            session.status().await,
            (
                Code::Unauthenticated,
                "Hello names a gateway id that is not its call's principal".to_owned()
            )
        );

        let mut session = open(bed.addr).await;
        let mut ahead = hello(GATEWAY);
        if let Some(Frame::Hello(hello)) = ahead.frame.as_mut() {
            hello.protocol_min = PROTOCOL_VERSION + 1;
            hello.protocol_max = PROTOCOL_VERSION + 2;
        }
        session.send(ahead).await;
        assert_eq!(
            session.status().await,
            (
                Code::FailedPrecondition,
                "no protocol version in common: the gateway speaks 2 to 3, this controller 1"
                    .to_owned()
            )
        );

        assert!(!bed.present(Instant::now()));
    }

    #[tokio::test]
    async fn a_replica_that_does_not_lead_answers_a_hello_with_not_leader_and_the_leader_it_knows()
    {
        let bed = bed(PATIENT).await;
        bed.step("controller-1@10.0.0.8:30100", 4);

        let mut session = open(bed.addr).await;
        session.send(hello(GATEWAY)).await;
        assert_eq!(session.frame().await, not_leader(1, 4, "10.0.0.8:30100"));
        assert_eq!(
            session.status().await,
            (
                Code::FailedPrecondition,
                "this controller does not lead: find the leader through Leader".to_owned()
            )
        );
        assert!(!bed.present(Instant::now()));
    }

    #[tokio::test]
    async fn a_floor_above_the_term_is_refused_at_the_hello_and_at_a_frame() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let refusal = (
            Code::FailedPrecondition,
            "the gateway's term floor 2 is above this controller's term 1".to_owned(),
        );

        let mut session = open(bed.addr).await;
        let mut high = hello(GATEWAY);
        high.term_floor = 2;
        session.send(high).await;
        assert_eq!(session.status().await, refusal);
        assert!(!bed.present(Instant::now()));

        let mut session = open(bed.addr).await;
        session.welcomed(1).await;
        // A floor at the term is a report and nothing else: the session
        // stands, and the frame after it is the one that is answered.
        session.send(frame(1, None)).await;
        session.send(frame(2, None)).await;
        assert_eq!(session.status().await, refusal);
    }

    /// The floor's word. A believed floor above the term is said to the
    /// election once a rising floor, at a later frame or at a `Hello`, and
    /// the refusal reads as before. A floor at the term says nothing. One
    /// past the believed range is refused by its field's name and says
    /// nothing.
    #[tokio::test]
    async fn a_floor_above_the_term_is_said_to_the_election_once_a_rising_floor() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let (door, said) = Said::door();
        bed.shared.floors.open(door);
        let refused = |floor: u64| {
            (
                Code::FailedPrecondition,
                format!("the gateway's term floor {floor} is above this controller's term 1"),
            )
        };
        let hello_under = |floor: u64| {
            let mut hello = hello(GATEWAY);
            hello.term_floor = floor;
            hello
        };

        // At a frame after the Welcome. The frame before it, at the term,
        // says nothing.
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;
        session.send(frame(1, None)).await;
        session.send(frame(3, None)).await;
        assert_eq!(session.status().await, refused(3));
        assert_eq!(said.terms(), [3]);

        // At a Hello: the same floor is said no second time, a higher one
        // is, and a lower one after it is refused and not said.
        let mut session = open(bed.addr).await;
        session.send(hello_under(3)).await;
        assert_eq!(session.status().await, refused(3));
        assert!(said.terms().is_empty());
        let mut session = open(bed.addr).await;
        session.send(hello_under(5)).await;
        assert_eq!(session.status().await, refused(5));
        let mut session = open(bed.addr).await;
        session.send(hello_under(4)).await;
        assert_eq!(session.status().await, refused(4));
        assert_eq!(said.terms(), [5]);

        // Past the believed range, at a Hello and at a later frame: the
        // field's name, and no word.
        let far = 2 + FLOOR_BELIEVED;
        let unbelieved = (
            Code::InvalidArgument,
            format!(
                "term_floor is {far}: a floor is believed up to {FLOOR_BELIEVED} above this controller's term 1"
            ),
        );
        let mut session = open(bed.addr).await;
        session.send(hello_under(far)).await;
        assert_eq!(session.status().await, unbelieved);
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;
        session.send(frame(far, None)).await;
        assert_eq!(session.status().await, unbelieved);
        assert!(said.terms().is_empty());
    }

    /// The word's log line: one a rising floor, with the gateway, the floor
    /// and the term.
    #[tokio::test]
    async fn the_floors_word_logs_one_line_a_rising_floor() {
        // The subscriber of the Welcome's log case, for the same reason.
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let (door, said) = Said::door();
        bed.shared.floors.open(door);
        for _ in 0..2 {
            let mut session = open(bed.addr).await;
            let mut high = hello(GATEWAY);
            high.term_floor = 3;
            session.send(high).await;
            assert_eq!(session.status().await.0, Code::FailedPrecondition);
        }
        assert_eq!(said.terms(), [3]);

        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let line = "a gateway's term floor is above this controller's term";
        assert_eq!(logged.matches(line).count(), 1, "{logged}");
        assert!(
            logged.contains("gateway=gateway-a floor=3 term=1"),
            "{logged}"
        );
    }

    /// The floors one process says spend a budget of terms. Fifteen of
    /// the whole believed range fit, each above the term the one before it
    /// left and each with its take's term. The sixteenth is past the
    /// budget: refused in the same words, it says nothing, and the log
    /// says so once.
    #[tokio::test]
    async fn a_floor_past_the_budget_of_terms_is_refused_as_before_and_says_nothing() {
        // The subscriber of the Welcome's log case, for the same reason.
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let bed = bed(PATIENT).await;
        let (door, said) = Said::door();
        bed.shared.floors.open(door);
        let mut term = 1;
        for _ in 1..FLOORS_SPENT_MAX / FLOOR_BELIEVED {
            let high = term + FLOOR_BELIEVED;
            let status = floor(&bed.shared, GATEWAY, high, term).expect("refused");
            assert_eq!(status.code(), Code::FailedPrecondition);
            assert_eq!(said.terms(), [high]);
            term = high + 1;
        }
        let high = term + FLOOR_BELIEVED;
        for _ in 0..2 {
            let status = floor(&bed.shared, GATEWAY, high, term).expect("refused");
            assert_eq!(status.code(), Code::FailedPrecondition);
            assert_eq!(
                status.message(),
                format!("the gateway's term floor {high} is above this controller's term {term}")
            );
        }
        assert!(said.terms().is_empty());

        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let line = "this process has too few terms left for a gateway's floor";
        assert_eq!(logged.matches(line).count(), 1, "{logged}");
    }

    /// The floors of every process move the lease to the reach and no
    /// further: a floor whose take would land past it is refused in the
    /// same words, it says nothing, and the log says so once.
    #[tokio::test]
    async fn a_floor_past_the_reach_is_refused_as_before_and_says_nothing() {
        // The subscriber of the Welcome's log case, for the same reason.
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let bed = bed(PATIENT).await;
        let (door, said) = Said::door();
        bed.shared.floors.open(door);
        // The last floor a word moves the lease for: its take is the reach.
        let term = FLOOR_REACH - 2;
        let status = floor(&bed.shared, GATEWAY, FLOOR_REACH - 1, term).expect("refused");
        assert_eq!(status.code(), Code::FailedPrecondition);
        assert_eq!(said.terms(), [FLOOR_REACH - 1]);
        let term = FLOOR_REACH;
        for high in [term + 1, term + 2] {
            let status = floor(&bed.shared, GATEWAY, high, term).expect("refused");
            assert_eq!(status.code(), Code::FailedPrecondition);
            assert_eq!(
                status.message(),
                format!("the gateway's term floor {high} is above this controller's term {term}")
            );
        }
        assert!(said.terms().is_empty());

        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let line = "a gateway's term floor is past the terms a floor may move the lease to";
        assert_eq!(logged.matches(line).count(), 1, "{logged}");
    }

    #[tokio::test]
    async fn a_kind_no_step_serves_yet_is_refused_by_its_name() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let kinds = [
            (
                Frame::HelloPage(proto::HelloPage::default()),
                Code::Unimplemented,
                "HelloPage is not served by this build",
            ),
            (
                Frame::SpecAck(proto::SpecAck::default()),
                Code::Unimplemented,
                "SpecAck is not served by this build",
            ),
            (
                Frame::Hello(proto::Hello::default()),
                Code::FailedPrecondition,
                "a session stands on this stream: a second Hello needs a new stream",
            ),
        ];
        for (kind, code, message) in kinds {
            let mut session = open(bed.addr).await;
            session.welcomed(1).await;
            session.send(frame(0, Some(kind))).await;
            assert_eq!(session.status().await, (code, message.to_owned()));
        }
    }

    const ENDED: &str = "this controller no longer leads: find the leader through Leader";

    /// Past the renew deadline with no renewal the fence is closed by the
    /// clock alone. The session learns at its next frame: a `NotLeader`
    /// precedes the status, under the term the replica still knows, and
    /// names nobody, since the record still names this replica.
    #[tokio::test]
    async fn past_the_renew_deadline_the_session_ends_with_not_leader_at_its_next_frame() {
        let settings = ElectionSettings {
            duration: Duration::from_secs(60),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_secs(1),
        };
        let mut bed = bed(settings).await;
        // The stream first, the take after it, and the Hello inside the
        // take's renew deadline. A host that holds the case past it has
        // the Hello answered NotLeader, which is no verdict on a session:
        // the take and the Hello are made again, on a new stream.
        let mut tries = 0;
        let mut session = loop {
            let mut session = open(bed.addr).await;
            bed.leads(1);
            session.send(hello(GATEWAY)).await;
            match session.frame().await.frame {
                Some(controller_frame::Frame::Welcome(_)) => break session,
                // The host held the case past the deadline: again.
                Some(controller_frame::Frame::NotLeader(_)) => {}
                other => panic!("a Welcome or a NotLeader answers the Hello: {other:?}"),
            }
            tries += 1;
            assert!(
                tries < 20,
                "the host held the case past its deadline twenty times"
            );
        };

        tokio::time::sleep(settings.renew_deadline + Duration::from_millis(20)).await;
        session.send(frame(1, None)).await;
        assert_eq!(session.frame().await, not_leader(2, 1, ""));
        assert_eq!(session.status().await, (Code::Aborted, ENDED.to_owned()));
    }

    /// A session with no frame in flight learns of a closed fence at the
    /// election's next step.
    #[tokio::test]
    async fn a_fence_that_closed_ends_an_idle_session_at_the_elections_next_step() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;

        // The lease is given up: the fence is closed, and the next look
        // sees a free lease.
        assert!(bed.elector.release().expect("the release"));
        bed.step("", 1);
        assert_eq!(session.frame().await, not_leader(2, 1, ""));
        assert_eq!(session.status().await, (Code::Aborted, ENDED.to_owned()));
    }

    #[tokio::test]
    async fn a_take_under_a_later_term_ends_the_session_welcomed_under_the_earlier_one() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;

        assert!(bed.elector.release().expect("the release"));
        bed.leads(2);
        assert_eq!(
            session.status().await,
            (
                Code::Aborted,
                "this controller leads under term 2 now and welcomed the session under term 1: say Hello again"
                    .to_owned()
            )
        );

        // The gateway says Hello again, on a new stream, under the term
        // that stands.
        let mut again = open(bed.addr).await;
        again.welcomed(2).await;
    }

    #[tokio::test]
    async fn a_second_session_ends_the_first_and_the_last_streams_end_starts_the_grace_period() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let mut first = open(bed.addr).await;
        first.welcomed(1).await;
        let mut second = open(bed.addr).await;
        second.welcomed(1).await;
        assert_eq!(
            first.status().await,
            (
                Code::Aborted,
                "a newer session of this gateway took this one's place".to_owned()
            )
        );
        // The superseded stream's end is not the gateway's: no period runs.
        assert!(bed.present(Instant::now() + 100 * GRACE_PERIOD));

        // The standing stream's end is: the gateway stays present for the
        // grace period, and no longer.
        drop(second);
        let started = Instant::now();
        while bed.present(Instant::now() + GRACE_PERIOD) {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the stream's end started no grace period in 20 s"
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert!(bed.present(Instant::now()));
    }

    /// A `Hello` past the ceiling of present gateways is refused by name,
    /// and the gateway that is present is not disturbed.
    #[tokio::test]
    async fn a_hello_past_the_ceiling_of_present_gateways_is_refused_by_name() {
        // The shared state builds its own table under the stock ceiling.
        let stock = bed(PATIENT).await;
        assert_eq!(stock.shared.gateways.ceiling(), PRESENT_MAX);
        assert_eq!(stock.shared.hello_deadline, HELLO_DEADLINE);

        let mut bed = bed_with(PATIENT, Some(1), None).await;
        bed.leads(1);
        let mut first = open(bed.addr).await;
        first.welcomed(1).await;

        let mut second = open_as(bed.addr, "gateway-b").await;
        second.send(hello("gateway-b")).await;
        assert_eq!(
            second.status().await,
            (
                Code::ResourceExhausted,
                "the controller holds its ceiling of 1 present gateways: say Hello again later"
                    .to_owned()
            )
        );
        assert!(bed.present(Instant::now()), "the present gateway stands");
        // The gateway that is present says Hello again and is welcomed.
        let mut again = open(bed.addr).await;
        again.welcomed(1).await;
    }

    /// A stream that says no Hello is ended at its deadline.
    #[tokio::test]
    async fn a_stream_that_says_no_hello_is_ended_at_its_deadline() {
        let mut bed = bed_with(PATIENT, None, Some(Duration::from_millis(50))).await;
        bed.leads(1);
        let mut session = open(bed.addr).await;
        assert_eq!(
            session.status().await,
            (
                Code::DeadlineExceeded,
                "no Hello came within 50ms of the stream's open".to_owned()
            )
        );
        assert!(!bed.present(Instant::now()));
        assert_eq!(HELLO_DEADLINE, Duration::from_secs(5));
    }

    /// A stream that said no Hello yet ends at the step that finds the
    /// fence closed, as a Hello under a closed fence does: the drain's
    /// give-up is such a step.
    #[tokio::test]
    async fn a_silent_stream_ends_at_the_step_that_finds_the_fence_closed() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let mut session = open(bed.addr).await;
        // A step that finds the fence open changes nothing.
        bed.leads(1);
        // The lease is given up and the election publishes its look.
        assert!(bed.elector.release().expect("the release"));
        bed.step("", 1);
        assert_eq!(session.frame().await, not_leader(1, 1, ""));
        assert_eq!(
            session.status().await,
            (
                Code::FailedPrecondition,
                "this controller does not lead: find the leader through Leader".to_owned()
            )
        );
    }

    /// One frame of each kind of the lifecycle, applied in order by the
    /// session to its own gateway's leases.
    #[tokio::test]
    async fn the_lifecycles_three_frames_are_applied_by_the_session() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let gateway = told(GATEWAY).gateway();
        let store = Arc::clone(&bed.shared.store);
        let t0 = Instant::now();
        let served = worker(&store, "http://a:8000");
        // The store's term is the led one, as a leader's is: an id made by
        // hand is of a term before it, and names its file by its block.
        store.set_term(1);
        for id in 1..=3 {
            store.issue(lease(gateway, id, t0)).expect("a lease");
        }

        let mut session = open(bed.addr).await;
        session.welcomed(1).await;
        session
            .send(frame(1, Some(Frame::Dispatched(a_dispatched(1)))))
            .await;
        session
            .send(frame(1, Some(Frame::Completed(a_completed(2, served)))))
            .await;
        session
            .send(frame(1, Some(Frame::Released(a_released(3)))))
            .await;
        // A second Hello is refused by name: the three before it were read
        // in order and none ended the session.
        session.send(hello(GATEWAY)).await;
        assert_eq!(
            session.status().await,
            (
                Code::FailedPrecondition,
                "a session stands on this stream: a second Hello needs a new stream".to_owned()
            )
        );
        let state =
            |id: u64| LeasesStore::get(&*store, gateway, LeaseId(id)).map(|record| record.state);
        assert!(matches!(state(1), Some(LeaseState::Dispatched { .. })));
        assert_eq!(state(2), None);
        assert_eq!(state(3), None);
        assert!(store.completed_ok_since(gateway, served, t0));
    }

    /// The session's rule holds for the lifecycle: a frame under a closed
    /// fence is not applied.
    #[tokio::test]
    async fn a_lifecycle_frame_under_a_closed_fence_is_not_applied() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let gateway = told(GATEWAY).gateway();
        let store = Arc::clone(&bed.shared.store);
        store.set_term(1);
        store
            .issue(lease(gateway, 1, Instant::now()))
            .expect("a lease");
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;

        assert!(bed.elector.release().expect("the release"));
        session
            .send(frame(1, Some(Frame::Released(a_released(1)))))
            .await;
        assert_eq!(session.frame().await, not_leader(2, 1, ""));
        assert_eq!(session.status().await, (Code::Aborted, ENDED.to_owned()));
        assert_eq!(
            LeasesStore::get(&*store, gateway, LeaseId(1)).map(|record| record.state),
            Some(LeaseState::ISSUED),
            "a fenced replica releases nothing"
        );
    }

    /// A session that was replaced applies no frame, whichever arm the
    /// loop's turn took.
    #[tokio::test]
    async fn a_replaced_sessions_frame_is_not_applied() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let gateway = told(GATEWAY).gateway();
        let store = Arc::clone(&bed.shared.store);
        let (first, second) = (
            worker(&store, "http://a:8000"),
            worker(&store, "http://b:8000"),
        );
        let now = Instant::now();
        let seat = bed
            .shared
            .gateways
            .welcome(gateway, GATEWAY, 1, now)
            .expect("a seat");
        let (sender, _out) = mpsc::channel(OUT_QUEUE);
        let mut answers = Answers { sender, seq: 0 };

        // While its seat stands the session's frame is applied.
        let report = frame(1, Some(Frame::Completed(a_completed(1, first))));
        assert!(judge(
            &bed.shared,
            GATEWAY,
            &super::Hand::new(gateway),
            1,
            &seat,
            &report,
            &mut answers
        )
        .await
        .is_none());
        assert!(store.completed_ok_since(gateway, first, now));

        let newer = bed
            .shared
            .gateways
            .welcome(gateway, GATEWAY, 1, now)
            .expect("a seat");
        let late = frame(1, Some(Frame::Completed(a_completed(1, second))));
        let status = judge(
            &bed.shared,
            GATEWAY,
            &super::Hand::new(gateway),
            1,
            &seat,
            &late,
            &mut answers,
        )
        .await
        .expect("the replaced session ends");
        assert_eq!(
            (status.code(), status.message()),
            (
                Code::Aborted,
                "a newer session of this gateway took this one's place"
            )
        );
        assert!(
            !store.completed_ok_since(gateway, second, now),
            "a replaced session writes nothing"
        );

        // The session that replaced it was never told: its frame is applied.
        let third = worker(&store, "http://c:8000");
        let report = frame(1, Some(Frame::Completed(a_completed(1, third))));
        assert!(judge(
            &bed.shared,
            GATEWAY,
            &super::Hand::new(gateway),
            1,
            &newer,
            &report,
            &mut answers
        )
        .await
        .is_none());
        assert!(store.completed_ok_since(gateway, third, now));
    }

    /// And for the floor: a lifecycle frame whose floor is above the term
    /// is refused before its arm.
    #[tokio::test]
    async fn a_lifecycle_frame_under_a_floor_above_the_term_is_not_applied() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let gateway = told(GATEWAY).gateway();
        let store = Arc::clone(&bed.shared.store);
        store.set_term(1);
        store
            .issue(lease(gateway, 1, Instant::now()))
            .expect("a lease");
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;

        session
            .send(frame(2, Some(Frame::Released(a_released(1)))))
            .await;
        assert_eq!(
            session.status().await,
            (
                Code::FailedPrecondition,
                "the gateway's term floor 2 is above this controller's term 1".to_owned()
            )
        );
        assert_eq!(
            LeasesStore::get(&*store, gateway, LeaseId(1)).map(|record| record.state),
            Some(LeaseState::ISSUED),
            "a frame the floor refuses is not applied"
        );
    }

    #[tokio::test]
    async fn a_lifecycle_frame_that_reads_wrong_ends_the_session_by_its_fields_name() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;
        session
            .send(frame(
                1,
                Some(Frame::Dispatched(proto::Dispatched::default())),
            ))
            .await;
        assert_eq!(
            session.status().await,
            (
                Code::InvalidArgument,
                "Dispatched.side is 0, which names none this version knows".to_owned()
            )
        );
    }

    #[test]
    fn a_hello_the_table_refused_is_answered_by_its_reason() {
        let collides = refusal(Refused::AnotherSubject);
        assert_eq!(collides.code(), Code::AlreadyExists);
        assert_eq!(
            collides.message(),
            "Hello's gateway id collides with another present gateway's in the controller's key: use another id"
        );
        let exhausted = refusal(Refused::Exhausted);
        assert_eq!(exhausted.code(), Code::ResourceExhausted);
        assert_eq!(
            exhausted.message(),
            "the controller cannot start another session"
        );
        let full = refusal(Refused::Full { ceiling: 7 });
        assert_eq!(full.code(), Code::ResourceExhausted);
        assert_eq!(
            full.message(),
            "the controller holds its ceiling of 7 present gateways: say Hello again later"
        );
        let closed = refusal(Refused::Stepped {
            read: 1,
            shown: None,
        });
        assert_eq!(closed.code(), Code::FailedPrecondition);
        assert_eq!(
            closed.message(),
            "this controller does not lead: find the leader through Leader"
        );
        let later = refusal(Refused::Stepped {
            read: 1,
            shown: Some(2),
        });
        assert_eq!(later.code(), Code::Aborted);
        assert_eq!(
            later.message(),
            "this controller leads under term 2 now and read this Hello under term 1: say Hello again"
        );
    }

    /// A frame the stream cannot read ends it with a status, before the
    /// `Welcome` and after it: one over the transport's limit, and one no
    /// frame decodes from.
    #[tokio::test]
    async fn a_frame_the_stream_cannot_read_ends_it_with_a_status() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        // Over the transport's 4 MiB a message.
        let mut over = hello(GATEWAY);
        if let Some(Frame::Hello(hello)) = over.frame.as_mut() {
            hello.version = "v".repeat(5 * 1024 * 1024);
        }
        let over_the_limit = |(code, message): (Code, String)| {
            assert_eq!(code, Code::ResourceExhausted, "{message}");
            assert!(
                message.starts_with("a Session frame is over the transport's limit: ")
                    && message.contains("4194304"),
                "{message}"
            );
        };
        // No `Dispatched` decodes from this byte.
        let odd = Odd {
            hello: None,
            dispatched: "\u{f}".to_owned(),
        };
        // The transport's own status, which is INTERNAL for a message its
        // codec could not decode.
        let undecoded = |(code, message): (Code, String)| {
            assert_eq!(code, Code::Internal, "{message}");
            assert!(!message.is_empty());
        };

        // As the first frame: no session, and the gateway is not present.
        let mut session = open(bed.addr).await;
        session.send(over.clone()).await;
        over_the_limit(session.status().await);
        let mut session = open_odd(bed.addr).await;
        session.send(odd.clone()).await;
        undecoded(session.status().await);
        assert!(!bed.present(Instant::now()));

        // After the Welcome.
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;
        session.send(over).await;
        over_the_limit(session.status().await);
        let mut session = open_odd(bed.addr).await;
        let said = match hello(GATEWAY).frame {
            Some(Frame::Hello(hello)) => hello,
            other => panic!("a Hello: {other:?}"),
        };
        session
            .send(Odd {
                hello: Some(said),
                dispatched: String::new(),
            })
            .await;
        assert!(matches!(
            session.frame().await.frame,
            Some(controller_frame::Frame::Welcome(_))
        ));
        session.send(odd).await;
        undecoded(session.status().await);
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

    /// The Welcome's log line reads a peer's release string through its
    /// bound: the one place a peer's own text reaches the log.
    #[tokio::test]
    async fn the_welcomes_log_line_reads_a_forged_release_string_as_the_fixed_word() {
        // A subscriber of the case's own keeps the lines of this thread,
        // the session's among them. A second one stays registered beside
        // it: with one alone a line is on or off by the thread that
        // reaches it first, and a case beside this one reaches it under
        // none.
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let mut session = open(bed.addr).await;
        let mut forged = hello(GATEWAY);
        if let Some(Frame::Hello(hello)) = forged.frame.as_mut() {
            hello.version = "1.0\nERROR forged".to_owned();
        }
        session.send(forged).await;
        assert!(matches!(
            session.frame().await.frame,
            Some(controller_frame::Frame::Welcome(_))
        ));

        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        assert!(logged.contains("a gateway is welcomed"), "{logged}");
        assert!(logged.contains("version=(unreadable)"), "{logged}");
        assert!(!logged.contains("forged"), "{logged}");
    }

    /// The log reads a peer's release string only when it is a short run of
    /// visible ASCII: no line end, no control byte and no length of the
    /// peer's choosing reaches a log line.
    #[test]
    fn the_log_reads_a_peers_release_string_only_when_it_is_short_and_visible() {
        assert_eq!(release("1.16.0-dev+590cd998"), "1.16.0-dev+590cd998");
        assert_eq!(release(""), "");
        let longest = "v".repeat(RELEASE_MAX);
        assert_eq!(release(&longest), longest);
        let over = "v".repeat(RELEASE_MAX + 1);
        for unread in [
            "1.0\nERROR forged",
            "1.0 beta",
            "1.0\u{7}",
            "1.0\u{e9}",
            over.as_str(),
        ] {
            assert_eq!(release(unread), "(unreadable)", "{unread:?}");
        }
    }

    const NO_KIND: &str = "GatewayVeto names no kind: KIND_UNSPECIFIED is not a veto";
    /// A minute, in the contract's microseconds.
    const MINUTE: u32 = 60_000_000;

    fn veto(worker: WorkerId, kind: i32, for_micros: u32, clear: bool) -> proto::GatewayFrame {
        frame(
            0,
            Some(Frame::Veto(proto::GatewayVeto {
                worker_id: worker.0,
                kind,
                for_micros,
                clear,
            })),
        )
    }

    /// Whether the store shows the gateway's veto on `worker` now.
    fn vetoed(bed: &Bed, worker: WorkerId) -> bool {
        let gateway = told(GATEWAY).gateway();
        bed.shared.store.vetoed(gateway, worker, Instant::now())
    }

    /// Waits, bounded, until the store shows the veto as `stands`.
    async fn shows(bed: &Bed, worker: WorkerId, stands: bool) {
        let started = Instant::now();
        while vetoed(bed, worker) != stands {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the store shows the veto as {stands} within 20 s"
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// The frame's arm: a duration sets its kind's veto, `clear` or a zero
    /// duration lifts it, and a veto of no kind ends the session by name.
    #[tokio::test]
    async fn a_veto_frame_sets_its_kinds_veto_and_a_clear_or_a_zero_duration_lifts_it() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let w = worker(&bed.shared.store, "http://a:8000");
        let mut session = open(bed.addr).await;
        assert_eq!(session.welcomed(1).await.vetoes_restored, 0);

        session.send(veto(w, 1, MINUTE, false)).await;
        shows(&bed, w, true).await;
        session.send(veto(w, 1, MINUTE, true)).await;
        shows(&bed, w, false).await;
        // A kind this build has no name for, lifted by a zero duration.
        session.send(veto(w, 9, MINUTE, false)).await;
        shows(&bed, w, true).await;
        session.send(veto(w, 9, 0, false)).await;
        shows(&bed, w, false).await;

        session.send(veto(w, 0, MINUTE, false)).await;
        assert_eq!(
            session.status().await,
            (Code::InvalidArgument, NO_KIND.to_owned())
        );
        assert!(!vetoed(&bed, w), "a veto of no kind is not kept");
    }

    /// A frame that meets a closed fence is not applied: the session ends
    /// with its `NotLeader` and the store shows no veto.
    #[tokio::test]
    async fn a_veto_that_meets_a_closed_fence_is_not_applied() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let w = worker(&bed.shared.store, "http://a:8000");
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;

        // The lease is given up and no step is published: the frame itself
        // meets the fence.
        assert!(bed.elector.release().expect("the release"));
        session.send(veto(w, 1, MINUTE, false)).await;
        assert_eq!(session.frame().await, not_leader(2, 1, ""));
        assert_eq!(session.status().await, (Code::Aborted, ENDED.to_owned()));
        assert!(!vetoed(&bed, w));
    }

    /// A gateway's vetoes ride its `Hello` and are held before the
    /// `Welcome` leaves. A list that miscounts, or holds a veto of no kind,
    /// refuses the `Hello` whole.
    #[tokio::test]
    async fn a_hellos_vetoes_are_held_before_the_welcome_and_a_malformed_list_refuses_it() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let w = worker(&bed.shared.store, "http://a:8000");
        let assertion = |worker: WorkerId, kind: i32, remaining_micros: u32| proto::VetoAssertion {
            worker_id: worker.0,
            kind,
            remaining_micros,
        };
        let listing = |veto_count: u32, vetoes: Vec<proto::VetoAssertion>| {
            frame(
                0,
                Some(Frame::Hello(proto::Hello {
                    gateway_id: GATEWAY.to_owned(),
                    protocol_min: PROTOCOL_VERSION,
                    protocol_max: PROTOCOL_VERSION,
                    veto_count,
                    vetoes,
                    ..proto::Hello::default()
                })),
            )
        };

        // One veto more than a list holds, and one kind more than a worker
        // holds: each is one more than a ceiling of the controller's.
        let over = u32::try_from(LISTED_MAX + 1).expect("a small number");
        let long = (0..u64::from(over))
            .map(|n| assertion(WorkerId(1_000 + n), 1, MINUTE))
            .collect();
        let kinds = i32::try_from(KINDS_MAX + 1).expect("a small number");
        let many = (1..=kinds).map(|kind| assertion(w, kind, MINUTE)).collect();
        let refused = [
            (
                listing(2, vec![assertion(w, 1, MINUTE)]),
                Code::InvalidArgument,
                "Hello counts 2 vetoes and lists 1".to_owned(),
            ),
            (
                listing(2, vec![assertion(w, 1, MINUTE), assertion(w, 0, MINUTE)]),
                Code::InvalidArgument,
                "Hello lists a veto of no kind: KIND_UNSPECIFIED is not a veto".to_owned(),
            ),
            (
                listing(over, long),
                Code::ResourceExhausted,
                format!("Hello lists {over} vetoes: a list holds {LISTED_MAX} at most"),
            ),
            (
                listing(u32::try_from(kinds).expect("a small number"), many),
                Code::ResourceExhausted,
                format!("Hello lists more than {KINDS_MAX} kinds of veto on one worker"),
            ),
        ];
        for (hello, code, message) in refused {
            let mut session = open(bed.addr).await;
            session.send(hello).await;
            assert_eq!(session.status().await, (code, message));
            assert!(!vetoed(&bed, w), "nothing of a refused list is applied");
            assert!(!bed.present(Instant::now()), "and no presence is made");
        }

        // One veto the controller holds, one on a worker it holds no row
        // for, one with nothing remaining.
        let mut session = open(bed.addr).await;
        let list = vec![
            assertion(w, 1, MINUTE),
            assertion(WorkerId(77), 1, MINUTE),
            assertion(w, 2, 0),
        ];
        session.send(listing(3, list)).await;
        let welcome = match session.frame().await.frame {
            Some(controller_frame::Frame::Welcome(welcome)) => welcome,
            other => panic!("a Welcome answers a Hello: {other:?}"),
        };
        assert_eq!(welcome.vetoes_restored, 1);
        // No wait: the veto was held before the Welcome left.
        assert!(vetoed(&bed, w));
    }

    /// A `Hello` of [`GATEWAY`] that lists `vetoes`, counted right.
    fn hello_listing(vetoes: Vec<proto::VetoAssertion>) -> proto::GatewayFrame {
        frame(
            0,
            Some(Frame::Hello(proto::Hello {
                gateway_id: GATEWAY.to_owned(),
                protocol_min: PROTOCOL_VERSION,
                protocol_max: PROTOCOL_VERSION,
                veto_count: u32::try_from(vetoes.len()).expect("a small number"),
                vetoes,
                ..proto::Hello::default()
            })),
        )
    }

    /// An entry with nothing remaining is no veto: it takes no place among
    /// a worker's kinds. A ceiling of kinds that stand and one more that
    /// ran out are a list the controller holds.
    #[tokio::test]
    async fn a_hellos_entry_with_nothing_remaining_takes_no_place_among_a_workers_kinds() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let w = worker(&bed.shared.store, "http://a:8000");
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        let mut vetoes: Vec<proto::VetoAssertion> = (1..=kinds)
            .map(|kind| proto::VetoAssertion {
                worker_id: w.0,
                kind,
                remaining_micros: MINUTE,
            })
            .collect();
        vetoes.push(proto::VetoAssertion {
            worker_id: w.0,
            kind: kinds + 1,
            remaining_micros: 0,
        });
        let mut session = open(bed.addr).await;
        session.send(hello_listing(vetoes)).await;
        let welcome = match session.frame().await.frame {
            Some(controller_frame::Frame::Welcome(welcome)) => welcome,
            other => panic!("a Welcome answers a Hello: {other:?}"),
        };
        assert_eq!(
            usize::try_from(welcome.vetoes_restored).ok(),
            Some(KINDS_MAX)
        );
    }

    /// A gateway holds a ceiling of vetoes, what its `Hello` lists whole:
    /// the frame that would set one more ends the session by name, and
    /// nothing of it is written.
    #[tokio::test]
    async fn a_veto_of_one_more_than_a_gateway_holds_ends_the_session_by_name() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        let vetoes: Vec<proto::VetoAssertion> = (0..VETOES_MAX / KINDS_MAX)
            .flat_map(|n| {
                let w = worker(&bed.shared.store, &format!("http://w{n}:8000"));
                (1..=kinds).map(move |kind| proto::VetoAssertion {
                    worker_id: w.0,
                    kind,
                    remaining_micros: MINUTE,
                })
            })
            .collect();
        let one_more = worker(&bed.shared.store, "http://one-more:8000");
        let mut session = open(bed.addr).await;
        session.send(hello_listing(vetoes)).await;
        let welcome = match session.frame().await.frame {
            Some(controller_frame::Frame::Welcome(welcome)) => welcome,
            other => panic!("a Welcome answers a Hello: {other:?}"),
        };
        assert_eq!(
            usize::try_from(welcome.vetoes_restored).ok(),
            Some(VETOES_MAX)
        );

        session.send(veto(one_more, 1, MINUTE, false)).await;
        assert_eq!(
            session.status().await,
            (
                Code::ResourceExhausted,
                format!(
                    "GatewayVeto would stand one veto more than the {VETOES_MAX} a gateway holds"
                )
            )
        );
        assert!(!vetoed(&bed, one_more), "the refused veto is not written");
    }

    /// A worker holds a ceiling of kinds for one gateway: the frame that
    /// would set one more ends the session by name, and what stands stays.
    #[tokio::test]
    async fn a_veto_of_one_kind_more_than_a_worker_holds_ends_the_session_by_name() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let w = worker(&bed.shared.store, "http://a:8000");
        let mut session = open(bed.addr).await;
        session.welcomed(1).await;
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        for kind in 1..=kinds {
            session.send(veto(w, kind, MINUTE, false)).await;
        }
        session.send(veto(w, kinds + 1, MINUTE, false)).await;
        assert_eq!(
            session.status().await,
            (
                Code::ResourceExhausted,
                format!(
                    "GatewayVeto would stand one kind more than the {KINDS_MAX} a worker holds for one gateway"
                )
            )
        );
        assert!(vetoed(&bed, w), "the kinds that stood stand");
    }

    /// The table's lock asks the seat's question again: a veto frame of a
    /// session replaced after `judge` asked ends it by the seat's own
    /// status and writes nothing.
    #[tokio::test]
    async fn a_veto_frame_of_a_session_replaced_since_the_question_ends_it_the_same_way() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let gateway = told(GATEWAY).gateway();
        let w = worker(&bed.shared.store, "http://a:8000");
        let now = Instant::now();
        let seat = bed
            .shared
            .gateways
            .welcome(gateway, GATEWAY, 1, now)
            .expect("a seat");
        let _newer = bed
            .shared
            .gateways
            .welcome(gateway, GATEWAY, 1, now)
            .expect("a seat");
        // Past the question `judge` asks: the arm itself, as for a frame
        // asked about before the newer session's `Welcome`.
        let frame = proto::GatewayVeto {
            worker_id: w.0,
            kind: 1,
            for_micros: MINUTE,
            clear: false,
        };
        let status = super::vetoed(&seat, &frame, now).expect("the replaced session ends");
        assert_eq!(
            (status.code(), status.message()),
            (
                Code::Aborted,
                "a newer session of this gateway took this one's place"
            )
        );
        assert!(!vetoed(&bed, w), "a replaced session writes no veto");
    }
}
