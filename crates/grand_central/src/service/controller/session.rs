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
    PROTOCOL_FLOOR, PROTOCOL_VERSION,
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
    let Some(protocol) = spoken(hello.protocol_min, hello.protocol_max) else {
        return Some(Status::failed_precondition(format!(
            "no protocol version in common: the gateway speaks {} to {}, this controller {PROTOCOL_FLOOR} to {PROTOCOL_VERSION}",
            hello.protocol_min, hello.protocol_max
        )));
    };
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
        protocol,
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

/// The protocol version a session speaks: the newest that a `Hello`'s range
/// from `min` to `max` and this build's own have in common, and `None`
/// where they share none.
const fn spoken(min: u32, max: u32) -> Option<u32> {
    let newest = if max < PROTOCOL_VERSION {
        max
    } else {
        PROTOCOL_VERSION
    };
    if newest >= PROTOCOL_FLOOR && newest >= min {
        Some(newest)
    } else {
        None
    }
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
            lifecycle::completed(&shared.store, hand, completed).err()
        }
        Some(Frame::Released(released)) => lifecycle::released(&shared.store, hand, released).err(),
        Some(Frame::Veto(veto)) => vetoed(seat, veto, now),
        Some(unserved @ (Frame::HelloPage(_) | Frame::SpecAck(_))) => Some(Status::unimplemented(
            format!("{} is not served by this build", kind(unserved)),
        )),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/session/tests.rs"]
mod tests;
