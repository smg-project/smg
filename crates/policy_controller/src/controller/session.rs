//! The `Session` call (design 5.2 and 5.3): one stream a gateway. `Hello`
//! comes first and `Welcome` answers it; from the `Welcome` the gateway is
//! present ([`Gateways`](super::gateways::Gateways)) and stays so for the
//! grace period after the stream's end.
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
//! One arm a kind of frame. `Hello` is served here, and the lifecycle's
//! three frames in [`lifecycle`]. The other kinds are refused by name
//! until the step that serves each adds its arm, as the service refuses an
//! unserved call.
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
use tracing::{debug, info};

use super::{
    gateways::{Refused, Seat},
    lifecycle,
    principal::Principal,
    service::Shared,
    PROTOCOL_VERSION,
};
use crate::{
    proto::{self, controller_frame, gateway_frame::Frame},
    store::GatewayId,
};

/// The stream rule's grace period (design 4.3): how long a gateway stays
/// present after its session ended with no new `Hello`.
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

/// The status of a frame the stream could not read. The transport's own
/// stands, but for a message over the transport's limit: the contract
/// never raises the transport's code for that one, and the code alone
/// tells the case.
fn unread(status: Status) -> Status {
    if status.code() == Code::OutOfRange {
        Status::resource_exhausted(format!(
            "a Session frame is over the transport's limit: {}",
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

fn millis(duration: Duration) -> u32 {
    u32::try_from(duration.as_millis()).unwrap_or(u32::MAX)
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
                    Err(status) => return Some(unread(status)),
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
    // One read: the verdict and the term the Welcome carries.
    let Some(term) = shared.fence.leader_term(now) else {
        return Some(not_leading(shared, answers).await);
    };
    if let Some(status) = floor(first.term_floor, term) {
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
        // No call decides yet: with no credit a gateway decides locally.
        // The call that issues leases sets these three.
        decide_window: 0,
        max_decide_streams: 0,
        dispatch_ttl_ms: 0,
        grace_period_ms: millis(shared.gateways.grace()),
        leader_poll_ms: millis(LEADER_POLL),
        // The design sends no page: nothing is restored from a Hello.
        leases_restored: 0,
        pins_restored: 0,
    };
    answers
        .frame(term, controller_frame::Frame::Welcome(welcome))
        .await;
    info!(
        gateway = %principal.subject,
        version = %release(&hello.version),
        term,
        "a gateway is welcomed"
    );

    let status = loop {
        tokio::select! {
            frame = inbound.message() => {
                let frame = match frame {
                    Ok(Some(frame)) => frame,
                    Ok(None) => break None,
                    Err(status) => break Some(unread(status)),
                };
                if let Some(status) = judge(shared, gateway, term, &seat, &frame, answers).await {
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
/// lost.
fn floor(floor: u64, term: u64) -> Option<Status> {
    (floor > term).then(|| {
        Status::failed_precondition(format!(
            "the gateway's term floor {floor} is above this controller's term {term}"
        ))
    })
}

/// The end of a session a newer one of its gateway replaced.
fn replaced() -> Status {
    Status::aborted("a newer session of this gateway took this one's place")
}

/// One frame after the `Welcome`: the seat, the fence, the floor, then its
/// kind's arm. `Some` ends the session; the frame is then not applied.
async fn judge(
    shared: &Shared,
    gateway: GatewayId,
    term: u64,
    seat: &Seat,
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
    if let Some(status) = floor(frame.term_floor, term) {
        return Some(status);
    }
    match &frame.frame {
        // A frame of no known kind tells its floor and nothing else.
        None => None,
        Some(Frame::Hello(_)) => Some(Status::failed_precondition(
            "a session stands on this stream: a second Hello needs a new stream",
        )),
        Some(Frame::Dispatched(dispatched)) => {
            lifecycle::dispatched(&shared.store, gateway, dispatched, now).err()
        }
        Some(Frame::Completed(completed)) => {
            lifecycle::completed(&shared.store, gateway, completed, now).err()
        }
        Some(Frame::Released(released)) => {
            lifecycle::released(&shared.store, gateway, released).err()
        }
        Some(unserved @ (Frame::HelloPage(_) | Frame::Veto(_) | Frame::SpecAck(_))) => Some(
            Status::unimplemented(format!("{} is not served by this build", kind(unserved))),
        ),
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
        judge, refusal, release, Answers, GRACE_PERIOD, HELLO_DEADLINE, LEADER_POLL, OUT_QUEUE,
        PRESENT_MAX, RELEASE_MAX,
    };
    use crate::{
        controller::{
            config::Scope,
            election::Standing,
            gateways::{
                tests::{lease, PATIENT},
                Gateways, Refused,
            },
            lifecycle::tests::{a_completed, a_dispatched, a_released, worker},
            principal::{tests::told, Layer, ENTRY},
            service::{Service, Shared},
            PROTOCOL_VERSION,
        },
        elector::{ElectionSettings, Elector, Tick},
        lease::LeaseId,
        proto::{
            self, controller_frame, gateway_frame::Frame,
            policy_controller_client::PolicyControllerClient,
        },
        store::{HealthSink, LeaseState, LeasesStore, MemoryStore},
    };

    const ID: &str = "controller-0";
    const ADDRESS: &str = "10.0.0.7:30100";
    const IDENTITY: &str = "controller-0@10.0.0.7:30100";
    const GATEWAY: &str = "gateway-a";

    /// A gateway listener that takes identities as told, served over one
    /// candidate's fence, with the election's writer in the case's hand.
    struct Bed {
        elector: Elector,
        standing: watch::Sender<Standing>,
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
        let (standing, stood) = watch::channel(Standing::default());
        let mut shared = Shared::new(
            ID.to_owned(),
            ADDRESS.to_owned(),
            &elector,
            stood,
            Arc::new(MemoryStore::new()),
            Arc::new(crate::Catalog::builtin()),
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
                    tick == Tick::Acquired { term } || tick == Tick::Renewed { term },
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
            self.standing.send_replace(Standing {
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
                    decide_window: 0,
                    max_decide_streams: 0,
                    grace_period_ms: 10_000,
                    dispatch_ttl_ms: 0,
                    leader_poll_ms: 1_000,
                    leases_restored: 0,
                    pins_restored: 0,
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

    #[tokio::test]
    async fn every_kind_but_hello_is_refused_by_its_name_until_its_step_serves_it() {
        let mut bed = bed(PATIENT).await;
        bed.leads(1);
        let kinds = [
            (
                Frame::HelloPage(proto::HelloPage::default()),
                Code::Unimplemented,
                "HelloPage is not served by this build",
            ),
            (
                Frame::Veto(proto::GatewayVeto::default()),
                Code::Unimplemented,
                "GatewayVeto is not served by this build",
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
            Some(LeaseState::Issued),
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
        assert!(judge(&bed.shared, gateway, 1, &seat, &report, &mut answers)
            .await
            .is_none());
        assert!(store.completed_ok_since(gateway, first, now));

        let newer = bed
            .shared
            .gateways
            .welcome(gateway, GATEWAY, 1, now)
            .expect("a seat");
        let late = frame(1, Some(Frame::Completed(a_completed(1, second))));
        let status = judge(&bed.shared, gateway, 1, &seat, &late, &mut answers)
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
        assert!(
            judge(&bed.shared, gateway, 1, &newer, &report, &mut answers)
                .await
                .is_none()
        );
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
            Some(LeaseState::Issued),
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
}
