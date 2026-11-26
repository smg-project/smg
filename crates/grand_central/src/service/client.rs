//! Clients for the controller's gateway protocol, behind the `client`
//! feature.
//!
//! Both clients dial a gateway listener, ask who leads and send a `Hello`
//! to open a session. The leader answers with a `Welcome`. The session
//! carries `Dispatched`, `Completed` and `Released` reports. The clients
//! raise their term floor from answers and refuse terms below it.
//!
//! [`Client`] is the reference client. It asks for a decision one request at
//! a time ([`Client::decide`]). [`Link`] shares the session and one decide
//! stream across a host's requests and releases the leases of a decision
//! its caller never takes.
//! [`DecisionGuard::asked`](crate::tracking::decision_guard::DecisionGuard::asked)
//! holds a linked decision and sends its reports on the session that stands
//! when it reports ([`Link::redial`]). A document the controller pushes reaches
//! the link's holder ([`Link::spec_pushes`]).
//!
//! A watch of the controller's workers opens on either client's channel
//! ([`Client::watch_workers`], [`Link::watch_workers`]). Once open it has
//! its own stream, so the session and asks can proceed while the watch
//! waits.
//!
//! The client takes its channel and makes no credential:
//! [`Client::connect`] dials an address in the clear, and
//! [`Client::connect_with`] an endpoint the caller configured with its own
//! TLS material. Every call states the gateway's id in one metadata entry
//! (`x-smg-principal`); a stated id is no credential. [`blocking::Client`]
//! drives the same calls for a caller that runs no runtime.
//!
//! No wait of [`Client`] for its peer but one outlasts [`ANSWER_WAIT`], on
//! any endpoint: the dial, each call from its send to its answer's end, the
//! session's call with its first frame, a report's wait for room in the
//! session's queue, the wait for the session's end, the decide stream's
//! open, an ask's waits for room in its queue and for its answer, the wait
//! for the decide stream's end, and a watch's open. An endpoint's own
//! time-outs end a wait sooner, and as the channel's error, not as the
//! client's [`Error::NoAnswer`]: [`Error::Status`] for a call and
//! [`Error::Connect`] for the dial. The one is a watch's next event
//! ([`Watch::next`]): a fleet that does not change tells nothing, and a
//! peer gone silent ends that wait through the channel's keepalive
//! ([`KEEPALIVE`]).
//!
//! A [`Link`] holds [`ANSWER_WAIT`] over its dial, its `Leader` ask and the
//! opens of its session, decide stream and watch. An ask waits its
//! caller's budget ([`Link::ask`]) and a report waits for nothing
//! ([`Link::queue_frame`]). The readers of the session and decide stream wait
//! with no bound of their own. The link sets the channel's keepalive,
//! which ends a session whose controller went silent ([`Link::connect`]).

use std::{
    future::{poll_fn, Future},
    pin::pin,
    task::Poll,
    time::Duration,
};

use prost::Message as _;
use tokio::sync::mpsc;
use tonic::{
    codegen::tokio_stream::{wrappers::ReceiverStream, Stream},
    metadata::AsciiMetadataValue,
    service::{interceptor::InterceptedService, Interceptor},
    transport::{Channel, Endpoint},
    Request, Status, Streaming,
};

use crate::service::{
    controller::{MAX_FRAME, MAX_REQUEST_ID, PRINCIPAL_ENTRY, PROTOCOL_VERSION},
    proto::{
        controller_frame, decide_answer, gateway_frame, grand_central_client::GrandCentralClient,
        Busy, Completed, ControllerFrame, DecideAnswer, DecideRequest, Decision, Dispatched,
        EncodePlaced, GatewayFrame, Hello, LeaderInfo, LeaderRequest, Released,
        WatchWorkersRequest, Welcome, WorkerEvent,
    },
};

mod link;
pub(crate) mod wire;

pub use link::{DecisionUnavailable, Link, SpecPushes};

/// Frames the client may queue toward the controller.
const SESSION_QUEUE: usize = 16;

/// The bytes one block hash takes in a frame: a fixed 64 bits of a packed
/// list.
const HASH: usize = 8;

/// The most block hashes one frame can hold. A frame's other fields take
/// room too, so the fit may cut a few more.
pub(crate) const MAX_HASHES: usize = MAX_FRAME / HASH;

/// The bound on each wait of the client for its peer: an answer a leader
/// under load can give, and a wait a blocking caller can afford. The client
/// lays it over the dial and over each call whole, since an endpoint's own
/// time-outs end at a call's response headers and a peer may stall after
/// them.
pub const ANSWER_WAIT: Duration = Duration::from_secs(5);

/// How often a channel [`Client::connect`] dials asks its peer for a sign
/// of life. A peer that answers none within [`ANSWER_WAIT`] ends the
/// channel's streams, a watch's wait among them.
pub const KEEPALIVE: Duration = Duration::from_secs(5);

type Stub = GrandCentralClient<InterceptedService<Channel, Principal>>;

/// Why a call of the client failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The address is no URI an endpoint can be made of.
    #[error("the address is not a URI: {0}")]
    Address(String),
    /// The gateway id is no value a metadata entry carries.
    #[error("the gateway id is no metadata value: {0:?}")]
    GatewayId(String),
    /// The dial failed.
    #[error("the connection failed: {0}")]
    Connect(#[source] tonic::transport::Error),
    /// The peer did not answer inside the client's bound. Every wait that
    /// bound ends is this kind and names the wait: the dial, the `Leader`
    /// ask, the session's welcome, the session's queue, the session's
    /// end, the decide stream's open, its queue, its end, the decision or
    /// the watch's open.
    /// A time-out the caller set on its own endpoint is not this
    /// kind: it ends a call as [`Error::Status`] and the dial as
    /// [`Error::Connect`].
    #[error("no answer within {wait:?}: {what}")]
    NoAnswer {
        /// The wait that ran out.
        what: &'static str,
        /// The bound it ran out under.
        wait: Duration,
    },
    /// The call ended with a status: the controller's answer, or the
    /// channel's own, as when a time-out the caller set on its endpoint
    /// ends the call.
    #[error("the call ended with a status: {0}")]
    Status(#[source] Box<Status>),
    /// The replica dialed does not lead. The address is the leader's as the
    /// replica names it, empty when it knows none.
    #[error("the replica does not lead; it names the leader at {0:?}")]
    NotLeader(String),
    /// The session ended, or answered the hello with a frame that is no
    /// welcome.
    #[error("the session answered the hello with no welcome: {0}")]
    NoWelcome(String),
    /// A `Leader` answer, a frame of the session or an answer of the decide
    /// stream came under a term below the client's floor. A gateway rejects
    /// an answer below its floor whatever it carries. A `Leader` answer below
    /// the floor is not returned. At the open the session is not taken, and
    /// the address a `NotLeader` names is not read; after the welcome the
    /// frame is read no further. A decide stream is dropped.
    #[error("an answer's term {term} is below the floor {floor}")]
    BelowFloor {
        /// The term the answer came under.
        term: u64,
        /// The floor the answer was read under.
        floor: u64,
    },
    /// An ask's request id is outside the call's range. The controller
    /// ends a stream at such a frame, so the client refuses it at its own
    /// door and sends nothing. A caller whose key is longer sends a digest
    /// of it, never a cut: a cut key can be two requests'.
    #[error("a request id of {0} bytes is outside the call's range of 1 to {most} bytes", most = MAX_REQUEST_ID)]
    RequestId(usize),
    /// An ask is over the call's bound on a frame as encoded with every
    /// block hash cut: the client drops no other field for size, and sends
    /// nothing.
    #[error("a DecideRequest of {0} bytes is over the call's limit of {limit} bytes with no block hash to cut", limit = MAX_FRAME)]
    TooLarge(usize),
    /// A field of an ask's facts is outside the range the call reads, named
    /// by its path in the contract. The controller ends a stream at such a
    /// frame, and a link's one stream is every caller's: the link refuses
    /// the frame at its own door and sends nothing ([`Link::ask`]).
    ///
    /// A hard demand of a host's request is refused under this kind too, as
    /// `constraints.demands`: this protocol version states none, so a
    /// handle sends no request that states one
    /// ([`DecisionGuard::asked`](crate::tracking::decision_guard::DecisionGuard::asked)).
    #[error("an ask's {0} is out of its range in this protocol version")]
    OutOfRange(&'static str),
    /// A link's report queue was full: its session took no report for as
    /// long as the queue covers. The session ended, and the reports its
    /// queue held are lost with it ([`Link::queue_frame`]).
    #[error("the link's report queue was full")]
    FrameQueueFull,
    /// A watch was asked of a link with no session standing.
    /// [`Link::ended`] gives the cause.
    #[error("the link has ended")]
    LinkEnded,
    /// A link was dialed again while its session stood, or a dial found
    /// its session let go ahead of its start. The next session took that
    /// one's place, as it does at the controller ([`Link::redial`]).
    #[error("the link was dialed again")]
    Redialed,
    /// The decide stream has no credit left: the gateway decides this
    /// request itself. Nothing was sent, and the stream stands.
    #[error("the decide stream has no credit left: the gateway decides this request itself")]
    NoCredit,
    /// The decide stream answered an ask with no usable decision. The
    /// reference client drops its stream on a mismatched id or missing
    /// answer. A handle that cannot read its decision gives back its leases.
    #[error("the decide stream answered an ask with no answer to it: {0}")]
    NoDecision(String),
    /// The blocking form could not build its runtime.
    #[error("no runtime for the blocking client: {0}")]
    Runtime(#[source] std::io::Error),
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Status(Box::new(status))
    }
}

/// States the gateway on every call: its one entry replaces any the request
/// carries under that key.
#[derive(Clone, Debug)]
struct Principal(AsciiMetadataValue);

impl Principal {
    /// Checks only that an entry can carry the id: the controller is the
    /// judge of a name.
    fn of(gateway_id: &str) -> Result<Self, Error> {
        gateway_id
            .parse()
            .map(Self)
            .map_err(|_| Error::GatewayId(gateway_id.to_owned()))
    }
}

impl Interceptor for Principal {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert(PRINCIPAL_ENTRY, self.0.clone());
        Ok(request)
    }
}

/// The gateway's session, open from its welcome until the client is
/// dropped or the controller ends the stream. After the welcome its answers
/// are read by [`Client::session_end`] and by a report that finds the
/// stream ended, and by nothing else: a controller that sends more
/// meanwhile fills the stream's window and waits there.
#[derive(Debug)]
struct Session {
    /// The queue toward the controller. Its drop ends the gateway's side of
    /// the stream.
    frames: mpsc::Sender<GatewayFrame>,
    /// The controller's answers. Its drop cancels the stream.
    answers: Streaming<ControllerFrame>,
    /// The number of the last frame queued: the hello's is 1.
    seq: u64,
    /// The status the stream ended with, once read. The stream tells it
    /// once; the session tells it at each ask.
    ended: Option<Status>,
    welcome: Welcome,
}

impl Session {
    /// Opens the stream with the hello as its first frame and takes the
    /// welcome: the call's answer and its first frame must come within
    /// `wait`. Returns the session and the term its welcome came under. The
    /// first frame takes the floor's check before its kind is read
    /// ([`read_under`]).
    async fn open(
        stub: &mut Stub,
        gateway_id: &str,
        floor: u64,
        wait: Duration,
    ) -> Result<(Self, u64), Error> {
        let (frames, queued) = mpsc::channel(SESSION_QUEUE);
        // This client takes no pushed document: its hello states no
        // generation.
        frames
            .send(hello(gateway_id, floor, 0))
            .await
            .map_err(|_| Error::NoWelcome("the session's queue is closed".to_owned()))?;
        let (answers, welcome, term) =
            Self::welcomed(stub, ReceiverStream::new(queued), floor, wait).await?;
        Ok((
            Self {
                frames,
                answers,
                seq: 1,
                ended: None,
                welcome,
            },
            term,
        ))
    }

    /// Opens a session on a stream whose first frame is its hello.
    async fn welcomed(
        stub: &mut Stub,
        frames: impl Stream<Item = GatewayFrame> + Send + 'static,
        floor: u64,
        wait: Duration,
    ) -> Result<(Streaming<ControllerFrame>, Welcome, u64), Error> {
        let opened = async {
            let mut answers = stub.session(frames).await?.into_inner();
            let first = answers.message().await?;
            Ok::<_, Status>((answers, first))
        };
        let (answers, first) = answered("the session's welcome", wait, opened).await??;
        match first.map(|frame| read_under(floor, frame)).transpose()? {
            Some(ControllerFrame {
                term,
                frame: Some(controller_frame::Frame::Welcome(welcome)),
                ..
            }) => Ok((answers, welcome, term)),
            Some(ControllerFrame {
                frame: Some(controller_frame::Frame::NotLeader(not_leader)),
                ..
            }) => Err(Error::NotLeader(not_leader.leader_address)),
            other => Err(Error::NoWelcome(format!("{other:?}"))),
        }
    }

    /// Queues one frame after the hello, numbered next and stating `floor`,
    /// within `wait`: a queue the peer stopped draining is a wait for the
    /// peer. A stream that ended tells the status it ended with, and once
    /// that status was read no frame is numbered or queued.
    async fn send(
        &mut self,
        floor: &mut u64,
        frame: gateway_frame::Frame,
        wait: Duration,
    ) -> Result<(), Error> {
        if let Some(status) = &self.ended {
            return Err(status.clone().into());
        }
        let frame = numbered(self.seq + 1, *floor, frame);
        match answered("the session's queue", wait, self.frames.send(frame)).await? {
            Ok(()) => {
                self.seq += 1;
                Ok(())
            }
            Err(_) => Err(self.end(floor, wait).await?.into()),
        }
    }

    /// The stream's end, awaited within `wait`: the status the controller
    /// ended it with, an `Ok` one for a clean end. A frame that comes first
    /// takes the floor's check ([`read_under`]), raises `floor` by its term
    /// and is read no further. The status is kept once read: the stream
    /// tells it once, and each later ask reads it again.
    async fn end(&mut self, floor: &mut u64, wait: Duration) -> Result<Status, Error> {
        if let Some(status) = &self.ended {
            return Ok(status.clone());
        }
        let ended = async {
            loop {
                match self.answers.message().await {
                    Ok(Some(frame)) => {
                        let term = read_under(*floor, frame)?.term;
                        *floor = (*floor).max(term);
                    }
                    Ok(None) => return Ok(Status::ok("the session's stream ended")),
                    Err(status) => return Ok::<_, Error>(status),
                }
            }
        };
        let status = answered("the session's end", wait, ended).await??;
        self.ended = Some(status.clone());
        Ok(status)
    }
}

/// The floor's check, at the one place every frame the session reads
/// passes: a frame under a term below `floor` is refused whatever it
/// carries, as the contract has a gateway reject it.
fn read_under(floor: u64, frame: ControllerFrame) -> Result<ControllerFrame, Error> {
    if frame.term < floor {
        return Err(Error::BelowFloor {
            term: frame.term,
            floor,
        });
    }
    Ok(frame)
}

/// A frame after the hello: numbered `seq`, stating `floor` as its term
/// floor.
fn numbered(seq: u64, floor: u64, frame: gateway_frame::Frame) -> GatewayFrame {
    GatewayFrame {
        seq,
        term_floor: floor,
        frame: Some(frame),
    }
}

/// The session's first frame: the hello of `gateway_id`, numbered 1, which
/// states `floor` as the term it saw and as its term floor. `generation`
/// is the controller's generation whose document the gateway's own engine
/// runs, and 0 where it runs none of the controller's.
fn hello(gateway_id: &str, floor: u64, generation: u64) -> GatewayFrame {
    GatewayFrame {
        seq: 1,
        term_floor: floor,
        frame: Some(gateway_frame::Frame::Hello(Hello {
            gateway_id: gateway_id.to_owned(),
            protocol_min: PROTOCOL_VERSION,
            protocol_max: PROTOCOL_VERSION,
            term_seen: floor,
            generation,
            version: env!("CARGO_PKG_VERSION").to_owned(),
            ..Hello::default()
        })),
    }
}

/// `step` under the client's bound: a wait for the peer that outlasts
/// `wait` is [`Error::NoAnswer`] and names `what`.
async fn answered<T>(
    what: &'static str,
    wait: Duration,
    step: impl Future<Output = T>,
) -> Result<T, Error> {
    tokio::time::timeout(wait, step)
        .await
        .map_err(|_| Error::NoAnswer { what, wait })
}

/// The `Leader` ask, whole: its send, its answer and the answer's end.
async fn ask_leader(stub: &mut Stub, wait: Duration) -> Result<LeaderInfo, Error> {
    let request = LeaderRequest {
        model: String::new(),
    };
    Ok(answered("the Leader ask", wait, stub.leader(request))
        .await??
        .into_inner())
}

/// Dials as `gateway_id` and takes the floor from the peer's leader answer.
async fn dial_leader(
    dial: impl Future<Output = Result<Channel, tonic::transport::Error>>,
    gateway_id: &str,
    wait: Duration,
) -> Result<(Stub, u64), Error> {
    let principal = Principal::of(gateway_id)?;
    let channel = answered("the dial", wait, dial)
        .await?
        .map_err(Error::Connect)?;
    let mut stub = GrandCentralClient::with_interceptor(channel, principal);
    let floor = ask_leader(&mut stub, wait).await?.term;
    Ok((stub, floor))
}

/// What the controller answered one ask.
#[derive(Clone, Debug, PartialEq)]
#[expect(
    clippy::exhaustive_enums,
    reason = "a caller answers each case, and a new case must stop its build"
)]
pub enum Answer {
    /// The controller's decision as the wire says it: a pick, a shed or a
    /// queue, with the leases it booked and its trace. One with no outcome
    /// is a deadline miss. Beside a pick of a request that stated encode
    /// items (`DecideRequest.encode`) comes where its items go
    /// (`DecideAnswer.encode`), and `None` beside every other decision.
    Decision(Decision, Option<EncodePlaced>),
    /// The controller decided nothing, for the reason it names: the gateway
    /// decides this request itself.
    Busy(Busy),
}

/// The client's decide stream between its asks. It holds nothing a new
/// stream must find: its credit starts again at the welcome's window.
#[derive(Debug)]
struct Decide {
    /// The queue toward the controller, of one ask. Its drop ends the
    /// gateway's side of the stream.
    requests: mpsc::Sender<DecideRequest>,
    /// The controller's answers. Its drop cancels the stream.
    answers: Streaming<DecideAnswer>,
    /// The asks the stream may still send: an ask spends one, a `Credit`
    /// frame adds its window, and the controller takes none back.
    credit: u32,
}

impl Decide {
    /// Opens the stream with `window` asks of credit; the call's answer
    /// must come within `wait`. A controller refuses the open where no
    /// session of the gateway stands.
    async fn open(stub: &mut Stub, window: u32, wait: Duration) -> Result<Self, Error> {
        let (requests, queued) = mpsc::channel(1);
        let answers = Self::opened(stub, ReceiverStream::new(queued), wait).await?;
        Ok(Self {
            requests,
            answers,
            credit: window,
        })
    }

    /// Opens the decide stream with the caller's request stream.
    async fn opened(
        stub: &mut Stub,
        requests: impl Stream<Item = DecideRequest> + Send + 'static,
        wait: Duration,
    ) -> Result<Streaming<DecideAnswer>, Error> {
        Ok(
            answered("the decide stream's open", wait, stub.decide(requests))
                .await??
                .into_inner(),
        )
    }

    /// Queues `request` within `wait` and spends one credit. With none
    /// left, the frames the stream already holds are read first, with no
    /// wait ([`Decide::held`]); with none still, the ask is
    /// [`Error::NoCredit`] and nothing is queued. A queue the stream's end
    /// closed tells the status the stream ended with.
    async fn send(
        &mut self,
        floor: &mut u64,
        request: DecideRequest,
        wait: Duration,
    ) -> Result<(), Error> {
        if self.credit == 0 {
            self.held(floor).await?;
        }
        if self.credit == 0 {
            return Err(Error::NoCredit);
        }
        let queued = self.requests.send(request);
        match answered("the decide stream's queue", wait, queued).await? {
            Ok(()) => {
                self.credit -= 1;
                Ok(())
            }
            Err(_) => Err(self.ended(floor, wait).await),
        }
    }

    /// Reads the frames the stream holds at this instant, each polled for
    /// once and never waited for: credit that has not come is none. A
    /// frame that carries an answer answers no ask, since none stands.
    async fn held(&mut self, floor: &mut u64) -> Result<(), Error> {
        loop {
            let polled = {
                let mut next = pin!(self.answers.message());
                poll_fn(|context| Poll::Ready(next.as_mut().poll(context))).await
            };
            let frame = match polled {
                Poll::Ready(Ok(Some(frame))) => frame,
                Poll::Ready(Ok(None)) => return Err(clean_end()),
                Poll::Ready(Err(status)) => return Err(status.into()),
                Poll::Pending => return Ok(()),
            };
            if let Some(unasked) = self.read(floor, frame)? {
                return Err(Error::NoDecision(format!("{unasked:?}")));
            }
        }
    }

    /// One frame of the stream under the floor's check, which the frame
    /// then raises: a `Credit` adds its window and is read no further
    /// (`None`); any other frame is handed back.
    fn read(
        &mut self,
        floor: &mut u64,
        frame: DecideAnswer,
    ) -> Result<Option<DecideAnswer>, Error> {
        if frame.term < *floor {
            return Err(Error::BelowFloor {
                term: frame.term,
                floor: *floor,
            });
        }
        *floor = frame.term;
        if let Some(decide_answer::Answer::Credit(credit)) = &frame.answer {
            self.credit = self.credit.saturating_add(credit.window);
            return Ok(None);
        }
        Ok(Some(frame))
    }

    /// The answer to the ask `id`, within `wait`: frames are read until
    /// one carries an answer, a `Credit` on the way adding its window. The
    /// id rides back on an answer as its request's key: an answer that
    /// names another id, or carries no answer this client reads, is
    /// [`Error::NoDecision`]. A stream that ends first tells its status.
    async fn answer(
        &mut self,
        floor: &mut u64,
        id: &[u8],
        wait: Duration,
    ) -> Result<Answer, Error> {
        let read = async {
            loop {
                let Some(frame) = self.answers.message().await? else {
                    return Err(clean_end());
                };
                let Some(frame) = self.read(floor, frame)? else {
                    continue;
                };
                if frame.request_id != id {
                    return Err(Error::NoDecision(format!("{frame:?}")));
                }
                return match frame.answer {
                    Some(decide_answer::Answer::Decision(decision)) => {
                        Ok(Answer::Decision(decision, frame.encode))
                    }
                    Some(decide_answer::Answer::Busy(busy)) => Ok(Answer::Busy(busy)),
                    // A `Credit` was read above: what is left is no kind
                    // of this client's protocol.
                    _ => Err(Error::NoDecision(format!(
                        "an answer of no kind this client reads, under the term {}",
                        frame.term
                    ))),
                };
            }
        };
        answered("the decision", wait, read).await?
    }

    /// Why a stream whose queue closed ended, awaited within `wait`: its
    /// status, as an error. A frame that comes first takes the floor's
    /// check and is read no further.
    async fn ended(&mut self, floor: &mut u64, wait: Duration) -> Error {
        let ended = async {
            loop {
                match self.answers.message().await {
                    Ok(Some(frame)) => {
                        self.read(floor, frame)?;
                    }
                    Ok(None) => return Ok(clean_end()),
                    Err(status) => return Ok::<_, Error>(status.into()),
                }
            }
        };
        match answered("the decide stream's end", wait, ended).await {
            Ok(Ok(error) | Err(error)) | Err(error) => error,
        }
    }
}

/// A decide stream's clean end, as the status an ask is told.
fn clean_end() -> Error {
    Status::ok("the decide stream ended").into()
}

/// The call's two bounds, held at the client's own door so that no frame
/// of the client ends a stream by them: the id's range, then the frame's
/// size as encoded. A frame over the size loses whole block hashes at its
/// tail until it fits, and the count cut is returned: a controller reads
/// a list from its head. No other field is dropped for size.
fn fit(request: &mut DecideRequest) -> Result<usize, Error> {
    let id = request.request_id.len();
    if !(1..=MAX_REQUEST_ID).contains(&id) {
        return Err(Error::RequestId(id));
    }
    let size = request.encoded_len();
    if size <= MAX_FRAME {
        return Ok(0);
    }
    let held = request
        .facts
        .as_ref()
        .map_or(0, |facts| facts.block_hashes.len());
    // A cut takes its hashes' bytes off the frame, and off the lengths
    // written before the list less than one hash more: one hash fewer than
    // the bytes over ask for is the least that can fit.
    let mut kept = held.saturating_sub((size - MAX_FRAME).div_ceil(HASH) - 1);
    loop {
        if let Some(facts) = &mut request.facts {
            facts.block_hashes.truncate(kept);
        }
        if request.encoded_len() <= MAX_FRAME {
            return Ok(held - kept);
        }
        kept = kept.checked_sub(1).ok_or(Error::TooLarge(size))?;
    }
}

/// A watch of the controller's workers, open on either client's channel
/// ([`Client::watch_workers`], [`Link::watch_workers`]).
#[derive(Debug)]
pub struct Watch {
    events: Streaming<WorkerEvent>,
}

impl Watch {
    async fn open(
        stub: &mut Stub,
        request: WatchWorkersRequest,
        wait: Duration,
    ) -> Result<Self, Error> {
        let response = answered("the watch's open", wait, stub.watch_workers(request)).await??;
        Ok(Self {
            events: response.into_inner(),
        })
    }

    /// The watch's next event, or none where the controller ended the
    /// stream with no error: it ends a stream whose watcher took nothing
    /// for a while after an event, and the watcher opens the watch again
    /// after the last event it took. A stream ended with a status tells it
    /// ([`Error::Status`]): `ABORTED` where its replica stopped leading.
    /// This wait holds no bound of the client's: a fleet that does not
    /// change tells nothing. A peer gone silent ends it through the
    /// channel's keepalive ([`KEEPALIVE`]). A caller that brings its own
    /// endpoint to [`Client::connect_with`] sets one on it.
    pub async fn next(&mut self) -> Result<Option<WorkerEvent>, Error> {
        Ok(self.events.message().await?)
    }
}

/// One gateway's side of the protocol over one channel.
#[derive(Debug)]
pub struct Client {
    gateway_id: String,
    stub: Stub,
    /// The bound on each wait for the peer.
    wait: Duration,
    session: Session,
    /// The decide stream between asks: none before the first ask, during
    /// one, and after one that ended as [`Error::BelowFloor`],
    /// [`Error::NoDecision`], [`Error::Status`] or [`Error::NoAnswer`]. An
    /// ask the door refuses (an id outside its range, a frame with no hash
    /// left to cut, no credit) leaves it where it stood.
    decide: Option<Decide>,
    /// The highest term an answer named.
    floor: u64,
    /// The block hashes cut from the asks sent so far.
    cut: usize,
}

impl Client {
    /// Dials `address`, a URI such as `http://127.0.0.1:30100`, in the
    /// clear, as the gateway `gateway_id`: [`Client::connect_with`] over a
    /// plain endpoint.
    pub async fn connect(address: &str, gateway_id: &str) -> Result<Self, Error> {
        let endpoint = Endpoint::from_shared(address.to_owned())
            .map_err(|error| Error::Address(error.to_string()))?
            .http2_keep_alive_interval(KEEPALIVE)
            .keep_alive_timeout(ANSWER_WAIT)
            .keep_alive_while_idle(true);
        Self::connect_with(endpoint, gateway_id).await
    }

    /// Dials an endpoint the caller configured (its TLS material and its
    /// time-outs, or none) as the gateway `gateway_id`, takes its term floor
    /// from the replica's `Leader` answer and opens the gateway's session.
    /// An id no metadata entry carries is refused before the dial; a replica
    /// that does not lead is [`Error::NotLeader`]; a first frame of the
    /// session below the floor is [`Error::BelowFloor`]. No wait for the
    /// replica outlasts [`ANSWER_WAIT`]: a dial, a `Leader` ask or a
    /// welcome that would is [`Error::NoAnswer`]. One that a time-out of
    /// the endpoint's own ends sooner is the channel's error:
    /// [`Error::Status`], or [`Error::Connect`] for the dial. The three
    /// wait one after another, so a connect whole can last three bounds.
    pub async fn connect_with(endpoint: Endpoint, gateway_id: &str) -> Result<Self, Error> {
        Self::connect_within(endpoint, gateway_id, ANSWER_WAIT).await
    }

    /// [`Client::connect_with`] with the bound on each wait as `wait`.
    async fn connect_within(
        endpoint: Endpoint,
        gateway_id: &str,
        wait: Duration,
    ) -> Result<Self, Error> {
        Box::pin(Self::over(endpoint.connect(), gateway_id, wait)).await
    }

    /// The client over the channel `dial` makes: the dial, the `Leader` ask
    /// and the session's open, each under `wait`.
    async fn over(
        dial: impl Future<Output = Result<Channel, tonic::transport::Error>>,
        gateway_id: &str,
        wait: Duration,
    ) -> Result<Self, Error> {
        let (mut stub, floor) = dial_leader(dial, gateway_id, wait).await?;
        let (session, term) = Session::open(&mut stub, gateway_id, floor, wait).await?;
        Ok(Self {
            gateway_id: gateway_id.to_owned(),
            stub,
            wait,
            session,
            decide: None,
            floor: floor.max(term),
            cut: 0,
        })
    }

    /// Asks the replica who leads, within the client's bound. An answer below
    /// the floor is [`Error::BelowFloor`]; otherwise its term raises the floor.
    pub async fn leader(&mut self) -> Result<LeaderInfo, Error> {
        let info = ask_leader(&mut self.stub, self.wait).await?;
        if info.term < self.floor {
            return Err(Error::BelowFloor {
                term: info.term,
                floor: self.floor,
            });
        }
        self.floor = info.term;
        Ok(info)
    }

    /// Reports a request this gateway sent to a worker: one `Dispatched`
    /// frame on the session, numbered after the last and stating the term
    /// floor. The client decides nothing and keeps no lease: the report is
    /// its caller's. The frame is queued within the client's bound and
    /// leaves when the connection sends it: a queue with no room in time
    /// is [`Error::NoAnswer`]. Once the session's end was read, by
    /// [`Client::session_end`] or by a report that found the queue closed,
    /// each report tells the status it ended with ([`Error::Status`]) and
    /// queues nothing: an `Ok` one after a clean end.
    pub async fn dispatched(&mut self, report: Dispatched) -> Result<(), Error> {
        self.report(gateway_frame::Frame::Dispatched(report)).await
    }

    /// Reports how a request ended on a worker: one `Completed` frame, sent
    /// as [`Client::dispatched`] sends its own.
    pub async fn completed(&mut self, report: Completed) -> Result<(), Error> {
        self.report(gateway_frame::Frame::Completed(report)).await
    }

    /// Gives a lease or a ticket up: one `Released` frame, sent as
    /// [`Client::dispatched`] sends its own.
    pub async fn released(&mut self, report: Released) -> Result<(), Error> {
        self.report(gateway_frame::Frame::Released(report)).await
    }

    async fn report(&mut self, frame: gateway_frame::Frame) -> Result<(), Error> {
        self.session.send(&mut self.floor, frame, self.wait).await
    }

    /// Asks the controller which worker one request goes to: one
    /// `DecideRequest` on the client's decide stream, answered by the
    /// controller's decision or by a `Busy`, for the gateway to decide
    /// itself. One ask at a time: its answer is read before the call
    /// returns. The client keeps no lease and decides nothing itself.
    ///
    /// The stream opens at the first ask, as a present gateway's: the
    /// session's hello stands by then. Its credit starts at the welcome's
    /// `decide_window`; an ask spends one, and a `Credit` frame read on the
    /// way to an answer adds its window. An ask with none left is
    /// [`Error::NoCredit`] and is sent nowhere.
    ///
    /// The call's two bounds are held before the send. An id outside the
    /// call's range is [`Error::RequestId`]. A frame over the call's size
    /// as encoded loses block hashes at its tail until it fits, counted by
    /// [`Client::hashes_cut`] once the ask is queued; one with none left
    /// to cut is [`Error::TooLarge`].
    ///
    /// An answer under a term below the floor is [`Error::BelowFloor`];
    /// one that is no answer to the ask is [`Error::NoDecision`]; a stream
    /// that ended tells its status ([`Error::Status`]); an open, a queue,
    /// an answer or a stream's end that outlasts the client's bound is
    /// [`Error::NoAnswer`]. These waits run one after another, so an ask
    /// whole can last three bounds. Each of these errors drops the stream,
    /// as does a call dropped before its end, and the next ask opens a new
    /// one at the welcome's window: an answer that comes late is never
    /// read. An ask given up may still be decided, and the lease it booked
    /// stands until it expires.
    pub async fn decide(&mut self, mut request: DecideRequest) -> Result<Answer, Error> {
        let cut = fit(&mut request)?;
        let wait = self.wait;
        // The stream is the ask's own while it is asked: an ask that ends
        // in an error, or is dropped, puts none back.
        let mut stream = match self.decide.take() {
            Some(stream) => stream,
            None => {
                let window = self.session.welcome.decide_window;
                Decide::open(&mut self.stub, window, wait).await?
            }
        };
        let id = request.request_id.clone();
        if let Err(refused) = stream.send(&mut self.floor, request, wait).await {
            if matches!(refused, Error::NoCredit) {
                self.decide = Some(stream);
            }
            return Err(refused);
        }
        self.cut += cut;
        let answer = stream.answer(&mut self.floor, &id, wait).await?;
        self.decide = Some(stream);
        Ok(answer)
    }

    /// Opens a watch of the controller's workers as `request` asks: a
    /// snapshot first or the tail after a place, of every model or of the
    /// ones it names. The stream is the watch's own: it shares the client's
    /// channel and its stated id, and neither the session nor an ask waits
    /// for it. An event's term is the one its place was written under, so
    /// it is not held to the client's floor: a watcher hands the place back
    /// at its resume, and the controller tells it to start again where the
    /// place is another history's. An open that outlasts the client's bound
    /// is [`Error::NoAnswer`]; one the controller refuses tells its status
    /// ([`Error::Status`]).
    pub async fn watch_workers(&mut self, request: WatchWorkersRequest) -> Result<Watch, Error> {
        Watch::open(&mut self.stub, request, self.wait).await
    }

    /// The session's end, awaited within the client's bound: the status the
    /// controller ended the stream with, the same at each ask once it is
    /// read. A frame that comes first raises the floor by its term and is
    /// read no further; one under a term below the floor is
    /// [`Error::BelowFloor`]. A session still open at the bound is
    /// [`Error::NoAnswer`].
    pub async fn session_end(&mut self) -> Result<Status, Error> {
        self.session.end(&mut self.floor, self.wait).await
    }

    /// The welcome that answered the session's hello.
    pub fn welcome(&self) -> &Welcome {
        &self.session.welcome
    }

    /// The block hashes the client cut from the tails of the asks it sent,
    /// each cut to fit its ask in a frame ([`Client::decide`]).
    pub fn hashes_cut(&self) -> usize {
        self.cut
    }

    /// The gateway this client speaks as.
    pub fn gateway_id(&self) -> &str {
        &self.gateway_id
    }

    /// The term floor: the highest term an answer named so far.
    pub fn term_floor(&self) -> u64 {
        self.floor
    }
}

/// The client for a caller that runs no runtime.
pub mod blocking {
    use std::{future::Future, io};

    use tokio::runtime::{Builder, Runtime};
    use tonic::{transport::Endpoint, Status};

    use super::{Answer, Error};
    use crate::service::proto::{
        Completed, DecideRequest, Dispatched, LeaderInfo, Released, Welcome,
    };

    /// [`super::Client`] driven to each call's end on a runtime of its own.
    /// It holds no protocol: every call is the async client's. The runtime
    /// keeps one thread, which drives the connection between calls: a
    /// report leaves without waiting for the next call. The runtime is this
    /// client's alone, so a call must come from a thread that runs no
    /// runtime. The drop waits for nothing the runtime still runs: a report
    /// made just before it may never leave.
    #[derive(Debug)]
    pub struct Client {
        // Declared before the runtime: its drop comes first, on a runtime
        // that still runs. No case holds the order.
        inner: super::Client,
        runtime: Background,
    }

    impl Client {
        /// [`super::Client::connect`].
        pub fn connect(address: &str, gateway_id: &str) -> Result<Self, Error> {
            Self::driven(super::Client::connect(address, gateway_id))
        }

        /// [`super::Client::connect_with`].
        pub fn connect_with(endpoint: Endpoint, gateway_id: &str) -> Result<Self, Error> {
            Self::driven(super::Client::connect_with(endpoint, gateway_id))
        }

        /// Drives `connect` to its end on a runtime of the client's own. A
        /// connect that fails drops that runtime, and the drop waits for
        /// nothing: a dial's name lookup runs on a thread no bound ends,
        /// and a drop that waited for it would hold the caller past the
        /// bound.
        pub(super) fn driven(
            connect: impl Future<Output = Result<super::Client, Error>>,
        ) -> Result<Self, Error> {
            let runtime = Background::start()?;
            let inner = runtime.drive(connect)?;
            Ok(Self { inner, runtime })
        }

        /// [`super::Client::leader`].
        pub fn leader(&mut self) -> Result<LeaderInfo, Error> {
            self.runtime.drive(self.inner.leader())
        }

        /// [`super::Client::dispatched`].
        pub fn dispatched(&mut self, report: Dispatched) -> Result<(), Error> {
            self.runtime.drive(self.inner.dispatched(report))
        }

        /// [`super::Client::completed`].
        pub fn completed(&mut self, report: Completed) -> Result<(), Error> {
            self.runtime.drive(self.inner.completed(report))
        }

        /// [`super::Client::released`].
        pub fn released(&mut self, report: Released) -> Result<(), Error> {
            self.runtime.drive(self.inner.released(report))
        }

        /// [`super::Client::session_end`].
        pub fn session_end(&mut self) -> Result<Status, Error> {
            self.runtime.drive(self.inner.session_end())
        }

        /// [`super::Client::decide`].
        pub fn decide(&mut self, request: DecideRequest) -> Result<Answer, Error> {
            self.runtime.drive(self.inner.decide(request))
        }

        /// [`super::Client::watch_workers`].
        pub fn watch_workers(
            &mut self,
            request: super::WatchWorkersRequest,
        ) -> Result<super::Watch, Error> {
            self.runtime.drive(self.inner.watch_workers(request))
        }

        /// [`super::Watch::next`] of a watch this client opened, on the
        /// client's runtime.
        pub fn next_of(
            &mut self,
            watch: &mut super::Watch,
        ) -> Result<Option<super::WorkerEvent>, Error> {
            self.runtime.drive(watch.next())
        }

        /// [`super::Client::welcome`].
        pub fn welcome(&self) -> &Welcome {
            self.inner.welcome()
        }

        /// [`super::Client::hashes_cut`].
        pub fn hashes_cut(&self) -> usize {
            self.inner.hashes_cut()
        }

        /// [`super::Client::gateway_id`].
        pub fn gateway_id(&self) -> &str {
            self.inner.gateway_id()
        }

        /// [`super::Client::term_floor`].
        pub fn term_floor(&self) -> u64 {
            self.inner.term_floor()
        }
    }

    /// The blocking client's runtime: one thread, which drives the
    /// connection between calls. Its drop shuts the runtime down in the
    /// background and waits for nothing: a name lookup still running on it
    /// would hold a drop that waited for as long as the lookup lasts.
    #[derive(Debug)]
    pub(super) struct Background(Option<Runtime>);

    impl Background {
        pub(super) fn start() -> Result<Self, Error> {
            Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .map(|runtime| Self(Some(runtime)))
                .map_err(Error::Runtime)
        }

        /// Drives `call` to its end on the runtime.
        pub(super) fn drive<T>(
            &self,
            call: impl Future<Output = Result<T, Error>>,
        ) -> Result<T, Error> {
            match &self.0 {
                Some(runtime) => runtime.block_on(call),
                // The runtime leaves at the drop alone, and no call follows
                // a drop.
                None => Err(Error::Runtime(io::Error::other("the runtime is shut down"))),
            }
        }
    }

    impl Drop for Background {
        fn drop(&mut self) {
            if let Some(runtime) = self.0.take() {
                runtime.shutdown_background();
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/service/client/tests.rs"]
pub(crate) mod tests;
