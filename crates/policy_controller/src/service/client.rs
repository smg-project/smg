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
        controller_frame, decide_answer, gateway_frame,
        policy_controller_client::PolicyControllerClient, Busy, Completed, ControllerFrame,
        DecideAnswer, DecideRequest, Decision, Dispatched, GatewayFrame, Hello, LeaderInfo,
        LeaderRequest, Released, WatchWorkersRequest, Welcome, WorkerEvent,
    },
};

mod link;
pub(crate) mod wire;

#[cfg(test)]
pub(crate) use link::tests::{poll, Peer};
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

type Stub = PolicyControllerClient<InterceptedService<Channel, Principal>>;

/// Why a call of the client failed.
#[derive(Debug, thiserror::Error)]
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
    let mut stub = PolicyControllerClient::with_interceptor(channel, principal);
    let floor = ask_leader(&mut stub, wait).await?.term;
    Ok((stub, floor))
}

/// What the controller answered one ask.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// The controller's decision as the wire says it: a pick, a shed or a
    /// queue, with the leases it booked and its trace. One with no outcome
    /// is a deadline miss.
    Decision(Decision),
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
                        Ok(Answer::Decision(decision))
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
mod tests {
    use std::{
        convert::Infallible,
        future::{ready, Ready},
        iter::repeat,
        net::{Ipv4Addr, SocketAddr},
        pin::Pin,
        sync::mpsc,
        time::Duration,
    };

    use axum::{
        body::{Body, Bytes},
        response::Response,
        Router,
    };
    use prost::Message as _;
    use tokio::{
        net::TcpListener,
        sync::mpsc::{unbounded_channel, UnboundedReceiver},
        task::{yield_now, JoinSet},
        time::{sleep, timeout},
    };
    use tonic::{
        codegen::tokio_stream::{iter, once, pending, Stream, StreamExt},
        server::{Grpc, StreamingService, UnaryService},
        service::{Interceptor, Routes},
        transport::{server::TcpIncoming, Endpoint, Server},
        Code, Request, Status, Streaming,
    };

    use super::{
        blocking, controller_frame, decide_answer, fit, gateway_frame, hello, numbered, Answer,
        Busy, Client, ControllerFrame, Decide, DecideAnswer, DecideRequest, Decision, Error,
        GatewayFrame, PolicyControllerClient, Principal, Session, Stub, Welcome, ANSWER_WAIT,
        MAX_FRAME, MAX_REQUEST_ID, PRINCIPAL_ENTRY, PROTOCOL_VERSION,
    };
    use crate::service::proto::{
        busy, Completed, Credit, Dispatched, LeaderInfo, LeaderRequest, NotLeader, Released,
        RequestFacts,
    };

    /// The bound a case gives a wait it wants ended.
    const SHORT: Duration = Duration::from_millis(300);
    /// The room a case gives the client to end a wait by its own bound.
    const ROOM: Duration = Duration::from_secs(10);

    #[test]
    fn every_call_states_the_gateway_in_one_metadata_entry() {
        let mut principal = Principal::of("gateway-1").expect("a plain id");
        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert(PRINCIPAL_ENTRY, "another".parse().expect("a value"));
        let request = principal.call(request).expect("the entry is stated");
        let stated: Vec<&str> = request
            .metadata()
            .get_all(PRINCIPAL_ENTRY)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .collect();
        assert_eq!(stated, ["gateway-1"], "one entry, the client's own");
    }

    #[test]
    fn an_id_no_metadata_entry_carries_is_refused_before_the_dial() {
        // Nothing listens at the address: a dial would fail by another name.
        let refused = blocking::Client::connect("http://127.0.0.1:9", "gate\nway");
        assert!(
            matches!(&refused, Err(Error::GatewayId(id)) if id == "gate\nway"),
            "{refused:?}"
        );
    }

    #[test]
    fn the_hello_is_the_first_frame_and_states_the_floor_twice() {
        let frame = hello("gateway-1", 7, 3);
        assert_eq!((frame.seq, frame.term_floor), (1, 7), "{frame:?}");
        let hello = match frame.frame {
            Some(gateway_frame::Frame::Hello(hello)) => Some(hello),
            _ => None,
        }
        .expect("the first frame is a hello");
        assert_eq!(hello.gateway_id, "gateway-1");
        assert_eq!(hello.term_seen, 7);
        assert_eq!(hello.generation, 3);
        assert_eq!(
            (hello.protocol_min, hello.protocol_max),
            (PROTOCOL_VERSION, PROTOCOL_VERSION)
        );
    }

    /// A peer that answers each call with gRPC headers and `body`, for as
    /// long as the case holds its task.
    async fn answering(
        body: fn() -> Body,
    ) -> (SocketAddr, JoinSet<Result<(), tonic::transport::Error>>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a listener");
        let address = listener.local_addr().expect("its address");
        let answer = Router::new().fallback(move || async move {
            Response::builder()
                .header("content-type", "application/grpc")
                .body(body())
                .expect("a response")
        });
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(Routes::from(answer))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        (address, serving)
    }

    /// An answer of headers and no frame.
    fn headers_alone() -> Body {
        Body::from_stream(pending::<Result<Bytes, Infallible>>())
    }

    /// An answer of headers and one message of no bytes (its flag and its
    /// length: five zeros) that never ends.
    fn one_message() -> Body {
        let message = once(Ok::<_, Infallible>(Bytes::from_static(&[0; 5])));
        Body::from_stream(message.chain(pending()))
    }

    /// An answer of headers and one frame, a welcome under `term`, that
    /// never ends.
    fn welcome_under(term: u64) -> Body {
        one_frame(term, controller_frame::Frame::Welcome(Welcome::default()))
    }

    /// An answer of headers and one frame, a `NotLeader` under `term` that
    /// names a leader, that never ends.
    fn not_leader_under(term: u64) -> Body {
        let named = NotLeader {
            leader_address: "http://127.0.0.1:9".to_owned(),
            ..NotLeader::default()
        };
        one_frame(term, controller_frame::Frame::NotLeader(named))
    }

    /// An answer of headers and one frame under `term` that never ends.
    fn one_frame(term: u64, frame: controller_frame::Frame) -> Body {
        frames(vec![under(term, Some(frame))])
    }

    /// A frame under `term` that carries `frame`.
    fn under(term: u64, frame: Option<controller_frame::Frame>) -> ControllerFrame {
        ControllerFrame {
            term,
            frame,
            ..ControllerFrame::default()
        }
    }

    /// An answer of headers and `frames` in turn, that never ends.
    fn frames(frames: Vec<ControllerFrame>) -> Body {
        let messages = frames.into_iter().map(|frame| {
            // A message on the wire: its flag, its length, its bytes.
            let length = u32::try_from(frame.encoded_len()).expect("a short frame");
            let mut message = vec![0];
            message.extend(length.to_be_bytes());
            message.extend(frame.encode_to_vec());
            Ok::<_, Infallible>(Bytes::from(message))
        });
        Body::from_stream(iter(messages).chain(pending()))
    }

    /// A welcome and a `NotLeader`, as frames.
    fn welcome() -> controller_frame::Frame {
        controller_frame::Frame::Welcome(Welcome::default())
    }

    fn not_leader() -> controller_frame::Frame {
        controller_frame::Frame::NotLeader(NotLeader::default())
    }

    /// A peer that takes each connection and answers nothing, for as long
    /// as the case holds its task.
    async fn mute() -> (SocketAddr, JoinSet<()>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a listener");
        let address = listener.local_addr().expect("its address");
        let mut serving = JoinSet::new();
        serving.spawn(async move {
            let mut held = Vec::new();
            while let Ok((socket, _)) = listener.accept().await {
                held.push(socket);
            }
        });
        (address, serving)
    }

    /// Whether `ended` is the client's own bound of `wait` run out on `what`.
    fn ran_out<T>(ended: &Result<T, Error>, what: &str, wait: Duration) -> bool {
        matches!(
            ended,
            Err(Error::NoAnswer { what: named, wait: bound }) if *named == what && *bound == wait
        )
    }

    /// The gateway's stub over a dial of `address`.
    async fn stub(address: SocketAddr) -> Stub {
        let channel = Endpoint::from_shared(format!("http://{address}"))
            .expect("a URI")
            .connect()
            .await
            .expect("the dial");
        let principal = Principal::of("gateway-1").expect("a plain id");
        PolicyControllerClient::with_interceptor(channel, principal)
    }

    /// The session's open against a peer whose answer stalls: the client's
    /// own bound ends it and names the welcome.
    async fn stalled_open(address: SocketAddr, ends: &'static str) {
        let mut stub = stub(address).await;
        let opened = timeout(ROOM, Session::open(&mut stub, "gateway-1", 0, SHORT))
            .await
            .expect(ends);
        assert!(
            ran_out(&opened, "the session's welcome", SHORT),
            "{opened:?}"
        );
    }

    /// The client's connect against a peer whose `Leader` answer stalls: the
    /// client's own bound ends it and names the ask.
    async fn stalled_ask(address: SocketAddr, ends: &'static str) {
        let endpoint = Endpoint::from_shared(format!("http://{address}")).expect("a URI");
        let asked = timeout(ROOM, Client::connect_within(endpoint, "gateway-1", SHORT))
            .await
            .expect(ends);
        assert!(ran_out(&asked, "the Leader ask", SHORT), "{asked:?}");
    }

    /// The `Leader` call of a peer that answers each ask with `answer`.
    #[derive(Clone, Copy)]
    struct Leads(fn() -> LeaderInfo);

    impl UnaryService<LeaderRequest> for Leads {
        type Response = LeaderInfo;
        type Future = Ready<Result<tonic::Response<Self::Response>, Status>>;

        fn call(&mut self, _request: Request<LeaderRequest>) -> Self::Future {
            ready(Ok(tonic::Response::new((self.0)())))
        }
    }

    /// A peer that answers each call with `answer`, for as long as the case
    /// holds its task.
    async fn leading(
        answer: fn() -> LeaderInfo,
    ) -> (SocketAddr, JoinSet<Result<(), tonic::transport::Error>>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a listener");
        let address = listener.local_addr().expect("its address");
        let answer = Router::new().fallback(move |request: axum::extract::Request| async move {
            Grpc::new(tonic_prost::ProstCodec::<LeaderInfo, LeaderRequest>::default())
                .unary(Leads(answer), request)
                .await
                .map(Body::new)
        });
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(Routes::from(answer))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        (address, serving)
    }

    #[tokio::test]
    async fn a_leader_answer_below_the_clients_floor_is_refused() {
        let (deciding, _deciding) = deciding(|| (Vec::new(), None)).await;
        let (stream, _stub) = stream(deciding, 0).await;
        let (mut client, _welcoming) = holding(stream, ROOM).await;
        let (leading, _leading) = leading(|| LeaderInfo {
            term: 7,
            address: "old".to_owned(),
            ..LeaderInfo::default()
        })
        .await;
        client.stub = stub(leading).await;
        client.floor = 8;

        let refused = client.leader().await;

        assert!(
            matches!(refused, Err(Error::BelowFloor { term: 7, floor: 8 })),
            "a stale leader answer is refused: {refused:?}"
        );
        assert_eq!(
            client.term_floor(),
            8,
            "the floor remains at its high-water mark"
        );
    }

    #[tokio::test]
    async fn a_welcome_that_never_comes_ends_the_wait_at_its_bound() {
        assert_eq!(ANSWER_WAIT, Duration::from_secs(5));
        let (address, _serving) = answering(headers_alone).await;
        stalled_open(
            address,
            "the client's own bound ends the wait for the welcome",
        )
        .await;
    }

    #[tokio::test]
    async fn a_session_call_nothing_answers_ends_the_wait_at_its_bound() {
        let (address, _serving) = mute().await;
        stalled_open(
            address,
            "the client's own bound ends a session call nothing answers",
        )
        .await;
    }

    #[tokio::test]
    async fn a_leader_ask_nothing_answers_ends_at_its_bound() {
        let (address, _serving) = mute().await;
        stalled_ask(
            address,
            "the client's own bound ends a Leader ask nothing answers",
        )
        .await;
    }

    #[tokio::test]
    async fn a_leader_ask_whose_headers_came_alone_ends_at_its_bound() {
        let (address, _serving) = answering(headers_alone).await;
        stalled_ask(
            address,
            "the client's own bound ends a Leader ask whose headers came alone",
        )
        .await;
    }

    #[tokio::test]
    async fn a_leader_ask_whose_answer_never_ends_ends_at_its_bound() {
        let (address, _serving) = answering(one_message).await;
        stalled_ask(
            address,
            "the client's own bound ends a Leader ask whose answer never ends",
        )
        .await;
    }

    #[tokio::test]
    async fn an_endpoints_own_time_out_ends_a_leader_ask_as_a_status() {
        let (address, _serving) = mute().await;
        let endpoint = Endpoint::from_shared(format!("http://{address}"))
            .expect("a URI")
            .timeout(SHORT);
        let asked = timeout(
            ROOM / 2,
            Client::connect_within(endpoint, "gateway-1", ROOM),
        )
        .await
        .expect("the endpoint's own time-out ends a Leader ask nothing answers");
        assert!(matches!(asked, Err(Error::Status(_))), "{asked:?}");
    }

    #[tokio::test]
    async fn a_dial_that_never_ends_ends_at_its_bound() {
        let dialed = timeout(
            ROOM,
            Client::over(std::future::pending(), "gateway-1", SHORT),
        )
        .await
        .expect("the client's own bound ends a dial that never ends");
        assert!(ran_out(&dialed, "the dial", SHORT), "{dialed:?}");
    }

    #[tokio::test]
    async fn a_first_frame_that_is_no_welcome_ends_the_open_with_no_wait() {
        let (address, _serving) = answering(one_message).await;
        let mut stub = stub(address).await;
        let opened = timeout(ROOM / 2, Session::open(&mut stub, "gateway-1", 0, ROOM))
            .await
            .expect("a first frame that is no welcome ends the open with no wait for more");
        assert!(
            matches!(&opened, Err(Error::NoWelcome(why)) if why.contains("frame: None")),
            "{opened:?}"
        );
    }

    #[tokio::test]
    async fn a_welcome_under_a_term_below_the_floor_is_refused() {
        let (address, _serving) = answering(|| welcome_under(6)).await;
        let mut stub = stub(address).await;
        let opened = Session::open(&mut stub, "gateway-1", 7, ROOM).await;
        assert!(
            matches!(opened, Err(Error::BelowFloor { term: 6, floor: 7 })),
            "{opened:?}"
        );
    }

    #[tokio::test]
    async fn a_not_leader_under_a_term_below_the_floor_is_refused_unread() {
        let (address, _serving) = answering(|| not_leader_under(6)).await;
        let mut stub = stub(address).await;
        let opened = Session::open(&mut stub, "gateway-1", 7, ROOM).await;
        assert!(
            matches!(opened, Err(Error::BelowFloor { term: 6, floor: 7 })),
            "{opened:?}"
        );
    }

    #[tokio::test]
    async fn a_welcome_under_the_floors_own_term_opens_the_session() {
        let (address, _serving) = answering(|| welcome_under(7)).await;
        let mut stub = stub(address).await;
        let opened = Session::open(&mut stub, "gateway-1", 7, ROOM).await;
        assert!(matches!(opened, Ok((_, 7))), "{opened:?}");
    }

    #[tokio::test]
    async fn a_welcome_under_a_term_above_the_floor_opens_the_session() {
        let (address, _serving) = answering(|| welcome_under(8)).await;
        let mut stub = stub(address).await;
        let opened = Session::open(&mut stub, "gateway-1", 7, ROOM).await;
        assert!(matches!(opened, Ok((_, 8))), "{opened:?}");
    }

    #[test]
    fn a_blocking_connect_that_fails_does_not_wait_for_a_lookup_it_started() {
        // The dial starts a blocking task, as a connector's name lookup is
        // one, and never ends. The task runs until the case lets it go.
        let (release, held) = mpsc::channel::<()>();
        let (told, ended) = mpsc::channel();
        let connecting = std::thread::spawn(move || {
            let dial = async move {
                let _lookup = tokio::task::spawn_blocking(move || held.recv());
                std::future::pending().await
            };
            let ended = blocking::Client::driven(Client::over(dial, "gateway-1", SHORT));
            told.send(ended.map(|_| ()))
        });
        let ended = ended
            .recv_timeout(ROOM)
            .expect("a failed connect returns while its lookup still runs");
        assert!(ran_out(&ended, "the dial", SHORT), "{ended:?}");
        drop(release);
        connecting
            .join()
            .expect("the connecting thread ends")
            .expect("the case took the connect's end");
    }

    #[test]
    fn the_blocking_plain_form_ends_a_stalled_ask_at_the_named_bound() {
        // The peer's runtime is the case's own: the blocking client brings
        // its own, from a thread that runs none.
        let peers = tokio::runtime::Runtime::new().expect("a runtime");
        let (address, _serving) = peers.block_on(answering(headers_alone));
        let (told, asked) = mpsc::channel();
        std::thread::spawn(move || {
            let asked = blocking::Client::connect(&format!("http://{address}"), "gateway-1");
            told.send(asked.map(|_| ()))
        });
        let asked = asked
            .recv_timeout(ROOM)
            .expect("the named bound ends the blocking plain form's stalled ask");
        assert!(ran_out(&asked, "the Leader ask", ANSWER_WAIT), "{asked:?}");
    }

    #[test]
    fn a_report_is_numbered_as_told_and_states_the_floor() {
        let report = gateway_frame::Frame::Dispatched(Dispatched::default());
        let frame = numbered(2, 7, report);
        assert_eq!((frame.seq, frame.term_floor), (2, 7), "{frame:?}");
        assert!(
            matches!(frame.frame, Some(gateway_frame::Frame::Dispatched(_))),
            "{frame:?}"
        );
    }

    #[tokio::test]
    async fn a_first_frame_that_carries_nothing_below_the_floor_is_refused() {
        let (address, _serving) = answering(|| frames(vec![under(6, None)])).await;
        let mut stub = stub(address).await;
        let opened = Session::open(&mut stub, "gateway-1", 7, ROOM).await;
        assert!(
            matches!(opened, Err(Error::BelowFloor { term: 6, floor: 7 })),
            "{opened:?}"
        );
    }

    #[tokio::test]
    async fn a_frame_after_the_welcome_below_the_floor_is_refused() {
        let (address, _serving) = answering(|| {
            frames(vec![
                under(7, Some(welcome())),
                under(6, Some(not_leader())),
            ])
        })
        .await;
        let mut stub = stub(address).await;
        let (mut session, _) = Session::open(&mut stub, "gateway-1", 7, ROOM)
            .await
            .expect("the welcome opens the session");
        let mut floor = 7;
        let ended = session.end(&mut floor, ROOM).await;
        assert!(
            matches!(ended, Err(Error::BelowFloor { term: 6, floor: 7 })),
            "{ended:?}"
        );
    }

    #[tokio::test]
    async fn a_frame_after_the_welcome_raises_the_floor_and_an_open_session_ends_the_wait() {
        let (address, _serving) = answering(|| {
            frames(vec![
                under(7, Some(welcome())),
                under(8, Some(not_leader())),
            ])
        })
        .await;
        let mut stub = stub(address).await;
        let (mut session, _) = Session::open(&mut stub, "gateway-1", 7, ROOM)
            .await
            .expect("the welcome opens the session");
        let mut floor = 7;
        let ended = timeout(ROOM, session.end(&mut floor, SHORT))
            .await
            .expect("the client's own bound ends the wait for the session's end");
        assert!(ran_out(&ended, "the session's end", SHORT), "{ended:?}");
        assert_eq!(floor, 8, "the frame's term raised the floor");
    }

    #[tokio::test]
    async fn a_report_into_a_session_whose_peer_stopped_reading_ends_at_its_bound() {
        let (address, _serving) = answering(|| welcome_under(0)).await;
        let mut stub = stub(address).await;
        let (mut session, _) = Session::open(&mut stub, "gateway-1", 0, ROOM)
            .await
            .expect("the welcome opens the session");
        let mut floor = 0;
        // The peer reads no frame: the stream's window fills, then the
        // session's queue, and a report waits for room.
        let waited: Result<(), Error> = timeout(ROOM * 2, async {
            loop {
                let report = gateway_frame::Frame::Completed(Completed {
                    lease_id: u64::MAX,
                    served: u64::MAX,
                    ttft_micros: u32::MAX,
                    tokens_in: u32::MAX,
                    tokens_out: u32::MAX,
                    duration_micros: u32::MAX,
                    ..Completed::default()
                });
                session.send(&mut floor, report, SHORT).await?;
            }
        })
        .await
        .expect("a peer that reads nothing leaves a report waiting for room");
        assert!(ran_out(&waited, "the session's queue", SHORT), "{waited:?}");
    }

    /// The session call of a peer that answers a welcome and then ends the
    /// stream as `end` says: clean, or with the status.
    struct Ends(fn() -> Result<(), Status>);

    impl StreamingService<GatewayFrame> for Ends {
        type Response = ControllerFrame;
        type ResponseStream = Pin<Box<dyn Stream<Item = Result<ControllerFrame, Status>> + Send>>;
        type Future = Ready<Result<tonic::Response<Self::ResponseStream>, Status>>;

        fn call(&mut self, _request: Request<Streaming<GatewayFrame>>) -> Self::Future {
            let end = (self.0)().err().map(Err);
            let answers = iter([Ok(under(0, Some(welcome())))]).chain(iter(end));
            ready(Ok(tonic::Response::new(Box::pin(answers))))
        }
    }

    /// A peer that answers each call with a welcome and then ends the
    /// stream as a controller does, its status in the trailers, for as long
    /// as the case holds its task.
    async fn ending(
        end: fn() -> Result<(), Status>,
    ) -> (SocketAddr, JoinSet<Result<(), tonic::transport::Error>>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a listener");
        let address = listener.local_addr().expect("its address");
        let answer = Router::new().fallback(move |request: axum::extract::Request| async move {
            Grpc::new(tonic_prost::ProstCodec::<ControllerFrame, GatewayFrame>::default())
                .streaming(Ends(end), request)
                .await
                .map(Body::new)
        });
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(Routes::from(answer))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        (address, serving)
    }

    /// A peer that answers each call with a welcome and keeps the frames
    /// the call sends, for as long as the case holds its task.
    async fn keeping() -> (
        SocketAddr,
        JoinSet<Result<(), tonic::transport::Error>>,
        UnboundedReceiver<GatewayFrame>,
    ) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a listener");
        let address = listener.local_addr().expect("its address");
        let (keep, kept) = unbounded_channel();
        let (call, mut calls) = unbounded_channel::<Body>();
        let answer = Router::new().fallback(move |request: axum::extract::Request| {
            // The call's frames are read by the peer's own task.
            let _ = call.send(request.into_body());
            async {
                Response::builder()
                    .header("content-type", "application/grpc")
                    .body(welcome_under(0))
                    .expect("a response")
            }
        });
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(Routes::from(answer))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        serving.spawn(async move {
            while let Some(sent) = calls.recv().await {
                let mut sent = sent.into_data_stream();
                let mut bytes = Vec::new();
                while let Some(Ok(chunk)) = sent.next().await {
                    bytes.extend(chunk);
                    // A message on the wire: its flag, its length, its bytes.
                    while let Some(length) = bytes.get(1..5) {
                        let length = u32::from_be_bytes(length.try_into().expect("four bytes"));
                        let end = 5 + usize::try_from(length).expect("a short frame");
                        let Some(message) = bytes.get(5..end) else {
                            break;
                        };
                        let frame = GatewayFrame::decode(message).expect("a gateway's frame");
                        let _ = keep.send(frame);
                        bytes.drain(..end);
                    }
                }
            }
            Ok(())
        });
        (address, serving, kept)
    }

    #[tokio::test]
    async fn an_ended_sessions_status_is_told_again_at_each_ask() {
        let (address, _serving) =
            ending(|| Err(Status::invalid_argument("the frame reads wrong"))).await;
        let mut stub = stub(address).await;
        let (mut session, _) = Session::open(&mut stub, "gateway-1", 0, ROOM)
            .await
            .expect("the welcome opens the session");
        let mut floor = 0;
        for ask in ["the first ask", "a second ask"] {
            let ended = session
                .end(&mut floor, ROOM)
                .await
                .expect("the stream's end is read");
            assert_eq!(
                ended.code(),
                Code::InvalidArgument,
                "{ask} reads the status the stream ended with: {ended:?}"
            );
        }
        // A report made straight after tells the same status and takes no
        // number, whether or not the connection has dropped the ended
        // call's request yet.
        let report = gateway_frame::Frame::Released(Released::default());
        let reported = session.send(&mut floor, report, ROOM).await;
        assert!(
            matches!(&reported, Err(Error::Status(status)) if status.code() == Code::InvalidArgument),
            "a report into the ended session tells the status it ended with: {reported:?}"
        );
        assert_eq!(session.seq, 1, "a refused report takes no number");
    }

    #[tokio::test]
    async fn a_clean_end_is_told_as_an_ok_status_at_each_ask() {
        let (address, _serving) = ending(|| Ok(())).await;
        let mut stub = stub(address).await;
        let (mut session, _) = Session::open(&mut stub, "gateway-1", 0, ROOM)
            .await
            .expect("the welcome opens the session");
        let mut floor = 0;
        for ask in ["the first ask", "a second ask"] {
            let ended = session
                .end(&mut floor, ROOM)
                .await
                .expect("the stream's end is read");
            assert_eq!(ended.code(), Code::Ok, "{ask} reads a clean end: {ended:?}");
        }
        let report = gateway_frame::Frame::Released(Released::default());
        let reported = session.send(&mut floor, report, ROOM).await;
        assert!(
            matches!(&reported, Err(Error::Status(status)) if status.code() == Code::Ok),
            "a report into a cleanly ended session tells its Ok status: {reported:?}"
        );
    }

    #[tokio::test]
    async fn each_report_is_numbered_after_the_last_and_states_the_floor_of_its_minute() {
        let (address, _serving, mut kept) = keeping().await;
        let mut stub = stub(address).await;
        let (mut session, _) = Session::open(&mut stub, "gateway-1", 0, ROOM)
            .await
            .expect("the welcome opens the session");
        for mut floor in [0, 0, 9] {
            let report = gateway_frame::Frame::Dispatched(Dispatched::default());
            session
                .send(&mut floor, report, ROOM)
                .await
                .expect("the report is queued");
        }
        let mut read = Vec::new();
        while read.len() < 4 {
            let frame = timeout(ROOM, kept.recv())
                .await
                .expect("the peer reads the session's frames")
                .expect("the peer keeps each frame");
            read.push((frame.seq, frame.term_floor));
        }
        assert_eq!(
            read,
            [(1, 0), (2, 0), (3, 0), (4, 9)],
            "the hello, then each report numbered after the last and stating the floor of its minute"
        );
    }

    #[tokio::test]
    async fn a_report_after_a_kept_end_is_refused_while_the_queue_still_takes_frames() {
        let (address, _serving, mut kept) = keeping().await;
        let mut stub = stub(address).await;
        let (mut session, _) = Session::open(&mut stub, "gateway-1", 0, ROOM)
            .await
            .expect("the welcome opens the session");
        // The minute after the end is read and before the connection drops
        // the ended call's request: the status is kept and the queue still
        // takes frames. This peer never ends the stream, so the case keeps
        // the status itself: an error's, then a clean end's.
        let mut floor = 0;
        let ends = [
            (
                Status::invalid_argument("the frame reads wrong"),
                Code::InvalidArgument,
            ),
            (Status::ok("the session's stream ended"), Code::Ok),
        ];
        for (ended, code) in ends {
            session.ended = Some(ended);
            let report = gateway_frame::Frame::Released(Released::default());
            let reported = session.send(&mut floor, report, ROOM).await;
            assert!(
                matches!(&reported, Err(Error::Status(status)) if status.code() == code),
                "a report after a kept end tells its status and queues nothing: {reported:?}"
            );
            assert_eq!(session.seq, 1, "a refused report takes no number");
        }
        // The peer's own read: with the end taken back, the next report is
        // the first frame to reach it after the hello.
        session.ended = None;
        let report = gateway_frame::Frame::Dispatched(Dispatched::default());
        session
            .send(&mut floor, report, ROOM)
            .await
            .expect("the report is queued");
        let mut read = Vec::new();
        for _ in 0..2 {
            let frame = timeout(ROOM, kept.recv())
                .await
                .expect("the peer reads a frame")
                .expect("the peer keeps each frame");
            read.push((frame.seq, frame.frame));
        }
        assert!(
            matches!(
                read.as_slice(),
                [
                    (1, Some(gateway_frame::Frame::Hello(_))),
                    (2, Some(gateway_frame::Frame::Dispatched(_)))
                ]
            ),
            "no refused report reached the peer: {read:?}"
        );
    }

    #[test]
    fn the_blocking_runtimes_drop_does_not_wait_for_a_task_it_still_runs() {
        // A blocking task stands for a name lookup a connector started: it
        // runs until the case lets it go.
        let (release, held) = mpsc::channel::<()>();
        let (told, dropped) = mpsc::channel();
        let dropping = std::thread::spawn(move || {
            let runtime = blocking::Background::start()?;
            runtime.drive(async move {
                drop(tokio::task::spawn_blocking(move || held.recv()));
                Ok(())
            })?;
            drop(runtime);
            told.send(())
                .map_err(|_| Error::NoWelcome("the case left".to_owned()))
        });
        dropped
            .recv_timeout(ROOM)
            .expect("the runtime's drop returns while its task still runs");
        drop(release);
        dropping
            .join()
            .expect("the dropping thread ends")
            .expect("the case took the drop's end");
    }

    /// How a scripted decide stream goes on after its frames: it stays
    /// open (`None`), ends clean, or ends with the status.
    type End = Option<Result<(), Status>>;

    /// A decide stream's script: its frames, then its end.
    type Script = fn() -> (Vec<DecideAnswer>, End);

    /// The decide call of a peer that answers its script's frames in turn,
    /// whatever the call sends, and goes on as the script's end says. With
    /// a drip it goes on instead with a credit of one ask each drip, for as
    /// long as the call stands.
    #[derive(Clone, Copy)]
    struct Decides(Script, Option<Duration>);

    impl StreamingService<DecideRequest> for Decides {
        type Response = DecideAnswer;
        type ResponseStream = Pin<Box<dyn Stream<Item = Result<DecideAnswer, Status>> + Send>>;
        type Future = Ready<Result<tonic::Response<Self::ResponseStream>, Status>>;

        fn call(&mut self, _request: Request<Streaming<DecideRequest>>) -> Self::Future {
            let (frames, end) = (self.0)();
            let frames = iter(frames.into_iter().map(Ok));
            let answers: Self::ResponseStream = match (self.1, end) {
                (Some(drip), _) => {
                    let credits = iter(repeat(())).then(move |()| async move {
                        sleep(drip).await;
                        Ok::<_, Status>(told(0, b"", credit(1)))
                    });
                    Box::pin(frames.chain(credits))
                }
                (None, None) => Box::pin(frames.chain(pending())),
                (None, Some(end)) => Box::pin(frames.chain(iter(end.err().map(Err)))),
            };
            ready(Ok(tonic::Response::new(answers)))
        }
    }

    /// A peer that answers each call as `script` says, for as long as the
    /// case holds its task.
    async fn deciding(
        script: Script,
    ) -> (SocketAddr, JoinSet<Result<(), tonic::transport::Error>>) {
        peer(Decides(script, None)).await
    }

    /// A peer that answers each call as [`Decides`] does, for as long as
    /// the case holds its task.
    async fn peer(decides: Decides) -> (SocketAddr, JoinSet<Result<(), tonic::transport::Error>>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a listener");
        let address = listener.local_addr().expect("its address");
        let answer = Router::new().fallback(move |request: axum::extract::Request| async move {
            Grpc::new(tonic_prost::ProstCodec::<DecideAnswer, DecideRequest>::default())
                .streaming(decides, request)
                .await
                .map(Body::new)
        });
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(Routes::from(answer))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        (address, serving)
    }

    /// An answer frame under `term` that names the ask `id`.
    fn told(
        term: u64,
        id: &[u8],
        answer: impl Into<Option<decide_answer::Answer>>,
    ) -> DecideAnswer {
        DecideAnswer {
            request_id: id.to_vec(),
            term,
            answer: answer.into(),
        }
    }

    /// A decision that states nothing, and a credit of `window`, as answers.
    fn decision() -> decide_answer::Answer {
        decide_answer::Answer::Decision(Decision::default())
    }

    fn credit(window: u32) -> decide_answer::Answer {
        decide_answer::Answer::Credit(Credit { window })
    }

    /// An ask named `id` that states nothing else.
    fn ask(id: &[u8]) -> DecideRequest {
        DecideRequest {
            request_id: id.to_vec(),
            ..DecideRequest::default()
        }
    }

    /// A decide stream with `window` asks of credit over a dial of
    /// `address`, and the stub that holds its channel.
    async fn stream(address: SocketAddr, window: u32) -> (Decide, Stub) {
        let mut stub = stub(address).await;
        let stream = Decide::open(&mut stub, window, ROOM)
            .await
            .expect("the stream opens");
        (stream, stub)
    }

    /// One ask on `stream`, queued and answered as the client's call does.
    async fn asked(stream: &mut Decide, floor: &mut u64, id: &[u8]) -> Result<Answer, Error> {
        stream.send(floor, ask(id), ROOM).await?;
        stream.answer(floor, id, ROOM).await
    }

    #[tokio::test]
    async fn a_credit_on_the_way_to_an_answer_adds_its_window() {
        let (address, _serving) = deciding(|| {
            (
                vec![told(1, b"", credit(5)), told(1, b"a", decision())],
                None,
            )
        })
        .await;
        let (mut stream, _stub) = stream(address, 3).await;
        let mut floor = 0;
        stream
            .send(&mut floor, ask(b"a"), ROOM)
            .await
            .expect("the ask is queued");
        assert_eq!(
            stream.credit, 2,
            "the ask spent one of the stream's credits"
        );
        let answer = stream.answer(&mut floor, b"a", ROOM).await;
        assert!(
            matches!(&answer, Ok(Answer::Decision(told)) if *told == Decision::default()),
            "{answer:?}"
        );
        assert_eq!(
            stream.credit, 7,
            "the credit frame added its window to the credit the stream held"
        );
        assert_eq!(floor, 1, "a frame's term raises the floor");
    }

    #[tokio::test]
    async fn an_ask_with_no_credit_is_refused_at_the_door_and_queues_nothing() {
        let (address, _serving) = deciding(|| (Vec::new(), None)).await;
        let (mut stream, _stub) = stream(address, 0).await;
        let refused = stream.send(&mut 0, ask(b"a"), ROOM).await;
        assert!(
            matches!(refused, Err(Error::NoCredit)),
            "an ask with no credit is refused at the door: {refused:?}"
        );
        assert_eq!(
            stream.requests.capacity(),
            1,
            "the queue's one place is free"
        );
    }

    #[tokio::test]
    async fn credit_the_stream_already_holds_is_read_at_the_door() {
        let (address, _serving) = deciding(|| (vec![told(0, b"", credit(2))], None)).await;
        let (mut stream, _stub) = stream(address, 0).await;
        let mut floor = 0;
        // The frame is on its way: an ask is refused until it has come, and
        // the first ask after that reads it with no wait.
        let queued = timeout(ROOM, async {
            loop {
                match stream.send(&mut floor, ask(b"a"), SHORT).await {
                    Err(Error::NoCredit) => yield_now().await,
                    queued => break queued,
                }
            }
        })
        .await
        .expect("the credit frame comes and the door reads it");
        queued.expect("the ask is queued on the credit the frame brought");
        assert_eq!(stream.credit, 1, "two came and the ask spent one");
    }

    #[tokio::test]
    async fn an_answer_no_ask_stands_for_is_no_decision_at_the_door() {
        let (address, _serving) = deciding(|| (vec![told(0, b"a", decision())], None)).await;
        let (mut stream, _stub) = stream(address, 0).await;
        let read = timeout(ROOM, async {
            loop {
                match stream.send(&mut 0, ask(b"b"), SHORT).await {
                    Err(Error::NoCredit) => yield_now().await,
                    read => break read,
                }
            }
        })
        .await
        .expect("the answer comes and the door reads it");
        assert!(
            matches!(read, Err(Error::NoDecision(_))),
            "an answer no ask stands for is no decision: {read:?}"
        );
    }

    #[tokio::test]
    async fn an_answer_or_a_credit_under_a_term_below_the_floor_is_refused() {
        let scripts: [Script; 2] = [
            || (vec![told(3, b"a", decision())], None),
            || (vec![told(3, b"", credit(9))], None),
        ];
        for script in scripts {
            let (address, _serving) = deciding(script).await;
            let (mut stream, _stub) = stream(address, 1).await;
            let mut floor = 5;
            let refused = asked(&mut stream, &mut floor, b"a").await;
            assert!(
                matches!(refused, Err(Error::BelowFloor { term: 3, floor: 5 })),
                "a frame below the floor is refused: {refused:?}"
            );
            assert_eq!(
                (floor, stream.credit),
                (5, 0),
                "a refused frame moves neither the floor nor the credit"
            );
        }
    }

    #[tokio::test]
    async fn a_busy_is_handed_out_as_it_came_and_its_term_raises_the_floor() {
        let (address, _serving) = deciding(|| {
            let busy = Busy {
                reason: busy::Reason::NotLeader.into(),
                retry_after_micros: 7,
            };
            let busy = decide_answer::Answer::Busy(busy);
            (vec![told(9, b"a", busy)], None)
        })
        .await;
        let (mut stream, _stub) = stream(address, 1).await;
        let mut floor = 5;
        let answer = asked(&mut stream, &mut floor, b"a").await;
        assert!(
            matches!(&answer, Ok(Answer::Busy(busy))
                if busy.reason() == busy::Reason::NotLeader && busy.retry_after_micros == 7),
            "{answer:?}"
        );
        assert_eq!(floor, 9);
    }

    #[tokio::test]
    async fn an_answer_that_is_not_the_asks_own_is_no_decision() {
        let scripts: [Script; 2] = [
            || (vec![told(1, b"b", decision())], None),
            || (vec![told(1, b"a", None::<decide_answer::Answer>)], None),
        ];
        for script in scripts {
            let (address, _serving) = deciding(script).await;
            let (mut stream, _stub) = stream(address, 1).await;
            let answer = asked(&mut stream, &mut 0, b"a").await;
            assert!(
                matches!(answer, Err(Error::NoDecision(_))),
                "an answer that is not the ask's own is no decision: {answer:?}"
            );
        }
    }

    #[tokio::test]
    async fn an_ask_on_a_stream_that_ended_tells_the_status_it_ended_with() {
        let scripts: [(Script, Code); 2] = [
            (
                || {
                    let closed = Status::unavailable("the listeners closed");
                    (Vec::new(), Some(Err(closed)))
                },
                Code::Unavailable,
            ),
            (|| (Vec::new(), Some(Ok(()))), Code::Ok),
        ];
        for (script, code) in scripts {
            let (address, _serving) = deciding(script).await;
            let (mut stream, _stub) = stream(address, 1).await;
            // The ask's send tells the end, or its answer does: whether the
            // connection has dropped the ended call's queue yet is its own.
            let answer = asked(&mut stream, &mut 0, b"a").await;
            assert!(
                matches!(&answer, Err(Error::Status(status)) if status.code() == code),
                "an ask on an ended stream tells its status: {answer:?}"
            );
        }
    }

    #[tokio::test]
    async fn an_ask_into_a_queue_the_streams_end_closed_tells_that_end_and_spends_nothing() {
        let (address, _serving) = deciding(|| {
            let closed = Status::unavailable("the listeners closed");
            (Vec::new(), Some(Err(closed)))
        })
        .await;
        let (mut stream, _stub) = stream(address, 1).await;
        timeout(ROOM, async {
            while !stream.requests.is_closed() {
                yield_now().await;
            }
        })
        .await
        .expect("the connection drops an ended call's queue");
        let refused = stream.send(&mut 0, ask(b"a"), ROOM).await;
        assert!(
            matches!(&refused, Err(Error::Status(status)) if status.code() == Code::Unavailable),
            "an ask into a closed queue tells the stream's end: {refused:?}"
        );
        assert_eq!(
            stream.credit, 1,
            "an ask the closed queue refused spent nothing"
        );
    }

    #[tokio::test]
    async fn an_ask_no_answer_comes_for_ends_at_the_clients_bound() {
        let (address, _serving) = deciding(|| (Vec::new(), None)).await;
        let (mut stream, _stub) = stream(address, 1).await;
        let mut floor = 0;
        stream
            .send(&mut floor, ask(b"a"), ROOM)
            .await
            .expect("the ask is queued");
        let waited = timeout(ROOM, stream.answer(&mut floor, b"a", SHORT))
            .await
            .expect("the client's own bound ends the wait for a decision");
        assert!(ran_out(&waited, "the decision", SHORT), "{waited:?}");
    }

    #[tokio::test]
    async fn a_decide_call_nothing_answers_ends_its_open_at_the_bound() {
        let (address, _serving) = mute().await;
        let mut stub = stub(address).await;
        let opened = timeout(ROOM, Decide::open(&mut stub, 1, SHORT))
            .await
            .expect("the client's own bound ends an open nothing answers");
        assert!(
            ran_out(&opened, "the decide stream's open", SHORT),
            "{opened:?}"
        );
    }

    #[tokio::test]
    async fn an_ask_into_a_stream_whose_peer_stopped_reading_ends_at_its_bound() {
        let (address, _serving) = answering(headers_alone).await;
        let (mut stream, _stub) = stream(address, u32::MAX).await;
        // The peer reads no frame: the stream's window fills, then the
        // queue's one place, and an ask waits for room.
        let waited: Result<(), Error> = timeout(ROOM * 2, async {
            loop {
                let ask = sized(1, 4_000, MAX_FRAME);
                stream.send(&mut 0, ask, SHORT).await?;
            }
        })
        .await
        .expect("a peer that reads nothing leaves an ask waiting for room");
        assert!(
            ran_out(&waited, "the decide stream's queue", SHORT),
            "{waited:?}"
        );
    }

    /// An ask named by `id` bytes that holds `hashes` block hashes, counted
    /// from zero, and is `size` bytes as encoded: its model's name is as
    /// long as the rest.
    fn sized(id: usize, hashes: usize, size: usize) -> DecideRequest {
        let mut request = DecideRequest {
            request_id: vec![7; id],
            facts: Some(RequestFacts {
                block_hashes: (0u64..).take(hashes).collect(),
                ..RequestFacts::default()
            }),
            constraints: None,
            again: false,
            ..Default::default()
        };
        // A longer name is written with a longer length: fill, then mend.
        let mut name = size - request.encoded_len();
        for _ in 0..4 {
            let facts = request.facts.as_mut().expect("the ask states facts");
            facts.model = "m".repeat(name);
            let now = request.encoded_len();
            if now == size {
                return request;
            }
            name = name + size - now;
        }
        panic!("no ask of {hashes} hashes is {size} bytes as encoded");
    }

    /// The block hashes `request` holds.
    fn hashes(request: &DecideRequest) -> &[u64] {
        request
            .facts
            .as_ref()
            .map_or(&[], |facts| &facts.block_hashes)
    }

    #[test]
    fn an_id_outside_the_calls_range_is_refused_at_the_door() {
        assert_eq!((MAX_REQUEST_ID, MAX_FRAME), (256, 65_536));
        for (id, taken) in [(0, false), (1, true), (256, true), (257, false)] {
            match fit(&mut ask(&vec![7; id])) {
                Ok(0) => assert!(taken, "an id of {id} bytes is in the range"),
                Err(Error::RequestId(told)) => assert!(!taken && told == id, "{id}: {told}"),
                other => panic!("an id of {id} bytes: {other:?}"),
            }
        }
    }

    #[test]
    fn a_frame_at_the_bound_keeps_its_hashes_and_one_past_it_loses_its_last() {
        let mut whole = sized(1, 4_000, MAX_FRAME);
        assert_eq!(fit(&mut whole).expect("a frame at the bound"), 0);
        assert_eq!(hashes(&whole).len(), 4_000);
        // The bytes over, and the hashes they cost: eight bytes a hash.
        for (over, cut) in [(1, 1), (8, 1), (9, 2), (17, 3)] {
            let mut request = sized(1, 4_000, MAX_FRAME + over);
            let fitted = fit(&mut request).expect("hashes to cut");
            assert_eq!(fitted, cut, "{over} bytes over");
            assert!(request.encoded_len() <= MAX_FRAME, "{over} bytes over");
            let head: Vec<u64> = (0u64..).take(4_000 - cut).collect();
            assert_eq!(hashes(&request), head, "the head stays and the tail goes");
            assert!(request
                .facts
                .is_some_and(|facts| facts.model.len() > 30_000));
        }
    }

    #[test]
    fn a_cut_that_shortens_the_lists_length_takes_one_hash_fewer() {
        // 2,048 hashes are 16,384 bytes, the least a length of three bytes
        // counts: with one hash fewer the length is written in two, so one
        // hash pays for nine bytes over.
        let mut request = sized(1, 2_048, MAX_FRAME + 9);
        let fitted = fit(&mut request).expect("hashes to cut");
        assert_eq!(fitted, 1, "one hash pays for nine bytes over");
        assert_eq!(request.encoded_len(), MAX_FRAME);
    }

    #[test]
    fn a_frame_past_the_bound_with_no_hash_left_is_the_callers_error() {
        // A lone hash and its list take ten bytes off a frame eleven over.
        for held in [0, 1] {
            let mut request = sized(1, held, MAX_FRAME + 11);
            let refused = fit(&mut request);
            assert!(
                matches!(refused, Err(Error::TooLarge(size)) if size == MAX_FRAME + 11),
                "{held} hashes: {refused:?}"
            );
        }
    }

    /// A client whose session a welcoming peer opened, with `stream` as its
    /// decide stream between asks and `wait` as its bound, and the
    /// welcoming peer's task.
    async fn holding(
        stream: Decide,
        wait: Duration,
    ) -> (Client, JoinSet<Result<(), tonic::transport::Error>>) {
        let (address, serving) = answering(|| welcome_under(0)).await;
        let mut stub = stub(address).await;
        let (session, _) = Session::open(&mut stub, "gateway-1", 0, ROOM)
            .await
            .expect("the welcome opens the session");
        let client = Client {
            gateway_id: "gateway-1".to_owned(),
            stub,
            wait,
            session,
            decide: Some(stream),
            floor: 0,
            cut: 0,
        };
        (client, serving)
    }

    #[tokio::test]
    async fn an_ask_the_door_refuses_keeps_the_stream_and_one_that_fails_drops_it() {
        let (address, _serving) = deciding(|| (Vec::new(), None)).await;
        let (stream, _stub) = stream(address, 0).await;
        let (mut client, _welcoming) = holding(stream, SHORT).await;

        let unnamed = client.decide(ask(b"")).await;
        assert!(matches!(unnamed, Err(Error::RequestId(0))), "{unnamed:?}");
        // One hash over the bound: the door cuts it, then finds no credit.
        let refused = client.decide(sized(1, 4_000, MAX_FRAME + 1)).await;
        assert!(matches!(refused, Err(Error::NoCredit)), "{refused:?}");
        assert_eq!(client.hashes_cut(), 0, "no cut frame was queued");
        let stream = client
            .decide
            .as_mut()
            .expect("an ask the door refused leaves the stream where it stood");
        stream.credit = 1;

        // Queued now, and the peer answers nothing: the bound ends the ask.
        let waited = timeout(ROOM, client.decide(sized(1, 4_000, MAX_FRAME + 1)))
            .await
            .expect("the client's own bound ends the ask");
        assert!(ran_out(&waited, "the decision", SHORT), "{waited:?}");
        assert_eq!(client.hashes_cut(), 1, "the queued ask's cut is counted");
        assert!(
            client.decide.is_none(),
            "an ask that failed drops the stream"
        );
    }

    #[tokio::test]
    async fn a_second_ask_rides_the_stream_the_first_left_and_its_credit() {
        // Each stream of this peer answers `a` and then `b`: on a stream of
        // its own the ask `b` would read an answer that is not its own.
        let (address, _serving) = deciding(|| {
            let answers = vec![told(0, b"a", decision()), told(0, b"b", decision())];
            (answers, None)
        })
        .await;
        let (stream, _stub) = stream(address, 2).await;
        let (mut client, _welcoming) = holding(stream, ROOM).await;

        for id in [b"a", b"b"] {
            let answer = client.decide(ask(id)).await;
            assert!(
                matches!(answer, Ok(Answer::Decision(_))),
                "a second ask rides the stream the first one left: {answer:?}"
            );
        }
        let refused = client.decide(ask(b"c")).await;
        assert!(
            matches!(refused, Err(Error::NoCredit)),
            "the credit two asks spent binds the third: {refused:?}"
        );
    }

    #[tokio::test]
    async fn an_end_the_door_finds_is_told_and_drops_the_stream() {
        let scripts: [(Script, Code); 2] = [
            (
                || {
                    let closed = Status::unavailable("the listeners closed");
                    (Vec::new(), Some(Err(closed)))
                },
                Code::Unavailable,
            ),
            (|| (Vec::new(), Some(Ok(()))), Code::Ok),
        ];
        for (script, code) in scripts {
            let (address, _serving) = deciding(script).await;
            let (stream, _stub) = stream(address, 0).await;
            let (mut client, _welcoming) = holding(stream, ROOM).await;
            // The end is on its way: an ask finds no credit until it has
            // come, and the first ask after that is told it.
            let found = timeout(ROOM, async {
                loop {
                    match client.decide(ask(b"a")).await {
                        Err(Error::NoCredit) => yield_now().await,
                        found => break found,
                    }
                }
            })
            .await
            .expect("the stream's end comes and the door reads it");
            assert!(
                matches!(&found, Err(Error::Status(status)) if status.code() == code),
                "the door tells the end it found: {found:?}"
            );
            assert!(
                client.decide.is_none(),
                "an end found at the door drops the stream"
            );
        }
    }

    #[tokio::test]
    async fn a_new_stream_opens_at_the_welcomes_window_and_its_answer_raises_the_clients_floor() {
        let (address, _serving) = deciding(|| (vec![told(4, b"a", decision())], None)).await;
        let (stream, stub) = stream(address, 0).await;
        let (mut client, _welcoming) = holding(stream, ROOM).await;
        // The next ask opens the client's own stream over the deciding
        // peer, at the window its welcome states.
        client.stub = stub;
        client.decide = None;
        client.session.welcome.decide_window = 3;

        let answer = client.decide(ask(b"a")).await;
        assert!(matches!(answer, Ok(Answer::Decision(_))), "{answer:?}");
        let credit = client.decide.as_ref().map(|stream| stream.credit);
        assert_eq!(
            credit,
            Some(2),
            "a new stream opens at the welcome's window"
        );
        assert_eq!(
            client.floor, 4,
            "an answer's term raises the client's own floor"
        );
    }

    #[tokio::test]
    async fn a_trickle_of_credit_frames_does_not_outlast_the_answers_bound() {
        // A credit frame each sixth of the bound and no answer: each frame
        // comes well inside the bound, and the wait for the answer still
        // ends at it.
        let (address, _serving) = peer(Decides(|| (Vec::new(), None), Some(SHORT / 6))).await;
        let (mut stream, _stub) = stream(address, 1).await;
        let mut floor = 0;
        stream
            .send(&mut floor, ask(b"a"), ROOM)
            .await
            .expect("the ask is queued");
        let waited = timeout(ROOM, stream.answer(&mut floor, b"a", SHORT))
            .await
            .expect("a trickle of credit frames does not outlast the answer's bound");
        assert!(ran_out(&waited, "the decision", SHORT), "{waited:?}");
    }

    #[tokio::test]
    async fn a_frame_below_the_floor_on_the_way_to_a_streams_end_is_refused() {
        let (address, _serving) = deciding(|| {
            let closed = Status::unavailable("the listeners closed");
            (vec![told(1, b"", credit(1))], Some(Err(closed)))
        })
        .await;
        let (mut stream, _stub) = stream(address, 1).await;
        timeout(ROOM, async {
            while !stream.requests.is_closed() {
                yield_now().await;
            }
        })
        .await
        .expect("the connection drops an ended call's queue");
        // The queue is closed and the stream still holds a frame under the
        // term 1: the read for the end takes the floor's check too.
        let refused = stream.send(&mut 2, ask(b"a"), ROOM).await;
        assert!(
            matches!(refused, Err(Error::BelowFloor { term: 1, floor: 2 })),
            "a frame below the floor is refused on the way to a stream's end: {refused:?}"
        );
    }
}
