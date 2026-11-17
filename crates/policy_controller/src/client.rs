//! The reference client of the controller's protocol: one gateway's side of
//! it, for the crate's own cases and for examples (the `client` feature; no
//! shipped binary builds it).
//!
//! It dials a controller's gateway listener, asks who leads and opens the
//! gateway's session: a `Hello`, answered by the leader's `Welcome`. On that
//! session it reports a request's life as its caller tells it: a
//! `Dispatched`, a `Completed`, a `Released`. The term of every answer raises
//! its term floor, and a frame of the session under a term below the floor
//! is refused. What a gateway's own host adds is not here: a fallback
//! engine, the decision a lease comes from, the spec push and its
//! acknowledgement, and reconnection.
//!
//! The client takes its channel and makes no credential:
//! [`Client::connect`] dials an address in the clear, and
//! [`Client::connect_with`] an endpoint the caller configured with its own
//! TLS material. Every call states the gateway's id in one metadata entry
//! (`x-smg-principal`); a stated id is no credential. [`blocking::Client`]
//! drives the same calls for a caller that runs no runtime.
//!
//! No wait of the client for its peer outlasts [`ANSWER_WAIT`], on any
//! endpoint: the dial, each call from its send to its answer's end, the
//! session's call with its first frame, a report's wait for room in the
//! session's queue and the wait for the session's end. An endpoint's own
//! time-outs end a wait sooner, and as the channel's error, not as the
//! client's [`Error::NoAnswer`]: [`Error::Status`] for a call and
//! [`Error::Connect`] for the dial.

use std::{future::Future, time::Duration};

use tokio::sync::mpsc;
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream,
    metadata::AsciiMetadataValue,
    service::{interceptor::InterceptedService, Interceptor},
    transport::{Channel, Endpoint},
    Request, Status, Streaming,
};

use crate::{
    controller::{PRINCIPAL_ENTRY, PROTOCOL_VERSION},
    proto::{
        controller_frame, gateway_frame, policy_controller_client::PolicyControllerClient,
        Completed, ControllerFrame, Dispatched, GatewayFrame, Hello, LeaderInfo, LeaderRequest,
        Released, Welcome,
    },
};

/// Frames the client may queue toward the controller.
const SESSION_QUEUE: usize = 16;

/// The bound on each wait of the client for its peer: an answer a leader
/// under load can give, and a wait a blocking caller can afford. The client
/// lays it over the dial and over each call whole, since an endpoint's own
/// time-outs end at a call's response headers and a peer may stall after
/// them.
pub const ANSWER_WAIT: Duration = Duration::from_secs(5);

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
    /// ask, the session's welcome, the session's queue or the session's
    /// end. A time-out the caller set on its own endpoint is not this
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
    /// A frame of the session came under a term below the client's floor.
    /// A gateway rejects a frame below its floor whatever it carries. At
    /// the open the session is not taken, and the address a `NotLeader`
    /// names is not read; after the welcome the frame is read no further.
    #[error("a frame's term {term} is below the floor {floor}")]
    BelowFloor {
        /// The term the frame came under.
        term: u64,
        /// The floor the frame was read under.
        floor: u64,
    },
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
        frames
            .send(hello(gateway_id, floor))
            .await
            .map_err(|_| Error::NoWelcome("the session's queue is closed".to_owned()))?;
        let opened = async {
            let mut answers = stub
                .session(ReceiverStream::new(queued))
                .await?
                .into_inner();
            let first = answers.message().await?;
            Ok::<_, Status>((answers, first))
        };
        let (answers, first) = answered("the session's welcome", wait, opened).await??;
        match first.map(|frame| read_under(floor, frame)).transpose()? {
            Some(ControllerFrame {
                term,
                frame: Some(controller_frame::Frame::Welcome(welcome)),
                ..
            }) => Ok((
                Self {
                    frames,
                    answers,
                    seq: 1,
                    ended: None,
                    welcome,
                },
                term,
            )),
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
/// states `floor` as the term it saw and as its term floor.
fn hello(gateway_id: &str, floor: u64) -> GatewayFrame {
    GatewayFrame {
        seq: 1,
        term_floor: floor,
        frame: Some(gateway_frame::Frame::Hello(Hello {
            gateway_id: gateway_id.to_owned(),
            protocol_min: PROTOCOL_VERSION,
            protocol_max: PROTOCOL_VERSION,
            term_seen: floor,
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

/// One gateway's side of the protocol over one channel.
#[derive(Debug)]
pub struct Client {
    gateway_id: String,
    stub: Stub,
    /// The bound on each wait for the peer.
    wait: Duration,
    session: Session,
    /// The highest term an answer named.
    floor: u64,
}

impl Client {
    /// Dials `address`, a URI such as `http://127.0.0.1:30100`, in the
    /// clear, as the gateway `gateway_id`: [`Client::connect_with`] over a
    /// plain endpoint.
    pub async fn connect(address: &str, gateway_id: &str) -> Result<Self, Error> {
        let endpoint = Endpoint::from_shared(address.to_owned())
            .map_err(|error| Error::Address(error.to_string()))?;
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
        Self::over(endpoint.connect(), gateway_id, wait).await
    }

    /// The client over the channel `dial` makes: the dial, the `Leader` ask
    /// and the session's open, each under `wait`.
    async fn over(
        dial: impl Future<Output = Result<Channel, tonic::transport::Error>>,
        gateway_id: &str,
        wait: Duration,
    ) -> Result<Self, Error> {
        let principal = Principal::of(gateway_id)?;
        let channel = answered("the dial", wait, dial)
            .await?
            .map_err(Error::Connect)?;
        let mut stub = PolicyControllerClient::with_interceptor(channel, principal);
        let floor = ask_leader(&mut stub, wait).await?.term;
        let (session, term) = Session::open(&mut stub, gateway_id, floor, wait).await?;
        Ok(Self {
            gateway_id: gateway_id.to_owned(),
            stub,
            wait,
            session,
            floor: floor.max(term),
        })
    }

    /// Asks the replica who leads, within the client's bound. The answer's
    /// term raises the floor.
    pub async fn leader(&mut self) -> Result<LeaderInfo, Error> {
        let info = ask_leader(&mut self.stub, self.wait).await?;
        self.floor = self.floor.max(info.term);
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

    use super::Error;
    use crate::proto::{Completed, Dispatched, LeaderInfo, Released, Welcome};

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

        /// [`super::Client::welcome`].
        pub fn welcome(&self) -> &Welcome {
            self.inner.welcome()
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
        task::JoinSet,
        time::timeout,
    };
    use tonic::{
        codegen::tokio_stream::{iter, once, pending, Stream, StreamExt},
        server::{Grpc, StreamingService},
        service::{Interceptor, Routes},
        transport::{server::TcpIncoming, Endpoint, Server},
        Code, Request, Status, Streaming,
    };

    use super::{
        blocking, controller_frame, gateway_frame, hello, numbered, Client, ControllerFrame, Error,
        GatewayFrame, PolicyControllerClient, Principal, Session, Stub, Welcome, ANSWER_WAIT,
        PRINCIPAL_ENTRY, PROTOCOL_VERSION,
    };
    use crate::proto::{Completed, Dispatched, NotLeader, Released};

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
        let frame = hello("gateway-1", 7);
        assert_eq!((frame.seq, frame.term_floor), (1, 7), "{frame:?}");
        let hello = match frame.frame {
            Some(gateway_frame::Frame::Hello(hello)) => Some(hello),
            _ => None,
        }
        .expect("the first frame is a hello");
        assert_eq!(hello.gateway_id, "gateway-1");
        assert_eq!(hello.term_seen, 7);
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
        let (address, _serving, _kept) = keeping().await;
        let mut stub = stub(address).await;
        let (mut session, _) = Session::open(&mut stub, "gateway-1", 0, ROOM)
            .await
            .expect("the welcome opens the session");
        // The minute after the end is read and before the connection drops
        // the ended call's request: the status is kept and the queue still
        // takes frames. This peer never ends the stream, so the case keeps
        // the status itself.
        session.ended = Some(Status::invalid_argument("the frame reads wrong"));
        let mut floor = 0;
        let report = gateway_frame::Frame::Released(Released::default());
        let reported = session.send(&mut floor, report, ROOM).await;
        assert!(
            matches!(&reported, Err(Error::Status(status)) if status.code() == Code::InvalidArgument),
            "a report after a kept end tells its status and queues nothing: {reported:?}"
        );
        assert_eq!(session.seq, 1, "a refused report takes no number");
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
}
