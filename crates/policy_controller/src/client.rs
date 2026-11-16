//! The reference client of the controller's protocol: one gateway's side of
//! it, for the crate's own cases and for examples (the `client` feature; no
//! shipped binary builds it).
//!
//! It dials a controller's gateway listener, asks who leads and opens the
//! gateway's session: a `Hello`, answered by the leader's `Welcome`. The term
//! of every answer raises its term floor. What a gateway's own host adds is
//! not here: a fallback engine, the spec push and its acknowledgement, and
//! reconnection.
//!
//! The client takes its channel and makes no credential:
//! [`Client::connect`] dials an address in the clear, and
//! [`Client::connect_with`] an endpoint the caller configured with its own
//! TLS material. Every call states the gateway's id in one metadata entry
//! (`x-smg-principal`); a stated id is no credential. [`blocking::Client`]
//! drives the same calls for a caller that runs no runtime.
//!
//! No wait of the client for its peer outlasts [`ANSWER_WAIT`], on any
//! endpoint: the dial, each call from its send to its answer's end, and the
//! session's call with its first frame. An endpoint's own time-outs end a
//! wait sooner.

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
        ControllerFrame, GatewayFrame, Hello, LeaderInfo, LeaderRequest, Welcome,
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
    /// The peer did not answer inside the client's bound. Every wait of the
    /// client that runs out ends as this kind and names the wait: the dial,
    /// the `Leader` ask or the session's welcome.
    #[error("no answer within {wait:?}: {what}")]
    NoAnswer {
        /// The wait that ran out.
        what: &'static str,
        /// The bound it ran out under.
        wait: Duration,
    },
    /// The controller answered the call with a status.
    #[error("the controller answered: {0}")]
    Status(#[source] Box<Status>),
    /// The replica dialed does not lead. The address is the leader's as the
    /// replica names it, empty when it knows none.
    #[error("the replica does not lead; it names the leader at {0:?}")]
    NotLeader(String),
    /// The session ended, or answered the hello with a frame that is no
    /// welcome.
    #[error("the session answered the hello with no welcome: {0}")]
    NoWelcome(String),
    /// The welcome came under a term below the client's floor. A gateway
    /// rejects a frame below its floor, so the session is not taken.
    #[error("the welcome's term {term} is below the floor {floor}")]
    BelowFloor {
        /// The term the welcome came under.
        term: u64,
        /// The floor the hello stated.
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
/// dropped. This form reads no frame after the welcome: a controller that
/// sends more fills the stream's window and waits there.
#[derive(Debug)]
struct Session {
    #[expect(
        dead_code,
        reason = "held: its drop ends the gateway's side of the stream"
    )]
    frames: mpsc::Sender<GatewayFrame>,
    #[expect(dead_code, reason = "held: its drop cancels the stream")]
    answers: Streaming<ControllerFrame>,
    welcome: Welcome,
}

impl Session {
    /// Opens the stream with the hello as its first frame and takes the
    /// welcome: the call's answer and its first frame must come within
    /// `wait`. Returns the session and the term its welcome came under. A
    /// welcome under a term below `floor` is refused: the contract has a
    /// gateway reject a frame below its floor.
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
        match first {
            Some(ControllerFrame {
                term,
                frame: Some(controller_frame::Frame::Welcome(welcome)),
                ..
            }) if term >= floor => Ok((
                Self {
                    frames,
                    answers,
                    welcome,
                },
                term,
            )),
            Some(ControllerFrame {
                term,
                frame: Some(controller_frame::Frame::Welcome(_)),
                ..
            }) => Err(Error::BelowFloor { term, floor }),
            Some(ControllerFrame {
                frame: Some(controller_frame::Frame::NotLeader(not_leader)),
                ..
            }) => Err(Error::NotLeader(not_leader.leader_address)),
            other => Err(Error::NoWelcome(format!("{other:?}"))),
        }
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
    /// that does not lead is [`Error::NotLeader`]; a welcome under a term
    /// below the floor is [`Error::BelowFloor`]. No wait for the replica
    /// outlasts [`ANSWER_WAIT`]: a dial, a `Leader` ask or a welcome that
    /// would is [`Error::NoAnswer`].
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
    use std::future::Future;

    use tokio::runtime::{Builder, Runtime};
    use tonic::transport::Endpoint;

    use super::Error;
    use crate::proto::{LeaderInfo, Welcome};

    /// [`super::Client`] driven to each call's end on a runtime of its own.
    /// It holds no protocol: every call is the async client's. The runtime
    /// is this client's alone, so a call, and the drop, must come from a
    /// thread that runs no runtime. The drop of a connected client waits
    /// for a name lookup that runtime still runs.
    #[derive(Debug)]
    pub struct Client {
        runtime: Runtime,
        inner: super::Client,
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
        /// connect that fails shuts that runtime down without waiting for
        /// it: a dial's name lookup runs on a thread no bound ends, and a
        /// drop that waited for it would hold the caller past the bound.
        pub(super) fn driven(
            connect: impl Future<Output = Result<super::Client, Error>>,
        ) -> Result<Self, Error> {
            let runtime = runtime()?;
            match runtime.block_on(connect) {
                Ok(inner) => Ok(Self { runtime, inner }),
                Err(error) => {
                    runtime.shutdown_background();
                    Err(error)
                }
            }
        }

        /// [`super::Client::leader`].
        pub fn leader(&mut self) -> Result<LeaderInfo, Error> {
            self.runtime.block_on(self.inner.leader())
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

    fn runtime() -> Result<Runtime, Error> {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(Error::Runtime)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        net::{Ipv4Addr, SocketAddr},
        sync::mpsc,
        time::Duration,
    };

    use axum::{
        body::{Body, Bytes},
        response::Response,
        Router,
    };
    use prost::Message as _;
    use tokio::{net::TcpListener, task::JoinSet, time::timeout};
    use tonic::{
        codegen::tokio_stream::{once, pending, StreamExt},
        service::{Interceptor, Routes},
        transport::{server::TcpIncoming, Endpoint, Server},
        Request,
    };

    use super::{
        blocking, controller_frame, gateway_frame, hello, Client, ControllerFrame, Error,
        PolicyControllerClient, Principal, Session, Stub, Welcome, ANSWER_WAIT, PRINCIPAL_ENTRY,
        PROTOCOL_VERSION,
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
        let frame = ControllerFrame {
            term,
            frame: Some(controller_frame::Frame::Welcome(Welcome::default())),
            ..ControllerFrame::default()
        };
        // A message on the wire: its flag, its length, its bytes.
        let length = u32::try_from(frame.encoded_len()).expect("a short frame");
        let mut message = vec![0];
        message.extend(length.to_be_bytes());
        message.extend(frame.encode_to_vec());
        let message = once(Ok::<_, Infallible>(Bytes::from(message)));
        Body::from_stream(message.chain(pending()))
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
}
