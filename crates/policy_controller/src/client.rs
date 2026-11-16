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
//! No wait of [`Client::connect`] outlasts [`ANSWER_WAIT`]. With
//! [`Client::connect_with`] the dial and the calls keep the caller's
//! time-outs, and the wait for the welcome keeps that bound.

use std::time::Duration;

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

/// The bound on each wait of the client for its peer, as long as a
/// controller waits for a session's hello: the dial and each call's answer
/// on the endpoint [`Client::connect`] builds, and the session's first frame
/// on any endpoint. An endpoint's own time-outs end at a call's response
/// headers, which a controller sends before it reads the hello.
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
    /// The controller answered the call with a status.
    #[error("the controller answered: {0}")]
    Status(#[source] Box<Status>),
    /// The replica dialed does not lead. The address is the leader's as the
    /// replica names it, empty when it knows none.
    #[error("the replica does not lead; it names the leader at {0:?}")]
    NotLeader(String),
    /// The session ended, answered the hello with a frame that is no
    /// welcome, or sent no frame within its bound.
    #[error("the session answered the hello with no welcome: {0}")]
    NoWelcome(String),
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
    /// welcome, which must come within `wait`. Returns the session and the
    /// term its welcome came under.
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
        let mut answers = stub
            .session(ReceiverStream::new(queued))
            .await?
            .into_inner();
        let first = tokio::time::timeout(wait, answers.message())
            .await
            .map_err(|_| Error::NoWelcome(format!("no frame within {wait:?}")))??;
        match first {
            Some(ControllerFrame {
                term,
                frame: Some(controller_frame::Frame::Welcome(welcome)),
                ..
            }) => Ok((
                Self {
                    frames,
                    answers,
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

/// The endpoint with its dial and each of its calls bounded by `wait`.
fn bounded(endpoint: Endpoint, wait: Duration) -> Endpoint {
    endpoint.connect_timeout(wait).timeout(wait)
}

async fn ask_leader(stub: &mut Stub) -> Result<LeaderInfo, Error> {
    let request = LeaderRequest {
        model: String::new(),
    };
    Ok(stub.leader(request).await?.into_inner())
}

/// One gateway's side of the protocol over one channel.
#[derive(Debug)]
pub struct Client {
    gateway_id: String,
    stub: Stub,
    session: Session,
    /// The highest term an answer named.
    floor: u64,
}

impl Client {
    /// Dials `address`, a URI such as `http://127.0.0.1:30100`, in the
    /// clear, as the gateway `gateway_id`: [`Client::connect_with`] over a
    /// plain endpoint whose dial and calls each end within [`ANSWER_WAIT`].
    pub async fn connect(address: &str, gateway_id: &str) -> Result<Self, Error> {
        let endpoint = Endpoint::from_shared(address.to_owned())
            .map_err(|error| Error::Address(error.to_string()))?;
        Self::connect_within(bounded(endpoint, ANSWER_WAIT), gateway_id, ANSWER_WAIT).await
    }

    /// Dials an endpoint the caller configured (its TLS material and its
    /// time-outs, or none) as the gateway `gateway_id`, takes its term floor
    /// from the replica's `Leader` answer and opens the gateway's session.
    /// An id no metadata entry carries is refused before the dial; a replica
    /// that does not lead is [`Error::NotLeader`]; a welcome that does not
    /// come within [`ANSWER_WAIT`] is [`Error::NoWelcome`].
    pub async fn connect_with(endpoint: Endpoint, gateway_id: &str) -> Result<Self, Error> {
        Self::connect_within(endpoint, gateway_id, ANSWER_WAIT).await
    }

    /// [`Client::connect_with`] with the welcome's bound as `wait`.
    async fn connect_within(
        endpoint: Endpoint,
        gateway_id: &str,
        wait: Duration,
    ) -> Result<Self, Error> {
        let principal = Principal::of(gateway_id)?;
        let channel = endpoint.connect().await.map_err(Error::Connect)?;
        let mut stub = PolicyControllerClient::with_interceptor(channel, principal);
        let floor = ask_leader(&mut stub).await?.term;
        let (session, term) = Session::open(&mut stub, gateway_id, floor, wait).await?;
        Ok(Self {
            gateway_id: gateway_id.to_owned(),
            stub,
            session,
            floor: floor.max(term),
        })
    }

    /// Asks the replica who leads. The answer's term raises the floor.
    pub async fn leader(&mut self) -> Result<LeaderInfo, Error> {
        let info = ask_leader(&mut self.stub).await?;
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
    use tokio::runtime::{Builder, Runtime};
    use tonic::transport::Endpoint;

    use super::Error;
    use crate::proto::{LeaderInfo, Welcome};

    /// [`super::Client`] driven to each call's end on a runtime of its own.
    /// It holds no protocol: every call is the async client's. The runtime
    /// is this client's alone, so a call, and the drop, must come from a
    /// thread that runs no runtime.
    #[derive(Debug)]
    pub struct Client {
        runtime: Runtime,
        inner: super::Client,
    }

    impl Client {
        /// [`super::Client::connect`].
        pub fn connect(address: &str, gateway_id: &str) -> Result<Self, Error> {
            let runtime = runtime()?;
            let inner = runtime.block_on(super::Client::connect(address, gateway_id))?;
            Ok(Self { runtime, inner })
        }

        /// [`super::Client::connect_with`].
        pub fn connect_with(endpoint: Endpoint, gateway_id: &str) -> Result<Self, Error> {
            let runtime = runtime()?;
            let inner = runtime.block_on(super::Client::connect_with(endpoint, gateway_id))?;
            Ok(Self { runtime, inner })
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
        time::Duration,
    };

    use axum::{
        body::{Body, Bytes},
        response::Response,
        Router,
    };
    use tokio::{net::TcpListener, task::JoinSet, time::timeout};
    use tonic::{
        codegen::tokio_stream::pending,
        service::{Interceptor, Routes},
        transport::{server::TcpIncoming, Endpoint, Server},
        Request,
    };

    use super::{
        blocking, bounded, gateway_frame, hello, Client, Error, PolicyControllerClient, Principal,
        Session, ANSWER_WAIT, PRINCIPAL_ENTRY, PROTOCOL_VERSION,
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

    /// A peer that answers each call with its headers and sends no frame,
    /// for as long as the case holds its task.
    async fn headers_alone() -> (SocketAddr, JoinSet<Result<(), tonic::transport::Error>>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a listener");
        let address = listener.local_addr().expect("its address");
        let headers = Router::new().fallback(|| async {
            Response::builder()
                .header("content-type", "application/grpc")
                .body(Body::from_stream(pending::<Result<Bytes, Infallible>>()))
                .expect("a response")
        });
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(Routes::from(headers))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        (address, serving)
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

    #[tokio::test]
    async fn a_welcome_that_never_comes_ends_the_wait_at_its_bound() {
        assert_eq!(ANSWER_WAIT, Duration::from_secs(5));
        let (address, _serving) = headers_alone().await;
        let channel = Endpoint::from_shared(format!("http://{address}"))
            .expect("a URI")
            .connect()
            .await
            .expect("the dial");
        let principal = Principal::of("gateway-1").expect("a plain id");
        let mut stub = PolicyControllerClient::with_interceptor(channel, principal);
        let opened = timeout(ROOM, Session::open(&mut stub, "gateway-1", 0, SHORT))
            .await
            .expect("the client's own bound ends the wait for the welcome");
        assert!(
            matches!(&opened, Err(Error::NoWelcome(why)) if why.starts_with("no frame within")),
            "{opened:?}"
        );
    }

    #[tokio::test]
    async fn the_plain_form_ends_a_call_that_nothing_answers() {
        let (address, _serving) = mute().await;
        let endpoint = Endpoint::from_shared(format!("http://{address}")).expect("a URI");
        let asked = timeout(
            ROOM,
            Client::connect_within(bounded(endpoint, SHORT), "gateway-1", SHORT),
        )
        .await
        .expect("the endpoint's own bound ends the ask");
        assert!(matches!(asked, Err(Error::Status(_))), "{asked:?}");
    }
}
