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

pub(crate) use super::link::tests::{poll, Peer};
use super::{
    blocking, controller_frame, decide_answer, fit, gateway_frame, hello, numbered, Answer, Busy,
    Client, ControllerFrame, Decide, DecideAnswer, DecideRequest, Decision, Error, GatewayFrame,
    GrandCentralClient, Principal, Session, Stub, Welcome, ANSWER_WAIT, MAX_FRAME, MAX_REQUEST_ID,
    PRINCIPAL_ENTRY, PROTOCOL_FLOOR, PROTOCOL_VERSION,
};
use crate::service::proto::{
    busy, Completed, Credit, Dispatched, EncodePlaced, LeaderInfo, LeaderRequest, NotLeader,
    Released, RequestFacts, Target,
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
        (PROTOCOL_FLOOR, PROTOCOL_VERSION)
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
    GrandCentralClient::with_interceptor(channel, principal)
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
async fn deciding(script: Script) -> (SocketAddr, JoinSet<Result<(), tonic::transport::Error>>) {
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
fn told(term: u64, id: &[u8], answer: impl Into<Option<decide_answer::Answer>>) -> DecideAnswer {
    DecideAnswer {
        request_id: id.to_vec(),
        term,
        answer: answer.into(),
        encode: None,
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
        matches!(&answer, Ok(Answer::Decision(told, None)) if *told == Decision::default()),
        "{answer:?}"
    );
    assert_eq!(
        stream.credit, 7,
        "the credit frame added its window to the credit the stream held"
    );
    assert_eq!(floor, 1, "a frame's term raises the floor");
}

/// The frame of an encode request's pick says where its items go
/// beside the decision (`DecideAnswer.encode`): the answer hands it
/// as the frame carried it, the encoders in their order.
#[tokio::test]
async fn an_answer_hands_where_an_encode_requests_items_go() {
    fn placed() -> EncodePlaced {
        let encoder = |id| Target {
            id,
            ..Target::default()
        };
        EncodePlaced {
            encoders: vec![encoder(9), encoder(4)],
            one: false,
        }
    }
    let (address, _serving) = deciding(|| {
        let frame = DecideAnswer {
            encode: Some(placed()),
            ..told(1, b"a", decision())
        };
        (vec![frame], None)
    })
    .await;
    let (mut stream, _stub) = stream(address, 1).await;
    let answer = asked(&mut stream, &mut 0, b"a").await;
    assert!(
        matches!(&answer, Ok(Answer::Decision(_, Some(told))) if *told == placed()),
        "{answer:?}"
    );
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
            matches!(answer, Ok(Answer::Decision(..))),
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
    assert!(matches!(answer, Ok(Answer::Decision(..))), "{answer:?}");
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
