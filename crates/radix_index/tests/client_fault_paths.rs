//! Client fault-path coverage: the `QueryOutcome` variants and the
//! string-mode (`Bytes`) keyspace that the happy-path smoke never
//! exercises. These are the branches the routing hot path relies on to
//! fall back correctly instead of silently corrupting a decision.

use std::{pin::Pin, sync::Arc, time::Duration};

use futures::Stream;
use radix_index::{
    client::{QueryOutcome, RemoteIndex},
    proto::{
        self,
        radix_index_server::{RadixIndex, RadixIndexServer},
    },
    server, Engine, EngineConfig,
};
use tonic::{Request, Response, Status, Streaming};

const MODEL: &str = "fault-model";
const BLOCK: u32 = 4;

/// A RadixIndex that accepts the Subscribe stream (so the client marks
/// itself connected) but NEVER answers a query — the deterministic way
/// to drive the client's deadline path without racing a fast loopback
/// answer (tokio's timer floor makes a sub-ms deadline unreliable).
#[derive(Default)]
struct SilentIndex;

#[tonic::async_trait]
impl RadixIndex for SilentIndex {
    type PublishStream = Pin<Box<dyn Stream<Item = Result<proto::PublishAck, Status>> + Send>>;
    type SubscribeStream = Pin<Box<dyn Stream<Item = Result<proto::Match, Status>> + Send>>;
    type PullStream = Pin<Box<dyn Stream<Item = Result<proto::Update, Status>> + Send>>;

    #[expect(
        clippy::disallowed_methods,
        reason = "test fixture: per-stream drainer task for the test's lifetime"
    )]
    async fn publish(
        &self,
        request: Request<Streaming<proto::Update>>,
    ) -> Result<Response<Self::PublishStream>, Status> {
        let mut inbound = request.into_inner();
        tokio::spawn(async move { while let Ok(Some(_)) = inbound.message().await {} });
        Ok(Response::new(Box::pin(futures::stream::pending())))
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "test fixture: per-stream drainer task for the test's lifetime"
    )]
    async fn subscribe(
        &self,
        request: Request<Streaming<proto::Query>>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        // Drain queries (so the client's send never backpressures) and
        // answer nothing: every query rides its deadline out.
        let mut inbound = request.into_inner();
        tokio::spawn(async move { while let Ok(Some(_)) = inbound.message().await {} });
        Ok(Response::new(Box::pin(futures::stream::pending())))
    }

    async fn pull(
        &self,
        _request: Request<proto::PullRequest>,
    ) -> Result<Response<Self::PullStream>, Status> {
        Ok(Response::new(Box::pin(futures::stream::empty())))
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture: fire-and-forget server for the test's lifetime"
)]
fn spawn_silent_index() -> String {
    let port = portpicker::pick_unused_port().expect("port");
    let addr = format!("127.0.0.1:{port}").parse().unwrap();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(RadixIndexServer::new(SilentIndex))
            .serve(addr)
            .await
    });
    format!("http://127.0.0.1:{port}")
}

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture: fire-and-forget server for the test's lifetime"
)]
fn spawn_index() -> String {
    let port = portpicker::pick_unused_port().expect("port");
    let engine = Arc::new(Engine::new(EngineConfig::default()));
    let addr = format!("127.0.0.1:{port}").parse().unwrap();
    tokio::spawn(server::serve(
        engine,
        addr,
        Vec::new(),
        Duration::from_secs(60),
    ));
    format!("http://127.0.0.1:{port}")
}

/// An empty content-hash slice is a silent no-op, not a panic: the
/// placement builder's `chain.last().expect(...)` is guarded, so a
/// sub-one-block request cannot take down the caller task.
#[tokio::test]
async fn empty_placement_is_a_noop_not_a_panic() {
    let dead_port = portpicker::pick_unused_port().expect("port");
    let client = RemoteIndex::connect(format!("http://127.0.0.1:{dead_port}"));
    // Neither of these may panic on an empty chain.
    client.publish_placement(MODEL, BLOCK, "grpc://x:1", &[]);
    client.publish_placement_bytes(MODEL, BLOCK, "grpc://x:1", &[]);
    // The client is still usable afterward.
    assert_eq!(
        client
            .query(MODEL, BLOCK, vec![1, 2], Duration::from_millis(20))
            .await,
        QueryOutcome::Disconnected
    );
}

/// A query against an index that is not reachable resolves `Disconnected`
/// immediately (the caller falls through to local state) — it never burns
/// its deadline on a stream that cannot answer.
#[tokio::test]
async fn query_against_a_dead_index_is_disconnected() {
    // A port with nothing listening: the subscribe driver never connects,
    // so `connected` stays false and every query fast-fails.
    let dead_port = portpicker::pick_unused_port().expect("port");
    let client = RemoteIndex::connect(format!("http://127.0.0.1:{dead_port}"));

    let outcome = client
        .query(MODEL, BLOCK, vec![1, 2, 3, 4], Duration::from_millis(50))
        .await;
    assert_eq!(outcome, QueryOutcome::Disconnected);
}

/// A live-stream query whose deadline elapses before the answer resolves
/// `Timeout` (NOT `Empty` — the distinction is what tells the caller
/// "fall back", not "the index says no overlap").
#[tokio::test]
async fn query_past_its_deadline_is_timeout_not_empty() {
    let url = spawn_silent_index();
    let client = RemoteIndex::connect(url);
    let hashes = vec![10u64, 20, 30, 40];

    // Once connected to the silent index, every query rides out its
    // deadline and resolves Timeout — distinct from Empty (which would be
    // read as "the index says no overlap" rather than "fall back").
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match client
            .query(MODEL, BLOCK, hashes.clone(), Duration::from_millis(100))
            .await
        {
            QueryOutcome::Timeout => break,
            QueryOutcome::Disconnected if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            other => panic!("expected Timeout once connected, got {other:?}"),
        }
    }
}

/// String-mode (`Bytes` keyspace) publish/query round-trips, and is
/// isolated from the `Tokens` keyspace: the same hashes under the same
/// model + block size live in disjoint keyspaces, so a Tokens query never
/// sees a Bytes holder and vice versa.
#[tokio::test]
async fn bytes_keyspace_roundtrips_and_is_isolated_from_tokens() {
    let url = spawn_index();
    let client = RemoteIndex::connect(url);
    let hashes = vec![7u64, 8, 9, 10];
    let tokens_holder = "grpc://10.0.0.1:9000";
    let bytes_holder = "grpc://10.0.0.2:9000";

    client.publish_placement(MODEL, BLOCK, tokens_holder, &hashes);
    client.publish_placement_bytes(MODEL, BLOCK, bytes_holder, &hashes);

    // Tokens query sees ONLY the tokens holder.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match client
            .query(MODEL, BLOCK, hashes.clone(), Duration::from_millis(50))
            .await
        {
            QueryOutcome::Scores(scores) => {
                assert_eq!(scores.len(), 1, "tokens keyspace holds one holder");
                assert_eq!(scores[0].0, tokens_holder);
                assert_eq!(scores[0].1, hashes.len() as u32);
                break;
            }
            _ if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            other => panic!("tokens placement never became queryable: {other:?}"),
        }
    }

    // Bytes query sees ONLY the bytes holder.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match client
            .query_bytes(MODEL, BLOCK, hashes.clone(), Duration::from_millis(50))
            .await
        {
            QueryOutcome::Scores(scores) => {
                assert_eq!(scores.len(), 1, "bytes keyspace holds one holder");
                assert_eq!(scores[0].0, bytes_holder);
                assert_eq!(scores[0].1, hashes.len() as u32);
                break;
            }
            _ if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            other => panic!("bytes placement never became queryable: {other:?}"),
        }
    }
}
