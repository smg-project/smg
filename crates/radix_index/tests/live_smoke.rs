//! Live end-to-end smoke: a real mock gRPC sim worker -> bridge ->
//! index service (over the wire) -> Subscribe query. Verifies the whole
//! event path: worker KV events, hash-only conversion, publish, apply,
//! and overlap scoring, with the query hashed exactly the way a gateway
//! would hash a request.

use std::{sync::Arc, time::Duration};

use futures::StreamExt;
use kv_index::compute_request_content_hashes;
use radix_index::{
    bridge, proto, proto::radix_index_client::RadixIndexClient, server, Engine, EngineConfig,
};
use smg_grpc_client::tokenspeed_scheduler::{tokenspeed_proto as ts, TokenSpeedSchedulerClient};
use tokio::sync::mpsc;

const MODEL: &str = "smoke-model";
const BLOCK: u32 = 4;

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture: fire-and-forget servers for the test's lifetime"
)]
#[tokio::test]
async fn worker_events_reach_the_index_over_the_wire() {
    // 1. Mock gRPC sim worker with KV events (block size 4).
    let worker_port = portpicker::pick_unused_port().expect("port for worker");
    let cfg = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: worker_port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 4,
        realistic: true,
        engine: mock_worker::engine::EngineParams {
            block_size: BLOCK,
            prefill_tps: 1_000_000.0,
            prefix_cache: true,
            ..mock_worker::engine::EngineParams::default()
        },
    });
    tokio::spawn(mock_worker::grpc::serve(
        cfg,
        "127.0.0.1".to_string(),
        worker_port,
    ));

    // 2. Index service over the wire.
    let index_port = portpicker::pick_unused_port().expect("port for index");
    let engine = Arc::new(Engine::new(EngineConfig::default()));
    let addr = format!("127.0.0.1:{index_port}").parse().unwrap();
    tokio::spawn(server::serve(
        Arc::clone(&engine),
        addr,
        Vec::new(),
        Duration::from_secs(60),
    ));

    // 3. Bridge: worker events -> index.
    let worker_url = format!("grpc://127.0.0.1:{worker_port}");
    let index_url = format!("http://127.0.0.1:{index_port}");
    let (tx, rx) = mpsc::channel::<proto::Update>(1024);
    let ledger = bridge::EpochLedger::default();
    tokio::spawn(bridge::worker_loop(
        worker_url.clone(),
        MODEL.to_string(),
        BLOCK,
        tx,
        ledger.clone(),
    ));
    tokio::spawn(bridge::run_publisher(rx, index_url.clone(), ledger.clone()));

    // 4. Drive one generate on the worker (8 tokens = 2 blocks) and wait
    //    for its stream to finish (prefill published at first chunk).
    let input_ids: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let client = {
        let mut attempt = 0;
        loop {
            match TokenSpeedSchedulerClient::connect(&worker_url).await {
                Ok(client) => break client,
                Err(_) if attempt < 50 => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(error) => panic!("worker never came up: {error}"),
            }
        }
    };
    let mut stream = client
        .generate(ts::GenerateRequest {
            request_id: "smoke-1".into(),
            tokenized: Some(ts::TokenizedInput {
                input_ids: input_ids.clone(),
                original_text: String::new(),
            }),
            sampling_params: Some(ts::SamplingParams {
                max_new_tokens: Some(4),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        })
        .await
        .expect("generate");
    while let Some(item) = stream.next().await {
        item.expect("generate stream item");
    }

    // 5. Query over the wire until the holder shows up (event propagation
    //    is async end to end).
    let mut index = RadixIndexClient::connect(index_url).await.expect("index");
    let hashes: Vec<u64> = compute_request_content_hashes(&input_ids, BLOCK as usize)
        .into_iter()
        .map(|h| h.0)
        .collect();
    let (query_tx, query_rx) = mpsc::channel::<proto::Query>(8);
    let outbound = tokio_stream::wrappers::ReceiverStream::new(query_rx);
    let mut answers = index
        .subscribe(tonic::Request::new(outbound))
        .await
        .expect("subscribe")
        .into_inner();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut best: Option<proto::HolderScore> = None;
    let mut query_id = 0u64;
    while tokio::time::Instant::now() < deadline {
        query_id += 1;
        query_tx
            .send(proto::Query {
                query_id,
                keyspace: Some(bridge::keyspace(MODEL, BLOCK)),
                content_hashes: hashes.clone(),
            })
            .await
            .expect("send query");
        let answer = answers
            .next()
            .await
            .expect("answer stream open")
            .expect("answer ok");
        assert_eq!(answer.query_id, query_id, "answers correlate by id");
        if let Some(top) = answer.scores.first() {
            best = Some(top.clone());
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let best = best.expect("the worker's blocks never reached the index");
    assert_eq!(best.holder, worker_url);
    assert_eq!(best.matched_blocks, 2, "both prompt blocks must match");
    assert!(best.event_fed, "bridge traffic is the event feed");
}

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture: fire-and-forget servers for the test's lifetime"
)]
#[tokio::test]
async fn client_placements_and_queries_roundtrip() {
    use radix_index::client::{QueryOutcome, RemoteIndex};

    let port = portpicker::pick_unused_port().expect("port");
    let engine = Arc::new(Engine::new(EngineConfig::default()));
    let addr = format!("127.0.0.1:{port}").parse().unwrap();
    tokio::spawn(server::serve(
        engine,
        addr,
        Vec::new(),
        Duration::from_secs(60),
    ));

    let client = RemoteIndex::connect(format!("http://127.0.0.1:{port}"));
    let tokens: Vec<u32> = (0..16).collect();
    let hashes: Vec<u64> = compute_request_content_hashes(&tokens, BLOCK as usize)
        .into_iter()
        .map(|h| h.0)
        .collect();

    // Before any placement: Empty (or Disconnected while dialing).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match client
            .query(MODEL, BLOCK, hashes.clone(), Duration::from_millis(50))
            .await
        {
            QueryOutcome::Empty => break,
            QueryOutcome::Scores(scores) => panic!("unexpected scores {scores:?}"),
            _ if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            outcome => panic!("index never became reachable: {outcome:?}"),
        }
    }

    client.publish_placement(MODEL, BLOCK, "grpc://10.0.0.1:9000", &hashes);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match client
            .query(MODEL, BLOCK, hashes.clone(), Duration::from_millis(50))
            .await
        {
            QueryOutcome::Scores(scores) => {
                assert_eq!(scores[0].0, "grpc://10.0.0.1:9000");
                assert_eq!(scores[0].1, 4, "16 tokens at block 4 = 4 blocks");
                break;
            }
            _ if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            outcome => panic!("placement never became queryable: {outcome:?}"),
        }
    }
}

/// Digest-enabled client end to end: a fresh chain is established with
/// a full send, re-publishes go out as digests, and a chain the index
/// never saw (digest-first) misses and is resent full — so it still
/// becomes queryable. Exercises the whole client digest + ack-resend
/// path over the real wire.
#[expect(
    clippy::disallowed_methods,
    reason = "test fixture: fire-and-forget servers for the test's lifetime"
)]
#[tokio::test]
async fn client_digest_roundtrip_establishes_and_recovers_misses() {
    use radix_index::client::{QueryOutcome, RemoteIndex};

    let port = portpicker::pick_unused_port().expect("port");
    let engine = Arc::new(Engine::new(EngineConfig::default()));
    tokio::spawn(server::serve(
        engine,
        format!("127.0.0.1:{port}").parse().unwrap(),
        Vec::new(),
        Duration::from_secs(60),
    ));
    let client = RemoteIndex::connect_with(format!("http://127.0.0.1:{port}"), true);
    let holder = "grpc://10.0.0.9:9000";

    let tokens: Vec<u32> = (100..132).collect();
    let hashes: Vec<u64> = compute_request_content_hashes(&tokens, BLOCK as usize)
        .into_iter()
        .map(|h| h.0)
        .collect();
    let expected_blocks = hashes.len() as u32;

    let queryable = |client: std::sync::Arc<RemoteIndex>, hashes: Vec<u64>| async move {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let QueryOutcome::Scores(scores) = client
                .query(MODEL, BLOCK, hashes.clone(), Duration::from_millis(50))
                .await
            {
                return scores[0].1;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("never became queryable");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };

    // First publish establishes (full send); becomes queryable.
    client.publish_placement(MODEL, BLOCK, holder, &hashes);
    assert_eq!(
        queryable(client.clone(), hashes.clone()).await,
        expected_blocks
    );

    // Re-publishes now go out as digests and confirm (no state change);
    // the chain stays queryable.
    for _ in 0..5 {
        client.publish_placement(MODEL, BLOCK, holder, &hashes);
    }
    assert_eq!(
        queryable(client.clone(), hashes.clone()).await,
        expected_blocks
    );
}
