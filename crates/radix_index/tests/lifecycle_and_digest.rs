//! End-to-end lifecycle relay and digest miss/resend recovery — the
//! removal and recovery paths the happy-path smoke never drives. A
//! dropped worker must stop being scored; a digest the index has
//! forgotten must be resent in full, never a silent under-match.

use std::{sync::Arc, time::Duration};

use radix_index::{
    client::{QueryOutcome, RemoteIndex},
    server, Engine, EngineConfig,
};

const MODEL: &str = "lifecycle-model";
const BLOCK: u32 = 4;

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture: fire-and-forget server for the test's lifetime"
)]
fn spawn_index(cfg: EngineConfig, sweep: Duration) -> String {
    let port = portpicker::pick_unused_port().expect("port");
    let engine = Arc::new(Engine::new(cfg));
    let addr = format!("127.0.0.1:{port}").parse().unwrap();
    tokio::spawn(server::serve(engine, addr, Vec::new(), sweep));
    format!("http://127.0.0.1:{port}")
}

/// Poll a query until `pred` holds on its outcome, or panic past the
/// deadline.
async fn wait_until(
    client: &RemoteIndex,
    hashes: &[u64],
    what: &str,
    mut pred: impl FnMut(&QueryOutcome) -> bool,
) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let outcome = client
            .query(MODEL, BLOCK, hashes.to_vec(), Duration::from_millis(50))
            .await;
        if pred(&outcome) {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("timed out waiting for {what}; last outcome {outcome:?}");
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// A dropped holder stops being scored (routing must not keep sending to
/// a gone worker), and a re-announce heals it back — the blocks were
/// only soft-retired, not lost.
#[tokio::test]
async fn dropped_holder_stops_scoring_then_readd_restores() {
    let url = spawn_index(EngineConfig::default(), Duration::from_secs(60));
    let client = RemoteIndex::connect(url);
    let hashes = vec![1u64, 2, 3, 4];
    let holder = "grpc://10.0.0.5:9000";

    client.publish_placement(MODEL, BLOCK, holder, &hashes);
    wait_until(
        &client,
        &hashes,
        "placement to become queryable",
        |o| matches!(o, QueryOutcome::Scores(s) if s[0].0 == holder),
    )
    .await;

    // Soft-retire: the holder must drop out of scoring.
    client.publish_dropped(MODEL, BLOCK, holder).await;
    wait_until(&client, &hashes, "dropped holder to stop scoring", |o| {
        matches!(o, QueryOutcome::Empty)
    })
    .await;

    // Re-announce heals it — the soft-retired blocks are scoreable again.
    client.publish_added(MODEL, BLOCK, holder).await;
    wait_until(
        &client,
        &hashes,
        "re-added holder to score again",
        |o| matches!(o, QueryOutcome::Scores(s) if s[0].0 == holder),
    )
    .await;
}

/// Digest miss recovery: after the index evicts an established chain
/// (TTL), the client's re-publish goes out as a `{tip, len}` digest the
/// index cannot confirm; the miss ack must resend the full chain so it
/// becomes queryable again — never a silent under-match.
#[tokio::test]
async fn evicted_digest_chain_is_resent_full_and_recovers() {
    // Short TTL + frequent sweep so an idle placement holder is evicted.
    let cfg = EngineConfig {
        inferred_ttl: Duration::from_millis(50),
        ..Default::default()
    };
    let url = spawn_index(cfg, Duration::from_millis(20));
    // Digest publishing ON: a re-publish of an established chain is a digest.
    let client = RemoteIndex::connect_with(url, true);
    let hashes = vec![101u64, 102, 103, 104];
    let holder = "grpc://10.0.0.9:9000";

    // Establish (full send) -> queryable.
    client.publish_placement(MODEL, BLOCK, holder, &hashes);
    wait_until(&client, &hashes, "chain to establish", |o| {
        matches!(o, QueryOutcome::Scores(_))
    })
    .await;

    // Queries don't refresh liveness, so the idle holder is swept away.
    wait_until(&client, &hashes, "index to evict the idle chain", |o| {
        matches!(o, QueryOutcome::Empty)
    })
    .await;

    // Now the client's cache still holds the tip, so each re-publish is a
    // digest the index misses. Recovery REQUIRES the miss-ack resend to
    // re-send the full chain; if that path were broken this would never
    // become queryable again and the test would time out.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        client.publish_placement(MODEL, BLOCK, holder, &hashes);
        tokio::time::sleep(Duration::from_millis(20)).await;
        if let QueryOutcome::Scores(scores) = client
            .query(MODEL, BLOCK, hashes.clone(), Duration::from_millis(50))
            .await
        {
            assert_eq!(scores[0].0, holder);
            assert_eq!(scores[0].1, hashes.len() as u32, "full chain resent");
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("digest miss never recovered: the full-chain resend did not restore the chain");
        }
    }
}
