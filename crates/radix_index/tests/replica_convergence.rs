//! Two LIVE replicas peered with each other over real gRPC, driven
//! through the real Publish surface. This is the drill that would have
//! caught the audit's move-only relay suppression: a dangling-parent
//! re-anchor MOVES blocks (net length delta zero — query answers
//! change, block count does not), and the peer must still converge.
//! The symmetric peering also exercises echo suppression: every relay
//! the peer applies is offered back, and must die as a no-op instead
//! of amplifying.

use std::{sync::Arc, time::Duration};

use radix_index::{
    bridge, engine::placement_chain, proto, proto::radix_index_client::RadixIndexClient, server,
    ContentHash, Engine, EngineConfig,
};
use tokio::sync::mpsc;

const MODEL: &str = "conv-model";
const BLOCK: u32 = 4;

fn keyspace_key() -> radix_index::engine::KeyspaceKey {
    radix_index::engine::KeyspaceKey {
        model: MODEL.into(),
        symbol_kind: radix_index::engine::SymbolKind::Tokens,
        block_size: BLOCK,
    }
}

fn stored(seq: u64, parent: Option<u64>, blocks: &[(u64, u64)]) -> proto::Update {
    proto::Update {
        keyspace: Some(bridge::keyspace(MODEL, BLOCK)),
        holder: "worker-a".into(),
        epoch: 1,
        seq,
        events: vec![proto::Event {
            kind: Some(proto::event::Kind::Stored(proto::Stored {
                parent_seq_hash: parent,
                blocks: blocks
                    .iter()
                    .map(|&(seq_hash, content_hash)| proto::Block {
                        seq_hash,
                        content_hash,
                    })
                    .collect(),
            })),
        }],
        added: None,
        dropped: false,
    }
}

/// Both engines' full answer for `chain`, as comparable rows.
fn answers(engine: &Engine, chain: &[u64]) -> Vec<(String, u32, u64)> {
    let hashes: Vec<ContentHash> = chain.iter().map(|&h| ContentHash(h)).collect();
    engine
        .find_matches(&keyspace_key(), &hashes)
        .into_iter()
        .map(|s| (s.holder, s.matched_blocks, s.total_blocks))
        .collect()
}

async fn converged(a: &Engine, b: &Engine, chains: &[Vec<u64>]) -> bool {
    for _ in 0..200 {
        let mut all_equal = true;
        for chain in chains {
            let left = answers(a, chain);
            if left.is_empty() || left != answers(b, chain) {
                all_equal = false;
                break;
            }
        }
        if all_equal {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture: fire-and-forget servers for the test's lifetime"
)]
#[tokio::test]
async fn peered_replicas_converge_through_moves_and_echoes() {
    let port_a = portpicker::pick_unused_port().expect("port a");
    let port_b = portpicker::pick_unused_port().expect("port b");
    let url_a = format!("http://127.0.0.1:{port_a}");
    let url_b = format!("http://127.0.0.1:{port_b}");

    let engine_a = Arc::new(Engine::new(EngineConfig::default()));
    let engine_b = Arc::new(Engine::new(EngineConfig::default()));
    tokio::spawn(server::serve(
        Arc::clone(&engine_a),
        format!("127.0.0.1:{port_a}").parse().unwrap(),
        vec![url_b.clone()],
        Duration::from_secs(60),
    ));
    tokio::spawn(server::serve(
        Arc::clone(&engine_b),
        format!("127.0.0.1:{port_b}").parse().unwrap(),
        vec![url_a.clone()],
        Duration::from_secs(60),
    ));

    let mut client_a = {
        let mut attempt = 0;
        loop {
            match RadixIndexClient::connect(url_a.clone()).await {
                Ok(client) => break client,
                Err(_) if attempt < 50 => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(error) => panic!("replica A never came up: {error}"),
            }
        }
    };

    // Event-fed chain on A only; relay must carry everything to B.
    // Keys 1..=6 with contents 101..=106; k5/k6 first extend the chain
    // at positions 4..5...
    let (tx, rx) = mpsc::channel::<proto::Update>(64);
    tx.send(stored(1, None, &[(1, 101), (2, 102), (3, 103), (4, 104)]))
        .await
        .unwrap();
    tx.send(stored(2, Some(4), &[(5, 105), (6, 106)]))
        .await
        .unwrap();
    // ...then re-arrive under a DANGLING parent: the engine re-anchors
    // at position 0, which is a same-key MOVE — zero net blocks, so a
    // length-delta relay heuristic would suppress it and B would keep
    // answering with the pre-move placement forever.
    tx.send(stored(3, Some(0xDEAD_BEEF), &[(5, 105), (6, 106)]))
        .await
        .unwrap();
    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);
    let mut acks = client_a
        .publish(tonic::Request::new(outbound))
        .await
        .expect("publish stream")
        .into_inner();
    for _ in 0..3 {
        tonic::codegen::tokio_stream::StreamExt::next(&mut acks)
            .await
            .expect("ack")
            .expect("ack ok");
    }

    // After the move, [105, 106] matches from position 0 and the old
    // six-deep prefix no longer fully matches; whatever A answers, B
    // must answer identically.
    let moved_chain = vec![105, 106];
    let original_chain = vec![101, 102, 103, 104];
    assert!(
        converged(
            &engine_a,
            &engine_b,
            &[moved_chain.clone(), original_chain.clone()]
        )
        .await,
        "replica B never converged with A: A={:?}/{:?} B={:?}/{:?}",
        answers(&engine_a, &moved_chain),
        answers(&engine_a, &original_chain),
        answers(&engine_b, &moved_chain),
        answers(&engine_b, &original_chain),
    );
    // The move really moved: depth-2 match at the NEW anchor.
    assert_eq!(answers(&engine_a, &moved_chain)[0].1, 2);

    // Placement idempotence + echo death: publish the SAME placement
    // to BOTH replicas (as fan-out publishers do). Each relays to the
    // other; the echo must apply as a no-op and stop.
    let contents: Vec<ContentHash> = (201..=204).map(ContentHash).collect();
    let chain: Vec<(u64, u64)> = placement_chain(&contents)
        .iter()
        .map(|b| (b.seq_hash.0, b.content_hash.0))
        .collect();
    let mut placement = stored(0, None, &chain);
    placement.holder = "worker-p".into();
    for url in [&url_a, &url_b] {
        let mut client = RadixIndexClient::connect(url.clone())
            .await
            .expect("client");
        let (tx, rx) = mpsc::channel::<proto::Update>(4);
        tx.send(placement.clone()).await.unwrap();
        drop(tx);
        let _ = client
            .publish(tonic::Request::new(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            ))
            .await
            .expect("publish placement")
            .into_inner();
    }
    let placement_chain_query = vec![201, 202, 203, 204];
    assert!(
        converged(
            &engine_a,
            &engine_b,
            std::slice::from_ref(&placement_chain_query)
        )
        .await,
        "placement never converged"
    );

    // Quiet-period stability: with no new publishes, block counts must
    // hold still — an echo loop would keep mutating (the fault drill
    // measured ~190x apply amplification from exactly that bug).
    let before = (engine_a.entry_count(), engine_b.entry_count());
    tokio::time::sleep(Duration::from_millis(500)).await;
    let after = (engine_a.entry_count(), engine_b.entry_count());
    assert_eq!(before, after, "state kept changing with no publisher");
    assert_eq!(
        answers(&engine_a, &placement_chain_query),
        answers(&engine_b, &placement_chain_query)
    );
}
