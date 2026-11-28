use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::RwLock;

use super::{
    is_join_peer, join_peer_name, DialOutcome, Dialer, GossipController, NodeState, NodeStatus,
    RetryManager, INIT_PEER_NAME,
};

fn node(name: &str) -> NodeState {
    NodeState {
        name: name.to_string(),
        address: "127.0.0.1:1".to_string(),
        status: NodeStatus::Alive as i32,
        version: 1,
        metadata: HashMap::new(),
    }
}

/// A connect that answers or refuses, counting its calls.
fn connect(
    calls: &Arc<AtomicUsize>,
    answer: bool,
) -> impl FnOnce(NodeState) -> std::future::Ready<anyhow::Result<()>> {
    let calls = Arc::clone(calls);
    move |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        std::future::ready(if answer {
            Ok(())
        } else {
            Err(anyhow::anyhow!("refused"))
        })
    }
}

/// A member answering resets its retry state; a failure records the
/// attempt and backs off; a dial inside that backoff runs no connect at
/// all; the backoff over, the next dial runs and its success starts the
/// peer afresh.
#[tokio::test]
async fn a_dial_connects_fails_or_backs_off_and_the_retry_state_follows() {
    let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();
    let peer = node("peer-a");
    let calls = Arc::new(AtomicUsize::new(0));
    let member_present = || true;

    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &peer,
        connect(&calls, true),
        member_present,
    )
    .await;
    assert!(matches!(outcome, DialOutcome::Connected), "{outcome:?}");
    assert_eq!(retry_managers["peer-a"].attempt_count(), 0);

    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &peer,
        connect(&calls, false),
        member_present,
    )
    .await;
    match outcome {
        DialOutcome::Failed {
            attempt,
            next_delay,
            ..
        } => {
            assert_eq!(attempt, 1);
            assert_eq!(next_delay, Duration::from_secs(1));
        }
        other => panic!("expected a failure, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::Relaxed), 2);

    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &peer,
        connect(&calls, true),
        member_present,
    )
    .await;
    assert!(
        matches!(outcome, DialOutcome::BackingOff { next_delay } if next_delay == Duration::from_secs(1)),
        "{outcome:?}"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        2,
        "a dial in backoff runs no connect"
    );
    assert_eq!(retry_managers["peer-a"].attempt_count(), 1);

    retry_managers.get_mut("peer-a").unwrap().last_attempt =
        Some(Instant::now() - Duration::from_secs(2));
    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &peer,
        connect(&calls, true),
        member_present,
    )
    .await;
    assert!(matches!(outcome, DialOutcome::Connected), "{outcome:?}");
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert_eq!(retry_managers["peer-a"].attempt_count(), 0);
    assert!(retry_managers["peer-a"].should_retry());
}

/// The join peer answering with nobody else in the table counts as an
/// attempt (the next dial waits out the backoff); answering with a
/// member in the table is a connection like any other.
#[tokio::test]
async fn the_join_peer_answering_alone_records_an_attempt() {
    let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();
    let join_peer = node(INIT_PEER_NAME);
    let calls = Arc::new(AtomicUsize::new(0));

    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &join_peer,
        connect(&calls, true),
        || false,
    )
    .await;
    assert!(
        matches!(outcome, DialOutcome::JoinPeerAlone { next_delay } if next_delay == Duration::from_secs(1)),
        "{outcome:?}"
    );
    assert_eq!(retry_managers[INIT_PEER_NAME].attempt_count(), 1);
    assert!(!retry_managers[INIT_PEER_NAME].should_retry());

    retry_managers.get_mut(INIT_PEER_NAME).unwrap().last_attempt =
        Some(Instant::now() - Duration::from_secs(2));
    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &join_peer,
        connect(&calls, true),
        || true,
    )
    .await;
    assert!(matches!(outcome, DialOutcome::Connected), "{outcome:?}");
    assert_eq!(retry_managers[INIT_PEER_NAME].attempt_count(), 0);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

/// Every configured join peer is a join peer: the second entry's
/// placeholder answering alone records an attempt as the first's does,
/// under a retry state of its own.
#[tokio::test]
async fn every_join_peer_is_dialed_under_its_own_retry_state() {
    assert_eq!(join_peer_name(0), INIT_PEER_NAME);
    assert_eq!(join_peer_name(2), "init_peer_2");
    assert!(is_join_peer(INIT_PEER_NAME) && is_join_peer("init_peer_1"));
    assert!(!is_join_peer("init_peerless") && !is_join_peer("peer-a"));

    let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let second = node(&join_peer_name(1));
    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &second,
        connect(&calls, true),
        || false,
    )
    .await;
    assert!(
        matches!(outcome, DialOutcome::JoinPeerAlone { .. }),
        "{outcome:?}"
    );
    assert_eq!(retry_managers["init_peer_1"].attempt_count(), 1);
    assert!(
        !retry_managers.contains_key(INIT_PEER_NAME),
        "the first join peer's retry state is untouched"
    );
}

/// While the table holds nobody to probe, the rounds dial the configured
/// join peers in turn, each under its placeholder; a member in the table
/// takes precedence over all of them.
#[test]
fn the_rounds_dial_the_join_peers_in_turn_while_the_table_is_empty() {
    let seeds: Vec<SocketAddr> = vec![
        "127.0.0.1:1".parse().unwrap(),
        "127.0.0.1:2".parse().unwrap(),
        "127.0.0.1:3".parse().unwrap(),
    ];
    let state = Arc::new(RwLock::new(BTreeMap::new()));
    let controller = GossipController::new(
        state,
        "127.0.0.1:9".parse().unwrap(),
        "self",
        seeds.clone(),
        Dialer::new(None),
    );
    let nobody = BTreeMap::new();

    let dialed: Vec<(String, String)> = (0..6)
        .map(|round| {
            let peer = controller
                .peer_to_dial(&nobody, round)
                .expect("a join peer to dial");
            (peer.name, peer.address)
        })
        .collect();
    assert_eq!(
        dialed,
        vec![
            ("init_peer".to_string(), seeds[0].to_string()),
            ("init_peer_1".to_string(), seeds[1].to_string()),
            ("init_peer_2".to_string(), seeds[2].to_string()),
            ("init_peer".to_string(), seeds[0].to_string()),
            ("init_peer_1".to_string(), seeds[1].to_string()),
            ("init_peer_2".to_string(), seeds[2].to_string()),
        ],
        "the entries are dialed in turn, round by round"
    );

    let members = BTreeMap::from([("peer-a".to_string(), node("peer-a"))]);
    let peer = controller.peer_to_dial(&members, 0).expect("a member");
    assert_eq!(peer.name, "peer-a", "a member in the table comes first");

    let alone = GossipController::new(
        Arc::new(RwLock::new(BTreeMap::new())),
        "127.0.0.1:9".parse().unwrap(),
        "self",
        Vec::new(),
        Dialer::new(None),
    );
    assert!(
        alone.peer_to_dial(&nobody, 0).is_none(),
        "no join peer configured, nobody to dial"
    );
}

/// Two failures in a row double the wait, and each peer keeps its own
/// retry state.
#[tokio::test]
async fn repeated_failures_lengthen_the_wait_per_peer() {
    let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();
    let peer_a = node("peer-a");
    let peer_b = node("peer-b");
    let calls = Arc::new(AtomicUsize::new(0));

    GossipController::dial_with_backoff(
        &mut retry_managers,
        &peer_a,
        connect(&calls, false),
        || true,
    )
    .await;
    retry_managers.get_mut("peer-a").unwrap().last_attempt =
        Some(Instant::now() - Duration::from_secs(5));
    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &peer_a,
        connect(&calls, false),
        || true,
    )
    .await;
    assert!(
        matches!(outcome, DialOutcome::Failed { attempt: 2, next_delay, .. } if next_delay == Duration::from_secs(2)),
        "{outcome:?}"
    );

    let outcome = GossipController::dial_with_backoff(
        &mut retry_managers,
        &peer_b,
        connect(&calls, true),
        || true,
    )
    .await;
    assert!(matches!(outcome, DialOutcome::Connected), "{outcome:?}");
    assert_eq!(retry_managers["peer-b"].attempt_count(), 0);
    assert_eq!(retry_managers["peer-a"].attempt_count(), 2);
}
