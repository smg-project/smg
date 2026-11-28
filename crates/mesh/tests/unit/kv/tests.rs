use std::time::Duration;

use super::*;

#[test]
fn test_derive_replica_id_deterministic() {
    let id1 = MeshKV::derive_replica_id("gateway-1");
    let id2 = MeshKV::derive_replica_id("gateway-1");
    assert_eq!(id1, id2, "Same server_name must produce same replica_id");
}

#[test]
fn test_derive_replica_id_different_names() {
    let id1 = MeshKV::derive_replica_id("gateway-1");
    let id2 = MeshKV::derive_replica_id("gateway-2");
    assert_ne!(
        id1, id2,
        "Different server_names should produce different replica_ids"
    );
}

#[test]
fn test_configure_crdt_prefix() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    assert_eq!(ns.prefix(), "worker:");
    assert!(kv.is_prefix_configured("worker:"));
}

#[test]
fn test_configs_prefix_auto_registered() {
    // The `config:` prefix is always registered at construction;
    // gateway code must not have to call configure_crdt_prefix for
    // it, and a redundant call would panic per the one-configure-
    // per-prefix rule.
    let kv = MeshKV::new("test-node".to_string());
    assert!(kv.is_prefix_configured("config:"));
    let configs = kv.configs();
    assert_eq!(configs.prefix(), "config:");
}

#[test]
fn test_configs_put_get_round_trip() {
    let kv = MeshKV::new("test-node".to_string());
    let configs = kv.configs();
    configs.put("config:rate_limit", b"100".to_vec());
    assert_eq!(
        configs.get("config:rate_limit"),
        Some(b"100".to_vec()),
        "config namespace round-trips through the shared CRDT store"
    );
}

#[test]
#[should_panic(expected = "already configured")]
fn test_configure_config_prefix_twice_panics() {
    // Application code must not try to reconfigure `config:` — the
    // auto-registration at MeshKV::new() already owns the slot.
    // This guards against accidental reconfiguration that would
    // replace the merge strategy or subscriber capacity.
    let kv = MeshKV::new("test-node".to_string());
    kv.configure_crdt_prefix("config:", MergeStrategy::LastWriterWins);
}

#[test]
fn test_configure_stream_prefix() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_stream_prefix(
        "td:",
        StreamConfig {
            max_buffer_bytes: 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    assert_eq!(ns.prefix(), "td:");
    assert_eq!(ns.routing(), StreamRouting::Broadcast);
    assert!(kv.is_prefix_configured("td:"));
}

#[test]
#[should_panic(expected = "already configured")]
fn test_duplicate_prefix_panics() {
    let kv = MeshKV::new("test-node".to_string());
    kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins); // panics
}

#[test]
#[should_panic(expected = "already configured")]
fn test_duplicate_prefix_across_modes_panics() {
    let kv = MeshKV::new("test-node".to_string());
    kv.configure_crdt_prefix("data:", MergeStrategy::LastWriterWins);
    kv.configure_stream_prefix(
        "data:",
        StreamConfig {
            max_buffer_bytes: 1024,
            routing: StreamRouting::Broadcast,
        },
    ); // panics
}

#[test]
fn test_crdt_put_get() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    ns.put("worker:7", b"healthy".to_vec());
    let val = ns.get("worker:7");
    assert_eq!(val, Some(b"healthy".to_vec()));
}

#[test]
fn test_crdt_delete() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    ns.put("worker:7", b"healthy".to_vec());
    ns.delete("worker:7");
    assert_eq!(ns.get("worker:7"), None);
}

#[test]
fn test_crdt_keys() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    ns.put("worker:1", b"a".to_vec());
    ns.put("worker:2", b"b".to_vec());
    ns.put("worker:3", b"c".to_vec());

    let mut keys = ns.keys("");
    keys.sort();
    assert_eq!(keys, vec!["worker:1", "worker:2", "worker:3"]);
}

#[test]
#[should_panic(expected = "does not match prefix")]
fn test_crdt_put_wrong_prefix_panics() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    ns.put("policy:1", b"wrong".to_vec()); // panics
}

#[test]
#[should_panic(expected = "only valid on Targeted")]
fn test_stream_publish_to_on_broadcast_panics() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_stream_prefix(
        "td:",
        StreamConfig {
            max_buffer_bytes: 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    ns.publish_to("peer-1", "td:model-x", Bytes::from("data")); // panics
}

#[tokio::test]
async fn test_crdt_subscribe() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    let mut sub = ns.subscribe("");
    ns.put("worker:7", b"healthy".to_vec());

    let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
        .await
        .expect("timeout")
        .expect("channel closed");

    assert_eq!(key, "worker:7");
    let frags = value.expect("put yields Some");
    assert_eq!(frags.len(), 1, "single local write is a single fragment");
    assert_eq!(frags[0].as_ref(), b"healthy");
}

#[tokio::test]
async fn test_crdt_subscribe_delete() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    let mut sub = ns.subscribe("");
    ns.put("worker:7", b"healthy".to_vec());
    ns.delete("worker:7");

    // First event: put
    let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
        .await
        .expect("timeout")
        .expect("channel closed");
    assert_eq!(key, "worker:7");
    assert!(value.is_some());

    // Second event: delete
    let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
        .await
        .expect("timeout")
        .expect("channel closed");
    assert_eq!(key, "worker:7");
    assert!(value.is_none(), "delete should notify with None");
}

#[test]
fn test_targeted_backpressure_drops_oldest() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_stream_prefix(
        "tree:page:",
        StreamConfig {
            max_buffer_bytes: 20, // tiny limit
            routing: StreamRouting::Targeted,
        },
    );

    ns.publish_to("peer-A", "tree:page:m1", Bytes::from("aaaaaaaaaa")); // 10 bytes
    ns.publish_to("peer-A", "tree:page:m2", Bytes::from("bbbbbbbbbb")); // 10 bytes
    ns.publish_to("peer-A", "tree:page:m3", Bytes::from("cccccccccc")); // over limit

    let drained = ns.drain_targeted_buffer();
    let total_bytes: usize = drained.iter().map(|(_, _, v)| v.len()).sum();
    assert!(
        total_bytes <= 20,
        "buffer should be within limit, got {total_bytes}"
    );
    // Oldest entry (m1) should have been dropped, keeping m2 and m3.
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].1, "tree:page:m2");
    assert_eq!(drained[1].1, "tree:page:m3");
}

#[test]
fn test_drain_targeted_buffer() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_stream_prefix(
        "tree:page:",
        StreamConfig {
            max_buffer_bytes: 1024,
            routing: StreamRouting::Targeted,
        },
    );

    ns.publish_to("peer-A", "tree:page:m1", Bytes::from("page1"));
    ns.publish_to("peer-B", "tree:page:m2", Bytes::from("page2"));

    let entries = ns.drain_targeted_buffer();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].0, "peer-A");
    assert_eq!(entries[1].0, "peer-B");

    // Buffer should be empty after drain.
    let entries2 = ns.drain_targeted_buffer();
    assert!(entries2.is_empty());
}

#[test]
fn test_collect_round_batch() {
    let kv = MeshKV::new("test-node".to_string());
    let targeted_ns = kv.configure_stream_prefix(
        "tree:page:",
        StreamConfig {
            max_buffer_bytes: 1024,
            routing: StreamRouting::Targeted,
        },
    );

    targeted_ns.publish_to("peer-A", "tree:page:m1", Bytes::from("page"));

    let batch = kv.collect_round_batch();
    assert_eq!(batch.targeted_entries.len(), 1);
    assert_eq!(batch.targeted_entries[0].0, "peer-A");

    // Second collect should be empty (buffers drained).
    let batch2 = kv.collect_round_batch();
    assert!(batch2.targeted_entries.is_empty());
}

#[test]
fn test_collect_round_batch_with_drain_callback() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_stream_prefix(
        "td:",
        StreamConfig {
            max_buffer_bytes: 1024,
            routing: StreamRouting::Broadcast,
        },
    );

    let _handle = ns.register_drain(Box::new(|| {
        vec![(
            "td:from-drain".to_string(),
            Bytes::from_static(b"drain-data"),
        )]
    }));

    let batch = kv.collect_round_batch();
    assert_eq!(batch.drain_entries.len(), 1);
    assert_eq!(batch.drain_entries[0].0, "td:from-drain");
}

#[test]
#[should_panic(expected = "drain already registered")]
fn test_duplicate_drain_registration_panics() {
    let kv = MeshKV::new("test-node".to_string());
    let ns = kv.configure_stream_prefix(
        "td:",
        StreamConfig {
            max_buffer_bytes: 1024,
            routing: StreamRouting::Broadcast,
        },
    );

    let _h1 = ns.register_drain(Box::new(Vec::new));
    let _h2 = ns.register_drain(Box::new(Vec::new)); // panics
}
