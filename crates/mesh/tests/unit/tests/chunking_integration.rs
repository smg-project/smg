//! Integration tests for the mesh stream chunking transport.
//! Exercises the sender + receiver halves of the chunking pipeline
//! through their public call seams without standing up a real gRPC
//! loop, so failures localize to the chunking / assembler / dispatch
//! code instead of being masked by transport flakiness.
//!
//! Scenarios covered:
//! - **round-trip, single chunk**: `publish()` of a small value reaches
//!   a subscribed receiver via the single-chunk fast path with no
//!   assembler state.
//! - **round-trip, multi-chunk**: oversized value is split by
//!   `chunk_value`, reassembled by `ChunkAssembler`, and delivered as a
//!   fragmented `Vec<Bytes>` — count of fragments and total payload
//!   match the source.
//! - **timeout**: partial assembly is collected then GC'd; subscriber
//!   never fires.
//! - **concurrent generations**: overlapping publishes to the same key
//!   (different generations) resolve to the newer generation's value;
//!   the older is dropped.
//! - **multi-key (multi-model)**: parallel chunked publishes to
//!   distinct keys don't cross-contaminate; each subscriber sees only
//!   its own payload.
//! - **peer-scope isolation**: two senders under the same key don't
//!   collide in the assembler; both payloads deliver.
//! - **tree-page bypass**: `tree:page:*` values are bounded by the
//!   repair-page size contract and therefore always take the
//!   single-chunk fast path in `dispatch_stream_batch` without
//!   touching the assembler. Asserted through the dispatch entry
//!   point, not just the shape of `chunk_value`'s return.

use std::{sync::Arc, time::Duration};

use bytes::Bytes;
use tokio::sync::mpsc::error::TryRecvError;

use crate::{
    kv::{MeshKV, StreamConfig, StreamNamespace, StreamRouting},
    service::gossip::StreamEntry,
    transport::{
        chunking::{build_stream_batches, chunk_value, next_generation},
        limits::{DEFAULT_MAX_CHUNKS_PER_BATCH, MAX_STREAM_CHUNK_BYTES},
        sync_stream::dispatch_stream_batch,
    },
};

/// Test-scoped chunk bytes. Smaller than production to keep test
/// allocations modest while still exercising the multi-chunk split
/// path. Every caller using this must stay below whatever cap is set;
/// tests that want to force a specific chunk count size the payload
/// against this value.
const TEST_MAX_CHUNK_BYTES: usize = 1024;

/// Drive one or more `StreamBatch`es produced on the sender side into
/// the receiver's dispatch path, flattening each batch's entries back
/// into a single dispatch call. This is the simulated wire: in
/// production the batches would serialize through gRPC, but within a
/// test we bypass prost and feed the `StreamEntry` sequences
/// directly.
fn deliver_batches(mesh_kv: &MeshKV, peer_id: &str, entries: Vec<StreamEntry>) {
    for batch in build_stream_batches(
        entries,
        DEFAULT_MAX_CHUNKS_PER_BATCH,
        MAX_STREAM_CHUNK_BYTES,
    ) {
        dispatch_stream_batch(mesh_kv, peer_id, batch.entries);
    }
}

/// Concatenate a fragmented subscriber payload into a contiguous
/// `Vec<u8>` for comparison assertions. Production subscribers that
/// need a contiguous buffer do the equivalent; tests match that shape.
fn flatten_payload(fragments: &[Bytes]) -> Vec<u8> {
    fragments.iter().flat_map(|b| b.iter().copied()).collect()
}

/// A receiver with the `td:` prefix configured as a 64 KiB broadcast stream:
/// where the round-trip tests start.
fn receiver_with_td_stream() -> (MeshKV, Arc<StreamNamespace>) {
    let mesh_kv = MeshKV::new("receiver".to_string());
    let ns = mesh_kv.configure_stream_prefix(
        "td:",
        StreamConfig {
            max_buffer_bytes: 64 * 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    (mesh_kv, ns)
}

// ---- round-trip: single chunk ----

#[tokio::test]
async fn round_trip_single_chunk_fires_subscriber() {
    let (mesh_kv, ns) = receiver_with_td_stream();
    let mut sub = ns.subscribe("model-1");

    let value = Bytes::from_static(b"hello-tenant-delta");
    let entries = chunk_value(
        "td:model-1".to_string(),
        next_generation(),
        value.clone(),
        TEST_MAX_CHUNK_BYTES,
    );
    assert_eq!(entries.len(), 1, "small value takes single-chunk fast path");
    assert_eq!(entries[0].total_chunks, 1);

    deliver_batches(&mesh_kv, "sender", entries);

    let (key, payload) = sub
        .receiver
        .recv()
        .await
        .expect("subscriber channel should yield event");
    assert_eq!(key, "td:model-1");
    let fragments = payload.expect("non-delete publish");
    assert_eq!(flatten_payload(&fragments), value.to_vec());
    assert_eq!(
        mesh_kv.chunk_assembler().in_flight(),
        0,
        "single-chunk fast path should not allocate assembler state"
    );
}

// ---- round-trip: multi-chunk ----

#[tokio::test]
async fn round_trip_multi_chunk_reassembles_in_fragments() {
    let (mesh_kv, ns) = receiver_with_td_stream();
    let mut sub = ns.subscribe("model-big");

    // 3 chunks of TEST_MAX_CHUNK_BYTES each, plus a short tail.
    let payload: Vec<u8> = (0u8..=255)
        .cycle()
        .take(3 * TEST_MAX_CHUNK_BYTES + 7)
        .collect();
    let value = Bytes::from(payload.clone());

    let entries = chunk_value(
        "td:model-big".to_string(),
        next_generation(),
        value,
        TEST_MAX_CHUNK_BYTES,
    );
    assert_eq!(entries.len(), 4, "3 full chunks + 1 tail chunk");
    assert!(entries.iter().all(|e| e.total_chunks == 4));

    deliver_batches(&mesh_kv, "sender", entries);

    let (key, fragments) = sub
        .receiver
        .recv()
        .await
        .expect("subscriber should fire on full reassembly");
    let fragments = fragments.expect("non-delete");
    assert_eq!(key, "td:model-big");
    assert_eq!(fragments.len(), 4, "one Bytes per chunk, no concat");
    assert_eq!(flatten_payload(&fragments), payload);
    assert_eq!(
        mesh_kv.chunk_assembler().in_flight(),
        0,
        "completed assembly removed from map"
    );
}

// ---- timeout: partial assembly GC'd ----

#[tokio::test]
async fn partial_assembly_times_out_and_is_gcd() {
    let (mesh_kv, ns) = receiver_with_td_stream();
    let mut sub = ns.subscribe("stalled");

    let payload = vec![0u8; 2 * TEST_MAX_CHUNK_BYTES];
    let all_entries = chunk_value(
        "td:stalled".to_string(),
        next_generation(),
        Bytes::from(payload),
        TEST_MAX_CHUNK_BYTES,
    );
    assert_eq!(all_entries.len(), 2);

    // Deliver only the first chunk; hold the second.
    let first = vec![all_entries[0].clone()];
    deliver_batches(&mesh_kv, "sender", first);
    assert_eq!(
        mesh_kv.chunk_assembler().in_flight(),
        1,
        "partial assembly retained"
    );

    // GC with a timeout shorter than the chunk's age-so-far would
    // evict instantly in test time; sleep just enough to guarantee
    // `Instant::elapsed()` crosses the threshold on coarse clocks.
    tokio::time::sleep(Duration::from_millis(10)).await;
    mesh_kv.chunk_assembler().gc(Duration::from_millis(1));
    assert_eq!(
        mesh_kv.chunk_assembler().in_flight(),
        0,
        "stale partial should be evicted by gc"
    );

    // Subscriber never fires on a GC'd partial — the value never
    // completed. try_recv returns Empty.
    match sub.receiver.try_recv() {
        Err(TryRecvError::Empty) => {}
        other => panic!("no event expected for GC'd partial, got {other:?}"),
    }
}

// ---- concurrent generations: same key, different publishes ----

#[tokio::test]
async fn concurrent_generations_newer_wins_older_dropped() {
    let (mesh_kv, ns) = receiver_with_td_stream();
    let mut sub = ns.subscribe("racing");

    let old_value: Vec<u8> = (0u8..=255).cycle().take(2 * TEST_MAX_CHUNK_BYTES).collect();
    let new_value: Vec<u8> = (128u8..=255)
        .chain(0u8..=127)
        .cycle()
        .take(2 * TEST_MAX_CHUNK_BYTES)
        .collect();
    assert_ne!(
        old_value, new_value,
        "sanity: distinct payloads for the two generations"
    );

    let gen_old = next_generation();
    let gen_new = next_generation();
    assert!(gen_new > gen_old, "sanity: generation counter monotonic");

    let old_entries = chunk_value(
        "td:racing".to_string(),
        gen_old,
        Bytes::from(old_value.clone()),
        TEST_MAX_CHUNK_BYTES,
    );
    let new_entries = chunk_value(
        "td:racing".to_string(),
        gen_new,
        Bytes::from(new_value.clone()),
        TEST_MAX_CHUNK_BYTES,
    );

    // Interleave old+new arrivals to mimic reordered delivery from a
    // single peer. Order: old[0], new[0], old[1], new[1].
    deliver_batches(
        &mesh_kv,
        "sender",
        vec![
            old_entries[0].clone(),
            new_entries[0].clone(),
            old_entries[1].clone(),
            new_entries[1].clone(),
        ],
    );

    let (key, fragments) = sub
        .receiver
        .recv()
        .await
        .expect("newer generation should complete reassembly");
    assert_eq!(key, "td:racing");
    let fragments = fragments.expect("non-delete");
    assert_eq!(
        flatten_payload(&fragments),
        new_value,
        "receiver reassembled the newer generation's payload"
    );
    assert!(
        sub.receiver.try_recv().is_err(),
        "older generation must not deliver after being superseded"
    );
    assert_eq!(
        mesh_kv.chunk_assembler().in_flight(),
        0,
        "completed newer-gen assembly removed; older was discarded"
    );
}

// ---- multi-key (multi-model): parallel chunked publishes don't cross-contaminate ----

#[tokio::test]
async fn multi_key_parallel_reassembly_isolates_per_key() {
    let (mesh_kv, ns) = receiver_with_td_stream();
    let mut sub_a = ns.subscribe("model-A");
    let mut sub_b = ns.subscribe("model-B");

    let value_a: Vec<u8> = (0u8..=255).cycle().take(2 * TEST_MAX_CHUNK_BYTES).collect();
    let value_b: Vec<u8> = (255u8..=255)
        .chain(0u8..=254)
        .cycle()
        .take(2 * TEST_MAX_CHUNK_BYTES)
        .collect();

    let a_entries = chunk_value(
        "td:model-A".to_string(),
        next_generation(),
        Bytes::from(value_a.clone()),
        TEST_MAX_CHUNK_BYTES,
    );
    let b_entries = chunk_value(
        "td:model-B".to_string(),
        next_generation(),
        Bytes::from(value_b.clone()),
        TEST_MAX_CHUNK_BYTES,
    );

    // Interleave so neither value finishes before the other starts.
    deliver_batches(
        &mesh_kv,
        "sender",
        vec![
            a_entries[0].clone(),
            b_entries[0].clone(),
            a_entries[1].clone(),
            b_entries[1].clone(),
        ],
    );

    let (key_a, frag_a) = sub_a.receiver.recv().await.expect("A fires");
    assert_eq!(key_a, "td:model-A");
    assert_eq!(flatten_payload(&frag_a.expect("non-delete")), value_a);

    let (key_b, frag_b) = sub_b.receiver.recv().await.expect("B fires");
    assert_eq!(key_b, "td:model-B");
    assert_eq!(flatten_payload(&frag_b.expect("non-delete")), value_b);

    assert!(
        sub_a.receiver.try_recv().is_err(),
        "A subscriber must not see B's payload"
    );
    assert!(
        sub_b.receiver.try_recv().is_err(),
        "B subscriber must not see A's payload"
    );
    assert_eq!(mesh_kv.chunk_assembler().in_flight(), 0);
}

// ---- peer-scope isolation: two senders with the same key don't collide ----

#[tokio::test]
async fn same_key_from_two_peers_reassembles_independently() {
    let (mesh_kv, ns) = receiver_with_td_stream();
    let mut sub = ns.subscribe("shared-key");

    // Both senders publish DIFFERENT payloads under the SAME key. The
    // assembler scopes in-flight state by (peer_id, key), so neither
    // sender's chunks land in the other's slot.
    let value_from_a: Vec<u8> = (0u8..=127).cycle().take(2 * TEST_MAX_CHUNK_BYTES).collect();
    let value_from_b: Vec<u8> = (128u8..=255)
        .cycle()
        .take(2 * TEST_MAX_CHUNK_BYTES)
        .collect();

    let a_entries = chunk_value(
        "td:shared-key".to_string(),
        next_generation(),
        Bytes::from(value_from_a.clone()),
        TEST_MAX_CHUNK_BYTES,
    );
    let b_entries = chunk_value(
        "td:shared-key".to_string(),
        next_generation(),
        Bytes::from(value_from_b.clone()),
        TEST_MAX_CHUNK_BYTES,
    );

    // Interleave across the two peers.
    dispatch_stream_batch(&mesh_kv, "peer-A", [a_entries[0].clone()]);
    dispatch_stream_batch(&mesh_kv, "peer-B", [b_entries[0].clone()]);
    dispatch_stream_batch(&mesh_kv, "peer-A", [a_entries[1].clone()]);
    dispatch_stream_batch(&mesh_kv, "peer-B", [b_entries[1].clone()]);

    // Two subscriber fires expected, one per (peer, key) assembly.
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    for _ in 0..2 {
        let (_k, frag) = sub.receiver.recv().await.expect("both peers complete");
        payloads.push(flatten_payload(&frag.expect("non-delete")));
    }
    payloads.sort();
    let mut expected = vec![value_from_a, value_from_b];
    expected.sort();
    assert_eq!(
        payloads, expected,
        "both payloads delivered without collision"
    );
    assert_eq!(mesh_kv.chunk_assembler().in_flight(), 0);
}

// ---- tree page: single-chunk fast path, no assembler touch ----

#[tokio::test]
async fn tree_page_takes_single_chunk_fast_path_no_assembler_state() {
    let mesh_kv = MeshKV::new("receiver".to_string());
    let ns = mesh_kv.configure_stream_prefix(
        "tree:page:",
        StreamConfig {
            max_buffer_bytes: 8 * 1024 * 1024,
            routing: StreamRouting::Targeted,
        },
    );
    let mut sub = ns.subscribe("model-x");

    // A realistic tree page — ~256 KB, well under any sensible chunk
    // bound, so chunk_value MUST return a single entry regardless of
    // the cap passed in.
    let page: Vec<u8> = (0u8..=255).cycle().take(256 * 1024).collect();
    let entries = chunk_value(
        "tree:page:model-x:0".to_string(),
        next_generation(),
        Bytes::from(page.clone()),
        MAX_STREAM_CHUNK_BYTES,
    );
    assert_eq!(
        entries.len(),
        1,
        "bounded tree:page value must stay single-chunk"
    );
    assert_eq!(entries[0].total_chunks, 1);

    let in_flight_before = mesh_kv.chunk_assembler().in_flight();
    let total_bytes_before = mesh_kv.chunk_assembler().total_bytes();

    dispatch_stream_batch(&mesh_kv, "peer-A", entries);

    // Subscriber fires immediately via the fast path.
    let (key, frag) = sub
        .receiver
        .recv()
        .await
        .expect("single-chunk tree page fires subscriber immediately");
    assert_eq!(key, "tree:page:model-x:0");
    assert_eq!(flatten_payload(&frag.expect("non-delete")), page);

    // The assembler must not have been touched for this path.
    assert_eq!(
        mesh_kv.chunk_assembler().in_flight(),
        in_flight_before,
        "single-chunk fast path must not allocate assembler state"
    );
    assert_eq!(
        mesh_kv.chunk_assembler().total_bytes(),
        total_bytes_before,
        "single-chunk fast path must not retain any chunk bytes"
    );
}
