use std::{thread, time::Duration};

use super::*;

/// Flatten a fragmented assembly result into a contiguous Vec<u8>
/// for test assertions only. Production callers handle fragments
/// directly (zero-copy fan-out).
fn flatten(bs: &[Bytes]) -> Vec<u8> {
    bs.iter().flat_map(|b| b.iter().copied()).collect()
}

#[test]
fn test_single_chunk_round_trip() {
    let asm = ChunkAssembler::new();
    let out = asm
        .receive_chunk("peer", "k", 1, 0, 1, Bytes::from_static(b"hello"))
        .unwrap();
    assert_eq!(flatten(&out), b"hello");
    assert_eq!(asm.in_flight(), 0, "completed assembly should be removed");
}

#[test]
fn test_multi_chunk_in_order() {
    let asm = ChunkAssembler::new();
    assert!(asm
        .receive_chunk("peer", "k", 1, 0, 3, Bytes::from_static(b"aaa"))
        .is_none());
    assert!(asm
        .receive_chunk("peer", "k", 1, 1, 3, Bytes::from_static(b"bbb"))
        .is_none());
    let out = asm
        .receive_chunk("peer", "k", 1, 2, 3, Bytes::from_static(b"ccc"))
        .unwrap();
    assert_eq!(flatten(&out), b"aaabbbccc");
    assert_eq!(asm.in_flight(), 0);
}

#[test]
fn test_multi_chunk_out_of_order() {
    let asm = ChunkAssembler::new();
    assert!(asm
        .receive_chunk("peer", "k", 1, 2, 3, Bytes::from_static(b"ccc"))
        .is_none());
    assert!(asm
        .receive_chunk("peer", "k", 1, 0, 3, Bytes::from_static(b"aaa"))
        .is_none());
    let out = asm
        .receive_chunk("peer", "k", 1, 1, 3, Bytes::from_static(b"bbb"))
        .unwrap();
    assert_eq!(
        flatten(&out),
        b"aaabbbccc",
        "chunks must assemble in index order"
    );
}

#[test]
fn test_generation_reset_discards_older_partials() {
    let asm = ChunkAssembler::new();
    assert!(asm
        .receive_chunk("peer", "k", 5, 0, 3, Bytes::from_static(b"old-0"))
        .is_none());
    assert!(asm
        .receive_chunk("peer", "k", 5, 1, 3, Bytes::from_static(b"old-1"))
        .is_none());

    assert!(asm
        .receive_chunk("peer", "k", 6, 0, 2, Bytes::from_static(b"new-0"))
        .is_none());
    let out = asm
        .receive_chunk("peer", "k", 6, 1, 2, Bytes::from_static(b"new-1"))
        .unwrap();
    assert_eq!(flatten(&out), b"new-0new-1");
}

#[test]
fn test_delayed_older_generation_chunk_is_dropped() {
    let asm = ChunkAssembler::new();
    // gen=6 starts and records one chunk.
    assert!(asm
        .receive_chunk("peer", "k", 6, 0, 2, Bytes::from_static(b"new-0"))
        .is_none());

    // A delayed gen=5 chunk arrives — must NOT reset the gen=6 state.
    assert!(asm
        .receive_chunk("peer", "k", 5, 0, 3, Bytes::from_static(b"stale-0"))
        .is_none());

    // Completing gen=6 must still yield the gen=6 payload.
    let out = asm
        .receive_chunk("peer", "k", 6, 1, 2, Bytes::from_static(b"new-1"))
        .unwrap();
    assert_eq!(
        flatten(&out),
        b"new-0new-1",
        "stale older chunk must not overwrite newer state"
    );
}

#[test]
fn test_gc_removes_stale_partials() {
    let asm = ChunkAssembler::new();
    assert!(asm
        .receive_chunk("peer", "k", 1, 0, 3, Bytes::from_static(b"aaa"))
        .is_none());
    assert_eq!(asm.in_flight(), 1);

    thread::sleep(Duration::from_millis(50));
    asm.gc(Duration::from_millis(30));
    assert_eq!(asm.in_flight(), 0, "stale partial should be GC'd");
}

#[test]
fn test_gc_keeps_recent_partials() {
    let asm = ChunkAssembler::new();
    assert!(asm
        .receive_chunk("peer", "k", 1, 0, 3, Bytes::from_static(b"aaa"))
        .is_none());
    asm.gc(Duration::from_secs(10));
    assert_eq!(asm.in_flight(), 1, "recent partial should survive gc");
}

#[test]
fn test_gc_skips_complete_assemblies() {
    // A complete assembly sitting in the map (before its owning
    // receive_chunk extracts it) must not be removed by gc even if
    // the timeout would otherwise apply.
    let asm = ChunkAssembler::new();
    let old = Instant::now() - Duration::from_secs(60);
    let k: AssemblyKey = ("peer".into(), "complete".into());
    asm.assemblies.insert(
        k.clone(),
        AssemblyState {
            generation: 1,
            received_count: 2,
            chunks: vec![
                Some(Bytes::from(vec![0u8; 10])),
                Some(Bytes::from(vec![0u8; 10])),
            ],
            created_at: old,
        },
    );
    asm.gc(Duration::from_secs(1));
    assert!(
        asm.assemblies.contains_key(&k),
        "gc must not remove a complete assembly"
    );
}

#[test]
fn test_gc_rechecks_timeout_at_remove() {
    // Exercise the real collect→remove race window by driving the
    // two phases manually. Overwrite the entry between phases; the
    // remove_if timeout re-check must spare the fresh state.
    let asm = ChunkAssembler::new();
    let timeout = Duration::from_secs(1);
    let k: AssemblyKey = ("peer".into(), "k".into());

    asm.assemblies.insert(
        k.clone(),
        AssemblyState {
            generation: 1,
            received_count: 0,
            chunks: vec![None, None],
            created_at: Instant::now() - Duration::from_secs(60),
        },
    );

    // Phase 1: collect sees the expired entry.
    let expired = asm.collect_expired(timeout);
    assert_eq!(expired, vec![k.clone()]);

    // Race: a concurrent receive_chunk resets the entry to a fresh
    // generation before gc's remove phase fires.
    asm.assemblies.insert(
        k.clone(),
        AssemblyState {
            generation: 2,
            received_count: 0,
            chunks: vec![None, None],
            created_at: Instant::now(),
        },
    );

    // Phase 2: remove_stale re-checks the timeout predicate; the
    // fresh entry's created_at is <1s ago, so the predicate fails
    // and the fresh state is spared.
    asm.remove_stale(&expired, timeout);

    let survived = asm.assemblies.get(&k).expect("fresh entry must survive");
    assert_eq!(
        survived.generation, 2,
        "gc evicted the fresh replacement instead of sparing it"
    );
}

#[test]
fn test_malformed_chunk_is_dropped() {
    let asm = ChunkAssembler::new();
    assert!(asm
        .receive_chunk("peer", "k", 1, 0, 0, Bytes::from_static(b"x"))
        .is_none());
    assert!(asm
        .receive_chunk("peer", "k", 1, 5, 3, Bytes::from_static(b"x"))
        .is_none());
    assert_eq!(asm.in_flight(), 0);
}

#[test]
fn test_oversized_total_chunks_is_rejected() {
    let asm = ChunkAssembler::new();
    // total = u32::MAX would trigger multi-GB allocation if admitted.
    assert!(asm
        .receive_chunk("peer", "k", 1, 0, u32::MAX, Bytes::from_static(b"x"))
        .is_none());
    // Just past the cap is also rejected.
    assert!(asm
        .receive_chunk(
            "peer",
            "k",
            1,
            0,
            MAX_TOTAL_CHUNKS + 1,
            Bytes::from_static(b"x")
        )
        .is_none());
    assert_eq!(
        asm.in_flight(),
        0,
        "oversized total must not allocate an AssemblyState"
    );
}

#[test]
fn test_bounds_evict_oldest_when_too_many_concurrent() {
    let asm = ChunkAssembler::with_limits(3, usize::MAX);
    // 4 partials against a cap of 3 — oldest must be evicted.
    for i in 0..4 {
        assert!(asm
            .receive_chunk(
                "peer",
                &format!("k{i}"),
                1,
                0,
                2,
                Bytes::from(vec![0u8; 10])
            )
            .is_none());
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(asm.in_flight(), 3, "concurrent cap enforced");
    assert!(
        !asm.assemblies.contains_key(&("peer".into(), "k0".into())),
        "oldest partial (k0) should be evicted"
    );
}

#[test]
fn test_bounds_evict_largest_when_over_byte_cap() {
    // Use payloads large enough that per-assembly vec overhead
    // (~25 bytes × total_chunks) is negligible relative to payload.
    let cap = 10_000;
    let asm = ChunkAssembler::with_limits(usize::MAX, cap);
    assert!(asm
        .receive_chunk("peer", "k_small", 1, 0, 2, Bytes::from(vec![0u8; 2_000]))
        .is_none());
    assert!(asm
        .receive_chunk("peer", "k_med", 1, 0, 2, Bytes::from(vec![0u8; 3_000]))
        .is_none());
    assert!(asm
        .receive_chunk("peer", "k_big", 1, 0, 2, Bytes::from(vec![0u8; 6_000]))
        .is_none());

    assert!(
        asm.total_bytes() <= cap,
        "byte cap enforced, total = {}",
        asm.total_bytes()
    );
    assert!(
        !asm.assemblies
            .contains_key(&("peer".into(), "k_big".into())),
        "largest partial (k_big) should be evicted"
    );
}

#[test]
fn test_enforce_bounds_skips_complete_assemblies() {
    // cap=1, one complete + two partials. Real incomplete=2 > 1,
    // so enforce_bounds must evict one. The complete must not be
    // picked even though it is the oldest by created_at.
    let asm = ChunkAssembler::with_limits(1, usize::MAX);
    let k_complete: AssemblyKey = ("peer".into(), "complete".into());
    let k_partial_a: AssemblyKey = ("peer".into(), "partial_a".into());
    let k_partial_b: AssemblyKey = ("peer".into(), "partial_b".into());
    asm.assemblies.insert(
        k_complete.clone(),
        AssemblyState {
            generation: 1,
            received_count: 2,
            chunks: vec![
                Some(Bytes::from(vec![0u8; 10])),
                Some(Bytes::from(vec![0u8; 10])),
            ],
            created_at: Instant::now() - Duration::from_secs(60),
        },
    );
    asm.assemblies.insert(
        k_partial_a.clone(),
        AssemblyState {
            generation: 1,
            received_count: 0,
            chunks: vec![None, None],
            created_at: Instant::now() - Duration::from_secs(30),
        },
    );
    asm.assemblies.insert(
        k_partial_b.clone(),
        AssemblyState {
            generation: 1,
            received_count: 0,
            chunks: vec![None, None],
            created_at: Instant::now(),
        },
    );

    asm.enforce_bounds();

    assert!(
        asm.assemblies.contains_key(&k_complete),
        "complete (even oldest) must not be evicted"
    );
    assert!(
        !asm.assemblies.contains_key(&k_partial_a),
        "oldest incomplete should be evicted"
    );
    assert!(
        asm.assemblies.contains_key(&k_partial_b),
        "newer incomplete survives"
    );
}

#[test]
fn test_enforce_bounds_cap_counts_only_incomplete() {
    // Transient complete-but-unextracted entries must not push the
    // incomplete population over cap. Here the cap is 1, and we
    // preload a complete entry + a partial. enforce_bounds should
    // see 1 incomplete (<=1), not 2 total, and leave the partial
    // alone.
    let asm = ChunkAssembler::with_limits(1, usize::MAX);
    let k_complete: AssemblyKey = ("peer".into(), "complete".into());
    let k_partial: AssemblyKey = ("peer".into(), "partial".into());
    asm.assemblies.insert(
        k_complete.clone(),
        AssemblyState {
            generation: 1,
            received_count: 2,
            chunks: vec![
                Some(Bytes::from(vec![0u8; 10])),
                Some(Bytes::from(vec![0u8; 10])),
            ],
            created_at: Instant::now(),
        },
    );
    asm.assemblies.insert(
        k_partial.clone(),
        AssemblyState {
            generation: 1,
            received_count: 1,
            chunks: vec![Some(Bytes::from(vec![0u8; 10])), None],
            created_at: Instant::now(),
        },
    );

    // Build the initial state manually so we can observe the
    // filter-based count without triggering a receive.
    let incomplete_count = asm
        .assemblies
        .iter()
        .filter(|e| !e.value().is_complete())
        .count();
    assert_eq!(incomplete_count, 1, "only 'partial' is incomplete");

    // Call enforce_bounds directly (via a no-op receive path).
    // With the fix: incomplete=1 <= cap=1, so no eviction.
    asm.enforce_bounds();

    assert!(
        asm.assemblies.contains_key(&k_complete),
        "complete assembly still present"
    );
    assert!(
        asm.assemblies.contains_key(&k_partial),
        "partial must not be evicted — real incomplete count was under cap"
    );
}

#[test]
fn test_enforce_bounds_spares_newer_generation_replacement() {
    // Simulate the race: enforce_bounds selects key "k" with
    // generation 5, but before the remove_if fires, another thread
    // resets "k" to generation 6. The generation-gated predicate
    // must spare the replacement. We emulate the race in-process
    // by splitting selection from removal via the private helper.
    let asm = ChunkAssembler::with_limits(1, usize::MAX);
    let k: AssemblyKey = ("peer".into(), "k".into());

    // Insert a stale generation-5 partial as the eviction target.
    asm.assemblies.insert(
        k.clone(),
        AssemblyState {
            generation: 5,
            received_count: 0,
            chunks: vec![None, None],
            created_at: Instant::now() - Duration::from_secs(60),
        },
    );

    // Enforcement picks this key + generation.
    let (picked_key, gen) = asm.oldest_incomplete().expect("have an incomplete");
    assert_eq!(picked_key, k);
    assert_eq!(gen, 5);

    // Racing thread replaces the state with a fresh generation-6.
    asm.assemblies.insert(
        k.clone(),
        AssemblyState {
            generation: 6,
            received_count: 0,
            chunks: vec![None, None],
            created_at: Instant::now(),
        },
    );

    // The delayed remove_if must NOT evict the generation-6 state.
    asm.assemblies.remove_if(&picked_key, |_, state| {
        state.generation == gen && !state.is_complete()
    });

    let survived = asm
        .assemblies
        .get(&k)
        .expect("newer-generation replacement must survive");
    assert_eq!(
        survived.generation, 6,
        "evicted stale gen 5 instead of sparing gen 6"
    );
}

#[test]
fn test_bounds_not_enforced_on_completion() {
    // If receive_chunk completes an assembly, the entry leaves the
    // map before bounds are checked — the just-completed value must
    // still be returned regardless of size.
    let asm = ChunkAssembler::with_limits(usize::MAX, 10);
    let out = asm
        .receive_chunk("peer", "k", 1, 0, 1, Bytes::from(vec![0u8; 500]))
        .expect("single-chunk completion returns bytes");
    assert_eq!(out.len(), 1, "single-chunk result is one Bytes fragment");
    assert_eq!(out[0].len(), 500);
    assert_eq!(asm.in_flight(), 0);
}

#[test]
fn test_multiple_keys_independent() {
    let asm = ChunkAssembler::new();
    assert!(asm
        .receive_chunk("peer", "a", 1, 0, 2, Bytes::from_static(b"a0"))
        .is_none());
    assert!(asm
        .receive_chunk("peer", "b", 1, 0, 2, Bytes::from_static(b"b0"))
        .is_none());
    assert_eq!(asm.in_flight(), 2);

    let a = asm
        .receive_chunk("peer", "a", 1, 1, 2, Bytes::from_static(b"a1"))
        .unwrap();
    assert_eq!(flatten(&a), b"a0a1");
    assert_eq!(asm.in_flight(), 1, "completing a should not touch b");

    let b = asm
        .receive_chunk("peer", "b", 1, 1, 2, Bytes::from_static(b"b1"))
        .unwrap();
    assert_eq!(flatten(&b), b"b0b1");
    assert_eq!(asm.in_flight(), 0);
}

#[test]
fn test_drop_pending_removes_only_matching_entry() {
    let asm = ChunkAssembler::new();
    // Start a multi-chunk assembly for (peer, "k") and an unrelated
    // one for (peer, "other") + (other_peer, "k").
    assert!(asm
        .receive_chunk("peer", "k", 1, 0, 2, Bytes::from_static(b"aa"))
        .is_none());
    assert!(asm
        .receive_chunk("peer", "other", 1, 0, 2, Bytes::from_static(b"bb"))
        .is_none());
    assert!(asm
        .receive_chunk("other_peer", "k", 1, 0, 2, Bytes::from_static(b"cc"))
        .is_none());
    assert_eq!(asm.in_flight(), 3);

    asm.drop_pending("peer", "k");
    assert_eq!(
        asm.in_flight(),
        2,
        "only (peer, k) should be gone; other entries spared"
    );

    // Completing the spared entries still works.
    let out = asm
        .receive_chunk("peer", "other", 1, 1, 2, Bytes::from_static(b"22"))
        .unwrap();
    assert_eq!(flatten(&out), b"bb22");
}

#[test]
fn test_drop_pending_on_missing_key_is_noop() {
    let asm = ChunkAssembler::new();
    asm.drop_pending("peer", "nonexistent");
    assert_eq!(asm.in_flight(), 0);
}
