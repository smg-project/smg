use super::*;

#[test]
fn next_generation_is_strictly_monotonic() {
    let a = next_generation();
    let b = next_generation();
    let c = next_generation();
    assert!(a < b && b < c, "{a} < {b} < {c}");
}

#[test]
fn test_single_chunk_fast_path() {
    let value = Bytes::from(vec![0u8; 100]);
    let entries = chunk_value("key".into(), 42, value.clone(), 1024);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].key, "key");
    assert_eq!(entries[0].generation, 42);
    assert_eq!(entries[0].chunk_index, 0);
    assert_eq!(entries[0].total_chunks, 1);
    assert_eq!(entries[0].data, value.to_vec());
}

#[test]
fn test_empty_value_is_single_chunk() {
    let entries = chunk_value("k".into(), 1, Bytes::new(), 1024);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].total_chunks, 1);
    assert!(entries[0].data.is_empty());
}

#[test]
fn test_multi_chunk_split_even_boundary() {
    // 3 × 10 bytes = 30 bytes total, max_chunk_bytes = 10 → 3 chunks
    let value = Bytes::from((0u8..30).collect::<Vec<_>>());
    let entries = chunk_value("k".into(), 7, value.clone(), 10);
    assert_eq!(entries.len(), 3);
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(e.chunk_index, i as u32);
        assert_eq!(e.total_chunks, 3);
        assert_eq!(e.generation, 7);
    }
    let reassembled: Vec<u8> = entries.iter().flat_map(|e| e.data.clone()).collect();
    assert_eq!(reassembled, value.to_vec());
}

#[test]
fn test_multi_chunk_split_uneven_boundary() {
    // 25 bytes, max_chunk_bytes = 10 → 3 chunks (10+10+5)
    let value = Bytes::from((0u8..25).collect::<Vec<_>>());
    let entries = chunk_value("k".into(), 1, value.clone(), 10);
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].data.len(), 10);
    assert_eq!(entries[1].data.len(), 10);
    assert_eq!(entries[2].data.len(), 5);
    let reassembled: Vec<u8> = entries.iter().flat_map(|e| e.data.clone()).collect();
    assert_eq!(reassembled, value.to_vec());
}

#[test]
fn test_chunk_at_exact_boundary() {
    // Value exactly equals max_chunk_bytes → single chunk (fast path).
    let value = Bytes::from(vec![0u8; 10]);
    let entries = chunk_value("k".into(), 1, value, 10);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].total_chunks, 1);
}

#[test]
fn test_one_byte_over_boundary_splits() {
    let value = Bytes::from(vec![0u8; 11]);
    let entries = chunk_value("k".into(), 1, value, 10);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].data.len(), 10);
    assert_eq!(entries[1].data.len(), 1);
}

#[test]
#[should_panic(expected = "max_chunk_bytes must be non-zero")]
fn test_zero_max_chunk_bytes_panics() {
    let _ = chunk_value("k".into(), 1, Bytes::from_static(b"x"), 0);
}

#[test]
fn test_tree_page_fast_path() {
    // tree:page:* stays under max_chunk_bytes by contract (2 MB vs 10 MB),
    // so it takes the single-chunk path and the debug_assert does not fire.
    let value = Bytes::from(vec![0u8; 1_000_000]);
    let entries = chunk_value("tree:page:model1:0".into(), 1, value, 10 * 1024 * 1024);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].total_chunks, 1);
}

#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "tree:page:*"))]
fn test_tree_page_split_panics_in_debug() {
    // If a tree page ever grows past max_chunk_bytes, debug build
    // catches it. Release build would just emit the split (valid
    // semantically, but violates the tree-sync contract).
    let value = Bytes::from(vec![0u8; 20]);
    let _ = chunk_value("tree:page:model1:0".into(), 1, value, 10);
}

#[test]
fn test_build_stream_batches_empty() {
    assert!(build_stream_batches(vec![], 5, 1024).is_empty());
}

#[test]
fn test_build_stream_batches_under_cap() {
    let entries = vec![
        StreamEntry {
            key: "a".into(),
            generation: 1,
            chunk_index: 0,
            total_chunks: 1,
            data: Bytes::from_static(&[1]),
        },
        StreamEntry {
            key: "b".into(),
            generation: 1,
            chunk_index: 0,
            total_chunks: 1,
            data: Bytes::from_static(&[2]),
        },
    ];
    let batches = build_stream_batches(entries, 5, 1024);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].entries.len(), 2);
}

#[test]
fn test_build_stream_batches_packs_over_cap() {
    let entries: Vec<StreamEntry> = (0..10)
        .map(|i| StreamEntry {
            key: format!("k{i}"),
            generation: 1,
            chunk_index: 0,
            total_chunks: 1,
            data: Bytes::copy_from_slice(&[i as u8]),
        })
        .collect();
    // 10 entries, cap=3 → 3 + 3 + 3 + 1 = 4 batches. No entry dropped.
    let batches = build_stream_batches(entries, 3, 1024);
    assert_eq!(batches.len(), 4);
    assert_eq!(batches[0].entries.len(), 3);
    assert_eq!(batches[1].entries.len(), 3);
    assert_eq!(batches[2].entries.len(), 3);
    assert_eq!(batches[3].entries.len(), 1);
    let total: usize = batches.iter().map(|b| b.entries.len()).sum();
    assert_eq!(total, 10, "every entry must be emitted (no truncation)");
}

#[test]
fn test_build_stream_batches_bytes_cap_splits_before_count_cap() {
    // 4 entries × 100 bytes = 400 bytes. Count cap = 10 (not hit).
    // Bytes cap = 250 → first batch holds 2 entries (200 ≤ 250;
    // adding a 3rd would push to 300 > 250), second holds the rest.
    let entries: Vec<StreamEntry> = (0..4)
        .map(|i| StreamEntry {
            key: format!("k{i}"),
            generation: 1,
            chunk_index: 0,
            total_chunks: 1,
            data: Bytes::from(vec![0u8; 100]),
        })
        .collect();
    let batches = build_stream_batches(entries, 10, 250);
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].entries.len(), 2);
    assert_eq!(batches[1].entries.len(), 2);
}

#[test]
fn test_build_stream_batches_oversized_entry_still_emitted() {
    // Single entry above the byte cap must still ship (at-most-once;
    // dropping would strand the value undeliverable). Emits as a
    // single-entry batch even though it exceeds max_bytes_per_batch.
    let entries = vec![StreamEntry {
        key: "big".into(),
        generation: 1,
        chunk_index: 0,
        total_chunks: 1,
        data: Bytes::from(vec![0u8; 500]),
    }];
    let batches = build_stream_batches(entries, 10, 100);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].entries.len(), 1);
    assert_eq!(batches[0].entries[0].data.len(), 500);
}

#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "max_chunks_per_batch"))]
fn test_build_stream_batches_zero_cap() {
    // Debug build: assert fires (contract violation).
    // Release build: runtime zero-check still returns empty as a safety net.
    let entries = vec![StreamEntry {
        key: "k".into(),
        generation: 1,
        chunk_index: 0,
        total_chunks: 1,
        data: Bytes::from_static(&[1]),
    }];
    let out = build_stream_batches(entries, 0, 1024);
    assert!(out.is_empty());
}

#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "max_bytes_per_batch"))]
fn test_build_stream_batches_zero_bytes_cap() {
    let entries = vec![StreamEntry {
        key: "k".into(),
        generation: 1,
        chunk_index: 0,
        total_chunks: 1,
        data: Bytes::from_static(&[1]),
    }];
    let out = build_stream_batches(entries, 5, 0);
    assert!(out.is_empty());
}
