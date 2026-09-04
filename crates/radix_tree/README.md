# radix-tree

A generic prefix-membership index: given per-holder chains of
content-addressed blocks, answer *"which holders already hold the
longest prefix of this chain, and how deep?"* — plus the write and
lifecycle operations a long-lived, multi-tenant index service needs
as first-class API. Zero SMG dependencies; everything it sees is a
hash.

Built as the ground-up replacement for the radix-index service's
import of the gateway's `kv_index::PositionalIndexer`; the service
(`crates/radix_index`) runs on it.

## API in one glance

```rust
let mut tree = RadixTree::new(Config::default());
let w = tree.create_holder("worker-7");            // generational id
tree.store(w, None, &[(key, content), ...])?;      // anchor a chain
tree.store(w, Some(parent_key), &more)?;           // extend it
tree.remove(w, &[key]);                            // event-feed eviction
tree.truncate_tail(w, keep);                       // prefix-closed capacity cut
tree.clear(w);                                     // epoch bump
tree.retire_holder(w);                             // frees everything, id recycled

let mut scratch = OverlapScratch::default();
let mut out = Vec::new();
tree.overlap(&chain_hashes, &mut scratch, &mut out);
// out: [{ holder, depth, total_blocks }]
tree.enumerate(w);                                 // (pos, key, content) for snapshots
```

The contract — exact matching semantics, convergence scope, and the
§4 alias rules — is what `tests/differential.rs` enforces: every core
must equal the reference model on every run.

## Structure

Prefixes form a trie of **chains**. A chain's contents are one
contiguous array, stored once no matter how many holders cover it;
membership is a short list of maximal runs pointing at interned
(hash-consed) holder sets; a query is one hash probe to the root
chain, a linear scan of contiguous contents to the divergence point,
a child-fork hop if needed, and a handful of span reads. Matching is
exact and content-verified — fingerprint collisions cannot
cross-credit.

`FlatTree` is the first-generation flat layout, kept as a second,
independently verified implementation: the test harness asserts BOTH
cores equal the reference model on every run.

## Verification

The crate was built harness-first:

- `tests/differential.rs` — both cores vs a representationally
  complete reference model AND the production `kv_index` oracle
  (dev-dependency), with full-state `audit()` at every checkpoint.
- `tests/fuzz_differential.rs` — wide-config fuzz plus a CHAOS mode
  that violates every contract precondition under a no-panic,
  audit-green, model-equal, deterministic-replay contract.
  `RADIX_FUZZ_SEEDS=10000` ran green.
- `tests/api.rs` — lifecycle, boundary, and exactness cases the model
  deliberately doesn't express.
- `tests/alloc_gate.rs` — counting-allocator gate: single-holder
  stores amortize to zero allocations.
- `tests/pinned_bench.rs` — the normative performance workload
  (`RADIX_BENCH_SIDE=oracle|r1|r3`, `RADIX_BENCH_SCALE=large`,
  `RADIX_BENCH_SOAK_SECS=n`).

## Measured (pinned workload: 12.8M holder-blocks, 256 holders)

| | old `PositionalIndexer` + glue | `RadixTree` |
|---|---|---|
| bytes / holder-block | 166.7 | **26.9** |
| worst cell (d78, 64 holders) cold p99 | 6.0 µs (unsound skip) | **7.6 µs exact** |
| single-holder query p50 | 917 ns | **292 ns** |
| writes, mixed stream | 5.5M blocks/s | 4.6M blocks/s |
| at 128M blocks: worst-cell p99 | 29 µs | **8.0 µs** |

No timestamps live in the tree: freshness policy (idle TTL per
holder) belongs to the consumer; chain data is freed by reference
counting.
