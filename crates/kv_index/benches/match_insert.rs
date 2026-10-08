//! Per-request cost of the cache-aware trees: the fused match-and-insert the
//! gateway runs once per routed request, plus the event-driven path's request
//! hashing and size check.
//!
//! Workload: 64 tenants (worker URLs) sharing a 512-token (2 KiB) system
//! prompt, each with its own 64-token (256-char) user turn. "hit" replays a
//! known prompt and routes it to its matched tenant, the steady state of a
//! warm tree; "miss" appends a never-seen turn each time; "hit_8_threads"
//! runs the hit path on eight threads against one tree and reports wall time
//! divided by operations.
#![expect(clippy::expect_used)]

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use kv_index::{
    compute_content_hash, compute_request_content_hashes, request_prefix_hashes, ChainBlockMap,
    ChainIndex, ContentHash, PositionalIndexer, SequenceHash, ShardedChainIndex, StoredBlock,
    TokenTree, Tree, WorkerBlockMap,
};

const TENANTS: usize = 64;
const SYSTEM_TOKENS: usize = 512;
const TURN_TOKENS: usize = 64;
const SYSTEM_CHARS: usize = 2048;
const TURN_CHARS: usize = 256;
const THREADS: usize = 8;

fn tenant(i: usize) -> String {
    format!("http://worker-{i:03}.inference.svc.cluster.local:8000")
}

fn mix(seed: u64) -> u64 {
    // splitmix64 step: cheap, deterministic, well spread.
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn token_prompt(turn: u64) -> Vec<u32> {
    let mut tokens: Vec<u32> = (0..SYSTEM_TOKENS as u64)
        .map(|i| (mix(i) % 150_000) as u32)
        .collect();
    tokens.extend((0..TURN_TOKENS as u64).map(|i| (mix(turn << 20 | i) % 150_000) as u32));
    tokens
}

fn text_prompt(turn: u64) -> String {
    const WORDS: [&str; 16] = [
        "the", "model", "answers", "with", "care", "and", "cites", "sources", "when", "asked",
        "about", "numbers", "dates", "names", "or", "places",
    ];
    let mut text = String::with_capacity(SYSTEM_CHARS + TURN_CHARS + 16);
    let mut i = 0u64;
    while text.len() < SYSTEM_CHARS {
        text.push_str(WORDS[(mix(i) % 16) as usize]);
        text.push(' ');
        i += 1;
    }
    text.push_str("\nUser: ");
    let mut j = 0u64;
    while text.len() < SYSTEM_CHARS + TURN_CHARS {
        text.push_str(WORDS[(mix(turn << 20 | j) % 16) as usize]);
        text.push(' ');
        j += 1;
    }
    text
}

fn warm_token_tree() -> (Arc<TokenTree>, Vec<String>, Vec<Vec<u32>>) {
    let tree = Arc::new(TokenTree::new());
    let tenants: Vec<String> = (0..TENANTS).map(tenant).collect();
    let prompts: Vec<Vec<u32>> = (0..TENANTS as u64).map(token_prompt).collect();
    for (prompt, tenant) in prompts.iter().zip(&tenants) {
        tree.insert_tokens(prompt, tenant);
    }
    (tree, tenants, prompts)
}

fn warm_string_tree() -> (Arc<Tree>, Vec<String>, Vec<String>) {
    let tree = Arc::new(Tree::new());
    let tenants: Vec<String> = (0..TENANTS).map(tenant).collect();
    let prompts: Vec<String> = (0..TENANTS as u64).map(text_prompt).collect();
    for (prompt, tenant) in prompts.iter().zip(&tenants) {
        tree.insert_text(prompt, tenant);
    }
    (tree, tenants, prompts)
}

/// Bench-owned tenant names by name, so the select closure can hand back the
/// matched tenant (the common hit outcome) with one lookup.
fn by_name(tenants: &[String]) -> HashMap<&str, &str> {
    tenants.iter().map(|t| (t.as_str(), t.as_str())).collect()
}

/// Rebuilt from scratch every `MISS_BATCH` misses (in the untimed setup), so
/// the miss path measures a tree of bounded, repeatable size.
const MISS_BATCH: u64 = 256;

fn bench_token_tree(c: &mut Criterion) {
    let (tree, tenants, prompts) = warm_token_tree();
    let names = by_name(&tenants);
    let mut group = c.benchmark_group("token_tree/match_and_insert");
    group.throughput(Throughput::Elements(1));

    let mut i = 0usize;
    group.bench_function("hit", |b| {
        b.iter(|| {
            let prompt = &prompts[i % TENANTS];
            i += 1;
            let result =
                tree.match_and_insert_with(prompt, |r| names.get(r.tenant.as_ref()).copied());
            assert!(result.matched_token_count >= SYSTEM_TOKENS);
            result.matched_token_count
        });
    });

    let miss_tree = RefCell::new(warm_token_tree().0);
    let mut turn = 1_000_000u64;
    group.bench_function("miss", |b| {
        b.iter_batched(
            || {
                turn += 1;
                if turn.is_multiple_of(MISS_BATCH) {
                    *miss_tree.borrow_mut() = warm_token_tree().0;
                }
                (token_prompt(turn), (turn % TENANTS as u64) as usize)
            },
            |(prompt, t)| {
                miss_tree
                    .borrow()
                    .match_and_insert_with(&prompt, |_| Some(tenants[t].as_str()))
                    .matched_token_count
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("hit_8_threads", |b| {
        b.iter_custom(|iters| {
            let per_thread = iters.div_ceil(THREADS as u64);
            let next = Arc::new(AtomicUsize::new(0));
            let start = Instant::now();
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let tree = Arc::clone(&tree);
                    let next = Arc::clone(&next);
                    let (names, prompts) = (&names, &prompts);
                    s.spawn(move || {
                        for _ in 0..per_thread {
                            let k = next.fetch_add(1, Ordering::Relaxed) % TENANTS;
                            tree.match_and_insert_with(&prompts[k], |r| {
                                names.get(r.tenant.as_ref()).copied()
                            });
                        }
                    });
                }
            });
            let elapsed = start.elapsed();
            // Normalise to the requested iteration count.
            Duration::from_secs_f64(
                elapsed.as_secs_f64() * iters as f64 / (per_thread * THREADS as u64) as f64,
            )
        });
    });
    group.finish();
}

fn bench_string_tree(c: &mut Criterion) {
    let (tree, tenants, prompts) = warm_string_tree();
    let names = by_name(&tenants);
    let mut group = c.benchmark_group("string_tree/match_and_insert");
    group.throughput(Throughput::Elements(1));

    let mut i = 0usize;
    group.bench_function("hit", |b| {
        b.iter(|| {
            let prompt = &prompts[i % TENANTS];
            i += 1;
            let result =
                tree.match_and_insert_with(prompt, |r| names.get(r.tenant.as_ref()).copied());
            assert!(result.matched_char_count >= SYSTEM_CHARS);
            result.matched_char_count
        });
    });

    let miss_tree = RefCell::new(warm_string_tree().0);
    let mut turn = 1_000_000u64;
    group.bench_function("miss", |b| {
        b.iter_batched(
            || {
                turn += 1;
                if turn.is_multiple_of(MISS_BATCH) {
                    *miss_tree.borrow_mut() = warm_string_tree().0;
                }
                (text_prompt(turn), (turn % TENANTS as u64) as usize)
            },
            |(prompt, t)| {
                miss_tree
                    .borrow()
                    .match_and_insert_with(&prompt, |_| Some(tenants[t].as_str()))
                    .matched_char_count
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("hit_8_threads", |b| {
        b.iter_custom(|iters| {
            let per_thread = iters.div_ceil(THREADS as u64);
            let next = Arc::new(AtomicUsize::new(0));
            let start = Instant::now();
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let tree = Arc::clone(&tree);
                    let next = Arc::clone(&next);
                    let (names, prompts) = (&names, &prompts);
                    s.spawn(move || {
                        for _ in 0..per_thread {
                            let k = next.fetch_add(1, Ordering::Relaxed) % TENANTS;
                            tree.match_and_insert_with(&prompts[k], |r| {
                                names.get(r.tenant.as_ref()).copied()
                            });
                        }
                    });
                }
            });
            let elapsed = start.elapsed();
            Duration::from_secs_f64(
                elapsed.as_secs_f64() * iters as f64 / (per_thread * THREADS as u64) as f64,
            )
        });
    });
    group.finish();
}

fn bench_event_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_path");
    let prompt = token_prompt(7);
    let prompt_2048: Vec<u32> = (0..2048u64).map(|i| (mix(i) % 150_000) as u32).collect();

    group.throughput(Throughput::Elements((prompt.len() / 16) as u64));
    group.bench_function("content_hashes/576_tokens/block_16", |b| {
        b.iter(|| compute_request_content_hashes(&prompt, 16).len());
    });
    group.throughput(Throughput::Elements((prompt_2048.len() / 16) as u64));
    group.bench_function("content_hashes/2048_tokens/block_16", |b| {
        b.iter(|| compute_request_content_hashes(&prompt_2048, 16).len());
    });

    // An indexer with 64 workers each holding 32 blocks: the size check the
    // policy runs per request before taking the event-driven path.
    let indexer = PositionalIndexer::new(64);
    for w in 0..TENANTS {
        let worker_id = indexer.intern_worker(&tenant(w)).expect("worker id");
        let mut worker_blocks = WorkerBlockMap::default();
        let blocks: Vec<StoredBlock> = (0..32usize)
            .map(|p| StoredBlock {
                seq_hash: SequenceHash::from(mix((w as u64) << 8 | p as u64)),
                content_hash: compute_content_hash(&prompt_2048[p * 16..(p + 1) * 16]),
            })
            .collect();
        indexer
            .apply_stored(worker_id, &blocks, None, &mut worker_blocks)
            .expect("store blocks");
    }
    group.throughput(Throughput::Elements(1));
    group.bench_function("indexer_current_size/64_workers", |b| {
        b.iter(|| indexer.current_size());
    });
    group.finish();
}

/// Blocks of a content chain as an engine hashes them (the engine hash is the chain hash).
fn chain_blocks(stream: u64, len: usize) -> Vec<StoredBlock> {
    let contents: Vec<ContentHash> = (0..len)
        .map(|p| compute_content_hash(&[stream as u32, (stream >> 32) as u32, p as u32]))
        .collect();
    contents
        .iter()
        .zip(request_prefix_hashes(&contents))
        .map(|(&content_hash, seq_hash)| StoredBlock {
            seq_hash,
            content_hash,
        })
        .collect()
}

/// A chain index in which `holders` workers hold one 88-block chain: the shape of the Mooncake
/// replay, where nineteen of twenty stores land on a run another worker built.
fn shared_run(holders: usize) -> (ChainIndex, Vec<StoredBlock>) {
    let index = ChainIndex::with_max_workers(64);
    let blocks = chain_blocks(1, 88);
    for w in 0..holders {
        let worker = index.intern_worker(&tenant(w)).expect("worker id");
        let mut map = ChainBlockMap::default();
        index
            .apply_stored(worker, &blocks, None, &mut map)
            .expect("store");
    }
    (index, blocks)
}

/// The store walk of the chain index per event block: a worker storing a chain other workers
/// already hold (the walk matches the run and writes the lane map), a store that diverges from
/// the shared chain halfway (the walk finds the divergence and splits the run), and a fresh
/// chain under the root (a new run, allocation included) as the control.
fn bench_chain_index_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_index_store");
    group.throughput(Throughput::Elements(88));

    let (index, blocks) = shared_run(19);
    let worker = index.intern_worker("newcomer").expect("worker id");
    let hashes: Vec<SequenceHash> = blocks.iter().map(|block| block.seq_hash).collect();
    group.bench_function("shared_chain/88_blocks/19_holders", |b| {
        b.iter_batched(
            || {
                let mut map = ChainBlockMap::default();
                index
                    .apply_stored(worker, &blocks, None, &mut map)
                    .expect("store");
                index.apply_removed(worker, &hashes, &mut map);
                map
            },
            |mut map| {
                index
                    .apply_stored(worker, &blocks, None, &mut map)
                    .expect("store");
                map
            },
            BatchSize::SmallInput,
        );
    });

    let mut forked = blocks[..44].to_vec();
    forked.extend(chain_blocks(2, 88).into_iter().skip(44));
    let forked = {
        // Re-chain the fork's engine hashes from the shared prefix.
        let contents: Vec<ContentHash> = forked.iter().map(|block| block.content_hash).collect();
        contents
            .iter()
            .zip(request_prefix_hashes(&contents))
            .map(|(&content_hash, seq_hash)| StoredBlock {
                seq_hash,
                content_hash,
            })
            .collect::<Vec<_>>()
    };
    group.bench_function("divergent_chain/88_blocks/split_at_44", |b| {
        b.iter_batched(
            || {
                let (index, _) = shared_run(19);
                let worker = index.intern_worker("forker").expect("worker id");
                (index, worker)
            },
            |(index, worker)| {
                let mut map = ChainBlockMap::default();
                index
                    .apply_stored(worker, &forked, None, &mut map)
                    .expect("store");
                (index, map)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("fresh_chain/88_blocks", |b| {
        b.iter_batched(
            || {
                let index = ChainIndex::with_max_workers(64);
                let worker = index.intern_worker("first").expect("worker id");
                (index, worker)
            },
            |(index, worker)| {
                let mut map = ChainBlockMap::default();
                index
                    .apply_stored(worker, &blocks, None, &mut map)
                    .expect("store");
                (index, map)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// The lookup of the chain index: a request of 88 blocks that twenty workers hold whole (the
/// Mooncake shape, one run walked), and the same request against two shards holding it on both.
fn bench_chain_index_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_index_lookup");
    group.throughput(Throughput::Elements(1));
    let (index, blocks) = shared_run(20);
    let request: Vec<ContentHash> = blocks.iter().map(|block| block.content_hash).collect();
    group.bench_function("88_blocks/20_holders/1_shard", |b| {
        b.iter(|| {
            let mut scored = 0usize;
            index.score_into(&request, |content| content.0, false, |_, _| scored += 1);
            scored
        });
    });
    let sharded = ShardedChainIndex::new(2, 64);
    for w in 0..20 {
        let worker = sharded
            .intern_worker_in(w % 2, &tenant(w))
            .expect("worker id");
        let mut map = ChainBlockMap::default();
        sharded
            .apply_stored(worker, &blocks, None, &mut map)
            .expect("store");
    }
    group.bench_function("88_blocks/20_holders/2_shards", |b| {
        b.iter(|| {
            let mut scored = 0usize;
            sharded.score_into(&request, |content| content.0, false, |_, _| scored += 1);
            scored
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_token_tree,
    bench_string_tree,
    bench_event_path,
    bench_chain_index_store,
    bench_chain_index_lookup
);
criterion_main!(benches);
