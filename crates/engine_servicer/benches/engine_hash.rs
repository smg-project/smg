//! What one block costs the KV relay's engine-hash check: the engines' block
//! hashes reproduced for one full block, chained on a parent and at a chain's
//! head, so a change to the encoding or the hashing reads in nanoseconds per
//! block beside its allocation pin.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use engine_servicer::engine_hash::{sglang_page, vllm_block};

/// A full block of a common size, token ids in the two-byte range.
const BLOCK_SIZE: usize = 16;

fn benches(c: &mut Criterion) {
    let tokens: Vec<u32> = (1_000..1_000 + BLOCK_SIZE as u32).collect();
    let parent = vllm_block(None, &tokens, None);
    let mut group = c.benchmark_group("engine_hash");
    group.throughput(Throughput::Elements(1));
    group.bench_function("vllm_block_chained", |b| {
        b.iter(|| vllm_block(black_box(Some(&parent)), black_box(&tokens), None));
    });
    group.bench_function("vllm_block_chain_head", |b| {
        b.iter(|| vllm_block(black_box(None), black_box(&tokens), None));
    });
    group.bench_function("sglang_page_chained", |b| {
        b.iter(|| sglang_page(black_box(Some(&parent)), black_box(&tokens)));
    });
    group.finish();
}

criterion_group!(hashes, benches);
criterion_main!(hashes);
