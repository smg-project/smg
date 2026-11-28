//! What the forced call's variant of a table costs to make (`Format::with_forced_call`): the
//! table's copy and the walk over its rows. A gateway makes the variant once per table and per
//! way the call is written and keeps it, so a request pays a lookup; the making is setup, stated
//! here as `throughput`'s `setup` rows state the engine's. One subject per shape: a table that
//! reads a thought, whose thought states stay, and a table that reads none, which becomes the
//! arguments state alone.
//!
//! A bench target of its own, beside `throughput`, as `grammar` is and for its reason: that
//! binary's rows are gated (`.github/bench-gates/symphony.toml`), and its set of callers
//! decides how the crate's hot path is inlined in it.
//!
//! `cargo bench -p smg-symphony --bench forced`; one core, release: the row per subject is one
//! variant made of the table's copy.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use symphony::{formats::Table, ForcedCall};

/// The tables under measurement: one that reads a thought, one that reads none.
const SUBJECTS: &[Table] = &[Table::Qwen3, Table::Llama4];

/// The variant per subject for the function a request names, as a gateway makes it of the
/// shared table: the copy, then the variant.
fn variant(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup");
    for table in SUBJECTS {
        let format = table.format();
        group.bench_function(BenchmarkId::new("forced_variant", table.name()), |b| {
            b.iter(|| black_box(format.clone().with_forced_call(ForcedCall::Arguments)));
        });
    }
    group.finish();
}

criterion_group!(benches, variant);
criterion_main!(benches);
