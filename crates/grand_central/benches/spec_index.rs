//! One validation of a policy spec by its `index:` keys: the overlap check
//! reads each key once, so a row grows with the keys and not by their
//! square. Ids: `engine/spec/validate_index/<keys>` at a thousand,
//! four thousand and nine thousand keys. The last is a document under the
//! size a spec document may have, 128 KiB (`MAX_DOCUMENT_BYTES`, the bound of
//! every document a source hands the controller), at the 13 to 14 bytes a
//! key this one writes.
//!
//! A bench target of its own: no pull-request gate reads it, and a row
//! added to the gated `decision` binary would move that binary
//! (`.github/bench-gates/grand-central-engine.toml`).
//!
//! Under the feature `in-process`, one apply on a running engine whose
//! document gained one profile at each of its generations, as a supplied
//! spec may. Ids:
//! `engine/spec/apply_grown/<profiles>` at ten, a hundred and a thousand
//! profiles. A row is the check of the document and its apply: the engine
//! keeps every profile, each made under a generation of its own, and holds
//! nothing for an earlier generation.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a document that fails to parse is a bug the unit tests catch"
)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_grand_central::document::PolicySpec;

/// The size a spec document may have, in bytes: the controller's
/// `MAX_DOCUMENT_BYTES`, private to the crate, so written again here. A case
/// of the controller's generations pins the number by the refusal's words.
const FILE_BOUND: usize = 128 << 10;

/// A document of `entries` index keys, a `prefix*` and a `*suffix` by turns
/// of two literal lengths, so no two overlap; with the bytes it writes.
fn indexed(entries: usize) -> (PolicySpec, usize) {
    let keys: Vec<String> = (0..entries)
        .map(|n| {
            if n % 2 == 0 {
                format!(r#""p{n:05}*":{{}}"#)
            } else {
                format!(r#""*{n:05}ss":{{}}"#)
            }
        })
        .collect();
    let document = format!(
        r#"{{"apiVersion":"smg.policy/v1","index":{{{}}},"profiles":{{"p":{{"pick":"round-robin"}}}}}}"#,
        keys.join(",")
    );
    let spec = serde_json::from_str(&document).expect("the document parses");
    (spec, document.len())
}

fn spec_validate_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/spec");
    // Ten samples: a row is one loop over the keys, the same at every run.
    group.sample_size(10);
    for entries in [1_000, 4_000, 9_000] {
        let (spec, bytes) = indexed(entries);
        assert!(bytes <= FILE_BOUND, "{entries} keys write {bytes} bytes");
        group.bench_with_input(
            BenchmarkId::new("validate_index", entries),
            &entries,
            |b, _| {
                b.iter(|| black_box(&spec).validate().expect("the document validates"));
            },
        );
    }
    group.finish();
}

/// The `round_robin` preset's document under `generation`, with `profiles`
/// profiles in all: the preset's own and a copy of it for each further one,
/// bound to a model of its own.
#[cfg(feature = "in-process")]
fn grown(profiles: usize, generation: u64) -> PolicySpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset's name");
    document.generation = generation;
    let own = document.profiles["round_robin"].clone();
    let binding = document.bindings[0].clone();
    for at in 1..profiles {
        let name = format!("model-{at:04}");
        document.profiles.insert(name.clone(), own.clone());
        let mut bound = binding.clone();
        bound.matcher.model.clone_from(&name);
        bound.profile = Some(name);
        document.bindings.insert(0, bound);
    }
    document
}

/// One apply on an engine whose document gained one profile at each of
/// its generations. The apply keeps every profile, each made under a
/// generation of its own.
#[cfg(feature = "in-process")]
fn spec_apply_grown(c: &mut Criterion) {
    use smg_grand_central::{document::AppliedSpec, engine::InProcess};
    let mut group = c.benchmark_group("engine/spec");
    group.sample_size(20);
    for profiles in [10, 100, 1_000] {
        let first = AppliedSpec::validate(&grown(1, 1)).expect("the document validates");
        let engine = InProcess::new(first);
        let mut generation = 1;
        for count in 2..=profiles {
            generation += 1;
            let next = AppliedSpec::validate(&grown(count, generation));
            engine
                .apply(next.expect("the document validates"))
                .expect("a later generation");
        }
        let mut document = grown(profiles, generation);
        group.bench_with_input(
            BenchmarkId::new("apply_grown", profiles),
            &profiles,
            |b, _| {
                b.iter(|| {
                    document.generation += 1;
                    let next = AppliedSpec::validate(black_box(&document));
                    engine
                        .apply(next.expect("the document validates"))
                        .expect("a later generation");
                });
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "in-process"))]
fn spec_apply_grown(_: &mut Criterion) {}

criterion_group!(benches, spec_validate_index, spec_apply_grown);
criterion_main!(benches);
