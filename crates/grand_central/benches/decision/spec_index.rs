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
//! document grew to its size by one profile a generation, as a supplied
//! spec may. Ids:
//! `engine/spec/apply_grown/<profiles>` at ten, a hundred and a thousand
//! profiles. A row is the check of the document and its apply: the engine
//! keeps every profile, each made under a generation of its own, and holds
//! nothing for an earlier generation.
//!
//! Under the same feature, one apply of the routing settings' document
//! with labelled models, each a profile of its own, on an engine that took
//! the models one a generation, as a gateway registers them. Ids:
//! `engine/spec/apply_labelled/<models>` at ten, a hundred and a thousand
//! models. A row is the check of the document and its apply, which builds
//! each profile's stages and compares them with the standing ones.
//!
//! A row's engine is built where the row runs: a filter that leaves a row
//! out builds none of it.
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

/// The routing settings' document under `generation`, with `models`
/// labelled models, each under round robin, beside a main policy of
/// `random`.
#[cfg(feature = "in-process")]
fn labelled(models: usize, generation: u64) -> PolicySpec {
    use smg_grand_central::document::RoutingSettings;
    let labels = (0..models)
        .map(|n| (format!("model-{n:04}"), "round_robin".to_owned()))
        .collect();
    let settings = RoutingSettings::default();
    let document = settings.policy_document_with_labels(&labels);
    let mut document = document.expect("the labelled document builds");
    document.generation = generation;
    document
}

/// One apply on an engine whose document, made by `sized`, grew to `size`
/// by one a generation: under the row `id` of the group. The engine is
/// built inside the row. The apply keeps every profile, each made under a
/// generation of its own.
#[cfg(feature = "in-process")]
fn apply_row(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    id: &str,
    size: usize,
    sized: fn(usize, u64) -> PolicySpec,
) {
    use smg_grand_central::{document::AppliedSpec, engine::InProcess};
    group.bench_with_input(BenchmarkId::new(id, size), &size, |b, &size| {
        let first = AppliedSpec::validate(&sized(1, 1)).expect("the document validates");
        let engine = InProcess::new(first);
        let mut generation = 1;
        for count in 2..=size {
            generation += 1;
            let next = AppliedSpec::validate(&sized(count, generation));
            engine
                .apply(next.expect("the document validates"))
                .expect("a later generation");
        }
        let mut document = sized(size, generation);
        b.iter(|| {
            document.generation += 1;
            let next = AppliedSpec::validate(black_box(&document));
            engine
                .apply(next.expect("the document validates"))
                .expect("a later generation");
        });
    });
}

/// One apply on an engine whose document grew to its size by one profile
/// a generation: a supplied spec's, and the routing settings' with a
/// labelled model for each.
#[cfg(feature = "in-process")]
fn spec_apply_grown(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/spec");
    group.sample_size(20);
    for size in [10, 100, 1_000] {
        apply_row(&mut group, "apply_grown", size, grown);
        apply_row(&mut group, "apply_labelled", size, labelled);
    }
    group.finish();
}

#[cfg(not(feature = "in-process"))]
fn spec_apply_grown(_: &mut Criterion) {}

criterion_group!(benches, spec_validate_index, spec_apply_grown);
criterion_main!(benches);
