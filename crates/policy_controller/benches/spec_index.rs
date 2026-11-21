//! One validation of a policy spec by its `index:` keys: the overlap check
//! reads each key once, so a row grows with the keys and not by their
//! square. Ids: `policy_engine/spec/validate_index/<keys>` at a thousand,
//! four thousand and nine thousand keys. The last is a document under the
//! size a spec document may have, 128 KiB (`MAX_DOCUMENT_BYTES`, the bound of
//! every document a source hands the controller), at the 13 to 14 bytes a
//! key this one writes.
//!
//! A bench target of its own: no pull-request gate reads it, and a row
//! added to the gated `decision` binary would move that binary
//! (`.github/bench-gates/policy-engine.toml`).
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a document that fails to parse is a bug the unit tests catch"
)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::document::PolicySpec;

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
    let mut group = c.benchmark_group("policy_engine/spec");
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

criterion_group!(benches, spec_validate_index);
criterion_main!(benches);
