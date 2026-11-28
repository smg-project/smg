//! The WebAssembly world's side of the wire contracts. Each record of the
//! table below carries its crate record's fields in the crate's order, less
//! the host's fields, with what the table names before and beside them. A
//! field of the crate stands under the name the crate's list spells for the
//! contracts, its underscores written as hyphens. The packed request block
//! keeps its u64 sections 8-byte aligned, as the world's comment states.

use crate::{
    affinity_record, candidate_record, contracts_dir, enum_mirrors, health_record, kebab_case,
    lease_record, load_row_record, media_record, outcome_record, pair_record, parse_world,
    pool_record, request_record, signal_cases, strings, target_record, zone_record, WIT_NAMESPACE,
    WIT_PACKAGE, WIT_VERSION, WORLD_WIT,
};

/// How a record of the world's `types` interface relates to the crate's.
struct WitMirror {
    record: &'static str,
    crate_record: &'static [&'static str],
    host_only: &'static [&'static str],
    /// Fields the WIT record carries before the crate's (an event's lease).
    before: &'static [&'static str],
    /// The fields of a record the crate keeps beside this one and the WIT
    /// record carries inside it, in that record's order (a candidate's
    /// affinity facts).
    beside: &'static [&'static str],
}

fn wit_mirrors() -> Vec<WitMirror> {
    vec![
        WitMirror {
            record: "load",
            crate_record: load_row_record(),
            host_only: &["resident_counted"],
            before: &[],
            beside: &[],
        },
        WitMirror {
            record: "candidate",
            crate_record: candidate_record(),
            host_only: &["position", "load_sampled_at_micros", "engine"],
            before: &[],
            beside: affinity_record(),
        },
        WitMirror {
            record: "media",
            crate_record: media_record(),
            host_only: &[],
            before: &[],
            beside: &[],
        },
        WitMirror {
            record: "request",
            crate_record: request_record(),
            host_only: &[
                "prompt_chars",
                "prompt_bytes",
                "now_micros",
                "session_counted",
                "affinity",
                "named",
                "model_fingerprint",
                "signal_cells",
                "client_key",
                "ring",
                "ring_positions",
                "decode_ring_positions",
                "session_key_from_rid",
            ],
            before: &[],
            beside: &[],
        },
        WitMirror {
            record: "zone-capacity",
            crate_record: zone_record(),
            host_only: &[],
            before: &[],
            beside: &[],
        },
        WitMirror {
            record: "pool",
            crate_record: pool_record(),
            host_only: &["reach", "reported", "engines", "word"],
            before: &[],
            beside: &[],
        },
        WitMirror {
            record: "lease",
            crate_record: lease_record(),
            host_only: &[
                "booked",
                "session_verdict",
                "issued_micros",
                "chars",
                "bytes",
            ],
            before: &[],
            beside: &[],
        },
        WitMirror {
            record: "pair",
            crate_record: pair_record(),
            host_only: &["decode_position", "decode_verdict", "decode_placing"],
            before: &[],
            beside: &[],
        },
        WitMirror {
            record: "target",
            crate_record: target_record(),
            host_only: &["position"],
            before: &[],
            beside: &[],
        },
        WitMirror {
            record: "completed",
            crate_record: outcome_record(),
            host_only: &[],
            before: &["lease"],
            beside: &[],
        },
    ]
}

/// `wire_modes` to `wire-modes`.
fn kebab_field(snake: &str) -> String {
    snake.replace('_', "-")
}

#[test]
fn the_wasm_world_resolves_and_mirrors_the_engine_record() {
    let world = parse_world().expect("the world resolves");
    let name = &world.resolve.packages[world.package].name;
    assert_eq!(
        name.to_string(),
        format!("{WIT_NAMESPACE}:{WIT_PACKAGE}@{WIT_VERSION}")
    );

    for mirror in wit_mirrors() {
        let record_name = mirror.record;
        let expected: Vec<String> = mirror
            .before
            .iter()
            .map(|name| (*name).to_owned())
            .chain(
                mirror
                    .crate_record
                    .iter()
                    .filter(|name| !mirror.host_only.contains(name))
                    .map(|name| kebab_field(name)),
            )
            .collect();
        let found = world
            .record_fields(record_name)
            .unwrap_or_else(|| panic!("record {record_name} is missing from types"));
        // The fields of a record the crate keeps beside this one are in the
        // world's record in their own order; the rest is the crate's record.
        let beside: Vec<String> = mirror.beside.iter().map(|name| kebab_field(name)).collect();
        let (inside, found): (Vec<String>, Vec<String>) =
            found.into_iter().partition(|name| beside.contains(name));
        assert_eq!(found, expected, "record {record_name} is not the crate's");
        assert_eq!(
            inside, beside,
            "record {record_name}: the record beside it is not in the world's in its order"
        );
    }
    assert_eq!(
        world
            .record_fields("dispatched")
            .expect("dispatched is a record"),
        strings(&["lease", "target", "side", "since-decision-micros"]),
        "the dispatch event carries the lifecycle hook's arguments"
    );

    for (_, _, wit_name, cases) in enum_mirrors() {
        let expected: Vec<String> = cases.iter().map(|case| kebab_case(case)).collect();
        let found = world
            .cases(wit_name)
            .unwrap_or_else(|| panic!("{wit_name} is missing from types"));
        assert_eq!(found, expected, "{wit_name} is not the crate's enum");
    }
    let expected: Vec<String> = signal_cases().iter().map(|case| kebab_case(case)).collect();
    assert_eq!(
        world.cases("signal").expect("signal is a variant"),
        expected
    );
    let expected: Vec<String> = health_record()
        .iter()
        .map(|flag| kebab_field(flag))
        .collect();
    assert_eq!(
        world.cases("health").expect("health is a flags type"),
        expected
    );
    assert_eq!(
        world
            .cases("wire-modes")
            .expect("wire-modes is a flags type"),
        strings(&["http", "grpc", "zmq"])
    );
    assert_eq!(
        world.cases("identity").expect("identity is a variant"),
        strings(&["none", "hash", "plain"])
    );
    assert_eq!(
        world.cases("decline").expect("decline is a variant"),
        strings(&["not-applicable", "budget", "internal"])
    );
    assert_eq!(
        world
            .cases("lifecycle-event")
            .expect("lifecycle-event is a variant"),
        strings(&[
            "issued",
            "dispatched",
            "completed",
            "released",
            "membership",
            "signal"
        ])
    );
}

/// The packed request block's layout as the world's comment states it: an
/// 80-byte header, the model bytes, zero bytes to the next multiple of 8,
/// then the u64 sections. The arithmetic is mirrored here so the alignment
/// rule is pinned beside the sentence that states it.
const PACKED_REQUEST_HEADER: usize = 80;

fn packed_hashes_offset(model_len: usize) -> usize {
    (PACKED_REQUEST_HEADER + model_len).div_ceil(8) * 8
}

#[test]
fn the_packed_request_block_keeps_its_u64_sections_aligned() {
    assert_eq!(packed_hashes_offset(12), 96, "a 12-byte model name");
    for model_len in 0..=255usize {
        let hashes = packed_hashes_offset(model_len);
        assert_eq!(hashes % 8, 0, "block hashes at a {model_len}-byte name");
        let tried = hashes + 8 * 16;
        assert_eq!(
            tried % 8,
            0,
            "tried after 16 hashes at a {model_len}-byte name"
        );
    }
    let wit =
        std::fs::read_to_string(contracts_dir().join(WORLD_WIT)).expect("the world file reads");
    for sentence in [
        "Request block: an 80-byte header",
        "zero bytes to the next multiple of 8",
        "puts the hashes at 96",
        "returns 8-aligned memory for these lists",
    ] {
        assert!(
            wit.contains(sentence),
            "the world states the alignment rule: {sentence:?}"
        );
    }
}
