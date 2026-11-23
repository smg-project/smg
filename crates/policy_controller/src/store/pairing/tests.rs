//! The cases of the pairing rule and of the derivation of the pair classes.
//! Every name is an opaque token: the rule compares names for equality and
//! reads none.

use super::{
    key, mismatch, runtime_specific, wire_mismatch, Member, Mismatch, PairClasses,
    PairingDescriptor, Refusal, LAYOUT_FACTS,
};
use crate::{
    document::spec::Pairing,
    engine::view::{WireModes, WorkerId},
};

mod written;

const MODES: [Pairing; 3] = [Pairing::Off, Pairing::Lenient, Pairing::Strict];
const COMPARING: [Pairing; 2] = [Pairing::Lenient, Pairing::Strict];

/// One worker as a case writes it: its runtime (empty when unknown), its
/// wires and what it reports.
#[derive(Clone, Debug, Default)]
struct Worker {
    runtime: &'static str,
    wires: WireModes,
    descriptor: PairingDescriptor,
}

/// A worker of `runtime` on the runtime-specific side that reports nothing.
fn worker(runtime: &'static str) -> Worker {
    Worker {
        runtime,
        wires: WireModes::GRPC,
        ..Worker::default()
    }
}

impl Worker {
    fn wires(mut self, wires: WireModes) -> Self {
        self.wires = wires;
        self
    }

    fn protocol(mut self, protocol: &str) -> Self {
        self.descriptor.protocol = Some(protocol.to_string());
        self
    }

    fn transport(mut self, transport: &str) -> Self {
        self.descriptor.transport = Some(transport.to_string());
        self
    }

    fn version(mut self, version: &str) -> Self {
        self.descriptor.version = Some(version.to_string());
        self
    }

    /// Report the layout fact a key spells `short` as `value`.
    fn layout(mut self, short: &str, value: &str) -> Self {
        let at = LAYOUT_FACTS
            .iter()
            .position(|&(fact, _)| fact == short)
            .expect("a layout fact");
        self.descriptor.kv_layout[at] = Some(value.to_string());
        self
    }

    fn member(&self, id: u64) -> Member<'_> {
        Member::new(
            WorkerId(id),
            self.wires,
            self.runtime,
            Some(&self.descriptor),
        )
    }
}

/// What the rule says of `prefill` beside `decode` under `mode`.
fn verdict(prefill: &Worker, decode: &Worker, mode: Pairing) -> Option<Mismatch> {
    mismatch(&prefill.member(1), &decode.member(2), mode)
}

/// What the wire's fact and then the rule say of the two.
fn wire_verdict(prefill: &Worker, decode: &Worker, mode: Pairing) -> Option<Mismatch> {
    wire_mismatch(&prefill.member(1), &decode.member(2), mode)
}

/// The members of a pool, numbered from `first`.
fn pool(workers: &[Worker], first: u64) -> Vec<Member<'_>> {
    workers
        .iter()
        .zip(first..)
        .map(|(worker, id)| worker.member(id))
        .collect()
}

/// The classes of the two pools: the prefills numbered from 1, the decodes
/// from 101.
fn derived(prefills: &[Worker], decodes: &[Worker], mode: Pairing) -> PairClasses {
    PairClasses::derive(&pool(prefills, 1), &pool(decodes, 101), mode)
}

/// Each prefill's class, in the order given.
fn class_of(classes: &PairClasses) -> Vec<u32> {
    classes.prefills().iter().map(|&(_, class)| class).collect()
}

/// Each decode's class, in the order given.
fn decode_class_of(classes: &PairClasses) -> Vec<u32> {
    classes.decodes().iter().map(|&(_, class)| class).collect()
}

/// The table: the decode classes each prefill class lists, from class 1.
fn table_of(classes: &PairClasses) -> Vec<Vec<u32>> {
    let table = classes.table();
    (1..=classes.classes())
        .map(|class| table.of(&class).to_vec())
        .collect()
}

fn strings(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_string()).collect()
}

#[test]
fn off_compares_nothing_of_the_handoff() {
    let prefill = worker("rt-a")
        .protocol("x")
        .transport("tx-a")
        .layout("page", "16")
        .version("1");
    let decode = worker("rt-b")
        .protocol("y")
        .transport("tx-b")
        .layout("page", "32")
        .version("2");
    assert_eq!(verdict(&prefill, &decode, Pairing::Off), None);
    // The same two differ on the first thing a comparing mode reads.
    for mode in COMPARING {
        assert_eq!(verdict(&prefill, &decode, mode), Some(Mismatch::Runtime));
    }
}

#[test]
fn a_known_difference_in_a_hard_fact_refuses_the_pair_in_both_comparing_modes() {
    let on = || worker("rt-a").transport("tx-a");
    let cases = [
        (worker("rt-a"), worker("rt-b"), Mismatch::Runtime),
        (on(), worker("rt-a").transport("tx-b"), Mismatch::Transport),
        (
            on().layout("dtype", "a"),
            on().layout("dtype", "b"),
            Mismatch::KvLayout("kv_cache_dtype"),
        ),
        (
            on().layout("page", "16"),
            on().layout("page", "32"),
            Mismatch::KvLayout("page_size"),
        ),
        (
            on().layout("attn", "a"),
            on().layout("attn", "b"),
            Mismatch::KvLayout("attention_backend"),
        ),
        (
            on().layout("model", "a"),
            on().layout("model", "b"),
            Mismatch::KvLayout("model_dtype"),
        ),
    ];
    for (prefill, decode, expected) in &cases {
        for mode in COMPARING {
            assert_eq!(
                verdict(prefill, decode, mode),
                Some(*expected),
                "{mode:?}: {prefill:?} beside {decode:?}"
            );
        }
    }
}

#[test]
fn a_fact_one_side_does_not_report_pairs_leniently_and_is_refused_strictly() {
    let known = || worker("rt-a").transport("tx-a").layout("page", "16");
    let cases = [
        (
            Worker {
                runtime: "",
                ..known()
            },
            "runtime",
        ),
        (worker("rt-a").layout("page", "16"), "transport"),
        (worker("rt-a").transport("tx-a"), "page_size"),
        (known().protocol("x"), "pairing_protocol"),
    ];
    for (other, component) in &cases {
        // Either side may be the one that differs in what it reports.
        for (prefill, decode) in [(&known(), other), (other, &known())] {
            assert_eq!(
                verdict(prefill, decode, Pairing::Lenient),
                None,
                "{component}"
            );
            assert_eq!(
                verdict(prefill, decode, Pairing::Strict),
                Some(Mismatch::Unknown(component))
            );
        }
    }
}

#[test]
fn what_neither_side_reports_is_unknown_for_a_runtime_or_a_transport_and_no_fact_for_a_layout() {
    // Neither names a runtime: it exists and is unknown.
    let nameless = worker("").transport("tx-a");
    assert_eq!(verdict(&nameless, &nameless, Pairing::Lenient), None);
    assert_eq!(
        verdict(&nameless, &nameless, Pairing::Strict),
        Some(Mismatch::Unknown("runtime"))
    );
    // Neither names a transport: the same.
    let silent = worker("rt-a");
    assert_eq!(verdict(&silent, &silent, Pairing::Lenient), None);
    assert_eq!(
        verdict(&silent, &silent, Pairing::Strict),
        Some(Mismatch::Unknown("transport"))
    );
    // Neither reports a layout fact: nothing to compare, and strict pairs.
    let bare = worker("rt-a").transport("tx-a");
    assert_eq!(verdict(&bare, &bare, Pairing::Strict), None);
}

#[test]
fn two_set_protocols_are_compared_whole_and_settle_the_pair() {
    let prefill = worker("rt-a")
        .protocol("x")
        .transport("tx-a")
        .layout("page", "16")
        .version("1");
    // The same protocol over facts that differ: the operator's word pairs.
    let same = worker("rt-a")
        .protocol("x")
        .transport("tx-b")
        .layout("page", "32")
        .version("2");
    // Another protocol over facts that agree: the operator's word refuses.
    let other = Worker {
        descriptor: PairingDescriptor {
            protocol: Some("y".to_string()),
            ..prefill.descriptor.clone()
        },
        ..prefill.clone()
    };
    for mode in COMPARING {
        assert_eq!(verdict(&prefill, &same, mode), None, "{mode:?}");
        assert_eq!(
            verdict(&prefill, &other, mode),
            Some(Mismatch::Explicit),
            "{mode:?}"
        );
    }
    // The runtime is read before the protocols.
    let elsewhere = Worker {
        runtime: "rt-b",
        ..same.clone()
    };
    assert_eq!(
        verdict(&prefill, &elsewhere, Pairing::Lenient),
        Some(Mismatch::Runtime)
    );
    // A protocol on one side alone: lenient falls through to the facts.
    let unset = worker("rt-a").transport("tx-b");
    assert_eq!(
        verdict(&prefill, &unset, Pairing::Lenient),
        Some(Mismatch::Transport)
    );
    assert_eq!(
        verdict(
            &prefill,
            &worker("rt-a").transport("tx-a"),
            Pairing::Lenient
        ),
        None
    );
}

#[test]
fn the_version_counts_under_strict_alone_and_only_when_both_sides_report_one() {
    let at = |version: &str| worker("rt-a").transport("tx-a").version(version);
    assert_eq!(verdict(&at("1"), &at("2"), Pairing::Lenient), None);
    assert_eq!(
        verdict(&at("1"), &at("2"), Pairing::Strict),
        Some(Mismatch::Version)
    );
    assert_eq!(verdict(&at("1"), &at("1"), Pairing::Strict), None);
    // A worker that reports none: a version is no fact every worker carries.
    let silent = worker("rt-a").transport("tx-a");
    assert_eq!(verdict(&at("1"), &silent, Pairing::Strict), None);
    assert_eq!(verdict(&silent, &at("1"), Pairing::Strict), None);
    // Every hard fact is read before it.
    assert_eq!(
        verdict(
            &at("1").layout("model", "a"),
            &at("2").layout("model", "b"),
            Pairing::Strict
        ),
        Some(Mismatch::KvLayout("model_dtype"))
    );
}

#[test]
fn a_mismatch_reads_as_the_component_it_names() {
    let read = |mismatch: Mismatch| mismatch.to_string();
    assert_eq!(read(Mismatch::Runtime), "runtime");
    assert_eq!(read(Mismatch::Explicit), "pairing_protocol");
    assert_eq!(read(Mismatch::Transport), "transport");
    assert_eq!(read(Mismatch::Version), "version");
    assert_eq!(read(Mismatch::KvLayout("page_size")), "page_size");
    assert_eq!(read(Mismatch::Unknown("transport")), "unknown transport");
}

#[test]
fn a_key_names_the_group_a_worker_pairs_in() {
    let named = worker("rt-a")
        .transport("tx-a")
        .layout("page", "16")
        .layout("dtype", "f16")
        .version("9");
    // The layout facts in their own order, and no version.
    assert_eq!(key(&named.member(1)), "rt-a/tx-a/dtype=f16,page=16");
    assert_eq!(
        key(&worker("rt-a").transport("tx-a").member(1)),
        "rt-a/tx-a/?"
    );
    assert_eq!(key(&worker("").member(1)), "unspecified/?/?");
    // A row with no descriptor reports nothing.
    let bare = Member::new(WorkerId(1), WireModes::GRPC, "rt-a", None);
    assert_eq!(key(&bare), "rt-a/?/?");
    assert_eq!(mismatch(&bare, &named.member(2), Pairing::Lenient), None);
    // An operator's protocol is the whole key.
    assert_eq!(key(&named.protocol("x").member(1)), "x");
}

#[test]
fn a_normalized_descriptor_holds_no_blank_value() {
    let written = PairingDescriptor {
        protocol: Some(String::new()),
        transport: Some(" a ".to_string()),
        version: Some(" ".to_string()),
        kv_layout: [
            Some("16".to_string()),
            Some("\t".to_string()),
            None,
            Some(" f ".to_string()),
        ],
    };
    assert!(!written.is_normalized());
    let mut read = written;
    read.normalize();
    assert_eq!(
        read,
        PairingDescriptor {
            protocol: None,
            transport: Some("a".to_string()),
            version: None,
            kv_layout: [Some("16".to_string()), None, None, Some("f".to_string())],
        }
    );
    assert!(read.is_normalized());
    // A descriptor that holds no blank stays as it is.
    let mut again = read.clone();
    again.normalize();
    assert_eq!(again, read);
    assert!(PairingDescriptor::default().is_normalized());
}

/// A worker of a descriptor that names `transport` as it is written, with
/// no door before the rule.
fn as_written(transport: &str) -> PairingDescriptor {
    PairingDescriptor {
        transport: Some(transport.to_string()),
        ..PairingDescriptor::default()
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a descriptor reaches the rule normalized")]
fn a_blank_value_stops_the_rule_in_a_debug_build() {
    let _member = Member::new(WorkerId(1), WireModes::GRPC, "", Some(&as_written("")));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a descriptor reaches the rule normalized")]
fn a_padded_value_stops_the_rule_in_a_debug_build() {
    let _member = Member::new(WorkerId(1), WireModes::GRPC, "", Some(&as_written(" a ")));
}

#[test]
fn only_http_alone_is_the_plain_side_of_the_wire_fact() {
    assert!(!runtime_specific(WireModes::HTTP));
    let (http, grpc, zmq) = (WireModes::HTTP, WireModes::GRPC, WireModes::ZMQ);
    for wires in [
        WireModes::NONE,
        grpc,
        zmq,
        http.union(grpc),
        http.union(zmq),
        grpc.union(zmq),
        http.union(grpc).union(zmq),
    ] {
        assert!(runtime_specific(wires), "{wires:?}");
    }
}

#[test]
fn a_worker_on_http_and_one_wire_more_never_pairs_with_an_http_only_one() {
    let on = |wires: WireModes| worker("rt-a").transport("tx-a").wires(wires);
    let both = WireModes::HTTP.union(WireModes::GRPC);
    let decodes = [on(WireModes::HTTP), on(both), on(WireModes::GRPC)];
    for mode in MODES {
        // The set of two is on the runtime-specific side, whichever role.
        let classes = derived(&[on(both)], &decodes, mode);
        assert_eq!(classes.list(1), [1, 2], "{mode:?}");
        let classes = derived(&[on(WireModes::HTTP)], &decodes, mode);
        assert_eq!(classes.list(1), [0], "{mode:?}");
    }
}

#[test]
fn two_known_runtimes_never_pair_on_the_runtime_specific_side_whatever_the_mode() {
    let (one, other, unknown) = (worker("rt-a"), worker("rt-b"), worker(""));
    for mode in MODES {
        assert_eq!(
            wire_verdict(&one, &other, mode),
            Some(Mismatch::Runtime),
            "{mode:?}"
        );
    }
    // An unknown runtime is not a different one: it pairs as the mode says.
    assert_eq!(wire_verdict(&one, &unknown, Pairing::Off), None);
    assert_eq!(wire_verdict(&unknown, &other, Pairing::Lenient), None);
    assert_eq!(
        wire_verdict(&one, &unknown, Pairing::Strict),
        Some(Mismatch::Unknown("runtime"))
    );
    // The plain side has no such fact: off pairs two runtimes there, and a
    // comparing mode refuses them by its own rule.
    let (one, other) = (one.wires(WireModes::HTTP), other.wires(WireModes::HTTP));
    assert_eq!(wire_verdict(&one, &other, Pairing::Off), None);
    assert_eq!(
        wire_verdict(&one, &other, Pairing::Lenient),
        Some(Mismatch::Runtime)
    );
}

#[test]
fn prefills_of_one_offer_share_a_class_and_a_class_is_numbered_as_its_first_prefill_comes() {
    let (a, b) = (
        worker("rt-a").transport("tx-a"),
        worker("rt-b").transport("tx-a"),
    );
    let prefills = [b.clone(), a.clone(), b.clone(), a.clone()];
    let decodes = [a.clone(), b, a];
    let classes = derived(&prefills, &decodes, Pairing::Lenient);
    assert_eq!(class_of(&classes), [1, 2, 1, 2]);
    assert_eq!(classes.classes(), 2);
    assert_eq!(classes.list(1), [1]);
    assert_eq!(classes.list(2), [0, 2]);
    // The decodes are numbered by themselves, as their first comes: the
    // first prefill class pairs with the second decode class, and the table
    // says so.
    assert_eq!(
        classes.decodes(),
        [(WorkerId(101), 1), (WorkerId(102), 2), (WorkerId(103), 1)]
    );
    assert_eq!(table_of(&classes), [[2], [1]]);
    assert_eq!(
        classes
            .prefills()
            .iter()
            .map(|&(id, _)| id)
            .collect::<Vec<_>>(),
        [WorkerId(1), WorkerId(2), WorkerId(3), WorkerId(4)]
    );
    // One list a prefill would hold six pairs; two lists are held.
    assert_eq!(classes.pairs(), 6);
    // Class 0 and a class past the last hold nobody.
    assert!(classes.list(0).is_empty() && classes.list(3).is_empty());
}

#[test]
fn two_descriptors_with_one_partner_list_are_one_class() {
    // Two versions of a prefill: lenient compares no version, so each lists
    // every decode and the two are one class.
    let at = |version: &str| worker("rt-a").transport("tx-a").version(version);
    let prefills = [at("1"), at("2")];
    let decodes = [at("1"), at("2"), at("1")];
    let lenient = derived(&prefills, &decodes, Pairing::Lenient);
    assert_eq!(class_of(&lenient), [1, 1]);
    assert_eq!(lenient.classes(), 1);
    assert_eq!(lenient.list(1), [0, 1, 2]);
    // The decodes of two versions pair with the same prefills: one class,
    // and the table is one entry.
    assert_eq!(decode_class_of(&lenient), [1, 1, 1]);
    assert_eq!(table_of(&lenient), [[1]]);
    // Strict compares the version: two lists, so two classes a side.
    let strict = derived(&prefills, &decodes, Pairing::Strict);
    assert_eq!(class_of(&strict), [1, 2]);
    assert_eq!(strict.list(1), [0, 2]);
    assert_eq!(strict.list(2), [1]);
    assert_eq!(decode_class_of(&strict), [1, 2, 1]);
    assert_eq!(table_of(&strict), [[1], [2]]);
}

#[test]
fn a_decode_that_names_no_transport_is_one_class_each_transports_class_lists_leniently() {
    let on = |transport: &str| worker("rt-a").transport(transport);
    let prefills = [on("tx-a"), on("tx-b")];
    let decodes = [on("tx-a"), worker("rt-a"), on("tx-b")];
    let lenient = derived(&prefills, &decodes, Pairing::Lenient);
    assert_eq!(class_of(&lenient), [1, 2]);
    assert_eq!(lenient.list(1), [0, 1]);
    assert_eq!(lenient.list(2), [1, 2]);
    assert_eq!(lenient.pairs(), 4);
    // The silent decode is one worker of one class, listed twice.
    assert_eq!(decode_class_of(&lenient), [1, 2, 3]);
    assert_eq!(table_of(&lenient), [[1, 2], [2, 3]]);
    // Strict refuses the silent decode: class 0, and no list names it.
    let strict = derived(&prefills, &decodes, Pairing::Strict);
    assert_eq!(strict.list(1), [0]);
    assert_eq!(strict.list(2), [2]);
    assert_eq!(decode_class_of(&strict), [1, 0, 2]);
    assert_eq!(table_of(&strict), [[1], [2]]);
}

#[test]
fn decodes_that_pair_with_the_same_prefills_share_a_class_whatever_else_tells_them_apart() {
    let on = |transport: &str| worker("rt-a").transport(transport);
    let prefills = [on("tx-a"), on("tx-b")];
    // Two versions of the first transport, the second transport, and two
    // silent decodes of which one reports a layout fact no prefill does.
    let decodes = [
        on("tx-a").version("1"),
        on("tx-a").version("2"),
        on("tx-b"),
        worker("rt-a"),
        worker("rt-a").layout("page", "16"),
    ];
    let classes = derived(&prefills, &decodes, Pairing::Lenient);
    assert_eq!(decode_class_of(&classes), [1, 1, 2, 3, 3]);
    // Seven pairs are held as four entries: the table grows with the
    // classes that differ, never with the workers.
    assert_eq!(classes.pairs(), 7);
    assert_eq!(table_of(&classes), [[1, 3], [2, 3]]);
}

#[test]
fn a_worker_nobody_pairs_with_is_class_zero_and_takes_no_number() {
    let on = |transport: &str| worker("rt-a").transport(transport);
    let prefills = [on("tx-x"), on("tx-a"), on("tx-x"), on("tx-b")];
    let decodes = [on("tx-y"), on("tx-b"), on("tx-a")];
    let classes = derived(&prefills, &decodes, Pairing::Lenient);
    assert_eq!(class_of(&classes), [0, 1, 0, 2]);
    assert_eq!(classes.classes(), 2);
    assert_eq!(classes.list(1), [2]);
    assert_eq!(classes.list(2), [1]);
    assert_eq!(decode_class_of(&classes), [0, 1, 2]);
    assert_eq!(table_of(&classes), [[2], [1]]);
    // A pair is made on the side, so nothing is refused there.
    assert_eq!(classes.refusal(WireModes::GRPC), None);
}

#[test]
fn off_is_one_class_on_the_plain_side_and_one_a_known_runtime_on_the_other() {
    // The plain side: off sorts nothing, every prefill lists every decode.
    let plain = |runtime: &'static str, transport: &str| {
        worker(runtime).transport(transport).wires(WireModes::HTTP)
    };
    let prefills = [
        plain("rt-a", "tx-a"),
        plain("rt-b", "tx-b"),
        plain("", "tx-c"),
    ];
    let decodes = [
        plain("rt-b", "tx-a"),
        plain("", "tx-b"),
        plain("rt-a", "tx-c"),
    ];
    let off = derived(&prefills, &decodes, Pairing::Off);
    assert_eq!(class_of(&off), [1, 1, 1]);
    assert_eq!(off.list(1), [0, 1, 2]);
    assert_eq!(decode_class_of(&off), [1, 1, 1]);
    assert_eq!(table_of(&off), [[1]]);

    // The runtime-specific side: the wire's fact stands under off. A class
    // per known runtime, each with the decodes of unknown runtime, and a
    // prefill of unknown runtime lists every decode.
    let on = |runtime: &'static str, transport: &str| worker(runtime).transport(transport);
    let prefills = [
        on("rt-a", "tx-a"),
        on("rt-b", "tx-b"),
        on("", "tx-c"),
        on("rt-a", "tx-d"),
    ];
    let decodes = [on("rt-b", "tx-a"), on("", "tx-b"), on("rt-a", "tx-c")];
    let off = derived(&prefills, &decodes, Pairing::Off);
    assert_eq!(class_of(&off), [1, 2, 3, 1]);
    assert_eq!(off.list(1), [1, 2]);
    assert_eq!(off.list(2), [0, 1]);
    assert_eq!(off.list(3), [0, 1, 2]);
    assert_eq!(decode_class_of(&off), [1, 2, 3]);
    assert_eq!(table_of(&off), [vec![2, 3], vec![1, 2], vec![1, 2, 3]]);
}

#[test]
fn a_refusal_names_each_pools_keys_and_what_the_two_disagree_on() {
    let on = |runtime: &'static str, transport: &str| worker(runtime).transport(transport);
    let prefills = [on("rt-a", "tx-b"), on("rt-a", "tx-a"), on("rt-a", "tx-b")];
    let decodes = [on("rt-b", "tx-a"), on("rt-a", "tx-c").layout("page", "16")];
    let classes = derived(&prefills, &decodes, Pairing::Lenient);
    assert_eq!(class_of(&classes), [0, 0, 0]);
    assert_eq!(classes.classes(), 0);
    assert_eq!(classes.pairs(), 0);
    // Each list sorted, each entry once.
    assert_eq!(
        classes.refusal(WireModes::GRPC),
        Some(&Refusal {
            prefill: strings(&["rt-a/tx-a/?", "rt-a/tx-b/?"]),
            decode: strings(&["rt-a/tx-c/page=16", "rt-b/tx-a/?"]),
            mismatches: strings(&["runtime", "transport"]),
        })
    );
    // The other side holds nobody: nothing is refused there.
    assert_eq!(classes.refusal(WireModes::HTTP), None);

    // Strict says what it does not know.
    let strict = derived(&[worker("rt-a")], &[worker("")], Pairing::Strict);
    assert_eq!(
        strict.refusal(WireModes::GRPC),
        Some(&Refusal {
            prefill: strings(&["rt-a/?/?"]),
            decode: strings(&["unspecified/?/?"]),
            mismatches: strings(&["unknown runtime"]),
        })
    );
}

#[test]
fn a_refusal_is_one_sides_of_the_wire_fact() {
    let plain = |transport: &str| worker("rt-a").transport(transport).wires(WireModes::HTTP);
    let on = |transport: &str| worker("rt-a").transport(transport);
    let prefills = [plain("tx-a"), on("tx-a")];
    let decodes = [plain("tx-b"), on("tx-a")];
    let classes = derived(&prefills, &decodes, Pairing::Lenient);
    assert_eq!(class_of(&classes), [0, 1]);
    // The plain side makes no pair and says why; the other side made one.
    assert_eq!(
        classes.refusal(WireModes::HTTP),
        Some(&Refusal {
            prefill: strings(&["rt-a/tx-a/?"]),
            decode: strings(&["rt-a/tx-b/?"]),
            mismatches: strings(&["transport"]),
        })
    );
    assert_eq!(classes.refusal(WireModes::GRPC), None);
    // A side one of whose pools holds nobody refuses nothing: the plain
    // prefill has no decode on its side at all.
    let lone = derived(&prefills, &decodes[1..], Pairing::Lenient);
    assert_eq!(class_of(&lone), [0, 1]);
    assert_eq!(lone.refusal(WireModes::HTTP), None);
}

/// A fleet drawn from a few values of every fact, so that offers repeat
/// and differ: the same fleet for the same seed.
fn fleet(seed: u64, workers: usize) -> Vec<Worker> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut draw = |of: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) % of) as usize
    };
    (0..workers)
        .map(|_| {
            let mut drawn = worker(["", "rt-a", "rt-b"][draw(3)]);
            if draw(3) == 0 {
                drawn = drawn.wires(WireModes::HTTP);
            }
            if let Some(transport) = [None, Some("tx-a"), Some("tx-b")][draw(3)] {
                drawn = drawn.transport(transport);
            }
            if draw(4) == 0 {
                drawn = drawn.protocol(["x", "y"][draw(2)]);
            }
            if draw(2) == 0 {
                drawn = drawn.layout("page", ["16", "32"][draw(2)]);
            }
            if draw(3) == 0 {
                drawn = drawn.version(["1", "2"][draw(2)]);
            }
            drawn
        })
        .collect()
}

/// Whether the two pair, worker by worker: on one side of the wire's fact
/// and refused by nothing.
fn pairs(prefill: &Member<'_>, decode: &Member<'_>, mode: Pairing) -> bool {
    runtime_specific(prefill.wires) == runtime_specific(decode.wires)
        && wire_mismatch(prefill, decode, mode).is_none()
}

/// The workers of a pool on one side of the wire's fact.
fn on_side<'a>(side: bool, pool: &[Member<'a>]) -> Vec<Member<'a>> {
    let on = |member: &&Member<'a>| runtime_specific(member.wires) == side;
    pool.iter().filter(on).copied().collect()
}

/// The refusal of one side of the wire's fact, worker by worker.
fn refusal_of(
    side: bool,
    prefills: &[Member<'_>],
    decodes: &[Member<'_>],
    mode: Pairing,
) -> Option<Refusal> {
    let (prefills, decodes) = (on_side(side, prefills), on_side(side, decodes));
    let paired = prefills
        .iter()
        .any(|prefill| decodes.iter().any(|decode| pairs(prefill, decode, mode)));
    if paired || prefills.is_empty() || decodes.is_empty() {
        return None;
    }
    let distinct = |mut words: Vec<String>| {
        words.sort();
        words.dedup();
        words
    };
    let mut mismatches = Vec::new();
    for prefill in &prefills {
        for decode in &decodes {
            mismatches.extend(wire_mismatch(prefill, decode, mode).map(|m| m.to_string()));
        }
    }
    Some(Refusal {
        prefill: distinct(prefills.iter().map(key).collect()),
        decode: distinct(decodes.iter().map(key).collect()),
        mismatches: distinct(mismatches),
    })
}

/// The derivation of the two pools against the rule run worker by worker.
fn holds_worker_by_worker(prefills: &[Member<'_>], decodes: &[Member<'_>], mode: Pairing) {
    let classes = PairClasses::derive(prefills, decodes, mode);
    assert_eq!(classes.prefills().len(), prefills.len());
    // The distinct lists, as their first prefill comes.
    let mut lists: Vec<Vec<u32>> = Vec::new();
    for (prefill, &(id, class)) in prefills.iter().zip(classes.prefills()) {
        assert_eq!(id, prefill.id);
        let list: Vec<u32> = (0..decodes.len() as u32)
            .filter(|&decode| pairs(prefill, &decodes[decode as usize], mode))
            .collect();
        assert_eq!(classes.list(class), list, "{mode:?}: {prefill:?}");
        let number = if list.is_empty() {
            0
        } else if let Some(at) = lists.iter().position(|known| *known == list) {
            at as u32 + 1
        } else {
            lists.push(list);
            lists.len() as u32
        };
        assert_eq!(class, number, "{mode:?}: {prefill:?}");
    }
    assert_eq!(classes.classes() as usize, lists.len());
    // The decodes that pair with the same prefill classes are one class,
    // numbered as its first decode comes.
    assert_eq!(classes.decodes().len(), decodes.len());
    let mut columns: Vec<Vec<u32>> = Vec::new();
    for (decode, &(id, class)) in decodes.iter().zip(classes.decodes()) {
        assert_eq!(id, decode.id);
        let mut column: Vec<u32> = prefills
            .iter()
            .zip(classes.prefills())
            .filter(|(prefill, _)| pairs(prefill, decode, mode))
            .map(|(_, &(_, class))| class)
            .collect();
        column.sort_unstable();
        column.dedup();
        let number = if column.is_empty() {
            0
        } else if let Some(at) = columns.iter().position(|known| *known == column) {
            at as u32 + 1
        } else {
            columns.push(column);
            columns.len() as u32
        };
        assert_eq!(class, number, "{mode:?}: {decode:?}");
    }
    // What a pair decision reads: a prefill pairs with a decode exactly
    // when its class lists the decode's, each list ascending and naming no
    // class twice.
    let table = classes.table();
    // The table is valid by the question a host asks where it builds one,
    // and it numbers densely, as the engine asks of a host: a list names
    // every decode class, so no row's class is above the count of the
    // entries.
    assert_eq!(table.validate(), Ok(()), "{mode:?}");
    let largest = classes.decodes().iter().map(|&(_, class)| class).max();
    assert!(
        largest.unwrap_or(0) as usize <= classes.entries(),
        "{mode:?}"
    );
    for (prefill, &(_, listing)) in prefills.iter().zip(classes.prefills()) {
        let listed = table.of(&listing);
        assert!(
            listed.windows(2).all(|pair| pair[0] < pair[1]),
            "{listed:?}"
        );
        for (decode, &(_, class)) in decodes.iter().zip(classes.decodes()) {
            assert_eq!(
                class != 0 && listed.contains(&class),
                pairs(prefill, decode, mode),
                "{mode:?}: {prefill:?} beside {decode:?}"
            );
        }
    }
    for (side, wires) in [(false, WireModes::HTTP), (true, WireModes::GRPC)] {
        assert_eq!(
            classes.refusal(wires),
            refusal_of(side, prefills, decodes, mode).as_ref(),
            "{mode:?}: side {side}"
        );
    }
}

#[test]
fn the_classes_hold_exactly_the_pairs_the_rule_makes_worker_by_worker() {
    // Small fleets refuse a side often; large ones repeat every offer.
    let mut refused = 0;
    for (prefills, decodes) in [(1, 1), (2, 3), (5, 4), (12, 17), (40, 60)] {
        for seed in 0..40 {
            let (prefills, decodes) = (fleet(seed, prefills), fleet(seed + 1_000, decodes));
            let (prefills, decodes) = (pool(&prefills, 1), pool(&decodes, 101));
            for mode in MODES {
                holds_worker_by_worker(&prefills, &decodes, mode);
                let classes = PairClasses::derive(&prefills, &decodes, mode);
                refused += usize::from(classes.refusal(WireModes::HTTP).is_some())
                    + usize::from(classes.refusal(WireModes::GRPC).is_some());
            }
        }
    }
    // The fleets drawn do reach the refusal.
    assert!(refused > 20, "{refused} refusals over the fleets drawn");
}
