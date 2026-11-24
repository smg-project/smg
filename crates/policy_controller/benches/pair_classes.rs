//! The cost of the pair classes at 128 prefills and 384 decodes of one
//! model:
//!
//! - `policy_pairs/derive/<fleet>/<mode>`: one derivation over the two
//!   pools;
//! - `policy_pairs/membership_write/<fleet>`: one membership write of a
//!   store that holds the fleet, under the default mode: a registration
//!   refreshed with the offer it had, every pool derived again, its classes
//!   carried, and published;
//! - `policy_pairs/apply_mode/<fleet>`: one generation of another mode
//!   applied to a store that holds the fleet: the standing membership
//!   derived again under it and published once.
//!
//! The fleets, each of one wire. A worker stands once in its pool, so the
//! decode pool is 384 rows in each:
//!
//! - `bare`: one runtime, nobody reports a descriptor, as a store holds its
//!   workers until a handshake fills one: one class a side leniently, none
//!   strictly;
//! - `uniform`: one runtime and one transport: one class a side;
//! - `mixed`: two runtimes and two transports, a quarter of the decodes
//!   naming no transport: leniently four prefill classes and six decode
//!   classes;
//! - `apart`: every prefill a transport of its own and three decodes of it:
//!   128 classes a side, each list one class;
//! - `overlap`: every prefill a transport of its own, one decode of it and
//!   256 decodes that name none: leniently 128 lists of two, the silent
//!   decodes one class that every list names;
//! - `widest`: every prefill names one fact of the seven the rule compares
//!   leniently, with a value of its own, and each decode names none, one or
//!   two of them: leniently every worker a class of its own, and a decode
//!   class pairs with all but a few prefill classes. A fleet built for a
//!   wide table: the rule derives one near the type's bound of one entry a
//!   pair of classes.
//!
//! Before the timed cells the bench prints, per fleet and mode, what the
//! derivation holds: each side's classes, the table's bytes (four an entry
//! and four a list's end), the bytes of one list a prefill (four a pair),
//! and the members a store of that mode lists in the decode pool.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use smg_policy_controller::{
    document::{AppliedSpec, Pairing, PolicySpec},
    engine::{Role, WireModes},
    store::{
        identity,
        pairing::{Member, PairClasses, PairingDescriptor},
        MemoryStore, WorkerFacts, WorkersStore,
    },
};

const PREFILLS: usize = 128;
const DECODES: usize = 384;
const MODEL: &str = "m";
const FLEETS: [&str; 6] = ["bare", "uniform", "mixed", "apart", "overlap", "widest"];
const MODES: [(&str, Pairing); 3] = [
    ("off", Pairing::Off),
    ("lenient", Pairing::Lenient),
    ("strict", Pairing::Strict),
];

/// A pair plan of the least-load profile on each side, less the two
/// admission filters: no fact of the store feeds their bits yet.
const PAIR_PLAN: &str = r#"{
  "apiVersion": "smg.policy/v1",
  "profiles": {
    "side": { "filter": ["healthy", "not-overloaded", "not-tried"], "score": ["least-load"], "pick": "best" }
  },
  "bindings": [
    { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "side", "decode": "side" } }
  ]
}"#;

/// What a worker reports: its runtime (empty when unknown) and its
/// descriptor.
type Offer = (String, Option<PairingDescriptor>);

/// A descriptor that names `transport` and nothing else.
fn on(transport: String) -> PairingDescriptor {
    PairingDescriptor {
        transport: Some(transport),
        ..PairingDescriptor::default()
    }
}

/// One fact a worker of the `widest` fleet names: the fact's number, of the
/// seven the rule compares leniently (a protocol, a runtime, a transport
/// and the four layout facts), and its value's.
type Atom = (usize, usize);

/// The fact two prefills of the `widest` fleet name; the six others are
/// named by 21 prefills each.
const NARROW: usize = 6;
/// How many prefills name each of the six other facts.
const WIDE: usize = 21;
/// A value no prefill names.
const FRESH: usize = WIDE;

/// A worker of the `widest` fleet that names `atoms` and nothing else.
fn naming(atoms: &[Atom]) -> Offer {
    let mut runtime = String::new();
    let mut descriptor = PairingDescriptor::default();
    for &(fact, value) in atoms {
        let value = format!("v{value}");
        match fact {
            0 => descriptor.protocol = Some(value),
            1 => runtime = value,
            2 => descriptor.transport = Some(value),
            layout => descriptor.kv_layout[layout - 3] = Some(value),
        }
    }
    (runtime, Some(descriptor))
}

/// The one fact prefill `index` of the `widest` fleet names, with a value
/// of its own: two prefills the narrow fact, 21 each of the others.
fn widest_prefill(index: usize) -> Atom {
    match index.checked_sub(2) {
        None => (NARROW, index),
        Some(rest) => (rest / WIDE, rest % WIDE),
    }
}

/// The decodes of the `widest` fleet: each names none, one or two of the
/// prefills' facts, and pairs with every prefill but those that name one of
/// its facts with another value. No two pair with the same prefills, so
/// every decode is a class of its own, and each misses few prefills: one
/// that names nothing, the narrow fact's two values and a fresh one, each
/// wide value and a fresh one a wide fact, then a wide value beside a
/// narrow one until the pool is full.
fn widest_decodes() -> Vec<Offer> {
    let wide = || (0..NARROW).flat_map(|fact| (0..WIDE).map(move |value| (fact, value)));
    let narrow = [0, 1, FRESH].map(|value| (NARROW, value));
    let mut named: Vec<Vec<Atom>> = vec![Vec::new()];
    named.extend(narrow.map(|atom| vec![atom]));
    named.extend(wide().map(|atom| vec![atom]));
    named.extend((0..NARROW).map(|fact| vec![(fact, FRESH)]));
    for beside in [narrow[0], narrow[1]] {
        named.extend(wide().map(|atom| vec![atom, beside]));
    }
    named.truncate(DECODES);
    named.iter().map(|atoms| naming(atoms)).collect()
}

/// What worker `index` of `fleet` reports.
fn offer(fleet: &str, role: Role, index: usize) -> Offer {
    let own = |index: usize| on(format!("tx-{index}"));
    match (fleet, role) {
        ("mixed", _) => {
            let runtime = ["rt-a", "rt-b"][index % 2].to_string();
            // Of every four workers of a runtime: two of one transport, one
            // of the other, and one of the other that, as a decode, names
            // none.
            let transport = match (index / 2) % 4 {
                0 | 2 => Some("tx-a"),
                1 => Some("tx-b"),
                _ => (role == Role::Prefill).then_some("tx-b"),
            };
            (
                runtime,
                transport.map(|transport| on(transport.to_string())),
            )
        }
        ("widest", Role::Prefill) => naming(&[widest_prefill(index)]),
        (_, _) => {
            let descriptor = match (fleet, role) {
                ("apart", Role::Prefill) | ("overlap", Role::Prefill) => Some(own(index)),
                ("apart", _) => Some(own(index % PREFILLS)),
                ("overlap", _) => (index < PREFILLS).then(|| own(index)),
                ("bare", _) => None,
                _ => Some(on("tx-a".to_string())),
            };
            ("rt-a".to_string(), descriptor)
        }
    }
}

/// The registrations of `fleet`: its prefills, then its decodes.
fn fleet(fleet: &str) -> Vec<WorkerFacts> {
    let worker = |role: Role, index: usize, (runtime, pairing): Offer| {
        let side = if role == Role::Prefill { "p" } else { "d" };
        let key = format!("grpc://{side}{index}:9000");
        WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role,
            wire_modes: WireModes::GRPC,
            engine_kind: runtime,
            models: vec![MODEL.to_string()],
            dp_size: 1,
            pairing: pairing.map(Arc::new),
            ..WorkerFacts::default()
        }
    };
    let prefills =
        (0..PREFILLS).map(|index| worker(Role::Prefill, index, offer(fleet, Role::Prefill, index)));
    let decodes: Vec<Offer> = if fleet == "widest" {
        widest_decodes()
    } else {
        (0..DECODES)
            .map(|index| offer(fleet, Role::Decode, index))
            .collect()
    };
    let decodes = (0..).zip(decodes);
    prefills
        .chain(decodes.map(|(index, offer)| worker(Role::Decode, index, offer)))
        .collect()
}

/// One role's workers as the pairing rule reads them, in id order, as the
/// store hands them to a derivation of the whole model.
fn members(fleet: &[WorkerFacts], role: Role) -> Vec<Member<'_>> {
    let mut members: Vec<Member<'_>> = fleet
        .iter()
        .filter(|facts| facts.role == role)
        .map(|facts| {
            Member::new(
                identity(&facts.key, facts.incarnation),
                facts.wire_modes,
                &facts.engine_kind,
                facts.pairing.as_deref(),
            )
        })
        .collect();
    members.sort_unstable_by_key(|member| member.id);
    members
}

/// Generation `number` of the pair plan, its document naming `mode`.
fn generation(number: u64, mode: Pairing) -> AppliedSpec {
    let mut document: PolicySpec = serde_json::from_str(PAIR_PLAN).expect("a well-formed document");
    document.generation = number;
    document.plans.pd_pair.pairing = mode;
    AppliedSpec::validate(&document).expect("the plan validates")
}

/// A store of `mode` that holds `fleet`: the mode reaches it as it does in
/// service, with a generation whose document names it.
fn store_of(fleet: &[WorkerFacts], mode: Pairing) -> MemoryStore {
    let store = MemoryStore::new();
    store
        .apply(generation(1, mode))
        .expect("the first generation");
    for facts in fleet {
        store.upsert(facts.clone());
    }
    store.publish_now();
    if let Some(first) = fleet.first() {
        let rows = fleet
            .iter()
            .filter(|facts| facts.role == first.role)
            .count();
        filled::filled(&store, MODEL, first.role, rows);
    }
    store
}

#[expect(clippy::print_stdout, reason = "the sizes are the bench's report")]
fn sizes() {
    for name in FLEETS {
        let fleet = fleet(name);
        let (prefills, decodes) = (
            members(&fleet, Role::Prefill),
            members(&fleet, Role::Decode),
        );
        for (mode_name, mode) in MODES {
            let classes = PairClasses::derive(&prefills, &decodes, mode);
            let entries = classes.entries();
            let decode_classes = classes.decodes().iter().map(|&(_, class)| class).max();
            let rows = store_of(&fleet, mode)
                .snapshot()
                .members(MODEL, Role::Decode)
                .expect("the decode pool");
            println!(
                "policy_pairs/sizes/{name}/{mode_name}: prefill_classes={} decode_classes={} \
                 table_bytes={} pair_list_bytes={} decode_rows={rows}",
                classes.classes(),
                decode_classes.unwrap_or(0),
                4 * (classes.classes() as usize + entries),
                4 * classes.pairs(),
            );
        }
    }
}

fn derivation(c: &mut Criterion) {
    sizes();
    let mut group = c.benchmark_group("policy_pairs/derive");
    for name in FLEETS {
        let fleet = fleet(name);
        let (prefills, decodes) = (
            members(&fleet, Role::Prefill),
            members(&fleet, Role::Decode),
        );
        for (mode_name, mode) in MODES {
            group.bench_function(format!("{name}/{mode_name}"), |b| {
                b.iter(|| PairClasses::derive(black_box(&prefills), black_box(&decodes), mode));
            });
        }
    }
    group.finish();
}

fn membership_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_pairs/membership_write");
    for name in FLEETS {
        let fleet = fleet(name);
        let store = store_of(&fleet, Pairing::default());
        let refreshed = fleet.first().expect("a prefill").clone();
        group.bench_function(name, |b| {
            b.iter_batched(
                || refreshed.clone(),
                |facts| {
                    let change = store.upsert(black_box(facts));
                    store.publish_now();
                    change
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// A generation of another mode on a store that holds the fleet: strict
/// over lenient, then lenient over strict, by turns. Each generation is
/// validated outside the timed call.
fn mode_change(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_pairs/apply_mode");
    for name in FLEETS {
        let store = store_of(&fleet(name), Pairing::Lenient);
        let mut number = 1;
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    number += 1;
                    let mode = if number % 2 == 0 {
                        Pairing::Strict
                    } else {
                        Pairing::Lenient
                    };
                    generation(number, mode)
                },
                |next| store.apply(black_box(next)).expect("above the standing"),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, derivation, membership_write, mode_change);
criterion_main!(benches);
