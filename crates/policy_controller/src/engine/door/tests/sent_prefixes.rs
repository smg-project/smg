//! Sent prefixes through the in-process door, with the decision's instant
//! handed in by each case.

use std::{
    error::Error,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Barrier,
    },
    thread,
    time::{Duration, Instant},
};

use super::{
    gone, request, Candidate, Door, Lifecycle, Needs, Refused, Request, RequestFacts, TieBand,
    GATEWAY, MODEL,
};
use crate::{
    document::{
        spec,
        spec::{
            AffinitySource, AppliedSpec, BindingSpec, Match, PolicySpec, ScoreSpec, ScoreStage,
            ShadowSpec,
        },
    },
    engine::{
        decision::ShadowKind,
        door::{decide_in, entry, Buffers, Decided, Taken, SHADOW_FLOOR},
        framework::Scorer,
        view::{HashKind, LoadRow, Phase, Role, WorkerId},
    },
    stages::catalog::Catalog,
    store::{
        BatchFacts, BatchOutcome, KvEvent, KvEventSink, LeasesStore, LoadSink, LoadSource,
        MemoryStore, Signals, StoredBlock, WorkerFacts, WorkerStatus, WorkersStore,
    },
};

type Case<T = ()> = Result<T, Box<dyn Error>>;

const HASHES: [u64; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/// The `cache_aware` preset as it stands: its affinity stage names no
/// source, so it reads `merged`.
pub(crate) fn merged(generation: u64) -> Result<PolicySpec, spec::SpecError> {
    let mut document = PolicySpec::preset("cache_aware")?;
    document.generation = generation;
    Ok(document)
}

pub(crate) fn use_sent_prefixes(document: &mut PolicySpec, source: AffinitySource) {
    for stage in document
        .profiles
        .values_mut()
        .flat_map(|profile| &mut profile.score)
    {
        if let ScoreSpec::Params(ScoreStage::PrefixAffinity(params)) = stage {
            params.source = Some(source);
        }
    }
}

pub(crate) fn applied(document: &PolicySpec) -> Result<AppliedSpec, spec::SpecError> {
    AppliedSpec::validate_digested(document, document.digest()?, &Catalog::builtin(), &[])
}

fn fleet() -> Arc<MemoryStore> {
    Arc::new(MemoryStore::with_publish_window(Duration::ZERO).with_deciders(4))
}

fn worker(
    store: &MemoryStore,
    key: &str,
    role: Role,
    models: &[&str],
    events: bool,
) -> Case<WorkerId> {
    let id = store
        .upsert(WorkerFacts {
            key: key.into(),
            engine: key.into(),
            incarnation: 1,
            role,
            models: models.iter().map(|model| (*model).into()).collect(),
            dp_size: 1,
            signals: Signals {
                events,
                ..Signals::default()
            },
            ..WorkerFacts::default()
        })
        .id();
    let row = WorkersStore::get(store, id).ok_or("the registered row")?;
    if row.status != WorkerStatus::Ready {
        store
            .set_status(id, WorkerStatus::Ready, row.revision)
            .ok_or("the status write")?;
    }
    store.publish_now();
    Ok(id)
}

fn load(store: &MemoryStore, id: WorkerId, running: u32, now: Instant) {
    store.record(
        id,
        0,
        LoadRow {
            running,
            ..LoadRow::default()
        },
        now,
        LoadSource::Stream,
    );
}

fn token_request(hashes: &[u64]) -> Request<'_> {
    let mut asked = request(b"prefix", 7);
    asked.facts.block_hashes = hashes;
    asked.facts.prompt_tokens = hashes.len() as u32 * 16;
    asked.facts.namespace_seed = 21;
    asked
}

fn taken<'a>(asked: &'a Request<'a>, now: Instant) -> Taken<'a, Request<'a>> {
    Taken {
        request: asked,
        gateway: GATEWAY,
        now,
        left: Duration::ZERO,
        pins: None,
        place: None,
    }
}

fn order(
    door: &Door,
    asked: &Request<'_>,
    now: Instant,
    buffers: &mut Buffers<()>,
) -> Result<Decided, Refused> {
    let store = &**door.store();
    let term = store.term();
    store.deciding(|world| {
        let taken = taken(asked, now);
        let entry = entry(world, &taken);
        door.order(world, 0, term, &taken, buffers, entry)
    })
}

fn issued(
    door: &Door,
    asked: &Request<'_>,
    now: Instant,
    buffers: &mut Buffers<()>,
) -> Case<WorkerId> {
    let decided = order(door, asked, now, buffers).map_err(|word| format!("{word:?}"))?;
    let lease = decided.lease.ok_or("a pick's lease")?.lease;
    door.report(GATEWAY, lease.id, gone());
    Ok(lease.booked)
}

fn depths(buffers: &Buffers<()>) -> Vec<(u32, u32)> {
    buffers
        .named
        .holders
        .iter()
        .map(|holder| (holder.row, holder.predicted_blocks))
        .collect()
}

fn slices(store: &MemoryStore, role: Role) -> Option<usize> {
    store.deciding(|world| world.sent_slices(MODEL, role))
}

/// A batch through the worker's lane with one stored block no request of
/// these cases holds: the store has heard the worker's cache events.
fn hear(store: &MemoryStore, id: WorkerId) -> BatchOutcome {
    let other = StoredBlock {
        seq_hash: kv_index::SequenceHash(999),
        content_hash: kv_index::ContentHash(999),
    };
    let event = KvEvent::Stored {
        parent: None,
        blocks: vec![other],
    };
    let facts = BatchFacts {
        sequence: 1,
        ..BatchFacts::default()
    };
    store.admit(id, 0, facts, &[event])
}

#[test]
fn a_worker_with_no_events_is_named_by_what_it_was_sent() -> Case {
    let store = fleet();
    let a = worker(&store, "a", Role::Single, &[MODEL], false)?;
    let b = worker(&store, "b", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let now = Instant::now() + Duration::from_secs(120);
    load(&store, a, 0, now);
    load(&store, b, 1, now);
    let mut buffers = Buffers::default();
    let asked = token_request(&HASHES[..8]);
    let first = issued(&door, &asked, now, &mut buffers)?;
    assert_eq!(first, a, "a cold request goes to the lighter worker");
    let later = now + Duration::from_millis(20);
    load(&store, a, 2, later);
    load(&store, b, 0, later);
    assert_eq!(issued(&door, &asked, later, &mut buffers)?, a);
    assert_eq!(depths(&buffers), vec![(u32::from(a > b), 8)]);
    Ok(())
}

/// A worker can ask for cache events and never publish one. Its lane is
/// never heard, so the record speaks for it as for a worker with no lane.
#[test]
fn a_worker_that_asks_for_events_and_publishes_none_keeps_its_prompt() -> Case {
    let store = fleet();
    let a = worker(&store, "a", Role::Single, &[MODEL], true)?;
    let b = worker(&store, "b", Role::Single, &[MODEL], true)?;
    let c = worker(&store, "c", Role::Single, &[MODEL], true)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let mut buffers = Buffers::default();
    let start = Instant::now() + Duration::from_secs(120);
    for id in [a, b, c] {
        load(&store, id, u32::from(id != a), start);
    }
    let asked = token_request(&HASHES[..8]);
    assert_eq!(issued(&door, &asked, start, &mut buffers)?, a);
    // From here its worker is the heaviest row: load alone would move the
    // prompt at every turn.
    for turn in 1..40 {
        let now = start + Duration::from_millis(20 * turn);
        load(&store, a, 2, now);
        load(&store, b, 0, now);
        load(&store, c, 0, now);
        assert_eq!(issued(&door, &asked, now, &mut buffers)?, a, "turn {turn}");
    }
    Ok(())
}

/// A lane's first batch hands its row to the index, with no publication
/// between. Its worker reported gone hands the row back to the record,
/// which still holds what the row was sent before.
#[test]
fn a_heard_lane_hands_its_row_to_the_index_and_a_worker_gone_hands_it_back() -> Case {
    let store = fleet();
    let a = worker(&store, "a", Role::Single, &[MODEL], true)?;
    let b = worker(&store, "b", Role::Single, &[MODEL], true)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let mut buffers = Buffers::default();
    let now = Instant::now() + Duration::from_secs(120);
    load(&store, a, 0, now);
    load(&store, b, 1, now);
    let asked = token_request(&HASHES[..8]);
    assert_eq!(issued(&door, &asked, now, &mut buffers)?, a);
    let row = u32::from(a > b);
    let later = now + Duration::from_millis(20);
    decide_only(&door, &asked, later, &mut buffers);
    assert_eq!(depths(&buffers), vec![(row, 8)]);

    assert_eq!(hear(&store, a), BatchOutcome::Applied);
    decide_only(&door, &asked, later, &mut buffers);
    assert!(buffers.named.holders.is_empty(), "the index speaks for it");
    // A lease on the heard worker writes nothing.
    let written = slices(&store, Role::Single);
    load(&store, a, 0, later);
    load(&store, b, 1, later);
    let other = token_request(&HASHES[..4]);
    assert_eq!(issued(&door, &other, later, &mut buffers)?, a);
    assert_eq!(slices(&store, Role::Single), written);

    KvEventSink::removed(&*store, a);
    decide_only(&door, &asked, later, &mut buffers);
    assert_eq!(depths(&buffers), vec![(row, 8)]);
    Ok(())
}

/// Where a request's binding names a shadow, its order writes the record
/// of sent prefixes as the plain order does. A prefix sent to a worker
/// with no events returns to it past a lighter one under both documents,
/// and the pool keeps the same slices.
#[test]
fn a_shadowed_order_records_the_prefixes_it_sent_as_the_plain_one() -> Case {
    let returning = |shadowed: bool| -> Case<([bool; 2], Option<usize>, u64)> {
        let store = fleet();
        let a = worker(&store, "a", Role::Single, &[MODEL], false)?;
        let b = worker(&store, "b", Role::Single, &[MODEL], false)?;
        let mut document = merged(1)?;
        if shadowed {
            let live = document.profiles.values().next();
            let twin = live.ok_or("the preset's profile")?.clone();
            document.profiles.insert("twin".into(), twin);
            document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
        }
        store.apply(applied(&document)?)?;
        let door = Door::new(store.clone());
        let asked = token_request(&HASHES[..8]);
        let placed = |now: Instant| -> Case<WorkerId> {
            let decided = door.decide(GATEWAY, &asked, now);
            let decided = decided.map_err(|word| format!("{word:?}"))?;
            let lease = decided.lease.ok_or("a pick's lease")?.lease;
            door.report(GATEWAY, lease.id, gone());
            Ok(lease.booked)
        };
        let now = Instant::now() + Duration::from_secs(120);
        load(&store, a, 0, now);
        load(&store, b, 1, now);
        let cold = placed(now)? == a;
        let later = now + Duration::from_millis(20);
        load(&store, a, 2, later);
        load(&store, b, 0, later);
        let back = placed(later)? == a;
        let agreed = door.shadowed(0, ShadowKind::Agreed);
        Ok(([cold, back], slices(&store, Role::Single), agreed))
    };
    let plain = returning(false)?;
    assert_eq!(plain.0, [true, true], "the prefix returns to its worker");
    assert_eq!(plain.2, 0, "no shadow is named");
    let shadowed = returning(true)?;
    assert_eq!((shadowed.0, shadowed.1), (plain.0, plain.1));
    assert_eq!(shadowed.2, 2, "the twin is asked beside both decisions");
    Ok(())
}

/// The `least_load` preset and, where `shadowed`, the `cache_aware`
/// preset's profile as its binding's shadow: a live profile that is named
/// no row, and a shadow that reads sent prefixes and is named rows of its
/// own.
fn least_load(generation: u64, shadowed: bool) -> Case<PolicySpec> {
    let mut document = PolicySpec::preset("least_load")?;
    document.generation = generation;
    if shadowed {
        let twin = merged(generation)?.profiles.into_values().next();
        let twin = twin.ok_or("the preset's profile")?;
        document.profiles.insert("twin".into(), twin);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    }
    Ok(document)
}

/// What two leases of one prompt left under a document.
#[derive(Debug, PartialEq)]
struct Landed {
    /// Whether each lease took the lighter worker.
    lighter: [bool; 2],
    /// The shadows that agreed, and those that took another worker.
    kinds: [u64; 2],
    /// The record's slices under the document, and behind the next one,
    /// which names no shadow.
    record: [Option<usize>; 2],
}

/// Behind a live profile that is named no row, a pool keeps the record of
/// sent prefixes for a shadow that is named rows of its own. Each lease
/// writes the live decision's worker under the prefixes that shadow's
/// naming folded, so the shadow reads where the requests went: it keeps a
/// returning prompt on the worker least load sent it to, past a lighter
/// one. The live decisions land as they land with no shadow, and a
/// document that names none keeps no record.
#[test]
fn a_shadow_of_its_own_rows_reads_where_the_live_decisions_went() -> Case {
    let landing = |shadowed: bool| -> Case<Landed> {
        let store = fleet();
        let a = worker(&store, "a", Role::Single, &[MODEL], false)?;
        let b = worker(&store, "b", Role::Single, &[MODEL], false)?;
        store.apply(applied(&least_load(1, shadowed)?)?)?;
        let door = Door::new(store.clone());
        let asked = token_request(&HASHES[..8]);
        let placed = |now: Instant| -> Case<WorkerId> {
            let decided = door.decide(GATEWAY, &asked, now);
            let decided = decided.map_err(|word| format!("{word:?}"))?;
            let lease = decided.lease.ok_or("a pick's lease")?.lease;
            door.report(GATEWAY, lease.id, gone());
            Ok(lease.booked)
        };
        let now = Instant::now() + Duration::from_secs(120);
        load(&store, a, 0, now);
        load(&store, b, 1, now);
        let first = placed(now)? == a;
        let later = now + Duration::from_millis(20);
        load(&store, a, 2, later);
        load(&store, b, 0, later);
        let second = placed(later)? == b;
        let kinds = [ShadowKind::Agreed, ShadowKind::OtherTarget];
        let kinds = kinds.map(|kind| door.shadowed(0, kind));
        let kept = slices(&store, Role::Single);
        // The document in force names no shadow again.
        store.apply(applied(&least_load(2, false)?)?)?;
        Ok(Landed {
            lighter: [first, second],
            kinds,
            record: [kept, slices(&store, Role::Single)],
        })
    };
    let plain = landing(false)?;
    assert_eq!(plain.lighter, [true, true], "least load takes the lighter");
    assert_eq!((plain.kinds, plain.record), ([0, 0], [None, None]));
    let shadowed = landing(true)?;
    assert_eq!(
        shadowed.lighter, plain.lighter,
        "the live decisions land the same"
    );
    assert_eq!(
        shadowed.kinds,
        [1, 1],
        "the shadow returns to the first worker"
    );
    assert_eq!(
        shadowed.record,
        [Some(2), None],
        "a slice a lease, then none"
    );
    Ok(())
}

/// A request whose shadow is not asked leaves no entry. Under the floor
/// no row is named for the shadow, so the lease writes nothing: not the
/// request's prefixes, and nothing of the naming the request before left
/// on the thread's buffers.
#[test]
fn a_decision_whose_shadow_is_not_asked_writes_no_record() -> Case {
    let store = fleet();
    let a = worker(&store, "a", Role::Single, &[MODEL], false)?;
    let b = worker(&store, "b", Role::Single, &[MODEL], false)?;
    store.apply(applied(&least_load(1, true)?)?)?;
    let door = Door::new(store.clone());
    let floor = u32::try_from(SHADOW_FLOOR.as_micros())?;
    let placed = |budget_micros: u32, now: Instant| -> Case<WorkerId> {
        let mut asked = token_request(&HASHES[..8]);
        asked.facts.budget_micros = budget_micros;
        let decided = door.decide(GATEWAY, &asked, now);
        let decided = decided.map_err(|word| format!("{word:?}"))?;
        let lease = decided.lease.ok_or("a pick's lease")?.lease;
        door.report(GATEWAY, lease.id, gone());
        Ok(lease.booked)
    };
    let now = Instant::now() + Duration::from_secs(120);
    load(&store, a, 0, now);
    load(&store, b, 1, now);
    assert_eq!(placed(floor, now)?, a);
    assert_eq!(slices(&store, Role::Single), Some(1));
    let later = now + Duration::from_millis(20);
    load(&store, a, 2, later);
    load(&store, b, 0, later);
    assert_eq!(placed(floor - 1, later)?, b);
    assert_eq!(door.shadowed(0, ShadowKind::Skipped), 1);
    assert_eq!(slices(&store, Role::Single), Some(1), "no entry for it");
    Ok(())
}

/// What two leases of one prompt in each of a model's two pools left.
#[derive(Debug, PartialEq)]
struct Beside {
    /// Whether each lease of the reader's pool took its first worker.
    returned: [bool; 2],
    /// The record's slices as each pool reads them: the shadowed
    /// binding's pool, then the reader's.
    record: [Option<usize>; 2],
    /// The shadows that were not asked.
    skipped: u64,
}

/// A model keeps one record of sent prefixes under one bound, and each
/// of its pools that keeps the record writes there. A shadow that would
/// be named rows of its own is not asked where another pool of its model
/// reads the record live: the entries its pool kept for it would stand
/// under the bound the live reader's stand under. That pool's leases and
/// the record's entries are the same with the shadow and without, and
/// each such decision counts its shadow as skipped.
#[test]
fn a_shadow_is_skipped_where_another_pool_of_its_model_reads_the_record() -> Case {
    let landing = |shadowed: bool| -> Case<Beside> {
        let store = fleet();
        let a = worker(&store, "a", Role::Single, &[MODEL], false)?;
        let b = worker(&store, "b", Role::Single, &[MODEL], false)?;
        let c = worker(&store, "c", Role::Prefill, &[MODEL], false)?;
        let d = worker(&store, "d", Role::Prefill, &[MODEL], false)?;
        // Least load decides the single pool, with the shadow or with
        // none. A live cache-aware profile decides the prefill pool.
        let mut document = least_load(1, shadowed)?;
        let aware = merged(1)?.profiles.into_values().next();
        let aware = aware.ok_or("the preset's profile")?;
        document.profiles.insert("aware".into(), aware);
        let mut reader = document.bindings[0].clone();
        reader.matcher.phase = vec![Phase::Prefill];
        reader.profile = Some("aware".into());
        reader.shadow = None;
        document.bindings[0].matcher.phase = vec![Phase::Single];
        document.bindings.push(reader);
        store.apply(applied(&document)?)?;
        let door = Door::new(store.clone());
        let placed = |phase: Phase, now: Instant| -> Case<WorkerId> {
            let mut asked = token_request(&HASHES[..8]);
            asked.facts.phase = phase;
            let decided = door.decide(GATEWAY, &asked, now);
            let decided = decided.map_err(|word| format!("{word:?}"))?;
            let lease = decided.lease.ok_or("a pick's lease")?.lease;
            door.report(GATEWAY, lease.id, gone());
            Ok(lease.booked)
        };
        let now = Instant::now() + Duration::from_secs(120);
        for (id, running) in [(a, 0), (b, 1), (c, 0), (d, 1)] {
            load(&store, id, running, now);
        }
        placed(Phase::Single, now)?;
        let cold = placed(Phase::Prefill, now)? == c;
        let later = now + Duration::from_millis(20);
        for (id, running) in [(a, 2), (b, 0), (c, 2), (d, 0)] {
            load(&store, id, running, later);
        }
        placed(Phase::Single, later)?;
        let back = placed(Phase::Prefill, later)? == c;
        Ok(Beside {
            returned: [cold, back],
            record: [Role::Single, Role::Prefill].map(|role| slices(&store, role)),
            skipped: door.shadowed(0, ShadowKind::Skipped),
        })
    };
    let plain = landing(false)?;
    assert_eq!(plain.returned, [true, true], "the prefix returns");
    // The reader's first lease writes its slice. The prompt's return to
    // the same worker is a fresh hit and writes none.
    assert_eq!(plain.record, [None, Some(1)], "the reader's one slice");
    assert_eq!(plain.skipped, 0, "no shadow is named");
    let shadowed = landing(true)?;
    assert_eq!(
        (shadowed.returned, shadowed.record),
        (plain.returned, plain.record),
        "the reader's leases and the record stand as with no shadow"
    );
    assert_eq!(shadowed.skipped, 2, "the pairing is skipped each time");
    Ok(())
}

#[test]
fn a_prefix_grown_by_a_turn_follows_its_worker() -> Case {
    let store = fleet();
    let a = worker(&store, "a", Role::Single, &[MODEL], false)?;
    let b = worker(&store, "b", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let mut buffers = Buffers::default();
    let now = Instant::now() + Duration::from_secs(120);
    load(&store, a, 0, now);
    load(&store, b, 1, now);
    assert_eq!(
        issued(&door, &token_request(&HASHES[..8]), now, &mut buffers)?,
        a
    );
    let later = now + Duration::from_millis(20);
    load(&store, a, 2, later);
    load(&store, b, 0, later);
    let long = token_request(&HASHES);
    assert_eq!(issued(&door, &long, later, &mut buffers)?, a);
    assert_eq!(depths(&buffers), vec![(u32::from(a > b), 8)]);
    assert_eq!(issued(&door, &long, later, &mut buffers)?, a);
    assert_eq!(depths(&buffers), vec![(u32::from(a > b), 12)]);
    Ok(())
}

#[test]
fn a_request_sent_a_moment_ago_writes_nothing() -> Case {
    let store = fleet();
    worker(&store, "a", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let mut buffers = Buffers::default();
    let now = Instant::now();
    let asked = token_request(&HASHES[..8]);
    issued(&door, &asked, now, &mut buffers)?;
    assert_eq!(slices(&store, Role::Single), Some(1));
    issued(&door, &asked, now, &mut buffers)?;
    assert_eq!(slices(&store, Role::Single), Some(1), "the same tick");
    issued(
        &door,
        &asked,
        now + Duration::from_micros(1 << 20),
        &mut buffers,
    )?;
    assert_eq!(slices(&store, Role::Single), Some(2), "the next tick");
    Ok(())
}

#[test]
fn a_long_request_is_written_in_slices_of_sixteen() -> Case {
    let store = fleet();
    worker(&store, "a", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let mut buffers = Buffers::default();
    let now = Instant::now();
    let hashes: Vec<u64> = (1..=40).collect();
    let asked = token_request(&hashes);
    issued(&door, &asked, now, &mut buffers)?;
    assert_eq!(slices(&store, Role::Single), Some(3));
    issued(&door, &asked, now, &mut buffers)?;
    assert_eq!(depths(&buffers), vec![(0, 40)]);
    assert_eq!(slices(&store, Role::Single), Some(3));
    for end in [16, 32] {
        issued(&door, &token_request(&hashes[..end]), now, &mut buffers)?;
        assert_eq!(depths(&buffers), vec![(0, end as u32)]);
        assert_eq!(
            slices(&store, Role::Single),
            Some(3),
            "every slice used this tick"
        );
    }
    // Sixteen, no more and no fewer: 32 cold blocks take two holds and 33
    // take three.
    for (first, blocks, holds) in [(100, 32, 2), (200, 33, 3)] {
        let before = slices(&store, Role::Single).ok_or("the record")?;
        let cold: Vec<u64> = (first..first + blocks).collect();
        issued(&door, &token_request(&cold), now, &mut buffers)?;
        assert_eq!(slices(&store, Role::Single), Some(before + holds));
    }
    Ok(())
}

#[test]
fn a_worker_whose_events_are_heard_is_neither_written_nor_listed() -> Case {
    let store = fleet();
    let event = worker(&store, "event", Role::Single, &[MODEL], true)?;
    let plain = worker(&store, "plain", Role::Single, &[MODEL], false)?;
    let other = worker(&store, "other", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    assert_eq!(hear(&store, event), BatchOutcome::Applied);
    let door = Door::new(store.clone());
    let mut buffers = Buffers::default();
    let now = Instant::now() + Duration::from_secs(120);
    load(&store, event, 0, now);
    load(&store, plain, 2, now);
    load(&store, other, 2, now);
    let asked = token_request(&HASHES[..8]);
    assert_eq!(issued(&door, &asked, now, &mut buffers)?, event);
    assert_eq!(slices(&store, Role::Single), Some(0));

    let later = now + Duration::from_millis(20);
    load(&store, event, 2, later);
    load(&store, plain, 0, later);
    assert_eq!(issued(&door, &asked, later, &mut buffers)?, plain);
    assert_eq!(slices(&store, Role::Single), Some(1));
    assert_eq!(
        worker(&store, "plain", Role::Single, &[MODEL], true)?,
        plain
    );
    let root = StoredBlock {
        seq_hash: kv_index::SequenceHash(1),
        content_hash: kv_index::ContentHash(1),
    };
    let event = KvEvent::Stored {
        parent: None,
        blocks: vec![root],
    };
    assert_eq!(
        store.admit(
            plain,
            0,
            BatchFacts {
                sequence: 1,
                ..BatchFacts::default()
            },
            &[event]
        ),
        BatchOutcome::Applied
    );
    decide_only(&door, &asked, later, &mut buffers);
    assert_eq!(buffers.named.holders.len(), 1);
    let holder = buffers.named.holders[0];
    assert_eq!((holder.overlap_blocks, holder.predicted_blocks), (1, 0));
    // The same numbers as windows of a text ask no cache events.
    let mut text = asked;
    text.facts.hash_kind = HashKind::Text;
    decide_only(&door, &text, later, &mut buffers);
    assert!(buffers.named.holders.is_empty());
    Ok(())
}

#[test]
fn a_namespace_keeps_its_prefixes_apart() -> Case {
    let store = fleet();
    worker(&store, "a", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store);
    let now = Instant::now();
    let mut buffers = Buffers::default();
    let mut asked = token_request(&HASHES);
    issued(&door, &asked, now, &mut buffers)?;
    decide_only(&door, &asked, now, &mut buffers);
    assert_eq!(depths(&buffers), vec![(0, 12)]);
    asked.facts.namespace_seed += 1;
    decide_only(&door, &asked, now, &mut buffers);
    assert!(buffers.named.holders.is_empty());
    Ok(())
}

#[test]
fn a_text_request_is_recorded_and_absent_hashes_are_not() -> Case {
    let store = fleet();
    worker(&store, "a", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let now = Instant::now();
    let mut buffers = Buffers::default();
    issued(&door, &token_request(&[]), now, &mut buffers)?;
    assert_eq!(slices(&store, Role::Single), Some(0));
    let mut asked = token_request(&HASHES);
    asked.facts.hash_kind = HashKind::Text;
    issued(&door, &asked, now, &mut buffers)?;
    assert_eq!(slices(&store, Role::Single), Some(1));
    decide_only(&door, &asked, now, &mut buffers);
    assert_eq!(depths(&buffers), vec![(0, 12)]);
    Ok(())
}

#[test]
fn a_refused_lease_writes_nothing() -> Case {
    let store = fleet();
    worker(&store, "a", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let now = Instant::now();
    let asked = token_request(&HASHES);
    let mut buffers = Buffers::default();
    let term = store.term() + 1;
    let refused = store.deciding(|world| {
        let taken = taken(&asked, now);
        let entry = entry(world, &taken);
        door.order(world, 0, term, &taken, &mut buffers, entry)
    });
    assert!(matches!(refused, Err(Refused::Unleased)));
    assert_eq!(slices(&store, Role::Single), Some(0));
    decide_only(&door, &asked, now, &mut buffers);
    assert!(buffers.named.holders.is_empty());
    Ok(())
}

#[test]
fn a_pairs_lease_writes_nothing() -> Case {
    let store = fleet();
    worker(&store, "prefill", Role::Prefill, &[MODEL], false)?;
    worker(&store, "decode", Role::Decode, &[MODEL], false)?;
    let mut document = PolicySpec::pair_presets("cache_aware", "least_load")?;
    document.generation = 1;
    for profile in document.profiles.values_mut() {
        profile
            .filter
            .retain(|filter| !matches!(filter.name(), "has-capacity" | "has-room"));
    }
    use_sent_prefixes(&mut document, AffinitySource::Merged);
    store.apply(applied(&document)?)?;
    let door = Door::new(store.clone());
    let mut asked = token_request(&HASHES);
    asked.facts.phase = Phase::Prefill;
    let mut buffers = Buffers::default();
    let now = Instant::now();
    // A prior naming left keys in this seat's buffer.
    decide_only(&door, &asked, now, &mut buffers);
    let decided = order(&door, &asked, now, &mut buffers).map_err(|word| format!("{word:?}"))?;
    assert!(decided.pair());
    assert_eq!(slices(&store, Role::Prefill), Some(0));
    Ok(())
}

fn decide_only(door: &Door, asked: &Request<'_>, now: Instant, buffers: &mut Buffers<()>) {
    door.store().deciding(|world| {
        decide_in(world, &taken(asked, now), buffers, &door.parts[0]);
    });
}

#[test]
fn a_decision_that_names_no_row_writes_nothing_of_the_one_before() -> Case {
    const BESIDE: &str = "beside";
    let store = fleet();
    let id = worker(&store, "a", Role::Single, &[MODEL, BESIDE], false)?;
    let mut document = merged(1)?;
    document
        .profiles
        .extend(PolicySpec::preset("least_load")?.profiles);
    document.bindings.insert(
        0,
        BindingSpec {
            matcher: Match {
                model: BESIDE.into(),
                ..Match::default()
            },
            profile: Some("least_load".into()),
            ..BindingSpec::default()
        },
    );
    store.apply(applied(&document)?)?;
    let door = Door::new(store.clone());
    let mut buffers = Buffers::default();
    let now = Instant::now();
    let asked = token_request(&HASHES);
    decide_only(&door, &asked, now, &mut buffers);
    let mut beside = token_request(&[]);
    beside.facts.model = BESIDE;
    assert_eq!(issued(&door, &beside, now, &mut buffers)?, id);
    assert_eq!(slices(&store, Role::Single), Some(0));
    decide_only(&door, &asked, now, &mut buffers);
    assert!(buffers.named.holders.is_empty());
    Ok(())
}

/// A scorer that reads nothing and panics, once armed, where it is asked
/// what it needs.
#[derive(Debug)]
struct AskedLate(Arc<AtomicBool>);

impl Lifecycle for AskedLate {}

impl Scorer for AskedLate {
    fn name(&self) -> &'static str {
        "asked-late"
    }

    fn needs(&self) -> Needs {
        assert!(!self.0.load(Ordering::Relaxed), "asked after its build");
        Needs::default()
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// A membership write under a profile that reads sent prefixes asks no
/// stage what it needs: the store reads what the engine's build kept of
/// each profile, as a decision does.
#[test]
fn a_membership_write_asks_no_stage_what_it_needs() -> Case {
    let store = fleet();
    worker(&store, "a", Role::Single, &[MODEL], false)?;
    let armed = Arc::new(AtomicBool::new(false));
    let stage = Arc::clone(&armed);
    let mut catalog = Catalog::builtin();
    catalog.scorer("asked-late", move || {
        Box::new(AskedLate(Arc::clone(&stage)))
    })?;
    let mut document = merged(1)?;
    for profile in document.profiles.values_mut() {
        profile.score.push("asked-late".into());
    }
    let digest = document.digest()?;
    store.apply(AppliedSpec::validate_digested(
        &document,
        digest,
        &catalog,
        &[],
    )?)?;
    assert_eq!(slices(&store, Role::Single), Some(0), "the pool's record");
    armed.store(true, Ordering::Relaxed);
    // A member joins: every pool is walked again, by what the build kept.
    worker(&store, "b", Role::Single, &[MODEL], false)?;
    assert_eq!(slices(&store, Role::Single), Some(0));
    Ok(())
}

/// A profile that is named no row reads no holder: its pool keeps no
/// record, and a lease over it writes none.
#[test]
fn a_profile_named_no_row_keeps_no_record_of_sent_prefixes() -> Case {
    let store = fleet();
    let id = worker(&store, "a", Role::Single, &[MODEL], false)?;
    let mut document = merged(1)?;
    for profile in document.profiles.values_mut() {
        // A second warm-up ruling: the profile keeps its column.
        profile.score.push(
            ScoreStage::WarmupSlice(spec::WarmupSliceParams {
                secs: Some(15),
                ..Default::default()
            })
            .into(),
        );
    }
    let spec = applied(&document)?;
    assert_eq!(spec.engine().named_for(MODEL, Phase::Single), 0);
    assert!(!spec.engine().reads_sent_for(MODEL, Phase::Single));
    store.apply(spec)?;
    assert_eq!(slices(&store, Role::Single), None);
    let door = Door::new(store.clone());
    let mut buffers = Buffers::default();
    let asked = token_request(&HASHES);
    assert_eq!(issued(&door, &asked, Instant::now(), &mut buffers)?, id);
    assert_eq!(slices(&store, Role::Single), None);
    Ok(())
}

#[test]
fn a_profile_that_reads_sent_prefixes_keeps_no_decider() -> Case {
    for source in [AffinitySource::Merged, AffinitySource::Predictor] {
        let mut document = merged(1)?;
        use_sent_prefixes(&mut document, source);
        document.bindings[0].matcher.model = MODEL.into();
        let spec = applied(&document)?;
        assert!(spec.needs().sent_prefixes);
        assert!(spec.engine().reads_sent_for(MODEL, Phase::Single));
        assert!(!spec.engine().reads_sent_for("unbound", Phase::Single));
        assert!(!spec.engine().keeps_deciders());
    }
    // The preset names no source and reads them too. `index` does not.
    assert!(applied(&merged(1)?)?.needs().sent_prefixes);
    let mut index = merged(1)?;
    use_sent_prefixes(&mut index, AffinitySource::Index);
    assert!(!applied(&index)?.needs().sent_prefixes);
    let mut old = serde_json::to_value(Needs::default())?;
    old.as_object_mut()
        .ok_or("the needs fields")?
        .remove("sent_prefixes");
    assert!(!serde_json::from_value::<Needs>(old)?.sent_prefixes);
    Ok(())
}

#[test]
fn readers_and_a_writer_share_one_record() -> Case {
    let store = fleet();
    let worker = worker(&store, "a", Role::Single, &[MODEL], false)?;
    store.apply(applied(&merged(1)?)?)?;
    let door = Door::new(store.clone());
    let now = Instant::now();
    let together = Barrier::new(4);
    thread::scope(|scope| {
        for thread in 0..4 {
            let (door, together) = (&door, &together);
            scope.spawn(move || {
                together.wait();
                for n in 0..32 {
                    let hashes = [1000 + thread * 32 + n, 2, 3, 4, 5, 6, 7, 8];
                    let decided = door.decide(GATEWAY, &token_request(&hashes), now);
                    let lease = decided
                        .expect("a decision")
                        .lease
                        .expect("a pick's lease")
                        .lease;
                    assert_eq!(lease.booked, worker);
                    door.report(GATEWAY, lease.id, gone());
                }
            });
        }
    });
    let mut buffers = Buffers::default();
    let mut written = 0;
    for n in 0..128 {
        let hashes = [1000 + n, 2, 3, 4, 5, 6, 7, 8];
        decide_only(&door, &token_request(&hashes), now, &mut buffers);
        // A write that met another thread at the record left its prefix
        // out whole.
        match depths(&buffers).as_slice() {
            [] => {}
            [(0, 8)] => written += 1,
            part => return Err(format!("a prefix written in part: {part:?}").into()),
        }
    }
    assert_eq!(slices(&store, Role::Single), Some(written));
    Ok(())
}
