//! The naming of a decision's rows from a host's index of cache events and
//! the engine's record of sent prefixes.

use std::{sync::atomic::AtomicUsize, thread, time::Duration};

use kv_index::{
    compute_request_content_hashes, request_prefix_hashes, ChainBlockMap, ChainIndex, ContentHash,
    StoredBlock,
};
use smg_worker::test_support::ready_worker;

use super::*;
use crate::{
    document::spec::{PolicySpec, ScoreStage, WarmupSliceParams},
    engine::{
        decision::{Pick, Shed, ShedReason, Target, Targets, Trace},
        framework::Scratch,
        prompt_hashes::{HashSeeds, PromptHashes},
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, Health, PoolFacts},
    },
};

const MODEL: &str = "model-a";
/// Tokens a block.
const BLOCK: u32 = 16;
/// An instant of the decision clock, in microseconds.
const NOW: u64 = 10_000_000;
const SECOND: u64 = 1_000_000;

/// The document of a built-in preset.
fn preset(name: &str) -> AppliedSpec {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    AppliedSpec::validate(&document).expect("the preset validates")
}

/// The cache-aware preset with its warm-up stage ruling by `secs` seconds
/// and four blocks, so a case's few blocks make a level.
fn warming_by(secs: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the preset's profile");
    profile.score[0] = ScoreStage::WarmupSlice(WarmupSliceParams {
        weight: 8.0,
        secs: Some(secs),
        blocks: Some(4),
        ..WarmupSliceParams::default()
    })
    .into();
    AppliedSpec::validate(&document).expect("the document validates")
}

fn address(at: usize) -> String {
    format!("http://worker-{at:03}:8000")
}

fn pool(count: usize) -> (Vec<Arc<dyn Worker>>, Vec<WorkerId>) {
    let workers = (0..count).map(|at| ready_worker(&address(at), MODEL));
    let ids = (0..count).map(|at| WorkerId(at as u64 + 1));
    (workers.collect(), ids.collect())
}

/// A host's indexes: one model's chain index, and how often it was asked.
#[derive(Default)]
struct Host {
    index: Option<Arc<ChainIndex>>,
    asked: AtomicUsize,
}

impl CacheIndexes for Host {
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        let index = self.index.as_ref().filter(|_| model == MODEL)?;
        Some(Arc::clone(index) as Arc<dyn CacheIndex>)
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        Some(BLOCK)
    }
}

/// An engine's affinity over a host with a chain index, or with none.
fn hosted(indexed: bool) -> (Affinity, Arc<Host>, Option<Arc<ChainIndex>>) {
    let index = indexed.then(|| Arc::new(ChainIndex::new()));
    let host = Arc::new(Host {
        index: index.clone(),
        asked: AtomicUsize::new(0),
    });
    let affinity = Affinity::default();
    affinity.set_indexes(Arc::clone(&host) as Arc<dyn CacheIndexes>);
    (affinity, host, index)
}

/// Stores `hashes`, a prompt's blocks from its first, for the worker at
/// `at`, as its event stream would.
fn store(index: &ChainIndex, at: usize, hashes: &[u64]) {
    let number = index.intern_worker(&address(at)).expect("a number");
    let contents: Vec<ContentHash> = hashes.iter().map(|&hash| ContentHash(hash)).collect();
    let blocks: Vec<StoredBlock> = contents
        .iter()
        .zip(request_prefix_hashes(&contents))
        .map(|(&content_hash, seq_hash)| StoredBlock {
            seq_hash,
            content_hash,
        })
        .collect();
    index
        .apply_stored(number, &blocks, None, &mut ChainBlockMap::new())
        .expect("the blocks are stored");
}

/// A prompt of `blocks` whole blocks, its tokens counted from `first`.
fn prompt(first: u32, blocks: u32) -> PromptHashes {
    let tokens: Vec<u32> = (first..first + blocks * BLOCK).collect();
    let spec = preset("cache_aware");
    PromptHashes::under_document(&spec, Some(BLOCK), Some(&tokens), None, HashSeeds::PLAIN)
}

/// The hashes of `prompt`'s blocks.
fn hashes_of(prompt: &PromptHashes) -> Vec<u64> {
    let mut facts = RequestFacts::default();
    prompt.write_to(&mut facts);
    facts.block_hashes.to_vec()
}

/// A request of `prompt` for the model, decided at `now` under `seed`.
fn request(prompt: &PromptHashes, seed: u64, now: u64) -> RequestFacts<'_> {
    let mut facts = RequestFacts {
        model: MODEL,
        seed,
        now_micros: now,
        ..RequestFacts::default()
    };
    prompt.write_to(&mut facts);
    facts
}

/// What a decision was named.
#[derive(Debug, Default, PartialEq)]
struct Named {
    holders: Vec<(u32, u32, u32)>,
    warming: Vec<u32>,
    thin: bool,
}

fn picked(position: u32, id: WorkerId) -> Decision {
    let mut targets = Targets::new();
    targets.push(Target {
        position,
        id,
        ..Target::default()
    });
    Decision::Pick(Pick {
        targets,
        trace: Trace::default(),
        session_verdict: SessionVerdict::None,
    })
}

fn shed() -> Decision {
    Decision::Shed(Shed {
        reason: ShedReason::PolicyDeclined,
        emptied_by: "a case",
        retry_after_secs: 0,
        wait_allowed: false,
        trace: Trace::default(),
    })
}

/// One list and what the engine keeps of it.
struct Bed {
    affinity: Affinity,
    pool: PoolAffinity,
    spec: AppliedSpec,
    workers: Vec<Arc<dyn Worker>>,
    ids: Vec<WorkerId>,
}

impl Bed {
    fn new(affinity: Affinity, spec: AppliedSpec, count: usize) -> Self {
        let (workers, ids) = pool(count);
        Self {
            affinity,
            pool: PoolAffinity::default(),
            spec,
            workers,
            ids,
        }
    }

    /// Decides `facts` by picking the worker at `target`, or by a shed, and
    /// says what the decision was named. `None` where it was named no row.
    fn decide(&self, facts: &RequestFacts<'_>, target: Option<u32>) -> Option<Named> {
        let mut named = None;
        let decision = decide(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            facts,
            |facts| {
                named = facts.named.map(|rows| Named {
                    holders: rows
                        .holders
                        .iter()
                        .map(|holder| (holder.row, holder.overlap_blocks, holder.predicted_blocks))
                        .collect(),
                    warming: rows.warming.to_vec(),
                    thin: rows.thin,
                });
                match target {
                    Some(position) => picked(position, self.ids[position as usize]),
                    None => shed(),
                }
            },
        );
        assert_eq!(decision.pick().is_some(), target.is_some());
        named
    }

    /// The holders a request of `prompt` is named, decided by a shed.
    fn holders(&self, prompt: &PromptHashes, now: u64) -> Vec<(u32, u32, u32)> {
        let named = self.decide(&request(prompt, 7, now), None);
        named.expect("rows named").holders
    }
}

/// The request's token blocks hash as an engine's stored blocks do: the
/// index finds a request by the hashes its facts carry.
#[test]
fn a_prompts_block_hashes_are_the_hashes_an_engine_stores() {
    let tokens: Vec<u32> = (100..100 + 5 * BLOCK + 3).collect();
    let spec = preset("cache_aware");
    let hashes =
        PromptHashes::under_document(&spec, Some(BLOCK), Some(&tokens), None, HashSeeds::PLAIN);
    let stored: Vec<u64> = compute_request_content_hashes(&tokens, BLOCK as usize)
        .iter()
        .map(|hash| hash.0)
        .collect();
    assert_eq!(stored.len(), 5);
    assert_eq!(hashes_of(&hashes), stored);
}

/// The index names the workers that hold a leading part of the request,
/// each with its depth, ascending by row. A worker that holds another
/// prompt is not named, and a fleet just admitted names no warming row.
#[test]
fn the_index_names_a_requests_holders_by_their_depth() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let hashes = hashes_of(&asked);
    store(&index, 5, &hashes[..2]);
    store(&index, 2, &hashes);
    store(&index, 6, &hashes_of(&prompt(9_000, 4)));
    let bed = Bed::new(affinity, preset("cache_aware"), 8);
    let named = bed.decide(&request(&asked, 7, NOW), None).expect("rows");
    assert_eq!(named.holders, [(2, 4, 0), (5, 2, 0)]);
    assert_eq!(named.warming, [] as [u32; 0]);
    // A request no worker holds a block of names nobody.
    assert!(bed.holders(&prompt(50_000, 4), NOW).is_empty());
}

/// Past the profile's sample the deepest depth is cut by a draw from the
/// decision's seed: sixteen rows, ascending, none twice, and another seed
/// draws another sixteen.
#[test]
fn the_cap_cuts_the_holders_of_one_depth_by_a_draw() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 2);
    for at in 0..24 {
        store(&index, at, &hashes_of(&asked));
    }
    let bed = Bed::new(affinity, preset("cache_aware"), 24);
    let rows = |seed| {
        let named = bed.decide(&request(&asked, seed, NOW), None).expect("rows");
        let rows: Vec<u32> = named.holders.iter().map(|&(row, _, _)| row).collect();
        assert_eq!(rows.len(), 16);
        assert!(rows.is_sorted_by(|a, b| a < b), "{rows:?}");
        assert!(rows.iter().all(|&row| row < 24));
        assert!(named.holders.iter().all(|&(_, held, _)| held == 2));
        rows
    };
    let first = rows(1);
    assert_eq!(first, rows(1), "one seed draws one set");
    assert!((2..40).any(|seed| rows(seed) != first));
}

/// With no index the record of sent prefixes speaks for every worker: a
/// pick writes the request's prefix for its worker, a later request that
/// begins alike names that worker by the blocks it was sent, and a worker
/// told gone is named no more.
#[test]
fn the_record_names_the_worker_a_prefix_was_sent_to() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    assert!(bed.holders(&asked, NOW).is_empty());
    // A shed writes nothing.
    assert!(bed.holders(&asked, NOW).is_empty());
    let named = bed.decide(&request(&asked, 7, NOW), Some(1));
    assert!(named.expect("rows").holders.is_empty());
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4)]);
    // A longer prompt that begins with it is named by the four blocks sent.
    let longer = prompt(1_000, 6);
    assert_eq!(bed.holders(&longer, NOW), [(1, 0, 4)]);
    // A second worker sent the longer prompt is the deeper holder of it,
    // and both hold the shorter one.
    bed.decide(&request(&longer, 7, NOW), Some(3));
    assert_eq!(bed.holders(&longer, NOW), [(3, 0, 6)]);
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4), (3, 0, 4)]);
    bed.affinity.forget(bed.ids[3]);
    assert_eq!(bed.holders(&longer, NOW), [(1, 0, 4)]);
    bed.affinity.sweep();
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4)]);
}

/// A prompt of more blocks than one turn of the record's lock writes is
/// recorded to its end.
#[test]
fn a_long_prompt_is_recorded_whole() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 40);
    let blocks = hashes_of(&asked).len() as u32;
    assert!(blocks > 32, "the document reads {blocks} blocks");
    bed.decide(&request(&asked, 7, NOW), Some(2));
    assert_eq!(bed.holders(&asked, NOW), [(2, 0, blocks)]);
}

/// The index speaks for a worker that holds blocks in it and the record for
/// every other: a pick of a worker the index speaks for writes no entry,
/// and no row is named twice.
#[test]
fn the_index_speaks_for_its_workers_and_the_record_for_the_others() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 0, &hashes_of(&asked)[..2]);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    bed.decide(&request(&asked, 7, NOW), Some(0));
    assert_eq!(bed.holders(&asked, NOW), [(0, 2, 0)]);
    bed.decide(&request(&asked, 7, NOW), Some(3));
    assert_eq!(bed.holders(&asked, NOW), [(0, 2, 0), (3, 0, 4)]);
}

/// A worker whose index is empty beside a fleet that holds a level is thin:
/// it is named as the list's one warming row, whatever its age.
#[test]
fn a_worker_with_nothing_indexed_beside_a_warm_fleet_is_named_thin() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let held = hashes_of(&prompt(1_000, 8));
    for at in 0..3 {
        store(&index, at, &held);
    }
    let bed = Bed::new(affinity, warming_by(60), 4);
    let named = bed.decide(&request(&prompt(50_000, 2), 7, NOW), None);
    let named = named.expect("rows");
    assert_eq!((named.warming, named.thin), (vec![3], true));
}

/// A fleet that is all young names nobody. A worker still short of its
/// first blocks beside workers that grew past them is named until its
/// window ends, and no longer.
#[test]
fn a_young_worker_is_named_until_its_window_ends() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let first = hashes_of(&prompt(1_000, 8));
    for at in 0..4 {
        store(&index, at, &first);
    }
    let bed = Bed::new(affinity, warming_by(1), 4);
    let other = prompt(50_000, 2);
    let warming = |now| {
        let named = bed.decide(&request(&other, 7, now), None).expect("rows");
        (named.warming, named.thin)
    };
    // Every worker was admitted a moment ago and none grew: nobody to
    // catch up to.
    assert_eq!(warming(NOW), (vec![], false));
    // Three workers gain their first blocks. The table is read again a
    // second later, when each worker's admission is some milliseconds old.
    let second = hashes_of(&prompt(2_000, 8));
    for at in 0..3 {
        store(&index, at, &second);
    }
    thread::sleep(Duration::from_millis(3));
    assert_eq!(warming(NOW + SECOND), (vec![3], false));
    // By the table that stands, past the end of the row's own window.
    assert_eq!(warming(NOW + 2 * SECOND - 1), (vec![], false));
}

/// A list's table is read once, then again by a decision a second later:
/// the decisions between read what stands and ask the host nothing.
#[test]
fn a_lists_table_stands_for_a_second() {
    let (affinity, host, _) = hosted(true);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    let reads = || host.asked.load(Ordering::Relaxed);
    assert_eq!(reads(), 0);
    bed.holders(&asked, NOW);
    assert_eq!(reads(), 1);
    for step in 1..100 {
        bed.holders(&asked, NOW + step * (SECOND / 100) - 1);
    }
    assert_eq!(reads(), 1);
    bed.holders(&asked, NOW + SECOND);
    assert_eq!(reads(), 2);
    // Blocks stored behind a read are named by the next one.
    let index = host.index.as_ref().expect("the index");
    store(index, 2, &hashes_of(&asked));
    assert!(bed.holders(&asked, NOW + SECOND + 1).is_empty());
    assert_eq!(bed.holders(&asked, NOW + 2 * SECOND), [(2, 4, 0)]);
}

/// A profile whose stages read no affinity is named no row: its decision
/// is handed the request as it stands, and the host is asked nothing.
#[test]
fn a_profile_that_reads_no_affinity_is_named_no_row() {
    let (affinity, host, _) = hosted(true);
    let bed = Bed::new(affinity, preset("least_load"), 4);
    let asked = prompt(1_000, 4);
    assert_eq!(bed.decide(&request(&asked, 7, NOW), Some(1)), None);
    assert_eq!(host.asked.load(Ordering::Relaxed), 0);
}

/// The records of `count` ready workers, as a host writes them.
fn records(ids: &[WorkerId]) -> Vec<Candidate> {
    let rows = (0u32..).zip(ids).map(|(position, &id)| Candidate {
        position,
        id,
        health: Health {
            healthy: true,
            ..Health::default()
        },
        ..Candidate::default()
    });
    rows.collect()
}

/// The stock profile's whole decision over the rows named: a request goes
/// to the worker that holds its prefix by the index, whatever the seed
/// draws, and with no index to the worker its prefix was sent to before.
#[test]
fn the_engine_decides_by_the_rows_named() {
    const WORKERS: usize = 32;
    for indexed in [true, false] {
        let (affinity, _, index) = hosted(indexed);
        let asked = prompt(1_000, 8);
        if let Some(index) = &index {
            store(index, 7, &hashes_of(&asked));
        }
        let bed = Bed::new(affinity, preset("cache_aware"), WORKERS);
        let rows = records(&bed.ids);
        let facts = PoolFacts {
            pool_size: WORKERS as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let mut scratch = Scratch::new();
        let mut place = |seed| {
            let request = request(&asked, seed, NOW);
            let decision = decide(
                &bed.affinity,
                &bed.pool,
                &bed.spec,
                &bed.workers,
                &bed.ids,
                &request,
                |request| {
                    let engine = bed.spec.engine();
                    engine.decide(
                        request,
                        &facts,
                        &rows,
                        &Constraints::default(),
                        &mut scratch,
                    )
                },
            );
            let pick = decision.pick().expect("a pick");
            let target = pick.targets.first().expect("a target");
            assert_eq!(target.id, bed.ids[target.position as usize]);
            target.position
        };
        let first = place(1);
        if indexed {
            assert_eq!(first, 7);
        }
        for seed in 2..64 {
            assert_eq!(place(seed), first, "indexed: {indexed}, seed {seed}");
        }
    }
}
