//! The naming of a decision's rows from a host's index of cache events and
//! the engine's record of sent prefixes.

use std::{
    cell::RefCell,
    sync::atomic::{AtomicU32, AtomicUsize},
    thread,
    time::Duration,
};

use kv_index::{
    compute_request_content_hashes, request_prefix_hashes, ChainBlockMap, ChainIndex, ContentHash,
    StoredBlock,
};
use smg_worker::test_support::ready_worker;

use super::*;
use crate::{
    document::spec::{
        AffinitySource, IndexSpec, PolicySpec, PrefixAffinityParams, ScoreStage, WarmupSliceParams,
    },
    engine::{
        decision::{Pick, Shed, ShedReason, Target, Targets, Trace},
        framework::Scratch,
        prompt_hashes::{HashSeeds, PromptHashes},
        records,
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, Health, PoolFacts},
    },
};

const MODEL: &str = "model-a";
/// Tokens a block.
const BLOCK: u32 = 16;
/// Bytes a window of text: a block of tokens in bytes.
const WINDOW: usize = (BLOCK * TEXT_BYTES_PER_TOKEN) as usize;
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

/// The cache-aware preset with its prefix stage reading placements from
/// `source` alone.
fn reading(source: AffinitySource) -> AppliedSpec {
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the preset's profile");
    let stage = profile
        .score
        .iter_mut()
        .find(|stage| stage.name() == "prefix-affinity")
        .expect("the prefix stage");
    *stage = ScoreStage::PrefixAffinity(PrefixAffinityParams {
        weight: 4.0,
        source: Some(source),
        ..PrefixAffinityParams::default()
    })
    .into();
    AppliedSpec::validate(&document).expect("the document validates")
}

fn address(at: usize) -> String {
    format!("http://worker-{at:03}:8000")
}

/// A list of `count` ready workers, with the id every decision knows each
/// by.
fn pool(count: usize) -> (Vec<Arc<dyn Worker>>, Vec<WorkerId>) {
    let workers: Vec<Arc<dyn Worker>> = (0..count)
        .map(|at| ready_worker(&address(at), MODEL))
        .collect();
    let ids = workers
        .iter()
        .map(|worker| records::worker_id(worker.as_ref()))
        .collect();
    (workers, ids)
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

/// The token ids of a prompt of `blocks` whole blocks, counted from `first`.
fn tokens(first: u32, blocks: u32) -> Vec<u32> {
    (first..first + blocks * BLOCK).collect()
}

/// A prompt of `blocks` whole blocks, its tokens counted from `first`, as
/// its host hashes it.
fn prompt(first: u32, blocks: u32) -> PromptHashes {
    hashed(&tokens(first, blocks))
}

/// What a host hashes of a tokenized prompt.
fn hashed(tokens: &[u32]) -> PromptHashes {
    let spec = preset("cache_aware");
    PromptHashes::under_document(&spec, Some(BLOCK), Some(tokens), None, HashSeeds::PLAIN)
}

/// What a host hashes of a prompt it holds as text.
fn hashed_text(text: &str) -> PromptHashes {
    let spec = preset("cache_aware");
    PromptHashes::under_document(&spec, Some(BLOCK), None, Some(text), HashSeeds::PLAIN)
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

/// The token ids a host lends, in no cache partition.
fn lend(tokens: &[u32]) -> PromptLent<'_> {
    PromptLent {
        tokens: Some(tokens),
        ..PromptLent::default()
    }
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
    /// How many times a decision asked its host for the prompt.
    asked: AtomicU32,
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
            asked: AtomicU32::new(0),
        }
    }

    /// Decides `facts`, whose host lends `lent`, by picking the worker at
    /// `target`, or by a shed, and says what the decision was named. `None`
    /// where it was named no row.
    fn decide_lent(
        &self,
        facts: &RequestFacts<'_>,
        lent: Option<PromptLent<'_>>,
        target: Option<u32>,
    ) -> Option<Named> {
        let named = RefCell::new(None);
        let prompt = || {
            self.asked.fetch_add(1, Ordering::Relaxed);
            lent
        };
        let decision = decide(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            &mut { *facts },
            prompt,
            |facts| {
                *named.borrow_mut() = facts.named.map(|rows| Named {
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
        named.into_inner()
    }

    /// [`Self::decide_lent`] for a host that lends no prompt.
    fn decide(&self, facts: &RequestFacts<'_>, target: Option<u32>) -> Option<Named> {
        self.decide_lent(facts, None, target)
    }

    /// The holders a request of `prompt` is named where its host lends no
    /// prompt, decided by a shed.
    fn holders(&self, prompt: &PromptHashes, now: u64) -> Vec<(u32, u32, u32)> {
        let named = self.decide(&request(prompt, 7, now), None);
        named.expect("rows named").holders
    }

    /// The holders a request of `tokens` is named where its host lends
    /// them, decided by a shed: nothing is recorded.
    fn sent(&self, tokens: &[u32], now: u64) -> Vec<(u32, u32, u32)> {
        let hashes = hashed(tokens);
        let named = self.decide_lent(&request(&hashes, 7, now), Some(lend(tokens)), None);
        named.expect("rows named").holders
    }

    /// Sends a request of `tokens` to the worker at `target`, and says the
    /// holders the decision was named.
    fn send(&self, tokens: &[u32], now: u64, target: u32) -> Vec<(u32, u32, u32)> {
        let hashes = hashed(tokens);
        let facts = request(&hashes, 7, now);
        let named = self.decide_lent(&facts, Some(lend(tokens)), Some(target));
        named.expect("rows named").holders
    }

    /// As [`Self::sent`] and [`Self::send`], for a request its host holds
    /// as `text`.
    fn text(&self, text: &str, target: Option<u32>) -> Vec<(u32, u32, u32)> {
        let hashes = hashed_text(text);
        let lent = PromptLent {
            text: Some(text),
            ..PromptLent::default()
        };
        let named = self.decide_lent(&request(&hashes, 7, NOW), Some(lent), target);
        named.expect("rows named").holders
    }

    /// The model's record of sent prefixes.
    fn record(&self) -> Arc<SentPrefixes> {
        let records = self.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    }

    /// The list's column for `facts` at `phase`.
    fn column(&self, facts: &RequestFacts<'_>, phase: Phase) -> Vec<AffinityFacts> {
        let mut filled = vec![AffinityFacts {
            overlap_blocks: 99,
            ..AffinityFacts::default()
        }];
        column(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            phase,
            facts,
            &mut filled,
        );
        filled
    }
}

/// The cache-aware preset on both sides of the pair plan.
fn paired() -> AppliedSpec {
    let mut document =
        PolicySpec::pair_presets("cache_aware", "cache_aware").expect("the pair presets");
    document.generation = 1;
    AppliedSpec::validate(&document).expect("the pair document validates")
}

/// What each row of a column holds: by the index, then by the record.
fn held(column: &[AffinityFacts]) -> Vec<(u32, u32)> {
    let held = |row: &AffinityFacts| (row.overlap_blocks, row.predicted_blocks);
    column.iter().map(held).collect()
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

/// With no index of cache events a tokenized request is named by the
/// model's token tree. A pick records the request's prefix for its worker
/// in the walk that named the request, a later request that begins alike is
/// named that worker by the blocks it was sent, a shed records nothing, and
/// a worker told gone is named no more.
#[test]
fn a_token_tree_names_the_worker_a_prefix_was_sent_to() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    assert!(bed.sent(&asked, NOW).is_empty());
    // A shed recorded nothing.
    assert!(bed.sent(&asked, NOW).is_empty());
    // The pick's own decision is named what stood before it.
    assert!(bed.send(&asked, NOW, 1).is_empty());
    assert_eq!(bed.sent(&asked, NOW), [(1, 0, 4)]);
    // A longer prompt that begins with it is named by the four blocks sent.
    let longer = tokens(1_000, 6);
    assert_eq!(bed.sent(&longer, NOW), [(1, 0, 4)]);
    // A second worker sent the longer prompt is the deeper holder of it,
    // and both hold the shorter one.
    assert_eq!(bed.send(&longer, NOW, 3), [(1, 0, 4)]);
    assert_eq!(bed.sent(&longer, NOW), [(3, 0, 6)]);
    assert_eq!(bed.sent(&asked, NOW), [(1, 0, 4), (3, 0, 4)]);
    bed.affinity.forget(&address(3));
    assert_eq!(bed.sent(&longer, NOW), [(1, 0, 4)]);
    assert_eq!(bed.sent(&asked, NOW), [(1, 0, 4)]);
    // A turn that goes on from a recorded prompt is recorded for the worker
    // that holds the rest.
    bed.send(&longer, NOW, 1);
    assert_eq!(bed.sent(&longer, NOW), [(1, 0, 6)]);
    // A prompt under one block is one no engine's cache serves: the tree
    // keeps none, and the request is decided all the same.
    let short: Vec<u32> = (7_000..7_000 + BLOCK - 1).collect();
    assert!(bed.send(&short, NOW, 2).is_empty());
    assert!(bed.sent(&short, NOW).is_empty());
    // The string tree was never made: no request came as text.
    let record = bed.record();
    assert!(record.tokens.get().is_some() && record.text.get().is_none());
}

/// A request its host did not tokenize is named by the model's string
/// tree, over its text, whether the model's workers publish cache events
/// or not: a worker is named by the whole windows of the text it was sent.
#[test]
fn a_string_tree_names_the_worker_a_text_was_sent_to() {
    for indexed in [false, true] {
        let (affinity, _, _) = hosted(indexed);
        let bed = Bed::new(affinity, preset("cache_aware"), 4);
        let text: String = ('a'..='z').cycle().take(4 * WINDOW).collect();
        assert!(bed.text(&text, None).is_empty());
        assert!(bed.text(&text, Some(2)).is_empty());
        assert_eq!(bed.text(&text, None), [(2, 0, 4)], "indexed: {indexed}");
        // A longer text that begins with it is named by the four windows
        // sent, and a text that parts from it inside its second window by
        // the one window they share.
        let longer = format!("{text}{}", "z".repeat(2 * WINDOW));
        assert_eq!(bed.text(&longer, None), [(2, 0, 4)]);
        let parted = format!("{}{}", &text[..WINDOW + 7], "#".repeat(3 * WINDOW));
        assert_eq!(bed.text(&parted, None), [(2, 0, 1)]);
        let record = bed.record();
        assert!(record.text.get().is_some() && record.tokens.get().is_none());
    }
}

/// Where the model's workers publish cache events, a tokenized request is
/// named by the index alone, as the classic policy names it: a pick is
/// recorded in no tree, and a worker the index holds nothing for is named
/// no holder.
#[test]
fn a_model_with_an_index_names_a_tokenized_request_by_the_index_alone() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = tokens(1_000, 4);
    store(&index, 0, &hashes_of(&hashed(&asked))[..2]);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    assert_eq!(bed.send(&asked, NOW, 3), [(0, 2, 0)]);
    assert_eq!(bed.sent(&asked, NOW), [(0, 2, 0)]);
    let record = bed.record();
    assert!(record.tokens.get().is_none() && record.text.get().is_none());
}

/// A profile whose prefix stage reads the record of sent prefixes alone is
/// named by the trees whatever index the model has, and one whose prefix
/// stage reads the index alone keeps no record and asks for no prompt.
#[test]
fn a_profile_is_named_by_the_records_its_prefix_stage_reads() {
    let asked = tokens(1_000, 4);
    let (affinity, _, index) = hosted(true);
    store(
        &index.expect("the index"),
        0,
        &hashes_of(&hashed(&asked))[..2],
    );
    let bed = Bed::new(affinity, reading(AffinitySource::Predictor), 4);
    assert!(bed.send(&asked, NOW, 3).is_empty());
    assert_eq!(bed.sent(&asked, NOW), [(3, 0, 4)]);
    // A host that lends no prompt has no tree read, and the index names
    // nobody for a profile that does not read it.
    assert!(bed.holders(&hashed(&asked), NOW).is_empty());

    let (affinity, _, index) = hosted(true);
    store(
        &index.expect("the index"),
        0,
        &hashes_of(&hashed(&asked))[..2],
    );
    let bed = Bed::new(affinity, reading(AffinitySource::Index), 4);
    assert_eq!(bed.send(&asked, NOW, 3), [(0, 2, 0)]);
    assert_eq!(bed.sent(&asked, NOW), [(0, 2, 0)]);
    assert_eq!(bed.asked.load(Ordering::Relaxed), 0);
    assert!(bed.affinity.records.lock().is_empty());
}

/// The host is asked for the prompt once a decision, and only where the
/// request's place keeps a record of sent prefixes: a profile that reads no
/// affinity never asks.
#[test]
fn the_prompt_is_asked_for_once_and_only_where_the_place_keeps_a_record() {
    let asked = tokens(1_000, 4);
    for indexed in [false, true] {
        let (affinity, _, _) = hosted(indexed);
        let bed = Bed::new(affinity, preset("cache_aware"), 4);
        bed.send(&asked, NOW, 1);
        assert_eq!(bed.asked.load(Ordering::Relaxed), 1, "indexed: {indexed}");
        bed.sent(&asked, NOW);
        assert_eq!(bed.asked.load(Ordering::Relaxed), 2);
    }
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("least_load"), 4);
    let hashes = hashed(&asked);
    let named = bed.decide_lent(&request(&hashes, 7, NOW), Some(lend(&asked)), Some(1));
    assert_eq!(named, None);
    assert_eq!(bed.asked.load(Ordering::Relaxed), 0);
}

/// Two cache partitions share no prefix: a request is named the worker its
/// own partition's prompt was sent to, and a request of the same tokens in
/// another partition, or in none, is named nobody.
#[test]
fn a_cache_partition_keeps_its_prefixes_apart() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    let hashes = hashed(&asked);
    let facts = request(&hashes, 7, NOW);
    let partition = |salt| CacheNamespace::derive(Some(salt), None, None);
    let under = |namespace, target| {
        let lent = PromptLent {
            namespace,
            ..lend(&asked)
        };
        let named = bed.decide_lent(&facts, Some(lent), target);
        named.expect("rows named").holders
    };
    assert!(under(partition("tenant-a"), Some(2)).is_empty());
    assert_eq!(under(partition("tenant-a"), None), [(2, 0, 4)]);
    assert!(under(partition("tenant-b"), None).is_empty());
    assert!(under(None, None).is_empty());
}

/// Many callers at once lose no write and wait on no lock of the record:
/// each of sixteen callers holds a conversation that grows a block a turn
/// and is sent to one worker, and every turn behind the first is named
/// that worker by all the blocks of the turn before.
#[test]
fn every_callers_next_turn_is_named_the_worker_of_its_last() {
    const CALLERS: u32 = 16;
    const TURNS: u32 = 48;
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 8);
    thread::scope(|scope| {
        for caller in 0..CALLERS {
            let bed = &bed;
            scope.spawn(move || {
                let own = caller % 8;
                let mut said = Vec::new();
                for turn in 1..=TURNS {
                    said.extend(tokens(caller * 1_000_000 + turn * BLOCK, 1));
                    let named = bed.send(&said, NOW, own);
                    let before = turn - 1;
                    let expected: &[(u32, u32, u32)] = &[(own, 0, before)];
                    assert_eq!(named, expected[..usize::from(before > 0)], "turn {turn}");
                }
            });
        }
    });
}

/// The timer's turn evicts a model's trees down to the document's bound
/// once the document's interval has passed since the record was made, and
/// not before. The prefixes least recently used go first.
#[test]
fn the_timers_turn_evicts_a_tree_down_to_its_bound() {
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let bound = u64::from(8 * BLOCK);
    document.index.insert(
        "*".to_string(),
        IndexSpec {
            max_size: bound,
            sweep_s: 2,
            ..IndexSpec::default()
        },
    );
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, spec, 4);
    let prompts: Vec<Vec<u32>> = (1..=4).map(|at| tokens(at * 100_000, 4)).collect();
    for (target, prompt) in (0u32..).zip(&prompts) {
        bed.send(prompt, NOW, target);
    }
    let record = bed.record();
    let size = || record.tokens().total_token_size();
    assert_eq!(size(), 16 * BLOCK as usize);
    bed.affinity.sweep(NOW + 2 * SECOND - 1);
    assert_eq!(size(), 16 * BLOCK as usize, "the interval has not passed");
    bed.affinity.sweep(NOW + 2 * SECOND);
    assert_eq!(size(), bound as usize);
    // The two prompts sent last stand, and the two sent first are gone.
    assert_eq!(bed.sent(&prompts[3], NOW), [(3, 0, 4)]);
    assert_eq!(bed.sent(&prompts[2], NOW), [(2, 0, 4)]);
    assert!(bed.sent(&prompts[0], NOW).is_empty());
    assert!(bed.sent(&prompts[1], NOW).is_empty());
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
fn candidates(ids: &[WorkerId]) -> Vec<Candidate> {
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
        let said = tokens(1_000, 8);
        let asked = hashed(&said);
        if let Some(index) = &index {
            store(index, 7, &hashes_of(&asked));
        }
        let bed = Bed::new(affinity, preset("cache_aware"), WORKERS);
        let rows = candidates(&bed.ids);
        let facts = PoolFacts {
            pool_size: WORKERS as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let scratch = RefCell::new(Scratch::new());
        let place = |seed| {
            let mut request = request(&asked, seed, NOW);
            let decision = decide(
                &bed.affinity,
                &bed.pool,
                &bed.spec,
                &bed.workers,
                &bed.ids,
                &mut request,
                || Some(lend(&said)),
                |request| {
                    let engine = bed.spec.engine();
                    engine.decide(
                        request,
                        &facts,
                        &rows,
                        &Constraints::default(),
                        &mut scratch.borrow_mut(),
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

/// A pair side's column holds one entry a worker of the side's list, in the
/// list's order: the leading blocks of the request the worker holds by the
/// index, what the index holds for the worker and what that gained since
/// its admission, and the worker's age at the request's instant.
#[test]
fn a_sides_column_holds_each_workers_facts_and_what_it_holds_of_the_request() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let hashes = hashes_of(&asked);
    store(&index, 0, &hashes[..2]);
    store(&index, 2, &hashes);
    store(&index, 2, &hashes_of(&prompt(9_000, 3)));
    let bed = Bed::new(affinity, paired(), 4);
    let indexed = |column: &[AffinityFacts]| -> Vec<(u32, u32)> {
        let indexed = |row: &AffinityFacts| (row.indexed_blocks, row.indexed_growth);
        column.iter().map(indexed).collect()
    };
    let first = bed.column(&request(&asked, 7, NOW), Phase::Prefill);
    assert_eq!(held(&first), [(2, 0), (0, 0), (4, 0), (0, 0)]);
    assert_eq!(indexed(&first), [(2, 0), (0, 0), (7, 0), (0, 0)]);
    // Half a second on, by the table that stands, each worker is half a
    // second older.
    let second = bed.column(&request(&asked, 7, NOW + SECOND / 2), Phase::Prefill);
    for (then, later) in first.iter().zip(&second) {
        assert_eq!(later.admitted_age_ms, then.admitted_age_ms + 500);
    }
    // The table read a second on holds what the index gained since.
    store(&index, 0, &hashes_of(&prompt(5_000, 5)));
    let third = bed.column(&request(&asked, 7, NOW + SECOND), Phase::Prefill);
    assert_eq!(indexed(&third), [(7, 5), (0, 0), (7, 0), (0, 0)]);
    assert_eq!(held(&third), held(&first));
    // A request hashed by nothing is held by nobody, and its column is whole.
    let bare = RequestFacts {
        model: MODEL,
        now_micros: NOW + SECOND,
        ..RequestFacts::default()
    };
    assert_eq!(held(&bed.column(&bare, Phase::Prefill)), [(0, 0); 4]);
}

/// A profile that reads affinity and is named no row is handed the column
/// whole: here its two warm-up stages rule by different windows, so no one
/// ruling names its warming rows. The column holds the index's depths. Such
/// a profile keeps no record of sent prefixes: its host is asked for no
/// prompt, and a pick is recorded in no tree.
#[test]
fn a_profile_named_no_row_is_handed_the_column() {
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the preset's profile");
    profile.score.push(
        ScoreStage::WarmupSlice(WarmupSliceParams {
            weight: 1.0,
            secs: Some(7),
            ..WarmupSliceParams::default()
        })
        .into(),
    );
    let spec = || AppliedSpec::validate(&document).expect("the document validates");
    assert_eq!(spec().engine().affinity_rows_cap(MODEL, Phase::Single), 0);
    let asked = prompt(1_000, 4);
    let seen = |bed: &Bed, target: Option<u32>| {
        let seen = RefCell::new(None);
        let decision = decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&asked, 7, NOW),
            || -> Option<PromptLent<'_>> { panic!("no prompt is asked for") },
            |facts| {
                assert!(facts.named.is_none());
                *seen.borrow_mut() = Some(held(facts.affinity));
                target.map_or_else(shed, |at| picked(at, bed.ids[at as usize]))
            },
        );
        assert_eq!(decision.pick().is_some(), target.is_some());
        seen.into_inner().expect("a decision")
    };
    let (affinity, _, index) = hosted(true);
    store(&index.expect("the index"), 2, &hashes_of(&asked));
    let bed = Bed::new(affinity, spec(), 4);
    assert_eq!(seen(&bed, None), [(0, 0), (0, 0), (4, 0), (0, 0)]);
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, spec(), 4);
    assert_eq!(seen(&bed, Some(1)), [(0, 0); 4]);
    assert_eq!(seen(&bed, None), [(0, 0); 4]);
    assert!(bed.affinity.records.lock().is_empty());
}
