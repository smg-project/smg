//! The naming of a decision's rows from a host's index of cache events, its
//! token tree and its string tree of sent prefixes, and the engine's record
//! of sent prefixes.

use std::{borrow::Cow, cell::RefCell, sync::atomic::AtomicUsize, thread, time::Duration};

use kv_index::{
    compute_request_content_hashes, request_prefix_hashes, ChainBlockMap, ChainIndex, ContentHash,
    GroupKind, SequenceHash, StoredBlock,
};
use smg_worker::test_support::ready_worker;

use super::*;
use crate::{
    document::spec::{
        PolicySpec, ProfileSpec, RulePlaced, ScoreStage, ShadowSpec, WarmupSliceParams,
    },
    engine::{
        decision::{OverlapSource, Pick, Shed, ShedReason, Target, Targets, Trace},
        framework::Scratch,
        prompt_hashes::{HashSeeds, PromptHashes},
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, Health, PoolFacts},
    },
};

impl Affinity {
    /// The bound, in placements, of the record of the prefixes sent to
    /// `model`'s workers: none where no list's table asked for one yet.
    pub(in crate::engine::in_process) fn bound_of(&self, model: &str) -> Option<usize> {
        let records = self.records.lock();
        let record = records.get(model)?;
        let bound = record.predictor.read().bound();
        Some(bound)
    }

    /// Runs `run` while a reader holds every record of sent prefixes, as a
    /// decision's naming on another thread does.
    pub(in crate::engine::in_process) fn while_held<R>(&self, run: impl FnOnce() -> R) -> R {
        let held: Vec<Arc<PrefixRecord>> = self.records.lock().values().cloned().collect();
        let _readers: Vec<_> = held.iter().map(|record| record.predictor.read()).collect();
        run()
    }
}

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
    ruled_by(warm_up(secs), |_| {})
}

/// The warm-up stage of [`warming_by`], for a case that states one more
/// of its numbers.
fn warm_up(secs: u64) -> WarmupSliceParams {
    WarmupSliceParams {
        weight: 8.0,
        secs: Some(secs),
        blocks: Some(4),
        ..WarmupSliceParams::default()
    }
}

/// The cache-aware preset with `stage` as its warm-up stage, its profile
/// as `change` then leaves it.
fn ruled_by(stage: WarmupSliceParams, change: impl FnOnce(&mut ProfileSpec)) -> AppliedSpec {
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the preset's profile");
    profile.score[0] = ScoreStage::WarmupSlice(stage).into();
    change(profile);
    AppliedSpec::validate(&document).expect("the document validates")
}

/// [`warming_by`] sixty seconds under a picker whose sample is `sample`
/// rows: the most holders, and the most warming rows, a decision is named.
fn sampled(sample: u32) -> AppliedSpec {
    use crate::document::spec::{BestParams, PickStage};
    ruled_by(warm_up(60), |profile| {
        profile.pick = PickStage::Best(BestParams { sample }).into();
    })
}

fn address(at: usize) -> String {
    format!("http://worker-{at:03}:8000")
}

fn pool(count: usize) -> (Vec<Arc<dyn Worker>>, Vec<WorkerId>) {
    let workers = (0..count).map(|at| ready_worker(&address(at), MODEL));
    let ids = (0..count).map(|at| WorkerId(at as u64 + 1));
    (workers.collect(), ids.collect())
}

/// A host's indexes: one model's chain index, and the host's own token tree
/// and string tree of the prompts it sent to the model's workers, where it
/// keeps them. And how often it was asked for the index or the token tree.
#[derive(Default)]
struct Host {
    index: Option<Arc<ChainIndex>>,
    asked: AtomicUsize,
    tree: Option<Arc<TokenTree>>,
    text: Option<Arc<StringTree>>,
}

impl CacheIndexes for Host {
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        // None while the index holds no block, as the trait asks of a host.
        let held = |index: &&Arc<ChainIndex>| model == MODEL && !index.is_empty();
        let index = self.index.as_ref().filter(held)?;
        Some(Arc::clone(index) as Arc<dyn CacheIndex>)
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        Some(BLOCK)
    }

    fn token_tree(&self, model: &str, phase: Phase) -> Option<Arc<TokenTree>> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        let single = model == MODEL && phase == Phase::Single;
        self.tree.as_ref().filter(|_| single).cloned()
    }

    fn string_tree(&self, model: &str, phase: Phase) -> Option<Arc<StringTree>> {
        let single = model == MODEL && phase == Phase::Single;
        self.text.as_ref().filter(|_| single).cloned()
    }
}

/// An engine's affinity over a host with a chain index, or with none.
fn hosted(indexed: bool) -> (Affinity, Arc<Host>, Option<Arc<ChainIndex>>) {
    let index = indexed.then(|| Arc::new(ChainIndex::new()));
    let host = Arc::new(Host {
        index: index.clone(),
        ..Host::default()
    });
    let affinity = Affinity::default();
    affinity.set_indexes(Arc::clone(&host) as Arc<dyn CacheIndexes>);
    (affinity, host, index)
}

/// An engine's affinity over a host that keeps a token tree of the prompts
/// it sent, a page a block, and lends it: with a chain index beside it, or
/// with none. The tree is the host's own, as its policy would read and
/// write it.
fn lending(indexed: bool) -> (Affinity, Arc<TokenTree>) {
    lending_by(BLOCK as usize, indexed)
}

/// [`lending`], the host's tree made with pages of `page` tokens. The
/// index it hands holds a block, as the trait asks of a host: it hands
/// none while its index is empty ([`CacheIndexes::index`]).
fn lending_by(page: usize, indexed: bool) -> (Affinity, Arc<TokenTree>) {
    let tree = Arc::new(TokenTree::with_config(page, Default::default()));
    let index = indexed.then(|| {
        let index = Arc::new(ChainIndex::new());
        store(&index, 0, &[7]);
        index
    });
    let host = Arc::new(Host {
        index,
        tree: Some(Arc::clone(&tree)),
        ..Host::default()
    });
    let affinity = Affinity::default();
    affinity.set_indexes(host as Arc<dyn CacheIndexes>);
    (affinity, tree)
}

/// An engine's affinity over a host that keeps a string tree of the prompts
/// it sent as text and lends it: with a chain index that holds a block
/// beside it, or with none. It lends no token tree.
fn lending_text(indexed: bool) -> (Affinity, Arc<StringTree>) {
    let tree = Arc::new(StringTree::new());
    let index = indexed.then(|| {
        let index = Arc::new(ChainIndex::new());
        store(&index, 0, &[7]);
        index
    });
    let host = Arc::new(Host {
        index,
        text: Some(Arc::clone(&tree)),
        ..Host::default()
    });
    let affinity = Affinity::default();
    affinity.set_indexes(host as Arc<dyn CacheIndexes>);
    (affinity, tree)
}

/// An engine's affinity over a host that lends its token tree and its
/// string tree and keeps no index: what a host that keeps both trees of a
/// model lends.
fn lending_both() -> (Affinity, Arc<TokenTree>, Arc<StringTree>) {
    let tree = Arc::new(TokenTree::with_config(BLOCK as usize, Default::default()));
    let text = Arc::new(StringTree::new());
    let host = Arc::new(Host {
        tree: Some(Arc::clone(&tree)),
        text: Some(Arc::clone(&text)),
        ..Host::default()
    });
    let affinity = Affinity::default();
    affinity.set_indexes(host as Arc<dyn CacheIndexes>);
    (affinity, tree, text)
}

/// Stores `hashes`, a prompt's blocks from its first, for the worker at
/// `at`, as its event stream would.
fn store(index: &ChainIndex, at: usize, hashes: &[u64]) {
    stored(index, at, hashes);
}

/// [`store`], handing back the worker's number and the map of its blocks,
/// which the end of its stream takes ([`ChainIndex::remove_worker`]).
fn stored(index: &ChainIndex, at: usize, hashes: &[u64]) -> (u32, ChainBlockMap) {
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
    let mut map = ChainBlockMap::new();
    index
        .apply_stored(number, &blocks, None, &mut map)
        .expect("the blocks are stored");
    (number, map)
}

/// The token ids of a prompt of `blocks` whole blocks, counted from `first`.
fn tokens(first: u32, blocks: u32) -> Vec<u32> {
    (first..first + blocks * BLOCK).collect()
}

/// What a host hashes of a tokenized prompt.
fn hashed(tokens: &[u32]) -> PromptHashes {
    let spec = preset("cache_aware");
    PromptHashes::under_document(&spec, Some(BLOCK), Some(tokens), None, HashSeeds::PLAIN)
}

/// A prompt of `blocks` whole blocks, its tokens counted from `first`.
fn prompt(first: u32, blocks: u32) -> PromptHashes {
    hashed(&tokens(first, blocks))
}

/// Bytes a window of text: a block's tokens at four bytes a token.
const WINDOW: usize = 4 * BLOCK as usize;

/// A prompt of text of `windows` whole windows, each a number counted from
/// `first` and written `char` by `char` behind its digits: a text that
/// begins with a shorter one of the same `first`. `fill` is one char of
/// some bytes, and a window is filled to its bytes.
fn text_of(first: u32, windows: u32, fill: char) -> String {
    let window = |at: u32| {
        let mut window = (first + at).to_string();
        while window.len() < WINDOW {
            window.push(fill);
        }
        assert_eq!(
            window.len(),
            WINDOW,
            "the fill ends on the window's last byte"
        );
        window
    };
    (0..windows).map(window).collect()
}

/// A prompt of text of `windows` whole windows of one byte a char.
fn text(first: u32, windows: u32) -> String {
    text_of(first, windows, '.')
}

/// What a host hashes of a prompt it holds as text.
fn worded(text: &str) -> PromptHashes {
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

/// [`picked`] as the binding's sessions rule places it alone, under the
/// rule's word of a request it places.
fn ruled(position: u32, id: WorkerId, kept: RulePlaced) -> Decision {
    let mut decision = picked(position, id);
    if let Decision::Pick(pick) = &mut decision {
        pick.targets.by_rule(kept);
    }
    decision
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

/// `decision` with nothing stated of its first target: the answer a lookup
/// was handed, where it stated what that target holds behind it.
fn unstated(mut decision: Decision) -> Decision {
    if let Decision::Pick(pick) = &mut decision {
        pick.targets.state(Overlap::NONE);
    }
    decision
}

/// `blocks` of `of` read from `source`, with no word of a margin: what the
/// lookup states of a pick's first target.
fn holds(blocks: u16, of: u16, source: OverlapSource) -> Overlap {
    Overlap {
        blocks,
        of,
        source,
        margin_held: false,
    }
}

/// The cache-aware preset with a second warm-up stage that rules by another
/// window: no one ruling names its warming rows, so the profile is named no
/// row and its decision is handed the column whole.
fn handed_whole() -> AppliedSpec {
    named_no_row(|document| document)
}

/// [`handed_whole`], its document as `change` leaves it.
fn named_no_row(change: impl FnOnce(PolicySpec) -> PolicySpec) -> AppliedSpec {
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
    let spec = AppliedSpec::validate(&change(document)).expect("the document validates");
    assert_eq!(spec.engine().affinity_rows_cap(MODEL, Phase::Single), 0);
    spec
}

/// `document` with every prefix stage of its profiles reading sent prefixes
/// alone (`source: predictor`).
fn reading_alone(mut document: PolicySpec) -> PolicySpec {
    use crate::document::spec::{AffinitySource, ScoreSpec};

    for profile in document.profiles.values_mut() {
        for stage in &mut profile.score {
            if let ScoreSpec::Params(ScoreStage::PrefixAffinity(prefix)) = stage {
                prefix.source = Some(AffinitySource::Predictor);
            }
        }
    }
    document
}

/// The cache-aware preset, its prefix stage reading sent prefixes alone,
/// under a picker whose sample is `sample` rows.
fn alone_sampled(sample: u32) -> AppliedSpec {
    use crate::document::spec::{BestParams, PickStage};

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the preset's profile");
    profile.pick = PickStage::Best(BestParams { sample }).into();
    let spec = AppliedSpec::validate(&reading_alone(document)).expect("the document validates");
    assert!(spec.engine().reads_sent_alone_for(MODEL, Phase::Single));
    assert_eq!(
        spec.engine().affinity_rows_cap(MODEL, Phase::Single),
        sample
    );
    spec
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
        self.decide_keyed(facts, None, target)
    }

    /// [`Self::decide`] for a request whose host names `key` as its entry
    /// in the token tree, where it names one.
    fn decide_keyed(
        &self,
        facts: &RequestFacts<'_>,
        key: Option<TokenKey<'_>>,
        target: Option<u32>,
    ) -> Option<Named> {
        self.decide_among(facts, None, key, target)
    }

    /// What a request its host holds to `places` is named, decided by a
    /// shed.
    fn held_to(&self, facts: &RequestFacts<'_>, places: &[u32]) -> Named {
        let named = self.decide_among(facts, Some(places), None, None);
        named.expect("rows named")
    }

    /// [`Self::decide_keyed`] for a request its host holds to `places`,
    /// where it states any.
    fn decide_among(
        &self,
        facts: &RequestFacts<'_>,
        places: Option<&[u32]>,
        key: Option<TokenKey<'_>>,
        target: Option<u32>,
    ) -> Option<Named> {
        self.decide_entered(facts, places, (key, None), target)
    }

    /// [`Self::decide_among`] for a request whose host names its entry in
    /// each tree it lends, where it names one: `key` in the token tree,
    /// `text` in the string tree.
    fn decide_entered(
        &self,
        facts: &RequestFacts<'_>,
        places: Option<&[u32]>,
        (key, text): (Option<TokenKey<'_>>, Option<TextKey<'_>>),
        target: Option<u32>,
    ) -> Option<Named> {
        let mut named = None;
        let decision = decide(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            &mut { *facts },
            places,
            |_| key,
            || text,
            |_| true,
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

    /// What the lookup states of the first target of `answer`, the decision
    /// of `facts`: the request as its host made it, with whatever that host
    /// fed, and `key` its entry in a lent tree where the host names one. The
    /// lookup returns the answer it was handed but for that statement.
    fn stated(
        &self,
        facts: &RequestFacts<'_>,
        key: Option<TokenKey<'_>>,
        answer: Decision,
    ) -> Overlap {
        self.stated_entered(facts, (key, None), answer)
    }

    /// [`Self::stated`] for a request whose host names its entry in each
    /// tree it lends, where it names one: `key` in the token tree, `text`
    /// in the string tree.
    fn stated_entered(
        &self,
        facts: &RequestFacts<'_>,
        (key, text): (Option<TokenKey<'_>>, Option<TextKey<'_>>),
        answer: Decision,
    ) -> Overlap {
        let returned = decide(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            &mut { *facts },
            None,
            |_| key,
            || text,
            |_| true,
            |_| answer,
        );
        assert_eq!(unstated(returned), answer, "the answer handed");
        returned.pick().map(Pick::overlap).unwrap_or_default()
    }

    /// [`Self::stated`] behind a pick of the worker at `row` by the
    /// profile's stages, for a request that names no entry.
    fn states(&self, facts: &RequestFacts<'_>, row: u32) -> Overlap {
        self.stated(facts, None, picked(row, self.ids[row as usize]))
    }

    /// The holders a request of `prompt` is named, decided by a shed.
    fn holders(&self, prompt: &PromptHashes, now: u64) -> Vec<(u32, u32, u32)> {
        let named = self.decide(&request(prompt, 7, now), None);
        named.expect("rows named").holders
    }

    /// The holders a request is named whose host names `key` as its entry
    /// in the token tree: the prompt's tokens behind `marker` ids of a
    /// partition's marker. Decided by picking the worker at `target`, which
    /// writes the request for it, or by a shed, which writes nothing.
    fn keyed(&self, key: &[u32], marker: usize, target: Option<u32>) -> Vec<(u32, u32, u32)> {
        let hashes = hashed(&key[marker..]);
        let key = TokenKey {
            tokens: Cow::Borrowed(key),
            marker,
        };
        let named = self.decide_keyed(&request(&hashes, 7, NOW), Some(key), target);
        named.expect("rows named").holders
    }

    /// The holders a request of text is named whose host names `key` as its
    /// entry in the string tree: the prompt's text behind `marker` chars of
    /// a partition's marker. Decided by picking the worker at `target`,
    /// which writes the request for it, or by a shed, which writes nothing.
    fn worded(&self, key: &str, marker: usize, target: Option<u32>) -> Vec<(u32, u32, u32)> {
        let behind = key
            .char_indices()
            .nth(marker)
            .map_or(key.len(), |(at, _)| at);
        let hashes = worded(&key[behind..]);
        let entry = TextKey {
            text: Cow::Borrowed(key),
            marker,
        };
        let facts = request(&hashes, 7, NOW);
        let named = self.decide_entered(&facts, None, (None, Some(entry)), target);
        named.expect("rows named").holders
    }

    /// The list's column for `facts` at `phase`, and what its fill read of
    /// the record of sent prefixes.
    fn column(
        &self,
        facts: &RequestFacts<'_>,
        phase: Phase,
    ) -> (Vec<AffinityFacts>, Option<PrefixRead>) {
        let mut filled = vec![AffinityFacts {
            overlap_blocks: 99,
            ..AffinityFacts::default()
        }];
        let read = column(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            phase,
            facts,
            &mut filled,
        );
        (filled, read)
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

/// The lookup is kept within the list's own workers. A worker outside the
/// list holds the request deeper than any worker of it, under a sample of
/// one: the one place goes to the list's deepest holder. A lookup over
/// every worker of the index would spend it on the worker outside, which
/// is no row of the list, and name nobody.
#[test]
fn a_holder_outside_the_list_takes_no_place_of_the_cap() {
    use crate::document::spec::{BestParams, PickStage};

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let hashes = hashes_of(&asked);
    store(&index, 2, &hashes[..2]);
    store(&index, 9, &hashes);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document.profiles.get_mut("cache_aware");
    let profile = profile.expect("the preset's profile");
    profile.pick = PickStage::Best(BestParams { sample: 1 }).into();
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let bed = Bed::new(affinity, spec, 4);
    assert_eq!(bed.holders(&asked, NOW), [(2, 2, 0)]);
}

/// A request its host holds to places is named its holders among the
/// places: the index is asked within them, so holders outside cannot fill
/// the sample. Of twenty workers that hold the whole prompt, a request held
/// to two is named both under every seed, where the sample of sixteen
/// would leave each out for some. The classic policy narrows its workers
/// first (`model_gateway/src/routers/common/placement.rs:474-484`) and
/// gathers the deepest holders among them, one or both by its cap's draw
/// (`model_gateway/src/policies/cache_aware.rs:2916-2939`, `:2951-2957`).
#[test]
fn a_held_request_is_named_its_holders_among_its_places() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 2);
    for at in 0..20 {
        store(&index, at, &hashes_of(&asked));
    }
    let bed = Bed::new(affinity, preset("cache_aware"), 20);
    for seed in 0..32 {
        let named = bed.held_to(&request(&asked, seed, NOW), &[3, 17]);
        assert_eq!(named.holders, [(3, 2, 0), (17, 2, 0)], "seed {seed}");
    }
    // A request that states no places is named the sample of the list.
    let named = bed.decide(&request(&asked, 7, NOW), None).expect("rows");
    assert_eq!(named.holders.len(), 16);
}

/// A shorter beginning held among the places is named where deeper holders
/// outside them are more than the sample: it is the deepest among the
/// places. The classic policy's whole gather finds it where none of its
/// drawn holders is a candidate
/// (`model_gateway/src/policies/cache_aware.rs:2931-2938`).
#[test]
fn a_held_request_is_named_the_deepest_holder_among_its_places() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 8);
    let hashes = hashes_of(&asked);
    for at in 0..17 {
        store(&index, at, &hashes);
    }
    store(&index, 19, &hashes[..3]);
    // The index knows the other place by a number alone.
    index.intern_worker(&address(18)).expect("a number");
    let bed = Bed::new(affinity, preset("cache_aware"), 20);
    for seed in 0..32 {
        let named = bed.held_to(&request(&asked, seed, NOW), &[18, 19]);
        assert_eq!(named.holders, [(19, 3, 0)], "seed {seed}");
        // Stated to no place, the seventeen deeper holders fill the sample.
        let named = bed.decide(&request(&asked, seed, NOW), None).expect("rows");
        assert!(named
            .holders
            .iter()
            .all(|&(row, held, _)| row < 17 && held == 8));
    }
    // The request's mask is the table's, kept to the places: a worker that
    // gains its first blocks while the table stands is not the index's to
    // name in that second, for a request held to it as for any other.
    store(&index, 18, &hashes);
    let named = bed.held_to(&request(&asked, 0, NOW), &[18, 19]);
    assert_eq!(named.holders, [(19, 3, 0)]);
    let named = bed.decide(&request(&asked, 0, NOW), None).expect("rows");
    assert!(named.holders.iter().all(|&(row, _, _)| row < 17));
}

/// A request held to places is named its warming rows among the places,
/// before the sample cuts them. Of twenty warming workers, a request held
/// to one is named it under every seed. The classic policy takes its
/// pool's warming workers whole and passes over each that is no candidate
/// (`model_gateway/src/policies/cache_aware.rs:2065-2069`).
#[test]
fn a_held_request_is_named_its_warming_rows_among_its_places() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let first = hashes_of(&prompt(1_000, 8));
    for at in 0..24 {
        store(&index, at, &first);
    }
    let bed = Bed::new(affinity, warming_by(1), 24);
    let other = prompt(50_000, 2);
    // The table's first read admits the workers, as a decision does.
    assert!(bed
        .held_to(&request(&other, 7, NOW), &[9])
        .warming
        .is_empty());
    // Four workers gain their first blocks, and twenty stay short of them.
    let second = hashes_of(&prompt(2_000, 8));
    for at in 20..24 {
        store(&index, at, &second);
    }
    thread::sleep(Duration::from_millis(3));
    for seed in 0..32 {
        let named = bed.held_to(&request(&other, seed, NOW + SECOND), &[9, 22]);
        assert_eq!(named.warming, [9], "seed {seed}");
        // Stated to no place, the sample cuts the twenty to sixteen.
        let named = bed.decide(&request(&other, seed, NOW + SECOND), None);
        assert_eq!(named.expect("rows").warming.len(), 16);
    }
}

/// The workers sent a prefix are named among the places too. Twenty
/// workers were sent the prompt, and a request held to two of them is named
/// both under every seed, by the blocks they were sent. The classic policy
/// takes every candidate among its tree's matched workers, with no cut
/// (`model_gateway/src/policies/cache_aware.rs:2541-2552`).
#[test]
fn a_held_request_is_named_the_workers_sent_its_prefix_among_its_places() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 20);
    let asked = prompt(1_000, 4);
    for at in 0..20 {
        bed.decide(&request(&asked, 7, NOW), Some(at));
    }
    for seed in 0..32 {
        let named = bed.held_to(&request(&asked, seed, NOW), &[3, 17]);
        assert_eq!(named.holders, [(3, 0, 4), (17, 0, 4)], "seed {seed}");
    }
    // Stated to no place, the sample cuts the twenty to sixteen.
    assert_eq!(bed.holders(&asked, NOW).len(), 16);
}

/// A token tree's workers are named among the places too. A tree hands the
/// eight workers of its deepest node at most, so the built-in sample of
/// sixteen never cuts them: under a sample of four, a request held to two of
/// the eight is named both under every seed. The classic policy takes every
/// candidate among the workers its tree hands
/// (`model_gateway/src/policies/cache_aware.rs:2541-2552`).
#[test]
fn a_held_request_is_named_a_trees_workers_among_its_places() {
    let (affinity, _) = lending(false);
    let bed = Bed::new(affinity, sampled(4), 20);
    let asked = tokens(1_000, 4);
    for at in 0..8 {
        bed.keyed(&asked, 0, Some(at));
    }
    let hashes = hashed(&asked);
    let key = || TokenKey {
        tokens: Cow::Borrowed(&asked),
        marker: 0,
    };
    for seed in 0..32 {
        let facts = request(&hashes, seed, NOW);
        let named = bed.decide_among(&facts, Some(&[2, 5]), Some(key()), None);
        let named = named.expect("rows named");
        assert_eq!(named.holders, [(2, 0, 4), (5, 0, 4)], "seed {seed}");
    }
    // Stated to no place, the sample cuts the eight to four.
    assert_eq!(bed.keyed(&asked, 0, None).len(), 4);
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
    // A turn that goes on from a recorded prompt writes the blocks it adds
    // for the worker that holds the rest.
    bed.decide(&request(&longer, 7, NOW), Some(1));
    assert_eq!(bed.holders(&longer, NOW), [(1, 0, 6)]);
    assert_eq!(bed.affinity.left_out(), 0);
}

/// A model's record of sent prefixes is its workers' model's, and so is its
/// bound: the index settings of the id the workers registered under,
/// whatever name a request gives the model.
#[test]
fn a_request_under_another_name_leaves_the_records_bound() {
    use crate::document::spec::IndexSpec;
    let (affinity, _, _) = hosted(false);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    // One placement under any other name, the default under the workers'.
    let one = IndexSpec {
        max_size: u64::from(BLOCK),
        ..IndexSpec::default()
    };
    document.index.insert("*".into(), one);
    document.index.insert(MODEL.into(), IndexSpec::default());
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let bed = Bed::new(affinity, spec, 4);
    let asked = prompt(1_000, 4);
    bed.decide(&request(&asked, 7, NOW), Some(1));
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4)]);
    // A request with no prompt that names the model by another name.
    let other = RequestFacts {
        model: "another-name",
        seed: 7,
        now_micros: NOW,
        ..RequestFacts::default()
    };
    bed.decide(&other, None);
    bed.affinity.sweep();
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4)]);
}

/// A model its workers registered after their first shares the record of
/// the id they registered first, and that id's index settings bound it: a
/// request for the later model reads them, before a request for the first
/// model and after one.
#[test]
fn a_later_model_of_its_workers_shares_the_first_models_record() {
    use smg_worker::{
        identity::builder::BasicWorkerBuilder, model_card::ModelCard, test_support::no_health_check,
    };

    use crate::document::spec::IndexSpec;
    const LATER: &str = "model-b";
    let (affinity, _, _) = hosted(false);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    // One placement under the later model's name, the default under the first's.
    let one = IndexSpec {
        max_size: u64::from(BLOCK),
        ..IndexSpec::default()
    };
    document.index.insert(LATER.into(), one);
    document.index.insert(MODEL.into(), IndexSpec::default());
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let mut bed = Bed::new(affinity, spec, 4);
    for (at, worker) in bed.workers.iter_mut().enumerate() {
        let models = vec![ModelCard::new(MODEL), ModelCard::new(LATER)];
        let built = <BasicWorkerBuilder>::new(address(at)).models(models);
        *worker = Arc::new(built.health_config(no_health_check()).build());
    }
    let asked = prompt(1_000, 4);
    let later = RequestFacts {
        model: LATER,
        ..request(&asked, 7, NOW)
    };
    bed.decide(&later, Some(1));
    bed.affinity.sweep();
    // The holders are asked under the first model's name.
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4)]);
    bed.decide(&later, None);
    bed.affinity.sweep();
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4)]);
}

/// A pick the binding's sessions rule placed alone is recorded by the
/// rule's own word of it, at both of the record's writes: the one behind a
/// decision named its rows, and the one behind a decision handed the column
/// whole. A pick the rule keeps from the profile leaves no prefix. One that
/// is the profile's own, the rule's default, is written as a pick by the
/// profile's stages is. A pick is written for the first of its targets its
/// caller sends to (`sends_to`), and for none where it sends to none. The
/// lookup returns the answer it is handed either
/// way, but for what the pick states of its first target
/// ([`a_pick_its_rule_keeps_from_the_profile_states_nothing`]): a pin the
/// keyed entry followed passes through it with its answer made. The live
/// profile of the column form is named no row, and its prefix stage reads
/// sent prefixes: the list keeps the record by that profile's own word,
/// with a shadow beside it or none, and the column shows
/// what that record holds. A gateway's sticky map hands a request it places
/// to no policy, so the policy's record never takes it
/// (`model_gateway/src/policies/registry.rs:791`, `:840`).
#[test]
fn a_pick_the_sessions_rule_placed_is_recorded_by_the_rules_word_at_both_writes() {
    for whole in [false, true] {
        let spec = if whole {
            // The shadow is the preset's profile, named rows. The live
            // profile's two warm-up stages rule by different windows: it is
            // named no row and reads the column whole.
            let mut document = PolicySpec::preset("cache_aware").expect("the preset");
            document.generation = 1;
            let twin = document.profiles["cache_aware"].clone();
            document.profiles.insert("twin".into(), twin);
            document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
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
            AppliedSpec::validate(&document).expect("the document validates")
        } else {
            preset("cache_aware")
        };
        let cap = spec.engine().affinity_rows_cap(MODEL, Phase::Single);
        assert_eq!(cap == 0, whole, "the form of the write");
        let record = spec.engine().pool_record_for(MODEL, Phase::Single);
        assert!(record.is_some(), "whole: {whole}: the list keeps a record");
        let (affinity, _, _) = hosted(false);
        let bed = Bed::new(affinity, spec, 4);
        // One decision of `asked` answered `answer`: the blocks the record
        // held for each worker when it was asked. The lookup returns the
        // answer it was handed.
        let decided = |asked: &PromptHashes, answer: Decision| {
            let mut held = vec![0; bed.ids.len()];
            let returned = decide(
                &bed.affinity,
                &bed.pool,
                &bed.spec,
                &bed.workers,
                &bed.ids,
                &mut request(asked, 7, NOW),
                None,
                |_| None,
                || None,
                // The caller sends to every worker but the one at 3.
                |target| target.position != 3,
                |facts| {
                    assert_eq!(facts.named.is_none(), whole);
                    match facts.named {
                        Some(rows) => {
                            for holder in rows.holders {
                                held[holder.row as usize] = holder.predicted_blocks;
                            }
                        }
                        None => {
                            for (row, worker) in facts.affinity.iter().enumerate() {
                                held[row] = worker.predicted_blocks;
                            }
                        }
                    }
                    answer
                },
            );
            assert_eq!(
                unstated(returned),
                answer,
                "whole: {whole}: the answer handed"
            );
            held
        };
        let (kept, own) = (prompt(1_000, 4), prompt(2_000, 4));
        // Kept from the profile: nothing is written, at one decision or two.
        let untold = ruled(1, bed.ids[1], RulePlaced::Untold);
        assert_eq!(decided(&kept, untold), [0; 4], "whole: {whole}");
        assert_eq!(decided(&kept, untold), [0; 4], "whole: {whole}: kept");
        assert_eq!(decided(&kept, shed()), [0; 4], "whole: {whole}: kept twice");
        // The profile's own: written for the rule's worker.
        let told = ruled(1, bed.ids[1], RulePlaced::Told);
        assert_eq!(decided(&own, told), [0; 4], "whole: {whole}");
        assert_eq!(decided(&own, shed()), [0, 4, 0, 0], "whole: {whole}: told");
        // The stages' pick of the first prompt is written as that one was,
        // behind a lead its caller passes over: the worker at 3.
        let mut by_stages = picked(2, bed.ids[2]);
        if let Decision::Pick(pick) = &mut by_stages {
            pick.targets.lead(Target {
                position: 3,
                id: bed.ids[3],
                ..Target::default()
            });
        }
        assert_eq!(decided(&kept, by_stages), [0; 4], "whole: {whole}");
        assert_eq!(
            decided(&kept, shed()),
            [0, 0, 4, 0],
            "whole: {whole}: the stages'"
        );
        // A pick whose one target its caller passes over: nothing is written.
        let (lost, alone) = (prompt(3_000, 4), picked(3, bed.ids[3]));
        assert_eq!(decided(&lost, alone), [0; 4], "whole: {whole}");
        assert_eq!(decided(&lost, shed()), [0; 4], "whole: {whole}: to none");
    }
}

/// A request's entry prints its sizes and none of its ids: no line the
/// engine formats carries a client's tokens.
#[test]
fn an_entrys_debug_form_holds_no_token() {
    let key = TokenKey {
        tokens: Cow::Owned(vec![4_242_424, 5_353_535]),
        marker: 1,
    };
    assert_eq!(format!("{key:?}"), "TokenKey { tokens: 2, marker: 1 }");
    let key = TextKey {
        text: Cow::Borrowed("a client's words"),
        marker: 2,
    };
    assert_eq!(format!("{key:?}"), "TextKey { text: 16, marker: 2 }");
}

/// The host's token tree takes the prompt of a pick the binding's sessions
/// rule placed alone by the rule's own word of it, as the record of hashed
/// prefixes does
/// ([`a_pick_the_sessions_rule_placed_is_recorded_by_the_rules_word_at_both_writes`]).
/// A pick the rule keeps from the profile leaves no prompt in the tree. One
/// that is the profile's own is written as a pick by the profile's stages
/// is. The lookup returns the answer it is handed either way, but for what
/// it states of the pick: nothing of one kept from the profile, and its
/// worker's entry of the match's rows of one that is the profile's own. A
/// gateway's sticky map hands a request it places to no policy, so the
/// policy's tree never takes that request's prompt
/// (`model_gateway/src/policies/registry.rs:791`, `:840`).
#[test]
fn a_pick_the_sessions_rule_placed_is_written_to_a_tree_by_the_rules_word() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let (kept, own) = (tokens(1_000, 4), tokens(2_000, 4));
    let miss = holds(0, 4, OverlapSource::None);
    let untold = ruled(1, bed.ids[1], RulePlaced::Untold);
    assert_eq!(answered_in_tree(&bed, &kept, untold), Overlap::NONE);
    assert!(
        bed.keyed(&kept, 0, None).is_empty(),
        "kept from the profile"
    );
    let told = ruled(1, bed.ids[1], RulePlaced::Told);
    assert_eq!(answered_in_tree(&bed, &own, told), miss);
    assert_eq!(bed.keyed(&own, 0, None), [(1, 0, 4)], "the rule's, told");
    assert_eq!(stated_in_tree(&bed, &kept, 2), miss);
    assert_eq!(bed.keyed(&kept, 0, None), [(2, 0, 4)], "the stages'");
}

/// A request whose host names its entry is decided inside the host's token
/// tree. A pick writes the whole prompt for its worker, a later request that
/// begins alike is named that worker by the blocks the two share, the
/// deepest holders alone are named, a shed writes nothing and a worker its
/// host took out of the tree is named no more. The record of hashed
/// prefixes stays empty.
#[test]
fn a_token_tree_names_the_worker_a_prefix_was_sent_to() {
    let (affinity, tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    assert!(bed.keyed(&asked, 0, None).is_empty());
    // The pick's own decision is named what stood before it.
    assert!(bed.keyed(&asked, 0, Some(1)).is_empty());
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 4)]);
    // A longer prompt that begins with it is named by the four blocks sent.
    let longer = tokens(1_000, 6);
    assert_eq!(bed.keyed(&longer, 0, Some(3)), [(1, 0, 4)]);
    // The worker sent the longer prompt is its one deepest holder, and both
    // hold the shorter one.
    assert_eq!(bed.keyed(&longer, 0, None), [(3, 0, 6)]);
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 4), (3, 0, 4)]);
    // The tree is the host's to keep: it takes out a worker that left.
    tree.remove_tenant_all(&TenantId::from(address(3)));
    assert_eq!(bed.keyed(&longer, 0, None), [(1, 0, 4)]);
    // A prompt under one block is one no engine's cache serves: the tree
    // keeps none, and the request is decided all the same.
    let short = &asked[..BLOCK as usize - 1];
    assert!(bed.keyed(short, 0, Some(2)).is_empty());
    assert!(bed.keyed(short, 0, None).is_empty());
    // A host that names no entry reads the record of hashed prefixes, which
    // no request above wrote.
    assert!(bed.holders(&prompt(1_000, 4), NOW).is_empty());
}

/// A tree's page is no unit of the request. The match is counted in the
/// request's blocks: by the size its host hashed it by, and by the
/// document's where the host hashed nothing. So a host's tree of 64-token
/// pages under a document of 16-token blocks names a prompt of 128 tokens
/// by eight blocks either way, the count the prefix stage sizes the request
/// by, and by four where its host hashed it by blocks of 32. The classic
/// policy divides the tokens its tree matched by the prompt's tokens,
/// whatever the page (`model_gateway/src/policies/cache_aware.rs:3474-3482`).
#[test]
fn a_tree_of_another_page_is_counted_in_the_requests_blocks() {
    let (affinity, _tree) = lending_by(4 * BLOCK as usize, false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 8);
    // Hashed by blocks of 16 tokens.
    assert!(bed.keyed(&asked, 0, Some(1)).is_empty());
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 8)]);
    // Hashed by nothing: the request states its size alone.
    let unhashed = RequestFacts {
        model: MODEL,
        seed: 7,
        now_micros: NOW,
        prompt_tokens: 8 * BLOCK,
        ..RequestFacts::default()
    };
    let key = || TokenKey {
        tokens: Cow::Borrowed(&asked),
        marker: 0,
    };
    let named = bed.decide_keyed(&unhashed, Some(key()), None);
    assert_eq!(named.expect("rows named").holders, [(1, 0, 8)]);
    // Hashed by blocks of 32 tokens, the size the model's workers reported:
    // the request's own size counts the match, not the document's.
    let spec = preset("cache_aware");
    let wide =
        PromptHashes::under_document(&spec, Some(2 * BLOCK), Some(&asked), None, HashSeeds::PLAIN);
    let named = bed.decide_keyed(&request(&wide, 7, NOW), Some(key()), None);
    assert_eq!(named.expect("rows named").holders, [(1, 0, 4)]);
}

/// Two conversations that begin alike for longer than a host hashes are each
/// named their own worker, by every block they were sent: the tree matches
/// the whole prompt. The record of hashed prefixes holds the first 256
/// blocks of a prompt, and reads the two as one.
#[test]
fn conversations_alike_past_the_hashed_blocks_are_named_their_own_workers() {
    let (affinity, _) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let head = tokens(1_000, 300);
    let turn = |tail: u32, blocks: u32| [head.as_slice(), &tokens(tail, blocks)].concat();
    bed.keyed(&turn(900_000, 1), 0, Some(0));
    bed.keyed(&turn(800_000, 1), 0, Some(3));
    assert_eq!(bed.keyed(&turn(900_000, 2), 0, None), [(0, 0, 301)]);
    assert_eq!(bed.keyed(&turn(800_000, 2), 0, None), [(3, 0, 301)]);
    // Named no entry, the same two requests are told apart by nothing.
    let hashes = hashed(&turn(900_000, 1));
    assert_eq!(hashes_of(&hashes), hashes_of(&hashed(&turn(800_000, 1))));
}

/// The naming of a cache partition is the host's. An entry named behind a
/// partition's marker shares no prefix with another partition's nor with a
/// plain one, and the marker is no part of the blocks a worker is named by.
#[test]
fn a_hosts_marker_keeps_two_partitions_apart() {
    let (affinity, _) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let marker = BLOCK as usize;
    // One page of ids no vocabulary holds, then the prompt.
    let under = |partition: u32, prompt: &[u32]| {
        let page = vec![partition | 1 << 31; marker];
        [page.as_slice(), prompt].concat()
    };
    let asked = tokens(1_000, 4);
    assert!(bed.keyed(&under(1, &asked), marker, Some(2)).is_empty());
    assert_eq!(bed.keyed(&under(1, &asked), marker, None), [(2, 0, 4)]);
    assert!(bed.keyed(&under(2, &asked), marker, None).is_empty());
    assert!(bed.keyed(&asked, 0, None).is_empty());
    // Another prompt of the partition shares the marker alone, and a
    // marker is no block of a prompt.
    assert!(bed
        .keyed(&under(1, &tokens(5_000, 4)), marker, None)
        .is_empty());
}

/// Many callers at once lose no write: each of sixty-four callers holds a
/// conversation that grows a block a turn and is sent to one worker, and
/// every turn behind the first is named that worker by all the blocks of the
/// turn before.
#[test]
fn every_callers_next_turn_is_named_the_worker_of_its_last() {
    const CALLERS: u32 = 64;
    const TURNS: u32 = 16;
    let (affinity, _) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 8);
    thread::scope(|scope| {
        for caller in 0..CALLERS {
            let bed = &bed;
            scope.spawn(move || {
                let own = caller % 8;
                let mut said = Vec::new();
                for turn in 1..=TURNS {
                    said.extend(tokens(caller * 1_000_000 + turn * BLOCK, 1));
                    let named = bed.keyed(&said, 0, Some(own));
                    let before: &[(u32, u32, u32)] = &[(own, 0, turn - 1)];
                    assert_eq!(named, before[..usize::from(turn > 1)], "turn {turn}");
                }
            });
        }
    });
}

/// The host and the engine read and write one tree. A prompt the host's own
/// policy stored for a worker is followed by the engine's next decision, and
/// what the engine decided stands in the host's tree under the worker's
/// address, as the host's own walk would have written it.
#[test]
fn the_host_and_the_engine_read_and_write_one_tree() {
    let (affinity, tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    // The host's policy placed a prompt on the worker at 2, in its own walk.
    let asked = tokens(1_000, 4);
    let at = address(2);
    tree.match_and_insert_with(&asked, |_| Some(at.as_str()));
    assert_eq!(bed.keyed(&asked, 0, None), [(2, 0, 4)]);
    // The engine places a longer one on the worker at 0: the host's tree
    // holds it for that worker, every token of it, and matches it whole.
    let longer = tokens(1_000, 6);
    assert_eq!(bed.keyed(&longer, 0, Some(0)), [(2, 0, 4)]);
    let held = tree.get_tenant_token_counts();
    assert_eq!(held.get(&address(0)), Some(&longer.len()));
    assert_eq!(held.get(&at), Some(&asked.len()));
    let found = tree.match_prefix_with_counts(&longer);
    assert_eq!(found.matched_token_count, longer.len());
    assert_eq!(&*found.tenant, address(0));
}

/// The tree stores a prompt for the first target of the pick its caller
/// sends to, asked best first, and for no worker where the caller sends to
/// none: a worker the caller passed over is named no holder.
///
/// The classic policy stores the worker it selected, among the candidates
/// its router can send to
/// (`model_gateway/src/policies/cache_aware.rs:3470-3491`, behind
/// `model_gateway/src/routers/common/placement.rs:504-509`, `:536-553`).
#[test]
fn a_tree_stores_a_prompt_for_the_first_target_its_caller_sends_to() {
    let (affinity, tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    // The workers at 1, 2 and 3, best first.
    let three = || {
        let mut targets = Targets::new();
        for position in 1..4 {
            targets.push(Target {
                position,
                id: bed.ids[position as usize],
                ..Target::default()
            });
        }
        Decision::Pick(Pick {
            targets,
            trace: Trace::default(),
            session_verdict: SessionVerdict::None,
        })
    };
    // One decision of `key` that picks the three, under a caller that sends
    // to the places `sent` names. Says the places the caller was asked of.
    let place = |key: &[u32], sent: &[u32]| {
        let hashes = hashed(key);
        let asked = RefCell::new(Vec::new());
        let entry = TokenKey {
            tokens: Cow::Borrowed(key),
            marker: 0,
        };
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&hashes, 7, NOW),
            None,
            |_| Some(entry),
            || None,
            |target| {
                asked.borrow_mut().push(target.position);
                sent.contains(&target.position)
            },
            |_| three(),
        );
        asked.into_inner()
    };
    // A caller that sends to the first target is asked of it alone.
    let (first, second, none) = (tokens(1_000, 4), tokens(2_000, 4), tokens(3_000, 4));
    assert_eq!(place(&first, &[1, 2, 3]), [1]);
    assert_eq!(bed.keyed(&first, 0, None), [(1, 0, 4)]);
    // One that passes the first over is stored for the second alone.
    assert_eq!(place(&second, &[2, 3]), [1, 2]);
    assert_eq!(bed.keyed(&second, 0, None), [(2, 0, 4)]);
    // One that sends to none is asked of each, and nothing is stored.
    assert_eq!(place(&none, &[]), [1, 2, 3]);
    assert!(bed.keyed(&none, 0, None).is_empty());
    let held = tree.get_tenant_token_counts();
    assert_eq!(held.get(&address(1)), Some(&first.len()));
    assert_eq!(held.get(&address(2)), Some(&second.len()));
    assert_eq!(held.len(), 2);
}

/// The host is asked for the entry once a decision, with the tree's page,
/// here four of the document's blocks, and only where it lends the place a
/// tree: not where it lends none, not for a model with an index of cache
/// events, and not under a profile that reads no affinity.
#[test]
fn the_entry_is_asked_for_once_and_only_where_the_host_lends_a_tree() {
    let hashes = prompt(1_000, 4);
    let pages = |lends: bool, indexed: bool, policy: &str| {
        let affinity = if lends {
            lending_by(4 * BLOCK as usize, indexed).0
        } else {
            hosted(indexed).0
        };
        let bed = Bed::new(affinity, preset(policy), 4);
        let mut pages = Vec::new();
        for _ in 0..2 {
            let asks = |page| {
                pages.push(page);
                None
            };
            let (list, ids) = (&bed.workers, &bed.ids);
            let facts = &mut request(&hashes, 7, NOW);
            decide(
                &bed.affinity,
                &bed.pool,
                &bed.spec,
                list,
                ids,
                facts,
                None,
                asks,
                || None,
                |_| true,
                |_| shed(),
            );
        }
        pages
    };
    assert_eq!(pages(true, false, "cache_aware"), [4 * BLOCK as usize; 2]);
    assert!(pages(false, false, "cache_aware").is_empty());
    assert!(pages(true, true, "cache_aware").is_empty());
    assert!(pages(true, false, "least_load").is_empty());
}

/// A request its host holds as text and names an entry for is decided
/// inside the host's string tree, as a tokenized one is inside its token
/// tree. A pick writes the whole text for its worker, a later request that
/// begins alike is named that worker by the windows the two share, the
/// deepest holders alone are named, a shed writes nothing and a worker its
/// host took out of the tree is named no more. The record of hashed
/// windows stays empty.
///
/// The classic policy matches and writes its string tree for a request
/// that holds no token ids (`select_worker_with_text` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_string_tree_names_the_worker_a_text_was_sent_to() {
    let (affinity, tree) = lending_text(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = text(1_000, 4);
    assert!(bed.worded(&asked, 0, None).is_empty());
    // The pick's own decision is named what stood before it.
    assert!(bed.worded(&asked, 0, Some(1)).is_empty());
    assert_eq!(bed.worded(&asked, 0, None), [(1, 0, 4)]);
    // A longer text that begins with it is named by the four windows sent.
    let longer = text(1_000, 6);
    assert_eq!(bed.worded(&longer, 0, Some(3)), [(1, 0, 4)]);
    // The worker sent the longer text is its one deepest holder, and both
    // hold the shorter one.
    assert_eq!(bed.worded(&longer, 0, None), [(3, 0, 6)]);
    assert_eq!(bed.worded(&asked, 0, None), [(1, 0, 4), (3, 0, 4)]);
    // The tree is the host's to keep: it takes out a worker that left.
    tree.remove_tenant_all(&TenantId::from(address(3)));
    assert_eq!(bed.worded(&longer, 0, None), [(1, 0, 4)]);
    // A text under one window names nobody, and is decided all the same.
    let short = &text(7_000, 1)[..WINDOW - 1];
    assert!(bed.worded(short, 0, Some(2)).is_empty());
    assert!(bed.worded(short, 0, None).is_empty());
    // A host that names no entry reads the record of hashed windows, which
    // no request above wrote.
    assert!(bed.holders(&worded(&asked), NOW).is_empty());
}

/// A request of text reads no index of cache events: an index speaks of
/// token blocks. So a host whose index holds a block still lends its string
/// tree, and a request of text is decided inside it. A tokenized request
/// under the same host is never asked for a text entry, and neither is a
/// request of text whose host lends no string tree: that one is named its
/// rows from the record of hashed windows, as before.
///
/// A host that keeps no index and lends both trees has each request
/// decided in the tree of its kind. A request of text is asked for its
/// token entry first, names none, and is decided in the string tree. A
/// tokenized one is decided in the token tree, and is never asked for a
/// text entry.
///
/// The classic policy asks its index only where the request holds token
/// ids, and sends every request of text to its string tree (`select_worker`
/// in `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_request_of_text_is_decided_in_the_string_tree_beside_an_index() {
    let (affinity, _tree) = lending_text(true);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = text(1_000, 4);
    assert!(bed.worded(&asked, 0, Some(2)).is_empty());
    assert_eq!(bed.worded(&asked, 0, None), [(2, 0, 4)]);
    // How often a decision of `facts` over `bed` asks for the text entry.
    let asks = |bed: &Bed, facts: &RequestFacts<'_>| {
        let asked = Cell::new(0);
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut { *facts },
            None,
            |_| None,
            || {
                asked.set(asked.get() + 1);
                None
            },
            |_| true,
            |_| picked(1, bed.ids[1]),
        );
        asked.get()
    };
    let (tokenized, of_text) = (prompt(1_000, 4), worded(&text(2_000, 4)));
    assert_eq!(asks(&bed, &request(&of_text, 7, NOW)), 1);
    assert_eq!(asks(&bed, &request(&tokenized, 7, NOW)), 0);
    // A host that lends its token tree alone: the record takes the text.
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    assert_eq!(asks(&bed, &request(&of_text, 7, NOW)), 0);
    assert_eq!(bed.holders(&of_text, NOW), [(1, 0, 4)]);

    // A host that lends both trees. How often a decision of `facts` asks
    // for each entry, the token tree's first, where the request names
    // `key` in the token tree and `words` in the string tree.
    let (affinity, sent_ids, sent_words) = lending_both();
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asks = |facts: &RequestFacts<'_>, key: Option<&[u32]>, words: Option<&str>| {
        let (keyed, worded) = (Cell::new(0), Cell::new(0));
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut { *facts },
            None,
            |_| {
                keyed.set(keyed.get() + 1);
                key.map(|tokens| TokenKey {
                    tokens: Cow::Borrowed(tokens),
                    marker: 0,
                })
            },
            || {
                worded.set(worded.get() + 1);
                words.map(|text| TextKey {
                    text: Cow::Borrowed(text),
                    marker: 0,
                })
            },
            |_| true,
            |_| picked(1, bed.ids[1]),
        );
        (keyed.get(), worded.get())
    };
    let (ids, words) = (tokens(3_000, 4), text(3_000, 4));
    let (tokenized, of_text) = (hashed(&ids), worded(&words));
    let entries = asks(&request(&of_text, 7, NOW), None, Some(&words));
    assert_eq!(entries, (1, 1), "a request of text");
    let entries = asks(&request(&tokenized, 7, NOW), Some(&ids), None);
    assert_eq!(entries, (1, 0), "a tokenized request");
    // Each prompt stands in the tree of its kind, for the worker picked.
    let found = sent_words.match_prefix_with_counts(&words);
    assert_eq!(found.matched_char_count, words.chars().count());
    assert_eq!(found.matched_tenants, [TenantId::from(address(1))]);
    let held = sent_ids.get_tenant_token_counts();
    assert_eq!(held.get(&address(1)), Some(&ids.len()));
    assert_eq!(sent_ids.total_token_size(), ids.len(), "no text");
}

/// A worker the model's index holds a block for is named by the string
/// tree's match as any other: a decision inside a tree reads no word of
/// who the index speaks for. The landed case beside an index stores the
/// index's block for another worker than the one the text goes to.
///
/// The classic policy sends every request of text to its string tree,
/// whatever its index holds (`select_worker` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_request_of_text_is_named_by_the_string_tree_for_a_worker_the_index_holds() {
    let tree = Arc::new(StringTree::new());
    let index = Arc::new(ChainIndex::new());
    // The index holds a block for the worker the text goes to.
    store(&index, 2, &[7]);
    let host = Arc::new(Host {
        index: Some(index),
        text: Some(Arc::clone(&tree)),
        ..Host::default()
    });
    let affinity = Affinity::default();
    affinity.set_indexes(host as Arc<dyn CacheIndexes>);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = text(1_000, 4);
    assert!(bed.worded(&asked, 0, Some(2)).is_empty());
    assert_eq!(bed.worded(&asked, 0, None), [(2, 0, 4)]);
    // The tree holds the text for that worker and no other.
    let found = tree.match_prefix_with_counts(&asked);
    assert_eq!(found.matched_char_count, asked.chars().count());
    assert_eq!(found.matched_tenants, [TenantId::from(address(2))]);
}

/// A string tree counts chars and a request of text is sized in bytes: a
/// match is turned into the bytes of the text it matched before it is
/// counted in the request's windows. So a text of several bytes a char is
/// named by every window it was sent, where a char read as a byte would
/// name fewer.
///
/// The classic policy divides the chars its tree matched by the text's
/// chars (`select_worker_with_text` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_text_of_several_bytes_a_char_is_counted_in_its_bytes() {
    let (affinity, _tree) = lending_text(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    for (first, fill) in [(2_000, 'é'), (3_000, '語'), (4_000, '🦀')] {
        let asked = text_of(first, 4, fill);
        assert_eq!(asked.len(), 4 * WINDOW);
        assert!(asked.chars().count() < asked.len(), "{fill}");
        assert!(bed.worded(&asked, 0, Some(1)).is_empty(), "{fill}");
        assert_eq!(bed.worded(&asked, 0, None), [(1, 0, 4)], "{fill}");
    }
    // The bytes of a match behind a marker, by its chars.
    assert_eq!(bytes_behind("é語🦀a", 0, 0), 0);
    assert_eq!(bytes_behind("é語🦀a", 0, 2), 5);
    assert_eq!(bytes_behind("é語🦀a", 1, 3), 7);
    assert_eq!(bytes_behind("é語🦀a", 1, 9), 8);
    // A match that ends inside the marker holds nothing of the prompt.
    assert_eq!(bytes_behind("é語🦀a", 3, 2), 0);
}

/// The naming of a cache partition is the host's, for text as for tokens.
/// An entry named behind a partition's marker shares no prefix with another
/// partition's nor with a plain one, and the marker's chars are no part of
/// the windows a worker is named by, whatever their bytes.
#[test]
fn a_hosts_marker_keeps_two_partitions_of_text_apart() {
    // Eighteen chars no prompt begins with: one of them of two bytes, and
    // then each of one byte. A key of one byte a char is counted by its
    // chars, with no pass over it.
    type Under = fn(u32, &str) -> String;
    let wide: Under = |partition, prompt| format!("\u{1}§{partition:016x}{prompt}");
    let narrow: Under = |partition, prompt| format!("\u{1}{partition:016x}\u{1}{prompt}");
    let marker = 18;
    let asked = text(1_000, 4);
    for (under, bytes) in [(wide, marker + 1), (narrow, marker)] {
        assert_eq!(under(1, "").len(), bytes);
        let (affinity, _) = lending_text(false);
        let bed = Bed::new(affinity, preset("cache_aware"), 4);
        assert!(bed.worded(&under(1, &asked), marker, Some(2)).is_empty());
        assert_eq!(bed.worded(&under(1, &asked), marker, None), [(2, 0, 4)]);
        assert!(bed.worded(&under(2, &asked), marker, None).is_empty());
        assert!(bed.worded(&asked, 0, None).is_empty());
        // Another prompt of the partition shares the marker alone, and a
        // marker is no window of a prompt.
        let other = under(1, &text(5_000, 4));
        assert!(bed.worded(&other, marker, None).is_empty());
        // A prompt that ends fourteen bytes short of its fourth window is
        // named by three: the marker's bytes fill no window of it.
        let cut = &asked[..4 * WINDOW - 14];
        assert_eq!(bed.worded(&under(1, cut), marker, None), [(2, 0, 3)]);
    }
}

/// The string tree stores a text for the first target of the pick its
/// caller sends to, asked best first, and for no worker where the caller
/// sends to none, as the token tree does
/// ([`a_tree_stores_a_prompt_for_the_first_target_its_caller_sends_to`]).
/// The host and the engine read and write one tree: what the engine decided
/// stands in the host's tree under the worker's address, every char of it.
#[test]
fn a_string_tree_stores_a_text_for_the_first_target_its_caller_sends_to() {
    let (affinity, tree) = lending_text(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    // The workers at 1 and 2, best first.
    let two = || {
        let mut targets = Targets::new();
        for position in 1..3 {
            targets.push(Target {
                position,
                id: bed.ids[position as usize],
                ..Target::default()
            });
        }
        Decision::Pick(Pick {
            targets,
            trace: Trace::default(),
            session_verdict: SessionVerdict::None,
        })
    };
    // One decision of `key` that picks the two, under a caller that sends
    // to the places `sent` names. Says the places the caller was asked of.
    let place = |key: &str, sent: &[u32]| {
        let hashes = worded(key);
        let asked = RefCell::new(Vec::new());
        let entry = TextKey {
            text: Cow::Borrowed(key),
            marker: 0,
        };
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&hashes, 7, NOW),
            None,
            |_| None,
            || Some(entry),
            |target| {
                asked.borrow_mut().push(target.position);
                sent.contains(&target.position)
            },
            |_| two(),
        );
        asked.into_inner()
    };
    let (first, second, none) = (text(1_000, 4), text(2_000, 4), text(3_000, 4));
    assert_eq!(place(&first, &[1, 2]), [1]);
    assert_eq!(bed.worded(&first, 0, None), [(1, 0, 4)]);
    // A caller that passes the first over is stored for the second alone.
    assert_eq!(place(&second, &[2]), [1, 2]);
    assert_eq!(bed.worded(&second, 0, None), [(2, 0, 4)]);
    // One that sends to none is asked of each, and nothing is stored.
    assert_eq!(place(&none, &[]), [1, 2]);
    assert!(bed.worded(&none, 0, None).is_empty());
    for (text, at) in [(&first, 1), (&second, 2)] {
        let found = tree.match_prefix_with_counts(text);
        assert_eq!(found.matched_char_count, text.chars().count());
        assert_eq!(found.matched_tenants, [TenantId::from(address(at))]);
    }
    assert_eq!(tree.get_tenant_char_count().len(), 2);
}

/// A request of text its host holds to places is named the workers its
/// string tree's match hands among those places, as a tokenized one is
/// ([`a_held_request_is_named_a_trees_workers_among_its_places`]).
#[test]
fn a_held_request_of_text_is_named_a_trees_workers_among_its_places() {
    let (affinity, _) = lending_text(false);
    let bed = Bed::new(affinity, sampled(4), 20);
    let asked = text(1_000, 4);
    for at in 0..8 {
        bed.worded(&asked, 0, Some(at));
    }
    let hashes = worded(&asked);
    let key = || TextKey {
        text: Cow::Borrowed(&asked),
        marker: 0,
    };
    for seed in 0..32 {
        let facts = request(&hashes, seed, NOW);
        let named = bed.decide_entered(&facts, Some(&[2, 5]), (None, Some(key())), None);
        let named = named.expect("rows named");
        assert_eq!(named.holders, [(2, 0, 4), (5, 0, 4)], "seed {seed}");
    }
    // Stated to no place, the sample cuts the eight to four.
    assert_eq!(bed.worded(&asked, 0, None).len(), 4);
}

/// A pick that meets a held record waits for nobody and leaves nothing out:
/// its thread keeps the write, reads it at its own namings and writes it on
/// behind its next pick. Until then another thread reads the record alone.
#[test]
fn a_pick_that_meets_a_held_record_keeps_its_prefix_for_its_thread() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    assert!(bed.holders(&asked, NOW).is_empty());
    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };
    // What a thread that keeps no write reads.
    let elsewhere = || {
        thread::scope(|scope| {
            let reader = scope.spawn(|| bed.holders(&asked, NOW));
            reader.join().expect("a reader")
        })
    };
    {
        let _reader = record.predictor.read();
        bed.decide(&request(&asked, 7, NOW), Some(2));
    }
    assert_eq!(bed.affinity.left_out(), 0);
    assert_eq!(bed.holders(&asked, NOW), [(2, 0, 4)]);
    assert!(elsewhere().is_empty());
    // The thread's next pick writes the kept prefix on.
    bed.decide(&request(&asked, 7, NOW), Some(2));
    assert_eq!(elsewhere(), [(2, 0, 4)]);
    // The whole prompt again, its last use a second old: the write that
    // would move it meets a held record and leaves nothing out.
    {
        let _reader = record.predictor.read();
        bed.decide(&request(&asked, 7, NOW + 2 * SECOND), Some(2));
    }
    assert_eq!(bed.affinity.left_out(), 0);
    // A leave stops a write a thread keeps: the thread names the worker no
    // more, and the write goes uncounted with the thread's buffers.
    let longer = prompt(1_000, 6);
    thread::scope(|scope| {
        let keeper = scope.spawn(|| {
            {
                let _reader = record.predictor.read();
                bed.decide(&request(&longer, 7, NOW + 2 * SECOND), Some(2));
            }
            let kept = bed.holders(&longer, NOW + 2 * SECOND);
            bed.affinity.forget(bed.ids[2]);
            (kept, bed.holders(&longer, NOW + 2 * SECOND))
        });
        let (kept, left) = keeper.join().expect("a keeper");
        assert_eq!(kept, [(2, 0, 6)]);
        assert!(left.is_empty());
    });
    assert_eq!(bed.affinity.left_out(), 0);
}

/// A thread keeps the newest eight writes. A longer prompt sent to the
/// worker a kept write of its leading part waits for takes that write's
/// place as the newest of them: the two turns are one write. A ninth write
/// pushes the oldest out, which is counted, and the thread names its worker
/// no more.
#[test]
fn a_thread_keeps_eight_writes_and_a_conversations_turns_are_one() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let first = prompt(1_000, 4);
    let second = prompt(1_000, 6);
    let other = prompt(2_000, 4);
    let ninth = prompt(9_000, 4);
    // The first read makes the model's record.
    assert!(bed.holders(&first, NOW).is_empty());
    thread::scope(|scope| {
        let keeper = scope.spawn(|| {
            let eight = bed.affinity.while_held(|| {
                // Another prompt's write is kept between the two turns.
                bed.decide(&request(&first, 7, NOW), Some(2));
                bed.decide(&request(&other, 7, NOW), Some(2));
                bed.decide(&request(&second, 7, NOW), Some(2));
                for more in 3..9 {
                    let asked = prompt(1_000 * more, 4);
                    bed.decide(&request(&asked, 7, NOW), Some(2));
                }
                let eight = (
                    bed.affinity.left_out(),
                    bed.holders(&other, NOW),
                    bed.holders(&second, NOW),
                );
                bed.decide(&request(&ninth, 7, NOW), Some(2));
                eight
            });
            let nine = (
                bed.affinity.left_out(),
                bed.holders(&other, NOW),
                bed.holders(&second, NOW),
                bed.holders(&ninth, NOW),
            );
            (eight, nine)
        });
        let (eight, nine) = keeper.join().expect("a keeper");
        assert_eq!(eight, (0, vec![(2, 0, 4)], vec![(2, 0, 6)]));
        // The oldest is the other prompt's: the second turn's is newer.
        assert_eq!(nine, (1, vec![], vec![(2, 0, 6)], vec![(2, 0, 4)]));
    });
    // The eight end unwritten with their thread's buffers, each counted.
    assert_eq!(bed.affinity.left_out(), 9);
}

/// A write behind a column that meets a held record is kept as one behind
/// a pick is: the thread's next column reads it beside what the record
/// holds, another thread's column reads the record alone, and the next
/// turn's write is kept in its turn. Once the record is free, the write
/// behind the thread's next column stands, the kept write is written on
/// behind it, and nothing waits.
#[test]
fn a_column_reads_the_write_its_thread_keeps_and_the_next_turn_is_kept() {
    let (affinity, _, _) = hosted(false);
    let prefills = Bed::new(affinity, paired(), 4);
    let asked = prompt(1_000, 4);
    let longer = prompt(1_000, 6);
    let stood = prompt(2_000, 4);
    // A column of the prompt and the write behind it for the worker `to`.
    let sent = |prompt: &PromptHashes, to: usize| {
        let facts = request(prompt, 7, NOW);
        let (column, read) = prefills.column(&facts, Phase::Prefill);
        let read = read.expect("the record speaks for the prefills");
        record_prefix(
            &prefills.pool,
            Phase::Prefill,
            &facts,
            read,
            prefills.ids[to],
        );
        held(&column)
    };
    let read =
        |prompt: &PromptHashes| held(&prefills.column(&request(prompt, 7, NOW), Phase::Prefill).0);
    // What a thread that keeps no write reads.
    let elsewhere = |prompt: &PromptHashes| {
        thread::scope(|scope| {
            let reader = scope.spawn(|| read(prompt));
            reader.join().expect("a reader")
        })
    };
    // The blocks a column reads for one worker of the four.
    let at = |worker: usize, blocks: u32| {
        let mut column = [(0, 0); 4];
        column[worker] = (0, blocks);
        column
    };
    // The record holds another prompt for the fourth worker.
    assert_eq!(sent(&stood, 3), [(0, 0); 4]);
    thread::scope(|scope| {
        let keeper = scope.spawn(|| {
            assert_eq!(
                prefills.affinity.while_held(|| sent(&asked, 1)),
                [(0, 0); 4]
            );
            assert_eq!(prefills.affinity.left_out(), 0);
            assert_eq!(read(&asked), at(1, 4));
            // A thread that keeps a write reads the record as before.
            assert_eq!(read(&stood), at(3, 4));
            assert_eq!(elsewhere(&asked), [(0, 0); 4]);
            // The next turn, the record held again: its column reads the
            // turn before, and its write is kept in its turn.
            assert_eq!(prefills.affinity.while_held(|| sent(&longer, 1)), at(1, 4));
            assert_eq!(read(&longer), at(1, 6));
            assert_eq!(elsewhere(&longer), [(0, 0); 4]);
            // The record free, the turn sent to the third worker: its write
            // stands and the kept write is written on behind it.
            assert_eq!(sent(&longer, 2), at(1, 6));
            assert_eq!(elsewhere(&longer), [(0, 0), (0, 6), (0, 6), (0, 0)]);
        });
        keeper.join().expect("a keeper");
    });
    // Nothing waited as the thread's buffers went.
    assert_eq!(prefills.affinity.left_out(), 0);
}

/// A pick of a worker the record holds the whole request for leaves nothing
/// out, however many workers hold it: a read keeps eight of them, and the
/// pick's own entry of the column says it of the others. At the tick of the
/// last use it writes nothing, and two seconds on its write adds no block.
#[test]
fn a_pick_of_a_holder_past_the_kept_rows_leaves_nothing_out() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, handed_whole(), 10);
    let asked = prompt(1_000, 4);
    for row in 0..10 {
        bed.decide(&request(&asked, 7, NOW), Some(row));
    }
    let (column, _) = bed.column(&request(&asked, 7, NOW), Phase::Single);
    assert_eq!(held(&column), [(0, 4); 10]);
    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };
    for now in [NOW, NOW + 2 * SECOND] {
        let _reader = record.predictor.read();
        bed.decide(&request(&asked, 7, now), Some(9));
    }
    assert_eq!(bed.affinity.left_out(), 0);
}

/// A worker that leaves behind a decision's read of the record and ahead
/// of its write is given no place: the record held none for it at the
/// leave, and the pick's write gives it none. So where the worker is told
/// gone, and where the standing workers are said whole without it.
#[test]
fn a_worker_that_leaves_under_its_first_pick_is_given_no_place() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    let holders_behind = |leave: &dyn Fn()| {
        let decision = decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&asked, 7, NOW),
            None,
            |_| None,
            || None,
            |_| true,
            |_| {
                leave();
                picked(2, bed.ids[2])
            },
        );
        assert!(decision.pick().is_some());
        bed.holders(&asked, NOW)
    };
    assert!(holders_behind(&|| bed.affinity.forget(bed.ids[2])).is_empty());
    let standing = [bed.ids[0], bed.ids[1], bed.ids[3]];
    assert!(holders_behind(&|| bed.affinity.keep(&standing)).is_empty());
}

/// A worker that left behind a decision's read of the record is given no
/// place where the write meets a record made anew: the list's table was
/// read for a model whose profile reads no affinity, the record left the
/// map, the worker left while the map held no record of its model, and the
/// first model's next table made another. A record leaves at the timer's
/// sweep once it holds no place, and with its places where the follow of
/// a host's registry lets it go.
#[test]
fn a_worker_that_left_is_given_no_place_in_a_record_made_anew() {
    let mut document = PolicySpec::preset("least_load").expect("the preset");
    let reader = PolicySpec::preset("cache_aware").expect("the preset");
    let mut binding = reader.bindings[0].clone();
    binding.matcher.model = MODEL.into();
    document.bindings.insert(0, binding);
    document.profiles.extend(reader.profiles);
    for let_go in [false, true] {
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let (affinity, _, _) = hosted(false);
        let bed = Bed::new(affinity, spec, 4);
        let asked = prompt(1_000, 4);
        let facts = request(&asked, 7, NOW);
        let other = RequestFacts {
            model: "model-b",
            ..facts
        };
        if let_go {
            // The follow names a record that holds a place alone.
            let (_, first) = bed.column(&facts, Phase::Single);
            let first = first.expect("the record speaks for the list");
            record_prefix(&bed.pool, Phase::Single, &facts, first, bed.ids[0]);
        }
        let (_, read) = bed.column(&facts, Phase::Single);
        let read = read.expect("the record speaks for the list");
        bed.column(&other, Phase::Single);
        if let_go {
            bed.affinity.let_go(&bed.affinity.unstated(|_| false));
        } else {
            bed.affinity.sweep();
        }
        assert!(bed.affinity.records.lock().is_empty(), "the record left");
        bed.affinity.forget(bed.ids[2]);
        bed.column(&facts, Phase::Single);
        record_prefix(&bed.pool, Phase::Single, &facts, read, bed.ids[2]);
        let behind = held(&bed.column(&facts, Phase::Single).0);
        assert_eq!(behind, [(0, 0); 4], "let go: {let_go}");
    }
}

/// Two records that leave the map in one sweep with unequal counts of
/// leaves, four and one: the record made next begins past the greater,
/// whichever of the two the map holds first.
#[test]
fn a_record_made_anew_begins_past_the_greatest_count_that_left() {
    for (many, one) in [(MODEL, "model-b"), ("model-b", MODEL)] {
        let (affinity, _, _) = hosted(false);
        // Each record is the map's alone and holds no place.
        drop(affinity.record_of(many, 8));
        for worker in 1..=3 {
            affinity.forget(WorkerId(worker));
        }
        drop(affinity.record_of(one, 8));
        affinity.forget(WorkerId(4));
        affinity.sweep();
        assert!(affinity.records.lock().is_empty(), "the two left");
        let anew = affinity.record_of(one, 8).left.load(Ordering::Relaxed);
        assert_eq!(anew, 5, "{many} left with four");
    }
}

/// A pin's answer made ahead of a leave and handed through the lookup
/// behind it gives its worker no place; with no leave between it writes.
#[test]
fn an_answer_made_ahead_of_a_leave_gives_its_worker_no_place() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    let holders_behind = |leave: &dyn Fn()| {
        let (leaves, made) = (bed.affinity.leaves(), picked(2, bed.ids[2]));
        leave();
        let decision = decide_behind(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&asked, 7, NOW),
            |_| None,
            || None,
            |_| true,
            || None,
            leaves,
            made,
        );
        assert!(decision.pick().is_some());
        bed.holders(&asked, NOW)
    };
    assert_eq!(holders_behind(&|| bed.affinity.forget(bed.ids[2])), []);
    let standing = [bed.ids[0], bed.ids[1], bed.ids[3]];
    assert_eq!(holders_behind(&|| bed.affinity.keep(&standing)), []);
    assert_eq!(holders_behind(&|| ()), [(2, 0, 4)]);
}

/// A list of no worker is read no table: its request is named no holder, no
/// record of sent prefixes is made, and the host is asked for no index and
/// no tree.
#[test]
fn a_list_of_no_worker_keeps_no_record() {
    let (affinity, host, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 0);
    let asked = prompt(1_000, 4);
    assert!(bed.holders(&asked, NOW).is_empty());
    assert!(bed.affinity.records.lock().is_empty());
    assert_eq!(host.asked.load(Ordering::Relaxed), 0);
}

/// A model's record leaves the map at the timer's sweep once its worker
/// left and no list's table holds it, and what it left out stays counted.
/// Until then it stays: a worker that stands keeps its prefixes when its
/// list is published again, and a table that stands reads its record.
#[test]
fn a_record_leaves_behind_its_models_last_worker() {
    let (affinity, _, _) = hosted(false);
    let mut bed = Bed::new(affinity, preset("cache_aware"), 1);
    let asked = prompt(1_000, 4);
    bed.decide(&request(&asked, 7, NOW), Some(0));
    bed.affinity.records.lock()[MODEL]
        .left_out
        .store(1, Ordering::Relaxed);
    // The list is published again, with no table yet; its worker stands.
    bed.pool = PoolAffinity::default();
    bed.affinity.sweep();
    assert_eq!(bed.holders(&asked, NOW), [(0, 0, 4)]);
    bed.affinity.forget(bed.ids[0]);
    bed.affinity.sweep();
    assert_eq!(bed.affinity.records.lock().len(), 1, "a table reads it");
    // The list falls, and its table with it.
    bed.pool = PoolAffinity::default();
    bed.affinity.sweep();
    assert!(bed.affinity.records.lock().is_empty());
    assert_eq!(bed.affinity.left_out(), 1);
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

/// A single request of token ids under a model whose index holds a block
/// is decided by the index alone: the record of sent prefixes is neither
/// read nor written for it. A worker the record holds the request for is
/// named no more once the index holds a block, and a pick beside the index
/// leaves no entry. A request of text reads no index, and the record speaks
/// for it as before.
///
/// The classic policy takes its event path wherever the model's index
/// holds a block and the request holds token ids, and reads no tree there
/// (`event_index_for` and `select_worker_scanned` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn the_index_decides_a_single_request_of_token_ids_alone() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    // The index holds no block yet, so its host hands none: the record
    // takes the request's prefix for the fourth worker.
    bed.decide(&request(&asked, 7, NOW), Some(3));
    assert_eq!(bed.holders(&asked, NOW), [(3, 0, 4)]);
    // A column filled beside no index keeps what it read of the record.
    let (_, before) = bed.column(&request(&asked, 7, NOW), Phase::Single);
    let before = before.expect("the record speaks while the index holds no block");
    // The index holds two of the request's blocks for the first worker, and
    // decides alone from the table's next read.
    store(&index, 0, &hashes_of(&asked)[..2]);
    assert_eq!(bed.holders(&asked, NOW + SECOND), [(0, 2, 0)]);
    // The list's column reads no record either: the fourth worker's entry
    // holds no prefix.
    let later = request(&asked, 7, NOW + SECOND);
    let (column, read) = bed.column(&later, Phase::Single);
    assert_eq!(held(&column), [(2, 0), (0, 0), (0, 0), (0, 0)]);
    assert!(read.is_none(), "the fill reads no record");
    // A pick of a third worker writes nothing, and neither does a write
    // behind the column filled before the index held a block: the table
    // that stands is asked at the write. The record holds the first pick's
    // four blocks and no more.
    bed.decide(&later, Some(2));
    record_prefix(&bed.pool, Phase::Single, &later, before, bed.ids[1]);
    assert_eq!(bed.holders(&asked, NOW + SECOND), [(0, 2, 0)]);
    {
        let standing = bed.pool.table.load();
        let table = standing.as_deref().expect("the list's table");
        let record = table.record.as_ref().expect("the model's record");
        assert_eq!(record.predictor.read().len(), 4, "one pick's blocks");
    }
    // The same windows as text: the record takes the pick's write.
    let words = worded(&text(1_000, 4));
    bed.decide(&request(&words, 7, NOW + SECOND), Some(2));
    assert_eq!(bed.holders(&words, NOW + SECOND), [(2, 0, 4)]);
}

/// A request hashed by its text asks no index, so the record of sent
/// prefixes speaks for every worker of its list. The pick of a worker the
/// index holds blocks for writes the text's prefix for it, and the repeat is
/// named that worker. So is a write that met a held record and waits with
/// its thread. The classic router sends a request with no token ids to its
/// tree of sent texts whatever its index of cache events holds
/// (`model_gateway/src/policies/cache_aware.rs:1511-1538`), and reads and
/// writes that tree for any worker of the list (`:3519-3596`).
///
/// A request hashed by token blocks answers as before beside it: its pick of
/// that worker writes no entry, and the index alone names the worker.
#[test]
fn a_request_of_text_is_recorded_for_a_worker_the_index_holds_blocks_for() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let tokens = prompt(1_000, 4);
    store(&index, 0, &hashes_of(&tokens)[..2]);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let said = text(1_000, 4);
    let asked = worded(&said);
    let facts = request(&asked, 7, NOW);
    assert_eq!(
        (facts.hash_kind, facts.block_hashes.len()),
        (HashKind::Text, 4)
    );
    assert!(
        bed.holders(&asked, NOW).is_empty(),
        "nobody was sent the text, whatever the index holds"
    );
    bed.decide(&facts, Some(0));
    assert_eq!(bed.holders(&asked, NOW), [(0, 0, 4)]);
    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };
    assert_eq!(record.predictor.read().len(), 4, "one placement a window");

    bed.decide(&request(&tokens, 7, NOW), Some(0));
    assert_eq!(bed.holders(&tokens, NOW), [(0, 2, 0)]);
    assert_eq!(
        record.predictor.read().len(),
        4,
        "a token request's pick of the worker writes nothing"
    );

    // Another text's write meets a held record: its thread reads it kept.
    let waiting = text(5_000, 4);
    let waiting = worded(&waiting);
    {
        let _reader = record.predictor.read();
        bed.decide(&request(&waiting, 7, NOW), Some(0));
    }
    assert_eq!(record.predictor.read().len(), 4, "the write waits");
    assert_eq!(bed.holders(&waiting, NOW), [(0, 0, 4)]);
    assert_eq!(bed.affinity.left_out(), 0);
}

/// A profile whose prefix stage reads sent prefixes alone
/// (`source: predictor`) is named the worker a request with token ids was
/// sent to, though the index holds a block for that worker, and its pick of
/// such a worker is written. No stage of it reads what the index holds, so
/// the record speaks to it for every worker of its list. So does a write
/// that met a held record and waits with its thread. Before, the record was
/// neither read nor written for such a worker, and the request was never
/// named one.
///
/// The gateway's own cache-aware selection reads its approximate token tree
/// alone where the model has no index of cache events that holds a block
/// (`model_gateway/src/policies/cache_aware.rs:1511-1534`, `event_index_for`
/// at `:1802-1818`). There one walk of the tree matches a request with token
/// ids against every worker a prefix was sent to and stores it for the
/// worker chosen (`select_worker_with_tokens`, `:3430-3516`). It has no
/// setting that reads the tree alone beside an index that holds blocks, and
/// it keeps one policy a model (`model_gateway/src/policies/registry.rs:242`):
/// the reading beside an index is the engine's own.
///
/// The preset, whose prefix stage reads cache events too, answers as before
/// over the same list: its pick of such a worker writes nothing. So the
/// profile that reads alone knows no send the preset placed there. Once the
/// index holds the request for the worker the record names, the two stand as
/// one holder with both counts.
#[test]
fn a_profile_that_reads_sent_prefixes_alone_is_named_a_worker_the_index_holds_blocks_for() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let other = hashes_of(&prompt(9_000, 1));
    store(&index, 0, &other);
    store(&index, 1, &other);
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 7, NOW);

    let merged = Bed::new(affinity, preset("cache_aware"), 2);
    merged.decide(&facts, Some(1));
    assert!(
        merged.holders(&asked, NOW).is_empty(),
        "the preset's pick of a worker with a block writes nothing"
    );

    // The same list, index and record under the profile that reads alone.
    let alone = Bed {
        pool: PoolAffinity::default(),
        spec: alone_sampled(2),
        ..merged
    };
    assert!(
        alone.holders(&asked, NOW).is_empty(),
        "no send the preset placed there is known"
    );
    alone.decide(&facts, Some(1));
    assert_eq!(alone.holders(&asked, NOW), [(1, 0, 4)]);
    let record = {
        let standing = alone.pool.table.load();
        let table = standing.as_deref().expect("the list's table");
        assert!(
            table.alone && !table.unheard_any,
            "the index holds a block for every worker"
        );
        Arc::clone(table.record.as_ref().expect("the model's record"))
    };
    assert_eq!(record.predictor.read().len(), 4, "one placement a block");

    // Another prompt's write meets a held record: its thread reads it kept.
    let waiting = prompt(5_000, 4);
    {
        let _reader = record.predictor.read();
        alone.decide(&request(&waiting, 7, NOW), Some(0));
    }
    assert_eq!(record.predictor.read().len(), 4, "the write waits");
    assert_eq!(alone.holders(&waiting, NOW), [(0, 0, 4)]);

    // The table read a second on finds the request's blocks in the index.
    store(&index, 1, &hashes_of(&asked));
    assert_eq!(alone.holders(&asked, NOW + SECOND), [(1, 4, 4)]);
    assert_eq!(alone.affinity.left_out(), 0);
}

/// Under a profile that reads sent prefixes alone, a worker the record and
/// the index both name stands once in a naming, with both counts. The
/// record's holders are named first and the index is asked in the room they
/// leave. Such a worker has taken two of the sample's places before the two
/// are made one, so the naming can stand short of the sample, by an entry
/// of the index's own.
#[test]
fn a_worker_the_record_and_the_index_both_name_stands_once_in_a_naming() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    for at in 0..3 {
        store(&index, at, &hashes_of(&asked));
    }
    let bed = Bed::new(affinity, alone_sampled(2), 3);
    bed.decide(&request(&asked, 7, NOW), Some(1));
    let mut short = 0;
    for seed in 0..32 {
        let named = bed.decide(&request(&asked, seed, NOW), None);
        match named.expect("rows").holders[..] {
            // The index's one place went to the worker the record named.
            [(1, 4, 4)] => short += 1,
            [(0, 4, 0), (1, 0, 4)] | [(1, 0, 4), (2, 4, 0)] => {}
            ref other => panic!("seed {seed}: {other:?}"),
        }
    }
    assert!(
        (1..32).contains(&short),
        "{short} of 32 namings stand one short"
    );
}

/// A profile that reads cache events too is named no send to a worker the
/// index holds a block for, beside a profile that reads sent prefixes alone
/// over the same list and record. The second sent a whole prompt to such a
/// worker, and keeps another such write with its thread where it met a held
/// record. A pair's decode side under the first reads past both: it still
/// holds the shorter prefix it sent itself to a worker with no block, for
/// that worker alone, in its list's column and in the holders a decode
/// asked alone is named. A single request under the first is named no send
/// at all: the index decides it alone.
#[test]
fn a_profile_that_reads_cache_events_is_named_no_send_to_a_worker_the_index_holds_blocks_for() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let other = hashes_of(&prompt(9_000, 1));
    store(&index, 0, &other);
    store(&index, 1, &other);
    // The index holds no block for the third worker: the decode side's
    // sends to it are written, two blocks of each prompt.
    let side = Bed::new(affinity, paired(), 3);
    let (written, waiting) = (prompt(1_000, 4), prompt(5_000, 4));
    for first in [1_000, 5_000] {
        let begun = prompt(first, 2);
        let facts = request(&begun, 7, NOW);
        let (_, read) = side.column(&facts, Phase::Decode);
        let read = read.expect("the record speaks for the third worker");
        record_prefix(&side.pool, Phase::Decode, &facts, read, side.ids[2]);
    }
    // What the side holds of each whole prompt: its list's column, and the
    // holders a decode asked alone is named.
    let holds = |side: &Bed| {
        [&written, &waiting].map(|prompt| {
            let facts = request(prompt, 7, NOW);
            let (column_of, _) = side.column(&facts, Phase::Decode);
            let asked_alone = RequestFacts {
                phase: Phase::Decode,
                ..facts
            };
            let named = side.decide(&asked_alone, None).expect("rows");
            (held(&column_of), named.holders)
        })
    };
    let short = (vec![(0, 0), (0, 0), (0, 2)], vec![(2, 0, 2)]);
    assert_eq!(holds(&side), [short.clone(), short.clone()]);

    // The profile that reads alone sends each whole prompt to a worker with
    // a block: one write stands in the record, the other waits.
    let alone = Bed {
        pool: PoolAffinity::default(),
        spec: alone_sampled(2),
        ..side
    };
    alone.decide(&request(&written, 7, NOW), Some(1));
    let record = {
        let standing = alone.pool.table.load();
        let table = standing.as_deref().expect("the list's table");
        Arc::clone(table.record.as_ref().expect("the model's record"))
    };
    let placements = record.predictor.read().len();
    {
        let _reader = record.predictor.read();
        alone.decide(&request(&waiting, 7, NOW), Some(0));
    }
    assert_eq!(record.predictor.read().len(), placements, "the write waits");
    assert_eq!(alone.holders(&written, NOW), [(1, 0, 4)]);
    assert_eq!(alone.holders(&waiting, NOW), [(0, 0, 4)]);

    // The side holds no more: the second worker was sent the first prompt's
    // two blocks too, and neither its entry nor a naming holds them.
    let side = Bed {
        pool: PoolAffinity::default(),
        spec: paired(),
        ..alone
    };
    assert_eq!(holds(&side), [short.clone(), short]);
    // A single request under the preset is named neither profile's send.
    let merged = Bed {
        pool: PoolAffinity::default(),
        spec: preset("cache_aware"),
        ..side
    };
    assert!(merged.holders(&written, NOW).is_empty());
    assert!(merged.holders(&waiting, NOW).is_empty());
    assert_eq!(merged.affinity.left_out(), 0);
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

/// Past the picker's sample the warming rows a decision is named are a
/// draw of the sample from its seed, each row as likely as another and
/// none twice. Within the sample every one is named.
#[test]
fn warming_rows_past_the_sample_are_a_draw_each_row_as_likely() {
    // Fourteen workers hold a level, and the ten behind them nothing: thin
    // beside a fleet whose median holds one.
    let bed = |sample| {
        let (affinity, _, index) = hosted(true);
        let index = index.expect("the index");
        let level = hashes_of(&prompt(1_000, 8));
        for at in 0..14 {
            store(&index, at, &level);
        }
        Bed::new(affinity, sampled(sample), 24)
    };
    let other = prompt(50_000, 2);
    let thin: Vec<u32> = (14..24).collect();
    let named = bed(16).decide(&request(&other, 1, NOW), None);
    let named = named.expect("rows");
    assert_eq!((named.warming, named.thin), (thin.clone(), true));
    let cut = bed(4);
    let mut drawn = std::collections::BTreeMap::new();
    for seed in 0..1_000 {
        let named = cut.decide(&request(&other, seed, NOW), None);
        let rows = named.expect("rows").warming;
        assert_eq!(rows.len(), 4, "seed {seed}");
        assert!(rows.is_sorted_by(|a, b| a < b), "{rows:?}");
        for row in rows {
            *drawn.entry(row).or_insert(0_u32) += 1;
        }
    }
    // 1,000 draws of 4 in 10: about 400 a row.
    assert_eq!(drawn.keys().copied().collect::<Vec<_>>(), thin);
    assert!(
        drawn.values().all(|count| (300..=500).contains(count)),
        "{drawn:?}"
    );
}

/// The holders' draw and the warming rows' draw are on streams of their
/// own. Two workers are the request's only holders and the list's only
/// warming rows, under a sample of one: over the seeds each list names
/// either worker beside either worker of the other. On one stream the two
/// draws would read one value, and one list's row would give the other's.
#[test]
fn the_holders_and_the_warming_rows_are_drawn_apart() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    // Two workers hold a level of blocks of their own.
    let level = hashes_of(&prompt(1_000, 8));
    for at in 0..2 {
        store(&index, at, &level);
    }
    // Two hold the request's one block and nothing more: thin.
    let asked = prompt(9_000, 1);
    for at in 2..4 {
        store(&index, at, &hashes_of(&asked));
    }
    let bed = Bed::new(affinity, sampled(1), 4);
    let mut pairs = std::collections::BTreeSet::new();
    for seed in 0..64 {
        let named = bed.decide(&request(&asked, seed, NOW), None);
        let named = named.expect("rows");
        assert_eq!(
            (named.holders.len(), named.warming.len()),
            (1, 1),
            "seed {seed}"
        );
        pairs.insert((named.holders[0].0, named.warming[0]));
    }
    assert_eq!(pairs.len(), 4, "{pairs:?}");
}

/// A document rules the warming rows by its own stage. One whose profile
/// holds no warm-up stage names nobody. One whose stage holds another
/// number rules by it: a worker that holds a quarter of the fleet's median
/// is thin under half of it, and is not under a fifth.
#[test]
fn a_document_rules_the_warming_rows_by_its_own_stage() {
    // Three workers hold a level, the fourth a quarter of what they hold,
    // and the fifth nothing.
    let warming = |spec| {
        let (affinity, _, index) = hosted(true);
        let index = index.expect("the index");
        let level = hashes_of(&prompt(1_000, 8));
        for at in 0..3 {
            store(&index, at, &level);
        }
        store(&index, 3, &level[..2]);
        let bed = Bed::new(affinity, spec, 5);
        let named = bed.decide(&request(&prompt(50_000, 2), 7, NOW), None);
        let named = named.expect("rows");
        (named.warming, named.thin)
    };
    assert_eq!(warming(warming_by(60)), (vec![3, 4], true));
    let unruled = ruled_by(warm_up(60), |profile| {
        profile.score.remove(0);
    });
    assert_eq!(warming(unruled), (vec![], false));
    let fifth = WarmupSliceParams {
        thin_ratio: Some(0.2),
        ..warm_up(60)
    };
    assert_eq!(warming(ruled_by(fifth, |_| {})), (vec![4], true));
}

/// A list's table is read once, then again by a decision a second later:
/// the decisions between read what stands and ask the host nothing.
#[test]
fn a_lists_table_stands_for_a_second() {
    let (affinity, host, _) = hosted(true);
    // The index holds a block, another worker's of another prompt: a host
    // hands no index that holds none, and is then asked for its tree too.
    let index = host.index.as_ref().expect("the index");
    store(index, 3, &hashes_of(&prompt(9_000, 1)));
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
    store(index, 2, &hashes_of(&asked));
    assert!(bed.holders(&asked, NOW + SECOND + 1).is_empty());
    assert_eq!(bed.holders(&asked, NOW + 2 * SECOND), [(2, 4, 0)]);
}

/// A number of the index that one worker handed back and the next worker
/// took names the new worker to the next decision: the list's table, which
/// kept the number for the first, is read again before its second is up.
#[test]
fn a_number_that_changed_hands_names_its_new_worker() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("an index");
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    // Worker 0 holds another prompt under its number when the table is read.
    let (number, map) = stored(&index, 0, &hashes_of(&prompt(5_000, 2)));
    assert!(bed.holders(&asked, NOW).is_empty());
    // Its stream ends and worker 1's opens: worker 1 takes the number and
    // alone holds the asked prompt.
    index.remove_worker(number, map);
    store(&index, 1, &hashes_of(&asked));
    assert_eq!(index.worker_id(&address(1)), Some(number));
    assert_eq!(bed.holders(&asked, NOW), [(1, 4, 0)]);
}

/// A number that changes hands under a decision's own lookup names nobody
/// to it: the table read the number for its former worker, and the lookup
/// finds the blocks of its new one. The next decision reads the table again
/// and names the new worker. A side's column holds nothing of the request
/// under such a lookup, and the next column holds the new worker's blocks.
#[test]
fn a_number_handed_on_under_a_lookup_names_nobody() {
    /// A chain index whose first lookup is met by a handoff: the stream of
    /// worker 0 ends and worker 1 stores the asked prompt under its number.
    struct HandedOn {
        index: ChainIndex,
        handoff: Mutex<Option<(u32, ChainBlockMap, Vec<u64>)>>,
    }

    impl CacheIndex for HandedOn {
        fn worker_number(&self, address: &str) -> Option<u32> {
            self.index.worker_id(address)
        }

        fn numbers_reused(&self) -> u64 {
            self.index.ids_reused()
        }

        fn worker_blocks(&self, number: u32) -> usize {
            self.index.worker_block_count(number)
        }

        fn deepest_into(
            &self,
            block_hashes: &[u64],
            block_size: u32,
            within: &[u64],
            cap: usize,
            below: &mut dyn FnMut(u32) -> u32,
            kept: &mut Vec<(u32, u32)>,
        ) {
            if let Some((number, map, hashes)) = self.handoff.lock().take() {
                self.index.remove_worker(number, map);
                store(&self.index, 1, &hashes);
            }
            CacheIndex::deepest_into(
                &self.index,
                block_hashes,
                block_size,
                within,
                cap,
                below,
                kept,
            );
        }
    }

    struct Lends(Arc<HandedOn>);

    impl CacheIndexes for Lends {
        fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
            Some(Arc::clone(&self.0) as Arc<dyn CacheIndex>)
        }

        fn block_size(&self, _: &str) -> Option<u32> {
            Some(BLOCK)
        }
    }

    let asked = prompt(1_000, 4);
    let handed_on = |spec: AppliedSpec| {
        let index = ChainIndex::new();
        // Worker 0 holds another prompt under its number when the table is
        // read.
        let (number, map) = stored(&index, 0, &hashes_of(&prompt(5_000, 2)));
        let handed = Arc::new(HandedOn {
            index,
            handoff: Mutex::new(Some((number, map, hashes_of(&asked)))),
        });
        let bed = Bed::new(Affinity::default(), spec, 4);
        bed.affinity
            .set_indexes(Arc::new(Lends(Arc::clone(&handed))) as Arc<dyn CacheIndexes>);
        (bed, handed, number)
    };

    let (bed, handed, number) = handed_on(preset("cache_aware"));
    assert!(bed.holders(&asked, NOW).is_empty());
    assert_eq!(handed.index.worker_id(&address(1)), Some(number));
    assert_eq!(bed.holders(&asked, NOW), [(1, 4, 0)]);

    // A side's column is filled by a lookup of its own.
    let (bed, handed, number) = handed_on(paired());
    let facts = request(&asked, 7, NOW);
    assert_eq!(held(&bed.column(&facts, Phase::Prefill).0), [(0, 0); 4]);
    assert_eq!(handed.index.worker_id(&address(1)), Some(number));
    assert_eq!(
        held(&bed.column(&facts, Phase::Prefill).0),
        [(0, 0), (4, 0), (0, 0), (0, 0)]
    );
}

/// A profile whose stages read no affinity is named no row: its decision
/// is handed the request as it stands, and the host is asked nothing. Under
/// a document none of whose bindings reads affinity, its list keeps no
/// table.
#[test]
fn a_profile_that_reads_no_affinity_is_named_no_row() {
    let (affinity, host, _) = hosted(true);
    let bed = Bed::new(affinity, preset("least_load"), 4);
    let asked = prompt(1_000, 4);
    assert_eq!(bed.decide(&request(&asked, 7, NOW), Some(1)), None);
    assert_eq!(host.asked.load(Ordering::Relaxed), 0);
    assert!(bed.pool.table.load().is_none(), "no table is read");
}

/// Under a document where one model's profile reads affinity in the single
/// phase and every other place reads none, each request is decided as under
/// a document of its own. A request of a phase that reads none is named no
/// row, and its list keeps no table. Another model's request of the
/// reader's phase is named none behind its list's table. The reader is
/// named its rows.
#[test]
fn a_profile_that_reads_affinity_is_named_its_rows_beside_one_that_reads_none() {
    let mut document = PolicySpec::preset("least_load").expect("the preset");
    let reader = PolicySpec::preset("cache_aware").expect("the preset");
    let mut binding = reader.bindings[0].clone();
    binding.matcher.model = MODEL.into();
    binding.matcher.phase = vec![Phase::Single];
    document.bindings.insert(0, binding);
    document.profiles.extend(reader.profiles);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = spec.engine();
    assert!(engine.needs_for(MODEL, Phase::Single).affinity);
    assert!(!engine.needs_for("model-b", Phase::Single).affinity);
    assert!(!engine.needs_for_phase(Phase::Decode).affinity);
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 2, &hashes_of(&asked));
    let bed = Bed::new(affinity, spec, 4);
    // The phase asked is the request's own: no binding of this one reads
    // affinity, so no table is read.
    let decode = RequestFacts {
        phase: Phase::Decode,
        ..request(&asked, 7, NOW)
    };
    assert_eq!(bed.decide(&decode, Some(1)), None);
    assert!(bed.pool.table.load().is_none(), "no table is read");
    // In the reader's phase the other model's profile reads none: its
    // list's table is what says so.
    let other = RequestFacts {
        model: "model-b",
        ..request(&asked, 7, NOW)
    };
    assert_eq!(bed.decide(&other, Some(1)), None);
    assert!(bed.pool.table.load().is_some(), "the list's table says so");
    assert_eq!(bed.holders(&asked, NOW), [(2, 4, 0)]);
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
            let mut request = request(&asked, seed, NOW);
            let decision = decide(
                &bed.affinity,
                &bed.pool,
                &bed.spec,
                &bed.workers,
                &bed.ids,
                &mut request,
                None,
                |_| None,
                || None,
                |_| true,
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
    let (first, _) = bed.column(&request(&asked, 7, NOW), Phase::Prefill);
    assert_eq!(held(&first), [(2, 0), (0, 0), (4, 0), (0, 0)]);
    assert_eq!(indexed(&first), [(2, 0), (0, 0), (7, 0), (0, 0)]);
    // Half a second on, by the table that stands, each worker is half a
    // second older.
    let (second, _) = bed.column(&request(&asked, 7, NOW + SECOND / 2), Phase::Prefill);
    for (then, later) in first.iter().zip(&second) {
        assert_eq!(later.admitted_age_ms, then.admitted_age_ms + 500);
    }
    // The table read a second on holds what the index gained since.
    store(&index, 0, &hashes_of(&prompt(5_000, 5)));
    let (third, _) = bed.column(&request(&asked, 7, NOW + SECOND), Phase::Prefill);
    assert_eq!(indexed(&third), [(7, 5), (0, 0), (7, 0), (0, 0)]);
    assert_eq!(held(&third), held(&first));
    // A request hashed by nothing is held by nobody, and its column is whole.
    let bare = RequestFacts {
        model: MODEL,
        now_micros: NOW + SECOND,
        ..RequestFacts::default()
    };
    assert_eq!(held(&bed.column(&bare, Phase::Prefill).0), [(0, 0); 4]);
}

/// With no index the record of sent prefixes speaks for a pair side's
/// workers: a pick written behind a column is what the side's next column
/// holds for its worker, and the two lists of a pair keep their workers
/// apart in the model's one record.
#[test]
fn a_sides_pick_is_recorded_and_its_next_column_holds_it() {
    let (affinity, _, _) = hosted(false);
    let prefills = Bed::new(affinity, paired(), 4);
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 7, NOW);
    let (column_of, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(0, 0); 4]);
    let read = read.expect("the record speaks for the prefills");
    record_prefix(
        &prefills.pool,
        Phase::Prefill,
        &facts,
        read,
        prefills.ids[1],
    );
    let sent_to_one = [(0, 0), (0, 4), (0, 0), (0, 0)];
    let (column_of, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), sent_to_one);
    // A worker that is not of the list writes nothing, and neither does a
    // write for another place than the list's table stands for.
    let read = read.expect("the record speaks for the prefills");
    record_prefix(&prefills.pool, Phase::Prefill, &facts, read, WorkerId(999));
    record_prefix(&prefills.pool, Phase::Decode, &facts, read, prefills.ids[2]);
    assert_eq!(
        held(&prefills.column(&facts, Phase::Prefill).0),
        sent_to_one
    );
    // The decodes are another list of the model, with workers of their own.
    let decode_pool = PoolAffinity::default();
    let (decodes, _) = pool(3);
    let decode_ids: Vec<WorkerId> = (100..103).map(WorkerId).collect();
    let mut filled = Vec::new();
    let mut decode_column = || {
        let read = column(
            &prefills.affinity,
            &decode_pool,
            &prefills.spec,
            &decodes,
            &decode_ids,
            Phase::Decode,
            &facts,
            &mut filled,
        );
        (held(&filled), read)
    };
    let (holds, read) = decode_column();
    assert_eq!(holds, [(0, 0); 3]);
    let read = read.expect("the record speaks for the decodes");
    record_prefix(&decode_pool, Phase::Decode, &facts, read, decode_ids[2]);
    assert_eq!(decode_column().0, [(0, 0), (0, 0), (0, 4)]);
    assert_eq!(
        held(&prefills.column(&facts, Phase::Prefill).0),
        sent_to_one
    );
    assert_eq!(prefills.affinity.left_out(), 0);
}

/// A pair side's column holds a text's prefix for a worker the index holds
/// blocks for. A request hashed by its text asks no index, so the record of
/// sent prefixes speaks for every worker of the side's list: its pick is
/// written behind the column for such a worker, and the side's next column
/// holds it. So where the index holds a block for every worker of the list.
/// The classic router keeps a tree of sent texts a side, and reads and
/// writes it for a request with no token ids whatever its index holds
/// (`model_gateway/src/policies/cache_aware.rs:1511-1538`, `:3519-3596`).
///
/// A request hashed by token blocks answers as before beside it: a pick of
/// a worker the index speaks for writes nothing, and a list whose every
/// worker the index speaks for reads no record.
#[test]
fn a_sides_column_holds_a_text_sent_to_a_worker_the_index_holds_blocks_for() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let tokens = prompt(1_000, 4);
    let other = hashes_of(&prompt(9_000, 1));
    store(&index, 0, &other);
    store(&index, 1, &hashes_of(&tokens)[..2]);
    let prefills = Bed::new(affinity, paired(), 3);
    let said = text(1_000, 4);
    let asked = worded(&said);

    // The index holds no block for the third prefill yet.
    let facts = request(&asked, 7, NOW);
    let (column_of, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(0, 0); 3]);
    let read = read.expect("the record speaks for the prefills");
    record_prefix(
        &prefills.pool,
        Phase::Prefill,
        &facts,
        read,
        prefills.ids[1],
    );
    let sent_to_one = [(0, 0), (0, 4), (0, 0)];
    assert_eq!(
        held(&prefills.column(&facts, Phase::Prefill).0),
        sent_to_one
    );
    let by_tokens = request(&tokens, 7, NOW);
    let by_index = [(0, 0), (2, 0), (0, 0)];
    let (column_of, read) = prefills.column(&by_tokens, Phase::Prefill);
    assert_eq!(held(&column_of), by_index);
    let read = read.expect("the record speaks for the third prefill");
    record_prefix(
        &prefills.pool,
        Phase::Prefill,
        &by_tokens,
        read,
        prefills.ids[1],
    );
    assert_eq!(
        held(&prefills.column(&by_tokens, Phase::Prefill).0),
        by_index
    );

    // The table read a second on finds a block for every prefill.
    store(&index, 2, &other);
    let later = NOW + SECOND;
    let (column_of, read) = prefills.column(&request(&asked, 7, later), Phase::Prefill);
    assert_eq!(held(&column_of), sent_to_one);
    assert!(read.is_some(), "a request of text reads the record");
    let (column_of, read) = prefills.column(&request(&tokens, 7, later), Phase::Prefill);
    assert_eq!(held(&column_of), by_index);
    assert!(read.is_none(), "a token request reads none");
    assert_eq!(prefills.affinity.left_out(), 0);
}

/// A pair side's column is read by the word its reader states, and the
/// write behind its pick by the word of its read. The pool's table is not
/// asked: its word is the profile's that a request of the phase asked alone
/// resolves to, which can be another's. A side whose profile reads sent
/// prefixes alone is handed what was sent to a worker the index holds a
/// block for, and its pick of such a worker is written. A side whose
/// profile reads cache events too is handed none of it over the same list,
/// and its pick of such a worker writes nothing, as before: the reader that
/// reads alone knows no send the other placed there. The public
/// [`column()`] answers as the second, under either table.
#[test]
fn a_sides_column_is_read_by_its_readers_word_of_the_record() {
    let pair = PolicySpec::pair_presets("cache_aware", "cache_aware");
    let mut pair = reading_alone(pair.expect("the pair presets"));
    pair.generation = 1;
    let read_alone = AppliedSpec::validate(&pair).expect("the pair document validates");
    for (spec, pools_word) in [(paired(), false), (read_alone, true)] {
        let (affinity, _, index) = hosted(true);
        let index = index.expect("the index");
        let other = hashes_of(&prompt(9_000, 1));
        store(&index, 0, &other);
        store(&index, 1, &other);
        let prefills = Bed::new(affinity, spec, 3);
        let asked = prompt(1_000, 4);
        let facts = request(&asked, 7, NOW);
        let side = |alone: bool| {
            let mut filled = Vec::new();
            let read = side_column(
                &prefills.affinity,
                &prefills.pool,
                &prefills.spec,
                &prefills.workers,
                &prefills.ids,
                Phase::Prefill,
                alone,
                &facts,
                &mut filled,
            );
            let read = read.expect("the record speaks for the third prefill");
            (held(&filled), read)
        };
        let record = |read: PrefixRead| {
            record_prefix(
                &prefills.pool,
                Phase::Prefill,
                &facts,
                read,
                prefills.ids[1],
            );
        };
        let nobody = [(0, 0); 3];

        let (holds, read) = side(false);
        assert_eq!(holds, nobody);
        record(read);
        assert_eq!(side(true).0, nobody, "a pick read by the index's word");

        let (holds, read) = side(true);
        assert_eq!(holds, nobody);
        record(read);
        assert_eq!(side(true).0, [(0, 0), (0, 4), (0, 0)]);
        assert_eq!(side(false).0, nobody, "the index speaks to that reader");
        assert_eq!(held(&prefills.column(&facts, Phase::Prefill).0), nobody);

        let standing = prefills.pool.table.load();
        let table = standing.as_deref().expect("the prefills' table");
        assert_eq!(table.alone, pools_word, "the pool's own word");
        assert_eq!(prefills.affinity.left_out(), 0);
    }
}

/// A single decision reads the record of sent prefixes by its own reader's
/// word, and never by a pair's side's. A binding of the decode phase alone
/// whose prefix stage reads cache events alone (`source: index`) is listed
/// ahead of a pair whose decode profile reads sent prefixes alone. The
/// decodes' pool keeps the record for that side, and a decode request asked
/// alone is decided by a profile that reads none of it: the table's word is
/// `false`. So that request is named the index's holder under a sample of
/// one, though the side sent the prompt to another worker. Its pick of a
/// worker the index holds a block for writes nothing, and its pick of a
/// worker with no block is written, as under any profile that reads cache
/// events. The side reads what it sent itself throughout.
///
/// Before, the pool's word there was the side's, and the record's holders
/// were named ahead of the index's for that request. The classic router has
/// no twin: it keeps one policy a side
/// (`model_gateway/src/policies/registry.rs:1338-1354`), so no second
/// profile decides over a side's list.
#[test]
fn a_single_decision_reads_the_record_by_its_own_word_beside_a_pairs_side() {
    use crate::document::spec::{AffinitySource, BestParams, PickStage, ScoreSpec};

    let pair = PolicySpec::pair_presets("round_robin", "cache_aware");
    let mut document = reading_alone(pair.expect("the pair presets"));
    document.generation = 1;
    let stock = PolicySpec::preset("cache_aware").expect("the preset");
    let mut ahead = stock.bindings[0].clone();
    ahead.matcher.phase = vec![Phase::Decode];
    ahead.profile = Some("ahead".into());
    document.bindings.insert(0, ahead);
    let by_index = stock.profiles.into_values().next();
    let mut by_index = by_index.expect("the preset's profile");
    for stage in &mut by_index.score {
        if let ScoreSpec::Params(ScoreStage::PrefixAffinity(prefix)) = stage {
            prefix.source = Some(AffinitySource::Index);
        }
    }
    by_index.pick = PickStage::Best(BestParams { sample: 1 }).into();
    document.profiles.insert("ahead".into(), by_index);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = spec.engine();
    assert!(!engine.reads_sent_for(MODEL, Phase::Decode));
    assert!(engine.side_reads_sent_alone(MODEL, Phase::Decode));
    assert_eq!(engine.pool_record_for(MODEL, Phase::Decode), Some(false));
    assert_eq!(engine.affinity_rows_cap(MODEL, Phase::Decode), 1);

    // The index holds the prompt for the third decode, another prompt's
    // block for the first two and nothing for the fourth.
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let other = hashes_of(&prompt(9_000, 1));
    store(&index, 0, &other);
    store(&index, 1, &other);
    store(&index, 2, &hashes_of(&asked));
    let decodes = Bed::new(affinity, spec, 4);
    let paired = request(&asked, 7, NOW);
    let asked_alone = RequestFacts {
        phase: Phase::Decode,
        ..paired
    };
    let side = || {
        let mut filled = Vec::new();
        let read = side_column(
            &decodes.affinity,
            &decodes.pool,
            &decodes.spec,
            &decodes.workers,
            &decodes.ids,
            Phase::Decode,
            true,
            &paired,
            &mut filled,
        );
        (held(&filled), read.expect("the record speaks to the side"))
    };

    // The pair's side sends the prompt to the second decode.
    let (holds, read) = side();
    assert_eq!(holds, [(0, 0), (0, 0), (4, 0), (0, 0)]);
    record_prefix(&decodes.pool, Phase::Decode, &paired, read, decodes.ids[1]);
    let sent = [(0, 0), (0, 4), (4, 0), (0, 0)];
    assert_eq!(side().0, sent);

    // The decode request asked alone is named the index's holder.
    let named = decodes.decide(&asked_alone, None).expect("rows");
    assert_eq!(named.holders, [(2, 4, 0)]);
    decodes.decide(&asked_alone, Some(0));
    assert_eq!(side().0, sent, "a pick of a worker with a block");
    decodes.decide(&asked_alone, Some(3));
    assert_eq!(side().0, [(0, 0), (0, 4), (4, 0), (0, 4)]);

    let standing = decodes.pool.table.load();
    let table = standing.as_deref().expect("the decodes' table");
    assert!(!table.alone && table.record.is_some());
    assert_eq!(decodes.affinity.left_out(), 0);
}

/// Each side of a pair is decided at its own phase and keeps both records
/// beside an index that holds a block: its column holds the index's blocks
/// for a worker the index speaks for and the prefix sent for another. The
/// index decides a single request alone.
#[test]
fn a_sides_column_keeps_the_record_beside_an_index() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 0, &hashes_of(&asked)[..2]);
    let prefills = Bed::new(affinity, paired(), 4);
    let facts = request(&asked, 7, NOW);
    let (column_of, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(2, 0), (0, 0), (0, 0), (0, 0)]);
    let read = read.expect("the record speaks for the other prefills");
    record_prefix(
        &prefills.pool,
        Phase::Prefill,
        &facts,
        read,
        prefills.ids[1],
    );
    let (column_of, _) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(2, 0), (0, 4), (0, 0), (0, 0)]);
}

/// A profile that reads affinity and is named no row is handed the column
/// whole: here its two warm-up stages rule by different windows, so no one
/// ruling names its warming rows.
#[test]
fn a_profile_named_no_row_is_handed_the_column() {
    let spec = handed_whole();
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 2, &hashes_of(&asked));
    let bed = Bed::new(affinity, spec, 4);
    let mut seen = None;
    let decision = decide(
        &bed.affinity,
        &bed.pool,
        &bed.spec,
        &bed.workers,
        &bed.ids,
        &mut request(&asked, 7, NOW),
        None,
        |_| None,
        || None,
        |_| true,
        |facts| {
            assert!(facts.named.is_none());
            seen = Some(held(facts.affinity));
            shed()
        },
    );
    assert!(decision.pick().is_none());
    let seen = seen.expect("a decision");
    assert_eq!(seen, [(0, 0), (0, 0), (4, 0), (0, 0)]);
}

/// A profile that is named no row and reads sent prefixes is handed what
/// its own picks sent: a request's pick writes its prefix for its worker,
/// and the column of the repeat holds it.
#[test]
fn a_profile_named_no_row_is_handed_the_prefix_its_pick_sent() {
    let asked = prompt(1_000, 4);
    let bed = Bed::new(hosted(false).0, handed_whole(), 4);
    assert!(bed.decide(&request(&asked, 7, NOW), Some(1)).is_none());
    let mut seen = None;
    decide(
        &bed.affinity,
        &bed.pool,
        &bed.spec,
        &bed.workers,
        &bed.ids,
        &mut request(&asked, 7, NOW),
        None,
        |_| None,
        || None,
        |_| true,
        |facts| {
            seen = Some(held(facts.affinity));
            shed()
        },
    );
    assert_eq!(seen.expect("a decision"), [(0, 0), (0, 4), (0, 0), (0, 0)]);
}

/// A profile that is named no row is handed a text's prefix for a worker
/// the index holds blocks for: its pick of that worker is written, and the
/// column of the repeat holds it. A request hashed by its text asks no
/// index, as under the classic router
/// (`model_gateway/src/policies/cache_aware.rs:1511-1538`, `:3519-3596`).
/// A request hashed by token blocks answers as before beside it: its pick
/// of that worker writes nothing, and its column holds the index's blocks.
#[test]
fn a_profile_named_no_row_is_handed_a_text_sent_to_a_worker_the_index_holds_blocks_for() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let tokens = prompt(1_000, 4);
    store(&index, 1, &hashes_of(&tokens)[..2]);
    let bed = Bed::new(affinity, handed_whole(), 4);
    let said = text(1_000, 4);
    let asked = worded(&said);
    let handed = |prompt: &PromptHashes| {
        let mut seen = None;
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(prompt, 7, NOW),
            None,
            |_| None,
            || None,
            |_| true,
            |facts| {
                seen = Some(held(facts.affinity));
                shed()
            },
        );
        seen.expect("a decision")
    };
    assert_eq!(handed(&asked), [(0, 0); 4]);
    assert!(bed.decide(&request(&asked, 7, NOW), Some(1)).is_none());
    assert_eq!(handed(&asked), [(0, 0), (0, 4), (0, 0), (0, 0)]);
    let by_index = [(0, 0), (2, 0), (0, 0), (0, 0)];
    assert_eq!(handed(&tokens), by_index);
    assert!(bed.decide(&request(&tokens, 7, NOW), Some(1)).is_none());
    assert_eq!(handed(&tokens), by_index);
}

/// A profile that is named no row and reads sent prefixes alone is handed
/// the prefix its pick sent to a worker the index holds blocks for: the
/// worker's entry of the column then holds both counts. Under the same
/// profile with a prefix stage that reads cache events too, the pick of that
/// worker writes nothing and its entry holds the index's blocks alone, as
/// before: ahead of the other profile's send and behind it.
#[test]
fn a_profile_named_no_row_reads_the_record_alone_for_a_worker_the_index_holds_blocks_for() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 1, &hashes_of(&asked)[..2]);
    let handed = |bed: &Bed| {
        let mut seen = None;
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&asked, 7, NOW),
            None,
            |_| None,
            || None,
            |_| true,
            |facts| {
                seen = Some(held(facts.affinity));
                shed()
            },
        );
        seen.expect("a decision")
    };
    let by_index = [(0, 0), (2, 0), (0, 0), (0, 0)];
    let merged = Bed::new(affinity, handed_whole(), 4);
    assert!(merged.decide(&request(&asked, 7, NOW), Some(1)).is_none());
    assert_eq!(handed(&merged), by_index);

    let alone = Bed {
        pool: PoolAffinity::default(),
        spec: named_no_row(reading_alone),
        ..merged
    };
    assert_eq!(handed(&alone), by_index);
    assert!(alone.decide(&request(&asked, 7, NOW), Some(1)).is_none());
    assert_eq!(handed(&alone), [(0, 0), (2, 4), (0, 0), (0, 0)]);

    // The first profile is handed none of that send.
    let merged = Bed {
        pool: PoolAffinity::default(),
        spec: handed_whole(),
        ..alone
    };
    assert_eq!(handed(&merged), by_index);
}

/// A pick states what its first target holds of the request, read from the
/// rows the engine named for its decision. A worker the index speaks for is
/// stated by the blocks the index holds for it, of the request's blocks. A
/// worker the rows do not name holds nothing: a miss of the request's
/// blocks, which is not "nothing stated". The index decides such a
/// request alone, so the miss's own pick writes nothing. Under a model
/// with no index a worker is stated by the blocks it was sent, read from
/// the predictor: there the miss's own pick wrote them. A shed holds no
/// pick.
#[test]
fn a_pick_states_its_workers_entry_of_the_rows_the_engine_named() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let hashes = hashes_of(&asked);
    store(&index, 5, &hashes[..2]);
    store(&index, 2, &hashes);
    let bed = Bed::new(affinity, preset("cache_aware"), 8);
    let facts = request(&asked, 7, NOW);
    assert_eq!(bed.states(&facts, 2), holds(4, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 5), holds(2, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 0), holds(0, 4, OverlapSource::None));
    // That pick wrote nothing: the index decides the request alone.
    assert_eq!(bed.states(&facts, 0), holds(0, 4, OverlapSource::None));
    assert_eq!(bed.stated(&facts, None, shed()), Overlap::NONE);
    // No index: the miss's pick wrote the request's prefix for its worker
    // in the record.
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 8);
    assert_eq!(bed.states(&facts, 0), holds(0, 4, OverlapSource::None));
    assert_eq!(bed.states(&facts, 0), holds(4, 4, OverlapSource::Predictor));
}

/// A pick states its worker's entry of a column the request's host filled,
/// read as an entry of the engine's own rows is: the larger of its two
/// readings, the index's on a tie, a miss where the entry holds nothing or
/// the column ends before the row. The engine names no row for such a
/// request and reads no table.
#[test]
fn a_pick_states_its_workers_entry_of_a_column_its_host_filled() {
    let (affinity, host, _) = hosted(true);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    let entry = |index: u32, predicted: u32| AffinityFacts {
        overlap_blocks: index,
        predicted_blocks: predicted,
        ..AffinityFacts::default()
    };
    let column = [entry(0, 0), entry(3, 0), entry(1, 4), entry(2, 2)];
    let facts = RequestFacts {
        affinity: &column,
        ..request(&asked, 7, NOW)
    };
    assert_eq!(bed.states(&facts, 1), holds(3, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 2), holds(4, 4, OverlapSource::Predictor));
    assert_eq!(bed.states(&facts, 3), holds(2, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 0), holds(0, 4, OverlapSource::None));
    // A row past a shorter column holds nothing, as a row no host named.
    let short = RequestFacts {
        affinity: &column[..3],
        ..facts
    };
    assert_eq!(bed.states(&short, 3), holds(0, 4, OverlapSource::None));
    assert_eq!(host.asked.load(Ordering::Relaxed), 0, "no index is asked");
    assert!(bed.pool.table.load().is_none(), "no table is read");
}

/// A pick states its worker's entry of the rows the request's host named in
/// a column's place: a row named by what it holds, a row not named as a
/// miss of the request's blocks.
#[test]
fn a_pick_states_its_workers_entry_of_the_rows_its_host_named() {
    let (affinity, host, _) = hosted(true);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    let holders = [
        AffinityHolder {
            row: 1,
            overlap_blocks: 3,
            predicted_blocks: 0,
        },
        AffinityHolder {
            row: 3,
            overlap_blocks: 0,
            predicted_blocks: 2,
        },
    ];
    let facts = RequestFacts {
        named: Some(AffinityRows {
            holders: &holders,
            warming: &[],
            thin: false,
            mean_in_flight: None,
        }),
        ..request(&asked, 7, NOW)
    };
    assert_eq!(bed.states(&facts, 1), holds(3, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 3), holds(2, 4, OverlapSource::Predictor));
    assert_eq!(bed.states(&facts, 0), holds(0, 4, OverlapSource::None));
    assert_eq!(host.asked.load(Ordering::Relaxed), 0, "no index is asked");
}

/// A profile that is named no row is handed the column whole, and its pick
/// states its worker's entry of that column.
#[test]
fn a_pick_states_its_workers_entry_of_the_column_the_engine_filled() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 2, &hashes_of(&asked)[..3]);
    let bed = Bed::new(affinity, handed_whole(), 4);
    let facts = request(&asked, 7, NOW);
    assert_eq!(bed.states(&facts, 2), holds(3, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 1), holds(0, 4, OverlapSource::None));
}

/// What the lookup states of `answer`'s first target, for a request of
/// `key`'s tokens whose host names them as its entry in the token tree it
/// lends: the request is decided inside the tree's match, and its descent
/// writes the prompt for the worker a pick the profile is told of goes to.
/// The lookup returns the answer it was handed but for that statement.
fn answered_in_tree(bed: &Bed, key: &[u32], answer: Decision) -> Overlap {
    let hashes = hashed(key);
    let entry = TokenKey {
        tokens: Cow::Borrowed(key),
        marker: 0,
    };
    bed.stated(&request(&hashes, 7, NOW), Some(entry), answer)
}

/// [`answered_in_tree`] behind a pick of the worker at `row` by the
/// profile's stages.
fn stated_in_tree(bed: &Bed, key: &[u32], row: u32) -> Overlap {
    answered_in_tree(bed, key, picked(row, bed.ids[row as usize]))
}

/// A request decided inside its host's token tree states its pick's entry
/// of the rows the tree's match named, as a pick over the engine's own rows
/// does. A prompt the tree holds for nobody is a miss of the request's
/// blocks. Once the tree holds it for a worker, a pick of that worker states
/// the match's tokens counted in the request's blocks, read as sent. A
/// longer prompt that begins with it states the blocks the two share, of
/// its own.
#[test]
fn a_pick_made_inside_a_hosts_token_tree_states_the_matchs_blocks() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    let miss = holds(0, 4, OverlapSource::None);
    assert_eq!(stated_in_tree(&bed, &asked, 1), miss);
    // That pick wrote the prompt for its worker, and the match names it.
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 4)]);
    let sent = holds(4, 4, OverlapSource::Predictor);
    assert_eq!(stated_in_tree(&bed, &asked, 1), sent);
    let longer = tokens(1_000, 6);
    let shared = holds(4, 6, OverlapSource::Predictor);
    assert_eq!(stated_in_tree(&bed, &longer, 1), shared);
    // No entry named: the record of hashed prefixes, which nothing wrote.
    let hashes = hashed(&asked);
    assert_eq!(bed.states(&request(&hashes, 7, NOW), 1), miss);
}

/// A request of text decided inside its host's string tree states its
/// pick's entry of the rows the tree's match named, as one decided inside
/// the token tree does
/// ([`a_pick_made_inside_a_hosts_token_tree_states_the_matchs_blocks`]): a
/// miss while the tree holds the text for nobody, then the bytes the match
/// holds counted in the request's windows, read as sent. A longer text
/// that begins with it states the windows the two share, of its own, and a
/// text of several bytes a char states its windows too. A pick that is
/// none of the match's holders states a miss.
#[test]
fn a_pick_made_inside_a_hosts_string_tree_states_the_matchs_windows() {
    let (affinity, _tree) = lending_text(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let stated = |key: &str, row: u32| {
        let hashes = worded(key);
        let entry = TextKey {
            text: Cow::Borrowed(key),
            marker: 0,
        };
        let answer = picked(row, bed.ids[row as usize]);
        bed.stated_entered(&request(&hashes, 7, NOW), (None, Some(entry)), answer)
    };
    let miss = holds(0, 4, OverlapSource::None);
    let sent = holds(4, 4, OverlapSource::Predictor);
    for (first, fill) in [(1_000, '.'), (2_000, '語')] {
        let asked = text_of(first, 4, fill);
        assert_eq!(stated(&asked, 1), miss, "{fill}");
        // That pick wrote the text for its worker, and the match names it.
        assert_eq!(bed.worded(&asked, 0, None), [(1, 0, 4)], "{fill}");
        assert_eq!(stated(&asked, 1), sent, "{fill}");
        let longer = text_of(first, 6, fill);
        let shared = holds(4, 6, OverlapSource::Predictor);
        assert_eq!(stated(&longer, 1), shared, "{fill}");
        // The match names the first worker alone, and the pick is the second.
        assert_eq!(stated(&asked, 2), miss, "{fill}");
    }
    // No entry named: the record of hashed windows, which nothing wrote.
    let hashes = worded(&text(1_000, 4));
    assert_eq!(bed.states(&request(&hashes, 7, NOW), 1), miss);
}

/// A pick that is none of the match's holders states a miss of the
/// request's blocks, whatever the match holds for another worker: the pick
/// states its own worker's entry. A gateway's classic policy observes the
/// match's share of such a request, whoever it chose
/// (`model_gateway/src/policies/cache_aware.rs:3103-3123`).
#[test]
fn a_pick_that_is_no_holder_of_a_trees_match_states_a_miss() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    let miss = holds(0, 4, OverlapSource::None);
    assert_eq!(stated_in_tree(&bed, &asked, 1), miss);
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 4)]);
    // The match names the first worker alone, and the pick is the second.
    assert_eq!(stated_in_tree(&bed, &asked, 2), miss);
    // The descent wrote the prompt for the second too: both hold it now.
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 4), (2, 0, 4)]);
    let sent = holds(4, 4, OverlapSource::Predictor);
    assert_eq!(stated_in_tree(&bed, &asked, 2), sent);
    assert_eq!(stated_in_tree(&bed, &asked, 1), sent);
}

/// A match under one block states a miss: the match is counted in the
/// request's blocks, and a match of less than one names no holder. Under a
/// tree of four-token pages, a prompt that shares its first eight tokens
/// with one the tree holds states a miss of its own four blocks. A prompt
/// shorter than one block always states a miss, even once the tree holds
/// every token of it for the worker picked. A gateway's classic policy
/// divides the tokens matched by the prompt's tokens, and observes that
/// last request whole (`model_gateway/src/policies/cache_aware.rs:3103-3123`).
#[test]
fn a_trees_match_under_one_block_states_a_miss() {
    const PAGE: usize = 4;
    let (affinity, tree) = lending_by(PAGE, false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    assert_eq!(
        stated_in_tree(&bed, &asked, 1),
        holds(0, 4, OverlapSource::None)
    );
    // Eight tokens of the prompt held, and then another prompt's.
    let mut branch = asked[..2 * PAGE].to_vec();
    branch.extend(tokens(5_000, 4).into_iter().skip(2 * PAGE));
    assert!(bed.keyed(&branch, 0, None).is_empty(), "no holder is named");
    assert_eq!(
        stated_in_tree(&bed, &branch, 1),
        holds(0, 4, OverlapSource::None)
    );
    // Three whole pages, and under one block.
    let short = &asked[..3 * PAGE];
    let none = holds(0, 1, OverlapSource::None);
    assert_eq!(stated_in_tree(&bed, short, 3), none);
    let held = tree.get_tenant_token_counts();
    assert!(held
        .get(&address(3))
        .is_some_and(|&held| held >= short.len()));
    assert_eq!(stated_in_tree(&bed, short, 3), none);
}

/// A holder the sample's draw leaves out states a miss. Eight workers hold
/// the prompt in the tree, and under a sample of four a decision is named
/// four of them. A pick of one of the four states the match's blocks. A
/// pick of one the draw left out states a miss of the request's blocks,
/// though the tree holds the prompt for it. A gateway's classic policy
/// counts every tenant of the match
/// (`model_gateway/src/policies/cache_aware.rs:3103-3123`).
#[test]
fn a_holder_the_samples_draw_leaves_out_states_a_miss() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, sampled(4), 20);
    let asked = tokens(1_000, 4);
    for at in 0..8 {
        bed.keyed(&asked, 0, Some(at));
    }
    let hashes = hashed(&asked);
    let (mut named_picks, mut left_out) = (0, 0);
    for seed in 0..32 {
        for row in 0..8 {
            let key = TokenKey {
                tokens: Cow::Borrowed(asked.as_slice()),
                marker: 0,
            };
            let mut named = Vec::new();
            let returned = decide(
                &bed.affinity,
                &bed.pool,
                &bed.spec,
                &bed.workers,
                &bed.ids,
                &mut request(&hashes, seed, NOW),
                None,
                |_| Some(key),
                || None,
                |_| true,
                |facts| {
                    let rows = facts.named.expect("rows named");
                    named = rows.holders.iter().map(|holder| holder.row).collect();
                    picked(row, bed.ids[row as usize])
                },
            );
            let stated = returned.pick().map(Pick::overlap).unwrap_or_default();
            assert_eq!(named.len(), 4, "seed {seed}: the sample");
            if named.contains(&row) {
                let sent = holds(4, 4, OverlapSource::Predictor);
                assert_eq!(stated, sent, "seed {seed}, row {row}: named");
                named_picks += 1;
            } else {
                let miss = holds(0, 4, OverlapSource::None);
                assert_eq!(stated, miss, "seed {seed}, row {row}: left out");
                left_out += 1;
            }
        }
    }
    assert!(named_picks > 0 && left_out > 0, "both kinds were picked");
}

/// A tree pick is counted by the document's block size where the request
/// states its tokens and no hash of them: the match in the request's
/// blocks, and the request itself. Sixty-four tokens are four blocks of
/// sixteen. Once the tree holds the prompt for a worker, a pick of it
/// states four of four.
#[test]
fn a_tree_pick_of_a_request_with_no_hashes_is_counted_by_the_documents_block() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    let facts = RequestFacts {
        model: MODEL,
        seed: 7,
        now_micros: NOW,
        prompt_tokens: 4 * BLOCK,
        ..RequestFacts::default()
    };
    let key = || TokenKey {
        tokens: Cow::Borrowed(asked.as_slice()),
        marker: 0,
    };
    let answer = picked(1, bed.ids[1]);
    let miss = holds(0, 4, OverlapSource::None);
    assert_eq!(bed.stated(&facts, Some(key()), answer), miss);
    let sent = holds(4, 4, OverlapSource::Predictor);
    assert_eq!(bed.stated(&facts, Some(key()), answer), sent);
}

/// A pick the binding's sessions rule placed alone and keeps from the
/// profile states nothing, over the engine's own rows and over a column its
/// host filled: the request is none of the profile's. One the rule places
/// under its default word is the profile's own, and states its worker's
/// entry as a pick by the profile's stages does. A gateway's sticky map
/// hands a request it places to no policy, so the policy observes nothing
/// of it (`model_gateway/src/policies/registry.rs:791`, `:840`).
#[test]
fn a_pick_its_rule_keeps_from_the_profile_states_nothing() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 1, &hashes_of(&asked));
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let column = [AffinityFacts {
        overlap_blocks: 4,
        ..AffinityFacts::default()
    }; 4];
    let named = request(&asked, 7, NOW);
    let fed = RequestFacts {
        affinity: &column,
        ..named
    };
    let untold = ruled(1, bed.ids[1], RulePlaced::Untold);
    let told = ruled(1, bed.ids[1], RulePlaced::Told);
    for facts in [&named, &fed] {
        assert_eq!(bed.stated(facts, None, untold), Overlap::NONE);
        assert_eq!(
            bed.stated(facts, None, told),
            holds(4, 4, OverlapSource::Index)
        );
    }
}

/// A decision whose profile reads no affinity states nothing, whatever its
/// host fed: under a document none of whose bindings reads affinity, and
/// under one where another model's profile does. The reader's pick states.
/// Nor does it state where its host lends a token tree and names the
/// request's entry: such a profile is decided inside no tree.
#[test]
fn a_profile_that_reads_no_affinity_states_nothing() {
    let asked = prompt(1_000, 4);
    let column = [AffinityFacts {
        overlap_blocks: 4,
        ..AffinityFacts::default()
    }; 4];
    let plain = request(&asked, 7, NOW);
    let fed = RequestFacts {
        affinity: &column,
        ..plain
    };
    let (affinity, _, _) = hosted(true);
    let bed = Bed::new(affinity, preset("least_load"), 4);
    assert_eq!(bed.states(&plain, 1), Overlap::NONE);
    assert_eq!(bed.states(&fed, 1), Overlap::NONE);
    let (affinity, tree) = lending(false);
    let bed = Bed::new(affinity, preset("least_load"), 4);
    let key = tokens(1_000, 4);
    assert_eq!(stated_in_tree(&bed, &key, 1), Overlap::NONE);
    assert!(
        tree.get_tenant_token_counts().is_empty(),
        "nothing is written"
    );
    // One model's profile reads affinity in the single phase. Another
    // model's reads none there.
    let mut document = PolicySpec::preset("least_load").expect("the preset");
    let reader = PolicySpec::preset("cache_aware").expect("the preset");
    let mut binding = reader.bindings[0].clone();
    binding.matcher.model = MODEL.into();
    binding.matcher.phase = vec![Phase::Single];
    document.bindings.insert(0, binding);
    document.profiles.extend(reader.profiles);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let (affinity, _, _) = hosted(true);
    let bed = Bed::new(affinity, spec, 4);
    let other = |facts: &RequestFacts<'_>| {
        bed.states(
            &RequestFacts {
                model: "model-b",
                ..*facts
            },
            1,
        )
    };
    assert_eq!(other(&plain), Overlap::NONE);
    assert_eq!(other(&fed), Overlap::NONE);
    assert_eq!(bed.states(&fed, 1), holds(4, 4, OverlapSource::Index));
}

/// A shadow that reads affinity states nothing of the live pick. Under a
/// binding whose own profile reads no affinity and whose shadow does, the
/// lookup fills the column for the shadow, and the pick the live profile
/// made says nothing of what its worker holds, over that column and over
/// one its host filled. With the two profiles the other way round the pick
/// states. The gateway's classic router observes a match ratio inside its
/// cache-aware policy alone
/// (`model_gateway/src/policies/cache_aware.rs:2686-2687`, `:3137`).
#[test]
fn a_shadow_that_reads_affinity_states_nothing_of_the_live_pick() {
    // One preset's profile live, and another's as its shadow.
    let bound = |live: &str, shadow: &str| {
        let mut document = PolicySpec::preset(live).expect("the preset");
        document.generation = 1;
        let other = PolicySpec::preset(shadow).expect("the preset");
        document.profiles.extend(other.profiles);
        document.bindings[0].shadow = Some(ShadowSpec::Profile(shadow.into()));
        AppliedSpec::validate(&document).expect("the document validates")
    };
    let asked = prompt(1_000, 4);
    let column = [AffinityFacts {
        overlap_blocks: 4,
        ..AffinityFacts::default()
    }; 4];
    let plain = request(&asked, 7, NOW);
    let fed = RequestFacts {
        affinity: &column,
        ..plain
    };
    // What the decision was given for the second worker, and what its pick
    // of that worker states.
    let decided = |bed: &Bed, facts: &RequestFacts<'_>| {
        let mut held = None;
        let returned = decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut { *facts },
            None,
            |_| None,
            || None,
            |_| true,
            |facts| {
                held = Some(facts.affinity_at(1).overlap_blocks);
                picked(1, bed.ids[1])
            },
        );
        (held, returned.pick().map(Pick::overlap).unwrap_or_default())
    };
    for (live, shadow, stated) in [
        ("least_load", "cache_aware", Overlap::NONE),
        (
            "cache_aware",
            "least_load",
            holds(4, 4, OverlapSource::Index),
        ),
    ] {
        let spec = bound(live, shadow);
        let needs = spec.engine().needs_for(MODEL, Phase::Single);
        assert!(needs.affinity, "{live} live: the binding reads affinity");
        let (affinity, _, index) = hosted(true);
        store(&index.expect("the index"), 1, &hashes_of(&asked));
        let bed = Bed::new(affinity, spec, 4);
        for facts in [&plain, &fed] {
            assert_eq!(
                decided(&bed, facts),
                (Some(4), stated),
                "{live} live, {shadow} its shadow"
            );
        }
    }
}

/// Three workers publish no blocks on their cache event streams.
/// The lightest worker takes the cold prompt. All 39 repeats stay there
/// after it becomes the heaviest worker.
#[test]
fn a_worker_that_asks_for_events_and_publishes_none_keeps_its_prompt() {
    use crate::engine::view::LoadRow;

    let (affinity, _, index) = hosted(true);
    let bed = Bed::new(affinity, preset("cache_aware"), 3);
    let index = index.expect("the cache index");
    for worker in &bed.workers {
        let number = index.intern_worker(worker.url()).expect("a worker number");
        assert_eq!(index.worker_block_count(number), 0);
    }

    let asked = prompt(1_000, 8);
    let mut rows = records(&bed.ids);
    let pool = PoolFacts {
        pool_size: 3,
        generation: 1,
        ..PoolFacts::default()
    };
    let mut scratch = Scratch::new();
    for turn in 0..40 {
        let now = NOW + turn * 20_000;
        let loads = if turn == 0 { [0, 1, 1] } else { [2, 0, 0] };
        for (row, running) in rows.iter_mut().zip(loads) {
            row.load = Some(LoadRow {
                running,
                ..LoadRow::default()
            });
            row.load_sampled_at_micros = now;
        }

        let decision = decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&asked, 7, now),
            None,
            |_| None,
            || None,
            |_| true,
            |facts| {
                bed.spec
                    .engine()
                    .decide(facts, &pool, &rows, &Constraints::default(), &mut scratch)
            },
        );
        let pick = decision.pick().expect("a pick");
        let target = pick.targets.first().expect("a target");
        assert_eq!(target.id, bed.ids[0], "turn {turn}");
    }
}

/// A pick records the eight-block prompt. A same-tick repeat preserves
/// its place in eviction order. A repeat one tick later renews its last use.
#[test]
fn a_request_sent_a_moment_ago_writes_nothing() {
    let asked = prompt(1_000, 8);
    let marker = prompt(2_000, 1);
    let first = request(&asked, 7, NOW);

    for (last, depth_after_eviction) in [(NOW, 7), (NOW + (1 << 20), 8)] {
        let (affinity, _, _) = hosted(false);
        let bed = Bed::new(affinity, preset("cache_aware"), 1);
        bed.decide(&first, Some(0));
        assert_eq!(bed.holders(&asked, NOW), [(0, 0, 8)]);

        bed.decide(&request(&marker, 7, NOW), Some(0));
        bed.decide(&first, Some(0));
        if last != NOW {
            bed.decide(&request(&asked, 7, last), Some(0));
        }

        // The older of the two ends loses one block.
        let record = bed.affinity.record_of(MODEL, 8);
        record.predictor.write().sweep(1);
        assert_eq!(
            bed.holders(&asked, last),
            [(0, 0, depth_after_eviction)],
            "prompt depth after one eviction at {last} microseconds"
        );
    }
}

/// Full write slices contain sixteen blocks. The last slice can be shorter.
/// Same-tick hits at the slice ends preserve their eviction order.
#[test]
fn a_long_request_is_written_in_slices_of_sixteen() {
    let cases: [(u32, u32, &[u32]); 3] = [
        (1_000, 40, &[16, 32, 40]),
        (2_000, 32, &[16, 32]),
        (3_000, 33, &[16, 32, 33]),
    ];
    let (affinity, _, _) = hosted(false);
    let previous = Bed::new(affinity, preset("cache_aware"), 2);

    // Give every block an older tick on a worker outside the next list.
    for &(first, blocks, _) in &cases {
        let asked = prompt(first, blocks);
        previous.decide(&request(&asked, 7, NOW - (1 << 20)), Some(1));
    }
    let bed = Bed::new(previous.affinity, previous.spec, 1);

    for (first, blocks, expected_ends) in cases {
        let asked = prompt(first, blocks);
        let facts = request(&asked, 7, NOW);
        let cold = bed.decide(&facts, Some(0)).expect("named rows");
        assert!(cold.holders.is_empty(), "cold prompt of {blocks} blocks");

        // Only the ends of writes receive the new tick on existing blocks.
        let mut ends = Vec::new();
        for end in 1..=blocks {
            let prefix = RequestFacts {
                block_hashes: &facts.block_hashes[..end as usize],
                prompt_tokens: end * BLOCK,
                ..facts
            };
            let (_, read) = bed.column(&prefix, Phase::Single);
            let read = read.expect("the prefix record").read;
            assert_eq!(read.blocks, end, "recorded depth for prefix {end}");
            if read.fresh {
                ends.push(end);
            }
        }
        assert_eq!(ends.as_slice(), expected_ends, "{blocks} cold blocks");
    }

    let marker = prompt(9_000, 1);
    bed.decide(&request(&marker, 7, NOW), Some(0));
    for end in [40, 16, 32] {
        let prefix = prompt(1_000, end);
        let named = bed
            .decide(&request(&prefix, 7, NOW), Some(0))
            .expect("named rows");
        assert_eq!(named.holders, [(0, 0, end)]);
    }

    // A write at any repeated end puts it after the marker.
    let record = bed.affinity.record_of(MODEL, 1);
    record.predictor.write().sweep(usize::MAX);
    assert_eq!(
        bed.holders(&marker, NOW),
        [(0, 0, 1)],
        "the marker stays newest"
    );
}

/// A repeat names the picked worker at a predicted depth of twelve blocks.
/// Changing only the namespace seed leaves the holder list empty.
#[test]
fn a_namespace_keeps_its_prefixes_apart() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 1);
    let asked = prompt(1_000, 12);
    let mut facts = request(&asked, 7, NOW);

    bed.decide(&facts, Some(0));
    assert_eq!(bed.holders(&asked, NOW), [(0, 0, 12)]);

    facts.namespace_seed ^= 1;
    let named = bed.decide(&facts, None).expect("rows named");
    assert!(named.holders.is_empty());
}

/// A pick with no hashes leaves the prefix record empty.
/// A pick for twelve text windows records all twelve for its worker.
#[test]
fn a_text_request_is_recorded_and_absent_hashes_are_not() {
    use crate::engine::view::TEXT_BYTES_PER_TOKEN;

    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 1);
    let empty = PromptHashes::default();
    bed.decide(&request(&empty, 7, NOW), Some(0));

    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };
    assert!(record.predictor.read().is_empty());

    let window = (BLOCK * TEXT_BYTES_PER_TOKEN) as usize;
    let text = "x".repeat(12 * window);
    let asked =
        PromptHashes::under_document(&bed.spec, Some(BLOCK), None, Some(&text), HashSeeds::PLAIN);
    let facts = request(&asked, 7, NOW);
    assert_eq!(
        (facts.hash_kind, facts.block_hashes.len()),
        (HashKind::Text, 12)
    );

    bed.decide(&facts, Some(0));
    assert_eq!(
        record.predictor.read().len(),
        12,
        "one placement per window"
    );
    assert_eq!(bed.holders(&asked, NOW), [(0, 0, 12)]);
}

/// A shed leaves the first prompt's twelve prefix keys in the buffer.
/// An unhashed least-load pick for another model leaves the first record empty.
/// A repeat of the first prompt still names no holder.
#[test]
fn a_decision_that_names_no_row_writes_nothing_of_the_one_before() {
    let mut document = PolicySpec::preset("least_load").expect("least-load preset");
    let reader = PolicySpec::preset("cache_aware").expect("cache-aware preset");
    let mut binding = reader.bindings[0].clone();
    binding.matcher.model = MODEL.into();
    document.bindings.insert(0, binding);
    document.profiles.extend(reader.profiles);
    let spec = AppliedSpec::validate(&document).expect("both bindings validate");

    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, spec, 1);
    let asked = prompt(1_000, 12);
    assert!(bed.holders(&asked, NOW).is_empty());
    assert_eq!(
        with_buffers(|buffers| buffers.keys.len()),
        12,
        "the first lookup leaves its prefix keys"
    );

    let empty = PromptHashes::default();
    let mut other = request(&empty, 7, NOW);
    other.model = "model-b";
    let rows = records(&bed.ids);
    let pool = PoolFacts {
        pool_size: 1,
        generation: 1,
        ..PoolFacts::default()
    };
    let decision = decide(
        &bed.affinity,
        &bed.pool,
        &bed.spec,
        &bed.workers,
        &bed.ids,
        &mut other,
        None,
        |_| None,
        || None,
        |_| true,
        |facts| {
            assert!(facts.named.is_none(), "least load receives no named rows");
            bed.spec.engine().decide(
                facts,
                &pool,
                &rows,
                &Constraints::default(),
                &mut Scratch::new(),
            )
        },
    );
    let pick = decision.pick().expect("least-load pick");
    assert_eq!(pick.targets.first().expect("first target").id, bed.ids[0]);

    {
        let records = bed.affinity.records.lock();
        let record = records.get(MODEL).expect("the first model's record");
        assert!(record.predictor.read().is_empty());
    }
    assert!(bed.holders(&asked, NOW).is_empty());
}

/// Under a source that reads the record alone the record's holders are
/// named first, and where they fill the profile's sample the index is not
/// asked. The decision is named the worker the prefix was sent to, though
/// the index holds the whole request for another, and it still goes to
/// that other worker where the filters refuse the named one.
#[test]
fn a_cap_the_records_holders_fill_is_not_read_past() {
    use crate::document::spec::{AffinitySource, BestParams, PickStage, ScoreSpec};

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the profile");
    profile.pick = PickStage::Best(BestParams { sample: 1 }).into();
    for stage in &mut profile.score {
        if let ScoreSpec::Params(ScoreStage::PrefixAffinity(prefix)) = stage {
            prefix.source = Some(AffinitySource::Predictor);
        }
    }
    let spec = AppliedSpec::validate(&document).expect("the document validates");

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 0, &hashes_of(&asked));
    let bed = Bed::new(affinity, spec, 2);
    let mut facts = request(&asked, 0, NOW);
    let first = bed.decide(&facts, Some(1)).expect("rows");
    assert_eq!(first.holders, [(0, 4, 0)]);

    let mut rows = records(&bed.ids);
    rows[1].health.healthy = false;
    let pool = PoolFacts {
        pool_size: 2,
        generation: 1,
        ..PoolFacts::default()
    };
    let decision = decide(
        &bed.affinity,
        &bed.pool,
        &bed.spec,
        &bed.workers,
        &bed.ids,
        &mut facts,
        None,
        |_| None,
        || None,
        |_| true,
        |facts| {
            assert_eq!(
                facts.named.expect("rows").holders,
                &[AffinityHolder {
                    row: 1,
                    overlap_blocks: 0,
                    predicted_blocks: 4,
                }]
            );
            bed.spec.engine().decide(
                facts,
                &pool,
                &rows,
                &Constraints::default(),
                &mut Scratch::new(),
            )
        },
    );
    let pick = decision.pick().expect("a healthy worker");
    assert_eq!(pick.targets.first().expect("a target").id, bed.ids[0]);

    // The index was not asked again: a lookup empties its answer first,
    // even under a cap of zero, and the first decision's answer stands.
    with_buffers(|buffers| {
        assert_eq!(buffers.kept.len(), 1, "the first index answer survives");
    });
}

/// Under a source that reads the record alone the record's holders are
/// named ahead of the index's, and the index is asked in the room they
/// leave. Where the record names fewer holders than the profile's sample
/// and the index holds the request for another worker, each holder stands
/// in the naming once: the record is not read again behind the index.
#[test]
fn a_record_named_ahead_is_not_named_again_behind_the_index() {
    use crate::document::spec::{AffinitySource, BestParams, PickStage, ScoreSpec};

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the profile");
    profile.pick = PickStage::Best(BestParams { sample: 3 }).into();
    for stage in &mut profile.score {
        if let ScoreSpec::Params(ScoreStage::PrefixAffinity(prefix)) = stage {
            prefix.source = Some(AffinitySource::Predictor);
        }
    }
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    assert_eq!(spec.engine().affinity_rows_cap(MODEL, Phase::Single), 3);

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 0, &hashes_of(&asked));
    let bed = Bed::new(affinity, spec, 3);
    // The pick writes the record: the prefix was sent to the second worker.
    let first = bed.decide(&request(&asked, 0, NOW), Some(1)).expect("rows");
    assert_eq!(first.holders, [(0, 4, 0)]);
    // One holder of the record and one of the index, under a sample of
    // three: the room left is not filled by the record a second time.
    assert_eq!(bed.holders(&asked, NOW), [(0, 4, 0), (1, 0, 4)]);
}

/// A naming is made once a decision, ahead of its filters. The holders the
/// record names stand in it by row, each with its depth, and they stand
/// whole where the filters refuse every worker. The model has no index: a
/// single request of token ids beside one is named by the index alone.
#[test]
fn a_naming_the_record_fills_stands_whole_where_every_holder_is_refused() {
    use crate::document::spec::{BestParams, PickStage};

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the profile");
    profile.pick = PickStage::Best(BestParams { sample: 2 }).into();
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    assert_eq!(spec.engine().affinity_rows_cap(MODEL, Phase::Single), 2);

    let (affinity, _, _) = hosted(false);
    let asked = prompt(1_000, 4);
    let bed = Bed::new(affinity, spec, 2);
    for row in 0..2 {
        bed.decide(&request(&asked, 0, NOW), Some(row))
            .expect("rows");
    }

    let pool = PoolFacts {
        pool_size: 2,
        generation: 1,
        ..PoolFacts::default()
    };
    let named_when = |healthy| {
        let mut rows = records(&bed.ids);
        for row in &mut rows {
            row.health.healthy = healthy;
        }
        let mut holders = Vec::new();
        let decision = decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&asked, 0, NOW),
            None,
            |_| None,
            || None,
            |_| true,
            |facts| {
                for holder in facts.named.expect("rows").holders {
                    holders.push((holder.row, holder.overlap_blocks, holder.predicted_blocks));
                }
                bed.spec.engine().decide(
                    facts,
                    &pool,
                    &rows,
                    &Constraints::default(),
                    &mut Scratch::new(),
                )
            },
        );
        assert_eq!(
            decision.pick().is_some(),
            healthy,
            "only healthy workers pass"
        );
        holders
    };

    let first = named_when(true);
    assert_eq!(first, [(0, 0, 4), (1, 0, 4)]);
    assert_eq!(named_when(false), first);
}

/// A list whose every worker the index holds a block for still reads the
/// record for a request hashed by its text, and writes its pick there: such
/// a request asks the index nothing. The classic router reads its tree of
/// sent texts for a request with no token ids, however many workers its
/// index of cache events names
/// (`model_gateway/src/policies/cache_aware.rs:1511-1538`, `:3519-3596`). A
/// request hashed by token blocks reads no record over the same list and
/// writes none, as before.
#[test]
fn a_pool_whose_workers_are_all_heard_reads_the_record_for_a_request_of_text() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let other = hashes_of(&prompt(9_000, 1));
    store(&index, 0, &other);
    store(&index, 1, &other);
    let bed = Bed::new(affinity, preset("cache_aware"), 2);
    let said = text(1_000, 4);
    let asked = worded(&said);
    let tokens = prompt(1_000, 4);

    let named = bed.decide(&request(&asked, 7, NOW), Some(1));
    assert!(
        named.expect("rows").holders.is_empty(),
        "nobody was sent the text before this pick"
    );
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4)]);
    bed.decide(&request(&tokens, 7, NOW), Some(1));
    assert!(
        bed.holders(&tokens, NOW).is_empty(),
        "a token request's pick writes nothing"
    );

    let standing = bed.pool.table.load();
    let table = standing.as_deref().expect("the list's table");
    assert!(
        table.heard_any && !table.unheard_any,
        "the index holds a block for every worker"
    );
    let record = table.record.as_ref().expect("the model's record");
    assert_eq!(record.predictor.read().len(), 4, "the text's four windows");
    let reads = |prompt: &PromptHashes| {
        let mut buffers = Buffers::new();
        let read = table.name(&request(prompt, 7, NOW), None, &mut buffers);
        (read.is_some(), buffers.keys.len())
    };
    assert_eq!(reads(&asked), (true, 4), "a request of text reads it");
    assert_eq!(reads(&tokens), (false, 0), "a token request reads none");
}

/// A single request of token ids under an index that holds a block reads
/// no record and writes none, whoever of the list the index speaks for: a
/// naming folds no key and a pick leaves no entry. Once the index holds no
/// block its host hands none, and from the table's next read the record is
/// read and written again.
#[test]
fn a_list_under_an_index_that_holds_a_block_reads_no_record() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let bed = Bed::new(affinity, preset("cache_aware"), 2);
    let asked = prompt(1_000, 4);
    let other = hashes_of(&prompt(9_000, 1))[0];

    // Each worker holds one block of another prompt. Its block map is kept
    // for its clear.
    let mut stored = [0, 1].map(|at| {
        let number = index.intern_worker(&address(at)).expect("a worker number");
        let mut map = ChainBlockMap::new();
        let block = StoredBlock {
            seq_hash: other.into(),
            content_hash: ContentHash(other),
        };
        index
            .apply_stored(number, &[block], None, &mut map)
            .expect("a stored block");
        (number, map)
    });

    let check = |now, reads| {
        let facts = request(&asked, 7, now);
        let named = bed.decide(&facts, Some(0)).expect("rows");
        assert!(
            named.holders.is_empty(),
            "no worker holds the prefix before this pick"
        );

        let standing = bed.pool.table.load();
        let table = standing.as_deref().expect("the list's table");
        let record = table.record.as_ref().expect("the model's record");
        let placements = record.predictor.read().len();

        let mut buffers = Buffers::new();
        let read = table.name(&facts, None, &mut buffers);
        assert_eq!(read.is_some(), reads, "record read at {now}");
        let blocks = if reads { 4 } else { 0 };
        assert_eq!(buffers.keys.len(), blocks, "folded keys at {now}");
        assert_eq!(placements, blocks, "placements after the pick at {now}");
    };

    check(NOW, false);
    // The first worker's block leaves: the index speaks for the second
    // alone, and still decides alone.
    let (number, map) = &mut stored[0];
    index.apply_cleared(*number, map);
    check(NOW + SECOND, false);
    // The last block leaves. The table that stands was read beside the
    // index, the next one is handed none.
    let (number, map) = &mut stored[1];
    index.apply_cleared(*number, map);
    check(NOW + 2 * SECOND - 1, false);
    check(NOW + 2 * SECOND, true);
}

/// A worker the index numbers and holds no block for is the record's to
/// speak for, where the record is asked for a request of token ids: here
/// for a side of a pair, which keeps the record beside an index. An empty
/// write of the worker's events leaves its entry of the side's column
/// holding the prefix it was sent. Once the index holds a block for it the
/// index speaks for it, and the record names it no more.
#[test]
fn an_empty_index_write_keeps_the_predicted_worker_named() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let prefills = Bed::new(affinity, paired(), 2);
    // The index holds a block throughout, the second worker's, as an
    // index its host lends does.
    let other = prompt(9_000, 1);
    store(&index, 1, &hashes_of(&other));
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 7, NOW);
    let (_, read) = prefills.column(&facts, Phase::Prefill);
    let read = read.expect("the record speaks for the first prefill");
    record_prefix(
        &prefills.pool,
        Phase::Prefill,
        &facts,
        read,
        prefills.ids[0],
    );

    store(&index, 0, &[]);
    let later = request(&asked, 7, NOW + SECOND);
    let (column_of, _) = prefills.column(&later, Phase::Prefill);
    assert_eq!(held(&column_of), [(0, 4), (0, 0)]);

    store(&index, 0, &hashes_of(&other));
    let later = request(&asked, 7, NOW + 2 * SECOND);
    let (column_of, _) = prefills.column(&later, Phase::Prefill);
    assert_eq!(
        held(&column_of),
        [(0, 0); 2],
        "the index speaks for the worker, and holds no block of the request"
    );
}

/// A worker the record names until the index holds blocks for it is named
/// by the index alone from the next read of the table: its entry in the
/// record names no second row.
#[test]
fn a_stale_prediction_cannot_name_a_worker_the_index_already_names() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let bed = Bed::new(affinity, preset("cache_aware"), 2);
    let asked = prompt(1_000, 4);

    bed.decide(&request(&asked, 7, NOW), Some(1))
        .expect("rows named");
    assert_eq!(bed.holders(&asked, NOW), [(1, 0, 4)]);

    store(&index, 1, &hashes_of(&asked));
    assert_eq!(
        bed.holders(&asked, NOW + SECOND),
        [(1, 4, 0)],
        "only the index names row 1 after the refresh"
    );
}

/// A list of more workers than one word of a mask holds is read whole.
/// Under no sample each of seventy workers is named by the record, and by
/// the index alone once its events arrived and the table is read again.
#[test]
fn a_pool_past_one_word_of_workers_is_read_whole() {
    const WORKERS: usize = 70;

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the profile");
    profile.pick = "best".into();
    let spec = AppliedSpec::validate(&document).expect("the document validates");

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    for at in 0..WORKERS {
        index.intern_worker(&address(at)).expect("a worker number");
    }
    let bed = Bed::new(affinity, spec, WORKERS);
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 7, NOW);
    for at in 0..WORKERS {
        bed.decide(&facts, Some(at as u32)).expect("rows");
    }

    let predicted: Vec<_> = (0..WORKERS).map(|at| (at as u32, 0, 4)).collect();
    assert_eq!(bed.holders(&asked, NOW), predicted);

    let hashes = hashes_of(&asked);
    for at in 0..WORKERS {
        store(&index, at, &hashes);
    }
    assert_eq!(
        bed.holders(&asked, NOW),
        predicted,
        "the standing table still names the predictions"
    );
    let indexed: Vec<_> = (0..WORKERS).map(|at| (at as u32, 4, 0)).collect();
    assert_eq!(bed.holders(&asked, NOW + SECOND), indexed);
}

/// A list is named the deepest holder among its own workers: a worker of
/// another list that was sent more of the request hides no shorter prefix
/// a worker of this list was sent.
#[test]
fn a_worker_outside_the_list_hides_no_shorter_prefix_inside_it() {
    let (affinity, _, _) = hosted(false);
    let both = Bed::new(affinity, preset("cache_aware"), 2);
    let four = prompt(1_000, 4);
    let three = prompt(1_000, 3);
    both.decide(&request(&four, 7, NOW), Some(0));
    both.decide(&request(&three, 7, NOW), Some(1));
    assert_eq!(both.holders(&four, NOW), [(0, 0, 4)], "the deeper holder");

    // The second worker alone, as another list of the model: the model's
    // one record holds what both workers were sent.
    let alone = Bed {
        workers: both.workers[1..].to_vec(),
        ids: both.ids[1..].to_vec(),
        pool: PoolAffinity::default(),
        ..both
    };
    assert_eq!(
        alone.holders(&four, NOW),
        [(0, 0, 3)],
        "its own three blocks"
    );
}

/// A profile whose prefix stage reads the record alone is named the
/// record's holders ahead of the index's: at a sample of one the worker
/// the prefix was sent to, and at two the index's holder beside it.
#[test]
fn a_source_that_reads_the_record_alone_names_its_holders_first() {
    use crate::document::spec::{AffinitySource, BestParams, PickStage, ScoreSpec};

    let spec = |sample| {
        let mut document = PolicySpec::preset("cache_aware").expect("the preset");
        let profile = document
            .profiles
            .get_mut("cache_aware")
            .expect("the preset's profile");
        for stage in &mut profile.score {
            if let ScoreSpec::Params(ScoreStage::PrefixAffinity(params)) = stage {
                params.source = Some(AffinitySource::Predictor);
            }
        }
        profile.pick = PickStage::Best(BestParams { sample }).into();
        AppliedSpec::validate(&document).expect("the document validates")
    };

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 0, &hashes_of(&asked));
    let facts = request(&asked, 0, NOW);

    let one = Bed::new(affinity, spec(1), 2);
    one.decide(&facts, Some(1));
    let named = one.decide(&facts, None).expect("rows at cap one");
    assert_eq!(
        named.holders,
        [(1, 0, 4)],
        "the recorded worker takes the slot"
    );

    let two = Bed {
        pool: PoolAffinity::default(),
        spec: spec(2),
        ..one
    };
    let named = two.decide(&facts, None).expect("rows at cap two");
    assert_eq!(named.holders, [(0, 4, 0), (1, 0, 4)]);
}

/// A removal takes a record's lock as a writer only where the record holds
/// the worker: it ends while a decision of another model holds that
/// model's record as a reader, and the worker's placement is gone.
#[test]
fn a_removal_waits_for_no_decider_of_another_model() {
    const READER_MODEL: &str = "model-b";

    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 1);
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 7, NOW);
    bed.decide(&facts, Some(0));
    assert_eq!(bed.holders(&asked, NOW), [(0, 0, 4)]);

    let reader_id = WorkerId(2);
    let reader_workers = [ready_worker(&address(1), READER_MODEL)];
    let reader_pool = PoolAffinity::default();
    decide(
        &bed.affinity,
        &reader_pool,
        &bed.spec,
        &reader_workers,
        &[reader_id],
        &mut RequestFacts {
            model: READER_MODEL,
            ..facts
        },
        None,
        |_| None,
        || None,
        |_| true,
        |_| picked(0, reader_id),
    );

    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(READER_MODEL).expect("the reader's record"))
    };
    let reading = record.predictor.read();
    assert!(!reading.holds(bed.ids[0]), "a worker of the other model");
    // The removal runs on a thread of its own: one that waited for the
    // reader is seen waiting, five seconds at most, and hangs no case.
    let (affinity, gone) = (&bed.affinity, bed.ids[0]);
    let ended = thread::scope(|scope| {
        let removal = scope.spawn(move || affinity.forget(gone));
        for _ in 0..5_000 {
            if removal.is_finished() {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let ended = removal.is_finished();
        assert!(reading.holds(reader_id), "the reader holds its record");
        drop(reading);
        ended
    });
    assert!(ended, "the removal waited for a reader of another model");
    assert!(bed.holders(&asked, NOW).is_empty(), "the placement is gone");
}

/// The cache-aware preset with a second prefix stage, which reads the
/// record alone, under a pick of the best of one: a profile one stage of
/// which reads cache events and another does not.
fn mixed() -> AppliedSpec {
    use crate::document::spec::{AffinitySource, BestParams, PickStage, PrefixAffinityParams};

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the preset profile");
    profile.pick = PickStage::Best(BestParams { sample: 1 }).into();
    profile.score.push(
        ScoreStage::PrefixAffinity(PrefixAffinityParams {
            weight: 0.01,
            source: Some(AffinitySource::Predictor),
            ..PrefixAffinityParams::default()
        })
        .into(),
    );
    let spec = AppliedSpec::validate(&document).expect("valid mixed profile");
    assert_eq!(spec.engine().affinity_rows_cap(MODEL, Phase::Single), 1);
    spec
}

/// One prefix stage that reads cache events has the index decide a single
/// request of token ids alone, whatever another reads. Beside a stage
/// that reads the record alone, the record is neither read nor written:
/// the worker the index names keeps a sample of one after the request
/// went to another worker, whose column entry holds no prefix.
#[test]
fn a_stage_that_reads_cache_events_keeps_the_index_holders_first() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the host index");
    let asked = prompt(1_000, 4);
    store(&index, 1, &hashes_of(&asked));
    let bed = Bed::new(affinity, mixed(), 2);
    let facts = request(&asked, 0, NOW);

    let cold = bed.decide(&facts, Some(0)).expect("named rows");
    assert_eq!(cold.holders, [(1, 4, 0)]);
    let (column, _) = bed.column(&facts, Phase::Single);
    assert_eq!(
        held(&column),
        [(0, 0), (4, 0)],
        "row 0 was sent the request, and no record holds it"
    );
    let warm = bed.decide(&facts, None).expect("named rows");
    assert_eq!(warm.holders, [(1, 4, 0)], "row 1 keeps the only slot");
}

/// Under the same profile a request of text reads no index. The stage that
/// reads the record alone is named what the record holds: the record takes
/// the request's pick, and the next request is named its worker.
///
/// The classic policy sends a request of text to its string tree, whatever
/// its index holds (`select_worker_scanned` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_mixed_profiles_request_of_text_reads_and_writes_the_record() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the host index");
    store(&index, 1, &hashes_of(&prompt(1_000, 4)));
    let bed = Bed::new(affinity, mixed(), 2);
    let words = worded(&text(1_000, 4));
    let facts = request(&words, 0, NOW);

    let cold = bed.decide(&facts, Some(0)).expect("named rows");
    assert!(cold.holders.is_empty(), "the index holds no text");
    let (column, _) = bed.column(&facts, Phase::Single);
    assert_eq!(held(&column), [(0, 4), (0, 0)], "the record holds the pick");
    let warm = bed.decide(&facts, None).expect("named rows");
    assert_eq!(warm.holders, [(0, 0, 4)]);
}

/// Nor does the index decide alone while it holds no block: its host hands
/// no index then, and the record takes a request of token ids as it does
/// under a profile whose every prefix stage reads cache events.
///
/// The classic policy reads its token tree while its index is empty
/// (`event_index_for` in `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_mixed_profiles_request_under_an_empty_index_reads_and_writes_the_record() {
    let (affinity, _, _) = hosted(true);
    let bed = Bed::new(affinity, mixed(), 2);
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 0, NOW);

    let cold = bed.decide(&facts, Some(0)).expect("named rows");
    assert!(cold.holders.is_empty(), "nothing is sent yet");
    let (column, _) = bed.column(&facts, Phase::Single);
    assert_eq!(held(&column), [(0, 4), (0, 0)], "the record holds the pick");
    let warm = bed.decide(&facts, None).expect("named rows");
    assert_eq!(warm.holders, [(0, 0, 4)]);
}

/// A pick of a worker the record holds no part of the request for is
/// written, though the record holds the whole request for another worker
/// at this tick: both then hold it.
#[test]
fn a_fresh_prefix_still_writes_for_a_new_target() {
    use crate::document::spec::{BestParams, PickStage};

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the preset profile");
    profile.pick = PickStage::Best(BestParams { sample: 1 }).into();
    let spec = AppliedSpec::validate(&document).expect("valid profile");

    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, spec, 2);
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 7, NOW);
    let cold = bed.decide(&facts, Some(0)).expect("named rows");
    assert!(cold.holders.is_empty());
    let warm = bed.decide(&facts, Some(1)).expect("named rows");
    assert_eq!(warm.holders, [(0, 0, 4)]);
    assert_eq!(
        bed.holders(&asked, NOW).len(),
        1,
        "only one holder is named"
    );

    let standing = bed.pool.table.load();
    let table = standing.as_deref().expect("the list table");
    let record = table.record.as_deref().expect("the prefix record");
    let mut keys = Vec::new();
    let mut holders = Vec::new();
    let (blocks, _) = table.read_record(record, &facts, table.alone, &mut keys, &mut holders);
    assert_eq!(blocks, 4);
    // The record names a block's holders in no order.
    holders.sort_unstable();
    assert_eq!(holders, bed.ids, "both workers hold the fourth block");
}

/// A pick of a worker that holds the whole request at this tick is a
/// repeat and writes nothing, though the sample left that worker unnamed.
/// A write would have moved the prompt's last use: the next block to leave
/// a full record is the prompt's own last one.
#[test]
fn a_fresh_target_outside_the_holder_cap_writes_nothing() {
    use crate::document::spec::{BestParams, PickStage};

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    let profile = document
        .profiles
        .get_mut("cache_aware")
        .expect("the preset profile");
    profile.pick = PickStage::Best(BestParams { sample: 1 }).into();
    // Four blocks per worker and one other block fill the record.
    document.index.entry(MODEL.into()).or_default().max_size = 9 * u64::from(BLOCK);
    let spec = AppliedSpec::validate(&document).expect("valid bounded profile");

    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, spec, 2);
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 7, NOW);
    for row in [0, 1] {
        bed.decide(&facts, Some(row)).expect("named rows");
    }
    let (before, _) = bed.column(&facts, Phase::Single);
    assert_eq!(held(&before), [(0, 4); 2]);

    let newer = prompt(9_000, 1);
    bed.decide(&request(&newer, 7, NOW), Some(0))
        .expect("named rows");
    let named = bed.holders(&asked, NOW);
    assert_eq!(named.len(), 1);
    let target = if named[0].0 == 0 { 1 } else { 0 };
    let repeat = bed.decide(&facts, Some(target)).expect("named rows");
    assert_eq!(
        repeat.holders,
        [(1 - target, 0, 4)],
        "the picked holder is omitted"
    );

    let incoming = prompt(50_000, 1);
    bed.decide(&request(&incoming, 7, NOW), Some(0))
        .expect("named rows");
    let (after, _) = bed.column(&facts, Phase::Single);
    assert_eq!(
        held(&after),
        [(0, 3); 2],
        "a repeat write would keep the fourth block"
    );
    assert_eq!(bed.holders(&newer, NOW), [(0, 0, 1)]);
}

/// A bound that came down forgets nothing by itself. The write that ends
/// behind it sweeps one step (`PrefixRecord::SWEEP` placements): of a
/// hundred placements at most the others stay.
#[test]
fn the_last_slice_sweeps_placements_owed_by_a_lower_bound() {
    let (affinity, _, _) = hosted(false);
    let mut bed = Bed::new(affinity, preset("cache_aware"), 1);
    let first = prompt(1_000, 100);
    bed.decide(&request(&first, 7, NOW), Some(0))
        .expect("rows for the first pick");
    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };

    let mut smaller = PolicySpec::preset("cache_aware").expect("the preset");
    smaller.generation = 2;
    smaller.index.entry(MODEL.into()).or_default().max_size = u64::from(BLOCK);
    bed.spec = AppliedSpec::validate(&smaller).expect("the smaller document");

    let cold = prompt(50_000, 4);
    let now = NOW + SECOND;
    assert!(bed.holders(&cold, now).is_empty(), "the new prompt is cold");
    assert_eq!(record.predictor.read().bound(), 1);
    assert_eq!(
        record.predictor.read().len(),
        100,
        "the sweep waits for the write"
    );

    bed.decide(&request(&cold, 7, now), Some(0))
        .expect("rows for the cold pick");
    let left = record.predictor.read().len();
    assert!(
        left <= 100 - PrefixRecord::SWEEP,
        "the cold write leaves {left} placements"
    );
}

/// The record's bound is the index's size in blocks of the size the
/// indexes report for the model. A size twice the document's halves the
/// bound at the next read of the table, and with none reported the
/// document's size serves again.
#[test]
fn a_block_size_of_record_moves_the_records_bound() {
    struct ReportedSize(Mutex<Option<u32>>);

    impl CacheIndexes for ReportedSize {
        fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
            None
        }

        fn block_size(&self, _: &str) -> Option<u32> {
            *self.0.lock()
        }
    }

    let reported = Arc::new(ReportedSize(Mutex::new(None)));
    let bed = Bed::new(Affinity::default(), preset("cache_aware"), 1);
    bed.affinity
        .set_indexes(Arc::clone(&reported) as Arc<dyn CacheIndexes>);

    let empty = PromptHashes::default();
    bed.holders(&empty, NOW);
    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };
    let bound = || record.predictor.read().bound();
    let original = bound();
    assert!(
        original > 1,
        "the starting bound allows at least two placements"
    );

    *reported.0.lock() = Some(2 * bed.spec.block_size());
    bed.holders(&empty, NOW + SECOND);
    assert_eq!(bound(), original / 2, "the reported size halves the bound");

    *reported.0.lock() = None;
    bed.holders(&empty, NOW + 2 * SECOND);
    assert_eq!(bound(), original, "the document size restores the bound");
}

/// A bound that comes down while a decision holds the record keeps its
/// sweep. The holder's own sweep finds nothing owed under the bound the
/// record still reads, and the work waits for the next writer all the
/// same.
#[test]
fn a_lower_bound_keeps_its_sweep_past_a_decider_that_holds_the_record() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 1);
    let first = prompt(1_000, 100);
    bed.decide(&request(&first, 7, NOW), Some(0))
        .expect("rows for the first pick");
    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };
    assert!(!record.sweeping.load(Ordering::Relaxed), "nothing is owed");
    let was = record.bound.load(Ordering::Relaxed);

    // A decision holds the record while another thread lowers the bound,
    // and sweeps once that thread waits for it.
    let mut held = record.predictor.write();
    assert_eq!(held.len(), 100, "the record holds the prompt whole");
    thread::scope(|scope| {
        scope.spawn(|| record.bound_to(1));
        // Five seconds at most for the bound's word to move: a lowering
        // that never moves it fails the case below and hangs nothing.
        for _ in 0..5_000 {
            if record.bound.load(Ordering::Relaxed) != was {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        // Time for a step the lowering must not take ahead of the guard:
        // a lowering that takes it later than this passes here unseen.
        thread::sleep(Duration::from_millis(20));
        record.sweep(&mut held);
        drop(held);
    });
    assert_eq!(record.bound.load(Ordering::Relaxed), 1, "the bound's word");
    assert_eq!(record.predictor.read().bound(), 1, "the lower bound");
    assert!(
        record.sweeping.load(Ordering::Relaxed),
        "the lower bound's sweep is still owed"
    );
    bed.affinity.sweep();
    let left = record.predictor.read().len();
    assert!(
        left <= 100 - PrefixRecord::SWEEP,
        "one step of the sweep leaves {left} placements"
    );
}

/// A mask that names `numbers`, a bit a number.
fn naming(numbers: &[u32]) -> Vec<u64> {
    let mut mask = vec![0u64; 2];
    for &number in numbers {
        set(&mut mask, number as usize);
    }
    mask
}

/// The blocks of `hashes` as a worker's main store names them to its view:
/// the worker's own hash of a block, here the block's prefix hash, and the
/// block's content hash.
fn named(hashes: &[u64]) -> Vec<(SequenceHash, ContentHash)> {
    let contents: Vec<ContentHash> = hashes.iter().map(|&hash| ContentHash(hash)).collect();
    let prefixes = request_prefix_hashes(&contents);
    prefixes.into_iter().zip(contents).collect()
}

/// The numbers the index gave the first `count` workers.
fn numbers(index: &ChainIndex, count: usize) -> Vec<u32> {
    let number = |at| index.worker_id(&address(at)).expect("a number");
    (0..count).map(number).collect()
}

/// On a fleet none of whose workers keeps cache groups, the lookup within
/// groups is the chain index's own capped lookup, holder for holder, under
/// every mask, cap and draw. Holders a host scored itself are kept by the
/// same rule: as many, of the same depths.
#[test]
fn the_lookup_within_groups_is_the_plain_lookup_where_no_worker_keeps_groups() {
    let index = ChainIndex::new();
    let request = [1u64, 2, 3];
    // Three blocks, two, two and one.
    for (at, depth) in [3usize, 2, 2, 1].into_iter().enumerate() {
        store(&index, at, &request[..depth]);
    }
    let all = numbers(&index, 4);
    let groups = GroupViews::new();
    let contents: Vec<ContentHash> = request.iter().map(|&hash| ContentHash(hash)).collect();
    let depths = |kept: &[(u32, u32)]| {
        let mut depths: Vec<u32> = kept.iter().map(|&(_, depth)| depth).collect();
        depths.sort_unstable();
        depths
    };
    let none: &[u32] = &[];
    for among in [&all[..], &all[1..], none] {
        let within = naming(among);
        for cap in [0usize, 1, 2, 3, 16] {
            for drawn in 0..2u32 {
                let said = format!("{among:?}, cap {cap}, draw {drawn}");
                let (mut plain, mut grouped) = (vec![(9, 9)], vec![(9, 9)]);
                index.deepest_into(&request, |hash| *hash, &within, cap, |_| drawn, &mut plain);
                let mut below = |_: u32| drawn;
                deepest_within_groups(
                    &index,
                    &groups,
                    &request,
                    BLOCK,
                    &within,
                    cap,
                    &mut below,
                    &mut grouped,
                );
                assert_eq!(grouped, plain, "{said}");
                let scores = index.find_matches(&contents, false).scores;
                let mut scored = vec![(9, 9)];
                deepest_of_within_groups(
                    scores,
                    &groups,
                    &request,
                    BLOCK,
                    &within,
                    cap,
                    &mut below,
                    &mut scored,
                );
                assert_eq!(depths(&scored), depths(&plain), "{said}");
                let holds = |&(number, depth): &(u32, u32)| {
                    let at = all.iter().position(|&held| held == number);
                    at.is_some_and(|at| [3, 2, 2, 1][at] == depth) && among.contains(&number)
                };
                assert!(scored.iter().all(holds), "{said}: {scored:?}");
            }
        }
    }
}

/// Where a worker keeps cache groups, each holder is lowered to what its
/// groups can serve, by the index crate's own rule, before the deepest are
/// kept. A holder lowered below another loses its place to it, and a holder
/// lowered to nothing leaves.
#[test]
fn a_holder_is_lowered_to_what_its_groups_serve_before_the_deepest_are_kept() {
    let index = ChainIndex::new();
    let request = [1u64, 2, 3];
    // Workers 0, 1 and 2 hold the three blocks and worker 3 holds two.
    for (at, depth) in [3usize, 3, 3, 2].into_iter().enumerate() {
        store(&index, at, &request[..depth]);
    }
    let all = numbers(&index, 4);
    let groups = GroupViews::new();
    // Worker 1's state group keeps the request's first block alone.
    let first = groups.get_or_create(all[1]);
    first.learn(1, Some(GroupKind::State), 0);
    first.main_stored(None, &named(&request));
    first.group_stored(1, &[named(&request)[0].0]);
    // Worker 2's keeps a block of another prompt and none of the request.
    let other = groups.get_or_create(all[2]);
    other.learn(1, Some(GroupKind::State), 0);
    other.main_stored(None, &named(&request));
    other.main_stored(None, &named(&[9]));
    other.group_stored(1, &[named(&[9])[0].0]);

    // What the index crate's rule says each worker serves of its depth.
    let contents: Vec<ContentHash> = request.iter().map(|&hash| ContentHash(hash)).collect();
    let prefixes = request_prefix_hashes(&contents);
    let served = |at: usize, depth: u32| {
        let view = groups.get(all[at]);
        view.map_or(depth, |view| {
            view.cap(&prefixes, depth, BLOCK as usize).depth
        })
    };
    let serves = [served(0, 3), served(1, 3), served(2, 3), served(3, 2)];
    assert_eq!(serves, [3, 1, 0, 2]);

    let within = naming(&all);
    let lookup = |cap: usize| {
        let mut kept = vec![(u32::MAX, 0)];
        let mut below = |_: u32| 0u32;
        deepest_within_groups(
            &index, &groups, &request, BLOCK, &within, cap, &mut below, &mut kept,
        );
        kept.sort_unstable();
        kept
    };
    // Every holder its groups leave a block, at the depth they serve.
    assert_eq!(lookup(16), [(all[0], 3), (all[1], 1), (all[3], 2)]);
    // The deepest two behind the lowering: worker 1 is as deep as worker 0
    // in the index, and worker 3 takes its place.
    assert_eq!(lookup(2), [(all[0], 3), (all[3], 2)]);
    assert_eq!(lookup(1), [(all[0], 3)]);
    // A worker the mask does not name is no holder.
    let mut kept = Vec::new();
    let mut below = |_: u32| 0u32;
    deepest_within_groups(
        &index,
        &groups,
        &request,
        BLOCK,
        &naming(&all[1..]),
        16,
        &mut below,
        &mut kept,
    );
    kept.sort_unstable();
    assert_eq!(kept, [(all[1], 1), (all[3], 2)]);

    // Holders a host scored itself are lowered and kept by the same rule.
    let scores = index.find_matches(&contents, false).scores;
    let mut scored = vec![(u32::MAX, 0)];
    deepest_of_within_groups(
        scores,
        &groups,
        &request,
        BLOCK,
        &within,
        2,
        &mut below,
        &mut scored,
    );
    scored.sort_unstable();
    assert_eq!(scored, lookup(2));

    // The chain a thread keeps is made anew for each request, and a deeper
    // holder extends the chain a shallower one made: a second request, whose
    // first block differs, held two blocks deep and three, the shallower
    // holder handed first. Each holder's state group keeps its last block
    // alone, so it is served its depth only under that request's own chain.
    let second = [7u64, 2, 3];
    store(&index, 4, &second);
    store(&index, 5, &second[..2]);
    let more = numbers(&index, 6);
    let (deep, shallow) = (more[4], more[5]);
    let chain = request_prefix_hashes(&second.map(ContentHash));
    for (number, depth) in [(deep, 3u32), (shallow, 2)] {
        let view = groups.get_or_create(number);
        view.learn(1, Some(GroupKind::State), 0);
        view.main_stored(None, &named(&second[..depth as usize]));
        view.group_stored(1, &[chain[depth as usize - 1]]);
        assert_eq!(view.cap(&chain, depth, BLOCK as usize).depth, depth);
    }
    deepest_of_within_groups(
        [(shallow, 2), (deep, 3)],
        &groups,
        &second,
        BLOCK,
        &naming(&[deep, shallow]),
        16,
        &mut below,
        &mut kept,
    );
    kept.sort_unstable();
    let mut wanted = [(deep, 3), (shallow, 2)];
    wanted.sort_unstable();
    assert_eq!(kept, wanted);

    // Of the depth the cap cuts, the draw says who stays. Workers 0 and 3
    // serve both of the request's first two blocks, and the cap leaves one.
    let cut = |drawn: u32| {
        let mut kept = Vec::new();
        let mut below = |_: u32| drawn;
        deepest_within_groups(
            &index,
            &groups,
            &request[..2],
            BLOCK,
            &within,
            1,
            &mut below,
            &mut kept,
        );
        kept
    };
    let mut both = [cut(0), cut(1)].concat();
    both.sort_unstable();
    assert_eq!(both, [(all[0], 2), (all[3], 2)]);
}
