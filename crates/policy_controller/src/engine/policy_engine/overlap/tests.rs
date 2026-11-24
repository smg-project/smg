//! The naming of a decision's rows from a host's index of cache events, its
//! token tree of sent prefixes and the engine's record of sent prefixes.

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
/// of the prompts it sent to the model's workers, where it keeps one. And
/// how often it was asked for either.
#[derive(Default)]
struct Host {
    index: Option<Arc<ChainIndex>>,
    asked: AtomicUsize,
    tree: Option<Arc<TokenTree>>,
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

    fn token_tree(&self, model: &str, phase: Phase) -> Option<Arc<TokenTree>> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        let single = model == MODEL && phase == Phase::Single;
        self.tree.as_ref().filter(|_| single).cloned()
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

/// What a host hashes of a tokenized prompt.
fn hashed(tokens: &[u32]) -> PromptHashes {
    let spec = preset("cache_aware");
    PromptHashes::under_document(&spec, Some(BLOCK), Some(tokens), None, HashSeeds::PLAIN)
}

/// A prompt of `blocks` whole blocks, its tokens counted from `first`.
fn prompt(first: u32, blocks: u32) -> PromptHashes {
    hashed(&tokens(first, blocks))
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
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    assert_eq!(spec.engine().affinity_rows_cap(MODEL, Phase::Single), 0);
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
        let mut named = None;
        let decision = decide(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            &mut { *facts },
            |_| key,
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
        let returned = decide(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            &mut { *facts },
            |_| key,
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
/// keyed entry followed passes through it with its answer made. A profile
/// that is named no row keeps no record of its own: its
/// list keeps one for a shadow that is named rows, and the column shows
/// what that record holds. A gateway's sticky map hands a request it places
/// to no policy, so the policy's record never takes it
/// (`model_gateway/src/policies/registry.rs:736-739`, `:782-783`).
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
                |_| None,
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
}

/// The host's token tree takes the prompt of a pick the binding's sessions
/// rule placed alone by the rule's own word of it, as the record of hashed
/// prefixes does
/// ([`a_pick_the_sessions_rule_placed_is_recorded_by_the_rules_word_at_both_writes`]).
/// A pick the rule keeps from the profile leaves no prompt in the tree. One
/// that is the profile's own is written as a pick by the profile's stages
/// is. The lookup returns the answer it is handed either way. A gateway's
/// sticky map hands a request it places to no policy, so the policy's tree
/// never takes that request's prompt
/// (`model_gateway/src/policies/registry.rs:736-739`, `:782-783`).
#[test]
fn a_pick_the_sessions_rule_placed_is_written_to_a_tree_by_the_rules_word() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    // One decision of the prompt answered `answer`, inside the host's tree.
    let decided = |asked: &[u32], answer: Decision| {
        let hashes = hashed(asked);
        let key = TokenKey {
            tokens: Cow::Borrowed(asked),
            marker: 0,
        };
        let returned = decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&hashes, 7, NOW),
            |_| Some(key),
            |_| true,
            |_| answer,
        );
        assert_eq!(returned, answer, "the answer handed");
    };
    let (kept, own) = (tokens(1_000, 4), tokens(2_000, 4));
    decided(&kept, ruled(1, bed.ids[1], RulePlaced::Untold));
    assert!(
        bed.keyed(&kept, 0, None).is_empty(),
        "kept from the profile"
    );
    decided(&own, ruled(1, bed.ids[1], RulePlaced::Told));
    assert_eq!(bed.keyed(&own, 0, None), [(1, 0, 4)], "the rule's, told");
    decided(&kept, picked(2, bed.ids[2]));
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
            |_| Some(entry),
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
                asks,
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

/// A pick that meets a held record waits for nobody: it writes nothing, the
/// record counts what it left out, and the next request of the prefix
/// writes it.
#[test]
fn a_pick_that_meets_a_held_record_leaves_its_prefix_out() {
    let (affinity, _, _) = hosted(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    assert!(bed.holders(&asked, NOW).is_empty());
    let record = {
        let records = bed.affinity.records.lock();
        Arc::clone(records.get(MODEL).expect("the model's record"))
    };
    {
        let _reader = record.predictor.read();
        bed.decide(&request(&asked, 7, NOW), Some(2));
    }
    assert_eq!(bed.affinity.left_out(), 1);
    assert!(bed.holders(&asked, NOW).is_empty());
    bed.decide(&request(&asked, 7, NOW), Some(2));
    assert_eq!(bed.holders(&asked, NOW), [(2, 0, 4)]);
    // The whole prompt again, its last use a second old: the write that
    // would move it meets a held record and leaves nothing out.
    {
        let _reader = record.predictor.read();
        bed.decide(&request(&asked, 7, NOW + 2 * SECOND), Some(2));
    }
    assert_eq!(bed.affinity.left_out(), 1);
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
            |_| None,
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
                |_| None,
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
        |_| None,
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
        |_| None,
        |_| true,
        |facts| {
            seen = Some(held(facts.affinity));
            shed()
        },
    );
    assert_eq!(seen.expect("a decision"), [(0, 0), (0, 4), (0, 0), (0, 0)]);
}

/// A pick states what its first target holds of the request, read from the
/// rows the engine named for its decision. A worker the index speaks for is
/// stated by the blocks the index holds for it, of the request's blocks. A
/// worker the rows do not name holds nothing: a miss of the request's
/// blocks, which is not "nothing stated". A worker the record of sent
/// prefixes speaks for is stated by the blocks it was sent, read from the
/// predictor: here the miss's own pick wrote them. A shed holds no pick.
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
    // That pick wrote the request's prefix for its worker in the record.
    assert_eq!(bed.states(&facts, 0), holds(4, 4, OverlapSource::Predictor));
    assert_eq!(bed.stated(&facts, None, shed()), Overlap::NONE);
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

/// A request decided inside its host's token tree states nothing of its
/// pick: the tree's match is a count of tokens and no entry of blocks. The
/// match names the worker all the same. The same request with no entry
/// named is decided over the rows the engine names, and states.
#[test]
fn a_pick_made_inside_a_hosts_token_tree_states_nothing() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    let hashes = hashed(&asked);
    let facts = request(&hashes, 7, NOW);
    let key = || TokenKey {
        tokens: Cow::Borrowed(asked.as_slice()),
        marker: 0,
    };
    let answer = picked(1, bed.ids[1]);
    assert_eq!(bed.stated(&facts, Some(key()), answer), Overlap::NONE);
    // The tree holds the prompt for that worker now, and its match names it.
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 4)]);
    assert_eq!(bed.stated(&facts, Some(key()), answer), Overlap::NONE);
    // No entry named: the record of hashed prefixes, which nothing wrote.
    assert_eq!(bed.states(&facts, 1), holds(0, 4, OverlapSource::None));
}

/// A pick the binding's sessions rule placed alone and keeps from the
/// profile states nothing, over the engine's own rows and over a column its
/// host filled: the request is none of the profile's. One the rule places
/// under its default word is the profile's own, and states its worker's
/// entry as a pick by the profile's stages does. A gateway's sticky map
/// hands a request it places to no policy, so the policy observes nothing
/// of it (`model_gateway/src/policies/registry.rs:736-739`, `:782-783`).
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
            |_| None,
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
            |_| None,
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
        |_| None,
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
        |_| None,
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

/// A naming is made once a decision, ahead of its filters. The holder the
/// record names and the holder the index names stand in it by row, each
/// with its depth, and they stand whole where the filters refuse every
/// worker.
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

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 1, &hashes_of(&asked));
    let bed = Bed::new(affinity, spec, 2);
    bed.decide(&request(&asked, 0, NOW), Some(0)).expect("rows");

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
            |_| None,
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
    assert_eq!(first, [(0, 0, 4), (1, 4, 0)]);
    assert_eq!(named_when(false), first);
}

/// A list whose every worker the index speaks for reads no record and
/// writes none: a naming folds no key and a pick leaves no entry. Once a
/// read of the table finds a worker with no block in the index, the record
/// is read and written again.
#[test]
fn a_pool_whose_workers_are_all_heard_reads_no_record() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let bed = Bed::new(affinity, preset("cache_aware"), 2);
    let asked = prompt(1_000, 4);
    let other = hashes_of(&prompt(9_000, 1))[0];

    // Keep the first worker's block map for the clear.
    let number = index.intern_worker(&address(0)).expect("a worker number");
    let mut map = ChainBlockMap::new();
    let block = StoredBlock {
        seq_hash: other.into(),
        content_hash: ContentHash(other),
    };
    index
        .apply_stored(number, &[block], None, &mut map)
        .expect("a stored block");
    store(&index, 1, &[other]);

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
        let read = table.name(&facts, &mut buffers);
        assert_eq!(read.is_some(), reads, "record read at {now}");
        let blocks = if reads { 4 } else { 0 };
        assert_eq!(buffers.keys.len(), blocks, "folded keys at {now}");
        assert_eq!(placements, blocks, "placements after the pick at {now}");
    };

    check(NOW, false);
    index.apply_cleared(number, &mut map);
    check(NOW + SECOND - 1, false);
    check(NOW + SECOND, true);
}

/// A worker the index numbers and holds no block for is the record's to
/// speak for: an empty write of its events leaves it named by the prefix
/// it was sent. Once the index holds a block for it the index speaks, and
/// a request it holds no block of there names nobody.
#[test]
fn an_empty_index_write_keeps_the_predicted_worker_named() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let bed = Bed::new(affinity, preset("cache_aware"), 2);
    // The index holds a block throughout, the second worker's, as an
    // index its host lends does.
    let other = prompt(9_000, 1);
    store(&index, 1, &hashes_of(&other));
    let asked = prompt(1_000, 4);
    bed.decide(&request(&asked, 7, NOW), Some(0))
        .expect("rows named");

    store(&index, 0, &[]);
    assert_eq!(bed.holders(&asked, NOW + SECOND), [(0, 0, 4)]);

    store(&index, 0, &hashes_of(&other));
    assert!(
        bed.holders(&asked, NOW + 2 * SECOND).is_empty(),
        "the worker's only indexed block belongs to another prompt"
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

/// The index's holders and the record's share the one cap. At a sample of
/// two the index's holder is named beside one of three recorded holders,
/// each of the three over the seeds, in row order. At a sample of one the
/// index's holder is named alone.
#[test]
fn the_two_lists_of_holders_keep_one_cap() {
    use crate::document::spec::{BestParams, PickStage};

    let [two, one] = [2, 1].map(|sample| {
        let mut document = PolicySpec::preset("cache_aware").expect("the preset");
        let profile = document
            .profiles
            .get_mut("cache_aware")
            .expect("the profile");
        profile.pick = PickStage::Best(BestParams { sample }).into();
        AppliedSpec::validate(&document).expect("the document validates")
    });

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 3, &hashes_of(&asked));
    index.intern_worker(&address(4)).expect("a worker number");
    let mut bed = Bed::new(affinity, two, 5);
    let facts = request(&asked, 7, NOW);
    for row in 0..3 {
        bed.decide(&facts, Some(row)).expect("rows");
    }

    let mut seen = [false; 3];
    for seed in 0..32 {
        let holders = bed
            .decide(&request(&asked, seed, NOW), None)
            .expect("rows")
            .holders;
        assert_eq!(holders.len(), 2, "seed {seed}");
        assert_eq!(holders[1], (3, 4, 0), "seed {seed}");
        let (row, overlap, predicted) = holders[0];
        assert!(row < holders[1].0, "seed {seed} named {holders:?}");
        assert_eq!((overlap, predicted), (0, 4), "seed {seed}");
        seen[row as usize] = true;
    }
    assert_eq!(seen, [true; 3], "each prediction takes the open slot");

    bed.spec = one;
    let named = bed
        .decide(&request(&asked, 0, NOW + SECOND), None)
        .expect("rows");
    assert_eq!(named.holders, [(3, 4, 0)]);
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
        |_| None,
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

/// One prefix stage that reads cache events keeps the index's holders
/// first, whatever another reads. Beside a stage that reads the record
/// alone, the worker the index names keeps a sample of one after the
/// prefix was sent to another worker.
#[test]
fn a_stage_that_reads_cache_events_keeps_the_index_holders_first() {
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

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the host index");
    let asked = prompt(1_000, 4);
    store(&index, 1, &hashes_of(&asked));
    let bed = Bed::new(affinity, spec, 2);
    let facts = request(&asked, 0, NOW);

    let cold = bed.decide(&facts, Some(0)).expect("named rows");
    assert_eq!(cold.holders, [(1, 4, 0)]);
    let (column, _) = bed.column(&facts, Phase::Single);
    assert_eq!(
        held(&column),
        [(0, 4), (4, 0)],
        "row 0 has the recorded prefix"
    );
    let warm = bed.decide(&facts, None).expect("named rows");
    assert_eq!(warm.holders, [(1, 4, 0)], "row 1 keeps the only slot");
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
    let (blocks, _) = table.read_record(record, &facts, &mut keys, &mut holders);
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
}
