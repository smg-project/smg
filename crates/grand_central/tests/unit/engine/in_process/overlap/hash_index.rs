//! A model under a hash index: its record of sent prefixes kept at the view's
//! boundaries, the model's first stored block, a boundary inside a block and
//! one past the hashes a request carries, a reload to another kind of index,
//! a pair side's column, and a request of text.

use super::*;

/// A model whose index is of the kind hash keeps the record of sent
/// prefixes at the view's boundaries: a pick writes the request's head at
/// each boundary it reaches for its worker, and the conversation's next
/// turn is named that worker. From the model's first cache event the
/// index decides a single request alone, as under a tree: the record is
/// neither read nor written.
#[test]
fn a_model_under_a_hash_index_is_named_the_worker_a_prefix_was_sent_to() {
    use crate::document::spec::{IndexKind, IndexSpec};
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    document.view.boundaries = vec![2 * BLOCK, 4 * BLOCK];
    let hash = IndexSpec {
        kind: IndexKind::Hash,
        ..IndexSpec::default()
    };
    document.index.insert("*".into(), hash);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let bed = Bed::new(affinity, spec, 4);
    let asked = prompt(1_000, 4);
    assert!(bed.holders(&asked, NOW).is_empty());
    // No worker reported a block yet: the record takes the pick.
    bed.decide(&request(&asked, 7, NOW), Some(3));
    assert_eq!(bed.holders(&asked, NOW), [(3, 0, 4)]);
    // The next turn goes on from the prompt and is named the same worker.
    let longer = prompt(1_000, 6);
    assert_eq!(bed.holders(&longer, NOW), [(3, 0, 4)]);
    // A conversation that begins alike is named it at the boundary they
    // share.
    let mut alike = tokens(1_000, 2);
    alike.extend(tokens(5_000, 2));
    assert_eq!(bed.holders(&hashed(&alike), NOW), [(3, 0, 2)]);
    // From the table's next read the index holds two of the prompt's blocks
    // for the first worker: the index decides alone, as under a tree, and
    // the record is not read.
    store(&index, 0, &hashes_of(&asked)[..2]);
    assert_eq!(bed.holders(&longer, NOW + SECOND), [(0, 2, 0)]);
    // Nor is it written: the pick of another worker names nobody more.
    bed.decide(&request(&longer, 7, NOW + SECOND), Some(1));
    assert_eq!(bed.holders(&longer, NOW + SECOND), [(0, 2, 0)]);
}

/// A hash index keys the record of sent prefixes at the view's boundaries
/// alone, as the classic policy keys its placement map. A prompt is held
/// at the deepest boundary it reaches, not at its last block. One that
/// parts from it between two boundaries is named its worker at the
/// boundary below. One that shares a head under the first boundary alone
/// is named nobody. A prompt that ends under the first boundary is named
/// nobody, and its pick writes nothing.
#[test]
fn a_hash_index_keys_the_record_at_the_boundaries_alone() {
    use crate::document::spec::{IndexKind, IndexSpec};
    let (affinity, _, _) = hosted(true);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    document.view.boundaries = vec![2 * BLOCK, 4 * BLOCK];
    let hash = IndexSpec {
        kind: IndexKind::Hash,
        ..IndexSpec::default()
    };
    document.index.insert("*".into(), hash);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let bed = Bed::new(affinity, spec, 4);
    let asked = prompt(1_000, 5);
    bed.decide(&request(&asked, 7, NOW), Some(3));
    assert_eq!(bed.holders(&asked, NOW), [(3, 0, 4)]);
    let mut parted = tokens(1_000, 3);
    parted.extend(tokens(5_000, 2));
    assert_eq!(bed.holders(&hashed(&parted), NOW), [(3, 0, 2)]);
    let mut alike = tokens(1_000, 1);
    alike.extend(tokens(5_000, 4));
    assert!(bed.holders(&hashed(&alike), NOW).is_empty());
    let short = prompt(1_000, 1);
    assert!(bed.holders(&short, NOW).is_empty());
    bed.decide(&request(&short, 7, NOW), Some(2));
    assert!(bed.holders(&short, NOW).is_empty());
    // The short prompt left no key behind: a prompt that goes on from it
    // past the first boundary is named the first prompt's worker alone.
    assert_eq!(bed.holders(&prompt(1_000, 2), NOW), [(3, 0, 2)]);
}

/// A boundary counts in whole blocks: one that ends inside a block keys a
/// request that holds the blocks under it, short of the boundary's tokens
/// as it is. A request with fewer whole blocks is named nobody.
#[test]
fn a_boundary_inside_a_block_keys_the_whole_blocks_under_it() {
    use crate::document::spec::{IndexKind, IndexSpec};
    let (affinity, _, _) = hosted(true);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    document.view.boundaries = vec![2 * BLOCK + BLOCK / 2];
    let hash = IndexSpec {
        kind: IndexKind::Hash,
        ..IndexSpec::default()
    };
    document.index.insert("*".into(), hash);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let bed = Bed::new(affinity, spec, 4);
    let asked = prompt(1_000, 2);
    bed.decide(&request(&asked, 7, NOW), Some(3));
    assert_eq!(bed.holders(&asked, NOW), [(3, 0, 2)]);
    assert!(bed.holders(&prompt(1_000, 1), NOW).is_empty());
}

/// A request carries `view.max_block_hashes` block hashes at most, so a
/// boundary past that many blocks keys nothing: a first turn that reaches
/// it is named nobody and leaves no key. A view that reads that deep keys
/// the same turn at the boundary.
#[test]
fn a_boundary_past_the_hashes_a_request_carries_keys_nothing() {
    use crate::document::spec::{IndexKind, IndexSpec};
    let turn: Vec<u32> = (1_000..11_000).collect();
    let held = |max_block_hashes: Option<u32>| {
        let (affinity, _, _) = hosted(true);
        let mut document = PolicySpec::preset("cache_aware").expect("the preset");
        document.generation = 1;
        document.view.boundaries = vec![8_192];
        if let Some(most) = max_block_hashes {
            document.view.max_block_hashes = most;
        }
        let hash = IndexSpec {
            kind: IndexKind::Hash,
            ..IndexSpec::default()
        };
        document.index.insert("*".into(), hash);
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let asked =
            PromptHashes::under_document(&spec, Some(BLOCK), Some(&turn), None, HashSeeds::PLAIN);
        let bed = Bed::new(affinity, spec, 4);
        bed.decide(&request(&asked, 7, NOW), Some(3));
        bed.holders(&asked, NOW)
    };
    // 256 hashes of 16 tokens end at 4,096 tokens: the boundary lies past.
    assert!(held(None).is_empty());
    assert_eq!(held(Some(512)), [(3, 0, 512)]);
}

/// A reload that changes how a model's record is keyed takes a new record:
/// a hash index's keys lie at its boundaries alone, and a tree's read of
/// them would count the first block and stop. What the model is sent under
/// the tree is read whole, by a longer prompt too.
#[test]
fn a_reload_to_another_index_kind_takes_a_new_record() {
    use crate::document::spec::{IndexKind, IndexSpec};
    let (affinity, _, _) = hosted(true);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    document.view.boundaries = vec![BLOCK, 4 * BLOCK];
    let hash = IndexSpec {
        kind: IndexKind::Hash,
        ..IndexSpec::default()
    };
    document.index.insert("*".into(), hash);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let mut bed = Bed::new(affinity, spec, 4);
    let asked = prompt(1_000, 4);
    bed.decide(&request(&asked, 7, NOW), Some(3));
    assert_eq!(bed.holders(&asked, NOW), [(3, 0, 4)]);
    // The same model under a tree: the list's table is read anew.
    document.generation = 2;
    document.index.clear();
    bed.spec = AppliedSpec::validate(&document).expect("the document validates");
    bed.pool = PoolAffinity::default();
    assert!(bed.holders(&asked, NOW).is_empty());
    bed.decide(&request(&asked, 7, NOW), Some(3));
    assert_eq!(bed.holders(&asked, NOW), [(3, 0, 4)]);
    assert_eq!(bed.holders(&prompt(1_000, 6), NOW), [(3, 0, 4)]);
}

/// A pair's side under a hash index reads and writes the record of sent
/// prefixes at the view's boundaries, as a single decision does. A pick
/// written behind a column is what the side's next column holds for its
/// worker, at the deepest boundary the request reaches. A prompt that
/// parts from it between two boundaries is held at the boundary below, and
/// one that shares a head under the first boundary alone by nobody.
#[test]
fn a_sides_column_under_a_hash_index_holds_a_pick_at_its_boundary() {
    use crate::document::spec::{IndexKind, IndexSpec};
    let (affinity, _, _) = hosted(false);
    let mut document =
        PolicySpec::pair_presets("cache_aware", "cache_aware").expect("the pair presets");
    document.generation = 1;
    document.view.boundaries = vec![2 * BLOCK, 4 * BLOCK];
    let hash = IndexSpec {
        kind: IndexKind::Hash,
        ..IndexSpec::default()
    };
    document.index.insert("*".into(), hash);
    let spec = AppliedSpec::validate(&document).expect("the pair document validates");
    let prefills = Bed::new(affinity, spec, 4);
    let asked = prompt(1_000, 5);
    let facts = request(&asked, 7, NOW);
    let (column_of, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(0, 0); 4]);
    let read = read.expect("the record speaks for the prefills");
    record_by_column(
        &prefills.pool,
        Phase::Prefill,
        &facts,
        read,
        &column_of,
        prefills.ids[1],
    );
    // Five blocks reach the boundary of four: the worker holds four.
    let (column_of, _) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(0, 0), (0, 4), (0, 0), (0, 0)]);
    let mut parted = tokens(1_000, 3);
    parted.extend(tokens(5_000, 2));
    let parted = hashed(&parted);
    let (column_of, _) = prefills.column(&request(&parted, 7, NOW), Phase::Prefill);
    assert_eq!(held(&column_of), [(0, 0), (0, 2), (0, 0), (0, 0)]);
    let mut alike = tokens(1_000, 1);
    alike.extend(tokens(5_000, 4));
    let alike = hashed(&alike);
    let (column_of, _) = prefills.column(&request(&alike, 7, NOW), Phase::Prefill);
    assert_eq!(held(&column_of), [(0, 0); 4]);
}

/// Under a hash index a request hashed by its text is recorded nowhere, as
/// the classic policy balances such a request by load: its pick writes
/// nothing into the record of sent prefixes and its next turn is named
/// nobody. A request of token ids of the same pool is written at its
/// boundary.
#[test]
fn a_request_of_text_under_a_hash_index_reads_and_writes_nothing() {
    use crate::document::spec::{IndexKind, IndexSpec};
    let (affinity, _, _) = hosted(false);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    document.view.boundaries = vec![2 * BLOCK, 4 * BLOCK];
    let hash = IndexSpec {
        kind: IndexKind::Hash,
        ..IndexSpec::default()
    };
    document.index.insert("*".into(), hash);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let bed = Bed::new(affinity, spec, 4);
    let written = || {
        let records = bed.affinity.records.lock();
        let record = records.get(MODEL);
        record.is_some_and(|record| !record.predictor.read().is_empty())
    };
    let asked = worded(&text(1_000, 6));
    let facts = request(&asked, 7, NOW);
    assert!(!facts.block_hashes.is_empty());
    bed.decide(&facts, Some(3));
    assert!(!written());
    assert!(bed.holders(&asked, NOW).is_empty());
    let ids = prompt(1_000, 6);
    bed.decide(&request(&ids, 7, NOW), Some(2));
    assert!(written());
    assert_eq!(bed.holders(&ids, NOW), [(2, 0, 4)]);
    assert!(bed.holders(&asked, NOW).is_empty());
}
