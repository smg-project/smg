//! A host's token tree and the requests decided inside it: when the host is
//! asked for an entry, a tree's page beside the request's blocks,
//! conversations alike past the blocks a host hashes, a partition's marker,
//! many callers at once, and the one tree the host and the engine read and
//! write.

use super::*;

/// A tree's page is no unit of the request. The match is counted in the
/// request's blocks: by the size its host hashed it by, and by the
/// document's where the host hashed nothing. So a host's tree of 64-token
/// pages under a document of 16-token blocks names a prompt of 128 tokens
/// by eight blocks either way, the count the prefix stage sizes the request
/// by, and by four where its host hashed it by blocks of 32. The classic
/// policy divides the tokens its tree matched by the prompt's tokens,
/// whatever the page (`select_worker_with_tokens`,
/// `model_gateway/src/policies/cache_aware.rs`).
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
/// plain one, and the marker is no part of the blocks a worker is named by,
/// nor of the match's two numbers.
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
    let named = bed.keyed_whole(&under(1, &asked), marker, None);
    assert_eq!(named.holders, [(2, 0, 4)]);
    assert_eq!(named.tree, Some(TreeMatch::new(64, 64)));
    assert!(bed.keyed(&under(2, &asked), marker, None).is_empty());
    assert!(bed.keyed(&asked, 0, None).is_empty());
    // Another prompt of the partition shares the marker alone, and a
    // marker is no block of a prompt: a match of nothing, of the prompt's
    // own sixty-four tokens.
    let named = bed.keyed_whole(&under(1, &tokens(5_000, 4)), marker, None);
    assert!(named.holders.is_empty());
    assert_eq!(named.tree, Some(TreeMatch::new(0, 64)));
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
