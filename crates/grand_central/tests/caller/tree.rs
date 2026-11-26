//! The program's token tree.
//!
//! The program keeps a token tree of the prompts it sent, one a model, and
//! lends it to the engine at its start ([`Program::start`]). A request states
//! its prompt's token ids and its entry in that tree ([`Asked::tokens`]).
//!
//! Under a document that reads the prefixes sent to each worker, the engine
//! decides a single request inside the tree's match. The same descent writes
//! the request for the worker it goes to. The tree stays the program's: it
//! takes a worker that left out of it and holds it to a size.
//!
//! Each case serves the turns of conversations and reads where each went and
//! what the tree holds. A gateway's own policy keeps such trees and reads them
//! the same way: `CacheAwarePolicy`
//! (`model_gateway/src/policies/cache_aware.rs`).
//!
//! [`Asked::tokens`]: crate::program::Asked::tokens
//! [`Program::start`]: crate::program::Program::start

use kv_index::TenantId;
use smg_grand_central::document::PolicySettings;

use crate::{
    pair::{fleet, PLAIN},
    program::{Asked, Program},
    served, under, under_pairs, url, worker, workers, MODEL,
};

/// The tokens of one page of the program's tree: the block of the document
/// the settings mean.
const PAGE: u32 = 16;

/// The policy a gateway's `policy: cache_aware` names, at the settings its
/// name alone gives it.
fn cache_aware() -> PolicySettings {
    PolicySettings::of_label("cache_aware").expect("a policy of that name")
}

/// A conversation's first prompt: `pages` pages of token ids that begin at
/// `from`, so two conversations that begin apart share no page.
fn begun(from: u32, pages: u32) -> Vec<u32> {
    (from..from + pages * PAGE).collect()
}

/// One turn served: `said`, the conversation so far, is the request's
/// prompt. The address the request was sent to.
fn turn(program: &Program, said: &[u32]) -> String {
    let answer = program.serve(&Asked {
        tokens: Some(said),
        ..PLAIN
    });
    assert_eq!((answer.status, answer.sent.len()), (200, 1));
    answer.sent.into_iter().next().expect("one send")
}

/// The workers the program's tree holds `said` whole for, by their
/// addresses in order. None where the tree holds a part of it alone, and
/// where the engine asked for no tree.
fn holders(program: &Program, said: &[u32]) -> Vec<String> {
    let Some(tree) = program.tree(MODEL) else {
        return Vec::new();
    };
    let found = tree.match_prefix_with_counts(said);
    if found.matched_token_count < said.len() {
        return Vec::new();
    }
    let holders = found.matched_tenants.iter();
    let mut holders: Vec<String> = holders.map(ToString::to_string).collect();
    holders.sort_unstable();
    holders
}

/// A conversation goes back to the worker its first turn was sent to, however
/// long it grows: the tree matches the whole prompt. Eight workers stand, and
/// a conversation of a thousand pages grows a page a turn.
///
/// A program that lends no tree is named no holder for such a prompt where no
/// index of cache events holds a block: nothing sends it back.
///
/// A gateway's own policy sends a request to a worker its tree's match found,
/// where the match is over its threshold: the test of `cache_threshold` in
/// `select_worker_with_tokens` (`model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_conversation_goes_back_to_the_worker_it_was_sent_to() {
    let fleet = workers(8);
    let (program, _loads, _task) = under(cache_aware(), &fleet, served);
    let mut said = begun(0, 1_000);
    let first = turn(&program, &said);
    for at in 1..32 {
        said.extend(begun(at * 1_000_000, 1));
        assert_eq!(turn(&program, &said), first, "turn {at}");
    }
}

/// The tree holds a prompt for the worker its request was sent to, and for no
/// other. Two workers serve four conversations. Behind each one's first turn
/// the tree holds its prompt whole for the worker the client's request went
/// to.
///
/// A gateway's own policy writes a request into its tree for the worker it
/// selected alone, inside the match: `match_and_insert_with_hash` in
/// `select_worker_with_tokens` (`model_gateway/src/policies/cache_aware.rs`).
#[test]
fn the_tree_holds_a_prompt_for_the_worker_it_was_sent_to() {
    let fleet = workers(2);
    let (program, _loads, _task) = under(cache_aware(), &fleet, served);
    for conversation in 0..4 {
        let said = begun(conversation * 1_000_000, 64);
        assert_eq!(holders(&program, &said), Vec::<String>::new());
        let sent = turn(&program, &said);
        assert_eq!(holders(&program, &said), [sent], "{conversation}");
    }
}

/// A worker that left holds nothing in the tree, and one that joins at its
/// address starts with nothing. The engine reads no worker that left, but the
/// tree is the program's to keep. Without the purge it would hold the worker's
/// prompts for good, and hand them to the worker that joins there.
///
/// The worker leaves under its address written without the scheme, which the
/// registry finds it by. The purge goes by the address the worker states, the
/// one the engine wrote its prompts under.
///
/// A gateway's own policy takes a removed worker out of every tree:
/// `remove_worker_by_url` (`model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_worker_that_left_holds_nothing_in_the_tree() {
    let fleet = workers(2);
    let (program, _loads, _task) = under(cache_aware(), &fleet, served);
    let said = begun(0, 64);
    let first = turn(&program, &said);
    assert_eq!(holders(&program, &said), [first.as_str()]);
    let bare = first.trim_start_matches("http://");
    assert!(program.leave(bare), "the registry finds the worker");
    assert_eq!(holders(&program, &said), Vec::<String>::new());
    let tree = program.tree(MODEL).expect("the model's tree");
    assert_eq!(tree.tenant_token_size(&TenantId::from(first.as_str())), 0);
    let at = (0..2).find(|at| url(*at) == first);
    let again = worker(at.expect("a worker of the fleet"));
    assert!(program.join(again), "a free address joins");
    assert_eq!(holders(&program, &said), Vec::<String>::new());
}

/// The program's pass holds its tree to a size. Eight conversations of 64
/// pages fill the tree with 8,192 tokens, and the pass at half of them leaves
/// at most that many.
///
/// A gateway's own policy runs such a pass over its trees on a timer:
/// `evict_trees` (`model_gateway/src/policies/cache_aware.rs`).
#[test]
fn the_programs_pass_holds_its_tree_to_a_size() {
    let fleet = workers(2);
    let (program, _loads, _task) = under(cache_aware(), &fleet, served);
    for conversation in 0..8 {
        turn(&program, &begun(conversation * 1_000_000, 64));
    }
    let tree = program.tree(MODEL).expect("the model's tree");
    let held = tree.total_token_size();
    assert_eq!(held, 8 * 64 * PAGE as usize);
    program.trim(held / 2);
    let left = tree.total_token_size();
    assert!(left > 0 && left <= held / 2, "{left} of {held}");
}

/// Under a document that reads no sent prefix the engine asks for no tree,
/// so none is made: the program's store stays empty, whatever its requests
/// state.
#[test]
fn a_document_that_reads_no_sent_prefix_asks_for_no_tree() {
    let fleet = workers(2);
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, served);
    turn(&program, &begun(0, 64));
    assert!(program.tree(MODEL).is_none());
}

/// A pair's request is decided inside no tree. The engine decides a single
/// request there and no other, and the program lends its tree to a single
/// request alone. Two prefills and two decodes stand under the same policy,
/// and a request states its prompt.
///
/// A gateway lends its trees to a single request alone:
/// `EngineCaches::token_tree` (`model_gateway/src/policies/lent_trees.rs`).
#[test]
fn a_pairs_request_is_lent_no_tree() {
    let fleet = fleet(2, 2);
    let (program, _loads, _task) = under_pairs(cache_aware(), &fleet, served);
    let said = begun(0, 64);
    let answer = program.serve(&Asked {
        tokens: Some(&said),
        ..PLAIN
    });
    assert_eq!((answer.status, answer.sent.len()), (200, 2));
    assert!(program.tree(MODEL).is_none());
}
