//! The cases of a prompt's entry in a token tree its caller lends
//! ([`Prompt::entry`]): each places a request by the one call, under a host
//! that lends its tree. They hold what the entry was asked to what the
//! request's decisions read, and what the tree then holds to the worker the
//! call sent to.

use std::{borrow::Cow, cell::Cell};

use kv_index::TokenTree;

use super::*;
use crate::engine::{
    decision_engine::TokenKey,
    policy_engine::single::tests::{lent, Also},
};

/// A caller's entry for its prompt: the prompt's token ids under no cache
/// partition, and how many times a decision asked for it.
struct Entry<'a> {
    tokens: &'a [u32],
    asked: Cell<u32>,
}

impl<'a> Entry<'a> {
    const fn of(tokens: &'a [u32]) -> Self {
        Self {
            tokens,
            asked: Cell::new(0),
        }
    }
}

impl TokenEntry for Entry<'_> {
    fn token_key(&self, _: usize) -> Option<TokenKey<'_>> {
        self.asked.set(self.asked.get() + 1);
        Some(TokenKey {
            tokens: Cow::Borrowed(self.tokens),
            marker: 0,
        })
    }
}

/// A single request over `members` that states `entry` for its prompt and
/// `admits` as its route's rule.
fn stating<'a>(
    members: &'a PublishedList,
    entry: &'a Entry<'a>,
    admits: Option<WorkerRule<'a>>,
) -> PlaceRequest<'a> {
    PlaceRequest {
        prompt: Prompt {
            tokens: Some(entry.tokens),
            entry: Some(entry),
            ..Prompt::NONE
        },
        admits,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    }
}

/// The workers `tree` holds `tokens` whole for, by their ids. A page is
/// sixteen tokens, and every prompt here is whole pages.
fn holders(tree: &TokenTree, tokens: &[u32]) -> Vec<WorkerId> {
    let found = tree.match_prefix_with_counts(tokens);
    if found.matched_token_count < tokens.len() {
        return Vec::new();
    }
    let ids = found.matched_tenants.iter();
    let mut ids: Vec<_> = ids.map(|address| WorkerId::from_address(address)).collect();
    ids.sort_unstable_by_key(|id| id.0);
    ids
}

/// A conversation placed by the one call goes back to its worker however
/// long it grows: its caller lends its token tree and states the prompt's
/// entry there, and each turn's decision is made inside the tree's match.
/// The entry is asked once a turn. Each call draws a seed of its own, so
/// no turn is held to its worker by a seed.
///
/// The classic policy answers the same: its tree matches the whole prompt,
/// and the share is over the threshold
/// (`model_gateway/src/policies/cache_aware.rs:3473-3491`).
#[test]
fn a_conversation_placed_by_the_one_call_goes_back_to_its_worker() {
    let (engine, _tree) = lent("cache_aware", Also::Nothing);
    let engine = hosted(engine);
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let sent = |tokens: &[u32]| {
        let entry = Entry::of(tokens);
        let request = stating(&members, &entry, None);
        let placement = engine.place(&request).expect("a placement");
        assert_eq!(entry.asked.get(), 1, "the entry is asked for once");
        placement.id()
    };
    let mut said: Vec<u32> = (0..16_000).collect();
    let first = sent(&said);
    for turn in 1..32 {
        said.extend(turn * 1_000_000..turn * 1_000_000 + 16);
        assert_eq!(sent(&said), first, "turn {turn}");
    }
}

/// The entry is asked where a decision of the call reads the lent tree, and
/// nowhere else: never by a place one pick decides, never by a whole
/// decision whose profile reads no record of sent prefixes, and never
/// where the caller lends no tree.
#[test]
fn a_call_whose_decision_reads_no_tree_never_asks_the_entry() {
    let tokens: Vec<u32> = (0..256).collect();
    let asked_of = |name: &str, lends: bool| {
        let engine = hosted(if lends {
            lent(name, Also::Nothing).0
        } else {
            preset(name)
        });
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let entry = Entry::of(&tokens);
        let request = stating(&members, &entry, None);
        engine.place(&request).expect("a placement");
        entry.asked.get()
    };
    assert_eq!(asked_of("round_robin", true), 0);
    assert_eq!(asked_of("least_load", true), 0);
    assert_eq!(asked_of("cache_aware", true), 1);
    assert_eq!(asked_of("cache_aware", false), 0);
}

/// A request that states no entry is decided as before, by the engine's
/// record of hashed prefixes, though its host lends a tree: the tree is
/// neither read nor written. A request that states one beside it is stored
/// in the tree for the worker it was sent to.
#[test]
fn a_request_that_states_no_entry_leaves_the_tree_alone() {
    let (engine, tree) = lent("cache_aware", Also::Nothing);
    let engine = hosted(engine);
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..256).collect();
    let plain = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    engine.place(&plain).expect("a placement");
    assert!(tree.get_tenant_token_counts().is_empty());

    let entry = Entry::of(&tokens);
    let placement = engine.place(&stating(&members, &entry, None));
    let sent = placement.expect("a placement").id();
    assert_eq!(holders(&tree, &tokens), [sent]);
}

/// The tree stores a prompt for the worker the call sends to. A rule that
/// refuses the worker a conversation is on has the next turn placed on
/// another target of the same decision, and the tree names that worker for
/// the turn's prompt and not the refused one. The turn behind it goes back
/// to the worker the request was sent to, in one decision.
///
/// The classic router sends to no worker the rule refuses, and its policy
/// stores the worker it selects among the ones the rule left
/// (`model_gateway/src/routers/common/placement.rs:504-509`,
/// `model_gateway/src/policies/cache_aware.rs:3470-3491`).
#[test]
fn a_prompt_is_stored_for_the_worker_its_rule_admits() {
    let (engine, tree) = lent("cache_aware", Also::Nothing);
    let engine = hosted(engine);
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let mut said: Vec<u32> = (0..4_096).collect();
    let first = {
        let entry = Entry::of(&said);
        let placement = engine.place(&stating(&members, &entry, None));
        placement.expect("a placement").id()
    };
    assert_eq!(holders(&tree, &said), [first]);

    // The rule refuses the conversation's worker from its second turn on.
    let admits = |worker: &dyn Worker| worker_id(worker) != first;
    let turn = |said: &[u32]| {
        let entry = Entry::of(said);
        let placement = engine.place(&stating(&members, &entry, Some(&admits)));
        assert_eq!(entry.asked.get(), 1, "one decision");
        placement.expect("a placement").id()
    };
    said.extend(1_000_000..1_000_016);
    let second = turn(&said);
    assert_ne!(second, first);
    assert_eq!(
        holders(&tree, &said),
        [second],
        "the refused pick is no holder"
    );
    said.extend(2_000_000..2_000_016);
    assert_eq!(turn(&said), second, "back to the worker it was sent to");
    assert_eq!(holders(&tree, &said), [second]);
}

/// A request whose route's rule takes no worker is shed under the rule's
/// name behind one decision, and that decision leaves no holder in the
/// tree: the call could send to none of its targets.
///
/// The classic router asks its policy for nothing over a list its rule
/// emptied (`model_gateway/src/routers/common/placement.rs:504-509`,
/// `:577-579`), so its tree stores nothing either.
#[test]
fn a_call_shed_under_the_rules_name_leaves_no_holder() {
    let (engine, tree) = lent("cache_aware", Also::Nothing);
    let engine = hosted(engine);
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..256).collect();
    let entry = Entry::of(&tokens);
    let admits = |_: &dyn Worker| false;
    let request = stating(&members, &entry, Some(&admits));
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    assert_eq!(entry.asked.get(), 1, "one decision");
    assert!(tree.get_tenant_token_counts().is_empty());
}

/// A call that decides again asks the entry once for each of its
/// decisions, and the tree stores the prompt once, for the worker the call
/// sends to. The route's rule takes one worker here, the one a retry
/// tried: the first decision passes it over, the rule refuses that
/// decision's targets, which stores nothing, and the second decision is
/// held to the rule's worker ([`hold_to_rule`]).
#[test]
fn a_call_that_decides_again_asks_the_entry_once_a_decision() {
    let (engine, tree) = lent("cache_aware", Also::Nothing);
    let engine = hosted(engine);
    let list = pool(16);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..256).collect();
    let entry = Entry::of(&tokens);
    let only = worker_id(list[15].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == only;
    let tried = [only];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        ..stating(&members, &entry, Some(&admits))
    };
    let placement = engine.place(&retry).expect("a placement");
    assert_eq!(placement.id(), only);
    assert_eq!(entry.asked.get(), 2, "one ask a decision");
    assert_eq!(holders(&tree, &tokens), [only]);
}

/// A rule that panics inside the tree's lookup passes its panic to the
/// caller, as a rule's panic does where the walk asks it. The rule here
/// panics once and admits every worker behind that: the first call still
/// ends in the panic, with no later target asked, nothing stored in the
/// tree and nothing left unheard. The request placed again goes through.
#[test]
fn a_rule_that_panics_inside_the_lookup_passes_its_panic_to_the_caller() {
    let (engine, tree) = lent("cache_aware", Also::Nothing);
    let engine = hosted(engine);
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..256).collect();
    let entry = Entry::of(&tokens);
    let calls = Cell::new(0);
    let admits = |_: &dyn Worker| {
        calls.set(calls.get() + 1);
        assert!(calls.get() > 1, "this rule panics once");
        true
    };
    let request = stating(&members, &entry, Some(&admits));
    let placed = catch_unwind(AssertUnwindSafe(|| engine.place(&request).is_ok()));
    assert!(placed.is_err(), "the rule's panic reaches the caller");
    assert_eq!(entry.asked.get(), 1, "one decision");
    assert_eq!(calls.get(), 1, "no target is asked behind the panic");
    assert!(tree.get_tenant_token_counts().is_empty());
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));

    let placement = engine.place(&request).expect("a placement");
    assert_eq!(holders(&tree, &tokens), [placement.id()]);
}
