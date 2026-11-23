//! A cell whose callers hold conversations.
//!
//! A turn carries the conversation so far and 64 tokens more, from 1,024
//! tokens up to 2,048, and then the caller begins another behind the head
//! every prompt shares. The cell's host keeps a token tree of the prompts it
//! sent and lends it to the engine, as a gateway lends the one its own
//! policy keeps ([`CacheIndexes::token_tree`]). A turn has its host state
//! the prompt's block hashes and hand its token ids, in the request the
//! runner states for it. Under `--entry place` a turn is asked by the one
//! call instead, which is handed the token ids and the turn's entry, and
//! under `--entry place_ruled` a route's rule too.
//!
//! A caller makes its prompts before its first turn, about half a kilobyte
//! a decision, so such a cell is run with a stated `--decisions`, and stops
//! at its start where the callers' prompts outgrow the token ids. Its line
//! ends with the turns that carried a conversation on (`turns`), those
//! placed on the worker of the turn before (`returned`) and their share
//! (`return_rate`).
//!
//! Such a cell makes more prompts for more decisions: its count of
//! instructions is the engine's entry and the guard's calls with all they
//! call, read from the two profiles by function.

use std::{
    borrow::Cow,
    hint::black_box,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Barrier,
    },
};

use kv_index::TokenTree;
use smg_policy_controller::engine::{
    CacheIndex, CacheIndexes, DecisionEngine, HashSeeds, Phase, PolicyEngine, PromptHashes,
    TokenEntry, TokenKey,
};

use super::{mix, place, ruled, stretch, Host, Run, CALLER_WARM, MODEL};

/// A conversation's sizes in tokens: the head every prompt begins with, a
/// conversation's first prompt, what a turn adds, and its last prompt.
const HEAD: usize = 32;
const BEGINS: usize = 1_024;
const GROWS: usize = 64;
const FULL: usize = 2_048;

/// The turns of one conversation.
const TURNS: u64 = ((FULL - BEGINS) / GROWS + 1) as u64;

/// The run's timed turns that carried a conversation on, and those of them
/// placed on the worker of the turn before. Each caller adds its own at its
/// end.
static CARRIED: AtomicU64 = AtomicU64::new(0);
static RETURNED: AtomicU64 = AtomicU64::new(0);

/// One turn's prompt: the conversation so far as token ids, and the hashes
/// of its full prompt.
pub(super) struct Turn<'a> {
    pub(super) tokens: &'a [u32],
    pub(super) hashes: &'a PromptHashes,
}

/// A turn's entry in its host's token tree: its token ids, under no cache
/// partition. The runner's maker and the one call's request state the same.
impl TokenEntry for Turn<'_> {
    fn token_key(&self, _: usize) -> Option<TokenKey<'_>> {
        Some(TokenKey {
            tokens: Cow::Borrowed(self.tokens),
            marker: 0,
        })
    }
}

/// A caller's conversations, made before its first turn: each one's full
/// prompt, as token ids and as the block hashes its host states. A turn
/// borrows the first tokens of one.
struct Said {
    tokens: Vec<u32>,
    hashes: Vec<PromptHashes>,
}

impl Said {
    /// The conversations that `turns` turns of `caller` go through. Behind
    /// the head a token is one more than the last, from a start no other
    /// caller's reach, so no two conversations hold the same block there.
    /// The ids fit: the run was held to that ahead of its callers
    /// ([`fits`]).
    fn by(engine: &PolicyEngine, caller: u64, turns: u64) -> Self {
        let conversations = turns.div_ceil(TURNS);
        let first = HEAD as u64 + caller * ids_of(turns);
        let mut last = first as u32 - 1;
        let (mut tokens, mut hashes) = (Vec::new(), Vec::new());
        for _ in 0..conversations {
            let begins = tokens.len();
            tokens.extend(0..HEAD as u32);
            tokens.extend((HEAD..FULL).map(|_| {
                last += 1;
                last
            }));
            let full = Some(&tokens[begins..]);
            hashes.push(engine.prompt_hashes(MODEL, full, None, HashSeeds::PLAIN));
        }
        Self { tokens, hashes }
    }

    /// The prompt of the caller's turn `at`, and whether the turn carries a
    /// conversation on: a conversation's first turn does not.
    fn turn(&self, at: u64) -> (Turn<'_>, bool) {
        let (conversation, step) = ((at / TURNS) as usize, (at % TURNS) as usize);
        let prompt = Turn {
            tokens: &self.tokens[conversation * FULL..][..BEGINS + step * GROWS],
            hashes: &self.hashes[conversation],
        };
        (prompt, step > 0)
    }
}

/// The token ids one caller's `turns` turns take behind the head.
fn ids_of(turns: u64) -> u64 {
    turns.div_ceil(TURNS) * (FULL - HEAD) as u64
}

/// What the cell asks of a run ahead of its callers: it asks the engine's
/// entry or the one call, which its callers ask, and every caller's prompts
/// fit the token ids. A caller that stopped alone would leave the others,
/// and the runner, at their start.
pub(super) fn fits(run: &Run) {
    assert!(
        matches!(run.entry.as_str(), "decide_at" | "place" | ruled::ENTRY),
        "the cell's callers ask the engine's entry or the one call"
    );
    let ids = run.callers as u64 * ids_of(CALLER_WARM + run.decisions);
    assert!(
        HEAD as u64 + ids <= u64::from(u32::MAX),
        "the callers' prompts outgrow the token ids: fewer callers or decisions"
    );
}

/// What the cell's host lends the engine: its own token tree of the prompts
/// it sent, a page the document's block. It keeps no index of cache events.
struct Lends(Arc<TokenTree>);

impl CacheIndexes for Lends {
    fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
        None
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        None
    }

    fn token_tree(&self, _: &str, _: Phase) -> Option<Arc<TokenTree>> {
        Some(Arc::clone(&self.0))
    }
}

/// What the cell makes behind its engine: the host's own tree, lent once,
/// at its start.
pub(super) fn lends(engine: &PolicyEngine) {
    let page = engine.spec().block_size().max(1) as usize;
    let tree = TokenTree::with_config(page, Default::default());
    engine
        .affinity()
        .set_indexes(Arc::new(Lends(Arc::new(tree))));
}

/// One caller's turns: each carries one of the caller's conversations on,
/// by the engine's own entry, under the seed the runner gives that caller's
/// turn, or by the one call where the run asks it ([`place::turn`]). Timed
/// as every caller's turns are ([`stretch`]).
pub(super) fn caller(host: &Host, start: &Barrier, decisions: u64, caller: u64) -> (u64, u64, u64) {
    let said = Said::by(host.engine, caller, CALLER_WARM + decisions);
    let by_call = place::asks();
    let (mut carried, mut returned, mut last) = (0u64, 0u64, None);
    let timed = stretch(start, decisions, |turn| {
        let (prompt, carries) = said.turn(turn);
        let seed = mix(caller << 40 | turn);
        let place = match by_call {
            Some(rule) => place::turn(host, black_box(&prompt), rule),
            None => host.place(black_box(seed), Some(&prompt)),
        };
        if carries && turn >= CALLER_WARM {
            carried += 1;
            returned += u64::from(place == last);
        }
        last = place;
        place
    });
    CARRIED.fetch_add(carried, Ordering::Relaxed);
    RETURNED.fetch_add(returned, Ordering::Relaxed);
    timed
}

/// The words the cell adds at the end of its result line: how many timed
/// turns carried a conversation on, how many of them went back to their
/// worker, and their share.
pub(super) fn words() -> String {
    let (carried, returned) = (
        CARRIED.load(Ordering::Relaxed),
        RETURNED.load(Ordering::Relaxed),
    );
    let rate = returned as f64 / carried.max(1) as f64;
    format!(" turns={carried} returned={returned} return_rate={rate:.4}")
}
