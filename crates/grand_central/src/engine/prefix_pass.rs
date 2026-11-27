//! What the last pass of a prefix stage measured: a slot of the thread, kept
//! until the stage says its word of the pick.
//!
//! The `prefix-affinity` stage of the `cache_aware` policy keeps it and takes
//! it through the calls of this file, and nothing else reads or writes it.
//! No entry of the engine asks the stage's word now, so the slot is written
//! and not read outside the stage's own cases.

use std::sync::atomic::{AtomicU32, Ordering};

/// The number of one prefix stage, by which what its pass measured is told
/// from another stage's: a profile may hold two. Only [`PrefixStageId::next`]
/// makes one, so a stage is handed no other stage's number, and none is `0`,
/// which the slot reads as no stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixStageId(u32);

/// The number the next prefix stage takes ([`PrefixStageId::next`]): none
/// takes `0`, which reads as no stage.
static NEXT_STAGE: AtomicU32 = AtomicU32::new(1);

impl PrefixStageId {
    /// The number of a new stage: the counter's next.
    pub(crate) fn next() -> Self {
        Self(NEXT_STAGE.fetch_add(1, Ordering::Relaxed))
    }
}

/// What a pass of a prefix stage measured, kept for what the stage says of
/// the pick made behind its decision ([`Scorer::said`]). A scorer's call is
/// handed no scratch of its decision's, and the pick is made behind it, so
/// the two numbers wait in a slot of the thread. That rests on a decision
/// that runs on one thread from its priced pass to its end, which holds
/// while the engine's decision never waits.
///
/// The slot and this type are private to the file. A stage reaches the slot
/// by three calls ([`keep_prefix_pass`], [`withdraw_prefix_pass`],
/// [`take_prefix_pass`]). A
/// pass keeps its numbers only where it prices the held prefix, and one that
/// orders nothing under the fleet's KV gate withdraws them, so what the slot
/// holds is of the stage's last pass.
///
/// [`Scorer::said`]: crate::engine::framework::Scorer::said
#[derive(Clone, Copy)]
struct PrefixPass {
    /// The stage whose pass it was, by its number ([`PrefixStageId`]): a
    /// profile may hold two, and one's pass says nothing of the other's.
    /// Zero is no stage.
    stage: u32,
    /// The deepest match of the pass's rows, in blocks.
    deepest: u32,
    /// The mean in flight the spill gate read.
    mean_in_flight: f32,
}

impl PrefixPass {
    /// No pass: no stage's number, and a match no row holds.
    const NONE: Self = Self {
        stage: 0,
        deepest: 0,
        mean_in_flight: 0.0,
    };
}

thread_local! {
    static LAST_PASS: std::cell::Cell<PrefixPass> = const { std::cell::Cell::new(PrefixPass::NONE) };
}

/// Forgets what a pass on this thread measured: a case's way to a slot no
/// earlier case wrote.
#[cfg(test)]
pub(crate) fn forget_prefix_pass() {
    LAST_PASS.set(PrefixPass::NONE);
}

/// Keeps what a pass of `stage` measured: the deepest match of its rows, in
/// blocks, and the mean in flight its spill gate read. The first stage to
/// keep on a thread holds the slot until it withdraws its pass or its word
/// is taken, and a later stage's pass says nothing of that one: no entry of
/// the engine empties the slot between decisions now. A later pass of the
/// same stage writes over its own numbers.
#[inline]
pub(crate) fn keep_prefix_pass(stage: PrefixStageId, deepest: u32, mean_in_flight: f32) {
    let stands = LAST_PASS.get().stage;
    if stands == 0 || stands == stage.0 {
        LAST_PASS.set(PrefixPass {
            stage: stage.0,
            deepest,
            mean_in_flight,
        });
    }
}

/// Withdraws what an earlier pass of `stage` left, for a pass that orders
/// nothing. What another stage's pass left stays.
#[inline]
pub(crate) fn withdraw_prefix_pass(stage: PrefixStageId) {
    if LAST_PASS.get().stage == stage.0 {
        LAST_PASS.set(PrefixPass::NONE);
    }
}

/// Empties the slot, whichever stage wrote it, and returns the deepest match
/// and the mean in flight where a pass of `stage` left them.
#[inline]
pub(crate) fn take_prefix_pass(stage: PrefixStageId) -> Option<(u32, f32)> {
    let pass = LAST_PASS.replace(PrefixPass::NONE);
    (pass.stage == stage.0).then_some((pass.deepest, pass.mean_in_flight))
}

#[cfg(test)]
#[path = "../../tests/unit/engine/prefix_pass/tests.rs"]
mod tests;
