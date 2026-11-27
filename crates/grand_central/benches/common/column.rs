//! The affinity column of the synthetic fleet, settled: every worker admitted
//! an hour ago with a full index, one in eight holding three quarters of a
//! 512-token request's prefix and one in eight a quarter. Its own file, for
//! the benches that feed a column: `mod.rs` is compiled into the gated
//! `decision` binary, whose rows keep to what that binary measures
//! (`.github/bench-gates/grand-central-engine.toml`).
use smg_grand_central::engine::AffinityFacts;

use crate::common::WorkerState;

/// The column `build_view`'s rows are decided with: one entry per worker, in
/// the fleet's order.
pub fn build_column(fleet: &[WorkerState], out: &mut Vec<AffinityFacts>) {
    out.clear();
    for w in fleet {
        let mut affinity = AffinityFacts::default();
        affinity.overlap_blocks = match w.id % 8 {
            3 => 24,
            5 => 8,
            _ => 0,
        };
        affinity.predicted_blocks = 0;
        affinity.admitted_age_ms = 3_600_000;
        affinity.indexed_growth = 50_000;
        affinity.indexed_blocks = 40_000 + w.live_rate;
        out.push(affinity);
    }
}
