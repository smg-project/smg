//! Process-lifetime detokenizer recoveries across all tokenizer instances (the
//! shape of `cache::activity`): how often incremental decoding found that the
//! decoder chain had rewritten text already emitted and streamed the remainder
//! instead of failing. Fixed-size snapshots, no locks; never reset.

use std::sync::atomic::{AtomicU64, Ordering};

static BYTE_RUN: AtomicU64 = AtomicU64::new(0);
static DECODER: AtomicU64 = AtomicU64::new(0);

/// Count one recovery. `byte_run` says an invalid byte collapsed a byte run
/// whose complete characters had been emitted; otherwise another decoder step
/// changed the emitted prefix.
pub(crate) fn note_prefix_mismatch(byte_run: bool) {
    let counter = if byte_run { &BYTE_RUN } else { &DECODER };
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Cumulative recoveries of one kind across all tokenizer instances.
#[derive(Debug, Clone, Copy)]
pub struct DecodeActivityStats {
    /// `byte_run` (an invalid byte collapsed emitted characters) or `decoder`
    /// (another decoder step rewrote the emitted prefix).
    pub kind: &'static str,
    /// Steps and finishes that emitted a best-effort remainder.
    pub prefix_mismatches: u64,
}

/// Read process-lifetime totals, including activity before a metrics recorder exists.
pub fn decode_activity_stats() -> [DecodeActivityStats; 2] {
    [("byte_run", &BYTE_RUN), ("decoder", &DECODER)].map(|(kind, counter)| DecodeActivityStats {
        kind,
        prefix_mismatches: counter.load(Ordering::Relaxed),
    })
}
