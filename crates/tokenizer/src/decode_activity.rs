//! Process-lifetime detokenizer recoveries across all tokenizer instances (the
//! shape of `cache::activity`): how often incremental decoding found that the
//! decoder chain had rewritten text already emitted and streamed the remainder
//! instead of failing. Fixed-size snapshots, no locks; never reset.

use std::sync::atomic::{AtomicU64, Ordering};

static BYTE_RUN: AtomicU64 = AtomicU64::new(0);
static DECODER: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// The recoveries counted on this thread, `[byte_run, decoder]`: a unit
    /// test's own, with the harness running each test on a thread of its own,
    /// while the process-wide totals add up every test of the binary.
    static THREAD_RECOVERIES: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
}

/// Count one recovery. `byte_run` says an invalid byte collapsed a byte run
/// whose complete characters had been emitted; otherwise another decoder step
/// changed the emitted prefix.
pub(crate) fn note_prefix_mismatch(byte_run: bool) {
    let counter = if byte_run { &BYTE_RUN } else { &DECODER };
    counter.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    THREAD_RECOVERIES.with(|cell| {
        let mut counts = cell.get();
        counts[if byte_run { 0 } else { 1 }] += 1;
        cell.set(counts);
    });
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

/// The recoveries counted on the calling thread, by kind: what a unit test
/// did itself, whatever the other tests of the binary do meanwhile.
#[cfg(test)]
pub(crate) fn thread_decode_activity_stats() -> [DecodeActivityStats; 2] {
    let [byte_run, decoder] = THREAD_RECOVERIES.with(std::cell::Cell::get);
    [("byte_run", byte_run), ("decoder", decoder)].map(|(kind, prefix_mismatches)| {
        DecodeActivityStats {
            kind,
            prefix_mismatches,
        }
    })
}
