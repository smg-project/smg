//! The engine's record of sent prefixes: a list of no worker, for which none is
//! made, and the block size the indexes report, which moves its bound.

use super::*;

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
