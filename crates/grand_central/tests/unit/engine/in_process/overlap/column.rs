//! A column of entries, one a worker: what a pair side's column holds, the
//! write behind its pick and one handed no entry for its worker, a side and a
//! request of text beside an index, and a decode request asked alone.

use super::*;

/// A pair side's column holds one entry a worker of the side's list, in the
/// list's order: the leading blocks of the request the worker holds by the
/// index, what the index holds for the worker and what that gained since
/// its admission, and the worker's age at the request's instant.
#[test]
fn a_sides_column_holds_each_workers_facts_and_what_it_holds_of_the_request() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let hashes = hashes_of(&asked);
    store(&index, 0, &hashes[..2]);
    store(&index, 2, &hashes);
    store(&index, 2, &hashes_of(&prompt(9_000, 3)));
    let bed = Bed::new(affinity, paired(), 4);
    let indexed = |column: &[AffinityFacts]| -> Vec<(u32, u32)> {
        let indexed = |row: &AffinityFacts| (row.indexed_blocks, row.indexed_growth);
        column.iter().map(indexed).collect()
    };
    let (first, _) = bed.column(&request(&asked, 7, NOW), Phase::Prefill);
    assert_eq!(held(&first), [(2, 0), (0, 0), (4, 0), (0, 0)]);
    assert_eq!(indexed(&first), [(2, 0), (0, 0), (7, 0), (0, 0)]);
    // Half a second on, by the table that stands, each worker is half a
    // second older.
    let (second, _) = bed.column(&request(&asked, 7, NOW + SECOND / 2), Phase::Prefill);
    for (then, later) in first.iter().zip(&second) {
        assert_eq!(later.admitted_age_ms, then.admitted_age_ms + 500);
    }
    // The table read a second on holds what the index gained since.
    store(&index, 0, &hashes_of(&prompt(5_000, 5)));
    let (third, _) = bed.column(&request(&asked, 7, NOW + SECOND), Phase::Prefill);
    assert_eq!(indexed(&third), [(7, 5), (0, 0), (7, 0), (0, 0)]);
    assert_eq!(held(&third), held(&first));
    // A request hashed by nothing is held by nobody, and its column is whole.
    let bare = RequestFacts {
        model: MODEL,
        now_micros: NOW + SECOND,
        ..RequestFacts::default()
    };
    assert_eq!(held(&bed.column(&bare, Phase::Prefill).0), [(0, 0); 4]);
}

/// With no index the record of sent prefixes speaks for a pair side's
/// workers: a pick written behind a column is what the side's next column
/// holds for its worker, and the two lists of a pair keep their workers
/// apart in the model's one record.
#[test]
fn a_sides_pick_is_recorded_and_its_next_column_holds_it() {
    let (affinity, _, _) = hosted(false);
    let prefills = Bed::new(affinity, paired(), 4);
    let asked = prompt(1_000, 4);
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
    let sent_to_one = [(0, 0), (0, 4), (0, 0), (0, 0)];
    let (column_of, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), sent_to_one);
    // A worker that is not of the list writes nothing, and neither does a
    // write for another place than the list's table stands for.
    let read = read.expect("the record speaks for the prefills");
    let (list, stranger) = (&prefills.pool, WorkerId(999));
    record_by_column(list, Phase::Prefill, &facts, read, &column_of, stranger);
    record_by_column(
        list,
        Phase::Decode,
        &facts,
        read,
        &column_of,
        prefills.ids[2],
    );
    assert_eq!(
        held(&prefills.column(&facts, Phase::Prefill).0),
        sent_to_one
    );
    // The decodes are another list of the model, with workers of their own.
    let decode_pool = PoolAffinity::default();
    let (decodes, _) = pool(3);
    let decode_ids: Vec<WorkerId> = (100..103).map(WorkerId).collect();
    let mut filled = Vec::new();
    let decode_column = |filled: &mut Vec<AffinityFacts>| {
        let read = column(
            &prefills.affinity,
            &decode_pool,
            &prefills.spec,
            &decodes,
            &decode_ids,
            Phase::Decode,
            &facts,
            filled,
        );
        (held(filled), read)
    };
    let (holds, read) = decode_column(&mut filled);
    assert_eq!(holds, [(0, 0); 3]);
    let read = read.expect("the record speaks for the decodes");
    record_by_column(
        &decode_pool,
        Phase::Decode,
        &facts,
        read,
        &filled,
        decode_ids[2],
    );
    assert_eq!(decode_column(&mut filled).0, [(0, 0), (0, 0), (0, 4)]);
    assert_eq!(
        held(&prefills.column(&facts, Phase::Prefill).0),
        sent_to_one
    );
    assert_eq!(prefills.affinity.left_out(), 0);
}

/// A write behind a column reads the picked worker's own entry of it, and a
/// worker the column holds no entry for is written as no repeat. The first
/// prefill is sent a prompt, so the next read finds the record holding the
/// prompt whole, its end used at this tick: a pick the column names among
/// the prompt's holders is a repeat, which writes nothing. The second
/// prefill's entry names it no holder, though the column's first entry
/// names one: its pick is recorded. The write for the third prefill is
/// handed an empty column: no entry says the fill found that worker there,
/// so the prompt is recorded for it. The fill gives every worker of its
/// list an entry, so no decision of the engine hands such a column: this
/// case hands one itself, to hold the write's own contract.
#[test]
fn a_write_handed_no_entry_for_its_worker_is_no_repeat() {
    let (affinity, _, _) = hosted(false);
    let prefills = Bed::new(affinity, paired(), 3);
    let asked = prompt(1_000, 4);
    let facts = request(&asked, 7, NOW);
    let write_for = |column: &[AffinityFacts], read: PrefixRead, at: usize| {
        let (list, id) = (&prefills.pool, prefills.ids[at]);
        record_by_column(list, Phase::Prefill, &facts, read, column, id);
    };
    let holds = || held(&prefills.column(&facts, Phase::Prefill).0);
    let (filled, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&filled), [(0, 0); 3]);
    let read = read.expect("the record speaks for the prefills");
    write_for(&filled, read, 0);

    // The record holds the prompt whole and fresh for the first prefill.
    let (filled, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&filled), [(0, 4), (0, 0), (0, 0)]);
    let read = read.expect("the record speaks for the prefills");
    assert!(read.read.whole && read.read.fresh, "{read:?}");
    // The second prefill's own entry names it no holder.
    write_for(&filled, read, 1);
    assert_eq!(holds(), [(0, 4), (0, 4), (0, 0)]);
    // The column holds no entry for the third prefill.
    write_for(&[], read, 2);
    assert_eq!(holds(), [(0, 4); 3]);
    assert_eq!(prefills.affinity.left_out(), 0);
}

/// A pair side's column holds a text's prefix for a worker the index holds
/// blocks for. A request hashed by its text asks no index, so the record of
/// sent prefixes speaks for every worker of the side's list: its pick is
/// written behind the column for such a worker, and the side's next column
/// holds it. So where the index holds a block for every worker of the list.
/// The classic router keeps a tree of sent texts a side, and reads and
/// writes it for a request with no token ids whatever its index holds
/// (`model_gateway/src/policies/cache_aware.rs:1511-1538`, `:3519-3596`).
///
/// A request hashed by token blocks answers as before beside it: a pick of
/// a worker the index speaks for writes nothing, and a list whose every
/// worker the index speaks for reads no record.
#[test]
fn a_sides_column_holds_a_text_sent_to_a_worker_the_index_holds_blocks_for() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let tokens = prompt(1_000, 4);
    let other = hashes_of(&prompt(9_000, 1));
    store(&index, 0, &other);
    store(&index, 1, &hashes_of(&tokens)[..2]);
    let prefills = Bed::new(affinity, paired(), 3);
    let said = text(1_000, 4);
    let asked = worded(&said);

    // The index holds no block for the third prefill yet.
    let facts = request(&asked, 7, NOW);
    let (column_of, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(0, 0); 3]);
    let read = read.expect("the record speaks for the prefills");
    record_by_column(
        &prefills.pool,
        Phase::Prefill,
        &facts,
        read,
        &column_of,
        prefills.ids[1],
    );
    let sent_to_one = [(0, 0), (0, 4), (0, 0)];
    assert_eq!(
        held(&prefills.column(&facts, Phase::Prefill).0),
        sent_to_one
    );
    let by_tokens = request(&tokens, 7, NOW);
    let by_index = [(0, 0), (2, 0), (0, 0)];
    let (column_of, read) = prefills.column(&by_tokens, Phase::Prefill);
    assert_eq!(held(&column_of), by_index);
    let read = read.expect("the record speaks for the third prefill");
    record_by_column(
        &prefills.pool,
        Phase::Prefill,
        &by_tokens,
        read,
        &column_of,
        prefills.ids[1],
    );
    assert_eq!(
        held(&prefills.column(&by_tokens, Phase::Prefill).0),
        by_index
    );

    // The table read a second on finds a block for every prefill.
    store(&index, 2, &other);
    let later = NOW + SECOND;
    let (column_of, read) = prefills.column(&request(&asked, 7, later), Phase::Prefill);
    assert_eq!(held(&column_of), sent_to_one);
    assert!(read.is_some(), "a request of text reads the record");
    let (column_of, read) = prefills.column(&request(&tokens, 7, later), Phase::Prefill);
    assert_eq!(held(&column_of), by_index);
    assert!(read.is_none(), "a token request reads none");
    assert_eq!(prefills.affinity.left_out(), 0);
}

/// Each side of a pair is decided at its own phase and keeps both records
/// beside an index that holds a block: its column holds the index's blocks
/// for a worker the index speaks for and the prefix sent for another. The
/// index decides a single request alone.
#[test]
fn a_sides_column_keeps_the_record_beside_an_index() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 0, &hashes_of(&asked)[..2]);
    let prefills = Bed::new(affinity, paired(), 4);
    let facts = request(&asked, 7, NOW);
    let (column_of, read) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(2, 0), (0, 0), (0, 0), (0, 0)]);
    let read = read.expect("the record speaks for the other prefills");
    record_by_column(
        &prefills.pool,
        Phase::Prefill,
        &facts,
        read,
        &column_of,
        prefills.ids[1],
    );
    let (column_of, _) = prefills.column(&facts, Phase::Prefill);
    assert_eq!(held(&column_of), [(2, 0), (0, 4), (0, 0), (0, 0)]);
}

/// A decode request asked alone is a single decision read at its own phase,
/// where the index does not decide alone. Under a profile that is named no
/// row and reads cache events too, its pick of a worker the index holds
/// blocks for writes nothing, as a pick over named rows writes nothing
/// there. So the same list, read next by a profile that reads sent prefixes
/// alone, is handed no send for that worker. Its pick of a worker the index
/// holds nothing for is written, and the first profile reads that send next.
#[test]
fn a_decode_request_asked_alone_writes_nothing_for_a_worker_the_index_holds_blocks_for() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 1, &hashes_of(&asked)[..2]);
    let decode = || RequestFacts {
        phase: Phase::Decode,
        ..request(&asked, 7, NOW)
    };
    let handed = |bed: &Bed| {
        let mut seen = None;
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut decode(),
            None,
            |_| None,
            || None,
            |_| true,
            |facts| {
                seen = Some(held(facts.affinity));
                shed()
            },
        );
        seen.expect("a decision")
    };
    let by_index = [(0, 0), (2, 0), (0, 0), (0, 0)];
    let merged = Bed::new(affinity, handed_whole(), 4);
    assert!(merged.decide(&decode(), Some(1)).is_none());
    assert_eq!(handed(&merged), by_index);
    // The record took nothing of that pick: a reader of the record alone
    // is handed the index's blocks and no send.
    let alone = Bed {
        pool: PoolAffinity::default(),
        spec: named_no_row(reading_alone),
        ..merged
    };
    assert_eq!(handed(&alone), by_index);
    // A pick of a worker the index holds nothing for is written: the index
    // does not decide a decode request alone.
    let merged = Bed {
        pool: PoolAffinity::default(),
        spec: handed_whole(),
        ..alone
    };
    assert!(merged.decide(&decode(), Some(3)).is_none());
    assert_eq!(handed(&merged), [(0, 0), (2, 0), (0, 0), (0, 4)]);
}
