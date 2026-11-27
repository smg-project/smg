//! The cases of the word an answer says at a term's mark.

use super::*;

/// The answer that carries the id at a term's mark says the word to
/// leave, only once it is handed over, and no id past it says another.
#[tokio::test]
async fn the_id_at_a_terms_mark_says_the_word_once_its_answer_is_handed_over() {
    let mut bed = leading([]);
    let (words, go) = heard(&bed);
    // The mark is three quarters of a term's ids. The id under it, the
    // mark's, and three past it.
    assert_eq!(u64::from(TERM_MARK) * 4, 3 << 32);
    bed.store.spend_ids(u32::MAX - TERM_MARK + 2);
    let under = bed.ask(frame(1, 1)).await;
    assert_eq!(lease_of(&under).id, 1 << 32 | u64::from(TERM_MARK - 1));
    // The decider is held inside the word: an answer that reaches the
    // case was handed over before the word was said.
    let at = bed.ask(frame(2, 1)).await;
    assert_eq!(lease_of(&at).id, 1 << 32 | u64::from(TERM_MARK));
    words
        .recv_timeout(PATIENCE)
        .expect("the mark's decider says the word");
    // A later word finds its go waiting. The ids past the mark say
    // none: the answer after each shows its decider said nothing.
    for _ in 0..8 {
        go.send(()).expect("the decider holds the other end");
    }
    for (id, past) in [(3, 1), (4, 2), (5, 3)] {
        let answer = bed.ask(frame(id, 1)).await;
        assert_eq!(lease_of(&answer).id, 1 << 32 | u64::from(TERM_MARK + past));
    }
    assert!(words.try_recv().is_err(), "one answer a term says the word");
    assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 0);
}

/// The mark's own id need not be one an answer carries: the mint steps
/// over an id on file. The first answer past the mark says the word
/// then, and once.
#[tokio::test]
async fn an_id_past_a_terms_mark_says_the_word_the_marks_own_never_said() {
    let mut bed = leading([]);
    let (words, go) = heard(&bed);
    for _ in 0..8 {
        go.send(()).expect("the decider holds the other end");
    }
    // The term's counter stands two ids past its mark.
    bed.store.spend_ids(u32::MAX - TERM_MARK - 2);
    let first = bed.ask(frame(1, 1)).await;
    assert_eq!(lease_of(&first).id, 1 << 32 | u64::from(TERM_MARK + 3));
    words
        .recv_timeout(PATIENCE)
        .expect("the first answer past the mark says the word");
    // The answer after each shows its decider said nothing.
    for (id, past) in [(2, 4), (3, 5)] {
        let answer = bed.ask(frame(id, 1)).await;
        assert_eq!(lease_of(&answer).id, 1 << 32 | u64::from(TERM_MARK + past));
    }
    assert!(words.try_recv().is_err(), "one answer a term says the word");
}

/// Each term says its own word: a leader past its mark under one term,
/// then under the next, hands the lease over twice.
#[tokio::test]
async fn two_terms_in_a_row_each_say_the_word_at_their_mark() {
    let mut bed = leading([]);
    let (words, go) = heard(&bed);
    let taken = bed.script.taken;
    for term in 1..=2 {
        go.send(()).expect("the decider holds the other end");
        if term > 1 {
            // The bed's candidate, alone over its lease, leaves and
            // takes again at one instant, as its election would.
            let left = bed.elector.step_down(None, taken);
            assert!(left.expect("a planned hand-over"));
            let took = bed.elector.tick(taken).expect("the take");
            assert_eq!(took, TickOutcome::Acquired { term });
            bed.store.set_term(term);
        }
        bed.store.spend_ids(u32::MAX - TERM_MARK + 1);
        let at = bed.ask(frame(1, 1)).await;
        assert_eq!(lease_of(&at).id, term << 32 | u64::from(TERM_MARK));
        // The answer after the mark's shows its decider said its word.
        let past = bed.ask(frame(2, 1)).await;
        assert_eq!(lease_of(&past).id, term << 32 | u64::from(TERM_MARK + 1));
        assert!(words.try_recv().is_ok(), "the word under term {term}");
        assert!(words.try_recv().is_err(), "one word under term {term}");
    }
}

/// Under the last term a replica can take, the mark says no word: a
/// hand-over there leaves no term to take. The term before it says its
/// own.
#[tokio::test]
async fn the_mark_of_the_last_term_says_no_word() {
    let mut bed = leading([]);
    let (words, go) = heard(&bed);
    go.send(()).expect("the decider holds the other end");
    // The bed's candidate, alone over its lease, leaves and takes again
    // at one instant, twice: above a term said, the term before the
    // last; then at a planned hand-over, the last under the ceiling.
    let (last, taken) = (TERM_CEILING - 1, bed.script.taken);
    for (term, above, says) in [(last - 1, Some(last - 2), true), (last, None, false)] {
        let left = bed.elector.step_down(above, taken);
        assert!(left.expect("a term that leaves one to take"));
        let took = bed.elector.tick(taken).expect("the take");
        assert_eq!(took, TickOutcome::Acquired { term });
        bed.store.set_term(term);
        bed.store.spend_ids(u32::MAX - TERM_MARK + 1);
        let at = bed.ask(frame(1, 1)).await;
        assert_eq!(lease_of(&at).id, term << 32 | u64::from(TERM_MARK));
        // The answer after the mark's shows its decider said what it
        // had to say.
        let past = bed.ask(frame(2, 1)).await;
        assert_eq!(lease_of(&past).id, term << 32 | u64::from(TERM_MARK + 1));
        assert_eq!(words.try_recv().is_ok(), says, "under term {term}");
    }
}
