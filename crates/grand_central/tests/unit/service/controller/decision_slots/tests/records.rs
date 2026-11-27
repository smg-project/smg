//! The cases of the record a served decision keeps, and of which decisions
//! keep one.

use super::*;

#[tokio::test]
async fn an_unsampled_served_decision_lays_out_no_rings() {
    let mut bed = leading([]);
    let answer = bed.ask(frame(1, 1)).await;
    assert!(matches!(outcome(&answer), Outcome::Pick(_)));
    assert!(bed.deciders.explain(1, GATEWAY, &[1]).is_none());
    assert!(bed.deciders.shared.samples.get().is_none());
    assert_eq!(bed.script.readers.lock().len(), 2);
}

/// A served decision that is kept is filed under the instant its
/// decider read its clock at, which is not the instant its frame was
/// received.
#[tokio::test]
async fn a_kept_served_decision_is_filed_under_its_deciders_instant() {
    // Every read of the case's clock shows the take plus this much.
    // The frame is received at the take.
    const LATER: Duration = Duration::from_millis(250);
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.generation = 1;
    document.explain.sample = 1;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), [Step::At(LATER)]);
    let answer = bed.ask(frame(1, 1)).await;
    assert!(matches!(outcome(&answer), Outcome::Pick(_)));
    let record = bed.deciders.explain(answer.term, GATEWAY, &[1]);
    let record = record.expect("the kept decision");
    assert_eq!(record.at, bed.script.taken + LATER);
}

#[test]
fn a_served_decision_that_keeps_no_record_answers_during_a_read() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.generation = 1;
    document.explain.sample = 2;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("the case's runtime");
    for id in 1..=2 {
        runtime.block_on(bed.ask(frame(id, 1)));
    }

    let shared = Arc::clone(&bed.deciders.shared);
    let rings = shared.samples.get().expect("the ring of the first sample");
    let _read = rings[0].lock();
    let answer = runtime.block_on(bed.ask(frame(3, 1)));
    assert!(matches!(outcome(&answer), Outcome::Pick(_)));
}

#[tokio::test]
async fn a_new_term_restarts_the_served_sample_count() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.generation = 1;
    document.explain.sample = 2;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let first = bed.ask(frame(1, 1)).await;
    assert!(matches!(outcome(&first), Outcome::Pick(_)));

    assert!(bed.elector.release().expect("the release"));
    assert_eq!(
        bed.elector.tick(bed.script.taken).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    bed.store.set_term(2);
    let answer = bed.ask(frame(2, 1)).await;
    assert_eq!(answer.term, 2);
    assert!(matches!(outcome(&answer), Outcome::Pick(_)));
    assert!(bed.deciders.explain(2, GATEWAY, &[2]).is_none());
}
