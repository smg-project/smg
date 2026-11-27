//! The cases of the fence at a decision: a replica that does not lead or is
//! past its renew deadline, a fence that closes or opens under a later term
//! while a decision is made, and the term a decision leaves under.

use super::*;

#[tokio::test]
async fn a_replica_that_does_not_lead_answers_not_leader_under_the_highest_term_it_knows() {
    // The third request is taken a whole budget after its receipt.
    let steps = [Duration::ZERO, Duration::ZERO, BUDGET].map(Step::At);
    let mut bed = bed(Some(preset("random", 1)), false, (2, 8), steps);
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 0));
    // The election saw another replica hold the lease under term 4.
    bed.standing.send_replace(ElectionObservation {
        holder: "controller-1@10.0.0.8:30100".to_owned(),
        term: 4,
        looked: true,
    });
    let answer = bed.ask(frame(2, 2)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 4));
    // The fence is read before the budget: the reason that still holds
    // for the next request is the one said.
    let answer = bed.ask(frame(3, 3)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 4));
    // One read of the clock each: nothing was built for any.
    assert_eq!(bed.script.readers.lock().len(), 3);
}

#[tokio::test]
async fn a_leader_past_its_renew_deadline_answers_not_leader_under_its_own_term() {
    let mut bed = leading([Step::At(FENCED)]);
    assert_eq!(bed.submit_at(frame(1, 1), GATEWAY, FENCED), None);
    let answer = bed.answer().await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
}

#[tokio::test]
async fn a_fence_that_closes_between_the_build_and_the_hand_over_sends_busy_in_the_decisions_place()
{
    // The first read finds the fence open and the decision is made; the
    // read at the hand-over is past the renew deadline.
    let mut bed = leading([Step::At(Duration::ZERO), Step::At(FENCED)]);
    let mut document = PolicySpec::preset("random").expect("a preset");
    document.generation = 2;
    document.explain.sample = 1;
    bed.store
        .apply(applied(&document))
        .expect("a later generation");
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 1));
    assert_eq!(bed.script.readers.lock().len(), 2);
    assert!(bed.deciders.explain(1, GATEWAY, &[1]).is_none());
    assert!(bed.deciders.shared.samples.get().is_none());
}

#[tokio::test]
async fn a_fence_open_under_a_later_term_at_the_hand_over_sends_busy_too() {
    let (reached, at_read) = std_mpsc::sync_channel(1);
    let (release, go) = std_mpsc::sync_channel(1);
    let mut bed = leading([Step::At(Duration::ZERO), Step::Hold { reached, go }]);
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_read
        .recv_timeout(PATIENCE)
        .expect("the decider reached the hand-over");
    // The decision was made under term 1. The candidate gives the lease
    // up and takes it again: the fence is open, under term 2.
    assert!(bed.elector.release().expect("the release"));
    let tick = bed.elector.tick(bed.script.taken).expect("the take");
    assert_eq!(tick, TickOutcome::Acquired { term: 2 });
    // The election's report tells the store the term of each take.
    bed.store.set_term(2);
    release.send(()).expect("the decider waits for the case");
    let answer = bed.answer().await;
    assert_eq!((busy(&answer), answer.term), (Reason::NotLeader, 2));
    // The next request is decided under the term that stands.
    let answer = bed.ask(frame(2, 2)).await;
    assert_eq!(answer.term, 2);
    assert!(bed.workers.contains(&picked(&answer)));
}

/// The term a decision leaves under is the fence's, the one both reads
/// showed. A higher one the election has seen since is a `Busy`'s to
/// carry.
#[tokio::test]
async fn a_decision_leaves_under_the_fences_term_while_the_election_has_seen_a_later_one() {
    let mut bed = leading([]);
    bed.standing.send_replace(ElectionObservation {
        holder: "controller-1@10.0.0.8:30100".to_owned(),
        term: 4,
        looked: true,
    });
    let answer = bed.ask(frame(1, 1)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(answer.term, 1);
}
