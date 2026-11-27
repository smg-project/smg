//! The cases of a request that is answered `Busy` or handed back by a full
//! queue, of an answer its stream does not take, and of the counts they
//! move.

use super::*;

/// A frame with under a microsecond of its budget left where its
/// decider takes it is answered there, and no request is made of it:
/// no stage is opened. A request that stated a budget of 0 would say
/// that its frame states none.
#[tokio::test]
async fn a_frame_with_nothing_left_is_answered_and_no_stage_is_opened() {
    let rows = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&rows);
    let spec = picking("seen", move || -> Box<dyn Picker> {
        let rows = Arc::clone(&kept);
        Box::new(Seen { rows })
    });
    let spent = BUDGET - Duration::from_nanos(999);
    let steps = [spent, Duration::ZERO].map(Step::At);
    let mut bed = bed(Some(spec), true, (1, 8), steps);
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert!(rows.lock().is_empty(), "no stage was opened");
    // The next frame is taken with its whole budget and is decided.
    let answer = bed.ask(frame(2, 2)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(rows.lock().len(), 1);
}

#[tokio::test]
async fn a_request_whose_deciders_queue_is_full_comes_back_at_once() {
    let (reached, at_read) = std_mpsc::sync_channel(1);
    let (release, go) = std_mpsc::sync_channel(1);
    // No decider and no depth asked: one thread behind a queue of one.
    let spec = Some(preset("random", 1));
    let mut bed = bed(spec, true, (0, 0), [Step::Hold { reached, go }]);
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_read
        .recv_timeout(PATIENCE)
        .expect("the decider took the first request");
    assert_eq!(bed.submit(frame(2, 2)), None, "the queue holds one");
    assert_eq!(bed.submit(frame(3, 3)), Some(Refused::Full));
    release.send(()).expect("the decider waits for the case");
    // The two that took a place are decided in the order they were
    // handed over; the third never reached the decider.
    for id in [1, 2] {
        let answer = bed.answer().await;
        assert_eq!(answer.request_id, [id]);
        assert!(bed.workers.contains(&picked(&answer)));
    }
    assert_eq!(bed.ask(frame(4, 4)).await.request_id, [4]);
}

#[tokio::test]
async fn an_answer_its_streams_full_queue_does_not_take_is_dropped_and_the_decider_goes_on() {
    let mut bed = leading([]);
    // A stream whose queue holds two answers and that reads none.
    let (sender, mut answers) = mpsc::channel(2);
    for id in 1..=3 {
        let job = Job {
            request: checked(frame(id, 1)),
            gateway: GATEWAY,
            received: bed.script.taken,
            answers: sender.clone(),
        };
        assert!(bed.deciders.submit(job).is_none());
    }
    // One decider decides a model's requests in order: it is past the
    // three when it answers a fourth, on a queue that is read.
    assert_eq!(bed.ask(frame(4, 1)).await.request_id, [4]);
    for id in [1, 2] {
        let answer = answers.recv().await.expect("the queue held it");
        assert_eq!(answer.frame().request_id, [id]);
    }
    assert!(
        answers.try_recv().is_err(),
        "the third found the queue full"
    );
    let dropped = |counts: &DecideStats| {
        [&counts.dropped_full, &counts.dropped_gone].map(|count| count.load(Ordering::Relaxed))
    };
    assert_eq!(dropped(&bed.counts), [1, 0]);
    // A stream that ended takes none: its answer is counted as gone.
    drop(answers);
    let job = Job {
        request: checked(frame(5, 1)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: sender,
    };
    assert!(bed.deciders.submit(job).is_none());
    assert_eq!(bed.ask(frame(6, 1)).await.request_id, [6]);
    assert_eq!(dropped(&bed.counts), [1, 1]);
}

/// Each `Busy` a decider answers is counted at its reason, and a
/// decision moves no count.
#[tokio::test]
async fn each_busy_is_counted_at_its_reason_and_a_decision_moves_no_count() {
    // A decision; a request taken with its budget gone; one whose
    // decision panics; two taken past the renew deadline.
    let steps = [
        Step::At(Duration::ZERO),
        Step::At(Duration::ZERO),
        Step::At(BUDGET),
        Step::Panic,
        Step::At(FENCED),
    ];
    let mut bed = bed(Some(preset("random", 1)), true, (1, 8), steps);
    let counted = |counts: &DecideStats| {
        counts
            .busy
            .each_ref()
            .map(|count| count.load(Ordering::Relaxed))
    };
    let answer = bed.ask(frame(1, 1)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(counted(&bed.counts), [0; 9]);
    assert_eq!(
        busy(&bed.ask(frame(2, 2)).await),
        Reason::BudgetGoneOnArrival
    );
    assert_eq!(busy(&bed.ask(frame(3, 3)).await), Reason::Unspecified);
    for id in [4, 5] {
        assert_eq!(busy(&bed.ask(frame(id, 1)).await), Reason::NotLeader);
    }
    // In the order of the reasons' numbers on the wire.
    assert_eq!(
        DecideStats::BUSY[3..5],
        ["budget_gone_on_arrival", "not_leader"]
    );
    assert_eq!(counted(&bed.counts), [1, 0, 0, 1, 2, 0, 0, 0, 0]);
    // The counts are sized by hand beside the wire's enum: every reason
    // the contract names has a place, and the number past them is no
    // reason. A reason added to the contract fails here, not silently.
    let places = i32::try_from(DecideStats::BUSY.len()).expect("a handful of reasons");
    assert!((0..places).all(|number| Reason::try_from(number).is_ok()));
    assert!(Reason::try_from(places).is_err());
}

#[tokio::test]
async fn a_budget_the_queue_used_up_is_busy_and_one_microsecond_of_it_is_a_decision() {
    let short = BUDGET - Duration::from_micros(1);
    // The first request is taken with one microsecond left, the second
    // with none, the third with under one: the engine reads what is
    // left in whole microseconds, and 0 as none.
    let under = BUDGET - Duration::from_nanos(999);
    let steps = [short, short, BUDGET, under].map(Step::At);
    let mut bed = leading(steps);
    let answer = bed.ask(frame(1, 1)).await;
    assert!(bed.workers.contains(&picked(&answer)));
    let answer = bed.ask(frame(2, 2)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!((answer.request_id.as_slice(), answer.term), (&[2][..], 1));
    let answer = bed.ask(frame(3, 3)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    // One read of the clock each: nothing was read for either after
    // the budget.
    assert_eq!(bed.script.readers.lock().len(), 4);
}

/// A replica that does not lead says so for a request it has no
/// decision for, too.
#[tokio::test]
async fn a_replica_that_does_not_lead_says_so_before_it_reads_a_prompt_as_text() {
    let mut bed = bed(Some(preset("random", 1)), false, (2, 8), []);
    let mut text = frame(1, 1);
    text.facts
        .as_mut()
        .expect("the case's frame has facts")
        .prompt_text = b"a prompt".to_vec();
    assert_eq!(busy(&bed.ask(text).await), Reason::NotLeader);
    assert_eq!(bed.counts.texts.load(Ordering::Relaxed), 0);
}

/// A text is said before the budget is read: the count holds every
/// text a leader took, whatever was left of its budget.
#[tokio::test]
async fn a_prompt_as_text_whose_budget_is_gone_is_still_counted_as_a_text() {
    let mut bed = leading([Step::At(BUDGET)]);
    let mut text = frame(1, 1);
    text.facts
        .as_mut()
        .expect("the case's frame has facts")
        .prompt_text = b"a prompt".to_vec();
    assert_eq!(busy(&bed.ask(text).await), Reason::Unspecified);
    assert_eq!(bed.counts.texts.load(Ordering::Relaxed), 1);
    // The same instant with no text: the budget's own reason.
    let answer = bed.ask(frame(2, 2)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
}

/// The budget is judged again where the answer is handed over, at the
/// read the fence takes there and with no clock read more. A decision
/// with a microsecond left leaves. One whose budget ended while it was
/// made leaves as `Busy` under the reason a late arrival is told, and
/// is counted apart from one. The case's budget is half the dispatch
/// span, so no lease's span ends where its decision's budget does.
#[tokio::test]
async fn a_decision_whose_budget_ended_while_it_was_made_leaves_as_busy_and_is_counted_apart() {
    let budget = DISPATCH_TTL / 2;
    let halved = |id, seed| budgeted(frame(id, seed), budget);
    let short = budget - Duration::from_micros(1);
    let under = budget - Duration::from_nanos(999);
    let steps = [
        Duration::ZERO,
        short,
        Duration::ZERO,
        under,
        Duration::ZERO,
        budget,
        budget,
    ]
    .map(Step::At);
    let mut bed = leading(steps);
    let answer = bed.ask(halved(1, 1)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 0);
    // Under one microsecond left is gone, as where a request is taken.
    for (id, count) in [(2, 1), (3, 2)] {
        let answer = bed.ask(halved(id, 2)).await;
        assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
        assert_eq!((answer.request_id.as_slice(), answer.term), (&[id][..], 1));
        assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), count);
    }
    // A request that arrived late is told the same reason at its
    // decider's first read, and is none of the count.
    let answer = bed.ask(halved(4, 4)).await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 2);
    assert_eq!(bed.script.readers.lock().len(), 7);
}
