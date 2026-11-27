//! The cases of a panic in a decision or in a decider's thread.

use super::*;

/// The lines a case's own subscriber wrote.
#[derive(Clone, Default)]
struct Logged(Arc<Mutex<Vec<u8>>>);

impl io::Write for Logged {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A decider of the case's own, on the case's thread: a streak of
/// faults writes one line, and a decision that does not fault ends the
/// streak.
#[test]
fn a_streak_of_faults_writes_one_line_and_a_decision_ends_it() {
    // The second read of the clock panics: the view is built by then.
    let fault = || [Step::At(Duration::ZERO), Step::Panic];
    let steps = fault().into_iter().chain(fault()).chain(fault());
    let bed = bed(Some(preset("random", 1)), true, (1, 8), steps);
    let mut decider = deciding(&bed);
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    // A second subscriber stays registered beside it: with one alone a
    // line is on or off by the thread that reaches it first.
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let written = || {
        let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        text.matches("a decision panicked").count()
    };
    tracing::subscriber::with_default(subscriber, || {
        let ask = |decider: &mut Deciding, id| {
            let handed = decider.answer(checked(frame(id, 1)), GATEWAY, bed.script.taken);
            handed.frame()
        };
        for id in 1..=3 {
            assert_eq!(busy(&ask(&mut decider, id)), Reason::Unspecified);
        }
        assert_eq!(written(), 1, "one line for the streak");
        // A decision that does not fault ends the streak.
        let answer = ask(&mut decider, 4);
        assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
        bed.script.steps.lock().extend(fault());
        assert_eq!(busy(&ask(&mut decider, 5)), Reason::Unspecified);
        assert_eq!(written(), 2, "a new streak, a new line");
    });
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 4);
}

/// A decider's thread holds the guard from its start: one that unwinds
/// is told, to a receiver taken after the word too; one that returns at
/// its queue's end, and a decision's own panic, are not.
#[tokio::test]
async fn a_thread_that_unwinds_is_told_and_one_that_returns_is_not() {
    let told = Arc::new(watch::Sender::new(false));
    let returned = watched(0, &told, || {}).expect("the thread starts").join();
    assert!(returned.is_ok() && !*told.subscribe().borrow());
    // No receiver stands at the unwind: the word is kept for the next.
    let unwound = watched(1, &told, || {
        panic!("the case's own panic, outside a decision")
    })
    .expect("the thread starts")
    .join();
    assert!(unwound.is_err() && *told.subscribe().borrow());

    // A decision that panics is caught: its decider's thread goes on,
    // and the deciders tell nothing.
    let spec = Some(preset("random", 1));
    let mut bed = bed(spec, true, (1, 8), [Step::Panic]);
    assert_eq!(busy(&bed.ask(frame(1, 1)).await), Reason::Unspecified);
    let answer = bed.ask(frame(2, 2)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert!(!*bed.deciders.ended().borrow());
    // The word a thread's unwind says is the one the deciders show, and
    // a receiver made after it reads it by its value.
    bed.deciders.tell_unwound();
    assert!(*bed.deciders.ended().borrow());
    let mut late = bed.deciders.ended();
    assert!(late.wait_for(|ended| *ended).await.is_ok());
}

#[tokio::test]
async fn a_decision_that_panics_is_answered_busy_with_no_reason_and_its_decider_decides_the_next() {
    let spec = Some(preset("random", 1));
    let mut bed = bed(spec, true, (1, 8), [Step::Panic]);
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::Unspecified, 1));
    assert_eq!(answer.request_id, [1]);
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
    let frame = frame(2, 2);
    assert_eq!(
        unleased(bed.ask(frame.clone()).await),
        bed.engines(&frame, 1)
    );
}

/// The `random` picker, which panics at the first pick of a case.
#[derive(Debug)]
struct PanicsOnce(Arc<AtomicBool>);

impl Lifecycle for PanicsOnce {}

impl Picker for PanicsOnce {
    fn name(&self) -> &'static str {
        "panics-once"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        let picked_before = self.0.swap(true, Ordering::Relaxed);
        assert!(picked_before, "the case's own panic, inside a stage");
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// A stage that panics inside a decision is held by the engine's own
/// catch, which refuses the request. Its decider answers it `Busy`
/// with no reason and counts it as a fault, as one its own catch held,
/// files no lease, and decides the next request.
#[tokio::test]
async fn a_stage_that_panics_in_a_decision_is_answered_busy_and_counted_as_a_fault() {
    let fired = Arc::new(AtomicBool::new(false));
    let spec = picking("panics-once", move || -> Box<dyn Picker> {
        Box::new(PanicsOnce(Arc::clone(&fired)))
    });
    let mut bed = bed(Some(spec), true, (1, 8), []);
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::Unspecified, 1));
    assert_eq!(answer.request_id, [1]);
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.live(GATEWAY), 0, "no lease was filed");
    let answer = bed.ask(frame(2, 2)).await;
    assert!(bed.workers.contains(&picked(&answer)), "{answer:?}");
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
}

/// A panic's payload can panic at its own drop. The decider frees it
/// under a catch of its own: the request is answered, its thread stands
/// and the deciders tell nothing.
#[tokio::test]
async fn a_panic_whose_payload_panics_at_its_drop_costs_its_own_request_alone() {
    let spec = Some(preset("random", 1));
    let mut bed = bed(spec, true, (1, 8), [Step::Bomb]);
    let ended = bed.deciders.ended();
    let answer = bed.ask(frame(1, 1)).await;
    assert_eq!((busy(&answer), answer.term), (Reason::Unspecified, 1));
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 1);
    let frame = frame(2, 2);
    assert_eq!(
        unleased(bed.ask(frame.clone()).await),
        bed.engines(&frame, 1)
    );
    assert!(!*ended.borrow(), "the thread stands");
}
