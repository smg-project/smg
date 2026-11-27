//! The cases of a lease behind its issue: how a lease is given up or ends,
//! who is told of it, and what an answer says of its lease's span.

use super::*;

/// [`Told`], whose issue call waits for its case: what falls while a
/// stage's own call runs is the case's to say.
#[derive(Debug)]
struct Waits {
    told: Told,
    reached: std_mpsc::SyncSender<()>,
    go: Arc<Mutex<std_mpsc::Receiver<()>>>,
    /// Its release call panics.
    bursts: bool,
}

impl Lifecycle for Waits {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        self.told.on_decision_issued(lease, decision);
        self.reached.send(()).expect("the case waits for this call");
        let word = self.go.lock().recv_timeout(PATIENCE);
        word.expect("the case lets the call go on");
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        assert!(!self.bursts, "the case's stage panics at a release");
        self.told.on_released(lease, reason);
    }
}

impl Picker for Waits {
    fn name(&self) -> &'static str {
        "waits"
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
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// A leader whose one picker waits inside its issue call: the bed, what
/// the engine told the stage of leases, and the case's two ends of the
/// wait, the call reached and the word that lets it go on.
fn waiting() -> (
    Bed,
    Arc<Mutex<Calls>>,
    std_mpsc::Receiver<()>,
    std_mpsc::SyncSender<()>,
) {
    waits(false)
}

/// [`waiting`]; where it `bursts`, the picker's release call panics.
fn waits(
    bursts: bool,
) -> (
    Bed,
    Arc<Mutex<Calls>>,
    std_mpsc::Receiver<()>,
    std_mpsc::SyncSender<()>,
) {
    let told = Arc::new(Mutex::new(Vec::new()));
    let (reached, at_issue) = std_mpsc::sync_channel(1);
    let (word, go) = std_mpsc::sync_channel(1);
    let (kept, go) = (Arc::clone(&told), Arc::new(Mutex::new(go)));
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("waits", move || {
            Box::new(Waits {
                told: Told(Arc::clone(&kept)),
                reached: reached.clone(),
                go: Arc::clone(&go),
                bursts,
            })
        })
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "waits".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    (bed(Some(spec), true, (2, 8), []), told, at_issue, word)
}

/// The budget is judged behind the lease's issue: what a stage's issue
/// call takes is the decision's time too. A decision that call outlasts
/// leaves as `Busy`, its lease is off the file, and the engine that
/// heard the issue hears the release.
#[tokio::test]
async fn a_decision_its_stages_issue_call_outlasts_leaves_as_busy_and_gives_its_lease_up() {
    let (mut bed, told, at_issue, word) = waiting();
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // The call runs under the turn of the leases' file, on the lease
    // filed ahead of it: no door of the leases answers until it ends.
    // The call took the budget: the next read of the clock, the
    // decision's last, shows the hour past it.
    bed.script.steps.lock().push_back(Step::At(BUDGET));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.live(GATEWAY), 0);
    let (id, late) = (1 << 32 | 1, Some(ReleaseReason::LateDecision));
    assert_eq!(*told.lock(), [(id, None), (id, late)]);
    // Two reads of the clock, as for a decision that leaves.
    assert_eq!(bed.script.readers.lock().len(), 2);
}

/// The fence is judged behind the lease's issue too: a word that closes
/// it while a stage's issue call runs stops the answer, and the lease
/// filed for it is given up as abandoned.
#[tokio::test]
async fn a_fence_closed_by_a_word_inside_a_stages_issue_call_stops_the_answer_and_its_lease() {
    let (mut bed, told, at_issue, word) = waiting();
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // The candidate gives the lead up while the call runs. A successor
    // takes at once on that word: no answer made under the term leaves
    // after it.
    assert!(bed.elector.release().expect("the release"));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(busy(&answer), Reason::NotLeader);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 0);
    assert_eq!(bed.store.live(GATEWAY), 0);
    let (id, abandoned) = (1 << 32 | 1, Some(ReleaseReason::Abandoned));
    assert_eq!(*told.lock(), [(id, None), (id, abandoned)]);
}

/// A lease another hand ended while its decision ran is not the
/// decision's to tell of. A release its gateway hands while the stage's
/// issue call runs waits for the turn that call runs under: the
/// give-up's door applies it first and finds none on file, and the
/// engine hears the issue and the gateway's release alone.
#[tokio::test]
async fn a_lease_another_hand_ended_is_told_nothing_at_the_give_up() {
    let (mut bed, told, at_issue, word) = waiting();
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // Its gateway gives the lease up while the stage's call runs.
    let id = LeaseId(1 << 32 | 1);
    let hand = Hand::new(GATEWAY);
    let gone = ReleaseReason::ClientGone;
    bed.store.hand(&hand, id, Said::Released(None, gone));
    bed.script.steps.lock().push_back(Step::At(BUDGET));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(*told.lock(), [(id.0, None), (id.0, Some(gone))]);
    assert_eq!(bed.store.live(GATEWAY), 0);
}

/// A decider that leaves its work on an empty queue applies what was
/// handed behind its last take, and the engine's lifecycle calls run
/// with it, outside any decision. A stage's call that panics there is
/// held as a decision's fault is, counted apart from the decisions',
/// and the decider stands for its next request.
#[tokio::test]
async fn a_stages_call_that_panics_as_a_decider_leaves_its_work_is_a_fault_it_outlives() {
    let (mut bed, told, at_issue, word) = waits(true);
    let ended = bed.deciders.ended();
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // Its gateway gives the lease up while the stage's call runs: the
    // frame is left to the decider at work, which applies it as it
    // leaves that work.
    let id = LeaseId(1 << 32 | 1);
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    bed.store.hand(&hand, id, gone);
    word.send(()).expect("the stage waits for the case");
    picked(&bed.answer().await);
    let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
    while faults() == 0 && !*ended.borrow() {
        assert!(
            waited.elapsed() < PATIENCE,
            "the decider never left its work"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!*ended.borrow(), "the decider's thread unwound");
    // The call's fault is its own count: no request went unanswered.
    assert_eq!(faults(), 1);
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
    // The lease left the file ahead of the call that panicked.
    assert_eq!(bed.store.live(GATEWAY), 0);
    assert_eq!(*told.lock(), [(id.0, None)]);
    // The decider stands: its next request is decided.
    assert_eq!(bed.submit(frame(2, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    word.send(()).expect("the stage waits for the case");
    picked(&bed.answer().await);
}

/// An answer its stream's queue does not take carries a lease no
/// gateway will name a dispatch for. Its decider hands the release
/// where the queue refuses the answer, full or closed: the lease is
/// off the file and its stage told, abandoned, before the next
/// decision reads a count. An answer the queue took keeps its lease.
#[tokio::test]
async fn a_lease_whose_answer_no_stream_took_is_given_up_where_it_is_dropped() {
    let (mut bed, told, at_issue, word) = waiting();
    let passed = || {
        at_issue
            .recv_timeout(PATIENCE)
            .expect("the stage's issue call");
        word.send(()).expect("the stage waits for the case");
    };
    let job = |bed: &Bed, id: u8, answers: &mpsc::Sender<wire::Decided>| Job {
        request: checked(frame(id, 1)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: answers.clone(),
    };
    // A stream whose queue holds one answer and that reads none: the
    // first answer waits in it, and the second finds it full.
    let (sender, answers) = mpsc::channel(1);
    for id in [1, 2] {
        assert!(bed.deciders.submit(job(&bed, id, &sender)).is_none());
        passed();
    }
    // One decider decides a model's requests in order: it is past the
    // two when it answers a third, on a queue that is read.
    assert_eq!(bed.submit(frame(3, 1)), None);
    passed();
    picked(&bed.answer().await);
    let id = |count: u64| 1 << 32 | count;
    let abandoned = Some(ReleaseReason::Abandoned);
    let mut heard = vec![
        (id(1), None),
        (id(2), None),
        (id(2), abandoned),
        (id(3), None),
    ];
    assert_eq!(*told.lock(), heard);
    assert_eq!(bed.store.live(GATEWAY), 2);
    // The stream ends, and the answer in its queue with it: a fourth
    // answer finds the queue closed.
    drop(answers);
    assert!(bed.deciders.submit(job(&bed, 4, &sender)).is_none());
    passed();
    assert_eq!(bed.submit(frame(5, 1)), None);
    passed();
    picked(&bed.answer().await);
    heard.extend([(id(4), None), (id(4), abandoned), (id(5), None)]);
    assert_eq!(*told.lock(), heard);
    assert_eq!(bed.store.live(GATEWAY), 3);
    let dropped = [&bed.counts.dropped_full, &bed.counts.dropped_gone];
    assert_eq!(dropped.map(|count| count.load(Ordering::Relaxed)), [1, 1]);
}

/// The hand of an unsent answer applies in place where the ring is
/// full, outside any decision. A stage's call that panics there is
/// held as a decision's fault is, counted apart from the decisions',
/// and the decider stands for its next request.
#[tokio::test]
async fn a_stages_panic_at_an_unsent_answers_give_up_is_a_fault_its_decider_outlives() {
    let (mut bed, told, at_issue, word) = waits(true);
    let ended = bed.deciders.ended();
    // A stream that ended: its answer finds the queue closed.
    let (sender, answers) = mpsc::channel(1);
    drop(answers);
    let job = Job {
        request: checked(frame(1, 1)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: sender,
    };
    assert!(bed.deciders.submit(job).is_none());
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // While the stage's call runs, its gateway gives the lease up and
    // fills the ring behind that frame: the decider's own hand finds
    // the ring full and applies in place, where the stage's release
    // call panics.
    let id = LeaseId(1 << 32 | 1);
    let hand = Hand::new(GATEWAY);
    let gone = || Said::Released(None, ReleaseReason::ClientGone);
    bed.store.hand(&hand, id, gone());
    for _ in 1..MemoryStore::HANDOVER {
        bed.store.hand(&hand, LeaseId(0), gone());
    }
    word.send(()).expect("the stage waits for the case");
    let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
    while faults() == 0 && !*ended.borrow() {
        assert!(
            waited.elapsed() < PATIENCE,
            "the decider never gave the lease up"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!*ended.borrow(), "the decider's thread unwound");
    // The call's fault is its own count: no decision panicked.
    assert_eq!(faults(), 1);
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
    // The lease left the file ahead of the call that panicked.
    assert_eq!(bed.store.live(GATEWAY), 0);
    assert_eq!(*told.lock(), [(id.0, None)]);
    // The decider stands: its next request is decided.
    assert_eq!(bed.submit(frame(2, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    word.send(()).expect("the stage waits for the case");
    picked(&bed.answer().await);
}

/// A give-up whose hand applies in place and meets another lease's
/// panicking call there is through: the call is held where the engine
/// is told, and the hand writes its frame behind it. The unsent
/// answer's lease leaves the file at the take behind its hand.
#[tokio::test]
async fn a_give_up_behind_another_leases_panic_at_a_full_ring_is_still_handed() {
    let (mut bed, told, at_issue, word) = waits(true);
    let ended = bed.deciders.ended();
    // The first lease: its answer is taken and the lease stays on file.
    assert_eq!(bed.submit(frame(1, 1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    word.send(()).expect("the stage waits for the case");
    picked(&bed.answer().await);
    assert_eq!(bed.store.live(GATEWAY), 1);
    // The second: a stream that ended, so its answer finds the queue
    // closed.
    let (sender, answers) = mpsc::channel(1);
    drop(answers);
    let job = Job {
        request: checked(frame(2, 1)),
        gateway: GATEWAY,
        received: bed.script.taken,
        answers: sender,
    };
    assert!(bed.deciders.submit(job).is_none());
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    // While the stage's call runs, the gateway gives the first lease up
    // and fills the ring behind that frame.
    let first = LeaseId(1 << 32 | 1);
    let hand = Hand::new(GATEWAY);
    let gone = || Said::Released(None, ReleaseReason::ClientGone);
    bed.store.hand(&hand, first, gone());
    for _ in 1..MemoryStore::HANDOVER {
        bed.store.hand(&hand, LeaseId(0), gone());
    }
    word.send(()).expect("the stage waits for the case");
    // Two held calls: the first lease's release, where the give-up's
    // hand applied in place, and the second lease's own, where the
    // decider takes its turn behind the hand and applies its frame.
    let (faults, waited) = (|| bed.store.lifecycle_faults(), Instant::now());
    while faults() < 2 && !*ended.borrow() {
        assert!(
            waited.elapsed() < PATIENCE,
            "the give-up was not handed behind the call that panicked"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!*ended.borrow(), "the decider's thread unwound");
    // No decision panicked: the calls' faults are their own count.
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
    assert_eq!(bed.store.live(GATEWAY), 0);
    assert_eq!(*told.lock(), [(first.0, None), (first.0 + 1, None)]);
}

/// The lease of an answer its stream's queue refuses is off its file
/// when its decider's give-up returns. The decider is at work there,
/// so the hand leaves the frame to it, and its look ahead of a
/// decision passes a held turn: the give-up takes the turn itself.
/// The decider is the case's own, at work on the case's thread. A
/// second thread, which may not apply at that file, reads it.
#[test]
fn a_refused_answers_lease_is_off_its_file_when_its_give_up_returns() {
    let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
    let mut decider = deciding(&bed);
    decider.working = Some(bed.store.at_work());
    let answer = decider.answer(checked(frame(1, 1)), GATEWAY, bed.script.taken);
    let leased = matches!(answer, wire::Decided::Decision { lease: Some(_), .. });
    assert!(leased, "the decision carries a lease");
    let live = || {
        thread::scope(|scope| {
            let read = scope.spawn(|| bed.store.live(GATEWAY));
            read.join().expect("the reading thread")
        })
    };
    assert_eq!(live(), 1);
    // Its stream refuses the answer.
    decider.unsent(GATEWAY, answer);
    assert_eq!(live(), 0, "the release waited for the decider's next take");
    assert_eq!(bed.counts.faults.load(Ordering::Relaxed), 0);
}

/// A lease's dispatch span counts from its decision's first read, and
/// its answer says what is left of the span at the last. A decision
/// inside its budget that leaves under a millisecond of the span has no
/// lease to hand over: it leaves as `Busy`, late, its lease given up.
/// One with no lease has no span to end, and leaves.
#[tokio::test]
async fn a_slow_decisions_lease_says_what_is_left_of_its_span_and_none_left_is_no_answer() {
    // A budget of twice the span: the span ends first.
    let slow = |id: u8| budgeted(frame(id, 1), DISPATCH_TTL * 2);
    let (mut bed, told, at_issue, word) = waiting();
    let id = |count: u64| 1 << 32 | count;
    // The first decision's stage leaves a millisecond of the span.
    let first = DISPATCH_TTL - Duration::from_millis(1);
    assert_eq!(bed.submit(slow(1)), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    bed.script.steps.lock().push_back(Step::At(first));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(lease_of(&answer).dispatch_ttl_ms, 1);
    let store: &dyn LeasesStore = bed.store.as_ref();
    let filed = store.get(GATEWAY, LeaseId(id(1)));
    let filed = filed.expect("the lease is on file");
    assert_eq!(filed.dispatch_deadline, bed.script.taken + DISPATCH_TTL);
    // The second is taken where the first left the clock, and its
    // stage leaves under a millisecond of the span, half its budget.
    let second = first + DISPATCH_TTL - Duration::from_micros(999);
    assert_eq!(bed.submit_at(slow(2), GATEWAY, first), None);
    at_issue
        .recv_timeout(PATIENCE)
        .expect("the stage's issue call");
    bed.script.steps.lock().push_back(Step::At(second));
    word.send(()).expect("the stage waits for the case");
    let answer = bed.answer().await;
    assert_eq!(busy(&answer), Reason::BudgetGoneOnArrival);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
    // A third of the same length, which the wires shed: no lease, so
    // no span to end, and it leaves as its shed.
    let third = second + DISPATCH_TTL - Duration::from_micros(999);
    let reads = [second, third].map(Step::At);
    bed.script.steps.lock().extend(reads);
    let wireless = constrained(slow(3), 0b010, None);
    assert_eq!(bed.submit_at(wireless, GATEWAY, second), None);
    let shed = declined(proto::ShedReason::NoEndpoints, "wires");
    assert_eq!(outcome(&bed.answer().await), &shed);
    assert_eq!(bed.counts.outlasted.load(Ordering::Relaxed), 1);
    let late = (id(2), Some(ReleaseReason::LateDecision));
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None), late]);
    assert_eq!(bed.store.live(GATEWAY), 1, "the first decision's lease");
}

#[tokio::test]
async fn the_engine_that_decided_is_told_of_a_lease_and_of_its_end_at_the_dispatch_deadline() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(hearing(&told, 1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    // Two leases at the take, the second dispatched.
    for request in 1..=2 {
        bed.ask(frame(request, 1)).await;
    }
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert!(store.dispatched(
        GATEWAY,
        LeaseId(id(2)),
        Dispatch::default(),
        bed.script.taken
    ));
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    // A decision past the dispatch deadline ends the first as
    // abandoned, and its engine is told; the dispatched lease stands.
    let late = DISPATCH_TTL + Duration::from_secs(1);
    bed.script.steps.lock().push_back(Step::At(late));
    assert_eq!(bed.submit_at(frame(3, 1), GATEWAY, late), None);
    bed.answer().await;
    let abandoned = (id(1), Some(ReleaseReason::Abandoned));
    assert_eq!(told.lock()[2..], [(id(3), None), abandoned]);
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
    assert_eq!(store.live(GATEWAY), 2);
}
