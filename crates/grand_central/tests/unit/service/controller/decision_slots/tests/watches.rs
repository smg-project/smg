//! The cases of the decider that applies what is handed to a file of
//! leases, of the decider a lease's events act as, and of the watch a
//! decider keeps while it waits for a job.

use super::*;

/// A request refused before anything is built applies what is handed
/// too, so a hand at a full ring waits for one request of the file's
/// decider, whatever its answer. The decider here is a former leader
/// behind its closed fence whose gateway still asks it: it refuses each
/// request and never rests, and the hand is through all the same.
#[tokio::test]
async fn a_refused_request_applies_what_a_full_ring_holds() {
    let mut bed = bed(Some(preset("random", 1)), true, (1, 8), []);
    bed.ask(frame(1, 1)).await;
    let hold = || {
        let (reached, held) = std_mpsc::sync_channel(1);
        let (go, word) = std_mpsc::sync_channel(1);
        (Step::Hold { reached, go: word }, held, go)
    };
    // The second request is held at its read of the clock: its decider
    // is at work, and what is handed is left to it.
    let (first, held, go) = hold();
    bed.script.steps.lock().push_back(first);
    assert_eq!(bed.submit(frame(2, 1)), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    // The ring fills, and one hand more waits at it.
    let gone = || Said::Released(None, ReleaseReason::ClientGone);
    let hand = Hand::new(GATEWAY);
    for _ in 0..MemoryStore::HANDOVER {
        bed.store.hand(&hand, LeaseId(0), gone());
    }
    let (through, handed) = std_mpsc::channel();
    let store = Arc::clone(&bed.store);
    let waiting = thread::spawn(move || {
        store.hand(&Hand::new(GATEWAY), LeaseId(0), gone());
        through.send(()).expect("the case waits for the hand");
    });
    // The lead is an hour gone at every read from here on. Two more
    // requests wait behind the held one, and the last is held at its
    // read of the clock: by then the decider has refused two requests
    // and has not rested.
    let hour = Duration::from_secs(3600);
    let (last, held, last_go) = hold();
    *bed.script.shown.lock() = hour;
    bed.script.steps.lock().extend([Step::At(hour), last]);
    assert_eq!(bed.submit(frame(3, 1)), None);
    assert_eq!(bed.submit(frame(4, 1)), None);
    go.send(()).expect("the decider holds the other end");
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock for the last request");
    let handed = handed.recv_timeout(PATIENCE);
    last_go.send(()).expect("the decider holds the other end");
    handed.expect("a refused request applies what the ring holds");
    waiting.join().expect("the hand's thread");
    for id in 2..=4 {
        let answer = bed.answer().await;
        assert_eq!(
            (busy(&answer), answer.request_id.as_slice()),
            (Reason::NotLeader, &[id][..])
        );
    }
}

/// A decider with a job at hand is at work at the leases' file: what a
/// session hands meanwhile is left to it, and it applies it before its
/// decision reads the rows' counts, ahead of its own issue. A frame
/// handed while no decider has a job is applied all the same.
#[tokio::test]
async fn a_frame_handed_while_a_decider_decides_is_applied_by_that_decider() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&told);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("told", move || Box::new(Told(Arc::clone(&kept))))
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "told".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    let mut bed = bed(Some(spec), true, (1, 8), []);
    let id = |count: u64| 1 << 32 | count;
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    bed.ask(frame(1, 1)).await;
    // The second decision is held at its first read of the clock: its
    // decider has the job at hand.
    let (reached, held) = std_mpsc::sync_channel(1);
    let (go, word) = std_mpsc::sync_channel(1);
    let hold = Step::Hold { reached, go: word };
    bed.script.steps.lock().push_back(hold);
    assert_eq!(bed.submit(frame(2, 1)), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    let hand = Hand::new(GATEWAY);
    bed.store.hand(&hand, LeaseId(id(1)), gone);
    assert_eq!(*told.lock(), [(id(1), None)], "left to the decider at work");
    go.send(()).expect("the decider holds the other end");
    bed.answer().await;
    let released = |count| (id(count), Some(ReleaseReason::ClientGone));
    assert_eq!(*told.lock(), [(id(1), None), released(1), (id(2), None)]);
    // The decider waits on its empty queue, or is about to: it applies
    // as it leaves its work, or the hand does.
    bed.store.hand(&hand, LeaseId(id(2)), gone);
    settled(|| told.lock().len() >= 4).await;
    assert_eq!(told.lock()[3], released(2));
}

/// The `random` picker, keeping the decider its thread acts as at each
/// call of a lease's issue and release.
#[derive(Debug)]
struct Named(Arc<Mutex<Vec<Option<DecisionSlot>>>>);

impl Lifecycle for Named {
    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        self.0.lock().push(DecisionSlot::acting());
    }

    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        self.0.lock().push(DecisionSlot::acting());
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Issued, EventKind::Released])
    }
}

impl Picker for Named {
    fn name(&self) -> &'static str {
        "named"
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

/// A decider's thread acts as its decider while it decides, and a
/// lease's event is told under the name of the decider whose file holds
/// the lease, whichever thread applies the frame.
#[tokio::test]
async fn a_decision_and_its_leases_events_act_as_the_leases_decider() {
    let named = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&named);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("named", move || Box::new(Named(Arc::clone(&kept))))
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "named".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    let mut bed = bed(Some(spec), true, (1, 8), []);
    let answer = bed.ask(frame(1, 1)).await;
    // The case's thread names no decider: its hand applies the release,
    // or the decider does as it leaves its work.
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    bed.store.hand(&hand, LeaseId(lease_of(&answer).id), gone);
    settled(|| named.lock().len() >= 2).await;
    assert_eq!(*named.lock(), [Some(DecisionSlot::new(0)); 2]);
    assert_eq!(DecisionSlot::acting(), None);
}

/// The `random` picker, keeping the name of the thread each release of
/// a lease is told on.
#[derive(Debug)]
struct Threads(Arc<Mutex<Vec<Option<String>>>>);

impl Lifecycle for Threads {
    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        let name = thread::current().name().map(str::to_owned);
        self.0.lock().push(name);
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Released])
    }
}

impl Picker for Threads {
    fn name(&self) -> &'static str {
        "threads"
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

/// A decider that waits for a job with a lease on its file keeps its
/// watch: a frame handed on the lull is left in the ring, and the
/// decider's own thread applies it within its look. The hand's thread
/// applies none. Once no lease stands on the file the watch ends, and
/// the decider wakes for a job alone.
#[tokio::test]
async fn a_frame_handed_on_a_lull_is_applied_by_its_leases_decider() {
    let threads = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&threads);
    let spec = picking("threads", move || Box::new(Threads(Arc::clone(&kept))));
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let taker = bed.deciders.of(MODEL);
    let answers = [bed.ask(frame(1, 1)).await, bed.ask(frame(2, 1)).await];
    // Two leases stand on its file: its look behind the answers
    // starts its watch, and it waits for a job from there.
    watching(&bed, 1).await;
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    for answer in &answers {
        bed.store.hand(&hand, LeaseId(lease_of(answer).id), gone);
    }
    settled(|| threads.lock().len() >= 2).await;
    let decider = || Some(format!("decide-{taker}"));
    assert_eq!(*threads.lock(), [decider(), decider()]);
    // No lease stands on the file: the decider's next look ends its
    // watch.
    watching(&bed, 0).await;
}

/// A controller's only decider keeps no watch. With leases on its
/// file and no job it rests, so a frame handed on the lull is applied
/// before its hand returns: by the hand in place, or by the decider's
/// last take where the hand meets its rest.
#[tokio::test]
async fn a_controllers_only_decider_keeps_no_watch() {
    let threads = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&threads);
    let spec = picking("threads", move || Box::new(Threads(Arc::clone(&kept))));
    let mut bed = bed(Some(spec), true, (1, 8), []);
    let answers = [bed.ask(frame(1, 1)).await, bed.ask(frame(2, 1)).await];
    // The decider leaves its work behind its last answer: the case
    // waits until the store shows no thread at work.
    settled(|| bed.store.working() == 0).await;
    assert_eq!(bed.store.watches(), 0, "two leases stand on its file");
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    for (applied, answer) in answers.iter().enumerate() {
        bed.store.hand(&hand, LeaseId(lease_of(answer).id), gone);
        let told = threads.lock().len();
        assert_eq!(told, applied + 1, "applied before its hand returned");
    }
    assert_eq!(bed.store.watches(), 0, "no look is owed");
}

/// Waits until `files` of the bed's files show a watch. A decider
/// starts its watch at its look, behind its answer, and ends it at a
/// look: a read right behind an answer may come first.
async fn watching(bed: &Bed, files: usize) {
    settled(|| bed.store.watches() == files).await;
}

/// A decider that waits with a lease on file waits its floor behind a
/// job, twice as long behind each look that applied no frame, up to
/// its cap, and its floor again behind a look that applied one: a
/// file whose frames flow is not quiet. Once no lease stands on the
/// file it waits with no bound. The decider is the case's own, and
/// the case's receive answers each wait as a queue with no job would.
#[test]
fn a_watching_deciders_wait_doubles_over_a_quiet_file_and_falls_at_a_frame() {
    let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
    let mut decider = deciding(&bed);
    decider.working = Some(bed.store.at_work());
    let answer = decider.answer(checked(frame(1, 1)), GATEWAY, bed.script.taken);
    let lease = LeaseId(lease_of(&answer.frame()).id);
    let hand = Hand::new(GATEWAY);
    let mut waits = Vec::new();
    let job = decider.wait_by(|within| {
        waits.push(within.map(|period| period.as_millis()));
        match waits.len() {
            // Handed in the seventh wait: the look behind it applies
            // a frame.
            7 => {
                let dispatch = Said::Dispatched(Dispatch::default(), bed.script.taken);
                bed.store.hand(&hand, lease, dispatch);
            }
            // Handed in the ninth: the look behind it applies the
            // release of the file's last lease.
            9 => {
                let gone = Said::Released(None, ReleaseReason::ClientGone);
                bed.store.hand(&hand, lease, gone);
            }
            10 => return Err(std_mpsc::RecvTimeoutError::Disconnected),
            _ => {}
        }
        Err(std_mpsc::RecvTimeoutError::Timeout)
    });
    assert!(job.is_none(), "the case's queue ends with no job");
    // Seven waits over the quiet file, two behind the frame, and no
    // bound behind the release.
    let mut wanted: Vec<_> = [1, 2, 4, 8, 16, 16, 16, 1, 2].map(Some).into();
    wanted.push(None);
    assert_eq!(waits, wanted);
    assert_eq!((LOOK_FLOOR.as_millis(), LOOK_CAP.as_millis()), (1, 16));
}

/// A decider with no place yet keeps no watch: before its first job
/// its wait is handed no period, as over a file that holds no lease.
#[test]
fn a_decider_with_no_place_yet_waits_with_no_bound() {
    let bed = bed(Some(preset("random", 1)), true, (1, 8), []);
    let mut decider = deciding(&bed);
    let mut waits = Vec::new();
    let job = decider.wait_by(|within| {
        waits.push(within);
        Err(std_mpsc::RecvTimeoutError::Disconnected)
    });
    assert!(job.is_none(), "the case's queue ends with no job");
    assert_eq!(waits, [None], "no place, so no watch and no period");
    assert_eq!(bed.store.watches(), 0);
}

/// A decision under a stage that may refuse by a report of leases
/// reads a completion its gateway reported of a lease on another
/// decider's file. That decider waits for a job with its watch over
/// the file, so the frames were left in its ring; the deciding thread
/// applies them before it reads its rows. The two models are taken by
/// two deciders and served by the same members.
#[tokio::test]
async fn a_decision_whose_stage_may_refuse_by_reports_applies_what_waits_at_the_other_files() {
    use crate::service::controller::lifecycle::tests::{a_completed, a_dispatched};

    let rows = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(seeing(&rows, Counted)), true, (2, 8), []);
    let models = a_model_each(&bed.deciders, 2);
    let served: Vec<&str> = models.iter().map(String::as_str).collect();
    for name in ["s0", "s1", "s2"] {
        serving(
            &bed.store,
            &bed.reads.objects,
            name,
            &served,
            Role::Single,
            WireModes::NONE,
        );
    }
    let hand = Hand::new(GATEWAY);
    let answer = bed.ask(frame_for(&models[0], 1, 1)).await;
    let lease = lease_of(&answer).id;
    // The first model's decider waits for a job on its watch over the
    // lease.
    watching(&bed, 1).await;
    let at = Instant::now();
    lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease), at).expect("the frame reads");
    lifecycle::completed(&bed.store, &hand, &a_completed(lease, picked(&answer)))
        .expect("the frame reads");
    // The other model's decision, at the other file, right behind the
    // completion: it reads no request in flight.
    bed.ask(frame_for(&models[1], 2, 1)).await;
    let rows = rows.lock();
    let read: u32 = rows[1].iter().map(|(count, _)| count).sum();
    assert_eq!(read, 0, "{rows:?}");
}
