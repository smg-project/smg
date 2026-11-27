//! The measurements of the deciders: what an idle watch costs, the lag of a
//! frame handed on a lull, and a closed loop of decisions.

use super::*;

/// The `random` picker, keeping the instant each release of a lease is
/// told at.
#[derive(Debug)]
struct Heard(Arc<Mutex<Vec<Instant>>>);

impl Lifecycle for Heard {
    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        self.0.lock().push(Instant::now());
    }
}

impl Picker for Heard {
    fn name(&self) -> &'static str {
        "heard"
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

/// The idle cost of a watch, for its measurement at `deciders`
/// deciders: one dispatched lease on each decider's file and no job,
/// for five seconds between two printed lines. A reader of the
/// threads named `decide-*` counts their wakes and their time on a
/// core over the pause. The tenth of a second before the first line
/// is not counted: each decider's wait grows to its cap in it.
async fn idle(deciders: usize) {
    use std::io::Write as _;

    use crate::service::controller::lifecycle::tests::a_dispatched;

    let mut out = io::stdout();
    let mut bed = bed(Some(preset("random", 1)), true, (deciders, 8), []);
    let models = a_model_each(&bed.deciders, deciders);
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
    for model in &models {
        let answer = bed.ask(frame_for(model, 1, 1)).await;
        let dispatched = a_dispatched(lease_of(&answer).id);
        lifecycle::dispatched(&bed.store, &hand, &dispatched, Instant::now())
            .expect("the frame reads");
    }
    thread::sleep(Duration::from_millis(100));
    writeln!(
        out,
        "the idle pause begins: a lease on each of {deciders} files, no job"
    )
    .expect("stdout");
    thread::sleep(Duration::from_secs(5));
    writeln!(out, "the idle pause ends").expect("stdout");
}

/// The idle cost of one decider's watch ([`idle`]).
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn one_idle_decider_measured() {
    idle(1).await;
}

/// The idle cost of eight deciders' watches ([`idle`]).
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn eight_idle_deciders_measured() {
    idle(8).await;
}

/// The lag of a frame handed on a lull, measured at one of two
/// deciders: from its hand to the decider's tell, over `FRAMES`
/// frames a cell. At
/// the floor each frame is handed within one shortest wait of its
/// decision. At the cap it is handed three longest waits behind its
/// decision and within one more: the decider's wait has grown to its
/// cap by then. Each is handed at a phase that sweeps that wait, so
/// no frame's instant is locked to a look.
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_lulls_lag_measured() {
    use std::io::Write as _;

    const FRAMES: usize = 500;
    let mut out = io::stdout();
    let heard = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&heard);
    let spec = picking("heard", move || Box::new(Heard(Arc::clone(&kept))));
    let mut bed = bed(Some(spec), true, (2, 8), []);
    let hand = Hand::new(GATEWAY);
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    let parts = u32::try_from(FRAMES).expect("five hundred");
    let cells = [
        ("at the floor", Duration::ZERO, LOOK_FLOOR),
        ("at the cap", 3 * LOOK_CAP, LOOK_CAP),
    ];
    let mut told = 0;
    for (cell, quiet, wait) in cells {
        let mut lags = Vec::with_capacity(FRAMES);
        for sent in 0..FRAMES {
            let answer = bed.ask(frame(1, 1)).await;
            // A step coprime to the sweep: each part of the wait once.
            let part = u32::try_from(sent * 7_919 % FRAMES).expect("under five hundred");
            thread::sleep(quiet + wait * part / parts);
            let handed = Instant::now();
            bed.store.hand(&hand, LeaseId(lease_of(&answer).id), gone);
            let at = loop {
                if let Some(at) = heard.lock().get(told) {
                    break *at;
                }
                assert!(
                    handed.elapsed() < PATIENCE,
                    "a handed frame waits one look at most"
                );
                std::hint::spin_loop();
            };
            told += 1;
            lags.push(at.saturating_duration_since(handed));
        }
        lags.sort_unstable();
        let at = |part: usize| lags[(FRAMES * part / 100).min(FRAMES - 1)].as_micros();
        writeln!(
                out,
                "the lag of a frame handed on a lull {cell}, {FRAMES} frames: p50 {} us, p99 {} us, the longest {} us",
                at(50),
                at(99),
                at(100)
            )
            .expect("stdout");
    }
}

/// One closed-loop client, measured: the share of its decisions that
/// read a count from before its last completion. The client asks,
/// names its dispatch, reports the completion `served` later and asks
/// again `think` later, so no lease of its own stands at a decision:
/// one that reads a request in flight read a count from before the
/// completion. Its requests are for one model, or for two in turn
/// that two deciders take and the same members serve: each decision
/// of the second form is made at another file than the completion
/// before it.
/// The stage that reads the counts is made by `stage`, and `under`
/// names it in the cell's lines.
async fn closed_loop<P: Picker + 'static>(
    stage: impl Fn(Seen) -> P + Send + Sync + 'static,
    under: &str,
) {
    use std::io::Write as _;

    use crate::service::controller::lifecycle::tests::{a_completed, a_dispatched};

    const ASKED: usize = 200;
    let mut out = io::stdout();
    let rows = Arc::new(Mutex::new(Vec::new()));
    let mut bed = bed(Some(seeing(&rows, stage)), true, (2, 8), []);
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
    let pause = |long: Duration| {
        if !long.is_zero() {
            thread::sleep(long);
        }
    };
    let millis = Duration::from_millis;
    let paces = [
        (millis(0), millis(0)),
        (millis(0), millis(2)),
        (millis(48), millis(0)),
        (millis(48), millis(20)),
    ];
    for (form, turn) in [("one model", [0, 0]), ("two models in turn", [0, 1])] {
        for (served, think) in paces {
            rows.lock().clear();
            for asked in 0..ASKED {
                let answer = bed.ask(frame_for(&models[turn[asked % 2]], 1, 1)).await;
                let lease = lease_of(&answer).id;
                lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease), Instant::now())
                    .expect("the frame reads");
                pause(served);
                let completed = a_completed(lease, picked(&answer));
                lifecycle::completed(&bed.store, &hand, &completed).expect("the frame reads");
                pause(think);
            }
            // The first decision of a cell stands behind the cell
            // before it: it is not counted.
            let rows = rows.lock();
            let behind = rows.iter().skip(1);
            let late = behind.filter(|read| read.iter().any(|(count, _)| *count > 0));
            writeln!(
                    out,
                    "a closed loop of {form}{under}, served {} ms, think {} ms: {} of {} decisions read a count from before the last completion",
                    served.as_millis(),
                    think.as_millis(),
                    late.count(),
                    ASKED - 1
                )
                .expect("stdout");
        }
    }
}

/// [`closed_loop`] under a stage that only picks by its counts.
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_closed_loop_measured() {
    closed_loop(|seen| seen, "").await;
}

/// [`closed_loop`] under a stage that says it may refuse by a report
/// of leases.
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_closed_loop_under_a_counting_stage_measured() {
    closed_loop(Counted, " under a stage that may refuse by a report").await;
}
