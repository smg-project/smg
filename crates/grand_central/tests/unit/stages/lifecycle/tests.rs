use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::{
    document::{
        applied::AppliedSpec,
        spec::{PolicySpec, PRESETS},
    },
    engine::view::{Health, WorkerId},
    tracking::lease::EventKind,
};

/// A stage that counts what reached it, to prove the box forwards every
/// hook and a stateless stage needs none.
#[derive(Default)]
struct Counting(AtomicUsize);

impl Lifecycle for Counting {
    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn on_signal(&self, _: &Signal<'_>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn hears(&self) -> EventSet {
        EventSet::of([EventKind::Released, EventKind::Signal])
    }
}

struct Stateless;

impl Lifecycle for Stateless {}

/// A stage whose state the calls keep.
struct Kept;

impl Lifecycle for Kept {
    fn needs_lifecycle(&self) -> bool {
        true
    }
}

#[test]
fn a_boxed_stage_says_what_its_stage_says_of_the_calls() {
    let kept: Box<dyn Lifecycle> = Box::new(Kept);
    assert!(Lifecycle::needs_lifecycle(&kept));
    let stateless: Box<dyn Lifecycle> = Box::new(Stateless);
    assert!(!Lifecycle::needs_lifecycle(&stateless));
}

/// A stage that declares no need takes no word unless it names one,
/// and a box says what its stage says. The two answers part: a stage
/// that takes a release only to drop state early needs no call, and
/// takes the word. A stage that needs the calls and names no word takes
/// every word on a lease. A release is a word on a lease, and a signal
/// is none.
#[test]
fn a_stage_takes_the_words_it_names_and_no_other() {
    use EventKind::{Completed, Dispatched, Issued, Released};
    let stateless: Box<dyn Lifecycle> = Box::new(Stateless);
    assert_eq!(stateless.hears(), EventSet::NONE);
    let counting: Box<dyn Lifecycle> = Box::new(Counting::default());
    let ends = counting.hears();
    assert!(ends.contains(Released) && !ends.contains(Completed));
    assert!(ends.intersects(EventSet::LEASE) && !counting.needs_lifecycle());
    assert!(!EventSet::of([EventKind::Signal]).intersects(EventSet::LEASE));
    let kept: Box<dyn Lifecycle> = Box::new(Kept);
    assert!(kept.needs_lifecycle());
    assert_eq!(kept.hears(), EventSet::LEASE);
    let both = ends.with(Issued);
    assert!(both.contains(Issued) && both.contains(Released) && !both.contains(Dispatched));
    assert_eq!(both.union(EventSet::ALL), EventSet::ALL);
    assert_eq!(
        EventSet::LEASE,
        EventSet::of([Issued, Dispatched, Completed, Released])
    );
}

/// The stock policies by the words their documents take. A rotation, a
/// draw and a pass-through take none: a request under them is owed no
/// word. A pick of two and least load take the dispatch, which moves
/// their count of sends, and a bucket the issue beside it. No stock
/// stage takes an end. The session table of `manual` takes a release:
/// a commit that refused every target takes back its decision's claim.
#[test]
fn the_stock_documents_name_the_words_they_take() {
    let hears = |name: &str| {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        AppliedSpec::validate(&document)
            .expect("the preset validates")
            .engine()
            .hears()
    };
    for silent in ["round_robin", "random", "passthrough"] {
        assert_eq!(hears(silent), EventSet::NONE, "{silent}");
    }
    let sends = EventSet::of([EventKind::Dispatched]);
    for counted in ["power_of_two", "least_load"] {
        assert_eq!(hears(counted), sends, "{counted}");
    }
    let booked = EventSet::of([EventKind::Issued, EventKind::Dispatched]);
    assert_eq!(hears("bucket"), booked);
    for name in PRESETS {
        let heard = hears(name);
        assert!(!heard.contains(EventKind::Completed), "{name}");
        let released = heard.contains(EventKind::Released);
        assert_eq!(released, name == "manual", "{name}");
    }
}

#[test]
fn a_boxed_stage_forwards_its_hooks_and_a_stateless_one_ignores_them() {
    let counting = Box::new(Counting::default());
    let lease = Lease::default();
    Lifecycle::on_released(&counting, &lease, ReleaseReason::ClientGone);
    Lifecycle::on_signal(
        &counting,
        &Signal::Health {
            worker: WorkerId(1),
            health: Health::default(),
        },
    );
    Lifecycle::on_membership(&counting, &Membership::Resized(WorkerId(1)));
    assert_eq!(
            counting.0.load(Ordering::Relaxed),
            2,
            "the two implemented hooks reached the stage through the box; the membership hook is its default no-op"
        );
    let stateless: Box<dyn Lifecycle> = Box::new(Stateless);
    stateless.on_released(&lease, ReleaseReason::Abandoned);
    stateless.on_dispatched(&lease, &Target::default(), Side::Prefill, 0);
}
