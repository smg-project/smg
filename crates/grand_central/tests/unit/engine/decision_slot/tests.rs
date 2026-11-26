use std::{cell::RefCell, sync::mpsc};

use super::{DecisionSlot, DecisionSlotGuard, LIVE};

#[test]
fn a_thread_acts_as_the_decider_it_entered_until_the_guard_drops() {
    assert_eq!(DecisionSlot::acting(), None);
    let outer = DecisionSlot::new(3).enter();
    assert_eq!(DecisionSlot::acting(), Some(DecisionSlot::new(3)));
    {
        let _inner = DecisionSlot::new(7).enter();
        assert_eq!(DecisionSlot::acting().map(DecisionSlot::number), Some(7));
    }
    assert_eq!(
        DecisionSlot::acting(),
        Some(DecisionSlot::new(3)),
        "the earlier name again"
    );
    drop(outer);
    assert_eq!(DecisionSlot::acting(), None);
}

#[test]
fn a_name_is_its_threads_and_no_others() {
    let _held = DecisionSlot::new(1).enter();
    let other = std::thread::spawn(DecisionSlot::acting)
        .join()
        .expect("the other thread");
    assert_eq!(other, None);
    assert_eq!(DecisionSlot::acting(), Some(DecisionSlot::new(1)));
}

#[test]
fn a_guard_dropped_while_a_later_one_lives_leaves_the_later_name() {
    let earlier = DecisionSlot::new(1).enter();
    let later = DecisionSlot::new(2).enter();
    drop(earlier);
    assert_eq!(DecisionSlot::acting(), Some(DecisionSlot::new(2)));
    let newest = DecisionSlot::new(3).enter();
    drop(later);
    assert_eq!(DecisionSlot::acting(), Some(DecisionSlot::new(3)));
    drop(newest);
    assert_eq!(DecisionSlot::acting(), None, "no guard is left");
}

/// A drop that leaves two guards: the thread acts as the newer of the
/// two, whichever of three went.
#[test]
fn a_drop_that_leaves_two_guards_leaves_the_newer_name() {
    for gone in 0..3 {
        let mut guards: Vec<_> = (1..=3)
            .map(|name| Some(DecisionSlot::new(name).enter()))
            .collect();
        guards[gone] = None;
        let newer = if gone == 2 { 2 } else { 3 };
        assert_eq!(
            DecisionSlot::acting(),
            Some(DecisionSlot::new(newer)),
            "the guard at {gone} of three went"
        );
        drop(guards);
        assert_eq!(DecisionSlot::acting(), None, "no guard is left");
    }
}

/// A guard kept in a field and renewed by assignment: the new guard
/// enters before the old one drops, and the name holds at each renewal.
#[test]
fn a_guard_renewed_by_assignment_keeps_its_name() {
    let mut held = DecisionSlot::new(4).enter();
    for _ in 0..3 {
        held = DecisionSlot::new(4).enter();
        assert_eq!(DecisionSlot::acting(), Some(DecisionSlot::new(4)));
    }
    drop(held);
    assert_eq!(DecisionSlot::acting(), None);
}

/// A guard's number is one more than the newest live guard's, so it
/// grows by one a renewal while any guard lives, and no two live guards
/// share one. Here a guard has outlived the renewals a `u32` counts: the
/// next renewal takes a number of its own, and the old guard's drop
/// leaves the renewed name acting.
#[test]
fn a_guard_that_outlives_the_renewals_a_u32_counts_shares_no_number() {
    let old = DecisionSlot::new(1).enter();
    let mut renewed = DecisionSlot::new(2).enter();
    // What that many renewals leave: the renewed guard's number.
    renewed.number = u32::MAX.into();
    LIVE.with_borrow_mut(|live| live.last_mut().expect("two guards live").0 = renewed.number);
    let next = DecisionSlot::new(2).enter();
    drop(renewed);
    drop(old);
    assert_eq!(
        DecisionSlot::acting(),
        Some(DecisionSlot::new(2)),
        "the old guard took its own number out, not the renewed one's"
    );
    drop(next);
    assert_eq!(DecisionSlot::acting(), None, "no guard is left");
}

/// The name acting once the kept guard dropped, then the names acting
/// under and after a guard entered there: `None` where the entry
/// panicked.
type AtEndReport = (
    Option<DecisionSlot>,
    Option<(Option<DecisionSlot>, Option<DecisionSlot>)>,
);

/// A value a thread keeps to its end: it drops the guard it kept, then
/// enters and drops another, and reports what the thread acted as.
struct AtEnd {
    kept: Option<DecisionSlotGuard>,
    report: mpsc::Sender<AtEndReport>,
}

impl Drop for AtEnd {
    fn drop(&mut self) {
        drop(self.kept.take());
        let after_kept = DecisionSlot::acting();
        let entered = std::panic::catch_unwind(|| {
            let turn = DecisionSlot::new(5).enter();
            let under = DecisionSlot::acting();
            drop(turn);
            (under, DecisionSlot::acting())
        });
        let _ = self.report.send((after_kept, entered.ok()));
    }
}

thread_local! {
    static AT_END: RefCell<Option<AtEnd>> = const { RefCell::new(None) };
}

/// At a thread's end its thread-locals go one after another, and one
/// first used ahead of the thread's first guard outlives the thread's
/// list of guards. A guard it kept drops there and leaves no name
/// acting, and a guard entered there names the thread until it drops.
#[test]
fn a_guard_kept_or_entered_past_its_threads_list_names_the_thread_while_it_lives() {
    let (report, reported) = mpsc::channel();
    std::thread::spawn(move || {
        // Used first: the thread's list is gone when this value goes.
        AT_END.with_borrow_mut(|at_end| *at_end = Some(AtEnd { kept: None, report }));
        let kept = DecisionSlot::new(4).enter();
        AT_END.with_borrow_mut(|at_end| {
            at_end.as_mut().expect("set above").kept = Some(kept);
        });
        assert_eq!(DecisionSlot::acting(), Some(DecisionSlot::new(4)));
    })
    .join()
    .expect("the thread ends");
    let (after_kept, entered) = reported.recv().expect("the value's report");
    assert_eq!(after_kept, None, "the kept guard's name went with it");
    assert_eq!(
        entered,
        Some((Some(DecisionSlot::new(5)), None)),
        "a guard entered there names the thread until it drops"
    );
}
