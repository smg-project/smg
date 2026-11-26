use std::{sync::atomic::AtomicBool, thread};

use super::*;

#[cfg(feature = "controller")]
impl SerialTurn {
    /// Every place left, taken for a case.
    pub(crate) fn fill(&self) -> Vec<impl Send + use<>> {
        std::iter::from_fn(|| self.enter::<fn()>(None)).collect()
    }

    /// One place, taken for a case.
    pub(crate) fn place(&self) -> Option<impl Send + use<>> {
        self.enter::<fn()>(None)
    }

    /// The turn by itself, so a case can hold it while callers enter.
    pub(crate) async fn hold(&self) -> tokio::sync::OwnedMutexGuard<()> {
        Arc::clone(&self.turn).lock_owned().await
    }

    pub(crate) fn available(&self) -> usize {
        self.waiting + 1 - self.let_in()
    }

    pub(crate) fn is_held(&self) -> bool {
        self.turn.try_lock().is_err()
    }
}

/// Of calls that leave together one alone is the last out: a place
/// goes back and the count is read in one step. Two places go back
/// from two threads at one instant, round after round.
#[test]
fn of_calls_that_leave_together_one_alone_is_the_last_out() {
    let turn = SerialTurn::new(32);
    let outs = Arc::new(AtomicUsize::new(0));
    for round in 1..=2_000 {
        let (go, ready) = (AtomicBool::new(false), AtomicUsize::new(0));
        thread::scope(|scope| {
            for _ in 0..2 {
                let outs = Arc::clone(&outs);
                let counted = move || {
                    outs.fetch_add(1, Ordering::SeqCst);
                };
                let place = turn.enter(Some(counted)).expect("a place is free");
                let (go, ready) = (&go, &ready);
                scope.spawn(move || {
                    ready.fetch_add(1, Ordering::SeqCst);
                    while !go.load(Ordering::SeqCst) {
                        std::hint::spin_loop();
                    }
                    drop(place);
                });
            }
            while ready.load(Ordering::SeqCst) < 2 {
                thread::yield_now();
            }
            go.store(true, Ordering::SeqCst);
        });
        assert_eq!(turn.let_in(), 0, "each place went back");
        assert_eq!(
            outs.load(Ordering::SeqCst),
            round,
            "one last call out a round"
        );
    }
}
