//! Bounded admission to a serial turn on the caller's threads.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use tokio::sync::Mutex;

/// Calls let in under a bound, one at a time. A call past the bound is
/// refused at once. The last call out is read in the step that gives its
/// place back.
#[derive(Debug)]
pub struct SerialTurn {
    waiting: usize,
    /// The calls let in: the one that runs and the ones that wait.
    let_in: Arc<AtomicUsize>,
    turn: Arc<Mutex<()>>,
}

impl SerialTurn {
    /// One running call and at most `waiting` calls awaiting their turn.
    pub fn new(waiting: usize) -> Self {
        Self {
            waiting,
            let_in: Arc::default(),
            turn: Arc::default(),
        }
    }

    /// Waits for a turn if a place is free. The returned guard keeps the
    /// turn and place until it is dropped. Dropping the wait gives back
    /// the place without running the call.
    pub async fn take(&self) -> Option<impl Send + use<>> {
        self.take_with(|| ()).await
    }

    /// A turn with a callback for the last place given back, including
    /// when a waiting caller leaves.
    pub(crate) async fn take_with<Out: FnOnce() + Send + 'static>(
        &self,
        out: Out,
    ) -> Option<impl Send + use<Out>> {
        let place = self.enter(Some(out))?;
        let turn = Arc::clone(&self.turn).lock_owned().await;
        Some((turn, place))
    }

    /// A place for one call more, or none where as many calls wait as may.
    fn enter<Out: FnOnce()>(&self, out: Option<Out>) -> Option<Place<Out>> {
        self.let_in
            .try_update(Ordering::AcqRel, Ordering::Acquire, |let_in| {
                let_in.checked_add(1).filter(|_| let_in <= self.waiting)
            })
            .ok()?;
        Some(Place {
            let_in: Arc::clone(&self.let_in),
            out,
        })
    }

    /// The calls let in at this instant, including the one in its turn.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn let_in(&self) -> usize {
        self.let_in.load(Ordering::Acquire)
    }

    /// Every place left, taken for a case.
    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn fill(&self) -> Vec<impl Send + use<>> {
        std::iter::from_fn(|| self.enter::<fn()>(None)).collect()
    }

    /// One place, taken for a case.
    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn place(&self) -> Option<impl Send + use<>> {
        self.enter::<fn()>(None)
    }

    /// The turn by itself, so a case can hold it while callers enter.
    #[cfg(all(test, feature = "controller"))]
    pub(crate) async fn hold(&self) -> tokio::sync::OwnedMutexGuard<()> {
        Arc::clone(&self.turn).lock_owned().await
    }

    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn available(&self) -> usize {
        self.waiting + 1 - self.let_in()
    }

    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn is_held(&self) -> bool {
        self.turn.try_lock().is_err()
    }
}

/// A call's place in the turn. It goes back when the call ends, or
/// when its caller leaves before it began. The call whose place goes
/// back while no other call is let in is the last call out, and its `out`
/// runs then: what the calls shared is let go by the last of them. The
/// place goes back and the count is read in one step, so of calls that
/// leave together one alone is the last.
struct Place<Out: FnOnce()> {
    let_in: Arc<AtomicUsize>,
    out: Option<Out>,
}

impl<Out: FnOnce()> Drop for Place<Out> {
    fn drop(&mut self) {
        let last = self.let_in.fetch_sub(1, Ordering::AcqRel) == 1;
        if let Some(out) = self.out.take().filter(|_| last) {
            out();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicBool, thread};

    use super::*;

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
}
