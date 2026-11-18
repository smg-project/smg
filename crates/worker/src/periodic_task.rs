use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use tracing::debug;

#[derive(Debug)]
pub struct PeriodicTask {
    debug_name: &'static str,
    shutdown_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl PeriodicTask {
    /// Spawn a background thread that periodically executes a task.
    pub fn spawn<F>(interval_secs: u64, debug_name: &'static str, task: F) -> Self
    where
        F: Fn() + Send + 'static,
    {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown_flag);

        let handle = thread::spawn(move || {
            let interval = Duration::from_secs(interval_secs);

            loop {
                // One timed wait per interval: the thread parks until the deadline. A
                // shutdown stores the flag and unparks it (an unpark before the park
                // leaves the token, so the park returns at once); a spurious wake-up
                // parks again for what is left of the interval. An interval the clock
                // cannot reach (the configuration does not bound it) has no deadline:
                // the thread parks until shutdown and the task never runs, as before.
                let deadline = Instant::now().checked_add(interval);
                loop {
                    if shutdown_clone.load(Ordering::Acquire) {
                        debug!("{} thread received shutdown signal", debug_name);
                        return;
                    }
                    match deadline {
                        Some(deadline) => {
                            let now = Instant::now();
                            if now >= deadline {
                                break;
                            }
                            thread::park_timeout(deadline - now);
                        }
                        None => thread::park(),
                    }
                }

                task();
            }
        });

        Self {
            debug_name,
            shutdown_flag,
            handle: Some(handle),
        }
    }
}

impl Drop for PeriodicTask {
    fn drop(&mut self) {
        self.shutdown_flag.store(true, Ordering::Release);

        if let Some(handle) = self.handle.take() {
            // The thread may be parked until its deadline: the unpark ends the wait now.
            handle.thread().unpark();
            // A task closure that (transitively) owns its own PeriodicTask —
            // e.g. by holding the last Arc to the struct storing it — would
            // run this drop on the task's own thread; joining would deadlock
            // (or error) on self-join. Detach instead: the shutdown flag is
            // already set, so the thread exits right after this drop returns.
            if handle.thread().id() == thread::current().id() {
                debug!(
                    "{} dropped from its own thread; detaching instead of self-joining",
                    self.debug_name
                );
                return;
            }
            match handle.join() {
                Ok(()) => debug!("{} thread shut down cleanly", self.debug_name),
                Err(_) => debug!("{} thread panicked during shutdown", self.debug_name),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicUsize, mpsc};

    use super::*;

    #[test]
    fn test_periodic_task_executes() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let _task = PeriodicTask::spawn(1, "test", move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Wait for at least one execution
        thread::sleep(Duration::from_millis(1200));
        assert!(counter.load(Ordering::SeqCst) >= 1);

        // Task will be stopped on drop
    }

    /// Drops `task` on a helper thread and waits for the drop under a bound
    /// far above any host pause: only a wait the shutdown did not wake
    /// reaches it, and that fails the case by name instead of hanging it.
    fn assert_the_drop_returns(task: PeriodicTask) {
        let (dropped, done) = mpsc::channel();
        thread::spawn(move || {
            drop(task);
            let _ = dropped.send(());
        });
        done.recv_timeout(Duration::from_secs(30))
            .expect("the drop woke the task's wait");
    }

    /// The drop ends a timed wait instead of waiting it out. The interval is
    /// a day, which no test run reaches, so the drop returning is the proof.
    #[test]
    fn test_periodic_task_responds_to_shutdown() {
        let task = PeriodicTask::spawn(24 * 60 * 60, "test", || {});
        // Let the task thread reach its timed wait, so the drop has to wake it.
        thread::sleep(Duration::from_millis(100));
        assert_the_drop_returns(task);
    }

    /// An interval the clock cannot reach never fires and never panics: the task
    /// thread parks until shutdown, and the drop wakes it. The thread has no
    /// deadline, so the drop returning is the proof.
    #[test]
    fn test_periodic_task_with_an_unreachable_interval_parks_until_shutdown() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let task = PeriodicTask::spawn(u64::MAX, "test", move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        thread::sleep(Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        assert_the_drop_returns(task);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    /// The wait form's cost is its wake-ups: a parked thread wakes once per
    /// interval (plus the one at shutdown), where the 100 ms slice form woke ten
    /// times a second. Every wake-up of a sleeping thread is one voluntary
    /// context switch in the task thread's own scheduler counters.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_periodic_task_wakes_once_per_interval() {
        use std::{
            path::{Path, PathBuf},
            sync::Mutex,
        };

        let task_dir: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let task_dir_clone = Arc::clone(&task_dir);
        let _task = PeriodicTask::spawn(1, "test", move || {
            // /proc/thread-self resolves to the running thread's task directory.
            if let Ok(p) = std::fs::read_link("/proc/thread-self") {
                *task_dir_clone.lock().unwrap() = Some(Path::new("/proc").join(p));
            }
        });

        // The first run records the task thread's directory.
        let start = Instant::now();
        let dir = loop {
            if let Some(d) = task_dir.lock().unwrap().clone() {
                break d;
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "the task never ran"
            );
            thread::sleep(Duration::from_millis(20));
        };
        let switches = || -> u64 {
            let status =
                std::fs::read_to_string(dir.join("status")).expect("the task thread's status");
            status
                .lines()
                .find_map(|l| l.strip_prefix("voluntary_ctxt_switches:"))
                .and_then(|v| v.trim().parse().ok())
                .expect("voluntary_ctxt_switches in the task thread's status")
        };

        let before = switches();
        thread::sleep(Duration::from_millis(2500));
        let wake_ups = switches() - before;

        // Two or three interval wake-ups in 2.5 s; the slice form made about 25.
        assert!(
            wake_ups <= 6,
            "the task thread woke {wake_ups} times in 2.5 s"
        );
    }
}
