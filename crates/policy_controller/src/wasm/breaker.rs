//! The plugin's breaker: traps, invalid answers and the budget hits a plugin
//! ran out of under its own bounds count as failures within a window; a
//! moving p50 past half the decision's budget or a p99 past it (one sample
//! per decision: the plugin's own charge in it, its constructor and stage
//! calls whatever ended them and a lazy `init` the decision paid, over the
//! decision's budget) counts as slowness; either opens the breaker for a
//! cool-down, during which every stage of the plugin takes its fallback
//! without a call. A report made during the cool-down, by a call admitted
//! before the open, is dropped: one incident is one opening. One breaker
//! per plugin, shared by its instances; the latency ring is per thread,
//! kept across the thread's rebuilds (no shared write per call).

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use super::config::BreakerSpec;

/// The failure window: when it started, on the runtime's clock, and the
/// failures counted in it. With none counted there is no window: the next
/// failure starts one at its own instant.
#[derive(Debug, Default)]
struct Window {
    start_micros: u64,
    failures: u32,
}

#[derive(Debug)]
pub(crate) struct Breaker {
    spec: BreakerSpec,
    /// Read and written in one step, under its lock, so a failure that
    /// starts a window and one that counts in it never interleave. A failure
    /// is a fault, rare beside a call, and a call reads `open_until` alone.
    window: Mutex<Window>,
    /// Microseconds on the runtime's clock until which the breaker is open;
    /// `0` when closed.
    open_until: AtomicU64,
    opens: AtomicU64,
}

impl Breaker {
    pub(crate) fn new(spec: BreakerSpec) -> Self {
        Self {
            spec,
            window: Mutex::new(Window::default()),
            open_until: AtomicU64::new(0),
            opens: AtomicU64::new(0),
        }
    }

    /// Whether the plugin is bypassed at `now`. A cool-down that has passed
    /// closes the breaker on this read: the calls resume, and the failure
    /// window counts afresh from the next failure.
    pub(crate) fn is_open(&self, now_micros: u64) -> bool {
        let until = self.open_until.load(Ordering::Relaxed);
        if until == 0 {
            return false;
        }
        if now_micros < until {
            return true;
        }
        let _ = self
            .open_until
            .compare_exchange(until, 0, Ordering::Relaxed, Ordering::Relaxed);
        false
    }

    /// One failure at `now`: opens the breaker at the window's threshold,
    /// and says whether this one opened it. A failure reported while the
    /// breaker is open, by a call admitted before it opened, is dropped: it
    /// counts in no window and opens nothing again.
    pub(crate) fn failure(&self, now_micros: u64) -> bool {
        let mut window = self.window.lock();
        if now_micros < self.open_until.load(Ordering::Relaxed) {
            return false;
        }
        if window.failures == 0
            || now_micros.saturating_sub(window.start_micros) > self.spec.window_micros
        {
            // A new window: the first failure in it, whether none was
            // counted (the first ever, the first after an open) or the last
            // window ran out. Two threads that race here both count their
            // failure in the new window: the second to take the lock finds
            // the window the first started.
            window.start_micros = now_micros;
            window.failures = 0;
        }
        window.failures = window.failures.saturating_add(1);
        let opened = window.failures >= self.spec.failures;
        if opened {
            self.open(&mut window, now_micros);
        }
        opened
    }

    /// The latency rule tripped: opens at once, and says whether it did. A
    /// slow window reported while the breaker is open, completed by
    /// decisions in flight at the open, is dropped like a failure.
    pub(crate) fn slow(&self, now_micros: u64) -> bool {
        let mut window = self.window.lock();
        if now_micros < self.open_until.load(Ordering::Relaxed) {
            return false;
        }
        self.open(&mut window, now_micros);
        true
    }

    /// Opens for the cool-down and forgets the window's count, so the next
    /// failure starts a window of its own. The caller holds the window, so
    /// one crossing of the threshold is one open.
    fn open(&self, window: &mut Window, now_micros: u64) {
        self.open_until.store(
            now_micros.saturating_add(self.spec.cooldown_micros).max(1),
            Ordering::Relaxed,
        );
        window.failures = 0;
        self.opens.fetch_add(1, Ordering::Relaxed);
    }

    /// How many times the breaker opened.
    pub(crate) fn opens(&self) -> u64 {
        self.opens.load(Ordering::Relaxed)
    }

    /// How long an opening lasts.
    pub(crate) fn cooldown_micros(&self) -> u64 {
        self.spec.cooldown_micros
    }
}

/// The last [`Latency::SAMPLES`] decisions of one plugin on one thread, read
/// every [`Latency::EVERY`] samples, each the plugin's charge in the decision
/// as a share of its budget (in thousandths): a p50 past half the budget or
/// a p99 past it is slow. The p99 of a window is its second-slowest sample,
/// so one outlier does not open the breaker, and a window that read slow is
/// forgotten, so it does not open it twice.
#[derive(Debug)]
pub(crate) struct Latency {
    ring: [u32; Self::SAMPLES],
    next: usize,
    filled: usize,
}

impl Latency {
    pub(crate) const SAMPLES: usize = 128;
    pub(crate) const EVERY: usize = 64;
    /// The frame, in the ring's unit.
    const FRAME: u32 = 1_000;

    pub(crate) const fn new() -> Self {
        Self {
            ring: [0; Self::SAMPLES],
            next: 0,
            filled: 0,
        }
    }

    /// Records one call of `micros` under a frame of `frame_micros` (a call
    /// without a frame is not a sample); `true` when this call completed a
    /// window and the window reads slow, which forgets the window.
    pub(crate) fn record(&mut self, micros: u32, frame_micros: u32) -> bool {
        if frame_micros == 0 {
            return false;
        }
        let share = u64::from(micros) * u64::from(Self::FRAME) / u64::from(frame_micros);
        self.ring[self.next] = u32::try_from(share).unwrap_or(u32::MAX);
        self.next = (self.next + 1) % Self::SAMPLES;
        self.filled = (self.filled + 1).min(Self::SAMPLES);
        if !self.next.is_multiple_of(Self::EVERY) || self.filled < 2 {
            return false;
        }
        let mut sorted = [0u32; Self::SAMPLES];
        sorted[..self.filled].copy_from_slice(&self.ring[..self.filled]);
        let window = &mut sorted[..self.filled];
        window.sort_unstable();
        // Nearest rank for the median: the ceil(n / 2)-th smallest. The tail
        // is the window's second-slowest call, so one outlier is not it.
        let p50 = window[(self.filled.div_ceil(2)).saturating_sub(1)];
        let p99 = window[self.filled - 2];
        let slow = p50 > Self::FRAME / 2 || p99 > Self::FRAME;
        if slow {
            self.next = 0;
            self.filled = 0;
        }
        slow
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{Breaker, Latency};
    use crate::wasm::config::BreakerSpec;

    fn spec() -> BreakerSpec {
        BreakerSpec {
            failures: 3,
            window_micros: 1_000,
            cooldown_micros: 5_000,
        }
    }

    #[test]
    fn the_threshold_of_failures_within_a_window_opens_for_the_cooldown() {
        let b = Breaker::new(spec());
        b.failure(10);
        b.failure(20);
        assert!(!b.is_open(30));
        b.failure(30);
        assert!(b.is_open(31));
        assert!(b.is_open(5_029));
        assert!(!b.is_open(5_030), "the cool-down passed: closed again");
        assert_eq!(b.opens(), 1);
    }

    #[test]
    fn failures_in_separate_windows_do_not_add_up() {
        let b = Breaker::new(spec());
        b.failure(0);
        b.failure(500);
        b.failure(2_000);
        assert!(!b.is_open(2_001), "the third failure opened a new window");
        b.failure(2_100);
        b.failure(2_200);
        assert!(b.is_open(2_201));
    }

    #[test]
    fn slowness_opens_at_once_and_a_failure_after_the_cooldown_reopens() {
        let b = Breaker::new(spec());
        b.slow(100);
        assert!(b.is_open(101));
        assert!(!b.is_open(6_000));
        b.failure(6_001);
        b.failure(6_002);
        b.failure(6_003);
        assert!(b.is_open(6_004));
        assert_eq!(b.opens(), 2);
    }

    /// A failure that finds none counted starts a window of its own, the
    /// first after an open like the first ever: a burst is not split at the
    /// end of a window that began before it.
    #[test]
    fn a_failure_that_finds_none_counted_starts_its_own_window() {
        // A cool-down shorter than the window.
        let spec = BreakerSpec {
            failures: 3,
            window_micros: 1_000,
            cooldown_micros: 100,
        };
        let b = Breaker::new(spec);
        // Past the clock's first window: this burst's window starts at 5,000.
        b.failure(5_000);
        b.failure(5_100);
        b.failure(5_200);
        assert!(b.is_open(5_201));
        assert!(!b.is_open(5_300), "the cool-down passed: closed again");
        // Three failures in 800 us, across the end of that window (6,000) and
        // of one counted from the open (6,200).
        b.failure(5_500);
        b.failure(5_900);
        assert!(!b.is_open(5_901));
        b.failure(6_300);
        assert!(
            b.is_open(6_301),
            "a burst after the open is counted in a window of its own"
        );
        assert_eq!(b.opens(), 2);

        // The first failure ever, inside the clock's first window: its window
        // starts at the failure, not at the clock's origin.
        let fresh = Breaker::new(spec);
        fresh.failure(600);
        fresh.failure(900);
        fresh.failure(1_100);
        assert!(
            fresh.is_open(1_101),
            "the first burst is counted in a window of its own"
        );
    }

    /// A failure reported while the breaker is open, by a call admitted
    /// before it opened, counts in no window and opens nothing again.
    #[test]
    fn a_failure_reported_while_open_neither_counts_nor_opens() {
        // A cool-down shorter than the window.
        let b = Breaker::new(BreakerSpec {
            failures: 3,
            window_micros: 1_000,
            cooldown_micros: 100,
        });
        b.failure(0);
        b.failure(0);
        b.failure(0);
        assert!(b.is_open(1));
        // Three calls already out at the open fail during the cool-down.
        b.failure(10);
        b.failure(20);
        b.failure(30);
        assert_eq!(b.opens(), 1, "a report while open opens nothing again");
        assert!(!b.is_open(100), "the cool-down was not restarted");
        // The count starts afresh after the cool-down: one failure is the
        // first of three, not the third.
        b.failure(110);
        assert!(
            !b.is_open(111),
            "the reports made while open were not counted"
        );
        b.failure(120);
        b.failure(130);
        assert!(b.is_open(131));
        assert_eq!(b.opens(), 2);
    }

    /// A slow window reported while the breaker is open, completed by
    /// decisions in flight at the open, opens nothing again.
    #[test]
    fn a_slow_window_reported_while_open_does_not_open_again() {
        let b = Breaker::new(spec());
        b.slow(100);
        b.slow(2_000);
        assert_eq!(b.opens(), 1);
        assert!(!b.is_open(5_100), "the cool-down was not restarted");
    }

    #[test]
    fn a_report_says_whether_it_opened_the_breaker() {
        let b = Breaker::new(spec());
        assert!(!b.failure(10));
        assert!(!b.failure(20));
        assert!(b.failure(30), "the third failure opened it");
        assert!(!b.failure(40), "a report while open opened nothing");
        assert!(!b.slow(50), "nor did a slow window");
        assert!(b.slow(5_030), "after the cool-down a slow window opens");
        // The same edge for a failure: the cool-down's last instant is
        // inside it, its end is not.
        let one = Breaker::new(BreakerSpec {
            failures: 1,
            window_micros: 1_000,
            cooldown_micros: 100,
        });
        assert!(one.failure(0));
        assert!(!one.failure(99), "the cool-down's last instant");
        assert!(one.failure(100), "the cool-down has passed");
    }

    /// Spins, then yields, until `ready`.
    fn wait(ready: impl Fn() -> bool) {
        let mut spins = 0_u32;
        while !ready() {
            spins += 1;
            if spins < 128 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// `racers` threads fail at the same instant past the window of a
    /// breaker that failed once, under a threshold of `failures`, round
    /// after round: the breakers as the rounds left them.
    fn race(racers: usize, failures: u32) -> Vec<Breaker> {
        const ROUNDS: usize = 4_000;
        let spec = BreakerSpec {
            failures,
            window_micros: 10,
            cooldown_micros: 1_000,
        };
        let breakers: Vec<Breaker> = (0..ROUNDS)
            .map(|_| {
                let breaker = Breaker::new(spec);
                // One failure: its window, ten microseconds from zero, has
                // run out by the instant of the race.
                breaker.failure(0);
                breaker
            })
            .collect();
        // The rounds started so far, and the failures reported in all.
        let started = AtomicUsize::new(0);
        let reported = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..racers {
                scope.spawn(|| {
                    for (round, breaker) in breakers.iter().enumerate() {
                        wait(|| started.load(Ordering::Acquire) > round);
                        // Past the window the breaker's one failure started.
                        breaker.failure(1_000);
                        reported.fetch_add(1, Ordering::Release);
                    }
                });
            }
            for round in 0..ROUNDS {
                started.store(round + 1, Ordering::Release);
                wait(|| reported.load(Ordering::Acquire) == racers * (round + 1));
            }
        });
        breakers
    }

    /// Two failures that race a window's rollover both count: under a
    /// threshold of two the breaker opens in every round.
    #[test]
    fn two_failures_racing_a_window_rollover_both_count() {
        let breakers = race(2, 2);
        let closed = breakers.iter().filter(|b| !b.is_open(1_001)).count();
        assert_eq!(
            closed,
            0,
            "{closed} of {} rollovers lost a failure",
            breakers.len()
        );
    }

    /// Three failures at once under a threshold of two cross it once: the
    /// breaker opens, and one open is counted.
    #[test]
    fn failures_racing_past_the_threshold_open_once() {
        let breakers = race(3, 2);
        let wrong = breakers.iter().filter(|b| b.opens() != 1).count();
        assert_eq!(
            wrong,
            0,
            "{wrong} of {} rounds did not open once",
            breakers.len()
        );
    }

    #[test]
    fn the_latency_window_reads_slow_on_p50_or_p99_past_the_frame() {
        let mut fast = Latency::new();
        let mut tripped = false;
        for _ in 0..Latency::SAMPLES {
            tripped |= fast.record(10, 100);
        }
        assert!(!tripped, "10 us calls under a 100 us frame are not slow");

        let mut half_slow = Latency::new();
        let mut tripped = false;
        for i in 0..Latency::SAMPLES {
            // One more than half the calls past half the frame: the median
            // lands on them.
            let micros = if i % 2 == 0 || i == 1 { 60 } else { 10 };
            tripped |= half_slow.record(micros, 100);
        }
        assert!(tripped, "a p50 past half the frame is slow");

        let mut one_outlier = Latency::new();
        let mut tripped = false;
        for i in 0..Latency::SAMPLES {
            let micros = if i == 5 { 500 } else { 10 };
            tripped |= one_outlier.record(micros, 100);
        }
        assert!(
            !tripped,
            "one call past the frame is not the p99 of the ring"
        );

        let mut tail = Latency::new();
        let mut tripped = false;
        for i in 0..Latency::SAMPLES {
            let micros = if i == 5 || i == 9 { 500 } else { 10 };
            tripped |= tail.record(micros, 100);
        }
        assert!(tripped, "two calls past the frame are the p99");
        let mut tripped_again = false;
        for _ in 0..Latency::SAMPLES {
            tripped_again |= tail.record(10, 100);
        }
        assert!(!tripped_again, "the window that read slow is forgotten");

        let mut mixed = Latency::new();
        let mut tripped = false;
        for i in 0..Latency::SAMPLES {
            // Sixty microseconds is slow under a 100 us frame and nothing
            // under a 1,000 us one: each call against its own frame.
            let frame = if i % 2 == 0 { 1_000 } else { 100 };
            tripped |= mixed.record(if frame == 1_000 { 60 } else { 10 }, frame);
        }
        assert!(!tripped, "each sample is judged against its own frame");

        let mut unbudgeted = Latency::new();
        assert!(!(0..Latency::SAMPLES).any(|_| unbudgeted.record(1_000, 0)));
    }
}
