//! The plugin's breaker: traps, invalid answers and the budget hits a plugin
//! ran out of under its own bounds count as failures within a window; a
//! moving p50 past half of the frames the plugin was given in a decision, at
//! most the decision's budget, or a p99 past them (one sample per decision:
//! the plugin's own charge in it, its constructor and stage calls whatever
//! ended them and a lazy `init` the decision paid, over those frames:
//! [`Latency::frame`]) counts as slowness; either opens the breaker for a
//! cool-down, during which every stage of the plugin takes its fallback
//! without a call. A report stamped during the cool-down, by a call admitted
//! before the open, is dropped whenever it arrives: one incident is one
//! opening. A delayed failure stamped more than one window before the latest
//! counted failure is also dropped. One breaker per plugin, shared by its
//! instances; the latency ring is per thread, or with a named decider's
//! instance where a kind keeps one, kept across rebuilds (no shared write per
//! call).

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use super::config::BreakerSpec;

/// The failure window: its earliest and latest instants on the runtime's
/// clock, and the failures counted in it. With none counted there is no
/// window: the next failure starts one at its own instant.
#[derive(Debug, Default)]
struct Window {
    start_micros: u64,
    end_micros: u64,
    failures: u32,
}

#[derive(Debug)]
pub(crate) struct Breaker {
    spec: BreakerSpec,
    /// Read and written in one step, under its lock, so a failure that
    /// starts a window and one that counts in it never interleave. A failure
    /// is a fault, rare beside a call, and a call reads `open_until` alone.
    window: Mutex<Window>,
    /// Microseconds on the runtime's clock at which the last opening ends:
    /// the breaker is open before it, and a report stamped before it is
    /// dropped. It stays there once the cool-down has passed (zero before
    /// the first opening), so no read of the breaker hides it from a report
    /// that was stamped inside the cool-down and arrives after its end.
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

    /// Whether the plugin is bypassed at `now`: before the last opening's
    /// end. Past it the calls resume, and the failure window counts afresh
    /// from the next failure (the opening forgot its count).
    pub(crate) fn is_open(&self, now_micros: u64) -> bool {
        now_micros < self.open_until.load(Ordering::Relaxed)
    }

    /// One failure at `now`: opens the breaker at the window's threshold,
    /// and says whether this one opened it. A failure stamped while the
    /// breaker was open, by a call admitted before it opened, is dropped
    /// whenever it arrives: it counts in no window and opens nothing again.
    /// A delayed failure stamped more than one window before the latest
    /// counted failure is also dropped because it fits no window with them.
    pub(crate) fn failure(&self, now_micros: u64) -> bool {
        let mut window = self.window.lock();
        if now_micros < self.open_until.load(Ordering::Relaxed) {
            return false;
        }
        if window.failures == 0
            || (now_micros > window.end_micros
                && now_micros - window.start_micros > self.spec.window_micros)
        {
            // A new window: the first failure in it, whether none was
            // counted (the first ever, the first after an open) or this is
            // newer than the last window. Two threads that race here both
            // count their failure in the new window when their instants fit.
            window.start_micros = now_micros;
            window.end_micros = now_micros;
            window.failures = 0;
        } else if now_micros < window.start_micros {
            if window.end_micros - now_micros > self.spec.window_micros {
                return false;
            }
            window.start_micros = now_micros;
        } else {
            window.end_micros = window.end_micros.max(now_micros);
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
            now_micros.saturating_add(self.spec.cooldown_micros),
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
/// as a share of the frames the plugin was given in it, at most the
/// decision's budget ([`Latency::frame`], in thousandths): a p50 past half
/// of them or a p99 past them is slow. The p99 of a window is its
/// second-slowest sample, so one outlier does not open the breaker, and a
/// window that read slow is forgotten, so it does not open it twice.
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

    /// The frame a decision's sample is taken over, in microseconds: the
    /// frames the plugin was given in the decision, `frames_micros` in all,
    /// at most the decision's budget as it began, `seed_micros`. A charged
    /// call is given its plugin's own frame under that budget, and the frame
    /// counts whole whatever the decision had left for the call: a plugin
    /// that others' draws starved is not read against the little it was
    /// left. A call is cut at its frame, so under a budget wider than a
    /// decision's frames a plugin's charge would pass no mark of the budget:
    /// a request in process may state none, and a stage is then shown the
    /// widest. So a plugin that answers in time reads slow too where its
    /// median takes over half of its own frames: a budget of twice its usual
    /// time keeps it under the mark. A guest's constructor is a call and adds
    /// its frame, so a guest bound as one stage is read over two frames.
    /// Every kind of plugin asks here.
    pub(crate) fn frame(seed_micros: u64, frames_micros: u64) -> u32 {
        u32::try_from(seed_micros.min(frames_micros)).unwrap_or(u32::MAX)
    }

    /// What a sample is: a charge of `micros` as a share of a frame of
    /// `frame_micros`, in the ring's unit. A call without a frame is none.
    /// Whoever asks whether a call is a sample asks here.
    pub(crate) fn share(micros: u32, frame_micros: u32) -> Option<u32> {
        if frame_micros == 0 {
            return None;
        }
        let share = u64::from(micros) * u64::from(Self::FRAME) / u64::from(frame_micros);
        Some(u32::try_from(share).unwrap_or(u32::MAX))
    }

    /// How many of the two marks a read holds a window to a sample of
    /// `share` is past: half the frame, where the median reads slow, and
    /// the frame, where the tail does.
    pub(crate) fn marks(share: u32) -> u8 {
        u8::from(share > Self::FRAME / 2) + u8::from(share > Self::FRAME)
    }

    /// Records one call of `micros` under a frame of `frame_micros`, when
    /// it is a sample ([`Latency::share`]); `true` when this call completed
    /// a window and the window reads slow, which forgets the window.
    pub(crate) fn record(&mut self, micros: u32, frame_micros: u32) -> bool {
        let Some(share) = Self::share(micros, frame_micros) else {
            return false;
        };
        self.ring[self.next] = share;
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
        let slow = Self::marks(p50) >= 1 || Self::marks(p99) == 2;
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
    use crate::plugins::shared::config::BreakerSpec;

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
    fn an_older_failure_outside_the_current_window_does_not_open() {
        let b = Breaker::new(BreakerSpec {
            failures: 2,
            window_micros: 1_000,
            cooldown_micros: 5_000,
        });
        assert!(!b.failure(2_000));
        assert!(!b.failure(0));
        assert_eq!(b.opens(), 0);
        assert!(!b.is_open(2_001));
    }

    #[test]
    fn an_older_failure_inside_the_window_counts_from_its_own_instant() {
        let b = Breaker::new(spec());
        assert!(!b.failure(1_000));
        assert!(!b.failure(500), "the older failure fits the window");
        assert!(b.failure(1_400), "three failures within 900 us open");

        let b = Breaker::new(spec());
        assert!(!b.failure(1_000));
        assert!(!b.failure(500));
        assert!(
            !b.failure(1_900),
            "a failure outside the window starts a new window"
        );
        assert_eq!(b.opens(), 0);
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

    /// A report stamped inside a cool-down is dropped whenever it arrives:
    /// after the cool-down's end too, and after another thread read the
    /// breaker closed there. The opening's end stays where it was, so the
    /// guard still finds it.
    #[test]
    fn a_report_stamped_inside_the_cooldown_is_dropped_after_a_closed_read() {
        let b = Breaker::new(BreakerSpec {
            failures: 1,
            window_micros: 1_000,
            cooldown_micros: 100,
        });
        assert!(b.failure(100), "the threshold of one opens until 200");
        // Another thread reads the breaker closed at the cool-down's end;
        // then a call that was in flight at the opening reports, with the
        // instant it read before that end.
        assert!(!b.is_open(200));
        assert!(!b.failure(199), "stamped inside the cool-down: dropped");
        assert!(!b.slow(199), "and a slow window stamped there");
        assert_eq!(b.opens(), 1, "nothing opened again");
        assert!(!b.is_open(200), "and the breaker reads closed");
        // A report stamped at the end is counted.
        assert!(b.failure(200));
        assert_eq!(b.opens(), 2);
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

        let mut at_the_frame = Latency::new();
        let mut tripped = false;
        for i in 0..Latency::SAMPLES {
            let micros = if i == 5 || i == 9 { 100 } else { 10 };
            tripped |= at_the_frame.record(micros, 100);
        }
        assert!(
            !tripped,
            "two calls at the frame, past its half, are not a slow p99"
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

    /// A sample's frame is the decision's budget or the frames of the
    /// plugin's calls in the decision, whichever is less. A request that
    /// states no budget seeds its decision with the widest one: the frames
    /// are the lesser there.
    #[test]
    fn a_samples_frame_is_the_lesser_of_the_budget_and_the_calls_frames() {
        assert_eq!(Latency::frame(1_000, 3_000), 1_000, "the budget");
        assert_eq!(Latency::frame(10_000, 3_000), 3_000, "the frames");
        let widest = u64::from(u32::MAX);
        assert_eq!(Latency::frame(widest, 300), 300, "no budget stated");
        assert_eq!(
            Latency::frame(u64::MAX, u64::MAX),
            u32::MAX,
            "past the ring's range both: the widest frame"
        );
    }
}
