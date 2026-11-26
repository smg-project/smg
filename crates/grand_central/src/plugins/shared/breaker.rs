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

/// What the ring did with one call ([`Latency::record`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Recorded {
    /// The call had no frame: it is no sample, and the ring is as it was.
    Nothing,
    /// The call is a sample, and the ring took it.
    Sample,
    /// The call is a sample that completed a window, and the window reads
    /// slow: the ring took the sample and forgot the window.
    SlowWindow,
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
    /// call adds its plugin's frame whole, whatever the decision had left
    /// for the call: a plugin that others' draws starved is not read
    /// against the little it was left. A frame may be wider than the
    /// budget, and the frames are cut at it here, once. Every kind of
    /// plugin asks here.
    ///
    /// A call is cut at its frame, so under a budget wider than a
    /// decision's frames a plugin's charge would pass no mark of the budget:
    /// a request in process may state none, and a stage is then shown the
    /// widest. So a plugin that answers in time reads slow too where its
    /// median takes over half of its own frames: a budget of twice its usual
    /// time keeps it under the mark. A guest's constructor is a call and adds
    /// its frame, so a guest bound as one stage is read over two frames.
    pub(crate) fn frame(seed_micros: u64, frames_micros: u64) -> u32 {
        u32::try_from(seed_micros.min(frames_micros)).unwrap_or(u32::MAX)
    }

    /// What a sample is: a charge of `micros` as a share of a frame of
    /// `frame_micros`, in the ring's unit. A call without a frame is none.
    /// Whoever asks whether a call was a sample reads what
    /// [`Latency::record`] says of it.
    fn share(micros: u32, frame_micros: u32) -> Option<u32> {
        if frame_micros == 0 {
            return None;
        }
        let share = u64::from(micros) * u64::from(Self::FRAME) / u64::from(frame_micros);
        Some(u32::try_from(share).unwrap_or(u32::MAX))
    }

    /// How many of the two marks a read holds a window to a sample of
    /// `share` is past: half the frame, where the median reads slow, and
    /// the frame, where the tail does.
    fn marks(share: u32) -> u8 {
        u8::from(share > Self::FRAME / 2) + u8::from(share > Self::FRAME)
    }

    /// Whether a mark stands between two charges under a frame of
    /// `frame_micros`: one is past a mark that the other is short of, so a
    /// read holds the two differently. Without a frame neither is a sample,
    /// and no mark stands between them. Only the WebAssembly kind asks, of
    /// the two ends of a time nothing read: no other kind arms a thread.
    #[cfg(feature = "wasm")]
    pub(crate) fn mark_between(micros: u32, other_micros: u32, frame_micros: u32) -> bool {
        Self::share(micros, frame_micros).map(Self::marks)
            != Self::share(other_micros, frame_micros).map(Self::marks)
    }

    /// Records one call of `micros` under a frame of `frame_micros`, when
    /// it is a sample, and says what it did with the call: nothing where
    /// the call has no frame, and a slow window where this call completed
    /// a window and the window reads slow, which forgets the window.
    pub(crate) fn record(&mut self, micros: u32, frame_micros: u32) -> Recorded {
        let Some(share) = Self::share(micros, frame_micros) else {
            return Recorded::Nothing;
        };
        self.ring[self.next] = share;
        self.next = (self.next + 1) % Self::SAMPLES;
        self.filled = (self.filled + 1).min(Self::SAMPLES);
        if !self.next.is_multiple_of(Self::EVERY) || self.filled < 2 {
            return Recorded::Sample;
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
        if !slow {
            return Recorded::Sample;
        }
        self.next = 0;
        self.filled = 0;
        Recorded::SlowWindow
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/shared/breaker/tests.rs"]
mod tests;
