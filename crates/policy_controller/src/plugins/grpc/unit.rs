//! The plugin's part in a decision and its breaker. The decision is the
//! breaker's unit: a plugin fails a decision once, however many of its opens
//! fail in it (each side of a pair, each binding), and its last close of the
//! decision takes the decision's one latency sample: what the decision
//! charged the plugin, over the frames the plugin was given in it, at most
//! the decision's budget ([`Latency::frame`]). The ring and its marks are
//! the in-process form's; no thread is ever armed here, since a remote
//! plugin's time is not the deciding thread's CPU. An opening writes the
//! one line every plugin kind writes.

use std::{
    cell::{Cell, RefCell},
    time::Instant,
};

use parking_lot::Mutex;
use thread_local::ThreadLocal;

use super::runtime::RECONNECT_PAUSE;
use crate::plugins::shared::{
    breaker::{Breaker, Latency},
    config::BreakerSpec,
    Part,
};

#[derive(Debug)]
pub(crate) struct Unit {
    /// The plugin's name, for the line an opening writes.
    name: &'static str,
    breaker: Breaker,
    /// Per thread: the latency ring, one sample per decision.
    latency: ThreadLocal<RefCell<Latency>>,
    /// Per thread: the plugin's part in the decision open there, what that
    /// decision charged the plugin and the frames of those charges
    /// ([`Unit::charge`]): its sample, and what the sample is taken over.
    part: ThreadLocal<Cell<(Part, u64, u64)>>,
    /// When a stream's failure last counted: a plugin's streams are one
    /// lane to the breaker.
    outage: Mutex<Option<Instant>>,
}

impl Unit {
    pub(crate) fn new(name: &'static str, spec: BreakerSpec) -> Self {
        Self {
            name,
            breaker: Breaker::new(spec),
            latency: ThreadLocal::new(),
            part: ThreadLocal::new(),
            outage: Mutex::new(None),
        }
    }

    /// One of the plugin's opens of decision `decision` on this thread; says
    /// whether the plugin already failed that decision (the one read of the
    /// cell an open makes). An open that reads `true` takes its fallback
    /// with no frame and no second count.
    pub(crate) fn enter(&self, decision: u64) -> bool {
        let cell = self.part.get_or_default();
        let (mut part, charged, frames) = cell.get();
        let (charged, frames) = if part.is_of(decision) {
            (charged, frames)
        } else {
            (0, 0)
        };
        let failed = part.open(decision);
        cell.set((part, charged, frames));
        failed
    }

    /// One failure at `now`, marked on the decision it fell in: its later
    /// opens fall back ([`Self::enter`]) and count no second failure
    /// ([`Self::failed_in`]). Outside a decision nothing is marked.
    pub(crate) fn fail(&self, now_micros: u64) {
        let cell = self.part.get_or_default();
        let (mut part, charged, frames) = cell.get();
        part.fail();
        cell.set((part, charged, frames));
        self.report(now_micros);
    }

    /// One failure at `now` with no decision to mark: a stream that broke
    /// behind the deciding threads. An opening writes its line.
    pub(crate) fn report(&self, now_micros: u64) {
        if self.breaker.failure(now_micros) {
            self.opened("failures");
        }
    }

    /// A stream's failure, dated `at` by its own lane: the instant its pause
    /// runs from. The breaker hears a plugin's streams as one lane: at most
    /// one failure a pause, which is all one lane can raise. So the streams
    /// that end together at a restart are one failure, one deciding thread
    /// and sixteen count the same at any phases, and a plugin that stays away
    /// counts once a pause. A decision asks for a failed stream a whole pause
    /// after its failure, so that open's failure is at or past the window by
    /// arithmetic. The apply asks without waiting the pause: a failed open
    /// it asked for inside the pause counts nothing.
    pub(crate) fn stream_failed(&self, at: Instant, now: impl FnOnce() -> u64) {
        let mut counted = self.outage.lock();
        if counted.is_some_and(|last| at.saturating_duration_since(last) < RECONNECT_PAUSE) {
            return;
        }
        *counted = Some(at);
        drop(counted);
        self.report(now());
    }

    /// Whether the plugin already failed decision `decision` on this thread;
    /// true until the thread's next decision opens it.
    pub(crate) fn failed_in(&self, decision: u64) -> bool {
        let (part, _, _) = self.part.get_or_default().get();
        part.failed_in(decision)
    }

    /// `micros` that decision `decision` charged the plugin on this thread
    /// for a round trip whose frame is `frame_micros`: part of the decision's
    /// sample and of what the sample is taken over, judged by the latency
    /// rule alone.
    pub(crate) fn charge(&self, decision: u64, micros: u64, frame_micros: u64) {
        let cell = self.part.get_or_default();
        let (part, charged, frames) = cell.get();
        if part.is_of(decision) {
            cell.set((
                part,
                charged.saturating_add(micros),
                frames.saturating_add(frame_micros),
            ));
        }
    }

    /// What decision `decision` charged the plugin on this thread so far:
    /// the plugin's own spend in the decision.
    pub(crate) fn charged(&self, decision: u64) -> u64 {
        let (part, charged, _) = self.part.get_or_default().get();
        if part.is_of(decision) {
            charged
        } else {
            0
        }
    }

    /// One of the plugin's closes of decision `decision` on this thread. The
    /// last one takes the decision's latency sample: the charges
    /// ([`Self::charge`]) over the decision's budget `seed_micros`, or over
    /// their frames where those are the lesser ([`Latency::frame`]). A
    /// decision that charged the plugin nothing records nothing; a window
    /// that reads slow opens the breaker at `now()`.
    pub(crate) fn leave(&self, decision: u64, seed_micros: u64, now: impl FnOnce() -> u64) {
        let cell = self.part.get_or_default();
        let (mut part, charged, frames) = cell.get();
        let last = part.close(decision);
        cell.set((part, charged, frames));
        if !last || charged == 0 {
            return;
        }
        let sample = u32::try_from(charged).unwrap_or(u32::MAX);
        let frame = Latency::frame(seed_micros, frames);
        let slow = self
            .latency
            .get_or(|| RefCell::new(Latency::new()))
            .borrow_mut()
            .record(sample, frame);
        if slow && self.breaker.slow(now()) {
            self.opened("latency");
        }
    }

    /// The breaker opened under `rule`: the one line of an opening, the
    /// same text and fields as the WebAssembly kind's. For the cool-down
    /// every stage of the plugin takes its fallback, and nothing else says
    /// so as it happens.
    fn opened(&self, rule: &'static str) {
        tracing::warn!(
            plugin = self.name,
            rule,
            cooldown_micros = self.breaker.cooldown_micros(),
            "the plugin's breaker opened: its stages take their fallback for the cool-down"
        );
    }

    /// Whether the plugin is bypassed at `now`.
    pub(crate) fn is_open(&self, now_micros: u64) -> bool {
        self.breaker.is_open(now_micros)
    }

    /// How many times the breaker opened.
    pub(crate) fn opens(&self) -> u64 {
        self.breaker.opens()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::Unit;
    use crate::plugins::shared::{breaker::Latency, config::BreakerSpec};

    const BUDGET: u64 = 1_000;

    fn unit() -> Unit {
        Unit::new("unit", BreakerSpec::default())
    }

    /// The charges of a decision are one sample, taken at the plugin's last
    /// close of it: on a pair neither side is past half the budget, the
    /// decision is, and a window of them reads slow at its first read.
    #[test]
    fn the_last_close_of_a_decision_takes_its_charges_as_one_sample() {
        let unit = unit();
        for decision in 1..=Latency::EVERY as u64 {
            assert!(!unit.enter(decision));
            assert!(!unit.enter(decision), "the other side of the pair");
            unit.charge(decision, 300, BUDGET);
            unit.charge(decision, 300, BUDGET);
            unit.leave(decision, BUDGET, || 5);
            assert_eq!(unit.opens(), 0, "one open is still to close");
            unit.leave(decision, BUDGET, || 5);
        }
        assert_eq!(unit.opens(), 1, "the window's first read");
    }

    /// Under a budget wider than the frames of a decision's round trips, the
    /// sample is taken over those frames: a request in process may state no
    /// budget, and its decision is then seeded with the widest. A plugin that
    /// used six tenths of its one frame reads slow there, where its charge
    /// over that budget read as nothing.
    #[test]
    fn a_charge_is_read_against_its_frame_under_a_wider_budget() {
        let unit = unit();
        let widest = u64::from(u32::MAX);
        for decision in 1..=Latency::EVERY as u64 {
            assert!(!unit.enter(decision));
            unit.charge(decision, 600, BUDGET);
            unit.leave(decision, widest, || 5);
        }
        assert_eq!(unit.opens(), 1, "the window's first read");
    }

    /// The frames of a decision add up: three round trips that each used
    /// four tenths of their frame are four tenths of the decision's frames,
    /// not twelve tenths of one.
    #[test]
    fn three_calls_under_half_their_frames_are_no_slow_sample() {
        let unit = unit();
        let widest = u64::from(u32::MAX);
        for decision in 1..=Latency::EVERY as u64 {
            assert!(!unit.enter(decision));
            for _ in 0..3 {
                unit.charge(decision, 400, BUDGET);
            }
            unit.leave(decision, widest, || 5);
        }
        assert_eq!(unit.opens(), 0);
    }

    /// A decision's charges start over with the decision: one slow decision
    /// in front of a window of fast ones is one slow sample.
    #[test]
    fn the_charges_of_a_decision_do_not_carry_into_the_next() {
        let unit = unit();
        for decision in 1..=Latency::EVERY as u64 {
            assert!(!unit.enter(decision));
            unit.charge(decision, if decision == 1 { 900 } else { 100 }, BUDGET);
            unit.leave(decision, BUDGET, || 5);
        }
        assert_eq!(unit.opens(), 0);
    }

    /// A charge is the decision it names: one that names another decision
    /// (a frame read after its decision closed) is nobody's sample.
    #[test]
    fn a_charge_that_names_another_decision_is_dropped() {
        let unit = unit();
        for decision in 1..=Latency::EVERY as u64 {
            assert!(!unit.enter(decision));
            unit.charge(decision + 1_000, 900, BUDGET);
            unit.leave(decision, BUDGET, || 5);
        }
        assert_eq!(unit.opens(), 0);
    }

    /// A decision that charged the plugin nothing is no sample: it does not
    /// thin a slow window out.
    #[test]
    fn a_decision_that_charged_nothing_is_no_sample() {
        let unit = unit();
        let idle = Latency::EVERY as u64 / 2;
        for decision in 1..=idle {
            assert!(!unit.enter(decision));
            unit.leave(decision, BUDGET, || 5);
        }
        for decision in idle + 1..=idle + Latency::EVERY as u64 {
            assert!(!unit.enter(decision));
            unit.charge(decision, 600, BUDGET);
            unit.leave(decision, BUDGET, || 5);
        }
        assert_eq!(unit.opens(), 1, "the 64 charged decisions are the window");
    }

    /// What a decision charged so far is the plugin's own spend in it, read
    /// by the decision's number: another decision's reads nothing, and the
    /// next decision starts over.
    #[test]
    fn what_a_decision_charged_so_far_is_read_by_its_number() {
        let unit = unit();
        assert!(!unit.enter(1));
        assert_eq!(unit.charged(1), 0);
        unit.charge(1, 300, BUDGET);
        unit.charge(1, 200, BUDGET);
        assert_eq!((unit.charged(1), unit.charged(2)), (500, 0));
        unit.leave(1, BUDGET, || 5);
        assert!(!unit.enter(2));
        assert_eq!(unit.charged(2), 0, "the next decision starts over");
        unit.leave(2, BUDGET, || 5);
    }

    /// A failure marks its decision for the plugin's later opens of it and
    /// for nothing after; a failure with no decision to mark counts alone.
    #[test]
    fn a_failure_marks_its_decision_and_a_report_marks_none() {
        let unit = unit();
        assert!(!unit.enter(1));
        unit.fail(5);
        assert!(unit.failed_in(1));
        assert!(unit.enter(1), "the other side finds the decision failed");
        unit.leave(1, BUDGET, || 5);
        unit.leave(1, BUDGET, || 5);
        assert!(!unit.enter(2), "the next decision starts clean");
        unit.report(5);
        assert!(!unit.failed_in(2), "a stream's failure marks no decision");
        unit.leave(2, BUDGET, || 5);
        // Five failures in the window: the default threshold.
        for _ in 0..3 {
            assert!(!unit.is_open(5));
            unit.report(5);
        }
        assert!(unit.is_open(5));
        assert_eq!(unit.opens(), 1);
    }

    /// The breaker hears a plugin's streams as one lane: at most one failure
    /// a pause. Two threads' streams end together, and each is asked for
    /// again a pause after its own failure: the first at 100, 200 and 300
    /// ms, the second, whose decisions come later, at 151 and 251. Under a
    /// threshold of four the count reads one thread's: closed through 251,
    /// open at 300. Counted by half a pause, the second thread's reopens
    /// count too and the breaker opens at 251.
    #[test]
    fn the_streams_of_one_plugin_count_as_one_lane_at_any_phases() {
        let unit = Unit::new(
            "unit",
            BreakerSpec {
                failures: 4,
                window_micros: 60_000_000,
                cooldown_micros: 60_000_000,
            },
        );
        let start = Instant::now();
        for millis in [0, 1, 100, 151, 200, 251] {
            unit.stream_failed(start + Duration::from_millis(millis), || 5);
        }
        assert_eq!(unit.opens(), 0, "three pauses, three failures");
        unit.stream_failed(start + Duration::from_millis(300), || 5);
        assert_eq!(unit.opens(), 1, "the fourth pause's failure opens");
    }
}
