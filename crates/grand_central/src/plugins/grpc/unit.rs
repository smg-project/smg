//! The plugin's part in a decision and its breaker. The decision is the
//! breaker's unit: a plugin fails a decision once, however many of its opens
//! fail in it (each side of a pair, each binding), and its last close of the
//! decision takes the decision's one latency sample: what the decision
//! charged the plugin, over the frames the plugin was given in it, at most
//! the decision's budget ([`Budget::sample_frame`]). The ring and its marks are
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
    breaker::{Breaker, Latency, Recorded},
    config::BreakerSpec,
    Budget, Part,
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
    /// ([`Self::charge`]) over the decision's `budget` as seeded, or over
    /// their frames where those are the lesser ([`Budget::sample_frame`]).
    /// A decision that charged the plugin nothing records nothing; a window
    /// that reads slow opens the breaker at `now()`.
    pub(crate) fn leave(&self, decision: u64, budget: &Budget, now: impl FnOnce() -> u64) {
        let cell = self.part.get_or_default();
        let (mut part, charged, frames) = cell.get();
        let last = part.close(decision);
        cell.set((part, charged, frames));
        if !last || charged == 0 {
            return;
        }
        let sample = u32::try_from(charged).unwrap_or(u32::MAX);
        let frame = budget.sample_frame(frames);
        let recorded = self
            .latency
            .get_or(|| RefCell::new(Latency::new()))
            .borrow_mut()
            .record(sample, frame);
        if recorded == Recorded::SlowWindow && self.breaker.slow(now()) {
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
#[path = "../../../tests/unit/plugins/grpc/unit/tests.rs"]
mod tests;
