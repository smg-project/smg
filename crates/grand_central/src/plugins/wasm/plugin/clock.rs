//! A call's bound on the clock and the judgement of its time: the epoch
//! ticks and their deadlines, the thread's CPU clock, and whose time a late
//! call's was. A child of the plugin's module, so that its items reach the
//! plugin, its binding and their cases, and nothing more of the crate.

use wasmtime::{Trap, UpdateDeadline};

use super::super::bindings::Watch;
use crate::plugins::shared::breaker::{Latency, Recorded};

/// The epoch ticker's period. A tick that finds a guest call in flight reads
/// its thread's CPU clock ([`tick`]): a call's bound on the clock counts the
/// time its thread had the CPU, and the call ends at the first tick that
/// finds it has had the bound. Fuel is the fine bound (one unit per
/// operator, a bulk memory operation by its size once it has run); the
/// ticks catch what the calibration underprices and the CPU a host call
/// spends.
pub(super) const EPOCH_TICK_MICROS: u64 = 1_000;

/// A bound of `bound_micros` on the wall clock in whole ticks, rounded up:
/// no deadline there ends a call sooner.
fn whole_ticks(bound_micros: u64) -> u64 {
    bound_micros.div_ceil(EPOCH_TICK_MICROS)
}

/// The epoch deadline of a call whose bound is `frame_micros` on the wall
/// clock: the bound in whole ticks, plus the tick in flight.
pub(super) fn deadline_ticks(frame_micros: u32) -> u64 {
    whole_ticks(u64::from(frame_micros)) + 1
}

/// The first epoch deadline of a call, in ticks from its start. A watched
/// call with no reading of its start meets its next tick, which takes the
/// reading its bound counts from. A call whose start was read meets the tick
/// its deadline on the wall clock falls on and none before it, so that
/// reading ends only a call that is late: no call the wall clock would let
/// run is ended because its thread is armed. A call nothing watches (an
/// `init`; a host with no thread clock) ends at that deadline.
pub(super) fn first_deadline(watched: bool, start_read: bool, bound_micros: u32) -> u64 {
    if watched && !start_read {
        1
    } else {
        deadline_ticks(bound_micros)
    }
}

/// The calling thread's CPU time, in microseconds; `None` on a host with no
/// thread clock, where a call's bound stays on the wall clock.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(super) fn thread_cpu_micros() -> Option<u64> {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
    let secs = u64::try_from(now.tv_sec).ok()?;
    let nanos = u64::try_from(now.tv_nsec).ok()?;
    Some(secs.saturating_mul(1_000_000).saturating_add(nanos / 1_000))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn thread_cpu_micros() -> Option<u64> {
    None
}

/// One tick with a guest call in flight on its store, run on the call's own
/// thread at the guest's next check. A watched call ends at the tick that
/// finds it has had its bound on the CPU since a reading, and runs on
/// otherwise, however late the wall clock says it is: the guest of a thread
/// the host held finishes in its own microseconds when the thread has the
/// CPU back. The reading is the call's start where the thread took one;
/// otherwise the call's first tick takes it, and the bound counts from
/// there. A first tick knows nothing of the time before it: it can end a
/// call (a call that has its bound on the CPU after it), it never says a
/// call was held. A call nothing watches ends here, on the wall clock.
pub(super) fn tick(watch: &mut Watch) -> wasmtime::Result<UpdateDeadline> {
    let (true, Some(cpu)) = (watch.watched, thread_cpu_micros()) else {
        return Err(Trap::Interrupt.into());
    };
    let since = match watch.start {
        Some(start) => start,
        None => *watch.first_tick.get_or_insert(cpu),
    };
    if cpu.saturating_sub(since) >= watch.bound_micros {
        return Err(Trap::Interrupt.into());
    }
    Ok(UpdateDeadline::Continue(1))
}

/// How a call ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum End {
    /// It answered or declined.
    Answered,
    /// It failed by itself: a trap, or its fuel ran out.
    Failed,
    /// Its bound on the clock ended it.
    OnTheClock,
}

/// What the end of a call says of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// The call's time is the plugin's own: it was not late, or its bound
    /// ended it, or a reading of its start says it had its bound, or it
    /// failed by itself.
    Own,
    /// Late, answered or declined, and a reading of its start says its
    /// thread had less than the bound on the CPU: the host held it.
    Held,
    /// Late, not ended, and no reading of its start: the host held it or it
    /// overran a bound its fuel did not price, and nothing says which.
    Unread,
}

/// A call's own time in its plugin's two sums of a decision, in
/// microseconds. With no reading a call's time is an interval, from its
/// fuel to its charge, and the two sums take its two ends; a reading makes
/// it one number. The low end is lent against a reading to come: on a host
/// with no thread clock none can, and a call's time is its charge in both.
/// The `init` a decision pays on a cold slot is timed the same way: its two
/// ends join the decision's sample, never the plugin's own spend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct OwnTime {
    /// For the decision's latency sample: what a reading says the call had;
    /// with none, its charge. The ring wants more than one slow sample, so
    /// it takes the high end: a guest slow in what its fuel does not price
    /// reads slow there.
    pub(super) sample: u64,
    /// For the plugin's own spend, which a later call that runs out is
    /// judged against: what a reading says the call had; with none, its
    /// fuel. One run-out is a verdict, so it convicts on the low end: the
    /// time a host held a call is never the plugin's spend. A host with no
    /// thread clock has no low end to lend: there it is the charge.
    pub(super) spend: u64,
}

impl OwnTime {
    /// The time of a call or an `init` a reading covers, one number: what it
    /// could spend, its fuel or the CPU its thread `had`, never more than
    /// its charge.
    pub(super) fn read(fuel_micros: u64, had: u64, charged: u64) -> Self {
        let read = fuel_micros.max(had).min(charged);
        Self {
            sample: read,
            spend: read,
        }
    }

    /// The time of one no reading covers, its two ends. With no thread
    /// clock (`watched` is false) nothing will ever read it, and its bound
    /// is the wall clock's: the time it took is its spend.
    pub(super) fn unread(watched: bool, fuel_micros: u64, charged: u64) -> Self {
        Self {
            sample: charged,
            spend: if watched { fuel_micros } else { charged },
        }
    }

    /// The two sums with `more` added.
    pub(super) fn plus(self, more: Self) -> Self {
        Self {
            sample: self.sample.saturating_add(more.sample),
            spend: self.spend.saturating_add(more.spend),
        }
    }
}

/// One thread's latency ring and its arm: whether the thread reads its CPU
/// clock at the start and the end of the plugin's calls
/// ([`ThreadRing::arm`]). The sample of a late call no reading covers is
/// the time the call took, which a hold of the host makes slow, so the ring
/// that is about to take one is armed until that sample has left it, and at
/// most one of them is ever in a window. A call that runs out under a
/// decision's remainder arms it too where the two ends of the time nothing
/// read part: by the time the plugin's unread calls took the run-out is its
/// own, by their fuel it is not. So does a sample slow at its high end alone
/// ([`ThreadRing::uncovered`]), late call or none: a window never holds two
/// slow samples no reading covered. Where the bound ended a call of the
/// decision under the plugin's own bound, such a sample arms the ring and
/// is not counted: at its bound a call is the plugin's.
#[derive(Debug)]
pub(super) struct ThreadRing {
    ring: Latency,
    /// The samples the ring still takes armed.
    armed: usize,
}

impl ThreadRing {
    pub(super) const fn new() -> Self {
        Self {
            ring: Latency::new(),
            armed: 0,
        }
    }

    /// Arms the ring's thread for the ring's length in samples: the sample
    /// of the decision in flight and the [`Latency::SAMPLES`] - 1 after it.
    /// The sample that follows those overwrites this decision's.
    pub(super) fn arm(&mut self) {
        self.armed = Latency::SAMPLES;
    }

    /// Whether the thread reads its CPU clock at the start and the end of a
    /// call.
    pub(super) fn armed(&self) -> bool {
        self.armed != 0
    }

    /// Whether a sample of `micros` is past a mark that `low_micros` is
    /// short of, both under a frame of `frame_micros`. The second is the
    /// same decision at the low end of the time nothing read, so such a
    /// sample is slow by that time alone: no reading covers it. A sample
    /// slow by both ends is not one, nor is a call that is no sample. The
    /// ring holds its marks: it says whether one stands between the two
    /// ([`Latency::mark_between`]).
    pub(super) fn uncovered(micros: u32, low_micros: u32, frame_micros: u32) -> bool {
        Latency::mark_between(micros, low_micros, frame_micros)
    }

    /// Takes one decision's sample of `micros`, whose low end is
    /// `low_micros`, under a frame of `frame_micros`. Says whether it is
    /// counted unread and whether a window it completed reads slow
    /// ([`ThreadRing::record`]). A sample no reading covers
    /// ([`ThreadRing::uncovered`]) on a ring nothing armed arms the ring,
    /// and it is counted unread. Where the bound `ended` a call of the
    /// decision under the plugin's own bound it arms the ring all the same
    /// and is not counted: the bound counted the time the thread had, so at
    /// its bound a call is the plugin's, never unread by its own verdict or
    /// through its decision's sample. A call the decision's remainder
    /// clamped is not one: the plugin failed nothing there.
    pub(super) fn take(
        &mut self,
        micros: u32,
        low_micros: u32,
        frame_micros: u32,
        ended: bool,
    ) -> (bool, bool) {
        let parted = !self.armed() && Self::uncovered(micros, low_micros, frame_micros);
        if parted {
            self.arm();
        }
        (parted && !ended, self.record(micros, frame_micros))
    }

    /// Records one call in the ring ([`Latency::record`]); `true` when a
    /// window it completed reads slow. The arm falls by one where a sample
    /// is written, so it is counted in the ring's own unit, and the ring
    /// says whether it took one ([`Recorded`]).
    pub(super) fn record(&mut self, micros: u32, frame_micros: u32) -> bool {
        let recorded = self.ring.record(micros, frame_micros);
        if recorded != Recorded::Nothing {
            self.armed = self.armed.saturating_sub(1);
        }
        recorded == Recorded::SlowWindow
    }
}

/// A call's own time and its verdict, judged once, at its end. A reading
/// covers a call whose start was read (its thread is armed): `cpu` reads the
/// thread's clock again, and the call's time is what it could spend, its
/// fuel or the CPU its thread had, never the time the host held it, late or
/// not. Late, such a call is the host's when its thread had less than the
/// bound and it answered or declined, the plugin's otherwise. A hold excuses
/// lateness, never a fault: a call that failed by itself is the plugin's
/// whatever its thread had. One call there is not read again: not late,
/// with fuel that covers the time it took, its time is its fuel whatever
/// its thread had. No reading covers any other call. It is sampled at its
/// charge (`fuel_micros` or `elapsed_micros`, whichever is more), what it
/// was before the bound counted the CPU, and late it is unread unless its
/// bound ended it; it spends its fuel, for nothing says how much of its
/// time its thread had. Time nothing read draws the decision's remainder,
/// never the plugin's own spend. A host with no thread clock watches no
/// call and can never read one: there a call spends its charge, the time
/// the wall clock that bounds it says it took.
///
/// Late is the wall clock at or past the bound in whole ticks, the earliest
/// a deadline there could have ended the call. Not a tick more: a call the
/// host held for between the two would be charged its hold and counted
/// nowhere, and two such samples in a ring read slow. Not the bound itself:
/// under a frame shorter than a tick a call inside its fuel runs past its
/// bound on the wall clock and is ended by nothing.
// A call that is not late on a thread that is not armed pays two compares
// here: inlined, it pays no call.
#[inline]
pub(super) fn judge(
    watch: &Watch,
    elapsed_micros: u64,
    fuel_micros: u64,
    end: End,
    cpu: impl FnOnce() -> Option<u64>,
) -> (OwnTime, Verdict) {
    let charged = fuel_micros.max(elapsed_micros);
    let late = watch.watched
        && elapsed_micros >= whole_ticks(watch.bound_micros).saturating_mul(EPOCH_TICK_MICROS);
    // Not late and covered by its fuel, a call takes its fuel in both arms
    // below: no reading can move its time, so none is taken.
    let Some(had) = watch
        .start
        .filter(|_| watch.watched && (late || fuel_micros < elapsed_micros))
        .and_then(|start| Some(cpu()?.saturating_sub(start)))
    else {
        let verdict = if late && end != End::OnTheClock {
            Verdict::Unread
        } else {
            Verdict::Own
        };
        return (
            OwnTime::unread(watch.watched, fuel_micros, charged),
            verdict,
        );
    };
    let verdict = if late && had < watch.bound_micros && end == End::Answered {
        Verdict::Held
    } else {
        Verdict::Own
    };
    (OwnTime::read(fuel_micros, had, charged), verdict)
}
