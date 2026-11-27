use std::{
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use super::*;

mod backends;
mod fence;
mod lost_records;
mod transitions;
mod uncertain_writes;

/// One second: the step between the tests' instants.
const S: Duration = Duration::from_secs(1);

/// How far ahead of the clock the tests' instants start. The elector
/// dates a first sight at the later of the caller's instant and its own
/// clock after the read, so a synthetic instant must never be overtaken
/// by the real clock. One second ahead was overtaken where a host held
/// the test or the lease file's writes for longer; an hour is not.
const AHEAD: Duration = S.saturating_mul(3_600);

fn settings() -> ElectionSettings {
    ElectionSettings::default()
}

fn candidate(lease: &MemoryLease, identity: &str) -> Elector {
    Elector::new(Box::new(lease.clone()), identity, settings()).expect("valid settings")
}

/// A held record no candidate wrote.
fn held_by(holder: &str, term: u64, duration_ms: u64) -> LeaseRecord {
    LeaseRecord {
        holder: holder.to_string(),
        term,
        acquired_at_micros: 1,
        renewed_at_micros: 1,
        duration_ms,
    }
}

impl Elector {
    /// The take of a lease where no record stood and the renewals after
    /// it, a period apart, up to `at`: the fence of such a take opens
    /// one renew deadline after it, a renewal behind it, and the last
    /// renewal here is the one at `at`. The wait is dated by the clock
    /// where the clock is past the instant handed, so `at` is a renew
    /// deadline ahead of the clock or more. An elector alone takes at
    /// `at` and leads at once. For the cases of this crate.
    pub(crate) fn lead_at(&mut self, at: Instant) {
        if self.alone {
            let take = self.tick(at).expect("the take");
            assert!(matches!(take, TickOutcome::Acquired { .. }), "{take:?}");
            assert!(self.fence.leader_term(at).is_some());
            return;
        }
        let (period, deadline) = (self.settings.renew_period, self.settings.renew_deadline);
        let steps = u32::try_from(deadline.as_nanos().div_ceil(period.as_nanos()))
            .expect("a count of periods");
        let took = at - period * steps;
        let take = self.tick(took).expect("the take");
        assert!(matches!(take, TickOutcome::Acquired { .. }), "{take:?}");
        for step in 1..=steps {
            assert_eq!(self.fence.leader_term(took + period * (step - 1)), None);
            let renewal = self.tick(took + period * step).expect("a renewal");
            assert!(
                matches!(renewal, TickOutcome::Renewed { .. }),
                "{renewal:?}"
            );
        }
        assert!(self.fence.leader_term(at).is_some());
    }
}

impl MemoryLease {
    /// The record gone by no holder's hand, as a deleted object or a
    /// deleted file: for the cases of this crate.
    pub(crate) fn lose(&self) {
        *self.record.lock() = None;
    }
}

#[test]
fn an_empty_identity_is_refused() {
    let err = Elector::new(Box::new(MemoryLease::new()), "", settings()).expect_err("refused");
    assert_eq!(err.to_string(), "election identity: the identity is empty");
}

#[test]
fn the_records_duration_is_whole_milliseconds_rounded_up() {
    let fine = ElectionSettings {
        duration: Duration::from_micros(1_900),
        renew_period: Duration::from_micros(1_000),
        renew_deadline: Duration::from_micros(1_800),
    };
    let lease = MemoryLease::new();
    let mut a = Elector::new(Box::new(lease.clone()), "a@host:1", fine).expect("settings");
    a.tick(Instant::now() + AHEAD).expect("acquire");
    let record = lease.clone().read().expect("read").expect("the record");
    assert_eq!(record.duration_ms, 2, "1.9 ms rounds up");
    assert!(record.duration() >= fine.duration);
    assert!(
        record.duration() > fine.renew_deadline,
        "a follower waits past the fence"
    );
    assert!(matches!(whole_millis_up(Duration::from_secs(3)), Ok(3_000)));
    assert!(matches!(whole_millis_up(Duration::from_millis(1)), Ok(1)));
}

#[test]
fn settings_out_of_order_are_refused_before_any_step() {
    let bad = [
        (
            ElectionSettings {
                renew_period: Duration::ZERO,
                ..settings()
            },
            "the renew period is zero",
        ),
        (
            ElectionSettings {
                renew_deadline: Duration::from_secs(1),
                ..settings()
            },
            "the renew deadline is not longer than the renew period",
        ),
        (
            ElectionSettings {
                duration: Duration::from_secs(2),
                ..settings()
            },
            "the duration is not longer than the renew deadline",
        ),
    ];
    for (settings, why) in bad {
        let err = Elector::new(Box::new(MemoryLease::new()), "a", settings).expect_err("refused");
        assert_eq!(err.to_string(), format!("election settings: {why}"));
    }
    // The design's profile for fewer leadership changes.
    let long = ElectionSettings {
        duration: Duration::from_secs(15),
        renew_period: Duration::from_secs(2),
        renew_deadline: Duration::from_secs(10),
    };
    assert!(long.validate().is_ok());
}

#[test]
fn a_duration_too_large_for_the_record_is_refused() {
    // The whole milliseconds fit; rounded up they do not.
    let edge = Duration::from_millis(u64::MAX) + Duration::from_nanos(1);
    for duration in [edge, Duration::MAX] {
        let settings = ElectionSettings {
            duration,
            renew_period: S,
            renew_deadline: duration - S,
        };
        let err = Elector::new(Box::new(MemoryLease::new()), "a", settings).expect_err("refused");
        assert_eq!(
            err.to_string(),
            "election settings: the duration is too large for the lease record"
        );
    }
}

#[test]
fn a_record_serializes_as_one_json_object() {
    let record = LeaseRecord {
        holder: "a@host:1".to_string(),
        term: 3,
        acquired_at_micros: 10,
        renewed_at_micros: 20,
        duration_ms: 3_000,
    };
    let json = serde_json::to_string(&record).expect("serialize");
    assert_eq!(
        json,
        r#"{"holder":"a@host:1","term":3,"acquired_at_micros":10,"renewed_at_micros":20,"duration_ms":3000}"#
    );
    let back: LeaseRecord = serde_json::from_str(&json).expect("parse");
    assert_eq!(back, record);
    assert!(!record.is_released());
    assert_eq!(record.duration(), Duration::from_secs(3));
}
