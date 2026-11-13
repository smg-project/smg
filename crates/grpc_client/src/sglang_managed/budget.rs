//! Original-expiry admission and a local monotonic budget.

use std::time::{Duration, Instant};

use tonic::Status;

use super::proto;

#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    expiry_unix_ms: u64,
    local: Instant,
}

impl Deadline {
    /// Remote timestamps never lengthen the caller's remaining budget.
    pub fn admit(
        budget: &proto::Budget,
        wall_now_ms: u64,
        monotonic_now: Instant,
        clock_offset_ms: i64,
        max_clock_skew_ms: u64,
        max_duration_ms: u64,
    ) -> Result<Self, Status> {
        if clock_offset_ms.unsigned_abs() > max_clock_skew_ms {
            return Err(Status::unavailable(
                "managed worker clock is outside tolerance",
            ));
        }
        let remaining = budget
            .original_expiry_unix_ms
            .saturating_sub(wall_now_ms)
            .min(budget.remaining_timeout_ms);
        if remaining == 0 {
            return Err(Status::deadline_exceeded("managed operation expired"));
        }
        if remaining > max_duration_ms
            || budget.original_expiry_unix_ms.saturating_sub(wall_now_ms) > max_duration_ms
        {
            return Err(Status::invalid_argument(
                "managed deadline exceeds retention bound",
            ));
        }
        let local = monotonic_now
            .checked_add(Duration::from_millis(remaining))
            .ok_or_else(|| Status::invalid_argument("managed deadline overflow"))?;
        Ok(Self {
            expiry_unix_ms: budget.original_expiry_unix_ms,
            local,
        })
    }

    /// Bind, query and reconnect use this same deadline, never a fresh timeout.
    pub fn remaining(self, now: Instant) -> Result<Duration, Status> {
        self.local
            .checked_duration_since(now)
            .filter(|d| !d.is_zero())
            .ok_or_else(|| Status::deadline_exceeded("managed operation expired"))
    }

    /// Tombstones remain until the latest legitimate start could arrive.
    pub fn expiry_unix_ms(self) -> u64 {
        self.expiry_unix_ms
    }
}
