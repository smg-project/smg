//! What the scorers' cases read of a cost term.

use super::{CostTerm, LoadField};
use crate::engine::view::LoadRow;

impl CostTerm {
    pub(in crate::stages) fn read(self, load: Option<&LoadRow>, in_flight: u32) -> Option<f64> {
        match self {
            Self::Load(field) => field.read(load?),
            Self::InFlight => Some(f64::from(in_flight)),
            // The walk leaves the sends to the pass behind it.
            Self::SentRequests | Self::SentTokens => Some(0.0),
        }
    }
}

impl LoadField {
    fn read(self, row: &LoadRow) -> Option<f64> {
        let value = match self {
            Self::Running => row.running,
            Self::Queued => row.queued,
            // A report of a queue's depth alone writes zero here: beside a
            // queue that zero is no count of its tokens.
            Self::WaitingUncachedTokens if row.waiting_uncached_tokens == 0 && row.queued > 0 => {
                return None
            }
            Self::WaitingUncachedTokens => row.waiting_uncached_tokens,
            Self::KvUsage => u32::from(row.kv_usage_permille),
            Self::ActiveKv => u32::from(row.active_kv_permille),
            Self::ResidentTokens if !row.resident_counted => return None,
            Self::ResidentTokens => row.resident_tokens,
        };
        Some(f64::from(value))
    }
}
