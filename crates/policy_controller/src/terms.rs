//! Fields a policy can use in a sum, in the units of the row that holds them.

use crate::view::{Candidate, LoadRow, RequestFacts};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Term {
    Load(LoadField),
    InFlight,
}

impl Term {
    pub(crate) fn named(name: &str) -> Option<Self> {
        let field = match name {
            "candidate.in_flight" => return Some(Self::InFlight),
            "load.active_kv_permille" => LoadField::ActiveKv,
            "load.kv_usage_permille" => LoadField::KvUsage,
            "load.queued" => LoadField::Queued,
            "load.resident_tokens" => LoadField::ResidentTokens,
            "load.running" => LoadField::Running,
            "load.waiting_uncached_tokens" => LoadField::WaitingUncachedTokens,
            _ => return None,
        };
        Some(Self::Load(field))
    }

    pub(crate) fn reads_load(self) -> bool {
        matches!(self, Self::Load(_))
    }

    pub(crate) fn read(self, values: &mut TermValues<'_, '_>) -> Option<f64> {
        match self {
            Self::Load(field) => field.read(values.load.as_ref()?),
            Self::InFlight => Some(f64::from(values.in_flight())),
        }
    }
}

/// The reads of one row. A sum and its missing-report price can ask for
/// the same count, so the first read serves both. The report is the one
/// the decision reads: none where it is past the document's age limit.
pub(crate) struct TermValues<'row, 'facts> {
    facts: &'row RequestFacts<'facts>,
    row: &'row Candidate,
    load: Option<LoadRow>,
    in_flight: Option<u32>,
}

impl<'row, 'facts> TermValues<'row, 'facts> {
    pub(crate) fn new(
        facts: &'row RequestFacts<'facts>,
        row: &'row Candidate,
        load: Option<LoadRow>,
    ) -> Self {
        Self {
            facts,
            row,
            load,
            in_flight: None,
        }
    }

    pub(crate) fn in_flight(&mut self) -> u32 {
        *self
            .in_flight
            .get_or_insert_with(|| self.facts.in_flight(self.row))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum LoadField {
    Running,
    Queued,
    WaitingUncachedTokens,
    KvUsage,
    ActiveKv,
    ResidentTokens,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sum_and_the_missing_report_price_share_one_count() {
        let facts = RequestFacts::default();
        let first = Candidate {
            in_flight: 13,
            ..Candidate::default()
        };
        let changed = Candidate {
            in_flight: 29,
            ..Candidate::default()
        };
        let mut values = TermValues::new(&facts, &first, None);
        assert_eq!(Term::InFlight.read(&mut values), Some(13.0));
        values.row = &changed;
        assert_eq!(values.in_flight(), 13);
    }
}
