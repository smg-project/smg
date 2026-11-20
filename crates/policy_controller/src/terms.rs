//! Fields a policy can use in a sum, in the units of the row that holds them.

use crate::view::LoadRow;

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

    #[cfg(test)]
    pub(crate) fn read(self, load: Option<&LoadRow>, in_flight: u32) -> Option<f64> {
        match self {
            Self::Load(field) => field.read(load?),
            Self::InFlight => Some(f64::from(in_flight)),
        }
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

#[cfg(test)]
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

/// Coefficients in field name order. A field left out has coefficient zero.
#[derive(Debug, Default)]
pub(crate) struct Coefficients {
    pub(crate) in_flight: f64,
    active_kv: f64,
    kv_usage: f64,
    queued: f64,
    pub(crate) resident_tokens: f64,
    running: f64,
    pub(crate) waiting_uncached_tokens: f64,
}

impl Coefficients {
    pub(crate) fn of(terms: &[(Term, f64)]) -> Self {
        let mut coefficients = Self::default();
        for &(term, coefficient) in terms {
            let field = match term {
                Term::InFlight => &mut coefficients.in_flight,
                Term::Load(field) => match field {
                    LoadField::ActiveKv => &mut coefficients.active_kv,
                    LoadField::KvUsage => &mut coefficients.kv_usage,
                    LoadField::Queued => &mut coefficients.queued,
                    LoadField::ResidentTokens => &mut coefficients.resident_tokens,
                    LoadField::Running => &mut coefficients.running,
                    LoadField::WaitingUncachedTokens => &mut coefficients.waiting_uncached_tokens,
                },
            };
            *field = coefficient;
        }
        coefficients
    }

    pub(crate) fn names_kv_or_running(&self) -> bool {
        self.active_kv != 0.0 || self.kv_usage != 0.0 || self.running != 0.0
    }

    /// Adds the report's terms in name order. A missing named value ends
    /// the sum under `over_least` and adds nothing otherwise.
    #[inline(always)]
    pub(crate) fn add_report<
        const RESIDENT: bool,
        const WAITING: bool,
        const KV_OR_RUNNING: bool,
    >(
        &self,
        mut sum: f64,
        row: &LoadRow,
        over_least: bool,
    ) -> Option<f64> {
        if KV_OR_RUNNING {
            sum += f64::from(row.active_kv_permille) * self.active_kv;
            sum += f64::from(row.kv_usage_permille) * self.kv_usage;
        }
        sum += f64::from(row.queued) * self.queued;
        if RESIDENT {
            if row.resident_counted {
                sum += f64::from(row.resident_tokens) * self.resident_tokens;
            } else if over_least {
                return None;
            }
        }
        if KV_OR_RUNNING {
            sum += f64::from(row.running) * self.running;
        }
        if WAITING {
            // A queue reported by depth alone gives no count of its tokens.
            if row.waiting_uncached_tokens != 0 || row.queued == 0 {
                sum += f64::from(row.waiting_uncached_tokens) * self.waiting_uncached_tokens;
            } else if over_least {
                return None;
            }
        }
        Some(sum)
    }
}
