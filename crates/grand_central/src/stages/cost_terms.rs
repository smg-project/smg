//! Fields a policy can use in a sum, in the units of the row that holds them.

use crate::{engine::view::LoadRow, tracking::dispatch_credit::DispatchCounts};

#[derive(Clone, Copy, Debug)]
pub(crate) enum CostTerm {
    Load(LoadField),
    InFlight,
    /// The sends to the row that no report of it shows yet.
    SentRequests,
    /// The prompt tokens of those sends: the count each one stated, and
    /// for a send that stated none the bytes of its text over four. A
    /// stage that names it asks for each request's size
    /// ([`Coefficients::sizes_sends`]).
    SentTokens,
}

impl CostTerm {
    /// Every name a document may write, with the term it reads.
    pub(crate) const NAMES: [(&str, Self); 9] = [
        ("candidate.in_flight", Self::InFlight),
        ("load.active_kv_permille", Self::Load(LoadField::ActiveKv)),
        ("load.kv_usage_permille", Self::Load(LoadField::KvUsage)),
        ("load.queued", Self::Load(LoadField::Queued)),
        (
            "load.resident_tokens",
            Self::Load(LoadField::ResidentTokens),
        ),
        ("load.running", Self::Load(LoadField::Running)),
        (
            "load.waiting_uncached_tokens",
            Self::Load(LoadField::WaitingUncachedTokens),
        ),
        ("sent.requests", Self::SentRequests),
        ("sent.tokens", Self::SentTokens),
    ];

    pub(crate) fn named(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find_map(|&(written, term)| (written == name).then_some(term))
    }

    pub(crate) fn reads_load(self) -> bool {
        matches!(self, Self::Load(_))
    }

    pub(crate) fn reads_sent(self) -> bool {
        matches!(self, Self::SentRequests | Self::SentTokens)
    }

    #[cfg(test)]
    pub(crate) fn read(self, load: Option<&LoadRow>, in_flight: u32) -> Option<f64> {
        match self {
            Self::Load(field) => field.read(load?),
            Self::InFlight => Some(f64::from(in_flight)),
            // The walk leaves the sends to the pass behind it.
            Self::SentRequests | Self::SentTokens => Some(0.0),
        }
    }
}

/// The family of names that read a signal the document declares:
/// `signal.<name>`. A stage resolves the name where it is built.
pub(crate) const SIGNAL_FAMILY: &str = "signal.";

/// The signal a written name reads: what stands behind `signal.`, where
/// the name is of that family.
pub(crate) fn signal_named(written: &str) -> Option<&str> {
    written.strip_prefix(SIGNAL_FAMILY)
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
    pub(crate) sent_requests: f64,
    pub(crate) sent_tokens: f64,
}

impl Coefficients {
    pub(crate) fn of(terms: &[(CostTerm, f64)]) -> Self {
        let mut coefficients = Self::default();
        for &(term, coefficient) in terms {
            let field = match term {
                CostTerm::InFlight => &mut coefficients.in_flight,
                CostTerm::Load(field) => match field {
                    LoadField::ActiveKv => &mut coefficients.active_kv,
                    LoadField::KvUsage => &mut coefficients.kv_usage,
                    LoadField::Queued => &mut coefficients.queued,
                    LoadField::ResidentTokens => &mut coefficients.resident_tokens,
                    LoadField::Running => &mut coefficients.running,
                    LoadField::WaitingUncachedTokens => &mut coefficients.waiting_uncached_tokens,
                },
                CostTerm::SentRequests => &mut coefficients.sent_requests,
                CostTerm::SentTokens => &mut coefficients.sent_tokens,
            };
            debug_assert!(
                coefficient != 0.0,
                "a named term has a coefficient other than zero"
            );
            *field = coefficient;
        }
        coefficients
    }

    pub(crate) fn names_kv_or_running(&self) -> bool {
        self.active_kv != 0.0 || self.kv_usage != 0.0 || self.running != 0.0
    }

    /// Whether a term prices the sends by their tokens (`sent.tokens`). A
    /// send of text alone is counted there by its text's bytes, which a
    /// host that can send a request on without reading its prompt states
    /// only where a stage asks for a request's size
    /// ([`Needs::prompt_size`](crate::document::Needs::prompt_size)). So a
    /// stage that holds such a term asks for it.
    pub(crate) fn sizes_sends(&self) -> bool {
        self.sent_tokens != 0.0
    }

    /// Adds the sends' two terms to `sum` in name order: the requests,
    /// then their tokens, stated or estimated. The one routine a pass
    /// that prices a profile's count of sends closes a row's sum by. A
    /// pass reads the count first, so it holds no sum across that read.
    #[inline(always)]
    pub(crate) fn add_sent(&self, sum: &mut f64, sent: DispatchCounts) {
        *sum += f64::from(sent.requests) * self.sent_requests;
        *sum += sent.tokens as f64 * self.sent_tokens;
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
