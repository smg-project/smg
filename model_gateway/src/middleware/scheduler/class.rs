//! Priority class + `X-SMG-Priority` header parser.

/// HTTP request header that conveys the desired priority class.
///
/// Case-insensitive; unknown values degrade to [`Class::Default`].
pub const PRIORITY_HEADER: &str = "x-smg-priority";

/// Service class assigned to an inbound request.
///
/// Numeric values are load-bearing: `Ord` is derived so the tenant clamp is
/// `std::cmp::min(header_class, max_class)`, and `repr(u8)` lets the scheduler
/// pack per-class inflight counts into a single `AtomicU64`. Serde encoding is
/// lowercase so YAML files use `system`/`interactive`/`default`/`bulk`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum Class {
    /// Background / batch jobs. Lowest priority; preemptible by everyone else.
    Bulk = 0,
    /// Unlabeled traffic. Middle of the road; what every request gets when no
    /// header is set.
    Default = 1,
    /// Latency-sensitive traffic (chat completions, autocomplete).
    Interactive = 2,
    /// Internal control-plane traffic. Highest priority; never preempted by
    /// external clients (the tenant clamp prevents any external tenant from
    /// landing here in practice).
    System = 3,
}

impl Class {
    /// All four variants in ascending priority order.
    pub const ALL: [Class; 4] = [Self::Bulk, Self::Default, Self::Interactive, Self::System];

    /// Parse a header value into a class. Case-insensitive, whitespace-tolerant.
    /// Unknown values (including the empty string) map to [`Class::Default`]
    /// — admission shouldn't fail because of a typo in a non-essential header.
    ///
    /// Avoids allocating: `eq_ignore_ascii_case` does the comparison in place
    /// rather than building a lowercased `String` on every request.
    pub fn parse_header(value: &str) -> Class {
        Self::lookup(value).unwrap_or(Self::Default)
    }

    /// Parse a class name an operator configured
    /// (`--priority-scheduler-default-max-class`). Case-insensitive and
    /// whitespace-tolerant like the header, but an unknown name is an error
    /// naming the four classes: a typo in a flag must surface at start-up,
    /// where the header parser's fallback to [`Class::Default`] would hide it.
    pub fn parse_name(value: &str) -> Result<Class, UnknownClassName> {
        Self::lookup(value).ok_or_else(|| UnknownClassName(value.trim().to_owned()))
    }

    /// The class `value` names, trimmed and case-insensitively; `None` for
    /// anything else, the empty string included.
    fn lookup(value: &str) -> Option<Class> {
        let trimmed = value.trim();
        Self::ALL
            .into_iter()
            .find(|class| trimmed.eq_ignore_ascii_case(class.as_str()))
    }

    /// Lowercase variant name. Used as a metrics label and structured-log field
    /// only; never serialized back over the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bulk => "bulk",
            Self::Default => "default",
            Self::Interactive => "interactive",
            Self::System => "system",
        }
    }
}

/// A configured class name that is none of the four classes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown priority class `{0}`; expected one of system, interactive, default, bulk")]
pub struct UnknownClassName(pub String);

#[cfg(test)]
#[path = "../../../tests/unit/middleware/scheduler/class/tests.rs"]
mod tests;
