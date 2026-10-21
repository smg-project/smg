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
mod tests {
    use super::*;

    #[test]
    fn test_parse_header_known_values() {
        assert_eq!(Class::parse_header("system"), Class::System);
        assert_eq!(Class::parse_header("interactive"), Class::Interactive);
        assert_eq!(Class::parse_header("default"), Class::Default);
        assert_eq!(Class::parse_header("bulk"), Class::Bulk);
    }

    #[test]
    fn test_parse_header_is_case_insensitive() {
        assert_eq!(Class::parse_header("Bulk"), Class::Bulk);
        assert_eq!(Class::parse_header("INTERACTIVE"), Class::Interactive);
        assert_eq!(Class::parse_header("SyStEm"), Class::System);
    }

    #[test]
    fn test_parse_header_unknown_defaults_to_default() {
        assert_eq!(Class::parse_header("urgent"), Class::Default);
        assert_eq!(Class::parse_header(""), Class::Default);
        assert_eq!(Class::parse_header("123"), Class::Default);
    }

    #[test]
    fn test_parse_name_accepts_the_four_classes_in_any_case() {
        assert_eq!(Class::parse_name("system"), Ok(Class::System));
        assert_eq!(Class::parse_name("Interactive"), Ok(Class::Interactive));
        assert_eq!(Class::parse_name(" default "), Ok(Class::Default));
        assert_eq!(Class::parse_name("BULK"), Ok(Class::Bulk));
    }

    #[test]
    fn test_parse_name_refuses_an_unknown_name_naming_the_four_classes() {
        for unknown in ["gold", "urgent", "", "123"] {
            let err = Class::parse_name(unknown).unwrap_err();
            assert_eq!(err, UnknownClassName(unknown.to_owned()));
            let message = err.to_string();
            assert!(message.contains(&format!("`{unknown}`")), "{message}");
            for class in Class::ALL {
                assert!(message.contains(class.as_str()), "{message}");
            }
        }
    }

    #[test]
    fn test_parse_header_tolerates_whitespace() {
        assert_eq!(Class::parse_header("  bulk  "), Class::Bulk);
        assert_eq!(Class::parse_header("\tinteractive\n"), Class::Interactive);
    }

    #[test]
    fn test_ord_min_implements_tenant_clamp() {
        // Tenant clamp = min(header_class, max_class).
        assert_eq!(
            std::cmp::min(Class::Interactive, Class::Default),
            Class::Default
        );
        assert_eq!(
            std::cmp::min(Class::System, Class::Interactive),
            Class::Interactive
        );
        assert_eq!(std::cmp::min(Class::Bulk, Class::System), Class::Bulk);
    }

    #[test]
    fn test_all_is_ascending() {
        assert_eq!(
            Class::ALL,
            [
                Class::Bulk,
                Class::Default,
                Class::Interactive,
                Class::System
            ]
        );
        // Ord order matches numeric order.
        assert!(Class::Bulk < Class::Default);
        assert!(Class::Default < Class::Interactive);
        assert!(Class::Interactive < Class::System);
    }

    #[test]
    fn test_as_str_returns_lowercase_variant_name() {
        assert_eq!(Class::Bulk.as_str(), "bulk");
        assert_eq!(Class::Default.as_str(), "default");
        assert_eq!(Class::Interactive.as_str(), "interactive");
        assert_eq!(Class::System.as_str(), "system");
    }

    #[test]
    fn test_priority_header_constant() {
        assert_eq!(PRIORITY_HEADER, "x-smg-priority");
    }
}
