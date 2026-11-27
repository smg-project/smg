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
