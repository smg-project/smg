use super::*;

#[test]
fn test_from_rmcp() {
    // `RmcpToolAnnotations` is `#[non_exhaustive]` in rmcp 1.7; build via
    // the chained setters instead of a struct literal.
    let rmcp = RmcpToolAnnotations::new()
        .read_only(true)
        .destructive(false)
        .idempotent(true)
        .open_world(false);
    let ann = ToolAnnotations::from_rmcp(&rmcp);
    assert!(ann.read_only);
    assert!(!ann.destructive);
}

#[test]
fn test_conservative_defaults() {
    let rmcp = RmcpToolAnnotations::new();
    let ann = ToolAnnotations::from_rmcp(&rmcp);
    assert!(!ann.read_only); // assume writes
    assert!(ann.destructive); // assume dangerous
    assert!(!ann.idempotent); // assume not safe to retry
    assert!(ann.open_world); // assume external access
}

#[test]
fn test_should_require_approval() {
    assert!(ToolAnnotations::new()
        .with_destructive(true)
        .should_require_approval());
    assert!(!ToolAnnotations::new()
        .with_destructive(true)
        .with_read_only(true)
        .should_require_approval());
}
