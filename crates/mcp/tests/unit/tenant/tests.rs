use super::*;

#[test]
fn test_tenant_id() {
    let id = TenantId::new("tenant-123");
    assert_eq!(id.as_str(), "tenant-123");
    assert_eq!(id.to_string(), "tenant-123");
}

#[test]
fn test_session_id_default() {
    let id1 = SessionId::default();
    let id2 = SessionId::default();
    assert_ne!(id1, id2);
}
