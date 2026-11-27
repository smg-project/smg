use super::*;

#[test]
fn canonical_forms_accepted() {
    assert!(is_canonical_serving_tenant_key("anonymous"));
    assert!(is_canonical_serving_tenant_key("auth:team-red"));
    assert!(is_canonical_serving_tenant_key("header:team-red"));
    assert!(is_canonical_serving_tenant_key("ip:127.0.0.1"));
    assert!(is_canonical_serving_tenant_key("ip:::1"));
}

#[test]
fn non_canonical_forms_rejected() {
    // Unprefixed (Explicit-only, admin-not-serving-path form).
    assert!(!is_canonical_serving_tenant_key("team-red"));
    // Empty id after a valid prefix.
    assert!(!is_canonical_serving_tenant_key("auth:"));
    assert!(!is_canonical_serving_tenant_key("header:"));
    // Padded id -- the header value is always trimmed first.
    assert!(!is_canonical_serving_tenant_key("header: team-red"));
    assert!(!is_canonical_serving_tenant_key("header:team-red "));
    // Outside to_str()'s alphabet.
    assert!(!is_canonical_serving_tenant_key("header:héllo"));
    assert!(!is_canonical_serving_tenant_key("header:team\u{1}red"));
    // Not a real IP address.
    assert!(!is_canonical_serving_tenant_key("ip:not-an-ip"));
    // Parseable but non-canonical: expanded/uppercase IPv6 never
    // matches the compressed lowercase form `into_key()` produces.
    assert!(!is_canonical_serving_tenant_key(
        "ip:2001:0DB8:0000:0000:0000:0000:0000:0001"
    ));
    // Unknown prefix.
    assert!(!is_canonical_serving_tenant_key("tenant:team-red"));
    assert!(!is_canonical_serving_tenant_key(""));
}

/// Stored records are scoped to the identities the gateway vouches for;
/// an address or an anonymous caller is no tenant.
#[test]
fn storage_tenant_is_the_canonical_key_of_a_vouched_identity_only() {
    let tenant_of = |identity: TenantIdentity| {
        storage_tenant(&RouteRequestMeta::new(canonical_tenant_key(identity)))
    };
    assert_eq!(
        tenant_of(TenantIdentity::Authenticated(Arc::from("team-red"))).as_deref(),
        Some("auth:team-red")
    );
    assert_eq!(
        tenant_of(TenantIdentity::Header(Arc::from("team-red"))).as_deref(),
        Some("header:team-red")
    );
    assert_eq!(
        tenant_of(TenantIdentity::IpAddress("127.0.0.1".parse().unwrap())),
        None
    );
    assert_eq!(tenant_of(TenantIdentity::Anonymous), None);
}

/// A text past the stack buffer takes the two-step path and reads the
/// same as the one-allocation path.
#[test]
fn a_key_past_the_stack_buffer_keeps_its_form() {
    let long = "x".repeat(300);
    let key = TenantIdentity::Header(Arc::from(long.as_str())).into_key();
    assert_eq!(key.as_str(), format!("header:{long}"));
    assert!(is_canonical_serving_tenant_key(key.as_str()));
    let short = TenantIdentity::Header(Arc::from("team-red")).into_key();
    assert_eq!(short.as_str(), "header:team-red");
}

#[test]
fn stack_text_refuses_a_write_past_its_capacity_and_keeps_what_it_had() {
    let mut text = StackText::<8>::default();
    assert!(text.write_str("abcd").is_ok());
    assert!(text.write_str("efghi").is_err());
    assert_eq!(text.as_str(), Some("abcd"));
    assert!(text.write_str("efgh").is_ok());
    assert_eq!(text.as_str(), Some("abcdefgh"));
}

#[test]
fn matches_what_tenant_identity_actually_produces() {
    assert!(is_canonical_serving_tenant_key(
        canonical_tenant_key(TenantIdentity::Authenticated(Arc::from("team-red"))).as_str()
    ));
    assert!(is_canonical_serving_tenant_key(
        canonical_tenant_key(TenantIdentity::Header(Arc::from("team-red"))).as_str()
    ));
    assert!(is_canonical_serving_tenant_key(
        canonical_tenant_key(TenantIdentity::IpAddress("127.0.0.1".parse().unwrap())).as_str()
    ));
    assert!(is_canonical_serving_tenant_key(
        canonical_tenant_key(TenantIdentity::Anonymous).as_str()
    ));
}
