use super::*;

#[test]
fn test_role_parsing() {
    assert_eq!("admin".parse::<Role>().unwrap(), Role::Admin);
    assert_eq!("ADMIN".parse::<Role>().unwrap(), Role::Admin);
    assert_eq!("user".parse::<Role>().unwrap(), Role::User);
    assert_eq!("USER".parse::<Role>().unwrap(), Role::User);
    assert!("invalid".parse::<Role>().is_err());
}

#[test]
fn test_role_display() {
    assert_eq!(Role::Admin.to_string(), "admin");
    assert_eq!(Role::User.to_string(), "user");
}

#[test]
fn test_role_is_admin() {
    assert!(Role::Admin.is_admin());
    assert!(!Role::User.is_admin());
}

#[test]
fn test_jwt_config_builder() {
    let config = JwtConfig::new("https://issuer.example.com", "api://my-app")
        .with_jwks_uri("https://issuer.example.com/.well-known/jwks.json")
        .with_role_mapping("Gateway.Admin", Role::Admin)
        .with_role_mapping("Gateway.User", Role::User);

    assert_eq!(config.issuer, "https://issuer.example.com");
    assert_eq!(config.audience, "api://my-app");
    assert!(config.jwks_uri.is_some());
    assert_eq!(config.role_mapping.len(), 2);
}

#[test]
fn test_control_plane_auth_config() {
    let mut config = ControlPlaneAuthConfig::default();
    assert!(!config.is_enabled());
    assert!(!config.has_jwt());
    assert!(!config.has_api_keys());

    config.jwt = Some(JwtConfig::new("https://issuer.example.com", "api://test"));
    assert!(config.is_enabled());
    assert!(config.has_jwt());

    config.api_keys.push(ApiKeyEntry::new(
        "test-key",
        "Test Key",
        "secret123",
        Role::Admin,
    ));
    assert!(config.has_api_keys());
}

#[test]
fn test_find_api_key() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![
            ApiKeyEntry::new("key1", "Key 1", "secret1", Role::Admin),
            ApiKeyEntry::new("key2", "Key 2", "secret2", Role::User),
        ],
        audit_enabled: true,
    };

    let found = config.find_api_key("secret1");
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, "key1");
    assert_eq!(found.unwrap().role, Role::Admin);

    let found = config.find_api_key("secret2");
    assert!(found.is_some());
    assert_eq!(found.unwrap().role, Role::User);

    assert!(config.find_api_key("invalid").is_none());
}

#[test]
fn test_api_key_hashing() {
    let entry = ApiKeyEntry::new("test", "Test Key", "my-secret-key", Role::Admin);

    // Verify the correct key works
    assert!(entry.verify("my-secret-key"));

    // Verify wrong keys don't work
    assert!(!entry.verify("wrong-key"));
    assert!(!entry.verify("my-secret-ke")); // prefix
    assert!(!entry.verify("my-secret-keyy")); // suffix
    assert!(!entry.verify("")); // empty
}

#[test]
fn test_api_key_debug_redacts_hash() {
    let entry = ApiKeyEntry::new("test", "Test Key", "secret", Role::Admin);
    let debug_str = format!("{entry:?}");

    // Should contain REDACTED, not the actual hash
    assert!(debug_str.contains("[REDACTED]"));
    // Should not contain the key or hash bytes
    assert!(!debug_str.contains("secret"));
}

#[test]
fn a_match_is_the_entry_whose_key_matched() {
    let set = ApiKeySet::new(vec![
        ApiKeyEntry::new("a", "A", "key-a", Role::Admin),
        ApiKeyEntry::new("b", "B", "key-b", Role::User),
        ApiKeyEntry::new("c", "C", "key-c", Role::User),
    ]);
    let found = set.find("key-c").expect("the last entry's key matches");
    assert_eq!(
        (found.id.as_str(), found.name.as_str(), found.role),
        ("c", "C", Role::User)
    );
    assert_eq!(
        set.find("key-a").map(|found| found.id.clone()).as_deref(),
        Some("a")
    );
    assert!(set.find("key-d").is_none());
}

#[test]
fn a_match_reads_the_set_it_was_found_in_after_a_reload() {
    let set = ApiKeySet::new(vec![ApiKeyEntry::new("a", "A", "key-a", Role::Admin)]);
    let found = set
        .find("key-a")
        .expect("the key matches before the reload");
    set.replace(vec![ApiKeyEntry::new("z", "Z", "key-z", Role::User)]);
    assert_eq!(found.id, "a");
    assert!(set.find("key-a").is_none());
    assert_eq!(
        set.find("key-z").map(|found| found.id.clone()).as_deref(),
        Some("z")
    );
}
