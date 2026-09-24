//! Canonical tenant identity and per-request metadata for serving paths.

use std::{net::IpAddr, sync::Arc};

pub const DEFAULT_TENANT_HEADER_NAME: &str = "x-smg-tenant-id";
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum TenantIdentity {
    Authenticated(Arc<str>),
    Header(Arc<str>),
    IpAddress(IpAddr),
    Anonymous,
    Explicit(Arc<str>),
}

impl TenantIdentity {
    #[must_use]
    pub fn into_key(self) -> TenantKey {
        let key = match self {
            Self::Authenticated(id) => format!("auth:{id}"),
            Self::Header(id) => format!("header:{id}"),
            Self::IpAddress(addr) => format!("ip:{addr}"),
            Self::Anonymous => "anonymous".to_string(),
            Self::Explicit(key) => key.to_string(),
        };
        TenantKey::from(key)
    }
}

pub use smg_external_router::tenant::{RouteRequestMeta, TenantKey};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPlaneCaller {
    tenant_key: TenantKey,
}

impl DataPlaneCaller {
    #[must_use]
    pub fn new(tenant_key: TenantKey) -> Self {
        Self { tenant_key }
    }

    #[must_use]
    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }

    #[inline]
    #[must_use]
    pub fn authenticated_from_sha256(hash: [u8; 32]) -> Self {
        Self::new(authenticated_tenant_key_from_sha256(hash))
    }
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum TenantResolutionError {
    #[error("admin routes require an explicit target tenant id")]
    MissingTargetTenant,
}

#[must_use]
pub fn canonical_tenant_key(identity: TenantIdentity) -> TenantKey {
    identity.into_key()
}

/// Whether `key` is a form [`TenantIdentity::into_key`] can actually
/// produce for a *serving-path* request — i.e. one of `auth:<non-empty>`,
/// `header:<non-empty>`, `ip:<valid address>`, or exactly `anonymous`.
///
/// Deliberately excludes [`TenantIdentity::Explicit`]'s raw, unprefixed
/// form: that variant is only ever constructed by
/// [`resolve_admin_target_tenant_key`] for admin routes, never by
/// [`middleware::tenant_resolution`](crate::middleware::tenant_resolution)'s
/// serving-path resolver. Any consumer that only ever looks keys up
/// against serving-path-resolved tenants (e.g. rate-limit policy) can use
/// this to catch a key that can never match a real request instead of
/// silently never applying.
#[must_use]
pub fn is_canonical_serving_tenant_key(key: &str) -> bool {
    if key == "anonymous" {
        return true;
    }
    if let Some(id) = key.strip_prefix("auth:") {
        return !id.is_empty();
    }
    if let Some(id) = key.strip_prefix("header:") {
        // Must match what a real header can decode to: trimmed, and
        // within HeaderValue::to_str()'s alphabet (HTAB or 0x20..=0x7E --
        // verified empirically; non-ASCII always fails to_str()).
        return !id.is_empty()
            && id.trim() == id
            && id.bytes().all(|b| b == b'\t' || (0x20..=0x7e).contains(&b));
    }
    if let Some(addr) = key.strip_prefix("ip:") {
        // Must round-trip: parse() accepts non-canonical forms (e.g.
        // expanded IPv6) that into_key() would never produce.
        return addr
            .parse::<IpAddr>()
            .is_ok_and(|parsed| parsed.to_string() == addr);
    }
    false
}

#[inline]
#[must_use]
pub fn authenticated_tenant_key_from_sha256(hash: [u8; 32]) -> TenantKey {
    let mut key = String::with_capacity(5 + hash.len() * 2);
    key.push_str("auth:");
    for byte in hash {
        key.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        key.push(HEX_DIGITS[(byte & 0x0f) as usize] as char);
    }
    TenantKey::from(key)
}

pub fn resolve_admin_target_tenant_key(
    tenant_key: &str,
) -> Result<TenantKey, TenantResolutionError> {
    let tenant_key = tenant_key.trim();
    if tenant_key.is_empty() {
        return Err(TenantResolutionError::MissingTargetTenant);
    }

    Ok(canonical_tenant_key(TenantIdentity::Explicit(Arc::from(
        tenant_key,
    ))))
}

pub fn resolve_admin_target_tenant_id(
    tenant_key: &str,
) -> Result<RouteRequestMeta, TenantResolutionError> {
    resolve_admin_target_tenant_key(tenant_key).map(RouteRequestMeta::new)
}

#[cfg(test)]
mod tests {
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
}
