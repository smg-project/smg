//! Canonical tenant identity and per-request metadata for serving paths.

use std::{
    fmt::{self, Write as _},
    net::IpAddr,
    sync::{Arc, LazyLock},
};

pub const DEFAULT_TENANT_HEADER_NAME: &str = "x-smg-tenant-id";

/// The longest tenant id accepted from a trusted header or a tenant API key:
/// the canonical key built from it (`header:<id>`, `auth:<id>`) is stored
/// with every response and conversation, in a 255-byte column on the
/// SQL backends.
pub const MAX_TENANT_ID_LEN: usize = 200;
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
    /// The canonical key, allocated once: a prefixed form is laid out on the
    /// stack and copied into the key's shared text, the anonymous key is one
    /// kept value, an explicit key shares the text it was given.
    #[must_use]
    pub fn into_key(self) -> TenantKey {
        match self {
            Self::Authenticated(id) => prefixed_key("auth:", &id),
            Self::Header(id) => prefixed_key("header:", &id),
            Self::IpAddress(addr) => prefixed_key("ip:", &addr),
            Self::Anonymous => ANONYMOUS_TENANT_KEY.clone(),
            Self::Explicit(key) => TenantKey::from(key),
        }
    }
}

/// The key of a caller the gateway knows nothing about.
static ANONYMOUS_TENANT_KEY: LazyLock<TenantKey> = LazyLock::new(|| TenantKey::new("anonymous"));

/// `<prefix><id>` as one allocation: formatted onto the stack (a serving-path
/// id is at most [`MAX_TENANT_ID_LEN`] bytes), then copied once into the key;
/// a longer text takes the two-step path through a `String`.
fn prefixed_key(prefix: &str, id: &dyn fmt::Display) -> TenantKey {
    let mut text = StackText::<256>::default();
    if write!(text, "{prefix}{id}").is_ok() {
        if let Some(text) = text.as_str() {
            return TenantKey::new(text);
        }
    }
    TenantKey::from(format!("{prefix}{id}"))
}

/// A fixed-capacity text buffer on the stack, written through [`fmt::Write`];
/// a write past the capacity fails and leaves the buffer as it was.
struct StackText<const N: usize> {
    bytes: [u8; N],
    len: usize,
}

impl<const N: usize> Default for StackText<N> {
    fn default() -> Self {
        Self {
            bytes: [0; N],
            len: 0,
        }
    }
}

impl<const N: usize> StackText<N> {
    /// The text written so far (every write copies whole strings, so the
    /// bytes are always a valid text).
    fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.bytes[..self.len]).ok()
    }
}

impl<const N: usize> fmt::Write for StackText<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self
            .len
            .checked_add(s.len())
            .filter(|end| *end <= N)
            .ok_or(fmt::Error)?;
        self.bytes[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

pub use smg_external_router::tenant::{storage_tenant, RouteRequestMeta, TenantKey};

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
}
