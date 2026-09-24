//! The tenant identity a request carries into a router.

use std::sync::Arc;

use axum::http::Extensions;
use uuid::Uuid;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TenantKey(Arc<str>);

impl TenantKey {
    #[must_use]
    pub fn new(key: impl AsRef<str>) -> Self {
        Self(Arc::from(key.as_ref()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for TenantKey {
    fn from(value: String) -> Self {
        Self(Arc::from(value))
    }
}

impl From<&str> for TenantKey {
    fn from(value: &str) -> Self {
        Self(Arc::from(value))
    }
}

#[derive(Debug, Clone)]
pub struct RouteRequestMeta {
    pub tenant_key: TenantKey,
    pub request_charge_id: Uuid,
    extensions: Extensions,
}

impl RouteRequestMeta {
    #[must_use]
    pub fn new(tenant_key: TenantKey) -> Self {
        Self {
            tenant_key,
            request_charge_id: Uuid::now_v7(),
            extensions: Extensions::new(),
        }
    }

    #[must_use]
    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }

    #[must_use]
    pub fn request_charge_id(&self) -> Uuid {
        self.request_charge_id
    }

    #[must_use]
    pub fn with_extension<T>(mut self, value: T) -> Self
    where
        T: Clone + Send + Sync + 'static,
    {
        self.extensions.insert(value);
        self
    }

    #[must_use]
    pub fn extension<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        self.extensions.get::<T>()
    }
}

impl PartialEq for RouteRequestMeta {
    fn eq(&self, other: &Self) -> bool {
        self.tenant_key == other.tenant_key && self.request_charge_id == other.request_charge_id
    }
}

impl Eq for RouteRequestMeta {}

/// What a router receives about the tenant behind a request.
pub type TenantRequestMeta = RouteRequestMeta;
