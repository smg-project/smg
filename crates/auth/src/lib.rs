//! Authentication and authorization module for control plane APIs.
//!
//! This module provides:
//! - JWT/OIDC authentication for external IDP integration
//! - API key authentication with role-based access
//! - Audit logging for control plane operations
//! - Middleware for securing admin and worker routes

mod audit;
mod config;
mod jwks;
mod jwt;
mod middleware;

pub use audit::{AuditEvent, AuditLogger, AuditOutcome};
pub use config::{ApiKeyEntry, ControlPlaneAuthConfig, JwtConfig, Role};
pub use jwt::{JwtValidator, JwtValidatorError};
pub use middleware::{
    bearer_token, control_plane_auth_middleware, AuthMethod, ControlPlaneAuthState, Principal,
    PrincipalExt,
};

/// Request ID for correlation in audit logs.
///
/// This type can be added to request extensions to provide request IDs
/// for audit logging. The middleware will extract this from request
/// extensions if present.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestId(pub String);
