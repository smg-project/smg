//! A worker's health: the thresholds past which a load report marks it
//! overloaded. The `runtime` feature adds its liveness, its circuit breaker
//! and its resilience settings.

#[cfg(feature = "runtime")]
pub mod circuit_breaker;
#[cfg(feature = "runtime")]
pub mod liveness;
pub mod overload;
#[cfg(feature = "runtime")]
pub mod resilience;
