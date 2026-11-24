//! A worker's health: the thresholds past which a load report marks it
//! overloaded. The `runtime` feature adds its liveness, its circuit breaker,
//! its resilience settings, the fleet's lifecycle (the probe schedule, the
//! status state machine, the admin fan-outs), the port between that
//! schedule (the port's collector) and the sink of what a probe's outcome
//! means for a worker's status, and the probe of the bootstrap port a
//! disaggregated worker advertises.

#[cfg(feature = "runtime")]
pub mod bootstrap_probe;
#[cfg(feature = "runtime")]
pub mod circuit_breaker;
#[cfg(feature = "runtime")]
pub mod liveness;
#[cfg(feature = "runtime")]
pub mod manager;
pub mod overload;
#[cfg(feature = "runtime")]
pub mod port;
#[cfg(feature = "runtime")]
pub mod resilience;
