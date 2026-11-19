//! Who a worker is: its address, parsed once and keyed, and the sampling
//! defaults a backend reports on its labels. The `runtime` feature adds the
//! worker itself, its builder and its metadata, and the errors a worker
//! operation reports.

#[cfg(feature = "runtime")]
pub mod basic;
#[cfg(feature = "runtime")]
pub mod builder;
pub mod endpoint;
#[cfg(feature = "runtime")]
pub mod error;
#[cfg(feature = "runtime")]
pub mod metadata;
pub mod sampling_defaults;
#[cfg(feature = "runtime")]
pub mod url_door;
