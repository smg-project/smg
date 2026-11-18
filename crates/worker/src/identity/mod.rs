//! Who a worker is: its address, parsed once and keyed, and the sampling
//! defaults a backend reports on its labels. The `runtime` feature adds the
//! errors a worker operation reports.

pub mod endpoint;
#[cfg(feature = "runtime")]
pub mod error;
pub mod sampling_defaults;
