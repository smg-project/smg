//! The ingestion drivers (design 5.1 and 6): what fills the store from the
//! engines. This step holds the load path: the row conversion, the scrape
//! parser, and the driver of the `GetLoads` poll and the metrics scrape
//! over the store's [`LoadSink`](crate::store::LoadSink) under its poll
//! arbitration. The KV subscription and the prober arrive in the step's
//! later PRs. Worker discovery runs in the controller, behind its
//! `discovery` feature.

pub mod loads;
pub mod probe;
pub mod rows;
pub mod scrape;
