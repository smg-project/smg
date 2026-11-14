//! The ingestion drivers (design 5.1 and 6): what fills the store from the
//! engines. This step holds the load path: the row conversion and the
//! scrape parser the gateway shares with the controller, and the driver of
//! the `GetLoads` poll and the metrics scrape over the store's
//! [`LoadSink`](crate::store::LoadSink) under its poll arbitration. The KV
//! subscription, the prober and discovery arrive in the step's later PRs.

pub mod loads;
pub mod probe;
pub mod rows;
pub mod scrape;
