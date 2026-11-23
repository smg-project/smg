//! A worker's load: its requests in flight, the monitor that polls the
//! workers and takes their pushed records, the counts of its polls that a
//! host reads, the port between it and the sink of what a load means for
//! routing, the load snapshots it publishes, the fleet's capacity, and the
//! merge of the metrics scraped from the workers into one exposition.

pub mod capacity;
pub mod counts;
pub mod inflight;
pub mod metrics_aggregator;
pub mod monitor;
pub mod port;
pub mod state;
