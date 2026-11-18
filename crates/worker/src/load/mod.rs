//! A worker's load: its requests in flight, and the merge of the metrics
//! scraped from the workers into one exposition.

pub mod inflight;
pub mod metrics_aggregator;
