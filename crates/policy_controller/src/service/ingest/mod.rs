//! The ingestion drivers fill the store from the engines. The load driver
//! polls for reports and scrapes metrics under the store's poll rule
//! ([`LoadSink`](crate::store::LoadSink)). The scrape parser turns metrics
//! into reports. Worker discovery runs in the controller, behind its
//! `discovery` feature.

pub mod loads;
pub mod probe;
pub mod scrape;
