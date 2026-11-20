//! The standalone policy controller: its server, protocol and clients, leader
//! election, replication to standbys and the drivers that read load from
//! workers.

#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "controller")]
pub mod controller;
pub mod elector;
#[cfg(feature = "ingest")]
pub mod ingest;
#[cfg(feature = "controller")]
pub mod proto;
pub mod replication;
#[cfg(feature = "client")]
pub mod watcher;
