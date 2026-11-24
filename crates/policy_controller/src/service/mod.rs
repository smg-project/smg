//! The standalone policy controller: its server, protocol and clients, leader
//! election, replication to standbys and the load monitor over its
//! workers.
//!
//! `worker_follower` is the task a gateway follows a controller's worker
//! list with.

#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "controller")]
pub mod controller;
pub mod elector;
#[cfg(feature = "controller")]
pub mod proto;
pub mod replication;
#[cfg(feature = "client")]
pub mod watcher;
#[cfg(feature = "client")]
pub mod worker_follower;
