//! A replica's worker objects: one object of the worker crate for every row
//! of the membership store that stands, and none beside.
//!
//! The store is the truth of who the workers are. What reads a worker's
//! signals (its loads, its cache events, its health) is the worker crate's
//! and works over that crate's object and registry. This module is the
//! projection between the two. [`Objects`] follows the store's membership
//! and writes the registry: a row arrives and its object is built, a row's
//! facts change and its object is replaced under the same id, a row goes and
//! its object is dropped. A leader and a standby run the same path over the
//! same rows.
//!
//! The holder is the registry's one writer. No job step writes it: under a
//! host that keeps a store a registration ends at the membership door, and
//! the row's change builds the object here.
//!
//! # The one table
//!
//! [`object()`] reads the table from row to object and [`facts()`] reads it
//! back. Each fact of the row lands in one field of the object, or is not
//! carried:
//!
//! | The row's fact | The object's field |
//! |---|---|
//! | `key` | `url` |
//! | `engine` | `dp_base_url`, the address the object dials, where the row names an engine that is not its key |
//! | `incarnation` | not carried: it is part of the row's id, which the holder keeps beside the object |
//! | `role` | `worker_type`, one to one |
//! | `wire_modes` | `connection_mode`: gRPC where the set holds it, else HTTP, else the host-local wire; an empty set is HTTP |
//! | `engine_kind` | `runtime_type`, as the worker crate parses the name; unspecified where it knows none |
//! | `models` | `models`, one card an id; no id is the wildcard |
//! | `dp_rank`, `dp_size` | `dp_rank`, `dp_size`, where the width is above one; else neither |
//! | `labels` | `labels` |
//! | `kv_block_size` | `kv_block_size` |
//! | `bootstrap_port` | `bootstrap_port` |
//! | `health` | the health checks: the two thresholds, the interval and the timeout in whole seconds, the checks off where the policy's probes are |
//! | `signals`, `reachable_from` | not carried: the object has no field for them, and a collector reads them of the row by the id the holder keeps |
//! | `zone`, `pairing`, `encode`, `vision`, `max_running`, `priority`, `overload` | not carried: a decision's facts, read of the store's rows and never of an object |
//!
//! A new object starts at its row's status. A replacement takes the
//! standing object's running state, including its status. A host that gets
//! health from the rows can use [`Objects::with_status_followed`] to copy
//! their status on tail changes and snapshots, for objects kept or laid
//! again. The holder checks the row's revision and writes under the
//! object's mutation lock. The host's breaker and vetoes still apply.
//! The id and the revision stand beside the object in the
//! holder ([`Standing`]). No credential and no connector is a row's fact. A
//! host hands the holder its connector ([`Objects::with_connector`]); a
//! credential has no door yet.
//!
//! # The adapters
//!
//! [`HealthAdapter`] writes probe outcomes and connect signals to the store
//! and sets the object's status to the status that stands there. The host
//! supplies the [`Lead`] and the term the adapter writes under.
//!
//! [`KvAdapter`] decodes the worker's KV events for the store's index, and
//! [`KvHost`] gives each subscription its row in the store. Every replica
//! runs the adapter over its own streams. The adapter keeps what a row's
//! stream showed of its cache groups, as the gateway's index does. A row
//! above rank 0 of a data-parallel worker gets no feed yet ([`NoFeed`]).
//! The module `kv` names the decode's limits.
//!
//! # The door
//!
//! The host hands out a [`Membership`] door. A host without election uses
//! [`StoreDoor`]. A replica that may lead uses the controller's door.
//!
//! # What the module does not hold
//!
//! It builds no registry, no engine and no task. The host builds the
//! registry and hands it in, and calls [`Objects::follow`] where the store
//! moved. It names nothing of the controller.

mod door;
mod found;
mod health;
mod kv;
mod lead;
mod object;
mod objects;

#[cfg(feature = "controller")]
pub(crate) use door::{drain, lay, take, Turn};
pub use door::{Membership, Refused, StoreDoor, Wrote};
pub use found::facts;
pub use health::HealthAdapter;
pub use kv::{KvAdapter, KvHost, NoFeed, Unfed};
pub use lead::{Lead, SoleLead};
pub use object::object;
pub use objects::{Members, Objects, Standing};
