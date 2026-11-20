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
//! | `drain_settle_secs` | the checks' drain window, where the row states one; else the object's default |
//! | `signals`, `reachable_from` | not carried: the object has no field for them, and a collector reads them of the row by the id the holder keeps. A host that is no gateway has the object of a row one gateway alone reaches built with its checks off ([`Objects::with_host_local_unchecked`]) |
//! | `zone`, `pairing`, `encode`, `vision`, `max_running`, `priority`, `demand_facts` | not carried: a decision's facts, read of the store's rows and never of an object |
//! | `overload` | not carried to the object: a decision's fact, read of the store's rows. [`facts()`] states the thresholds of the object's own registration over the standing row's |
//!
//! A new object starts at its row's status. A replacement takes the
//! standing object's running state, including its status. A host that gets
//! health from the rows can use [`Objects::with_status_followed`] to copy
//! their status on tail changes and snapshots, for objects kept or laid
//! again. The holder checks the row's revision under the object's
//! mutation lock and hands the registry the status, as the health
//! adapter does: the registry's subscribers hear the change, and a
//! promotion comes back as any promotion does. The host's vetoes stand.
//! The id and the revision stand beside the object in the
//! holder ([`Standing`]). No credential and no connector is a row's fact. A
//! host hands the holder its connector ([`Objects::with_connector`]). It
//! keeps its workers' credentials in [`Credentials`] and hands the keeper
//! to the holder ([`Objects::with_credentials`]). A credential is bound to
//! the rows its registrations wrote: a row no registration of this host
//! stated takes none. Where a registration's answer says so, the host has
//! the holder lay the address's objects again ([`Objects::lay_again`]).
//! [`registration_at_door_keeping`] makes these calls with the keeper the
//! host's holder was built with.
//!
//! A host whose registrations run in its own process needs none of that.
//! Its registration built a whole object, and the holder keeps it as the
//! row's ([`Objects::with_built_kept`]): the door hands it in with the
//! write ([`Membership::found_built`], [`Objects::handing`]). The object
//! holds what the table does not carry, its credential first.
//!
//! # The adapters
//!
//! [`HealthAdapter`] writes probe outcomes and connect signals to the store
//! and sets the object's status to the status that stands there. The host
//! supplies the [`Lead`] and the term the adapter writes under. Its
//! projection ([`HealthAdapter::project`]) writes what the object knows of
//! its worker onto the row: the object's liveness veto is the row's stalled
//! bit, for every gateway. The worker crate's collector drives it through
//! the adapter: at each move of that veto, for an object laid over another,
//! and for every object where a collector starts. A host that runs the
//! collector calls no projection of its own.
//!
//! [`LoadAdapter`] writes load records and poll answers to the store. The
//! store decides when the collector polls and keeps the overload verdict.
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
//! [`registration_at_door`] writes the created workers' rows, and hands
//! each built worker to the door with its write.
//! [`removal_at_door`] marks the selected rows draining, waits for the
//! workers to settle and removes their rows. The steps write the store
//! through the door. [`Objects`] remains the registry's one writer.
//!
//! The health loop's refresh of a worker's engine has an entry of its own
//! ([`Membership::refreshed`]). It asks no maker, writes the row's labels
//! and no other fact, and hands the rebuilt worker in with the write.
//!
//! # What the module does not hold
//!
//! It builds no registry and no engine. The host builds the registry and
//! hands it in, and calls [`Objects::follow`] where the store moved.
//! [`KvFeed::run`], its follow of the cache events, is a future the host
//! drives. The collector under it starts a task for each worker it streams
//! and ends them at the stop. It names nothing of the controller.

mod credentials;
mod door;
mod found;
mod health;
mod kv;
mod kv_feed;
mod lead;
mod loads;
mod object;
mod objects;
pub mod steps;

pub use credentials::{Credentials, KeptCredential, RegistrationInFlight};
#[cfg(all(test, feature = "controller"))]
pub(crate) use door::tests::{
    check_event_signal, check_expectations, check_guard, check_health_scope, check_maker,
    check_provider, check_refresh,
};
#[cfg(feature = "controller")]
pub(crate) use door::{drain, lay, refresh, take, Turn};
pub use door::{Expect, Membership, Refused, Remover, StoreDoor, Wrote};
pub use found::facts;
pub use health::HealthAdapter;
pub use kv::{KvAdapter, KvHost, NoFeed, Unfed};
pub use kv_feed::KvFeed;
pub use lead::{Lead, SoleLead};
pub use loads::LoadAdapter;
pub use object::object;
pub use objects::{Members, Objects, Standing};
pub use steps::{
    maker_of, registration_at_door, registration_at_door_keeping, removal_at_door, remover_of,
    DrainWorkersAtDoor, NoKeeper, RemoveWorkersAtDoor, Unasked, WriteWorkersAtDoor, DISCOVERY,
    DRAIN_WORKERS_AT_DOOR_STEP, REMOVE_WORKERS_AT_DOOR_STEP, WRITE_WORKERS_AT_DOOR_STEP,
};
