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
//! | `dp_rank`, `dp_size` | `dp_size` where the width is above one, and `dp_rank` with it unless the row is the engine whole: under its engine's own address at rank 0; else neither |
//! | `labels` | `labels` |
//! | `kv_block_size` | `kv_block_size` |
//! | `bootstrap_port` | `bootstrap_port` |
//! | `health` | the health checks: the two thresholds, the interval and the timeout in whole seconds, the checks off where the policy's probes are |
//! | `drain_settle_secs` | the checks' drain window, where the row states one; else the object's default |
//! | `signals`, `reachable_from` | not carried: the object has no field for them, and a collector reads them of the row by the id the holder keeps. A host that is no gateway has the object of a row one gateway alone reaches built with its checks off ([`Objects::with_host_local_unchecked`]) |
//! | `zone`, `encode`, `max_running`, `priority` | not carried: a decision's facts, read of the store's rows and never of an object |
//! | `vision`, `demand_facts` | not carried to the object: a decision's facts, which the holder keeps beside the object and lends with each list (`Objects::lent`) |
//! | `pairing` | not carried to the object: a decision's fact, read of the store's rows. The object derives its own at its build ([`pairing_descriptor`] reads it). [`facts()`] keeps the standing row's, and so does the controller's door, whose log carries none |
//! | `overload` | not carried to the object, where the store judges overload: a decision's fact, read of the store's rows. Under a host whose own load monitor judges it ([`Objects::with_overload_thresholds`]), the object's thresholds: the row's own, each over the host's default for its signal, the usage as a ratio. A row of the external runtime takes no default. [`facts()`] states the thresholds of the object's own registration over the standing row's. A replacement by id states them whole: one it leaves out is cleared ([`Over::Replaced`]) |
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
//! holder ([`RowRevision`]), with every fact of the row the object does not
//! hold as stated: inside the crate the two together state the row
//! (`Objects::row_of`). No credential and no connector is a row's fact. A
//! host hands the holder its connector ([`Objects::with_connector`]). It
//! keeps its workers' credentials in [`Credentials`] and hands the keeper
//! to the holder ([`Objects::with_credentials`]). A credential is bound to
//! the rows its registrations wrote: a row no registration of this host
//! stated takes none. Where a registration's answer says so, the host has
//! the holder lay the address's objects again ([`Objects::lay_again`]).
//! [`registration_into_store_keeping_credentials`] makes these calls with the
//! keeper the host's holder was built with.
//!
//! # The adapters
//!
//! [`HealthAdapter`] writes probe outcomes and connect signals to the store
//! and sets the object's status to the status that stands there. The host
//! supplies the [`Lead`] and the term the adapter writes under.
//!
//! [`load_monitor`] is the worker crate's own load monitor over the
//! holder's registry, with no adapter under it. It publishes the workers'
//! reports as its snapshot, which an engine that decides over its host's
//! lists reads, and writes the overload verdict on the objects. A host
//! runs it where its decisions read the snapshot, and writes no load to
//! the store then.
//!
//! [`KvAdapter`] decodes the worker's KV events for the store's index, and
//! [`KvHost`] gives each subscription its row in the store. Every replica
//! runs the adapter over its own streams. The adapter keeps what a row's
//! stream showed of its cache groups, as the gateway's index does. A row
//! above rank 0 of a data-parallel worker gets no feed yet ([`SkipReason`]).
//! The module `kv` names the decode's limits.
//!
//! # The door
//!
//! The host hands out a [`MembershipWriter`] door. The crate holds one: the
//! controller's, for a replica that may lead. It holds none over a store
//! alone.
//!
//! [`registration_into_store`] writes the created workers' rows, and hands
//! each built worker to the door with its write.
//! [`removal_from_store`] marks the selected rows draining, waits for the
//! workers to settle and removes their rows. The steps write the store
//! through the door. [`Objects`] remains the registry's one writer.
//!
//! The health loop's refresh of a worker's engine has an entry of its own
//! ([`MembershipWriter::refreshed`]). It asks no maker, and writes the row's
//! labels and no other fact.
//!
//! # What the module does not hold
//!
//! It builds no registry and no engine. The host builds the registry and
//! hands it in, and calls [`Objects::follow`] where the store moved.
//! [`KvFeed::run`], its follow of the cache events, is a future the host
//! drives. The collector under it starts a task for each worker it streams
//! and ends them at the stop. It names nothing of the controller.

mod credentials;
mod found;
mod health;
mod kv;
mod kv_feed;
mod lead;
mod loads;
mod members;
mod membership_writer;
mod object;
mod objects;
#[cfg(feature = "in-process")]
pub(crate) mod reach;
mod remainder;
pub(crate) mod steps;

pub use credentials::{Credentials, KeptCredential, RegistrationInFlight};
#[cfg(feature = "controller")]
pub(crate) use found::stated;
pub use found::{facts, pairing_descriptor};
pub use health::HealthAdapter;
pub use kv::{KvAdapter, KvHost, SkipReason, SkippedSubscriptions};
pub use kv_feed::KvFeed;
pub use lead::{Lead, SoleLead};
pub use loads::load_monitor;
pub use members::Members;
#[cfg(feature = "in-process")]
pub use members::{EncodeColumns, OpenAdmission};
#[cfg(all(test, feature = "controller"))]
pub(crate) use membership_writer::tests::{
    check_event_signal, check_expectations, check_guard, check_health_scope, check_maker,
    check_one_row, check_provider, check_rank_rows, check_refresh, FindOne, Rows, StoreDoor,
};
#[cfg(feature = "controller")]
pub(crate) use membership_writer::{
    drain, lay, lay_one, lay_ready, refresh, stating, take, Ready, Stated, Turn,
};
pub use membership_writer::{
    MembershipWriteOutcome, MembershipWriteRefused, MembershipWriter, Over, Precondition, Remover,
};
pub use object::object;
pub use objects::{Objects, RowRevision};
pub use steps::{
    registrant_of, registration_into_store, registration_into_store_keeping_credentials,
    removal_from_store, remover_of, DrainWorkersInStore, NoCredentials, RegistrantError,
    RemoveWorkersFromStore, WriteWorkersToStore, DISCOVERY, DRAIN_WORKERS_IN_STORE_STEP,
    REMOVE_WORKERS_FROM_STORE_STEP, WRITE_WORKERS_TO_STORE_STEP,
};
