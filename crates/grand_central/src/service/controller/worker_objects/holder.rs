//! The holder of the replica's worker objects over the controller's store.

use std::sync::Arc;

use smg_grpc_client::engine_port::GrpcConnector;
use smg_worker::{
    connection::engine::EngineConnector, health::overload::OverloadThresholds,
    registry::WorkerRegistry,
};

#[cfg(test)]
use crate::service::controller::Config;
use crate::{
    engine::CacheIndexes,
    store::MemoryStore,
    worker_host::{EventIndexes, KvTables, Members, Objects},
};

/// The replica's worker objects: their holder and the store their rows
/// stand in. The registry is built here, and the holder alone
/// writes it. Each object connects to its engine through the engine clients'
/// gRPC connector, so a worker of the host-local wire has no connection
/// here. The replica cannot dial a row that one gateway alone reaches: such
/// a row's object is built with its checks off, and no probe is sent to it.
///
/// The object of an engine's row states its row's overload thresholds over
/// the program's default, which the controller's start states
/// ([`Config::worker_overload`](crate::service::controller::Config::worker_overload)),
/// and an external row's object its row's alone: what a load monitor over
/// the registry judges the worker by
/// ([`load_monitor`](crate::worker_host::load_monitor)). One set a worker:
/// a registration's own values, each over the start's for an engine's
/// worker. The start's are stated while the document in force leaves its
/// overload switch on, and an engine's row states its own alone where
/// every binding switches it off: the document door states the word
/// ([`Objects::state_overload_switch`]).
/// The service runs that monitor while the replica leads, and it writes
/// each verdict on the worker's object.
///
/// An object's status leads its row's. A decision reads a worker's status
/// on its object. While the replica leads, a probe's outcome, a connect
/// signal and a drain are written on the object, and the store is told.
/// While it follows, the holder lays each status its leader's log tells.
/// So the holder follows no log of its own store.
pub(crate) struct WorkerObjects {
    store: Arc<MemoryStore>,
    objects: Arc<Objects<Members>>,
    /// The KV tables the holder was handed, kept beside it: a reader of the
    /// replica's tables is handed these, the ones its feed writes.
    tables: Arc<KvTables>,
}

impl WorkerObjects {
    /// The objects of `store`'s rows, none built yet: the write that makes
    /// a row lays its object. `overload` is the thresholds the object of an
    /// engine's row states for each signal its row names none for: the
    /// start's.
    pub(crate) fn stating(store: Arc<MemoryStore>, overload: OverloadThresholds) -> Self {
        Self::with_connector(store, Arc::new(GrpcConnector), overload)
    }

    /// [`Self::stating`] the default thresholds of a start, for a case that
    /// sets no others.
    #[cfg(test)]
    pub(crate) fn new(store: Arc<MemoryStore>) -> Self {
        Self::stating(store, Config::DEFAULT_WORKER_OVERLOAD)
    }

    /// [`Self::stating`] with `connector` making each object's engine
    /// connection.
    pub(super) fn with_connector(
        store: Arc<MemoryStore>,
        connector: Arc<dyn EngineConnector>,
        overload: OverloadThresholds,
    ) -> Self {
        // The KV tables are the replica's own, made here once: the holder
        // states its rows to them and the feed writes them.
        let tables = Arc::new(KvTables::default());
        Self {
            store,
            objects: Arc::new(
                Objects::new(Arc::new(WorkerRegistry::new()))
                    .with_connector(connector)
                    .with_host_local_unchecked()
                    .with_overload_thresholds(overload)
                    .with_kv_tables(Arc::clone(&tables)),
            ),
            tables,
        }
    }

    /// The store the objects' rows stand in.
    pub(crate) fn store(&self) -> Arc<MemoryStore> {
        Arc::clone(&self.store)
    }

    /// The replica's KV tables: the ones the holder was handed at its build.
    pub(crate) fn kv_tables(&self) -> Arc<KvTables> {
        Arc::clone(&self.tables)
    }

    /// The indexes of cache events the replica's engine reads: its KV
    /// tables' own, a worker found by its address among the worker objects.
    /// The store is lent them once, at the controller's build.
    pub(crate) fn cache_indexes(&self) -> Arc<dyn CacheIndexes> {
        Arc::new(EventIndexes::new(Arc::clone(&self.tables), &self.objects))
    }

    /// The holder shared with readers of the replica's worker objects.
    pub(crate) fn holder(&self) -> Arc<Objects<Members>> {
        Arc::clone(&self.objects)
    }

    /// The holder, for a case that reads what stands.
    #[cfg(test)]
    pub(crate) fn objects(&self) -> &Objects<Members> {
        &self.objects
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/service/controller/worker_objects/holder/tests.rs"]
pub(crate) mod tests;
