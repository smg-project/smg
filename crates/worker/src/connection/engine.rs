//! A worker's engine connection as the worker runtime sees it.
//!
//! [`EngineConnection`] holds the calls the runtime makes on a connected
//! engine: liveness, the health and load reads and the admin calls, each in
//! the worker's own types. [`EngineConnector`] makes a connection. The host
//! implements both over its own client type and hands the connector to the
//! worker builder; the request path reads its own client type back from the
//! handle with `downcast_arc`.

use std::{any::Any, fmt, sync::Arc, time::Duration};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use tokio::sync::OnceCell;

use crate::worker::{ProfileOptions, RuntimeType, WorkerLoadResponse, WorkerType};

type ConnectionCell = OnceCell<Arc<dyn EngineConnection>>;

/// A worker's connection generation. Compatible replacements adopt the slot
/// itself so a probe from either worker can retire its observed generation.
pub struct EngineConnectionCache {
    slot: ArcSwap<ArcSwap<ConnectionCell>>,
}

impl EngineConnectionCache {
    /// Start with the cell the builder prepared, possibly already connected.
    pub(crate) fn new(cell: ConnectionCell) -> Self {
        Self {
            slot: ArcSwap::from_pointee(ArcSwap::from_pointee(cell)),
        }
    }

    /// Pin the generation for one acquisition, probe or handshake.
    pub fn load_full(&self) -> Arc<ConnectionCell> {
        self.slot.load().load_full()
    }

    /// Follow future reconnects as well as the connection established now.
    pub(crate) fn adopt(&self, previous: &Self) {
        self.slot.store(previous.slot.load_full());
    }

    /// Retire only the cell the probe found dead. A later generation may
    /// already be serving requests by the time this probe resumes.
    pub fn evict(&self, observed: &Arc<ConnectionCell>) {
        self.slot
            .load()
            .compare_and_swap(observed, Arc::new(OnceCell::new()));
    }
}

/// The reason a gRPC or ZMQ worker built without an [`EngineConnector`]
/// reports when it is asked to connect, with the two ways out.
pub const NO_ENGINE_CONNECTOR: &str = "the worker was built without an engine connector: \
    hand one to its builder, or build it through the host's name for the builder";

/// An engine's answer to an admin call, in the worker's own words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdminAnswer {
    /// Whether the engine did what it was asked.
    pub success: bool,
    /// What the engine said about it; an engine may say nothing.
    pub message: String,
}

/// The calls the worker runtime makes on a worker's connection to its
/// engine, over gRPC or ZMQ. The worker holds the host's client behind this
/// trait and shares it with the request path.
///
/// The engine's two event subscriptions are no call of this trait: they
/// answer the wire's own messages, so the host's event monitor reads them
/// from the host's client, read back through this handle. A connection of
/// another type feeds that monitor no KV events and no load records.
#[async_trait]
pub trait EngineConnection: Any + Send + Sync {
    /// Whether the engine's wire carries the admin calls: the cache flush
    /// and the profile start and stop.
    fn carries_admin_calls(&self) -> bool;

    /// Whether the connection still serves. A connection that cannot
    /// reconnect in place answers `false` once its engine is gone.
    fn is_alive(&self) -> bool;

    /// Ask the engine whether it is healthy.
    async fn health_check(&self) -> Result<bool, tonic::Status>;

    /// Read the engine's load report.
    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status>;

    /// Flush the engine's KV cache; `timeout_s` is the wait the engine is
    /// asked to honor.
    async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, tonic::Status>;

    /// Start a profiling run on the engine with the worker's `options`.
    async fn start_profile(&self, options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status>;

    /// Stop the profiling run in flight on the engine.
    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status>;

    /// The KV transfer engine id the engine's server info reports; `None`
    /// when it reports none.
    async fn kv_engine_id(&self) -> Result<Option<String>, tonic::Status>;
}

impl dyn EngineConnection {
    /// The connection as the host's own client type `T`: the same
    /// allocation, or `None` when another type stands behind the handle.
    pub fn downcast_arc<T: EngineConnection>(self: Arc<Self>) -> Option<Arc<T>> {
        let any: Arc<dyn Any + Send + Sync> = self;
        any.downcast().ok()
    }
}

/// What the ZMQ handshake needs to know of a worker: where it listens, what
/// it serves, the role it was registered with and the handshake's bounds.
#[derive(Debug)]
pub struct ZmqLink {
    /// The address requests dial, without a rank suffix.
    pub base_url: String,
    /// The model the worker serves.
    pub model_id: String,
    /// The engine's runtime, which selects the wire dialect.
    pub runtime: RuntimeType,
    /// The role the worker was registered with. A connector refuses an
    /// engine that declared another at the handshake.
    pub worker_type: WorkerType,
    /// The handshake address, when the worker's spec overrides the derived
    /// one.
    pub handshake_address: Option<String>,
    /// How many engines the link must find; `None` takes the count declared
    /// at the handshake.
    pub engine_count: Option<usize>,
    /// The per-stage silence bound of the startup handshake.
    pub startup: Duration,
    /// The bound on a request over the link, from its dispatch to its last
    /// message; unbounded when `None`.
    pub request_timeout: Option<Duration>,
}

/// Makes a worker's engine connection. The host implements it over its own
/// client. A worker gets it from its builder, handed in or as the builder's
/// default ([`ConnectorDefault`]); a worker with neither refuses a gRPC or
/// ZMQ connect.
///
/// The host's event monitor subscribes through the host's own client alone:
/// a connector that hands out another connection leaves its workers without
/// KV events, and their load polled.
#[async_trait]
pub trait EngineConnector: fmt::Debug + Send + Sync {
    /// Connect to the gRPC server of an engine of `runtime` at `base_url`.
    /// An error is the reason as the worker reports it.
    async fn connect_grpc(
        &self,
        base_url: &str,
        runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String>;

    /// Run the ZMQ handshake for `link` and check the role each engine
    /// declared against the worker's. An error is the reason as the worker
    /// reports it.
    async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String>;
}

/// The connector a builder gives a worker it was handed none for. It is a
/// type parameter of the builder, as a hasher is of a map: this module knows
/// no client, so its own default is [`NoConnector`], and a host names the
/// builder with the type that answers its own.
pub trait ConnectorDefault {
    /// The connector, or `None` for a builder without a default.
    fn default_connector() -> Option<Arc<dyn EngineConnector>>;
}

/// No default: a gRPC or ZMQ worker built without a connector refuses its
/// connect with [`NO_ENGINE_CONNECTOR`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NoConnector;

impl ConnectorDefault for NoConnector {
    fn default_connector() -> Option<Arc<dyn EngineConnector>> {
        None
    }
}
