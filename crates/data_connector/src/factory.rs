//! Factory function to create storage backends based on configuration.

use std::{sync::Arc, time::Duration};

use tokio::sync::oneshot;
use tracing::info;
use url::Url;

use crate::{
    bounded::{
        BoundedConversationItemStorage, BoundedConversationStorage, BoundedResponseStorage,
        CallBound,
    },
    config::{HistoryBackend, MemoryConfig, OracleConfig, PostgresConfig, RedisConfig},
    core::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    hooked::{HookedConversationItemStorage, HookedConversationStorage, HookedResponseStorage},
    hooks::StorageHook,
    memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage},
    noop::{NoOpConversationItemStorage, NoOpConversationStorage, NoOpResponseStorage},
    oracle::{OracleConversationItemStorage, OracleConversationStorage, OracleResponseStorage},
    postgres::{
        PostgresConversationItemStorage, PostgresConversationStorage, PostgresResponseStorage,
        PostgresStore,
    },
    redis::{
        RedisConversationItemStorage, RedisConversationStorage, RedisResponseStorage, RedisStore,
    },
};

/// Complete storage handles returned by the factory.
pub struct StorageBundle {
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
}

/// Configuration for creating storage backends
pub struct StorageFactoryConfig<'a> {
    pub backend: &'a HistoryBackend,
    pub oracle: Option<&'a OracleConfig>,
    pub postgres: Option<&'a PostgresConfig>,
    pub redis: Option<&'a RedisConfig>,
    /// Bounds of the memory backend; `None` = [`MemoryConfig::default`].
    pub memory: Option<&'a MemoryConfig>,
    /// Optional storage hook. When provided, all three storage backends are
    /// wrapped in `Hooked*Storage` that runs before/after hooks around every
    /// storage operation.
    pub hook: Option<Arc<dyn StorageHook>>,
}

/// Why the start-up contact with a store failed. The contact, one connection
/// and one round trip, is bounded by the backend's store-call bound
/// (`op_timeout_secs`) on Postgres and Redis, where the client library and
/// the pool bound their own steps at twice that, and by the pool's wait
/// (`pool_timeout_secs`) on Oracle, whose driver connects synchronously on a
/// blocking thread; so a store that accepts the connection and never answers
/// is the contact's own verdict, with the bound in the text, for every
/// backend; a store that refuses the connection is a verdict at once. The
/// gateway does not start with a history backend it cannot reach.
#[derive(Debug)]
pub(crate) enum StoreContactError {
    /// The connection or the round trip failed: refused, reset, rejected.
    Failed(String),
    /// Nothing came back within the bound.
    NoAnswer,
}

/// The start-up contact of a store whose driver connects synchronously:
/// `connect` runs on a thread of its own, outside the runtime's blocking
/// pool, and its answer is awaited under `bound` (`None`: no bound, as a
/// pool with no wait bound). Nothing back within the bound is `NoAnswer`; a
/// `connect` that failed, or whose thread ended without an answer (the
/// helper runs the connect and the bootstrap, so the text names neither), is
/// `Failed` with the cause. The bound ends the wait, not the thread: it
/// returns when the driver does, and the process's exit ends it. It is not
/// a `spawn_blocking` task because the runtime's drop waits for those: a
/// start-up that gave up on the connect must still hand its error to the
/// operator and exit.
pub(crate) async fn contact_blocking<T: Send + 'static>(
    bound: Option<Duration>,
    connect: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, StoreContactError> {
    let (tx, rx) = oneshot::channel();
    std::thread::Builder::new()
        .name("history-store-start-up".to_string())
        .spawn(move || {
            // The receiver is gone when the bound fired first: nothing to do.
            let _ = tx.send(connect());
        })
        .map_err(|e| {
            StoreContactError::Failed(format!("the start-up thread could not start: {e}"))
        })?;
    let answer = match bound {
        Some(bound) => match tokio::time::timeout(bound, rx).await {
            Ok(answer) => answer,
            Err(_elapsed) => return Err(StoreContactError::NoAnswer),
        },
        None => rx.await,
    };
    match answer {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(cause)) => Err(StoreContactError::Failed(cause)),
        Err(_gone) => Err(StoreContactError::Failed(
            "the start-up thread ended without an answer".to_string(),
        )),
    }
}

/// The start-up error for a store contact that failed: the backend, the store
/// (its URL with the password masked) and, when nothing came back, the bound.
fn contact_error(backend: &str, store: &str, bound: Duration, err: StoreContactError) -> String {
    match err {
        StoreContactError::Failed(cause) => {
            format!("the {backend} history backend at {store} is unreachable at start-up: {cause}")
        }
        StoreContactError::NoAnswer => format!(
            "the {backend} history backend at {store} did not answer within the start-up bound of {} s",
            bound.as_secs()
        ),
    }
}

/// The URL with its password masked, for log and error text and the
/// gateway's configuration view: the userinfo password and a `password` query
/// pair alike (the Postgres driver reads the latter too).
pub fn redacted_url(url: &str) -> String {
    match Url::parse(url) {
        Ok(mut parsed) => {
            if parsed.password().is_some() {
                let _ = parsed.set_password(Some("****"));
            }
            if parsed
                .query_pairs()
                .any(|(key, _)| key.eq_ignore_ascii_case("password"))
            {
                let pairs: Vec<(String, String)> = parsed
                    .query_pairs()
                    .map(|(key, value)| {
                        let value = if key.eq_ignore_ascii_case("password") {
                            "****".to_string()
                        } else {
                            value.into_owned()
                        };
                        (key.into_owned(), value)
                    })
                    .collect();
                parsed.query_pairs_mut().clear().extend_pairs(pairs);
            }
            parsed.to_string()
        }
        Err(_) => "<redacted>".to_string(),
    }
}

/// Create all configured storage handles.
///
/// # Errors
/// Returns error string if required configuration is missing or initialization fails
pub async fn create_storage(config: StorageFactoryConfig<'_>) -> Result<StorageBundle, String> {
    let bundle = match config.backend {
        HistoryBackend::Memory => {
            let memory = config.memory.copied().unwrap_or_default();
            memory.validate()?;
            info!(
                "Initializing data connector: Memory (at most {} responses and {} items, retention {})",
                memory.max_responses,
                memory.max_items,
                memory
                    .retention()
                    .map_or_else(|| "none".to_string(), |r| format!("{} s", r.as_secs()))
            );
            StorageBundle {
                response_storage: Arc::new(MemoryResponseStorage::with_config(&memory)),
                conversation_storage: Arc::new(MemoryConversationStorage::with_config(&memory)),
                conversation_item_storage: Arc::new(MemoryConversationItemStorage::with_config(
                    &memory,
                )),
            }
        }
        HistoryBackend::None => {
            info!("Initializing data connector: None (no persistence)");
            StorageBundle {
                response_storage: Arc::new(NoOpResponseStorage::new()),
                conversation_storage: Arc::new(NoOpConversationStorage::new()),
                conversation_item_storage: Arc::new(NoOpConversationItemStorage::new()),
            }
        }
        HistoryBackend::Oracle => {
            let oracle_cfg = config
                .oracle
                .ok_or("oracle configuration is required when history_backend=oracle")?;
            // The same numeric bounds as the Postgres and Redis paths, before
            // any connection.
            oracle_cfg
                .validate_bounds()
                .map_err(|e| format!("Invalid Oracle configuration: {e}"))?;

            info!(
                "Initializing data connector: Oracle ATP (pool: {}-{}, op_timeout_secs: {})",
                oracle_cfg.pool_min, oracle_cfg.pool_max, oracle_cfg.op_timeout_secs
            );

            // The driver runs each call on a blocking thread, which the bound
            // does not interrupt: the request is answered at the bound; the
            // thread returns when the driver does (each round trip of a pooled
            // connection ends within its own call timeout of twice the bound).
            let storages = bounded(
                create_oracle_storage(oracle_cfg).await?,
                CallBound::new("oracle", oracle_cfg.op_timeout()),
            );

            info!("Data connector initialized successfully: Oracle ATP");
            storages
        }
        HistoryBackend::Postgres => {
            let postgres_cfg = config
                .postgres
                .ok_or("Postgres configuration is required when history_backend=postgres")?;
            // Every entry path (the CLI, the Python bindings, a RouterConfig
            // built by hand) reaches this point: the numeric bounds are checked
            // here, so a bound of 0 s is refused by name instead of failing
            // every call once the gateway is up. The connection string is left
            // to the driver (the CLI's `validate()` checks its URL shape).
            postgres_cfg
                .validate_bounds()
                .map_err(|e| format!("Invalid Postgres configuration: {e}"))?;

            let log_db_url = redacted_url(&postgres_cfg.db_url);

            info!(
                "Initializing data connector: Postgres (db_url: {}, pool_max: {}, op_timeout_secs: {})",
                log_db_url, postgres_cfg.pool_max, postgres_cfg.op_timeout_secs
            );

            let storages = bounded(
                create_postgres_storage(postgres_cfg).await?,
                CallBound::new("postgres", postgres_cfg.op_timeout()),
            );

            info!("Data connector initialized successfully: Postgres");
            storages
        }
        HistoryBackend::Redis => {
            let redis_cfg = config
                .redis
                .ok_or("Redis configuration is required when history_backend=redis")?;
            // The same numeric bounds as the Postgres path; the URL is the
            // driver's (a socket path included).
            redis_cfg
                .validate_bounds()
                .map_err(|e| format!("Invalid Redis configuration: {e}"))?;

            let log_redis_url = redacted_url(&redis_cfg.url);

            info!(
                "Initializing data connector: Redis (url: {}, pool_max: {}, op_timeout_secs: {})",
                log_redis_url, redis_cfg.pool_max, redis_cfg.op_timeout_secs
            );

            let storages = bounded(
                create_redis_storage(redis_cfg).await?,
                CallBound::new("redis", redis_cfg.op_timeout()),
            );

            info!("Data connector initialized successfully: Redis");
            storages
        }
    };

    // Wrap backends in hooked storage when a hook is provided.
    if let Some(hook) = config.hook {
        info!("Wrapping storage backends with hook");
        Ok(StorageBundle {
            response_storage: Arc::new(HookedResponseStorage::new(
                bundle.response_storage,
                hook.clone(),
            )),
            conversation_storage: Arc::new(HookedConversationStorage::new(
                bundle.conversation_storage,
                hook.clone(),
            )),
            conversation_item_storage: Arc::new(HookedConversationItemStorage::new(
                bundle.conversation_item_storage,
                hook,
            )),
        })
    } else {
        Ok(bundle)
    }
}

/// Every call to a remote store under the backend's call bound (see the
/// `bounded` module); the hook wrappers go around these, so a hook's own time
/// does not count against the bound.
fn bounded(bundle: StorageBundle, bound: CallBound) -> StorageBundle {
    StorageBundle {
        response_storage: Arc::new(BoundedResponseStorage::new(bundle.response_storage, bound)),
        conversation_storage: Arc::new(BoundedConversationStorage::new(
            bundle.conversation_storage,
            bound,
        )),
        conversation_item_storage: Arc::new(BoundedConversationItemStorage::new(
            bundle.conversation_item_storage,
            bound,
        )),
    }
}

/// Create Oracle storage backends with a single shared connection pool.
async fn create_oracle_storage(oracle_cfg: &OracleConfig) -> Result<StorageBundle, String> {
    use crate::oracle::OracleStore;

    let store = OracleStore::new(oracle_cfg)?;
    // The start-up connection under the pool's wait, as the Postgres and
    // Redis contacts run under their call bound; a pool wait of 0 is no bound
    // (then nothing can fail to answer in time, and the text's 0 s is unused).
    let bound = oracle_cfg.pool_timeout();
    let conn = store.contact(bound).await.map_err(|err| {
        contact_error(
            "Oracle",
            &oracle_cfg.connect_descriptor,
            bound.unwrap_or_default(),
            err,
        )
    })?;
    // The tables, the schema version and the indexes, on that connection and
    // on the same kind of thread, so no round trip of the start-up runs on a
    // runtime worker; unbounded, as before: a migration takes what it takes,
    // and a store that stops answering after the login holds the start-up
    // here (named in the change, not changed).
    let bootstrap_store = store.clone();
    contact_blocking(None, move || {
        bootstrap_store.bootstrap(
            &conn,
            &[
                OracleConversationStorage::init_schema,
                OracleConversationItemStorage::init_schema,
                OracleResponseStorage::init_schema,
            ],
        )?;
        drop(conn);
        Ok(())
    })
    .await
    .map_err(|err| match err {
        StoreContactError::Failed(cause) => cause,
        StoreContactError::NoAnswer => "the start-up bootstrap gave no answer".to_string(),
    })?;

    Ok(StorageBundle {
        response_storage: Arc::new(OracleResponseStorage::new(store.clone())),
        conversation_storage: Arc::new(OracleConversationStorage::new(store.clone())),
        conversation_item_storage: Arc::new(OracleConversationItemStorage::new(store)),
    })
}

async fn create_postgres_storage(postgres_cfg: &PostgresConfig) -> Result<StorageBundle, String> {
    let store = PostgresStore::new(postgres_cfg.clone())?;
    let bound = postgres_cfg.op_timeout();
    store.contact(bound).await.map_err(|err| {
        contact_error("Postgres", &redacted_url(&postgres_cfg.db_url), bound, err)
    })?;
    // The tables, the schema version and the indexes: once per database, in
    // one transaction under the schema lock, so gateways starting together
    // take turns and a fresh database starts at the latest version.
    store
        .bootstrap()
        .await
        .map_err(|err| format!("failed to initialize the Postgres schema: {err}"))?;
    let postgres_resp = PostgresResponseStorage::new(store.clone())
        .await
        .map_err(|err| format!("failed to initialize Postgres response storage: {err}"))?;
    let postgres_conv = PostgresConversationStorage::new(store.clone());
    let postgres_item = PostgresConversationItemStorage::new(store);

    Ok(StorageBundle {
        response_storage: Arc::new(postgres_resp),
        conversation_storage: Arc::new(postgres_conv),
        conversation_item_storage: Arc::new(postgres_item),
    })
}

async fn create_redis_storage(redis_cfg: &RedisConfig) -> Result<StorageBundle, String> {
    let store = RedisStore::new(redis_cfg.clone())?;
    let bound = redis_cfg.op_timeout();
    store
        .contact(bound)
        .await
        .map_err(|err| contact_error("Redis", &redacted_url(&redis_cfg.url), bound, err))?;
    let redis_resp = RedisResponseStorage::new(store.clone());
    let redis_conv = RedisConversationStorage::new(store.clone());
    let redis_item = RedisConversationItemStorage::new(store);

    Ok(StorageBundle {
        response_storage: Arc::new(redis_resp),
        conversation_storage: Arc::new(redis_conv),
        conversation_item_storage: Arc::new(redis_item),
    })
}

#[cfg(test)]
#[path = "../tests/unit/factory/tests.rs"]
mod tests;
