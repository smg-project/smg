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

            info!(
                "Initializing data connector: Oracle ATP (pool: {}-{})",
                oracle_cfg.pool_min, oracle_cfg.pool_max
            );

            let storages = create_oracle_storage(oracle_cfg).await?;

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
mod tests {
    use std::{net::TcpListener, time::Instant};

    use serde_json::json;

    use super::*;
    use crate::{
        core::{NewConversation, NewConversationItem, StoredResponse},
        test_support::RefusedAddress,
    };

    /// A listener that completes the TCP handshake and never answers: the
    /// store that accepts bytes and says nothing.
    fn black_hole() -> (TcpListener, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local port");
        let port = listener.local_addr().expect("local address").port();
        (listener, port)
    }

    /// The store-call bound of the test configurations, the start-up
    /// contact's bound: short, so the black-hole tests are quick, and under
    /// the client's and pool's backstops (twice it) by construction.
    const BOUND_SECS: u64 = 2;
    const BOUND: Duration = Duration::from_secs(BOUND_SECS);

    fn postgres_config(port: u16) -> PostgresConfig {
        PostgresConfig {
            db_url: format!("postgres://u:s3cr3t@127.0.0.1:{port}/db"),
            pool_max: 1,
            op_timeout_secs: BOUND_SECS,
            schema: None,
        }
    }

    #[test]
    fn redacted_masks_the_password_in_the_userinfo_and_in_the_query() {
        assert_eq!(
            redacted_url("postgres://u:s3cr3t@h:5432/db"),
            "postgres://u:****@h:5432/db"
        );
        assert_eq!(
            redacted_url("postgres://u@h/db?password=s3cr3t&sslmode=require"),
            "postgres://u@h/db?password=****&sslmode=require"
        );
        assert_eq!(
            redacted_url("postgres://u:s3cr3t@h/db?Password=s3cr3t"),
            "postgres://u:****@h/db?Password=****"
        );
        assert_eq!(
            redacted_url("redis://:s3cr3t@h:6379/0"),
            "redis://:****@h:6379/0"
        );
        assert_eq!(redacted_url("not a url"), "<redacted>");
    }

    fn redis_config(port: u16) -> RedisConfig {
        RedisConfig {
            url: format!("redis://:s3cr3t@127.0.0.1:{port}/0"),
            pool_max: 1,
            op_timeout_secs: BOUND_SECS,
            schema: None,
            retention_days: None,
        }
    }

    /// The tests' hang guard: a start-up verdict, the contact's own at
    /// `BOUND` or a backstop's at twice it, must come within it; the guard
    /// only turns a start-up that never answers into a failure. A loaded
    /// runner changes the tests' duration, never their verdict.
    const HANG_GUARD: Duration = Duration::from_secs(30);

    /// `create_storage` for one remote backend, given up on at the hang
    /// guard: `None` is a start-up that never gave a verdict.
    async fn start_up(
        postgres: Option<&PostgresConfig>,
        redis: Option<&RedisConfig>,
    ) -> Option<Result<StorageBundle, String>> {
        let backend = if postgres.is_some() {
            HistoryBackend::Postgres
        } else {
            HistoryBackend::Redis
        };
        tokio::time::timeout(
            HANG_GUARD,
            create_storage(StorageFactoryConfig {
                backend: &backend,
                oracle: None,
                postgres,
                redis,
                memory: None,
                hook: None,
            }),
        )
        .await
        .ok()
    }

    #[tokio::test]
    async fn a_postgres_store_that_accepts_and_never_answers_is_refused_within_the_start_up_bound()
    {
        let (_hole, port) = black_hole();
        let started = Instant::now();
        let verdict = start_up(Some(&postgres_config(port)), None)
            .await
            .expect("the start-up contact gives a verdict within its bound");
        let err = verdict
            .err()
            .expect("a store that never answers is refused at start-up");
        assert!(
            err.contains("Postgres history backend")
                && err.contains("did not answer within the start-up bound of 2 s"),
            "{err}"
        );
        assert!(!err.contains("s3cr3t"), "the password is masked: {err}");
        assert!(
            started.elapsed() >= BOUND,
            "the verdict waited for the bound"
        );
    }

    #[tokio::test]
    async fn a_redis_store_that_accepts_and_never_answers_is_refused_within_the_start_up_bound() {
        let (_hole, port) = black_hole();
        let started = Instant::now();
        let verdict = start_up(None, Some(&redis_config(port)))
            .await
            .expect("the start-up contact gives a verdict within its bound");
        let err = verdict
            .err()
            .expect("a store that never answers is refused at start-up");
        // The client's connect and the pool's create are bounded at twice the
        // store-call bound: the start-up contact's own verdict comes first,
        // the same as for Postgres. Only that verdict names the bound (a
        // backstop's reads "is unreachable at start-up: ..."), so the text
        // tells which came first; no clock on the test's side judges it.
        assert!(
            err.contains("Redis history backend")
                && err.contains("did not answer within the start-up bound of 2 s"),
            "{err}"
        );
        assert!(!err.contains("s3cr3t"), "the password is masked: {err}");
        assert!(
            started.elapsed() >= BOUND,
            "the verdict waited for the bound"
        );
    }

    /// A start-up connect that never returns (a store that accepts and never
    /// answers behind a driver that connects synchronously) is a verdict at
    /// the bound; a connect that fails is a verdict at once; no bound is no
    /// bound. The bound ends the wait, not the connect's thread, which runs
    /// on, outside the runtime.
    #[tokio::test]
    async fn a_blocking_start_up_connect_that_never_returns_is_refused_at_the_bound() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let returned = Arc::new(AtomicBool::new(false));
        let flag = returned.clone();
        let started = Instant::now();
        let verdict = contact_blocking(Some(Duration::from_secs(1)), move || {
            std::thread::sleep(Duration::from_secs(3));
            flag.store(true, Ordering::SeqCst);
            Ok::<(), String>(())
        })
        .await;
        let elapsed = started.elapsed();
        assert!(
            matches!(verdict, Err(StoreContactError::NoAnswer)),
            "{verdict:?}"
        );
        assert!(
            elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(3),
            "verdict after {elapsed:?}; the bound is 1 s"
        );
        assert!(
            !returned.load(Ordering::SeqCst),
            "the connect's thread is still running when the verdict is given"
        );

        let failed = contact_blocking(Some(Duration::from_secs(1)), || {
            Err::<(), String>("refused".into())
        })
        .await;
        assert!(
            matches!(failed, Err(StoreContactError::Failed(ref cause)) if cause == "refused"),
            "{failed:?}"
        );
        let unbounded = contact_blocking(None, || Ok::<u8, String>(7)).await;
        assert!(matches!(unbounded, Ok(7)), "{unbounded:?}");
    }

    /// The connect the bound gave up on runs on a thread of its own, not in
    /// the runtime's blocking pool: a runtime that shuts down after the
    /// verdict does not wait for it, so the start-up error reaches the
    /// operator and the process exits. A connect in the blocking pool holds
    /// the runtime's drop for as long as the driver blocks, which is what this
    /// test refuses: the runtime of its own, the verdict at the bound, then
    /// the drop, all within a few seconds while the connect sleeps on.
    #[test]
    fn a_start_up_connect_the_bound_gave_up_on_does_not_hold_the_runtime_s_shutdown() {
        use std::sync::mpsc;

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("a runtime of the test's own");
            let verdict = runtime.block_on(contact_blocking(Some(Duration::from_secs(1)), || {
                std::thread::sleep(Duration::from_secs(30));
                Ok::<(), String>(())
            }));
            assert!(
                matches!(verdict, Err(StoreContactError::NoAnswer)),
                "{verdict:?}"
            );
            drop(runtime);
            let _ = done_tx.send(());
        });
        done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the runtime shut down while the connect's thread still ran");
    }

    /// The Oracle start-up against a store that accepts and never answers
    /// gives a verdict within the pool wait: the bound's own where the driver
    /// connects, a refusal at once where it cannot (its client library absent
    /// on the test host); never a start that does not end.
    #[tokio::test]
    async fn an_oracle_store_that_accepts_and_never_answers_is_refused_within_the_start_up_bound() {
        let (_hole, port) = black_hole();
        let oracle = OracleConfig {
            wallet_path: None,
            connect_descriptor: format!("127.0.0.1:{port}/svc"),
            external_auth: false,
            username: "u".to_string(),
            password: "s3cr3t".to_string(),
            pool_min: 1,
            pool_max: 1,
            pool_timeout_secs: BOUND_SECS,
            schema: None,
        };
        let verdict = tokio::time::timeout(
            HANG_GUARD,
            create_storage(StorageFactoryConfig {
                backend: &HistoryBackend::Oracle,
                oracle: Some(&oracle),
                postgres: None,
                redis: None,
                memory: None,
                hook: None,
            }),
        )
        .await
        .expect("the start-up gives a verdict within its bound");
        let err = verdict
            .err()
            .expect("a store that never answers is refused at start-up");
        assert!(
            err.contains("Oracle history backend")
                && (err.contains("did not answer within the start-up bound of 2 s")
                    || err.contains("is unreachable at start-up")),
            "{err}"
        );
        assert!(
            !err.contains("s3cr3t"),
            "the password is not in the text: {err}"
        );
    }

    #[tokio::test]
    async fn an_unreachable_redis_store_is_refused_at_start_up_like_postgres() {
        let refused = RefusedAddress::reserve();
        let err = start_up(None, Some(&redis_config(refused.port())))
            .await
            .expect("the start-up contact gives a verdict within its bound")
            .err()
            .expect("an unreachable store is refused at start-up");
        assert!(
            err.contains("Redis history backend") && err.contains("is unreachable at start-up"),
            "{err}"
        );
        assert!(!err.contains("s3cr3t"), "the password is masked: {err}");
    }

    #[tokio::test]
    async fn an_unreachable_postgres_store_is_refused_at_start_up() {
        let refused = RefusedAddress::reserve();
        let err = start_up(Some(&postgres_config(refused.port())), None)
            .await
            .expect("the start-up contact gives a verdict within its bound")
            .err()
            .expect("an unreachable store is refused at start-up");
        assert!(
            err.contains("Postgres history backend") && err.contains("is unreachable at start-up"),
            "{err}"
        );
        assert!(
            err.contains("refused"),
            "the cause of the refusal is in the text the operator sees: {err}"
        );
        assert!(!err.contains("s3cr3t"), "the password is masked: {err}");
    }

    /// The three stores of `bundle` take a write each and read it back.
    async fn stores_and_reads_back(bundle: StorageBundle) {
        let (resp, conv, items) = (
            bundle.response_storage,
            bundle.conversation_storage,
            bundle.conversation_item_storage,
        );
        let mut response = StoredResponse::new(None);
        response.input = json!("hello");
        let id = resp.store_response(response).await.unwrap();
        assert!(resp.get_response(&id).await.unwrap().is_some());

        let conversation = conv
            .create_conversation(NewConversation {
                id: None,
                metadata: None,
                tenant: None,
            })
            .await
            .unwrap();
        assert!(conv
            .get_conversation(&conversation.id)
            .await
            .unwrap()
            .is_some());

        let item = items
            .create_item(NewConversationItem {
                id: None,
                response_id: None,
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([]),
                status: Some("completed".to_string()),
                tenant: None,
            })
            .await
            .unwrap();
        assert!(items.get_item(&item.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_create_storage_memory() {
        let config = StorageFactoryConfig {
            backend: &HistoryBackend::Memory,
            oracle: None,
            postgres: None,
            redis: None,
            memory: None,
            hook: None,
        };
        stores_and_reads_back(create_storage(config).await.unwrap()).await;
    }

    #[tokio::test]
    async fn test_create_storage_none() {
        let config = StorageFactoryConfig {
            backend: &HistoryBackend::None,
            oracle: None,
            postgres: None,
            redis: None,
            memory: None,
            hook: None,
        };
        let bundle = create_storage(config).await.unwrap();
        let (resp, conv) = (bundle.response_storage, bundle.conversation_storage);

        // NoOp storage should accept writes but return nothing on reads
        let mut response = StoredResponse::new(None);
        response.input = json!("hello");
        let id = resp.store_response(response).await.unwrap();
        assert!(resp.get_response(&id).await.unwrap().is_none());
        assert!(conv
            .get_conversation(&"nonexistent".into())
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_create_storage_oracle_missing_config() {
        let err = create_storage(StorageFactoryConfig {
            backend: &HistoryBackend::Oracle,
            oracle: None,
            postgres: None,
            redis: None,
            memory: None,
            hook: None,
        })
        .await
        .err()
        .expect("should fail");
        assert!(err.contains("oracle configuration is required"));
    }

    /// A store config reaches the factory from every entry path (the CLI, the
    /// Python bindings, a config built by hand); a bound of 0 s is refused by
    /// name here, before any connection, instead of failing every store call
    /// once the gateway is up.
    #[tokio::test]
    async fn a_zero_call_bound_is_refused_by_the_factory_before_any_connection() {
        let postgres_cfg = PostgresConfig {
            db_url: "postgres://user:pass@localhost:5432/mydb".to_string(),
            pool_max: 4,
            op_timeout_secs: 0,
            schema: None,
        };
        let err = create_storage(StorageFactoryConfig {
            backend: &HistoryBackend::Postgres,
            oracle: None,
            postgres: Some(&postgres_cfg),
            redis: None,
            memory: None,
            hook: None,
        })
        .await
        .err()
        .expect("a zero bound is refused");
        assert!(err.contains("op_timeout_secs"), "{err}");

        let redis_cfg = RedisConfig {
            url: "redis://localhost:6379".to_string(),
            pool_max: 4,
            op_timeout_secs: 0,
            retention_days: Some(1),
            schema: None,
        };
        let err = create_storage(StorageFactoryConfig {
            backend: &HistoryBackend::Redis,
            oracle: None,
            postgres: None,
            redis: Some(&redis_cfg),
            memory: None,
            hook: None,
        })
        .await
        .err()
        .expect("a zero bound is refused");
        assert!(err.contains("op_timeout_secs"), "{err}");
    }

    #[tokio::test]
    async fn test_create_storage_postgres_missing_config() {
        let err = create_storage(StorageFactoryConfig {
            backend: &HistoryBackend::Postgres,
            oracle: None,
            postgres: None,
            redis: None,
            memory: None,
            hook: None,
        })
        .await
        .err()
        .expect("should fail");
        assert!(err.contains("Postgres configuration is required"));
    }

    #[tokio::test]
    async fn test_create_storage_redis_missing_config() {
        let err = create_storage(StorageFactoryConfig {
            backend: &HistoryBackend::Redis,
            oracle: None,
            postgres: None,
            redis: None,
            memory: None,
            hook: None,
        })
        .await
        .err()
        .expect("should fail");
        assert!(err.contains("Redis configuration is required"));
    }

    #[tokio::test]
    async fn test_create_storage_with_hook() {
        use std::sync::Arc;

        use async_trait::async_trait;

        use crate::{
            context::RequestContext,
            hooks::{BeforeHookResult, ExtraColumns, HookError, StorageHook, StorageOperation},
        };

        struct NoOpHook;

        #[async_trait]
        impl StorageHook for NoOpHook {
            async fn before(
                &self,
                _op: StorageOperation,
                _ctx: Option<&RequestContext>,
                _payload: &serde_json::Value,
            ) -> Result<BeforeHookResult, HookError> {
                Ok(BeforeHookResult::default())
            }

            async fn after(
                &self,
                _op: StorageOperation,
                _ctx: Option<&RequestContext>,
                _payload: &serde_json::Value,
                _result: &serde_json::Value,
                extra: &ExtraColumns,
            ) -> Result<ExtraColumns, HookError> {
                Ok(extra.clone())
            }
        }

        let config = StorageFactoryConfig {
            backend: &HistoryBackend::Memory,
            oracle: None,
            postgres: None,
            redis: None,
            memory: None,
            hook: Some(Arc::new(NoOpHook)),
        };
        stores_and_reads_back(create_storage(config).await.unwrap()).await;
    }
}
