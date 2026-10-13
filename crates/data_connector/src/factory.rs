//! Factory function to create storage backends based on configuration.

use std::{sync::Arc, time::Duration};

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
/// (`op_timeout_secs`): the client library and the pool bound their own steps
/// at twice that, so a store that accepts the connection and never answers is
/// the contact's own verdict, with the bound in the text, for either backend;
/// a store that refuses the connection is a verdict at once. The gateway does
/// not start with a history backend it cannot reach.
#[derive(Debug)]
pub(crate) enum StoreContactError {
    /// The connection or the round trip failed: refused, reset, rejected.
    Failed(String),
    /// Nothing came back within the bound.
    NoAnswer,
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

/// The URL with its password masked, for log and error text: the userinfo
/// password and a `password` query pair alike (the Postgres driver reads the
/// latter too).
fn redacted(url: &str) -> String {
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

            let storages = create_oracle_storage(oracle_cfg)?;

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

            let log_db_url = redacted(&postgres_cfg.db_url);

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

            let log_redis_url = redacted(&redis_cfg.url);

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
fn create_oracle_storage(oracle_cfg: &OracleConfig) -> Result<StorageBundle, String> {
    use crate::oracle::OracleStore;

    let store = OracleStore::new(
        oracle_cfg,
        &[
            OracleConversationStorage::init_schema,
            OracleConversationItemStorage::init_schema,
            OracleResponseStorage::init_schema,
        ],
    )?;

    Ok(StorageBundle {
        response_storage: Arc::new(OracleResponseStorage::new(store.clone())),
        conversation_storage: Arc::new(OracleConversationStorage::new(store.clone())),
        conversation_item_storage: Arc::new(OracleConversationItemStorage::new(store)),
    })
}

async fn create_postgres_storage(postgres_cfg: &PostgresConfig) -> Result<StorageBundle, String> {
    let store = PostgresStore::new(postgres_cfg.clone())?;
    let bound = postgres_cfg.op_timeout();
    store
        .contact(bound)
        .await
        .map_err(|err| contact_error("Postgres", &redacted(&postgres_cfg.db_url), bound, err))?;
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
        .map_err(|err| contact_error("Redis", &redacted(&redis_cfg.url), bound, err))?;
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
    use crate::core::{NewConversation, NewConversationItem, StoredResponse};

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

    /// A port nothing listens on: the connection is refused.
    fn closed_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local port");
        listener.local_addr().expect("local address").port()
    }

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
            redacted("postgres://u:s3cr3t@h:5432/db"),
            "postgres://u:****@h:5432/db"
        );
        assert_eq!(
            redacted("postgres://u@h/db?password=s3cr3t&sslmode=require"),
            "postgres://u@h/db?password=****&sslmode=require"
        );
        assert_eq!(
            redacted("postgres://u:s3cr3t@h/db?Password=s3cr3t"),
            "postgres://u:****@h/db?Password=****"
        );
        assert_eq!(
            redacted("redis://:s3cr3t@h:6379/0"),
            "redis://:****@h:6379/0"
        );
        assert_eq!(redacted("not a url"), "<redacted>");
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

    /// `create_storage` for one remote backend, given up on well after the
    /// start-up bound: `None` is a start-up that never gave a verdict.
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
            BOUND + Duration::from_secs(20),
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
        // and names the bound, the same as for Postgres.
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
        assert!(
            started.elapsed() <= BOUND + Duration::from_secs(5),
            "the verdict came within the bound"
        );
    }

    #[tokio::test]
    async fn an_unreachable_redis_store_is_refused_at_start_up_like_postgres() {
        let port = closed_port();
        let err = start_up(None, Some(&redis_config(port)))
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
        let port = closed_port();
        let err = start_up(Some(&postgres_config(port)), None)
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
        let bundle = create_storage(config).await.unwrap();
        let (resp, conv, items) = (
            bundle.response_storage,
            bundle.conversation_storage,
            bundle.conversation_item_storage,
        );

        // Verify they work end-to-end
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
        let bundle = create_storage(config).await.unwrap();
        let (resp, conv, items) = (
            bundle.response_storage,
            bundle.conversation_storage,
            bundle.conversation_item_storage,
        );

        // Verify hooked storage works end-to-end
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
}
