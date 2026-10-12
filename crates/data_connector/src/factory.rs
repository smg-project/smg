//! Factory function to create storage backends based on configuration.

use std::sync::Arc;

use tracing::info;
use url::Url;

use crate::{
    config::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig},
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
    /// Optional storage hook. When provided, all three storage backends are
    /// wrapped in `Hooked*Storage` that runs before/after hooks around every
    /// storage operation.
    pub hook: Option<Arc<dyn StorageHook>>,
}

/// Create all configured storage handles.
///
/// # Errors
/// Returns error string if required configuration is missing or initialization fails
pub async fn create_storage(config: StorageFactoryConfig<'_>) -> Result<StorageBundle, String> {
    let bundle = match config.backend {
        HistoryBackend::Memory => {
            info!("Initializing data connector: Memory");
            StorageBundle {
                response_storage: Arc::new(MemoryResponseStorage::new()),
                conversation_storage: Arc::new(MemoryConversationStorage::new()),
                conversation_item_storage: Arc::new(MemoryConversationItemStorage::new()),
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

            let log_db_url = match Url::parse(&postgres_cfg.db_url) {
                Ok(mut url) => {
                    if url.password().is_some() {
                        let _ = url.set_password(Some("****"));
                    }
                    url.to_string()
                }
                Err(_) => "<redacted>".to_string(),
            };

            info!(
                "Initializing data connector: Postgres (db_url: {}, pool_max: {})",
                log_db_url, postgres_cfg.pool_max
            );

            let storages = create_postgres_storage(postgres_cfg).await?;

            info!("Data connector initialized successfully: Postgres");
            storages
        }
        HistoryBackend::Redis => {
            let redis_cfg = config
                .redis
                .ok_or("Redis configuration is required when history_backend=redis")?;

            let log_redis_url = match Url::parse(&redis_cfg.url) {
                Ok(mut url) => {
                    if url.password().is_some() {
                        let _ = url.set_password(Some("****"));
                    }
                    url.to_string()
                }
                Err(_) => "<redacted>".to_string(),
            };

            info!(
                "Initializing data connector: Redis (url: {}, pool_max: {})",
                log_redis_url, redis_cfg.pool_max
            );

            let storages = create_redis_storage(redis_cfg)?;

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

fn create_redis_storage(redis_cfg: &RedisConfig) -> Result<StorageBundle, String> {
    let store = RedisStore::new(redis_cfg.clone())?;
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
    use serde_json::json;

    use super::*;
    use crate::core::{NewConversation, NewConversationItem, StoredResponse};

    #[tokio::test]
    async fn test_create_storage_memory() {
        let config = StorageFactoryConfig {
            backend: &HistoryBackend::Memory,
            oracle: None,
            postgres: None,
            redis: None,
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
            hook: None,
        })
        .await
        .err()
        .expect("should fail");
        assert!(err.contains("oracle configuration is required"));
    }

    #[tokio::test]
    async fn test_create_storage_postgres_missing_config() {
        let err = create_storage(StorageFactoryConfig {
            backend: &HistoryBackend::Postgres,
            oracle: None,
            postgres: None,
            redis: None,
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
            })
            .await
            .unwrap();
        assert!(items.get_item(&item.id).await.unwrap().is_some());
    }
}
