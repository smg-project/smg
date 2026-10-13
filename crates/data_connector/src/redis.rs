//! Redis storage implementation using RedisStore helper

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_redis::{Connection, Manager, Pool, PoolConfig, Runtime, Timeouts};
use redis::{AsyncCommands, AsyncConnectionConfig};
use serde_json::Value;

use crate::{
    common::{parse_json_value, parse_raw_response, resolve_extra_column_values},
    config::RedisConfig,
    context::current_extra_columns,
    core::{
        make_item_id, ConditionalStore, Conversation, ConversationId, ConversationItem,
        ConversationItemId, ConversationItemResult, ConversationItemStorage,
        ConversationItemStorageError, ConversationMetadata, ConversationResult,
        ConversationStorage, ConversationStorageError, ListParams, NewConversation,
        NewConversationItem, ResponseId, ResponseResult, ResponseStorage, ResponseStorageError,
        SortOrder, StoredResponse, TERMINAL_RESPONSE_STATUSES,
    },
    factory::StoreContactError,
    schema::SchemaConfig,
};

pub(crate) struct RedisStore {
    pool: Pool,
    retention_days: Option<u64>,
    pub(crate) schema: Arc<SchemaConfig>,
}

impl RedisStore {
    pub fn new(config: RedisConfig) -> Result<Self, String> {
        let schema = config.schema.clone().unwrap_or_default();
        schema.validate()?;
        let schema = Arc::new(schema);

        // Each call as a whole is bounded by the `bounded` wrappers the
        // factory installs. The pool and the client library bound their own
        // steps (connecting, a response, waiting for a pooled connection,
        // creating one, recycling one) at twice that bound: backstops that
        // clean up after a call the wrapper has already answered, never the
        // first to fire. Left to its defaults the library bounded a response
        // at 500 ms and a connection at 1 s, which failed every call to a
        // store a few hundred milliseconds away.
        let backstop = config.op_timeout().saturating_mul(2);
        let connection = AsyncConnectionConfig::new()
            .set_connection_timeout(Some(backstop))
            .set_response_timeout(Some(backstop));
        let manager = Manager::new_with_config(config.url.as_str(), connection)
            .map_err(|e| format!("Invalid Redis connection URL: {e}"))?;
        let mut pool_config = PoolConfig::new(config.pool_max);
        pool_config.timeouts = Timeouts {
            wait: Some(backstop),
            create: Some(backstop),
            recycle: Some(backstop),
        };
        let pool = Pool::builder(manager)
            .config(pool_config)
            .runtime(Runtime::Tokio1)
            .build()
            .map_err(|e| format!("Failed to build Redis connection pool: {e}"))?;
        Ok(Self {
            pool,
            retention_days: config.retention_days,
            schema,
        })
    }

    /// The start-up contact: one connection and one round trip within
    /// `bound`, so a store that refuses the connection, or accepts it and never
    /// answers, is a verdict before the gateway serves.
    pub(crate) async fn contact(&self, bound: Duration) -> Result<(), StoreContactError> {
        let round_trip = async {
            let mut conn = self.pool.get().await.map_err(|e| e.to_string())?;
            redis::cmd("PING")
                .query_async::<String>(&mut *conn)
                .await
                .map_err(|e| e.to_string())?;
            Ok::<(), String>(())
        };
        match tokio::time::timeout(bound, round_trip).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(cause)) => Err(StoreContactError::Failed(cause)),
            Err(_elapsed) => Err(StoreContactError::NoAnswer),
        }
    }
}

impl Clone for RedisStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            retention_days: self.retention_days,
            schema: self.schema.clone(),
        }
    }
}

pub(super) struct RedisConversationStorage {
    store: RedisStore,
}

impl RedisConversationStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn conversation_key(&self, id: &str) -> String {
        match &self.store.schema.owner {
            Some(owner) => format!("{owner}:conversation:{id}"),
            None => format!("conversation:{id}"),
        }
    }

    fn parse_metadata(
        metadata: Option<String>,
    ) -> Result<Option<ConversationMetadata>, ConversationStorageError> {
        crate::common::parse_conversation_metadata(metadata)
            .map_err(ConversationStorageError::StorageError)
    }

    /// The tenant stored on a conversation's hash: `None` when the schema
    /// skips the column or the record carries none.
    async fn read_tenant(
        &self,
        conn: &mut Connection,
        key: &str,
    ) -> Result<Option<String>, ConversationStorageError> {
        let s = &self.store.schema.conversations;
        if s.is_skipped("tenant") {
            return Ok(None);
        }
        conn.hget(key, s.col("tenant"))
            .await
            .map_err(ConversationStorageError::from_redis)
    }
}

#[async_trait]
impl ConversationStorage for RedisConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> Result<Conversation, ConversationStorageError> {
        let conversation = Conversation::new(input);
        let id_str = conversation.id.0.as_str();
        let created_at: DateTime<Utc> = conversation.created_at;
        let metadata_json = conversation
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;

        let s = &self.store.schema.conversations;

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::pool_unavailable(&e))?;
        let key = self.conversation_key(id_str);

        let mut pipe = redis::pipe();
        pipe.hset(&key, s.col("id"), id_str);
        if !s.is_skipped("created_at") {
            pipe.hset(&key, s.col("created_at"), created_at.to_rfc3339());
        }
        if !s.is_skipped("metadata") {
            if let Some(meta) = metadata_json {
                pipe.hset(&key, s.col("metadata"), meta);
            }
        }
        if !s.is_skipped("tenant") {
            if let Some(tenant) = &conversation.tenant {
                pipe.hset(&key, s.col("tenant"), tenant);
            }
        }

        // Append extra columns from hooks or defaults
        let hook_extra = current_extra_columns().unwrap_or_default();
        for (name, val) in resolve_extra_column_values(s, &hook_extra) {
            if let Some(v) = val {
                pipe.hset(&key, name, v);
            }
        }

        if let Some(days) = self.store.retention_days {
            pipe.expire(&key, (days * 24 * 60 * 60) as i64);
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(ConversationStorageError::from_redis)?;

        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let s = &self.store.schema.conversations;

        let id_str = id.0.as_str();
        let key = self.conversation_key(id_str);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::pool_unavailable(&e))?;

        let exists: bool = conn
            .exists(&key)
            .await
            .map_err(ConversationStorageError::from_redis)?;
        if !exists {
            return Ok(None);
        }

        let created_at = if s.is_skipped("created_at") {
            Utc::now()
        } else {
            let created_at_str: String = conn
                .hget(&key, s.col("created_at"))
                .await
                .map_err(ConversationStorageError::from_redis)?;
            DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?
                .with_timezone(&Utc)
        };

        let metadata = if s.is_skipped("metadata") {
            None
        } else {
            let metadata_json: Option<String> = conn
                .hget(&key, s.col("metadata"))
                .await
                .map_err(ConversationStorageError::from_redis)?;
            Self::parse_metadata(metadata_json)?
        };

        let tenant = self.read_tenant(&mut conn, &key).await?;
        Ok(Some(Conversation::with_parts(
            id.clone(),
            created_at,
            metadata,
            tenant,
        )))
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let s = &self.store.schema.conversations;

        let id_str = id.0.as_str();
        let key = self.conversation_key(id_str);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::pool_unavailable(&e))?;

        let exists: bool = conn
            .exists(&key)
            .await
            .map_err(ConversationStorageError::from_redis)?;
        if !exists {
            return Ok(None);
        }

        if !s.is_skipped("metadata") {
            let col_meta = s.col("metadata");
            let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;

            if let Some(meta) = metadata_json {
                conn.hset::<_, _, _, ()>(&key, col_meta, meta)
                    .await
                    .map_err(ConversationStorageError::from_redis)?;
            } else {
                conn.hdel::<_, _, ()>(&key, col_meta)
                    .await
                    .map_err(ConversationStorageError::from_redis)?;
            }
        }

        let created_at = if s.is_skipped("created_at") {
            Utc::now()
        } else {
            let col_created = s.col("created_at");
            let created_at_str: String = conn
                .hget(&key, col_created)
                .await
                .map_err(ConversationStorageError::from_redis)?;
            DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?
                .with_timezone(&Utc)
        };

        let tenant = self.read_tenant(&mut conn, &key).await?;
        Ok(Some(Conversation::with_parts(
            id.clone(),
            created_at,
            metadata,
            tenant,
        )))
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let id_str = id.0.as_str();
        let key = self.conversation_key(id_str);
        let items_key = format!("{key}:items");

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::pool_unavailable(&e))?;

        // The pipeline answers one count per DEL (the conversation key, then
        // its item index); the conversation existed if its key was deleted.
        let (deleted, _item_index_deleted): (usize, usize) = redis::pipe()
            .del(&key)
            .del(&items_key)
            .query_async(&mut conn)
            .await
            .map_err(ConversationStorageError::from_redis)?;

        Ok(deleted > 0)
    }
}

pub(super) struct RedisConversationItemStorage {
    store: RedisStore,
}

impl RedisConversationItemStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn item_key(&self, id: &str) -> String {
        match &self.store.schema.owner {
            Some(owner) => format!("{owner}:item:{id}"),
            None => format!("item:{id}"),
        }
    }

    fn conv_items_key(&self, conv_id: &str) -> String {
        match &self.store.schema.owner {
            Some(owner) => format!("{owner}:conversation:{conv_id}:items"),
            None => format!("conversation:{conv_id}:items"),
        }
    }

    /// Parse a Redis hash map into a `ConversationItem`, returning errors for
    /// corrupted data instead of silently substituting defaults.
    /// Fields listed in `skip_columns` use defaults since they were not stored.
    fn build_item_from_map(
        &self,
        map: &HashMap<String, String>,
        fallback_id: &str,
    ) -> Result<ConversationItem, ConversationItemStorageError> {
        let si = &self.store.schema.conversation_items;

        let col_id = si.col("id");
        let id = ConversationItemId(
            map.get(col_id)
                .cloned()
                .unwrap_or_else(|| fallback_id.to_string()),
        );

        let response_id = if si.is_skipped("response_id") {
            None
        } else {
            map.get(si.col("response_id")).cloned()
        };

        let item_type = if si.is_skipped("item_type") {
            "message".to_string()
        } else {
            let col_item_type = si.col("item_type");
            map.get(col_item_type)
                .filter(|s| !s.is_empty())
                .cloned()
                .ok_or_else(|| {
                    ConversationItemStorageError::StorageError(format!(
                        "item {fallback_id} missing {col_item_type}"
                    ))
                })?
        };

        let role = if si.is_skipped("role") {
            None
        } else {
            map.get(si.col("role")).cloned()
        };

        let status = if si.is_skipped("status") {
            None
        } else {
            map.get(si.col("status")).cloned()
        };

        let content = if si.is_skipped("content") {
            Value::Null
        } else {
            match map.get(si.col("content")) {
                Some(s) => serde_json::from_str(s)
                    .map_err(ConversationItemStorageError::SerializationError)?,
                None => Value::Null,
            }
        };

        let created_at = if si.is_skipped("created_at") {
            Utc::now()
        } else {
            let col_created = si.col("created_at");
            let created_at_str = map.get(col_created).ok_or_else(|| {
                ConversationItemStorageError::StorageError(format!(
                    "item {fallback_id} missing {col_created}"
                ))
            })?;
            DateTime::parse_from_rfc3339(created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| {
                    ConversationItemStorageError::StorageError(format!(
                        "item {fallback_id} invalid {col_created}: {e}"
                    ))
                })?
        };
        let tenant = if si.is_skipped("tenant") {
            None
        } else {
            map.get(si.col("tenant")).cloned()
        };

        Ok(ConversationItem {
            id,
            response_id,
            item_type,
            role,
            content,
            status,
            created_at,
            tenant,
        })
    }
}

/// The sorted-set score of a conversation link: `added_at` in milliseconds, with the
/// microseconds as the fraction. Links a microsecond apart keep their order (on whole
/// milliseconds they tied, and Redis orders a tie by the member, the random item id); a
/// whole-millisecond stamp scores as before, so links already stored keep their place.
fn link_score(added_at: DateTime<Utc>) -> f64 {
    added_at.timestamp_micros() as f64 / 1000.0
}

#[async_trait]
impl ConversationItemStorage for RedisConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> Result<ConversationItem, ConversationItemStorageError> {
        let NewConversationItem {
            id: opt_id,
            response_id,
            item_type,
            role,
            content,
            status,
            tenant,
        } = item;
        let id = opt_id.unwrap_or_else(|| make_item_id(&item_type));
        let created_at = Utc::now();
        let content_json = serde_json::to_string(&content)?;

        let conversation_item = ConversationItem {
            id,
            response_id,
            item_type,
            role,
            content,
            status,
            created_at,
            tenant,
        };

        let si = &self.store.schema.conversation_items;
        let id_str = conversation_item.id.0.as_str();
        let key = self.item_key(id_str);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;

        let mut pipe = redis::pipe();

        pipe.hset(&key, si.col("id"), id_str);
        if !si.is_skipped("response_id") {
            if let Some(rid) = &conversation_item.response_id {
                pipe.hset(&key, si.col("response_id"), rid);
            }
        }
        if !si.is_skipped("item_type") {
            pipe.hset(&key, si.col("item_type"), &conversation_item.item_type);
        }
        if !si.is_skipped("role") {
            if let Some(r) = &conversation_item.role {
                pipe.hset(&key, si.col("role"), r);
            }
        }
        if !si.is_skipped("content") {
            pipe.hset(&key, si.col("content"), &content_json);
        }
        if !si.is_skipped("status") {
            if let Some(s) = &conversation_item.status {
                pipe.hset(&key, si.col("status"), s);
            }
        }
        if !si.is_skipped("created_at") {
            pipe.hset(&key, si.col("created_at"), created_at.to_rfc3339());
        }
        if !si.is_skipped("tenant") {
            if let Some(tenant) = &conversation_item.tenant {
                pipe.hset(&key, si.col("tenant"), tenant);
            }
        }

        // Append extra columns from hooks or defaults
        let hook_extra = current_extra_columns().unwrap_or_default();
        for (name, val) in resolve_extra_column_values(si, &hook_extra) {
            if let Some(v) = val {
                pipe.hset(&key, name, v);
            }
        }

        if let Some(days) = self.store.retention_days {
            pipe.expire(&key, (days * 24 * 60 * 60) as i64);
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(ConversationItemStorageError::from_redis)?;

        Ok(conversation_item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = self.conv_items_key(cid);

        let score = link_score(added_at);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        conn.zadd::<_, _, _, ()>(&key, iid, score)
            .await
            .map_err(ConversationItemStorageError::from_redis)?;

        Ok(())
    }

    async fn link_items(
        &self,
        conversation_id: &ConversationId,
        items: &[(ConversationItemId, DateTime<Utc>)],
    ) -> ConversationItemResult<()> {
        if items.is_empty() {
            return Ok(());
        }

        let cid = conversation_id.0.as_str();
        let key = self.conv_items_key(cid);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;

        let mut pipe = redis::pipe();
        for (item_id, added_at) in items {
            let score = link_score(*added_at);
            pipe.zadd(&key, item_id.0.as_str(), score).ignore();
        }
        pipe.query_async::<()>(&mut *conn)
            .await
            .map_err(ConversationItemStorageError::from_redis)?;

        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let cid = conversation_id.0.as_str();
        let key = self.conv_items_key(cid);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;

        let mut min = "-inf".to_string();
        let mut max = "+inf".to_string();
        let mut cursor_score: Option<f64> = None;
        let mut cursor_id: Option<String> = None;

        if let Some(after_id) = &params.after {
            let score: Option<f64> = conn
                .zscore(&key, after_id)
                .await
                .map_err(ConversationItemStorageError::from_redis)?;
            if let Some(s) = score {
                cursor_score = Some(s);
                cursor_id = Some(after_id.clone());
                match params.order {
                    SortOrder::Asc => min = s.to_string(),
                    SortOrder::Desc => max = s.to_string(),
                }
            }
        }

        let fetch_limit = if cursor_score.is_some() {
            (params.limit + 32) as isize
        } else {
            params.limit as isize
        };

        let item_ids: Vec<String> = match params.order {
            SortOrder::Asc => conn
                .zrangebyscore_limit(&key, min, max, 0, fetch_limit)
                .await
                .map_err(ConversationItemStorageError::from_redis)?,
            SortOrder::Desc => conn
                .zrevrangebyscore_limit(&key, max, min, 0, fetch_limit)
                .await
                .map_err(ConversationItemStorageError::from_redis)?,
        };

        let item_ids: Vec<String> = if let (Some(_), Some(ref c_id)) = (cursor_score, &cursor_id) {
            item_ids
                .into_iter()
                .skip_while(|id| id != c_id)
                .skip(1)
                .take(params.limit)
                .collect()
        } else {
            item_ids.into_iter().take(params.limit).collect()
        };

        if item_ids.is_empty() {
            return Ok(Vec::<ConversationItem>::new());
        }

        let mut pipe = redis::pipe();
        for iid in &item_ids {
            pipe.hgetall(self.item_key(iid));
        }

        let results: Vec<HashMap<String, String>> = pipe
            .query_async(&mut conn)
            .await
            .map_err(ConversationItemStorageError::from_redis)?;

        let mut items: Vec<ConversationItem> = Vec::with_capacity(results.len());
        for (i, map) in results.into_iter().enumerate() {
            if map.is_empty() {
                continue;
            }

            items.push(self.build_item_from_map(&map, &item_ids[i])?);
        }

        Ok(items)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        let iid = item_id.0.as_str();
        let key = self.item_key(iid);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;

        let map: HashMap<String, String> = conn
            .hgetall(&key)
            .await
            .map_err(ConversationItemStorageError::from_redis)?;

        if map.is_empty() {
            return Ok(None);
        }

        self.build_item_from_map(&map, iid).map(Some)
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = self.conv_items_key(cid);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        let score: Option<f64> = conn
            .zscore(&key, iid)
            .await
            .map_err(ConversationItemStorageError::from_redis)?;

        Ok(score.is_some())
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = self.conv_items_key(cid);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::pool_unavailable(&e))?;
        conn.zrem::<_, _, ()>(&key, iid)
            .await
            .map_err(ConversationItemStorageError::from_redis)?;

        Ok(())
    }
}

pub(super) struct RedisResponseStorage {
    store: RedisStore,
}

impl RedisResponseStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn response_key(&self, id: &str) -> String {
        match &self.store.schema.owner {
            Some(owner) => format!("{owner}:response:{id}"),
            None => format!("response:{id}"),
        }
    }

    fn safety_key(&self, identifier: &str) -> String {
        match &self.store.schema.owner {
            Some(owner) => format!("{owner}:safety:{identifier}:responses"),
            None => format!("safety:{identifier}:responses"),
        }
    }

    /// The key that points from an output item id at the response that
    /// produced it: the lookup behind `item_reference` input items. Written
    /// with the response (a hash cannot be searched by a field of its JSON
    /// payload short of scanning every response) and removed with it.
    fn output_item_key(&self, item_id: &str) -> String {
        match &self.store.schema.owner {
            Some(owner) => format!("{owner}:output_item:{item_id}"),
            None => format!("output_item:{item_id}"),
        }
    }

    /// The output-item keys pointing at the responses stored under `keys`,
    /// read from each hash's `raw_response` (before the hashes go).
    async fn output_item_keys_of_many(
        &self,
        conn: &mut Connection,
        keys: &[String],
    ) -> ResponseResult<Vec<String>> {
        let sr = &self.store.schema.responses;
        if sr.is_skipped("raw_response") || keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut pipe = redis::pipe();
        for key in keys {
            pipe.hget(key, sr.col("raw_response"));
        }
        let raws: Vec<Option<String>> = pipe
            .query_async(conn)
            .await
            .map_err(ResponseStorageError::from_redis)?;
        let mut out = Vec::new();
        for raw in raws {
            let raw = parse_raw_response(raw).map_err(ResponseStorageError::StorageError)?;
            out.extend(
                output_item_ids(&raw)
                    .into_iter()
                    .map(|item_id| self.output_item_key(item_id)),
            );
        }
        Ok(out)
    }

    /// The fields of the response's hash, in the order they are written,
    /// skipping the configured `skip_columns` and appending the extra
    /// columns of hooks or defaults.
    fn hash_fields<'a>(
        &'a self,
        response: &'a StoredResponse,
    ) -> Result<Vec<(&'a str, String)>, ResponseStorageError> {
        let sr = &self.store.schema.responses;
        let mut fields: Vec<(&str, String)> = vec![(sr.col("id"), response.id.0.clone())];
        if !sr.is_skipped("previous_response_id") {
            if let Some(prev) = &response.previous_response_id {
                fields.push((sr.col("previous_response_id"), prev.0.clone()));
            }
        }
        if !sr.is_skipped("conversation_id") {
            if let Some(cid) = &response.conversation_id {
                fields.push((sr.col("conversation_id"), cid.clone()));
            }
        }
        if !sr.is_skipped("input") {
            fields.push((sr.col("input"), serde_json::to_string(&response.input)?));
        }
        if !sr.is_skipped("created_at") {
            fields.push((sr.col("created_at"), response.created_at.to_rfc3339()));
        }
        if !sr.is_skipped("safety_identifier") {
            if let Some(safety) = &response.safety_identifier {
                fields.push((sr.col("safety_identifier"), safety.clone()));
            }
        }
        if !sr.is_skipped("model") {
            if let Some(model) = &response.model {
                fields.push((sr.col("model"), model.clone()));
            }
        }
        if !sr.is_skipped("tenant") {
            if let Some(tenant) = &response.tenant {
                fields.push((sr.col("tenant"), tenant.clone()));
            }
        }
        if !sr.is_skipped("raw_response") {
            fields.push((
                sr.col("raw_response"),
                serde_json::to_string(&response.raw_response)?,
            ));
        }
        let hook_extra = current_extra_columns().unwrap_or_default();
        for (name, val) in resolve_extra_column_values(sr, &hook_extra) {
            if let Some(v) = val {
                fields.push((name, v));
            }
        }
        Ok(fields)
    }

    /// The writes that go with a stored response besides its hash: one key
    /// per output item id (with the response's retention) and the hash's
    /// expiry.
    fn index_writes(&self, pipe: &mut redis::Pipeline, key: &str, response: &StoredResponse) {
        let sr = &self.store.schema.responses;
        let response_id_str = response.id.0.as_str();
        if !sr.is_skipped("raw_response") {
            for item_id in output_item_ids(&response.raw_response) {
                let item_key = self.output_item_key(item_id);
                match self.store.retention_days {
                    Some(days) => {
                        pipe.set_ex(item_key, response_id_str, days * 24 * 60 * 60);
                    }
                    None => {
                        pipe.set(item_key, response_id_str);
                    }
                }
            }
        }
        if let Some(days) = self.store.retention_days {
            pipe.expire(key, (days * 24 * 60 * 60) as i64);
        }
    }

    /// Index the response by its safety identifier, if it has one.
    async fn index_by_identifier(
        &self,
        conn: &mut Connection,
        response: &StoredResponse,
    ) -> ResponseResult<()> {
        let sr = &self.store.schema.responses;
        if sr.is_skipped("safety_identifier") {
            return Ok(());
        }
        if let Some(safety) = &response.safety_identifier {
            let safety_key = self.safety_key(safety);
            let score = response.created_at.timestamp_millis() as f64;
            conn.zadd::<_, _, _, ()>(safety_key, response.id.0.as_str(), score)
                .await
                .map_err(ResponseStorageError::from_redis)?;
        }
        Ok(())
    }

    /// Build a `StoredResponse` from the Redis hash map returned by `HGETALL`.
    /// Fields listed in `skip_columns` use defaults since they were not stored.
    fn build_response_from_map(
        &self,
        map: HashMap<String, String>,
        fallback_id: &str,
    ) -> Result<StoredResponse, ResponseStorageError> {
        let s = &self.store.schema.responses;

        let col_id = s.col("id");
        let id = ResponseId(
            map.get(col_id)
                .cloned()
                .unwrap_or_else(|| fallback_id.to_string()),
        );

        let previous_response_id = if s.is_skipped("previous_response_id") {
            None
        } else {
            map.get(s.col("previous_response_id"))
                .map(|v| ResponseId(v.clone()))
        };
        let conversation_id = if s.is_skipped("conversation_id") {
            None
        } else {
            map.get(s.col("conversation_id")).cloned()
        };

        let input = if s.is_skipped("input") {
            Value::Array(vec![])
        } else {
            parse_json_value(map.get(s.col("input")).cloned())
                .map_err(ResponseStorageError::StorageError)?
        };

        let created_at = if s.is_skipped("created_at") {
            Utc::now()
        } else {
            let col_created = s.col("created_at");
            let created_at_str = map.get(col_created).ok_or_else(|| {
                ResponseStorageError::StorageError(format!(
                    "response {fallback_id} missing {col_created}"
                ))
            })?;
            DateTime::parse_from_rfc3339(created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| {
                    ResponseStorageError::StorageError(format!(
                        "response {fallback_id} invalid {col_created}: {e}"
                    ))
                })?
        };

        let safety_identifier = if s.is_skipped("safety_identifier") {
            None
        } else {
            map.get(s.col("safety_identifier")).cloned()
        };
        let model = if s.is_skipped("model") {
            None
        } else {
            map.get(s.col("model")).cloned()
        };
        let raw_response = if s.is_skipped("raw_response") {
            Value::Null
        } else {
            parse_raw_response(map.get(s.col("raw_response")).cloned())
                .map_err(ResponseStorageError::StorageError)?
        };
        let tenant = if s.is_skipped("tenant") {
            None
        } else {
            map.get(s.col("tenant")).cloned()
        };

        Ok(StoredResponse {
            id,
            previous_response_id,
            input,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
            tenant,
        })
    }

    /// The responses stored under `response_ids`, in that order, those the
    /// caller's `tenant` may see; with them the ids whose hash is gone (the
    /// retention took it), which belong in no index any more.
    async fn visible_responses(
        &self,
        conn: &mut Connection,
        response_ids: &[String],
        tenant: Option<&str>,
    ) -> ResponseResult<(Vec<StoredResponse>, Vec<String>)> {
        if response_ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut pipe = redis::pipe();
        for id in response_ids {
            pipe.hgetall(self.response_key(id));
        }
        let results: Vec<HashMap<String, String>> = pipe
            .query_async(conn)
            .await
            .map_err(ResponseStorageError::from_redis)?;

        let mut out: Vec<StoredResponse> = Vec::with_capacity(results.len());
        let mut gone: Vec<String> = Vec::new();
        for (i, map) in results.into_iter().enumerate() {
            if map.is_empty() {
                gone.push(response_ids[i].clone());
                continue;
            }
            let response = self.build_response_from_map(map, &response_ids[i])?;
            if response.is_visible_to(tenant) {
                out.push(response);
            }
        }
        Ok((out, gone))
    }
}

#[async_trait]
impl ResponseStorage for RedisResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let response_id = response.id.clone();
        let key = self.response_key(&response_id.0);
        let fields = self.hash_fields(&response)?;

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;

        let mut pipe = redis::pipe();
        for (field, value) in &fields {
            pipe.hset(&key, *field, value);
        }
        self.index_writes(&mut pipe, &key, &response);
        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(ResponseStorageError::from_redis)?;
        self.index_by_identifier(&mut conn, &response).await?;

        Ok(response_id)
    }

    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> ResponseResult<ConditionalStore> {
        let sr = &self.store.schema.responses;
        let key = self.response_key(&response.id.0);
        let fields = self.hash_fields(&response)?;

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;

        // The status check and the hash write run as one script: nothing
        // closes the record between them.
        let script = redis::Script::new(&store_if_open_script());
        let mut invocation = script.key(&key);
        invocation.arg(sr.col("raw_response"));
        for (field, value) in &fields {
            invocation.arg(*field).arg(value);
        }
        let outcome: String = invocation
            .invoke_async(&mut conn)
            .await
            .map_err(ResponseStorageError::from_redis)?;
        match outcome.as_str() {
            "stored" => {
                let mut pipe = redis::pipe();
                self.index_writes(&mut pipe, &key, &response);
                pipe.query_async::<()>(&mut conn)
                    .await
                    .map_err(ResponseStorageError::from_redis)?;
                self.index_by_identifier(&mut conn, &response).await?;
                Ok(ConditionalStore::Stored)
            }
            "closed" => {
                drop(conn);
                Ok(match self.get_response(&response.id).await? {
                    Some(current) => ConditionalStore::Closed(Box::new(current)),
                    None => ConditionalStore::Missing,
                })
            }
            _ => Ok(ConditionalStore::Missing),
        }
    }

    async fn update_response_object_if_open(
        &self,
        id: &ResponseId,
        raw_response: Value,
    ) -> ResponseResult<ConditionalStore> {
        let sr = &self.store.schema.responses;
        if sr.is_skipped("raw_response") {
            // No object field: nothing a transition could move.
            return Ok(ConditionalStore::Stored);
        }
        let key = self.response_key(&id.0);
        let json_raw_response = serde_json::to_string(&raw_response)?;
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;
        // The same script as the full store, with the object's field alone.
        let script = redis::Script::new(&store_if_open_script());
        let outcome: String = script
            .key(&key)
            .arg(sr.col("raw_response"))
            .arg(sr.col("raw_response"))
            .arg(&json_raw_response)
            .invoke_async(&mut conn)
            .await
            .map_err(ResponseStorageError::from_redis)?;
        match outcome.as_str() {
            "stored" => {
                // The output-item keys and the expiry follow the object.
                let mut moved = StoredResponse::new(None);
                moved.id = id.clone();
                moved.raw_response = raw_response;
                let mut pipe = redis::pipe();
                self.index_writes(&mut pipe, &key, &moved);
                pipe.query_async::<()>(&mut conn)
                    .await
                    .map_err(ResponseStorageError::from_redis)?;
                Ok(ConditionalStore::Stored)
            }
            "closed" => {
                drop(conn);
                Ok(match self.get_response(id).await? {
                    Some(current) => ConditionalStore::Closed(Box::new(current)),
                    None => ConditionalStore::Missing,
                })
            }
            _ => Ok(ConditionalStore::Missing),
        }
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let id = response_id.0.as_str();
        let key = self.response_key(id);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;

        let map: HashMap<String, String> = conn
            .hgetall(&key)
            .await
            .map_err(ResponseStorageError::from_redis)?;

        if map.is_empty() {
            return Ok(None);
        }

        self.build_response_from_map(map, id).map(Some)
    }

    async fn find_response_by_output_item(
        &self,
        item_id: &str,
    ) -> ResponseResult<Option<StoredResponse>> {
        if self.store.schema.responses.is_skipped("raw_response") {
            return Ok(None);
        }
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;

        let response_id: Option<String> = conn
            .get(self.output_item_key(item_id))
            .await
            .map_err(ResponseStorageError::from_redis)?;
        match response_id {
            Some(id) => self.get_response(&ResponseId(id)).await,
            None => Ok(None),
        }
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let sr = &self.store.schema.responses;

        let id = response_id.0.as_str();
        let key = self.response_key(id);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;

        let output_item_keys = self
            .output_item_keys_of_many(&mut conn, std::slice::from_ref(&key))
            .await?;

        if sr.is_skipped("safety_identifier") {
            conn.del::<_, ()>(&key)
                .await
                .map_err(ResponseStorageError::from_redis)?;
        } else {
            let col_safety = sr.col("safety_identifier");

            let (safety, ()): (Option<String>, ()) = redis::pipe()
                .atomic()
                .hget(&key, col_safety)
                .del(&key)
                .query_async(&mut conn)
                .await
                .map_err(ResponseStorageError::from_redis)?;

            if let Some(s) = safety {
                conn.zrem::<_, _, ()>(self.safety_key(&s), id)
                    .await
                    .map_err(ResponseStorageError::from_redis)?;
            }
        }

        if !output_item_keys.is_empty() {
            conn.del::<_, ()>(output_item_keys)
                .await
                .map_err(ResponseStorageError::from_redis)?;
        }

        Ok(())
    }

    async fn expire_response(
        &self,
        response_id: &ResponseId,
        after: Duration,
    ) -> ResponseResult<()> {
        let key = self.response_key(response_id.0.as_str());
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;

        // The record and the output-item keys pointing at it go together;
        // the identifier index tolerates a member whose record is gone.
        let mut keys = self
            .output_item_keys_of_many(&mut conn, std::slice::from_ref(&key))
            .await?;
        keys.push(key);
        let seconds = i64::try_from(after.as_secs().max(1)).unwrap_or(i64::MAX);
        let mut pipe = redis::pipe();
        for key in &keys {
            pipe.expire(key, seconds);
        }
        let _: () = pipe
            .query_async(&mut conn)
            .await
            .map_err(ResponseStorageError::from_redis)?;
        Ok(())
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let sr = &self.store.schema.responses;
        if sr.is_skipped("safety_identifier") {
            return Ok(Vec::new());
        }

        let key = self.safety_key(identifier);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;

        // The whole index, newest first: the tenant is a field of each
        // record, so the limit applies after the records the caller may not
        // see are dropped.
        let response_ids: Vec<String> = conn
            .zrevrange(&key, 0, -1)
            .await
            .map_err(ResponseStorageError::from_redis)?;

        let (mut out, gone) = self
            .visible_responses(&mut conn, &response_ids, tenant)
            .await?;
        // The ids whose hash the retention took leave the index here.
        if !gone.is_empty() {
            conn.zrem::<_, _, ()>(&key, &gone)
                .await
                .map_err(ResponseStorageError::from_redis)?;
        }
        if let Some(limit) = limit {
            out.truncate(limit);
        }
        Ok(out)
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> ResponseResult<usize> {
        let sr = &self.store.schema.responses;
        if sr.is_skipped("safety_identifier") {
            return Ok(0);
        }

        let key = self.safety_key(identifier);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::pool_unavailable(&e))?;

        let response_ids: Vec<String> = conn
            .zrange(&key, 0, -1)
            .await
            .map_err(ResponseStorageError::from_redis)?;
        if response_ids.is_empty() {
            return Ok(0);
        }

        // Only the records the caller may see go; the others keep their
        // place in the index; the ids whose hash is already gone leave it too.
        let (visible, gone) = self
            .visible_responses(&mut conn, &response_ids, tenant)
            .await?;
        let visible_ids: Vec<String> = visible.into_iter().map(|response| response.id.0).collect();
        let count = visible_ids.len();
        if count == 0 {
            if !gone.is_empty() {
                conn.zrem::<_, _, ()>(&key, &gone)
                    .await
                    .map_err(ResponseStorageError::from_redis)?;
            }
            return Ok(0);
        }

        let response_keys: Vec<String> =
            visible_ids.iter().map(|id| self.response_key(id)).collect();
        let output_item_keys = self
            .output_item_keys_of_many(&mut conn, &response_keys)
            .await?;

        let mut pipe = redis::pipe();
        for response_key in &response_keys {
            pipe.del(response_key);
        }
        for item_key in &output_item_keys {
            pipe.del(item_key);
        }
        // An index left empty goes with its last member.
        let mut leaving = visible_ids;
        leaving.extend(gone);
        pipe.zrem(&key, &leaving);

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(ResponseStorageError::from_redis)?;

        Ok(count)
    }
}

/// The script of [`ResponseStorage::store_response_if_open`]: `KEYS[1]` is
/// the response's hash, `ARGV[1]` the name of the field holding the stored
/// object, `ARGV[2..]` the hash's fields and values. It writes the fields
/// only while the stored object's status is not terminal
/// ([`TERMINAL_RESPONSE_STATUSES`]; a hash without one counts as open) and
/// answers `stored`, `closed` or `missing`.
fn store_if_open_script() -> String {
    let closed = TERMINAL_RESPONSE_STATUSES
        .iter()
        .map(|status| format!("status == '{status}'"))
        .collect::<Vec<_>>()
        .join(" or ");
    format!(
        "
local raw = redis.call('HGET', KEYS[1], ARGV[1])
if not raw then return 'missing' end
local status = ''
local ok, doc = pcall(cjson.decode, raw)
if ok and type(doc) == 'table' and type(doc['status']) == 'string' then status = doc['status'] end
if {closed} then return 'closed' end
for i = 2, #ARGV - 1, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1]) end
return 'stored'
"
    )
}

/// The ids of the output items of a stored response (`raw_response.output[].id`).
fn output_item_ids(raw_response: &Value) -> Vec<&str> {
    raw_response
        .get("output")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn output_item_ids_reads_the_output_array_only() {
        let raw = json!({
            "id": "resp_1",
            "output": [
                {"type": "reasoning", "id": "rs_1"},
                {"type": "message", "id": "msg_1"},
                {"type": "message"}
            ]
        });
        assert_eq!(output_item_ids(&raw), vec!["rs_1", "msg_1"]);
        assert!(output_item_ids(&json!({"id": "resp_2"})).is_empty());
        assert!(output_item_ids(&Value::Null).is_empty());
    }

    #[test]
    fn link_scores_a_microsecond_apart_keep_their_order() {
        let base = DateTime::from_timestamp_micros(1_700_000_000_123_456).unwrap();
        let later = base + chrono::TimeDelta::microseconds(1);
        assert!(link_score(base) < link_score(later));
        assert!(link_score(later) < link_score(later + chrono::TimeDelta::microseconds(1)));
    }

    #[test]
    fn link_score_of_a_whole_millisecond_is_its_millisecond_count() {
        // The score links were stored with before, so stored and new links sort together.
        let whole = DateTime::from_timestamp_millis(1_700_000_000_123).unwrap();
        assert_eq!(
            link_score(whole).to_bits(),
            1_700_000_000_123.0_f64.to_bits()
        );
    }
}
