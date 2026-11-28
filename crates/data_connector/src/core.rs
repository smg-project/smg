// Core types for the data connector module.
// Contains all traits, data types, error types, and IDs for all storage backends.

use std::{
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter, Write},
    sync::Arc,
};

use async_trait::async_trait;
use chrono::{DateTime, SubsecRound, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};

use crate::common::describe_error;

// ============================================================================
// Shared helpers
// ============================================================================

/// Generate a 50-character hex string from 25 cryptographically random bytes.
/// Used by both `ConversationId::new()` and `make_item_id()`.
fn random_hex_id() -> String {
    let mut rng = rand::rng();
    let mut bytes = [0u8; 25];
    rng.fill_bytes(&mut bytes);
    let mut hex_string = String::with_capacity(50);
    for b in &bytes {
        // Writing to a String is infallible; discard the always-Ok result.
        let _ = write!(hex_string, "{b:02x}");
    }
    hex_string
}

// ============================================================================
// PART 1: Conversation Storage
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConversationId(pub String);

impl ConversationId {
    pub fn new() -> Self {
        Self(format!("conv_{}", random_hex_id()))
    }
}

impl Default for ConversationId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for ConversationId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ConversationId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl Display for ConversationId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Metadata payload persisted with a conversation
pub type ConversationMetadata = JsonMap<String, Value>;

/// Input payload for creating a conversation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NewConversation {
    /// Optional conversation ID (if None, a random ID will be generated)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<ConversationId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ConversationMetadata>,
    /// The tenant the conversation is scoped to (see [`Conversation::tenant`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

/// Stored conversation data structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Conversation {
    pub id: ConversationId,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ConversationMetadata>,
    /// The tenant the conversation is scoped to: the identity the gateway
    /// resolved for the request that created it, when it resolved one. A
    /// conversation stored under a tenant is served only to that tenant; one
    /// stored under none is served as before (see [`tenant_scope_allows`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

impl Conversation {
    pub fn new(new_conversation: NewConversation) -> Self {
        Self {
            id: new_conversation.id.unwrap_or_default(),
            created_at: Utc::now(),
            metadata: new_conversation.metadata,
            tenant: new_conversation.tenant,
        }
    }

    pub fn with_parts(
        id: ConversationId,
        created_at: DateTime<Utc>,
        metadata: Option<ConversationMetadata>,
        tenant: Option<String>,
    ) -> Self {
        Self {
            id,
            created_at,
            metadata,
            tenant,
        }
    }

    /// Whether a caller the gateway resolved `tenant` for may read, write
    /// into or delete this conversation (see [`tenant_scope_allows`]).
    pub fn is_visible_to(&self, tenant: Option<&str>) -> bool {
        tenant_scope_allows(self.tenant.as_deref(), tenant)
    }
}

/// The one rule of tenant scope, for responses and conversations alike: a
/// record stored under a tenant is served only to a caller of that tenant; a
/// record stored under none is served to every caller, as every record was
/// before records carried a tenant. A record has no tenant when the gateway
/// runs without tenant identity, when it resolved none for the request that
/// stored it (a caller known by its address only, or anonymous), or when it
/// was stored before this field existed.
pub fn tenant_scope_allows(record: Option<&str>, caller: Option<&str>) -> bool {
    match record {
        None => true,
        Some(record) => Some(record) == caller,
    }
}

/// The responses of a chain (oldest first, as [`ResponseStorage::get_response_chain`]
/// answers it) that a caller of `tenant` may see: the newest records down to
/// the first one stored under another tenant, where the history ends as it
/// would at a missing record. Empty when the chain's head itself is not
/// visible.
pub fn visible_chain(
    mut responses: Vec<Arc<StoredResponse>>,
    tenant: Option<&str>,
) -> Vec<Arc<StoredResponse>> {
    let hidden_up_to = responses
        .iter()
        .rposition(|response| !response.is_visible_to(tenant))
        .map_or(0, |index| index + 1);
    responses.split_off(hidden_up_to)
}

/// Result alias for conversation storage operations
pub type ConversationResult<T> = Result<T, ConversationStorageError>;

/// Error type for conversation storage operations
#[derive(Debug, thiserror::Error)]
pub enum ConversationStorageError {
    #[error("Conversation not found: {0}")]
    ConversationNotFound(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    /// The call exceeded the store's bound (`op_timeout_secs`): the store did
    /// not answer in time.
    #[error("Storage timeout: {0}")]
    Timeout(String),

    /// The store itself is not available: no connection to be had, the
    /// connection gone, an I/O or protocol failure beneath the call, the
    /// server shutting down or out of connections. Answered `503
    /// storage_unavailable`, as a timed-out call is.
    #[error("Storage unavailable: {0}")]
    Unavailable(String),

    /// A storage hook refused the operation on purpose: the endpoint's own
    /// failure, not an outage.
    #[error("Rejected by the storage hook: {0}")]
    Rejected(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Trait describing the CRUD interface for conversation storage backends
#[async_trait]
pub trait ConversationStorage: Send + Sync + 'static {
    async fn create_conversation(&self, input: NewConversation)
        -> ConversationResult<Conversation>;

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>>;

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>>;

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool>;

    /// How many conversations the store holds, for a store that can tell
    /// without a query (the in-memory backend); `None` otherwise.
    fn stored_record_count(&self) -> Option<usize> {
        None
    }
}

// ============================================================================
// PART 2: ConversationItem Storage
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConversationItemId(pub String);

impl Display for ConversationItemId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ConversationItemId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ConversationItemId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    pub id: ConversationItemId,
    pub response_id: Option<String>,
    pub item_type: String,
    pub role: Option<String>,
    pub content: Value,
    pub status: Option<String>,
    pub created_at: DateTime<Utc>,
    /// The tenant the item is scoped to: the identity the gateway resolved
    /// for the request that created it (a conversation write, a stored
    /// response), when it resolved one. An item is referenced by id (an
    /// `item_reference`, an item added under its own id) only by its tenant;
    /// one stored under none is served as before (see [`tenant_scope_allows`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

impl ConversationItem {
    /// Whether a caller the gateway resolved `tenant` for may reference or
    /// adopt this item by its id (see [`tenant_scope_allows`]).
    pub fn is_visible_to(&self, tenant: Option<&str>) -> bool {
        tenant_scope_allows(self.tenant.as_deref(), tenant)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewConversationItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<ConversationItemId>,
    pub response_id: Option<String>,
    pub item_type: String,
    pub role: Option<String>,
    pub content: Value,
    pub status: Option<String>,
    /// The tenant the item is scoped to (see [`ConversationItem::tenant`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortOrder {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListParams {
    pub limit: usize,
    pub order: SortOrder,
    pub after: Option<String>, // item_id cursor
}

/// Strictly increasing `added_at` stamps for the items one caller links to a conversation.
///
/// Every backend lists a conversation's items by `added_at`, then by item id, so items that
/// share a stamp come back in id order, which is random. A caller that links several items in
/// one operation stamps each of them from one sequence: the current time, or one microsecond
/// past the previous stamp when the clock has not moved at least that far. Stamps are whole
/// microseconds, the resolution the SQL backends store, so two stamps this sequence tells
/// apart stay apart in every backend.
#[derive(Debug, Default)]
pub struct AddedAtSequence {
    last: Option<DateTime<Utc>>,
}

impl AddedAtSequence {
    pub fn new() -> Self {
        Self::default()
    }

    /// The next stamp: later than every stamp this sequence handed out before.
    pub fn next_added_at(&mut self) -> DateTime<Utc> {
        let now = Utc::now().trunc_subsecs(6);
        let next = match self.last {
            Some(last) if now <= last => last + chrono::TimeDelta::microseconds(1),
            _ => now,
        };
        self.last = Some(next);
        next
    }
}

pub type ConversationItemResult<T> = Result<T, ConversationItemStorageError>;

#[derive(Debug, thiserror::Error)]
pub enum ConversationItemStorageError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    /// The call exceeded the store's bound (`op_timeout_secs`): the store did
    /// not answer in time.
    #[error("Storage timeout: {0}")]
    Timeout(String),

    /// The store itself is not available: no connection to be had, the
    /// connection gone, an I/O or protocol failure beneath the call, the
    /// server shutting down or out of connections. Answered `503
    /// storage_unavailable`, as a timed-out call is.
    #[error("Storage unavailable: {0}")]
    Unavailable(String),

    /// A storage hook refused the operation on purpose: the endpoint's own
    /// failure, not an outage.
    #[error("Rejected by the storage hook: {0}")]
    Rejected(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

#[async_trait]
pub trait ConversationItemStorage: Send + Sync + 'static {
    /// Store `item` under its id (`item.id`, or a fresh one made for its
    /// type) and answer it as stored. An item the store already holds under
    /// that id is kept as it is and answered, neither replaced nor merged: a
    /// stored item is written once and has no update path, and a caller that
    /// creates under an existing id means the stored item (the response
    /// persist, for a turn whose input echoes an item of an earlier response
    /// with the id it came with; the Conversations API, for an item added
    /// under its own id). The same answer on every backend.
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem>;

    /// Link `item_id` into the conversation at `added_at`. An item the
    /// conversation already has is moved, neither duplicated nor refused: its
    /// one link takes the new stamp, so the item lists once, at its new
    /// position, on every backend (the SQL stores upsert on the
    /// (conversation, item) key).
    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()>;

    /// Batch-link multiple items to a conversation in a single operation,
    /// each link as [`Self::link_item`] (an item the conversation already
    /// has moves). Default implementation loops over `link_item`; backends
    /// may override with a more efficient batched approach.
    async fn link_items(
        &self,
        conversation_id: &ConversationId,
        items: &[(ConversationItemId, DateTime<Utc>)],
    ) -> ConversationItemResult<()> {
        for (item_id, added_at) in items {
            self.link_item(conversation_id, item_id, *added_at).await?;
        }
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>>;

    /// Get a single item by ID
    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>>;

    /// Check if an item is linked to a conversation
    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool>;

    /// Delete an item link from a conversation (does not delete the item itself)
    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()>;

    /// How many items the store holds, for a store that can tell without a
    /// query (the in-memory backend); `None` otherwise.
    fn stored_record_count(&self) -> Option<usize> {
        None
    }
}

/// Helper to build id prefix based on item_type
pub fn make_item_id(item_type: &str) -> ConversationItemId {
    let hex_string = random_hex_id();

    let prefix = match item_type {
        "message" => "msg",
        "reasoning" => "rs",
        "mcp_call" => "mcp",
        "mcp_list_tools" => "mcpl",
        "function_call" => "fc",
        other => {
            // Fallback: first 3 letters of type or "itm"
            let fallback: String = other.chars().take(3).collect();
            if fallback.is_empty() {
                return ConversationItemId(format!("itm_{hex_string}"));
            }
            return ConversationItemId(format!("{fallback}_{hex_string}"));
        }
    };
    ConversationItemId(format!("{prefix}_{hex_string}"))
}

// ============================================================================
// PART 3: Response Storage
// ============================================================================

/// Response identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponseId(pub String);

impl ResponseId {
    pub fn new() -> Self {
        Self(ulid::Ulid::new().to_string())
    }
}

impl Display for ResponseId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Default for ResponseId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for ResponseId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ResponseId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Stored response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponse {
    /// Unique response ID
    pub id: ResponseId,

    /// ID of the previous response in the chain (if any)
    pub previous_response_id: Option<ResponseId>,

    /// Input items as JSON array
    pub input: Value,

    /// When this response was created
    pub created_at: DateTime<Utc>,

    /// Safety identifier for content moderation
    pub safety_identifier: Option<String>,

    /// Model used for generation
    pub model: Option<String>,

    /// Conversation id if associated with a conversation
    #[serde(default)]
    pub conversation_id: Option<String>,

    /// Raw OpenAI response payload
    #[serde(default)]
    pub raw_response: Value,

    /// The tenant the response is scoped to: the identity the gateway
    /// resolved for the request that stored it, when it resolved one. A
    /// response stored under a tenant is read, continued and deleted only by
    /// that tenant; one stored under none is served as before (see
    /// [`tenant_scope_allows`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

impl StoredResponse {
    pub fn new(previous_response_id: Option<ResponseId>) -> Self {
        Self {
            id: ResponseId::new(),
            previous_response_id,
            input: Value::Array(vec![]),
            created_at: Utc::now(),
            safety_identifier: None,
            model: None,
            conversation_id: None,
            raw_response: Value::Null,
            tenant: None,
        }
    }

    /// Whether a caller the gateway resolved `tenant` for may read, continue
    /// or delete this response (see [`tenant_scope_allows`]).
    pub fn is_visible_to(&self, tenant: Option<&str>) -> bool {
        tenant_scope_allows(self.tenant.as_deref(), tenant)
    }
}

/// The statuses a response does not leave any more: a record in one of them
/// is closed, and [`ResponseStorage::store_response_if_open`] refuses to
/// rewrite it. The one list behind the gateway's notion of a closed record
/// and the stores' conditional writes.
pub const TERMINAL_RESPONSE_STATUSES: [&str; 4] =
    ["completed", "incomplete", "failed", "cancelled"];

/// `values` as a quoted, comma-separated SQL list (`'a', 'b'`), for the
/// stores' `IN (...)` predicates over the status lists above.
pub(crate) fn sql_string_list(values: &[&str]) -> String {
    values
        .iter()
        .map(|value| format!("'{value}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

impl StoredResponse {
    /// Whether the stored object's status is one a response does not leave
    /// any more.
    pub fn is_closed(&self) -> bool {
        self.raw_response["status"]
            .as_str()
            .is_some_and(|status| TERMINAL_RESPONSE_STATUSES.contains(&status))
    }
}

/// What [`ResponseStorage::store_response_if_open`] found under the id.
#[derive(Debug, Clone)]
pub enum ConditionalStore {
    /// The record was open (or carried no status) and now reads as the
    /// response given.
    Stored,
    /// The record had reached a terminal status and stays as it is: this is
    /// it.
    Closed(Box<StoredResponse>),
    /// Nothing is stored under the id; nothing was written.
    Missing,
}

/// Response chain - a sequence of related responses. The records are shared
/// (`Arc`): a backend that holds them in memory hands out the ones it holds,
/// and a caller that rebuilds a conversation from a chain reads them in place
/// rather than taking a copy of every record, with its items, on every turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChain {
    /// The responses in chronological order
    pub responses: Vec<Arc<StoredResponse>>,

    /// Metadata about the chain
    pub metadata: HashMap<String, Value>,
}

impl Default for ResponseChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseChain {
    pub fn new() -> Self {
        Self {
            responses: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a response to the chain
    pub fn add_response(&mut self, response: StoredResponse) {
        self.responses.push(Arc::new(response));
    }

    /// Add a response the backend holds shared, without copying it.
    pub fn add_shared(&mut self, response: Arc<StoredResponse>) {
        self.responses.push(response);
    }
}

/// Error type for response storage operations
#[derive(Debug, thiserror::Error)]
pub enum ResponseStorageError {
    #[error("Response not found: {0}")]
    ResponseNotFound(String),

    #[error("Invalid chain: {0}")]
    InvalidChain(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    /// The call exceeded the store's bound (`op_timeout_secs`): the store did
    /// not answer in time.
    #[error("Storage timeout: {0}")]
    Timeout(String),

    /// The store itself is not available: no connection to be had, the
    /// connection gone, an I/O or protocol failure beneath the call, the
    /// server shutting down or out of connections. Answered `503
    /// storage_unavailable`, as a timed-out call is.
    #[error("Storage unavailable: {0}")]
    Unavailable(String),

    /// A storage hook refused the operation on purpose: the endpoint's own
    /// failure, not an outage.
    #[error("Rejected by the storage hook: {0}")]
    Rejected(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type ResponseResult<T> = Result<T, ResponseStorageError>;

/// What a caller can tell about a failed store call without reading the
/// backend's text.
pub trait StorageFailure {
    /// The call exceeded the store's bound (`op_timeout_secs`): the store is
    /// not answering for now.
    fn timed_out(&self) -> bool;
    /// Whether the store itself is not available: not answering within its
    /// bound, no connection to be had, the connection gone or failing beneath
    /// the call, the server shutting down or refusing connections. The API
    /// reports that as a 503 the client may retry; anything the store decided
    /// or anything about the record (a constraint it refused, a hook's
    /// rejection, a record not found or undecodable, an invalid chain) keeps
    /// the endpoint's own answer. The backends classify at the source (see
    /// [`Failure`]).
    fn is_unavailable(&self) -> bool;
}

macro_rules! storage_failure_impl {
    ($error:ty) => {
        impl StorageFailure for $error {
            fn timed_out(&self) -> bool {
                matches!(self, Self::Timeout(_))
            }
            fn is_unavailable(&self) -> bool {
                matches!(self, Self::Timeout(_) | Self::Unavailable(_))
            }
        }

        impl $error {
            /// The error for a classified backend failure.
            pub fn from_failure(failure: Failure) -> Self {
                match failure {
                    Failure::Unavailable(text) => Self::Unavailable(text),
                    Failure::Storage(text) => Self::StorageError(text),
                }
            }

            /// The error for a failure to take a connection from the pool.
            pub fn pool_unavailable(error: &dyn std::error::Error) -> Self {
                Self::from_failure(Failure::pool(error))
            }

            /// The error for a Redis failure, classified.
            pub fn from_redis(error: redis::RedisError) -> Self {
                Self::from_failure(Failure::redis(&error))
            }

            /// The error for a Postgres failure, classified.
            pub fn from_pg(error: tokio_postgres::Error) -> Self {
                Self::from_failure(Failure::postgres(&error))
            }
        }
    };
}
storage_failure_impl!(ResponseStorageError);
storage_failure_impl!(ConversationStorageError);
storage_failure_impl!(ConversationItemStorageError);

/// A backend failure, classified where it happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// The store is not available.
    Unavailable(String),
    /// The store answered with a failure of the operation.
    Storage(String),
}

impl Failure {
    /// The failure of taking a connection from the pool: the store is not
    /// available, whatever the pool's reason (no connection, a timeout); the
    /// cause chain stays in the text (`error connecting to server: Connection
    /// refused`, not the kind alone).
    pub fn pool(error: &dyn std::error::Error) -> Self {
        Self::Unavailable(format!(
            "no connection from the pool: {}",
            describe_error(error)
        ))
    }

    /// A Redis error: the connection and I/O kinds (refused, dropped, timed
    /// out, the cluster unreachable, a protocol failure the client cannot
    /// recover from) are the store being unavailable; the rest (a wrong
    /// type, a refused command) are the store's answer.
    pub fn redis(error: &redis::RedisError) -> Self {
        let text = error.to_string();
        if error.is_connection_refusal()
            || error.is_connection_dropped()
            || error.is_io_error()
            || error.is_timeout()
            || error.is_cluster_error()
            || error.is_unrecoverable_error()
        {
            Self::Unavailable(text)
        } else {
            Self::Storage(text)
        }
    }

    /// A Postgres error: one without a server answer (the connection closed,
    /// a timeout, an I/O or protocol failure) is the store being unavailable,
    /// and so is a server answer that says the server is not serving: a
    /// connection exception (class 08), insufficient resources such as too
    /// many connections (class 53), an operator intervention such as a
    /// shutdown or `cannot_connect_now` (57P..), or a write that reached a
    /// read-only standby after a failover (`25006`). Any other server answer
    /// (a constraint, a refused statement) is the store's own.
    pub fn postgres(error: &tokio_postgres::Error) -> Self {
        let text = describe_error(error);
        let Some(db) = error.as_db_error() else {
            return Self::Unavailable(text);
        };
        let code = db.code().code();
        if code.starts_with("08")
            || code.starts_with("53")
            || code.starts_with("57P")
            || *db.code() == tokio_postgres::error::SqlState::READ_ONLY_SQL_TRANSACTION
        {
            Self::Unavailable(text)
        } else {
            Self::Storage(text)
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/core/failure_tests.rs"]
mod failure_tests;

/// Trait for response storage
#[async_trait]
pub trait ResponseStorage: Send + Sync {
    /// Store a response under its id.
    ///
    /// A record already stored under the same id is replaced: a background
    /// response is written as `queued` before its work starts and again with
    /// its terminal object when the work ends, under the one id its client
    /// polls.
    async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId>;

    /// Store `response` under its id unless the record stored there has
    /// reached a terminal status ([`TERMINAL_RESPONSE_STATUSES`]) or is
    /// missing: the write of a background response's status transition
    /// (`in_progress`, `cancelled`, `failed`, the terminal
    /// object) is one conditional write, so a record closed on one gateway
    /// is never rewritten by another whatever the timing.
    ///
    /// The default reads then writes, two round trips with a window between
    /// them; every backend of this crate overrides it with one atomic
    /// operation (a check under the lock, a conditional `UPDATE`, a script).
    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> ResponseResult<ConditionalStore> {
        match self.get_response(&response.id).await? {
            None => Ok(ConditionalStore::Missing),
            Some(current) if current.is_closed() => Ok(ConditionalStore::Closed(Box::new(current))),
            Some(_) => {
                self.store_response(response).await?;
                Ok(ConditionalStore::Stored)
            }
        }
    }

    /// Moves the stored object of the record under `id` to `raw_response`
    /// unless the record has reached a terminal status or is missing: the
    /// write of a status transition (`in_progress`, `cancelled`, `failed`),
    /// which changes the object alone. The columns written when the record
    /// was created (the request's context, a hook's extra columns) are left
    /// as they are: a transition written under no request context must not
    /// reset them.
    ///
    /// The default reads the record and writes it whole through
    /// [`Self::store_response_if_open`]; every backend of this crate
    /// overrides it with one conditional write of the object's column.
    async fn update_response_object_if_open(
        &self,
        id: &ResponseId,
        raw_response: Value,
    ) -> ResponseResult<ConditionalStore> {
        match self.get_response(id).await? {
            None => Ok(ConditionalStore::Missing),
            Some(current) if current.is_closed() => Ok(ConditionalStore::Closed(Box::new(current))),
            Some(mut current) => {
                current.raw_response = raw_response;
                self.store_response_if_open(current).await
            }
        }
    }

    /// Get a response by ID
    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>>;

    /// Delete a response
    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()>;

    /// Get the chain of responses leading to a given response.
    ///
    /// Walks `previous_response_id` links from the given response backwards,
    /// collecting up to `max_depth` responses (or unlimited if `None`).
    /// Returns responses in chronological order (oldest first).
    ///
    /// The default implementation calls `self.get_response()` in a loop with
    /// cycle detection to prevent infinite loops from self-referencing chains.
    /// Backends that can walk the chain more efficiently (e.g. with a single
    /// lock or a recursive SQL query) should override this.
    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
        let mut chain = ResponseChain::new();
        let mut current_id = Some(response_id.clone());
        let mut seen = HashSet::new();

        while let Some(ref lookup_id) = current_id {
            if let Some(limit) = max_depth {
                if seen.len() >= limit {
                    break;
                }
            }

            // Cycle detection: error if we've already visited this ID.
            if !seen.insert(lookup_id.clone()) {
                return Err(ResponseStorageError::InvalidChain(format!(
                    "cycle detected at response {}",
                    lookup_id.0
                )));
            }

            let fetched = self.get_response(lookup_id).await?;
            match fetched {
                Some(response) => {
                    current_id.clone_from(&response.previous_response_id);
                    chain.add_response(response);
                }
                None => break,
            }
        }

        chain.responses.reverse();
        Ok(chain)
    }

    /// The part of [`Self::get_response_chain`] a caller of `tenant` may see:
    /// the newest records down to the first it may not see (this walk reads
    /// that one and stops); none for a hidden, missing or unstored id.
    ///
    /// The Responses loaders read history through this method, not through
    /// [`Self::get_response_chain`]: a backend that overrides that one with a
    /// faster walk overrides this one too, as the in-memory store does.
    async fn get_response_chain_for(
        &self,
        response_id: &ResponseId,
        tenant: Option<&str>,
    ) -> ResponseResult<ResponseChain> {
        let mut chain = ResponseChain::new();
        let mut seen = HashSet::new();
        let mut next = Some(response_id.clone());
        while let Some(id) = next.take() {
            if seen.contains(&id) {
                return Err(ResponseStorageError::InvalidChain(format!(
                    "cycle detected at response {}",
                    id.0
                )));
            }
            match self.get_response(&id).await? {
                Some(response)
                    if response.is_visible_to(tenant)
                        && !(seen.is_empty() && is_unstored(&response)) =>
                {
                    next.clone_from(&response.previous_response_id);
                    chain.add_response(response);
                }
                _ => break,
            }
            seen.insert(id);
        }
        chain.responses.reverse();
        Ok(chain)
    }

    /// Find the stored response whose `raw_response.output` carries an item
    /// with the given id (the lookup behind `item_reference` input items).
    ///
    /// The default answers `None`: a backend without an index over output
    /// items reports the item as unknown rather than scanning a table.
    async fn find_response_by_output_item(
        &self,
        item_id: &str,
    ) -> ResponseResult<Option<StoredResponse>> {
        let _ = item_id;
        Ok(None)
    }

    /// List recent responses for a safety identifier: those a caller of
    /// `tenant` may see (see [`StoredResponse::is_visible_to`]), newest first.
    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>>;

    /// Delete the responses for a safety identifier that a caller of `tenant`
    /// may see; answers how many went.
    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> ResponseResult<usize>;

    /// Ask the store to drop the response under `response_id` on its own
    /// once `after` has passed, where the store can keep such a deadline (a
    /// key with a time to live). The default keeps none: the gateway's own
    /// timer deletes the record while it runs, and a store without deadlines
    /// answers through [`Self::delete_unstored_older_than`].
    async fn expire_response(
        &self,
        response_id: &ResponseId,
        after: std::time::Duration,
    ) -> ResponseResult<()> {
        let _ = (response_id, after);
        Ok(())
    }

    /// Delete the records whose object says `store: false` and that reached a
    /// terminal status before `cutoff` (see `unstored_terminal_before`):
    /// what a gateway that stopped inside such a record's polling window, or
    /// died, left behind in a store that outlives it. The default deletes
    /// nothing: a store without a query over the stored object answers 0.
    async fn delete_unstored_older_than(&self, cutoff: DateTime<Utc>) -> ResponseResult<usize> {
        let _ = cutoff;
        Ok(0)
    }

    /// How many responses the store holds, for a store that can tell without
    /// a query (the in-memory backend); `None` otherwise.
    fn stored_record_count(&self) -> Option<usize> {
        None
    }
}

/// Whether `response` is the record of a response its client asked not to
/// store (`store: false` in its object): a background run's record, kept for
/// its polling window only. Such a record is no previous response and its
/// items are no stored items, as the public API answers for an unstored id.
pub fn is_unstored(response: &StoredResponse) -> bool {
    response.raw_response.get("store").and_then(Value::as_bool) == Some(false)
}

/// Whether `response` is [unstored](is_unstored) and reached a terminal
/// status ([`TERMINAL_RESPONSE_STATUSES`]) before `cutoff`: at its object's
/// `completed_at` when it carries one, at the record's creation otherwise.
pub fn unstored_terminal_before(response: &StoredResponse, cutoff: DateTime<Utc>) -> bool {
    let raw = &response.raw_response;
    if !is_unstored(response) {
        return false;
    }
    if !response.is_closed() {
        return false;
    }
    let ended = raw
        .get("completed_at")
        .and_then(Value::as_i64)
        .unwrap_or_else(|| response.created_at.timestamp());
    ended < cutoff.timestamp()
}

impl Default for StoredResponse {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
#[path = "../tests/unit/core/tests.rs"]
mod tests;
