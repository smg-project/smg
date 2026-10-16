// Core types for the data connector module.
// Contains all traits, data types, error types, and IDs for all storage backends.

use std::{
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter, Write},
};

use async_trait::async_trait;
use chrono::{DateTime, SubsecRound, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};

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
    mut responses: Vec<StoredResponse>,
    tenant: Option<&str>,
) -> Vec<StoredResponse> {
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

/// Response chain - a sequence of related responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChain {
    /// The responses in chronological order
    pub responses: Vec<StoredResponse>,

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
            crate::common::describe_error(error)
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
        let text = crate::common::describe_error(error);
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
mod failure_tests {
    use super::*;

    #[test]
    fn a_redis_connection_or_io_failure_is_unavailable_and_an_answer_is_not() {
        let gone = redis::RedisError::from((redis::ErrorKind::Io, "connection reset"));
        assert!(matches!(Failure::redis(&gone), Failure::Unavailable(_)));
        let refused = redis::RedisError::from(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "refused",
        ));
        assert!(matches!(Failure::redis(&refused), Failure::Unavailable(_)));
        let wrong_type =
            redis::RedisError::from((redis::ErrorKind::UnexpectedReturnType, "WRONGTYPE"));
        assert!(matches!(Failure::redis(&wrong_type), Failure::Storage(_)));
        // A protocol parse failure leaves the connection unusable: the client
        // library counts it unrecoverable, and so does the classification.
        let broken = redis::RedisError::from((redis::ErrorKind::Parse, "bad frame"));
        assert!(matches!(Failure::redis(&broken), Failure::Unavailable(_)));
    }

    #[test]
    fn a_pool_failure_keeps_its_cause_chain() {
        #[derive(Debug)]
        struct Outer(std::io::Error);
        impl Display for Outer {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                f.write_str("error connecting to server")
            }
        }
        impl std::error::Error for Outer {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let error = Outer(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "Connection refused (os error 111)",
        ));
        let Failure::Unavailable(text) = Failure::pool(&error) else {
            panic!("a pool failure is unavailable");
        };
        assert!(
            text.contains("error connecting to server: Connection refused (os error 111)"),
            "{text}"
        );
    }

    #[test]
    fn only_a_timeout_or_an_unavailable_store_is_unavailable() {
        assert!(ResponseStorageError::Timeout("bound".into()).is_unavailable());
        assert!(ResponseStorageError::Timeout("bound".into()).timed_out());
        assert!(ResponseStorageError::Unavailable("gone".into()).is_unavailable());
        assert!(!ResponseStorageError::Unavailable("gone".into()).timed_out());
        assert!(!ConversationStorageError::StorageError("duplicate key".into()).is_unavailable());
        assert!(!ConversationStorageError::Rejected("tenant".into()).is_unavailable());
        assert!(
            !ConversationItemStorageError::from_failure(Failure::Storage("x".into()))
                .is_unavailable()
        );
        assert!(
            ConversationItemStorageError::from_failure(Failure::Unavailable("x".into()))
                .is_unavailable()
        );
    }
}

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
                    chain.responses.push(response);
                }
                None => break,
            }
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
    /// terminal status before `cutoff` (see [`unstored_terminal_before`]):
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
/// status (the statuses a response does not leave: completed, incomplete,
/// failed, cancelled) before `cutoff`: at its object's `completed_at` when it
/// carries one, at the record's creation otherwise.
pub fn unstored_terminal_before(response: &StoredResponse, cutoff: DateTime<Utc>) -> bool {
    let raw = &response.raw_response;
    if !is_unstored(response) {
        return false;
    }
    if !matches!(
        raw.get("status").and_then(Value::as_str),
        Some("completed" | "incomplete" | "failed" | "cancelled")
    ) {
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
mod tests {
    use std::collections::HashSet;

    use super::*;

    // ========================================================================
    // ConversationId tests
    // ========================================================================

    #[test]
    fn conversation_id_new_has_conv_prefix() {
        let id = ConversationId::new();
        assert!(
            id.0.starts_with("conv_"),
            "ConversationId should start with 'conv_', got: {id}"
        );
    }

    #[test]
    fn conversation_id_new_generates_unique_ids() {
        let ids: HashSet<String> = (0..100).map(|_| ConversationId::new().0).collect();
        assert_eq!(ids.len(), 100, "all 100 ConversationIds should be unique");
    }

    #[test]
    fn conversation_id_new_has_consistent_length() {
        // "conv_" (5 chars) + 50 hex chars = 55 total
        for _ in 0..10 {
            let id = ConversationId::new();
            assert_eq!(
                id.0.len(),
                55,
                "ConversationId should be 55 chars (conv_ + 50 hex), got {} chars: {id}",
                id.0.len()
            );
        }
    }

    #[test]
    fn conversation_id_default_works_same_as_new() {
        let id = ConversationId::default();
        assert!(
            id.0.starts_with("conv_"),
            "Default ConversationId should start with 'conv_', got: {id}"
        );
        assert_eq!(id.0.len(), 55, "Default ConversationId should be 55 chars");
    }

    #[test]
    fn conversation_id_from_string() {
        let id = ConversationId::from("my_custom_id".to_string());
        assert_eq!(id.0, "my_custom_id");
    }

    #[test]
    fn conversation_id_from_str() {
        let id = ConversationId::from("my_custom_id");
        assert_eq!(id.0, "my_custom_id");
    }

    #[test]
    fn conversation_id_display() {
        let id = ConversationId::from("conv_abc123");
        assert_eq!(format!("{id}"), "conv_abc123");
    }

    // ========================================================================
    // ConversationItemId tests
    // ========================================================================

    #[test]
    fn conversation_item_id_from_string() {
        let id = ConversationItemId::from("item_123".to_string());
        assert_eq!(id.0, "item_123");
    }

    #[test]
    fn conversation_item_id_from_str() {
        let id = ConversationItemId::from("item_456");
        assert_eq!(id.0, "item_456");
    }

    #[test]
    fn conversation_item_id_display() {
        let id = ConversationItemId::from("msg_abc");
        assert_eq!(format!("{id}"), "msg_abc");
    }

    // ========================================================================
    // ResponseId tests
    // ========================================================================

    #[test]
    fn response_id_new_generates_valid_ulid() {
        let id = ResponseId::new();
        // ULID strings are 26 characters, uppercase alphanumeric (Crockford Base32)
        assert_eq!(
            id.0.len(),
            26,
            "ULID string should be 26 chars, got {} chars: {}",
            id.0.len(),
            id.0
        );
        assert!(
            id.0.chars().all(|c| c.is_ascii_alphanumeric()),
            "ULID should contain only alphanumeric characters, got: {}",
            id.0
        );
    }

    #[test]
    fn response_id_new_generates_unique_ids() {
        let ids: HashSet<String> = (0..100).map(|_| ResponseId::new().0).collect();
        assert_eq!(ids.len(), 100, "all 100 ResponseIds should be unique");
    }

    #[test]
    fn response_id_default_works_same_as_new() {
        let id = ResponseId::default();
        assert_eq!(id.0.len(), 26, "Default ResponseId should be 26-char ULID");
    }

    #[test]
    fn response_id_from_string() {
        let id = ResponseId::from("resp_custom".to_string());
        assert_eq!(id.0, "resp_custom");
    }

    #[test]
    fn response_id_from_str() {
        let id = ResponseId::from("resp_custom");
        assert_eq!(id.0, "resp_custom");
    }

    // ========================================================================
    // make_item_id() tests
    // ========================================================================

    #[test]
    fn make_item_id_message_prefix() {
        let id = make_item_id("message");
        assert!(
            id.0.starts_with("msg_"),
            "message type should produce 'msg_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_reasoning_prefix() {
        let id = make_item_id("reasoning");
        assert!(
            id.0.starts_with("rs_"),
            "reasoning type should produce 'rs_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_mcp_call_prefix() {
        let id = make_item_id("mcp_call");
        assert!(
            id.0.starts_with("mcp_"),
            "mcp_call type should produce 'mcp_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_mcp_list_tools_prefix() {
        let id = make_item_id("mcp_list_tools");
        assert!(
            id.0.starts_with("mcpl_"),
            "mcp_list_tools type should produce 'mcpl_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_function_call_prefix() {
        let id = make_item_id("function_call");
        assert!(
            id.0.starts_with("fc_"),
            "function_call type should produce 'fc_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_unknown_type_uses_first_3_chars() {
        let id = make_item_id("custom_type");
        assert!(
            id.0.starts_with("cus_"),
            "unknown type 'custom_type' should produce 'cus_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_empty_type_uses_itm() {
        let id = make_item_id("");
        assert!(
            id.0.starts_with("itm_"),
            "empty type string should produce 'itm_' prefix, got: {id}"
        );
    }

    #[test]
    fn make_item_id_correct_length() {
        // Each known prefix: prefix + "_" + 50 hex chars
        let test_cases = vec![
            ("message", "msg_"),
            ("reasoning", "rs_"),
            ("mcp_call", "mcp_"),
            ("mcp_list_tools", "mcpl_"),
            ("function_call", "fc_"),
        ];

        for (item_type, prefix) in test_cases {
            let id = make_item_id(item_type);
            let expected_len = prefix.len() + 50;
            assert_eq!(
                id.0.len(),
                expected_len,
                "make_item_id(\"{item_type}\") should be {expected_len} chars ('{prefix}' + 50 hex), got {} chars: {id}",
                id.0.len()
            );
        }

        // Unknown type: first 3 chars + "_" + 50 hex = 54 chars
        let id = make_item_id("custom_type");
        assert_eq!(
            id.0.len(),
            54,
            "unknown type should be 54 chars (3 char prefix + '_' + 50 hex), got {} chars: {id}",
            id.0.len()
        );

        // Empty type: "itm_" + 50 hex = 54 chars
        let id = make_item_id("");
        assert_eq!(
            id.0.len(),
            54,
            "empty type should be 54 chars ('itm_' + 50 hex), got {} chars: {id}",
            id.0.len()
        );
    }

    // ========================================================================
    // Conversation tests
    // ========================================================================

    #[test]
    fn conversation_new_generates_id_if_none_provided() {
        let conv = Conversation::new(NewConversation {
            id: None,
            metadata: None,
            tenant: None,
        });
        assert!(
            conv.id.0.starts_with("conv_"),
            "should generate a ConversationId when none provided, got: {}",
            conv.id
        );
    }

    #[test]
    fn conversation_new_uses_provided_id() {
        let custom_id = ConversationId::from("my_conv_id");
        let conv = Conversation::new(NewConversation {
            id: Some(custom_id.clone()),
            metadata: None,
            tenant: None,
        });
        assert_eq!(conv.id, custom_id, "should use the provided ConversationId");
    }

    #[test]
    fn conversation_with_parts_preserves_all_fields() {
        let id = ConversationId::from("test_id");
        let created_at = Utc::now();
        let mut metadata = ConversationMetadata::new();
        metadata.insert("key".to_string(), Value::String("value".to_string()));

        let conv = Conversation::with_parts(
            id.clone(),
            created_at,
            Some(metadata.clone()),
            Some("header:team-red".to_string()),
        );

        assert_eq!(conv.id, id);
        assert_eq!(conv.created_at, created_at);
        assert_eq!(conv.metadata, Some(metadata));
        assert_eq!(conv.tenant.as_deref(), Some("header:team-red"));
    }

    /// A chain is served down to the first record of another tenant; a head
    /// of another tenant hides the whole chain.
    #[test]
    fn a_chain_is_cut_at_the_first_record_of_another_tenant() {
        let stored = |id: &str, tenant: Option<&str>| {
            let mut stored = StoredResponse::new(None);
            stored.id = ResponseId::from(id);
            stored.tenant = tenant.map(str::to_string);
            stored
        };
        let ids = |chain: Vec<StoredResponse>| {
            chain
                .into_iter()
                .map(|stored| stored.id.0)
                .collect::<Vec<_>>()
        };
        let a = Some("header:a");
        let chain = || {
            vec![
                stored("resp_1", None),
                stored("resp_2", Some("header:b")),
                stored("resp_3", a),
                stored("resp_4", a),
            ]
        };
        assert_eq!(ids(visible_chain(chain(), a)), ["resp_3", "resp_4"]);
        assert!(
            visible_chain(chain(), Some("header:b")).is_empty(),
            "the head is a's"
        );
        assert!(visible_chain(chain(), None).is_empty());
        let shared = vec![stored("resp_1", None), stored("resp_2", None)];
        assert_eq!(
            ids(visible_chain(shared, Some("header:b"))),
            ["resp_1", "resp_2"],
            "records without a tenant are served as before"
        );
        assert!(visible_chain(Vec::new(), a).is_empty());
    }

    /// A record stored under a tenant is served to that tenant only; a record
    /// stored under none is served to every caller, as before.
    #[test]
    fn tenant_scope_serves_a_tenanted_record_to_its_tenant_only() {
        assert!(tenant_scope_allows(None, None));
        assert!(tenant_scope_allows(None, Some("header:b")));
        assert!(tenant_scope_allows(Some("header:a"), Some("header:a")));
        assert!(!tenant_scope_allows(Some("header:a"), Some("header:b")));
        assert!(!tenant_scope_allows(Some("header:a"), None));

        let mut response = StoredResponse::new(None);
        assert!(response.is_visible_to(Some("auth:anyone")));
        response.tenant = Some("auth:team-red".to_string());
        assert!(response.is_visible_to(Some("auth:team-red")));
        assert!(!response.is_visible_to(Some("auth:team-blue")));
        assert!(!response.is_visible_to(None));

        let conversation = Conversation::new(NewConversation {
            id: None,
            metadata: None,
            tenant: Some("header:team-red".to_string()),
        });
        assert_eq!(conversation.tenant.as_deref(), Some("header:team-red"));
        assert!(conversation.is_visible_to(Some("header:team-red")));
        assert!(!conversation.is_visible_to(Some("header:team-blue")));
        assert!(!conversation.is_visible_to(None));
    }

    // ========================================================================
    // StoredResponse tests
    // ========================================================================

    #[test]
    fn stored_response_new_none_has_no_previous() {
        let resp = StoredResponse::new(None);
        assert!(
            resp.previous_response_id.is_none(),
            "new(None) should have no previous_response_id"
        );
    }

    #[test]
    fn stored_response_new_some_has_correct_previous() {
        let prev_id = ResponseId::from("prev_123");
        let resp = StoredResponse::new(Some(prev_id.clone()));
        assert_eq!(
            resp.previous_response_id,
            Some(prev_id),
            "new(Some(id)) should set previous_response_id"
        );
    }

    #[test]
    fn stored_response_default_works() {
        let resp = StoredResponse::default();
        assert!(
            resp.previous_response_id.is_none(),
            "default() should have no previous_response_id"
        );
        assert_eq!(
            resp.input,
            Value::Array(vec![]),
            "default input should be empty array"
        );
        assert_eq!(
            resp.raw_response,
            Value::Null,
            "default raw_response should be Null"
        );
    }

    // ========================================================================
    // ResponseChain tests
    // ========================================================================

    #[test]
    fn response_chain_new_creates_empty_chain() {
        let chain = ResponseChain::new();
        assert!(
            chain.responses.is_empty(),
            "new chain should have no responses"
        );
        assert!(
            chain.metadata.is_empty(),
            "new chain should have no metadata"
        );
    }

    #[test]
    fn response_chain_add_response_appends() {
        let mut chain = ResponseChain::new();
        let r1 = StoredResponse::new(None);
        let r2 = StoredResponse::new(None);
        let r1_id = r1.id.clone();
        let r2_id = r2.id.clone();

        chain.add_response(r1);
        assert_eq!(chain.responses.len(), 1, "chain should have 1 response");

        chain.add_response(r2);
        assert_eq!(chain.responses.len(), 2, "chain should have 2 responses");
        assert_eq!(chain.responses[0].id, r1_id, "first response should be r1");
        assert_eq!(chain.responses[1].id, r2_id, "second response should be r2");
    }

    // ========================================================================
    // AddedAtSequence tests
    // ========================================================================

    #[test]
    fn added_at_sequence_stamps_strictly_increase_at_microsecond_resolution() {
        let mut stamps = AddedAtSequence::new();
        let started = Utc::now();
        let mut previous = stamps.next_added_at();
        assert!(previous >= started.trunc_subsecs(6));
        // A tight loop asks faster than the clock moves: every stamp still gets its own
        // microsecond...
        for _ in 0..1000 {
            let next = stamps.next_added_at();
            assert!(next.timestamp_micros() > previous.timestamp_micros());
            assert_eq!(
                next.timestamp_subsec_nanos() % 1000,
                0,
                "whole microseconds"
            );
            previous = next;
        }
        // ...and the sequence runs at most one microsecond per stamp ahead of the clock.
        assert!(previous <= Utc::now() + chrono::TimeDelta::microseconds(1001));
    }

    #[test]
    fn added_at_sequence_follows_the_clock_once_it_has_moved() {
        let mut stamps = AddedAtSequence::new();
        let first = stamps.next_added_at();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let second = stamps.next_added_at();
        assert!(second - first >= chrono::TimeDelta::milliseconds(1));
    }

    /// A store that overrides neither conditional write runs the trait's
    /// read-then-write defaults: a record that cannot be read is an error to
    /// the caller, never an open record, and nothing is written; a closed
    /// record is handed back and left as it is; a missing one takes nothing;
    /// an open one takes the write.
    #[tokio::test]
    async fn the_default_conditional_writes_write_nothing_over_a_record_they_cannot_read() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use parking_lot::Mutex;

        enum Read {
            Fails,
            Answers(Option<Box<StoredResponse>>),
        }

        struct PlainStore {
            read: Mutex<Read>,
            stores: AtomicUsize,
        }

        #[async_trait]
        impl ResponseStorage for PlainStore {
            async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId> {
                self.stores.fetch_add(1, Ordering::SeqCst);
                Ok(response.id)
            }

            async fn get_response(
                &self,
                _response_id: &ResponseId,
            ) -> ResponseResult<Option<StoredResponse>> {
                match &*self.read.lock() {
                    Read::Fails => Err(ResponseStorageError::Timeout("read timed out".to_string())),
                    Read::Answers(record) => Ok(record.as_deref().cloned()),
                }
            }

            async fn delete_response(&self, _response_id: &ResponseId) -> ResponseResult<()> {
                Ok(())
            }

            async fn list_identifier_responses(
                &self,
                _identifier: &str,
                _tenant: Option<&str>,
                _limit: Option<usize>,
            ) -> ResponseResult<Vec<StoredResponse>> {
                Ok(Vec::new())
            }

            async fn delete_identifier_responses(
                &self,
                _identifier: &str,
                _tenant: Option<&str>,
            ) -> ResponseResult<usize> {
                Ok(0)
            }
        }

        let record = |status: &str| {
            let mut response = StoredResponse::new(None);
            response.id = ResponseId::from("resp_1");
            response.raw_response = serde_json::json!({"id": "resp_1", "status": status});
            response
        };
        let moved = serde_json::json!({"id": "resp_1", "status": "in_progress"});
        let store = PlainStore {
            read: Mutex::new(Read::Fails),
            stores: AtomicUsize::new(0),
        };
        let stores = || store.stores.load(Ordering::SeqCst);

        assert!(
            store
                .store_response_if_open(record("completed"))
                .await
                .is_err(),
            "a failed read is the caller's error, not an open record"
        );
        assert!(store
            .update_response_object_if_open(&ResponseId::from("resp_1"), moved.clone())
            .await
            .is_err());
        assert_eq!(
            stores(),
            0,
            "nothing is written while the record's state is unknown"
        );

        *store.read.lock() = Read::Answers(Some(Box::new(record("cancelled"))));
        let ConditionalStore::Closed(current) = store
            .store_response_if_open(record("completed"))
            .await
            .unwrap()
        else {
            panic!("a cancelled record is closed");
        };
        assert_eq!(current.raw_response["status"], "cancelled");
        assert!(matches!(
            store
                .update_response_object_if_open(&ResponseId::from("resp_1"), moved.clone())
                .await
                .unwrap(),
            ConditionalStore::Closed(_)
        ));
        assert_eq!(stores(), 0, "a closed record is left as it is");

        *store.read.lock() = Read::Answers(None);
        assert!(matches!(
            store
                .store_response_if_open(record("completed"))
                .await
                .unwrap(),
            ConditionalStore::Missing
        ));
        assert!(matches!(
            store
                .update_response_object_if_open(&ResponseId::from("resp_1"), moved.clone())
                .await
                .unwrap(),
            ConditionalStore::Missing
        ));
        assert_eq!(stores(), 0, "nothing is written under a missing id");

        *store.read.lock() = Read::Answers(Some(Box::new(record("queued"))));
        assert!(matches!(
            store
                .store_response_if_open(record("completed"))
                .await
                .unwrap(),
            ConditionalStore::Stored
        ));
        assert!(matches!(
            store
                .update_response_object_if_open(&ResponseId::from("resp_1"), moved)
                .await
                .unwrap(),
            ConditionalStore::Stored
        ));
        assert_eq!(stores(), 2, "an open record takes both writes");
    }
}
