//! Data connector module for response storage and conversation storage.
//!
//! Provides storage backends for:
//! - Conversations
//! - Conversation items
//! - Responses
//!
//! Supported backends:
//! - Memory (default)
//! - None (no-op)
//! - Oracle ATP
//! - Postgres
//! - Redis

mod common;
pub mod config;
pub mod context;
mod core;
mod factory;
mod hooked;
pub mod hooks;
mod memory;
mod noop;
mod oracle;
mod oracle_migrations;
mod postgres;
mod postgres_migrations;
mod redis;
pub mod schema;
pub(crate) mod versioning;

// Re-export config types
// Re-export core types and traits
pub use core::{
    AddedAtSequence, Conversation, ConversationId, ConversationItem, ConversationItemId,
    ConversationItemStorage, ConversationStorage, ListParams, NewConversation, NewConversationItem,
    ResponseId, ResponseStorage, ResponseStorageError, SortOrder, StoredResponse,
};

pub use config::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig};
// Re-export hook infrastructure
pub use context::{
    current_extra_columns, current_request_context, with_extra_columns, with_request_context,
    RequestContext,
};
// Re-export factory
pub use factory::{create_storage, StorageBundle, StorageFactoryConfig};
pub use hooks::{BeforeHookResult, ExtraColumns, HookError, StorageHook, StorageOperation};
// Re-export memory implementations for testing
pub use memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage};
// Re-export schema config types
pub use schema::{ColumnDef, SchemaConfig, TableConfig};
