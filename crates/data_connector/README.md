# data-connector

**Version:** 2.0.0 | **License:** Apache-2.0

Pluggable storage abstraction for the Shepherd Model Gateway (SMG). Provides
trait-based backends for persisting conversations, conversation items, and
responses. Backend selection is a runtime decision driven by configuration,
allowing the same application binary to target in-memory storage during
development and a production database in deployment.

## Architecture

### Core Traits

The crate defines three async traits in `core.rs`. Every backend implements all
three:

| Trait | Responsibility |
|-------|---------------|
| `ConversationStorage` | CRUD for conversation records (create, get, update, delete) |
| `ConversationItemStorage` | Item creation, linking items to conversations, cursor-based listing, deletion of links |
| `ResponseStorage` | Store/retrieve responses, walk response chains, index and query by safety identifier |

All traits are `Send + Sync + 'static` and use `async_trait` so they can be
held behind `Arc<dyn Trait>`.

### Factory Pattern

`create_storage()` is the single entry point. It accepts a
`StorageFactoryConfig` (backend selector plus optional per-backend configs) and
returns a `StorageTuple`:

```rust
pub type StorageTuple = (
    Arc<dyn ResponseStorage>,
    Arc<dyn ConversationStorage>,
    Arc<dyn ConversationItemStorage>,
);
```

### Supported Backends

| Backend | Variant | Description |
|---------|---------|-------------|
| **Memory** | `HistoryBackend::Memory` | In-process `HashMap`/`BTreeMap` storage. Default. No persistence across restarts. Suitable for development and testing. |
| **None / NoOp** | `HistoryBackend::None` | Accepts all writes silently, returns empty on reads. Use when persistence is intentionally disabled. |
| **PostgreSQL** | `HistoryBackend::Postgres` | Production backend using `tokio-postgres` with `deadpool` connection pooling. Fully async. Tables are auto-created on first connection. |
| **Redis** | `HistoryBackend::Redis` | Production backend using `deadpool-redis`. Supports optional TTL-based data retention (`retention_days`). |
| **Oracle ATP** | `HistoryBackend::Oracle` | Enterprise backend using the synchronous `oracle` crate. Async bridging is handled via `tokio::task::spawn_blocking`. Tables are auto-created on initialization. |

## Usage

```rust
use data_connector::{
    create_storage, StorageFactoryConfig, HistoryBackend,
    NewConversation, NewConversationItem, StoredResponse,
};
use serde_json::json;
use std::sync::Arc;

// Build factory config -- memory backend needs no extra configuration.
let config = StorageFactoryConfig {
    backend: &HistoryBackend::Memory,
    oracle: None,
    postgres: None,
    redis: None,
    hook: None,
};

let (responses, conversations, items) = create_storage(config).await.unwrap();

// Create a conversation
let conv = conversations
    .create_conversation(NewConversation { id: None, metadata: None })
    .await
    .unwrap();

// Store a response
let mut resp = StoredResponse::new(None);
resp.input = json!([{"role": "user", "content": "Hello"}]);
let resp_id = responses.store_response(resp).await.unwrap();

// Create and link a conversation item
let item = items
    .create_item(NewConversationItem {
        id: None,
        response_id: Some(resp_id.0.clone()),
        item_type: "message".to_string(),
        role: Some("user".to_string()),
        content: json!([]),
        status: Some("completed".to_string()),
    })
    .await
    .unwrap();

items.link_item(&conv.id, &item.id, chrono::Utc::now()).await.unwrap();
```

## Configuration

Backend selection is controlled by the `HistoryBackend` enum
(`"memory"`, `"none"`, `"oracle"`, `"postgres"`, `"redis"` when deserialized
from JSON/YAML). Each database backend has a dedicated config struct.

### `PostgresConfig`

| Field | Type | Description |
|-------|------|-------------|
| `db_url` | `String` | Connection URL (`postgres://user:pass@host:port/dbname`). Validated for scheme, host, and database name. |
| `pool_max` | `usize` | Maximum connections in the deadpool pool (default helper: 16). Must be > 0. |
| `op_timeout_secs` | `u64` | Bound in seconds on one call to the store (pool wait, connect and the statement; default 5, must be > 0). A call past it fails as a `Timeout` error, which the gateway answers `503 storage_unavailable`. |
| `schema` | `Option<SchemaConfig>` | Optional schema customization. See [Schema Configuration](#schema-configuration). |

Call `validate()` to check the URL before use.

### `RedisConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `String` | -- | Connection URL (`redis://` or `rediss://`). |
| `pool_max` | `usize` | 16 | Maximum pool connections. |
| `retention_days` | `Option<u64>` | `Some(30)` | TTL in days for stored data. `None` disables expiration. |
| `op_timeout_secs` | `u64` | 5 | Bound in seconds on one call to the store (pool wait, connect and the command; must be > 0). Replaces the client library's own 500 ms response / 1 s connection defaults. A call past it fails as a `Timeout` error, which the gateway answers `503 storage_unavailable`. |
| `schema` | `Option<SchemaConfig>` | `None` | Optional schema customization. See [Schema Configuration](#schema-configuration). |

Call `validate()` to check the URL before use.

### `OracleConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wallet_path` | `Option<String>` | `None` | Path to ATP wallet / TLS config directory. |
| `connect_descriptor` | `String` | -- | DSN, e.g. `tcps://host:port/service`. |
| `external_auth` | `bool` | `false` | Use OS/external authentication. |
| `username` | `String` | -- | Database username. |
| `password` | `String` | -- | Database password (redacted in `Debug` output). |
| `pool_min` | `usize` | 1 | Minimum pool connections. |
| `pool_max` | `usize` | 16 | Maximum pool connections. |
| `pool_timeout_secs` | `u64` | 30 | Connection acquisition timeout in seconds. |
| `schema` | `Option<SchemaConfig>` | `None` | Optional schema customization. See [Schema Configuration](#schema-configuration). |

### Schema Configuration

All three database backends (Oracle, Postgres, Redis) accept an optional
`SchemaConfig` that lets you customize table names and column names without
modifying source code. When `schema` is omitted, all backends use their
default table and column names — zero behavioral change.

```yaml
postgres:
  db_url: "postgres://user:pass@localhost:5432/mydb"
  pool_max: 16

  schema:
    owner: "myschema"           # Oracle: schema prefix (MYSCHEMA."TABLE")
                                # Redis: key prefix ("myschema:conversation:{id}")
                                # Postgres: ignored (use search_path for schema control)
    version: 2                  # Skip migrations 1–2 (database already at v2)
    auto_migrate: true          # Opt in to automatic migration (default: false)

    conversations:
      table: "my_conversations" # Overrides default "conversations"
      columns:
        id: "conv_id"           # Overrides column name "id" -> "conv_id"
        metadata: "conv_meta"   # Overrides column name "metadata" -> "conv_meta"

    responses:
      table: "my_responses"
      columns:
        safety_identifier: "user_identifier"

    conversation_items:
      table: "my_items"

    conversation_item_links:
      table: "my_links"
```

`SchemaConfig` has two types:

| Type | Fields | Purpose |
|------|--------|---------|
| `SchemaConfig` | `owner`, `version`, `auto_migrate`, `conversations`, `responses`, `conversation_items`, `conversation_item_links` | Top-level config with an optional owner/prefix, migration control, and per-table settings |
| `TableConfig` | `table`, `columns` | Per-table config: physical table name and a map of logical-to-physical column name overrides |

Key behaviors:

- **`col(field)`** returns the physical column name for a logical field name.
  If no override is configured, the logical name is returned unchanged.
- **`qualified_table(owner)`** returns `OWNER."TABLE"` when an owner is set
  (used by Oracle), or just the table name otherwise.
- **Validation** runs at startup. All identifiers must match `[a-zA-Z0-9_]+`.
  Invalid identifiers are rejected before any queries execute.
- **Redis**: Only `owner` (key prefix) and `columns` (hash field names) affect
  Redis behavior. The `table` field is ignored for Redis key patterns — keys
  always use hardcoded entity names (`conversation`, `item`, `response`).

## Storage Hooks

Hooks let you inject custom logic before and after storage operations without
modifying backend code. Use cases include audit logging, multi-tenancy field
population, PII redaction, and custom validation.

### `StorageHook` Trait

Implement the `StorageHook` trait (in `hooks.rs`):

```rust
#[async_trait]
pub trait StorageHook: Send + Sync + 'static {
    async fn before(
        &self,
        operation: StorageOperation,
        context: Option<&RequestContext>,
        payload: &Value,
    ) -> Result<BeforeHookResult, HookError>;

    async fn after(
        &self,
        operation: StorageOperation,
        context: Option<&RequestContext>,
        payload: &Value,
        result: &Value,
        extra: &ExtraColumns,
    ) -> Result<ExtraColumns, HookError>;
}
```

- `before()` returning `Continue(extra_columns)` proceeds with the operation.
  The extra columns map is forwarded to the backend for persistence.
- `before()` returning `Reject(reason)` aborts the operation with an error.
- `before()` returning `Err(_)` logs a warning and **continues** (non-fatal).
- `after()` receives the result and extra columns from `before()`. It can
  return modified extra columns for the caller.

### Wiring a Hook

Pass the hook to `create_storage()` via `StorageFactoryConfig`:

```rust
let hook = Arc::new(MyHook::new());
let config = StorageFactoryConfig {
    backend: &HistoryBackend::Postgres,
    postgres: Some(pg_config),
    hook: Some(hook),
    ..Default::default()
};
let (responses, conversations, items) = create_storage(config).await?;
```

### Request Context

`RequestContext` is a per-request key-value bag (populated from HTTP headers,
middleware, etc.) made available to hooks via tokio task-local storage:

```rust
use data_connector::context::{with_request_context, RequestContext};

let mut ctx = RequestContext::new();
ctx.set("tenant_id", "acme-corp");
ctx.set("user_id", "user-42");

// Run storage operations with context available to hooks
with_request_context(ctx, async {
    conversations.create_conversation(input).await
}).await;
```

### Extra Columns

Extra columns let hooks persist custom fields alongside core data. They are
defined in `SchemaConfig` and populated by hook `before()` calls.

```yaml
schema:
  responses:
    extra_columns:
      TENANT_ID:
        sql_type: "VARCHAR(128)"
      STORED_BY:
        sql_type: "VARCHAR(128)"
        default_value: "system"   # Used when hook doesn't provide a value
  conversations:
    extra_columns:
      TENANT_ID:
        sql_type: "VARCHAR(128)"
      CREATED_BY:
        sql_type: "VARCHAR(128)"
```

Value resolution order for each extra column on write:
1. Hook-provided value from `ExtraColumns` (via `before()` result)
2. `default_value` from `ColumnDef` in schema config
3. `NULL` (if neither is available)

Extra columns are write-side enrichment (audit trail, tenant ID, etc.). They
are included in DDL for schema creation and in INSERT statements, but are
**not** read back into `StoredResponse`/`Conversation`/`ConversationItem`
structs on SELECT.

### Skip Columns

Skip columns let you omit core columns from DDL, INSERT, and SELECT
statements. This is useful when migrating to a schema that doesn't have all
default columns.

```yaml
schema:
  responses:
    skip_columns:
      - raw_response
      - safety_identifier
```

Skipped columns use default values when reading (e.g. `None` for optional
fields, empty collections for lists/maps, `Value::Null` for JSON).

### WASM Storage Hooks

For sandboxed, language-agnostic hooks, use the WASM Component Model bridge.
The WIT interface is in `wasm/src/interface/storage/storage-hooks.wit` and
the Rust bridge is `WasmStorageHook` in the `smg-wasm` crate (behind the
`storage-hooks` feature flag).

```rust
use smg_wasm::WasmStorageHook;  // requires: smg-wasm with "storage-hooks" feature

let wasm_bytes = std::fs::read("path/to/storage_hook.component.wasm")?;
let hook = WasmStorageHook::new(&wasm_bytes)?;
// Use Arc::new(hook) in StorageFactoryConfig
```

See `examples/wasm/wasm-guest-storage-hook/` for a complete guest example
demonstrating multi-tenancy and audit trail extra columns.

## Data Model

### Conversations

A `Conversation` has an `id` (`ConversationId`), a `created_at` timestamp, and
optional JSON `metadata`. Conversations act as containers for ordered sets of
conversation items.

### Conversation Items

A `ConversationItem` represents a single turn or event within a conversation.
Items are created independently and then linked to a conversation via
`link_item()`. Listing uses cursor-based pagination (`ListParams` with `limit`,
`order`, and an optional `after` cursor).

Fields: `id`, `response_id` (optional back-reference), `item_type`, `role`,
`content` (JSON), `status`, `created_at`.

### Responses

A `StoredResponse` captures a complete model interaction: input, output,
instructions, tool calls, metadata, model identifier, and raw response payload.
Responses support chaining via `previous_response_id` and can be queried by
`safety_identifier` for content moderation workflows. The `ResponseChain`
struct reconstructs the chronological sequence of related responses.

### Tenant scope

`StoredResponse` and `Conversation` carry an optional `tenant`: the identity
the gateway resolved for the request that stored them (a tenant API key or a
trusted tenant header, as its canonical key). Every backend persists it (a
nullable `tenant` column on `responses`, `conversations` and `conversation_items`, a hash field in
Redis). The rule, `tenant_scope_allows`, is the same for both record types:

- a record stored under a tenant is served only to a caller of that tenant
  (`is_visible_to`); the gateway answers another tenant's id as it answers an
  unknown one;
- a record stored under no tenant (no tenant identity configured, a caller the
  gateway resolved no tenant for, or a row from before the column existed) is
  served to every caller, as before.

The listing and the deletion by `safety_identifier` take the caller's tenant
and apply the same rule in the store's own query. Conversation items carry the
tenant as well: an existing item is referenced (`item_reference`) or added
under its own id only by its tenant. The gateway bounds a tenant id to 200
characters, so the canonical key fits the column.

## ID Generation

| ID Type | Format | Example |
|---------|--------|---------|
| `ConversationId` | `conv_` + 50 random hex chars (25 bytes) | `conv_a1b2c3...` |
| `ConversationItemId` | `{prefix}_` + 50 random hex chars | `msg_d4e5f6...` |
| `ResponseId` | ULID (26 chars, lexicographically sortable, millisecond precision) | `01ARZ3NDEKTSV4RRFFQ69G5FAV` |

**Item type prefixes:**

| `item_type` | Prefix |
|-------------|--------|
| `message` | `msg` |
| `reasoning` | `rs` |
| `mcp_call` | `mcp` |
| `mcp_list_tools` | `mcpl` |
| `function_call` | `fc` |
| (other) | first 3 chars of the type, or `itm` if empty |

## Start-up Contact

With a Postgres or Redis history backend configured, `create_storage` makes one
bounded contact with the store before anything else: a connection and one round
trip (`SELECT 1` / `PING`) within the backend's store-call bound
(`op_timeout_secs`, 5 s by default). A store that refuses the connection is a
named error at once (the backend, the URL with its password masked, the cause);
a store that accepts the connection and never answers is one when the bound
passes (`did not answer within the start-up bound of 5 s`): the client library
and the pool bound their own steps at twice the store-call bound, so the
contact's verdict is the first to fire for either backend. The gateway exits
before it binds its API port: it does not start without a history backend it
cannot reach. Oracle's contact is bounded by its own `pool_timeout_secs`.

## Database Schema

All database backends auto-create their schemas on first connection. The
following default table names are used (configurable via `SchemaConfig`):

| Default Table | Purpose |
|---------------|---------|
| `conversations` | Conversation records with metadata |
| `conversation_items` | Individual items (messages, tool calls, etc.) |
| `conversation_item_links` | Join table linking items to conversations with ordering (`added_at`) |
| `responses` | Stored model responses with chaining and safety identifier indexing |

PostgreSQL additionally creates an index on
`conversation_item_links(conversation_id, added_at)` for efficient
cursor-based listing.

Column names within each table can also be overridden via `SchemaConfig`.
If you rename a column in config, the corresponding database column must
already exist with that name.

### Schema Versioning

The data connector includes a built-in migration system that tracks applied
schema changes in a `_schema_versions` table. Each backend (Oracle, Postgres)
defines its own migration list with backend-specific DDL. Redis has no
structural schema and does not use migrations.

**Safe by default**: `auto_migrate` defaults to `false`. When pending
migrations are detected, startup **fails with the exact SQL statements**
needed so you can review and apply them manually. Set `auto_migrate: true`
to opt in to automatic migration.

Normal SQL history backend startup currently enforces only the **core history**
migrations for conversations, conversation items, and responses. Optional
subsystem schema (for example skills metadata or background-mode queue tables)
must be managed by the subsystem that actually uses those tables rather than
being forced on every history-backed deployment.

On startup (Postgres: in one transaction under an advisory lock, so gateways
starting together on one database take turns and all come up):
1. The `_schema_versions` tracking table is created
2. The store's schema version is compared with the migrations the build knows: a database ahead of the build is refused (see "A database ahead of the build" below). Postgres does this here, before any history table is touched; Oracle after its history tables exist (step 3), as the first step of its migration run
3. Tables are created if they don't exist (`CREATE TABLE` / `CREATE TABLE IF NOT EXISTS`)
4. A fresh database (no responses table before this start) is recorded at the
   latest version: its tables were just created in their final shape, so no
   migration is pending and no `auto_migrate` is needed
5. Pending migrations are checked:
   - If `auto_migrate: true` → migrations are applied automatically
   - If `auto_migrate: false` (default) → startup fails with actionable SQL if migrations are pending
6. Indexes are created last, once the columns they cover exist

Current core history migrations:

| Version | Description |
|---------|-------------|
| 1 | Add `safety_identifier` column to responses |
| 2 | Remove legacy `user_id` column from responses |
| 3 | Drop redundant `output`, `metadata`, `instructions`, `tool_calls` columns from responses |
| 4 | Add the nullable `tenant` column to responses, conversations and conversation items |

#### A database ahead of the build

A database whose `_schema_versions` is above the latest migration a build
knows (a newer build migrated it forward; the gateway was rolled back) is
refused at startup with a message that names both versions: the build does
not know the shape of a newer schema and would otherwise start and fail its
first read of a changed column. Upgrade the gateway, or set `version` to the
database's version in the schema config to run the build against it anyway
(logged once at startup).

#### Controlling migrations

| Config field | Type | Default | Description |
|---|---|---|---|
| `auto_migrate` | `bool` | `false` | Set to `true` to apply migrations automatically on startup (the environment variable `DB_AUTO_MIGRATE=true` sets the default). When `false`, startup fails with the exact SQL if migrations are pending. |
| `version` | `u32` (optional) | `None` | Starting version — migrations up to this number are skipped. Use when your database is already at a known version, or to run a build against a database whose schema version is ahead of it (refused at startup otherwise, see above). |

Example: opt in to automatic migration:

```yaml
oracle:
  schema:
    auto_migrate: true
```

Example: database already at version 2, skip those migrations:

```yaml
oracle:
  schema:
    version: 2
    auto_migrate: true
```

#### Concurrency safety

- **Postgres**: Uses `pg_advisory_lock` to serialize migrations across
  concurrent application instances.
- **Oracle**: DDL statements use PL/SQL exception handling for idempotency.
  Duplicate version records from concurrent instances are detected and skipped
  (ORA-00001).

## Testing

Run the unit tests (Memory and NoOp backends, config validation, ID generation):

```bash
cargo test -p data-connector
```

Integration tests against live PostgreSQL, Redis, or Oracle instances require
the corresponding backend to be running and configured.
