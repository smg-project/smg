//! The model gateway: the library behind the `smg` binary (`main.rs`), its
//! Python and Go bindings, and its integration tests.
//!
//! The crate root lists concerns, one module each. Where things live:
//!
//! - [`app_context`]: the composition root (`AppContext` and its builder), what every handler reads.
//! - [`config`]: the router configuration: its types, builder, bounds and validation.
//! - [`endpoints`]: handlers that do not route inference: tokenize, parse, conversations, stored responses, the model inventory.
//! - [`health`]: liveness and readiness, the readiness snapshot and the optional probe listener.
//! - [`mesh`]: gateway-side glue for the router mesh: the adapters and their wiring, and the discovery of its router peers.
//! - [`middleware`]: the serving routes' tower layers: authentication, limits, logging, the scheduler, tenant resolution.
//! - [`observability`]: metrics, logging, tracing, heap profiles, the in-flight tracker.
//! - [`policies`]: worker selection policies and their registry.
//! - [`rate_limit`]: per-tenant token and request rate limiting.
//! - [`routers`]: the request paths (HTTP, gRPC, external providers) and what they share.
//! - [`server`]: start-up and its phases, the route table, the listeners and their limits, graceful shutdown, the posture log, the private-key-file check, the process TLS crypto provider and the HTTP client constructors that need it.
//! - [`service_discovery`]: worker discovery providers and the reconciler.
//! - [`tenant`]: tenant identity, keys and per-request metadata.
//! - [`version`]: the build identity.
//! - [`wasm`]: WebAssembly module support and its routes.
//! - [`worker`]: the worker domain: identity, registry, health, load, resilience, monitoring, and the RL control plane's read-only view of the registry.
//! - [`workflow`]: the workflow engine and its steps: worker registration and lifecycle, tokenizer, MCP and WASM registration.

pub mod app_context;
pub mod config;
pub mod endpoints;
pub(crate) mod fs_watch;
pub mod health;
pub mod mesh;
pub mod middleware;
pub mod observability;
pub mod policies;
pub mod rate_limit;
pub mod routers;
pub mod secrets;
pub mod server;
pub mod service_discovery;
pub mod tenant;
pub mod version;
pub mod wasm;
pub mod worker;
pub mod workflow;
