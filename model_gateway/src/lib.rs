//! The model gateway: the library behind the `smg` binary (`main.rs`), its
//! Python and Go bindings, and its integration tests.
//!
//! The crate root lists concerns, one module each. Where things live:
//!
//! - [`app_context`]: the composition root (`AppContext` and its builder), what every handler reads.
//! - [`clients`]: the HTTP client factory, the process TLS provider installed first.
//! - [`config`]: the router configuration: its types, builder, bounds and validation.
//! - [`endpoints`]: handlers that do not route inference: tokenize, parse, conversations, stored responses, the model inventory.
//! - [`health`]: liveness and readiness, the readiness snapshot and the optional probe listener.
//! - [`mesh`]: gateway-side glue for the router mesh: the adapters and their wiring.
//! - [`mesh_discovery`]: discovery of the mesh's router peers.
//! - [`middleware`]: the serving routes' tower layers: authentication, limits, logging, the scheduler, tenant resolution.
//! - [`observability`]: metrics, logging, tracing, heap profiles, the in-flight tracker.
//! - [`policies`]: worker selection policies and their registry.
//! - [`rate_limit`]: per-tenant token and request rate limiting.
//! - [`rl_adapter`]: the RL control plane's read-only view of the worker registry.
//! - [`routers`]: the request paths (HTTP, gRPC, external providers) and what they share.
//! - [`server`]: start-up and its phases, the route table, the listeners and their limits, graceful shutdown, the posture log.
//! - [`service_discovery`]: worker discovery providers and the reconciler.
//! - [`tenant`]: tenant identity, keys and per-request metadata.
//! - [`tls`]: the process TLS crypto provider and the private-key-file check.
//! - [`version`]: the build identity.
//! - [`wasm`]: WebAssembly module support and its routes.
//! - [`worker`]: the worker domain: identity, registry, health, load, resilience, monitoring.
//! - [`workflow`]: the workflow engine and its steps: worker registration and lifecycle, tokenizer, MCP and WASM registration.

pub mod app_context;
pub mod clients;
pub mod config;
pub mod endpoints;
pub mod health;
pub mod mesh;
pub mod mesh_discovery;
pub mod middleware;
pub mod observability;
pub mod policies;
pub mod rate_limit;
pub mod rl_adapter;
pub mod routers;
pub mod server;
pub mod service_discovery;
pub mod tenant;
pub mod tls;
pub mod version;
pub mod wasm;
pub mod worker;
pub mod workflow;
