//! The model gateway: the library behind the `smg` binary (`main.rs`), its
//! Python and Go bindings, and its integration tests.
//!
//! The crate root lists concerns, one module each. Where things live:
//!
//! - [`app_context`]: the composition root (`AppContext` and its builder), what every handler reads.
//! - [`config`]: the router configuration: its types, builder, bounds and validation.
//! - [`endpoints`]: handlers that do not route inference: tokenize, parse, conversations, stored responses, the model inventory.
//! - [`mesh`]: gateway-side glue for the router mesh: the adapters and their wiring, and the discovery of its router peers.
//! - [`middleware`]: the serving routes' tower layers: authentication, limits, logging, the scheduler, tenant resolution.
//! - [`observability`]: metrics, logging, tracing, heap profiles, the in-flight tracker.
//! - [`policies`]: worker selection policies and their registry, and the KV index the cache-aware policy reads.
//! - [`rate_limit`]: per-tenant token and request rate limiting.
//! - [`rl_adapter`]: the RL control plane's read-only view of the worker registry.
//! - [`routers`]: the request paths (HTTP, gRPC, external providers) and what they share.
//! - [`server`]: start-up and its phases, the route table, the listeners and their limits, graceful shutdown, the posture log, the private-key-file check, the process TLS crypto provider and the HTTP client constructors that need it; in [`server::health`](mod@server::health), liveness and readiness, the readiness snapshot and the optional probe listener.
//! - [`tenant`]: tenant identity, keys and per-request metadata.
//! - [`version`]: the build identity.
//! - [`wasm`]: WebAssembly module support and its routes.
//! - [`workflow`]: the workflow engine and its steps: worker registration and lifecycle, tokenizer, MCP and WASM registration.
//!
//! The worker domain (identity, registry, health, load, monitoring) is the
//! `smg_worker` crate, and worker discovery (its providers and the
//! reconciler) is the `smg_service_discovery` crate: the gateway names their
//! types by the crates' paths.

pub mod admin;
pub mod app_context;
mod cli;
pub mod config;
mod discovery_host;
pub mod endpoints;
pub mod mesh;
pub mod middleware;
pub mod observability;
pub mod policies;
pub(crate) mod policy_controller;
pub mod rate_limit;
pub mod rl_adapter;
pub mod routers;
pub mod secrets;
pub mod server;
pub mod tenant;
pub mod version;
pub mod wasm;
pub mod workflow;

/// Runs the gateway from the process's command-line arguments.
///
/// The argument parser exits the process after printing help or rejecting
/// arguments. The executable owns allocator selection and registration.
pub fn main_from_args() -> Result<std::process::ExitCode, Box<dyn std::error::Error>> {
    cli::main_from_args()
}

/// The lib's tests run under the allocator the executable runs under
/// (`main.rs`), so a test can read what a path allocates on its own thread
/// through the allocator's counters (`routers::grpc::testing`).
#[cfg(all(test, not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static TEST_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
