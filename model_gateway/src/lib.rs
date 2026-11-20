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

#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[doc(hidden)]
pub use tikv_jemallocator::Jemalloc as __StockAllocator;

/// Installs the stock executable's global allocator and options for a long-running gateway.
///
/// Write this once at the binary's root. The allocator uses prefixed symbols
/// so native libraries can keep their own allocator. On MSVC and musl the
/// macro expands to nothing. See [`main_from_args`] for registration.
///
/// The expansion exports the options under an unsafe attribute, so a crate
/// that forbids unsafe code cannot take the macro.
#[macro_export]
macro_rules! stock_allocator {
    () => {
        #[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
        const _: () = {
            #[global_allocator]
            static GLOBAL_ALLOCATOR: $crate::__StockAllocator = $crate::__StockAllocator;

            #[expect(
                unsafe_code,
                reason = "the allocator reads this NUL-terminated options string by its exported name"
            )]
            #[unsafe(export_name = "_rjem_malloc_conf")]
            static MALLOC_CONF: &[u8; 100] = $crate::observability::metrics::SERVER_MALLOC_CONF;
        };
    };
}

/// Runs the gateway from the process's command-line arguments.
///
/// The argument parser exits the process after printing help or rejecting
/// arguments. A binary that wants the stock allocator declares it at its
/// root and registers it before this call:
///
/// ```no_run
/// smg::stock_allocator!();
///
/// fn main() -> Result<std::process::ExitCode, Box<dyn std::error::Error>> {
///     smg::observability::metrics::register_jemalloc_as_global_allocator();
///     smg::main_from_args()
/// }
/// ```
///
/// The macro alone gives the allocator and its options. The registration
/// turns the allocator's statistics on, and where it was not called this
/// call says so once on stderr. A binary that declares no allocator runs
/// on Rust's default, the system allocator, and the stock build does not.
///
/// The call builds its own runtime and blocks its thread until the server
/// stops, so it must not be made from async code. A call that served has
/// closed the process's logging and tracing when it returns. A version
/// flag and `--print-policy-spec` answer and return before either starts.
/// A host whose process goes on after the server calls
/// [`server::startup_with`] instead.
pub fn main_from_args() -> Result<std::process::ExitCode, Box<dyn std::error::Error>> {
    main_with(server::Extensions::default())
}

/// [`main_from_args`] with the host's extensions.
pub fn main_with(
    extensions: server::Extensions,
) -> Result<std::process::ExitCode, Box<dyn std::error::Error>> {
    #[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
    if !observability::metrics::jemalloc_is_global() {
        use std::{io::Write, sync::Once};

        static WARN: Once = Once::new();
        WARN.call_once(|| {
            let _ = writeln!(
                std::io::stderr(),
                "warning: jemalloc was not registered as the global allocator, so allocator statistics are off. A binary on the stock allocator declares smg::stock_allocator!() and calls smg::observability::metrics::register_jemalloc_as_global_allocator() before this call. See smg::main_from_args."
            );
        });
    }
    cli::main_from_args(extensions)
}

/// The lib's tests run under the allocator the executable runs under
/// (`main.rs`), so a test can read what a path allocates on its own thread
/// through the allocator's counters (`routers::grpc::testing`).
#[cfg(all(test, not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static TEST_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
