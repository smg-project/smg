//! The standalone policy controller (design 2: the `remote` mode of the
//! policy engine): the crate's command line over the stock build's stages.
//! A host with stages of its own writes this `main` over its own catalog.

use std::process::ExitCode;

use smg_policy_controller::{controller, Catalog};

/// The process's allocator, the one the gateway's binary registers, on the
/// targets the gateway registers it. A burst of registrations builds the
/// store's world again at every write: the system allocator keeps resident
/// what those builds free, hundreds of megabytes where the heap holds a few.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The allocator's settings, a copy of the gateway's for a long-running
/// server: a background thread hands freed pages back to the system ten
/// seconds after their last use, the threads share sixteen arenas at most,
/// and an allocation up to 32 MB takes the room the last one freed. The
/// allocator reads them from this symbol as it starts, and the
/// `_RJEM_MALLOC_CONF` environment variable after it, entry by entry.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[expect(
    unsafe_code,
    reason = "the allocator reads its settings from this exported symbol; a NUL-terminated byte string nothing in Rust dereferences"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

fn main() -> ExitCode {
    controller::run(Catalog::builtin())
}
