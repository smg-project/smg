//! The standalone policy controller (design 2: the `remote` mode of the
//! policy engine): the crate's command line over the stock build's stages.
//! A host with stages of its own writes this `main` over its own catalog.

use std::process::ExitCode;

use smg_policy_controller::{service::controller, stages::StageRegistry};

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
/// and an allocation up to 32 MiB takes the room the last one freed. A case
/// of this crate holds the copy to the gateway's constant.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
const SETTINGS: &[u8; 100] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

/// The settings without their first entry, for an Apple target: the
/// allocator compiles no background thread there and does not finish its
/// start-up where one is asked. Freed pages go back as the threads allocate.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
const SETTINGS_WITHOUT_THREAD: &[u8; 77] = match (SETTINGS.first_chunk(), SETTINGS.last_chunk()) {
    (Some(b"background_thread:true,"), Some(rest)) => rest,
    _ => panic!("the settings do not start with the background thread's entry"),
};

// The two chunks are the whole of the settings.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
const _: () =
    assert!(SETTINGS.len() == b"background_thread:true,".len() + SETTINGS_WITHOUT_THREAD.len());

/// The symbol the allocator reads its settings from as it starts. It reads
/// the `_RJEM_MALLOC_CONF` environment variable after it, entry by entry.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
#[expect(
    unsafe_code,
    reason = "the allocator reads its settings from this exported symbol; a NUL-terminated byte string nothing in Rust dereferences"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] = SETTINGS;

/// The same symbol on an Apple target.
#[cfg(target_vendor = "apple")]
#[expect(
    unsafe_code,
    reason = "the allocator reads its settings from this exported symbol; a NUL-terminated byte string nothing in Rust dereferences"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 77] = SETTINGS_WITHOUT_THREAD;

fn main() -> ExitCode {
    controller::run(StageRegistry::builtin())
}
