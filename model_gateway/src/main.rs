use std::process::ExitCode;

use smg::observability::metrics::register_jemalloc_as_global_allocator;

// Jemalloc as the global allocator: glibc malloc retains freed pages badly
// under the gateway's allocation churn. Prefixed symbols only — vendored C
// libraries keep their own malloc.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Jemalloc's run-time options for a long-running server (see
// `SERVER_MALLOC_CONF`): the executable exports them as jemalloc's
// application-provided `malloc_conf`, as the Python extension does for the
// router launched from Python and the Go FFI library for a Go host.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
use smg::observability::metrics::SERVER_MALLOC_CONF;
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[expect(
    unsafe_code,
    reason = "jemalloc reads its options from this exported symbol; a NUL-terminated byte string nothing in Rust dereferences"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] = SERVER_MALLOC_CONF;

fn main() -> Result<ExitCode, Box<dyn std::error::Error>> {
    register_jemalloc_as_global_allocator();
    smg::main_from_args()
}
