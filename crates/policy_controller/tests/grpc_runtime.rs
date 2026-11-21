//! The gRPC plugin runtime's constructor in a process out of descriptors or
//! out of threads. Two cases in a test binary of their own: each lowers a
//! limit of the process, which no other case could run beside, so the two
//! take turns. The thread case is Linux's alone.
#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::sync::{Mutex, PoisonError};

use rustix::process::{getrlimit, setrlimit, Resource, Rlimit};
use smg_policy_controller::plugins::{grpc::Runtime, shared::OpenDecisions};

/// One case at a time lowers a limit of the process.
static LIMITS: Mutex<()> = Mutex::new(());

/// The I/O thread builds its event loop and hands the result back before
/// the constructor returns: a host that cannot build one reads the error
/// where it asked. With no hand-back the constructor returns a runtime that
/// serves nothing, and a plugin loaded over it fails at its connect frame
/// for a reason that names no cause.
#[test]
fn a_runtime_that_cannot_be_built_is_the_constructors_error() {
    let _turn = LIMITS.lock().unwrap_or_else(PoisonError::into_inner);
    let decisions = OpenDecisions::new();
    let before = getrlimit(Resource::Nofile);
    // No descriptor is left to open, and an event loop opens two.
    setrlimit(
        Resource::Nofile,
        Rlimit {
            current: Some(0),
            maximum: before.maximum,
        },
    )
    .expect("the limit is lowered");
    let built = Runtime::new(&decisions);
    setrlimit(Resource::Nofile, before).expect("the limit is put back");
    assert!(
        built.is_err(),
        "the constructor returned a runtime whose I/O thread has no event loop"
    );
    // With descriptors again the same constructor builds.
    Runtime::new(&decisions).expect("a runtime");
}

/// A host that cannot start the I/O thread reads the constructor's error
/// from async code as from any other. The thread's closure owns no event
/// loop, so nothing that blocks is dropped on the caller. A loop built by the
/// constructor and moved into the closure is dropped on the calling thread
/// when the spawn fails, and inside an async context that drop panics.
/// Linux alone: its limit counts threads, where another system's counts
/// processes and the spawn succeeds whatever the limit.
#[cfg(target_os = "linux")]
#[test]
#[expect(
    clippy::print_stdout,
    reason = "a run the limit does not hold in reports it on stdout under --nocapture"
)]
fn a_thread_that_cannot_start_is_the_constructors_error_inside_an_async_caller() {
    let _turn = LIMITS.lock().unwrap_or_else(PoisonError::into_inner);
    let decisions = OpenDecisions::new();
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the caller's loop");
    let before = getrlimit(Resource::Nproc);
    let built = caller.block_on(async {
        // No thread is left to start: the user's limit goes under what
        // already runs.
        setrlimit(
            Resource::Nproc,
            Rlimit {
                current: Some(1),
                maximum: before.maximum,
            },
        )
        .expect("the limit is lowered");
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Runtime::new(&decisions).map(|_| ())
        }));
        setrlimit(Resource::Nproc, before).expect("the limit is put back");
        built
    });
    if matches!(built, Ok(Ok(()))) {
        // The limit does not hold for this user (the superuser): the thread
        // started, and there is nothing to read here.
        println!(
            "a thread that cannot start: this run asserted another lawful world: \
             the thread limit does not hold for this user"
        );
        return;
    }
    assert!(
        matches!(built, Ok(Err(_))),
        "inside an async context the constructor returned no error: it panicked"
    );
}
