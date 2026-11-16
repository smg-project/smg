//! The gRPC plugin runtime's constructor in a process out of descriptors.
//! One case in a test binary of its own: it lowers the process's descriptor
//! limit, which no other case could run beside.
#![cfg(any(target_os = "linux", target_os = "macos"))]

use rustix::process::{getrlimit, setrlimit, Resource, Rlimit};
use smg_policy_controller::{grpc::Runtime, plugins::OpenDecisions};

/// The I/O thread's event loop is built by the constructor, on the calling
/// thread: a host that cannot build it reads the error where it asked.
/// Built on the I/O thread, the constructor returns a runtime that serves
/// nothing, and a plugin loaded over it fails at its connect frame for a
/// reason that names no cause.
#[test]
fn a_runtime_that_cannot_be_built_is_the_constructors_error() {
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
