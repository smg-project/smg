//! The program of the `in_process` example, driven: a program that decides
//! in its own process serves its requests by the one call the engine has
//! for a request ([`PolicyEngine::place`]).
//!
//! The program is the example's own module (`examples/in_process/program.rs`),
//! taken in here as it stands. A test target is a program outside the
//! library: it reaches the crate's public names and no other. Each case
//! starts the program over the worker crate's test workers, with that
//! crate's test feed as its load monitor and a function of its own as its
//! sends, serves requests through it as a route would, and reads what the
//! client and the workers saw. No case names a stage of the document, looks
//! inside a decision or keeps a lease.
//!
//! What the program asks of the engine stands in one place, the methods of
//! its `Engine`. Each method beside `place` is a call the host contract
//! still owes a program, and leaves that list with the change that pays it.
//!
//! One module a plan holds the plan's cases. A single request is served so
//! far (`caller/single.rs`). The example's own stage
//! (`examples/in_process/engine_load.rs`) is taken in beside the program,
//! and its cases decide over records made by hand (`caller/stage.rs`).
//!
//! [`PolicyEngine::place`]: smg_policy_controller::engine::PolicyEngine::place

#![expect(clippy::expect_used, reason = "a case that stops on a broken setup")]

use std::{future::Future, sync::Arc};

use program::{settings, Program};
use smg_policy_controller::document::{AppliedSpec, PolicySettings};
use smg_worker::{identity::basic::Worker, load::state::TestFeed, test_support::ready_worker};

#[path = "../examples/in_process/engine_load.rs"]
mod engine_load;
#[path = "../examples/in_process/program.rs"]
mod program;
#[path = "caller/single.rs"]
mod single;
#[path = "caller/stage.rs"]
mod stage;

/// The model every case serves.
const MODEL: &str = "model-a";

/// The address of the worker at `at`.
fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

/// A ready worker that serves [`MODEL`], at [`url`]`(at)`.
fn worker(at: usize) -> Arc<dyn Worker> {
    ready_worker(&url(at), MODEL)
}

/// `workers` such workers, in the order of their addresses.
fn workers(workers: usize) -> Vec<Arc<dyn Worker>> {
    (0..workers).map(worker).collect()
}

/// A send that every worker answers a 200.
fn served(_: &dyn Worker) -> u16 {
    200
}

/// The program started from `spec` over `fleet`, with `sends` as its sends:
/// the program, its load monitor's stand-in, whose channel the engine
/// holds, and the engine's task to spawn.
fn started(
    spec: AppliedSpec,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    let mut channel = None;
    let loads = TestFeed::attached_to(|handed| channel = handed);
    let (program, task) = Program::start(spec, channel, sends);
    for worker in fleet {
        assert!(program.join(Arc::clone(worker)), "a free address joins");
    }
    (program, loads, task)
}

/// [`started`] under the document a gateway that sets `policy` means.
fn under(
    policy: PolicySettings,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    let document = settings(policy).policy_document();
    let document = document.expect("the settings have a document");
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    started(spec, fleet, sends)
}
