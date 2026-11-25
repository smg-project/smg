//! The program of the `in_process` example, driven: a program that decides
//! in its own process serves its requests by the one call the engine has
//! for a request ([`InProcess::place`]).
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
//! A pair's decode asked again is asked of the placement `place` answered,
//! where that placement says its decode still stands chosen.
//!
//! One module a plan holds the plan's cases: a single request
//! (`caller/single.rs`), a pair of a prefill worker and a decode worker
//! (`caller/pair.rs`), and an encode request, whose items go to encoders
//! ahead of its pair (`caller/encode.rs`). The example's own stage
//! (`examples/in_process/engine_load.rs`) is taken in beside the program,
//! and its cases decide over records made by hand (`caller/stage.rs`). The
//! program's own ring (`examples/in_process/ring.rs`) is taken in too: its
//! cases read the ring a generation holds and expect the sends its walk
//! says (`caller/lent_ring.rs`). The wire a retry stays on has its cases
//! beside them (`caller/demand.rs`).
//!
//! [`InProcess::place`]: smg_grand_central::engine::InProcess::place

#![expect(clippy::expect_used, reason = "a case that stops on a broken setup")]

use std::{future::Future, sync::Arc};

use program::{encode_settings, pair_settings, settings, Plan, Program};
use smg_grand_central::document::{AppliedSpec, PolicySettings, RoutingSettings};
use smg_worker::{identity::basic::Worker, load::state::TestFeed, test_support::ready_worker};

#[path = "caller/demand.rs"]
mod demand;
#[path = "caller/encode.rs"]
mod encode;
#[path = "../examples/in_process/engine_load.rs"]
mod engine_load;
#[path = "caller/lent_ring.rs"]
mod lent_ring;
#[path = "caller/pair.rs"]
mod pair;
#[path = "../examples/in_process/program.rs"]
mod program;
#[path = "../examples/in_process/ring.rs"]
mod ring;
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

/// The program started from `spec` over `fleet`, routing to the kinds of
/// worker `plan` says, with `sends` as its sends: the program, its load
/// monitor's stand-in, whose channel the engine holds, and the engine's
/// task to spawn.
fn started_as(
    plan: Plan,
    spec: AppliedSpec,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    let mut channel = None;
    let loads = TestFeed::attached_to(|handed| channel = handed);
    let (program, task) = Program::start(plan, spec, channel, sends);
    for worker in fleet {
        assert!(program.join(Arc::clone(worker)), "a free address joins");
    }
    (program, loads, task)
}

/// [`started_as`] over single workers.
fn started(
    spec: AppliedSpec,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    started_as(Plan::Single, spec, fleet, sends)
}

/// [`started_as`] under `settings`: the document they mean, and the kinds
/// of worker their mode says.
fn under_settings(
    settings: &RoutingSettings,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    let plan = Plan::of(settings).expect("a mode the program serves");
    let document = settings.policy_document();
    let document = document.expect("the settings have a document");
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    started_as(plan, spec, fleet, sends)
}

/// The program under the document a gateway that sets `policy` means, over
/// single workers.
fn under(
    policy: PolicySettings,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    under_settings(&settings(policy), fleet, sends)
}

/// The program under the document a gateway means that routes to pairs,
/// both sides under `policy`.
fn under_pairs(
    policy: PolicySettings,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    under_settings(&pair_settings(policy), fleet, sends)
}

/// The program under the document a gateway means that routes to pairs
/// behind encoders, the pair's two sides under `policy`.
fn under_encode(
    policy: PolicySettings,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    under_settings(&encode_settings(policy), fleet, sends)
}
