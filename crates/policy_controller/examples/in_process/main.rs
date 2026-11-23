//! A program that decides in its own process.
//!
//! The engine chooses the worker each request goes to, inside the program
//! that serves the request. This example is such a program, as small as one
//! can be. It keeps its workers in the worker crate's registry, starts the
//! engine once from routing settings, over that registry, and serves each
//! request by the one call the engine has for a request
//! ([`PolicyEngine::place`]): it keeps the placement to the request's end
//! and says when the request went out and how it ended.
//!
//! `program.rs` is the program. This file runs it over three workers that
//! serve nothing: a send is a line on the output, and one worker answers
//! every send with a server error, so a request that meets it is asked
//! again past it.
//!
//! `engine_load.rs` is a stage of the program's own, a scorer added in
//! code. This file shows it once over records made by hand, as a program
//! tests a stage of its own.
//!
//! ```text
//! cargo run -p smg-policy-controller --features in-process --example in_process
//! ```
//!
//! [`PolicyEngine::place`]: smg_policy_controller::engine::PolicyEngine::place

#![expect(
    clippy::print_stdout,
    reason = "an example says on its output what it did"
)]

use std::{error::Error, sync::Arc};

use engine_load::{decided_by_engine_load, stages};
use program::{settings, Asked, Program};
use smg_policy_controller::{
    document::{AppliedSpec, PolicySettings, PolicySpec},
    engine::{Candidate, Constraints, PoolFacts, RequestFacts, Role, Scratch, Target, WorkerId},
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus},
};

mod engine_load;
mod program;

/// The model the example's workers serve.
const MODEL: &str = "model-a";

/// The worker that answers every send with a server error.
const FAILING: &str = "http://worker-1:8000";

/// A ready worker at `address` that serves [`MODEL`], with no health
/// checker behind it.
fn worker(address: &str) -> Arc<dyn Worker> {
    let health = HealthCheckConfig {
        disable_health_check: true,
        ..HealthCheckConfig::default()
    };
    let worker = <BasicWorkerBuilder>::new(address)
        .model(ModelCard::new(MODEL))
        .health_config(health)
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// The example's network: a send is a line on the output.
fn send(worker: &dyn Worker) -> u16 {
    let status = if worker.url() == FAILING { 502 } else { 200 };
    println!("  sent to {}, answered {status}", worker.url());
    status
}

/// The program's own scorer, asked once over records made by hand: an
/// engine of two ranks that holds three requests on one of them, and a
/// worker that stands alone and holds two. The request goes to the worker
/// that stands alone, past the idle rank of the busier engine.
fn own_stage() -> Result<(), Box<dyn Error>> {
    let mut document = PolicySpec::preset("round_robin")?;
    decided_by_engine_load(&mut document);
    let engine = document.validate_in(&stages()?, &[])?;
    // A record made ready is an engine of its own. The two ranks state the
    // engine they share, and the worker behind them the next engine of the
    // list: an engine states its place among the engines of its list.
    let rank = |position: u32, dp_rank: u16, in_flight: u32| Candidate {
        engine: 0,
        dp_rank,
        dp_size: 2,
        in_flight,
        ..Candidate::ready(position, WorkerId(u64::from(position)), Role::Single)
    };
    let alone = Candidate {
        engine: 1,
        in_flight: 2,
        ..Candidate::ready(2, WorkerId(2), Role::Single)
    };
    let records = [rank(0, 0, 3), rank(1, 1, 0), alone];
    let facts = RequestFacts {
        model: MODEL,
        ..RequestFacts::default()
    };
    let (pool, cx) = (PoolFacts::default(), Constraints::default());
    let decision = engine.decide(&facts, &pool, &records, &cx, &mut Scratch::new());
    let first = decision.pick().and_then(|pick| pick.targets.first());
    assert_eq!(first, Some(&Target::of(&alone)), "the lighter engine");
    println!("the program's own scorer, over records made by hand:");
    println!("  an engine holds 3 requests over two ranks and a worker alone holds 2:");
    println!(
        "  the request goes to the worker alone, at place {}",
        alone.position
    );
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    // The document a configuration's `policy: round_robin` means.
    let document = settings(PolicySettings::RoundRobin).policy_document()?;
    let spec = AppliedSpec::validate(&document)?;
    // This program has no load monitor, so it hands the engine no channel.
    let (program, task) = Program::start(spec, None, send);
    #[expect(
        clippy::disallowed_methods,
        reason = "the engine's task runs for the life of the process"
    )]
    let _task = tokio::spawn(task);

    let addresses = ["http://worker-0:8000", FAILING, "http://worker-2:8000"];
    for address in addresses {
        assert!(program.join(worker(address)), "a free address joins");
    }
    // The yield hands the engine's task its first poll: there it reads the
    // registry whole and tells the document's stages each worker.
    tokio::task::yield_now().await;
    let asked = Asked {
        model: MODEL,
        headers: None,
    };
    for request in 1..=4 {
        println!("request {request}:");
        let answer = program.serve(&asked);
        println!("  the client reads {}", answer.status);
    }

    for address in addresses {
        assert!(program.leave(address), "a worker that joined leaves");
    }
    // The registry's word of each leave woke the task: at this yield it
    // tells the document's stages each one.
    tokio::task::yield_now().await;
    let answer = program.serve(&asked);
    println!(
        "with no worker left the client reads {} after {} sends: {:?}",
        answer.status,
        answer.sent.len(),
        answer.unplaced
    );
    own_stage()
}
