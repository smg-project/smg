//! A program that decides in its own process.
//!
//! The engine chooses the worker each request goes to, inside the program
//! that serves the request. This example is such a program, as small as one
//! can be. It keeps its workers in the worker crate's registry, starts the
//! engine once from routing settings, over that registry, and serves each
//! request by the one call the engine has for a request
//! ([`InProcess::place`]): it keeps the placement to the request's end
//! and says when the request went out and how it ended.
//!
//! `program.rs` is the program. This file runs it five times, over workers
//! that serve nothing: a send is a line on the output. The first run is over
//! three single workers, one of which answers every send with a server
//! error, so a request that meets it is asked again past it. The second is
//! over pairs of a prefill worker and a decode worker: the settings' mode
//! says so, a request lends the two lists, and the same one call names both
//! of its workers. The third is over such a pair behind two encoders: a
//! request lends the three lists, one that carries an item states it, and
//! the call names the item's encoder too. The fourth is over a pair whose
//! decode goes out behind its prefill's answer: the program asks the engine
//! for the decode again, behind that answer. The fifth is under a policy
//! that places a routing key by a ring: the program lends its own
//! (`ring.rs`), and a key goes to the worker that ring puts it on.
//!
//! `engine_load.rs` is a stage of the program's own, a scorer added in
//! code. This file shows it once over records made by hand, as a program
//! tests a stage of its own.
//!
//! ```text
//! cargo run -p smg-grand-central --features in-process --example in_process
//! ```
//!
//! [`InProcess::place`]: smg_grand_central::engine::InProcess::place

#![expect(
    clippy::print_stdout,
    reason = "an example says on its output what it did"
)]

use std::{error::Error, sync::Arc};

use engine_load::{decided_by_engine_load, stages};
use http::{HeaderMap, HeaderValue};
use program::{encode_settings, pair_settings, settings, Asked, Plan, Program};
use smg_grand_central::{
    document::{AppliedSpec, DecodePick, PolicySettings, PolicySpec},
    engine::{
        Candidate, Constraints, Identity, PoolFacts, RequestFacts, Role, Scratch, Target, WorkerId,
    },
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus, WorkerType},
};

mod engine_load;
mod program;
mod ring;

/// The model the example's workers serve.
const MODEL: &str = "model-a";

/// The worker that answers every send with a server error.
const FAILING: &str = "http://worker-1:8000";

/// A ready worker of `kind` at `address` that serves [`MODEL`], with no
/// health checker behind it.
fn worker(address: &str, kind: WorkerType) -> Arc<dyn Worker> {
    let health = HealthCheckConfig {
        disable_health_check: true,
        ..HealthCheckConfig::default()
    };
    let worker = <BasicWorkerBuilder>::new(address)
        .model(ModelCard::new(MODEL))
        .worker_type(kind)
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
    let rank = |position: u32, dp_rank: u16, in_flight: u32| {
        let mut row = Candidate::ready(position, WorkerId(u64::from(position)), Role::Single);
        row.engine = 0;
        row.dp_rank = dp_rank;
        row.dp_size = 2;
        row.in_flight = in_flight;
        row
    };
    let mut alone = Candidate::ready(2, WorkerId(2), Role::Single);
    alone.engine = 1;
    alone.in_flight = 2;
    let records = [rank(0, 0, 3), rank(1, 1, 0), alone];
    let mut facts = RequestFacts::default();
    facts.model = MODEL;
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
    // The document a configuration's `policy: round_robin` means, and the
    // kinds of worker its mode routes to: single workers.
    let settings = settings(PolicySettings::RoundRobin);
    let plan = Plan::of(&settings).ok_or("a mode the program serves")?;
    let spec = AppliedSpec::validate(&settings.policy_document()?)?;
    // This program has no load monitor, so it hands the engine no channel.
    let (program, task) = Program::start(plan, spec, None, send);
    #[expect(
        clippy::disallowed_methods,
        reason = "the engine's task runs for the life of the process"
    )]
    let _task = tokio::spawn(task);

    let addresses = ["http://worker-0:8000", FAILING, "http://worker-2:8000"];
    for address in addresses {
        let single = worker(address, WorkerType::Regular);
        assert!(program.join(single), "a free address joins");
    }
    // The yield hands the engine's task its first poll: there it reads the
    // registry whole and tells the document's stages each worker.
    tokio::task::yield_now().await;
    let asked = Asked {
        model: MODEL,
        headers: None,
        items: &[],
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
    pairs(&asked).await?;
    encodes(&asked).await?;
    in_turn(&asked).await?;
    by_ring().await?;
    own_stage()
}

/// The program over pairs: the same settings under the mode of a gateway
/// that routes to a prefill worker and a decode worker, two of each. A
/// request lends the two lists, and its one call names a prefill and a
/// decode: both are sent, the prefill first.
async fn pairs(asked: &Asked<'_>) -> Result<(), Box<dyn Error>> {
    let settings = pair_settings(PolicySettings::RoundRobin);
    let plan = Plan::of(&settings).ok_or("a mode the program serves")?;
    let spec = AppliedSpec::validate(&settings.policy_document()?)?;
    let (program, task) = Program::start(plan, spec, None, send);
    #[expect(
        clippy::disallowed_methods,
        reason = "the engine's task runs for the life of the process"
    )]
    let _task = tokio::spawn(task);
    let prefills = ["http://prefill-0:8000", "http://prefill-1:8000"];
    let decodes = ["http://decode-0:8000", "http://decode-1:8000"];
    for address in prefills {
        let prefill = worker(address, WorkerType::Prefill);
        assert!(program.join(prefill), "a free address joins");
    }
    for address in decodes {
        let decode = worker(address, WorkerType::Decode);
        assert!(program.join(decode), "a free address joins");
    }
    tokio::task::yield_now().await;
    for request in 1..=2 {
        println!("a pair's request {request}:");
        let answer = program.serve(asked);
        println!("  the client reads {}", answer.status);
    }
    Ok(())
}

/// The program over a pair behind encoders: the same settings under the
/// mode of a gateway that sends a request's items to encode workers. Each
/// request lends the three lists. One that carries an item states it, and
/// its one call names the item's encoder beside the pair: this program
/// sends the item first. The same request with no item is placed as its
/// pair.
async fn encodes(asked: &Asked<'_>) -> Result<(), Box<dyn Error>> {
    let settings = encode_settings(PolicySettings::RoundRobin);
    let plan = Plan::of(&settings).ok_or("a mode the program serves")?;
    let spec = AppliedSpec::validate(&settings.policy_document()?)?;
    let (program, task) = Program::start(plan, spec, None, send);
    #[expect(
        clippy::disallowed_methods,
        reason = "the engine's task runs for the life of the process"
    )]
    let _task = tokio::spawn(task);
    let fleet = [
        ("http://encode-0:8000", WorkerType::Encode),
        ("http://encode-1:8000", WorkerType::Encode),
        ("http://prefill-0:8000", WorkerType::Prefill),
        ("http://decode-0:8000", WorkerType::Decode),
    ];
    for (address, kind) in fleet {
        assert!(program.join(worker(address, kind)), "a free address joins");
    }
    tokio::task::yield_now().await;
    // An item's key is its caller's hash of the item's content.
    let items = [Identity::plain("the hash of an item's content")];
    let carrying = Asked {
        items: &items,
        ..*asked
    };
    println!("an encode request of one item:");
    let answer = program.serve(&carrying);
    println!("  the client reads {}", answer.status);
    println!("the same request with no item:");
    let answer = program.serve(asked);
    println!("  the client reads {}", answer.status);
    Ok(())
}

/// The program over a pair with two decodes, with a route that sends the
/// decode behind its prefill's answer. The document picks the decode behind
/// the prefill, which is the document's own word and no setting's. A
/// request's one call books its prefill alone. Behind the prefill's answer
/// the program states the request again and asks the placement for its
/// decode: the decode that is sent is the one the engine names then.
async fn in_turn(asked: &Asked<'_>) -> Result<(), Box<dyn Error>> {
    let settings = pair_settings(PolicySettings::RoundRobin);
    let plan = Plan::of(&settings).ok_or("a mode the program serves")?;
    let mut document = settings.policy_document()?;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let spec = AppliedSpec::validate(&document)?;
    let (program, task) = Program::start(plan, spec, None, send);
    let program = program.in_turn();
    #[expect(
        clippy::disallowed_methods,
        reason = "the engine's task runs for the life of the process"
    )]
    let _task = tokio::spawn(task);
    let fleet = [
        ("http://prefill-0:8000", WorkerType::Prefill),
        ("http://decode-0:8000", WorkerType::Decode),
        ("http://decode-1:8000", WorkerType::Decode),
    ];
    for (address, kind) in fleet {
        assert!(program.join(worker(address, kind)), "a free address joins");
    }
    tokio::task::yield_now().await;
    for request in 1..=2 {
        println!("a pair's request {request}, sent in turn:");
        let answer = program.serve(asked);
        println!("  the client reads {}", answer.status);
    }
    Ok(())
}

/// The program under `consistent_hashing` over three single workers, each
/// request under a routing key its header names. The program lends its own
/// ring with the request, so a key's requests go to the worker that ring
/// puts the key on: the first of the ring's walk from where the key stands.
async fn by_ring() -> Result<(), Box<dyn Error>> {
    let settings = settings(PolicySettings::ConsistentHashing);
    let plan = Plan::of(&settings).ok_or("a mode the program serves")?;
    let spec = AppliedSpec::validate(&settings.policy_document()?)?;
    let (program, task) = Program::start(plan, spec, None, send);
    #[expect(
        clippy::disallowed_methods,
        reason = "the engine's task runs for the life of the process"
    )]
    let _task = tokio::spawn(task);
    let addresses = [
        "http://worker-0:8000",
        "http://worker-2:8000",
        "http://worker-3:8000",
    ];
    for address in addresses {
        let single = worker(address, WorkerType::Regular);
        assert!(program.join(single), "a free address joins");
    }
    tokio::task::yield_now().await;
    for key in ["session-a", "session-b"] {
        let lists = program.published(MODEL).ok_or("the model's generation")?;
        let walk: Vec<&str> = lists.ring().walked(key).collect();
        println!("the routing key {key}, whose walk of the program's ring is {walk:?}:");
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", HeaderValue::from_static(key));
        let asked = Asked {
            model: MODEL,
            headers: Some(&headers),
            items: &[],
        };
        for _ in 0..2 {
            let answer = program.serve(&asked);
            println!("  the client reads {}", answer.status);
        }
    }
    Ok(())
}
