//! `smg_policy_decisions_total` on the scrape: one count per decision of the
//! hosted policy engine, by the model, the profile that decided, the outcome
//! and the mode, with its help text.

use std::sync::Arc;

use smg::{
    config::PolicyConfig,
    policies::engine_host::{Decider, HostRequest},
    worker::{test_support::worker_of, Worker, WorkerType},
};

#[test]
fn every_decision_of_the_hosted_engine_counts_once_by_profile_outcome_and_mode() {
    let handle = crate::metrics::recorder();
    let round_robin = PolicyConfig::RoundRobin;
    let decider = Decider::from_configs(&round_robin, &round_robin, &round_robin)
        .expect("the engine serves round robin");
    // Ready workers with no health checker behind them: the pool a decision
    // over a live engine sees.
    let workers: Vec<Arc<dyn Worker>> = ["http://decisions-a:8000", "http://decisions-b:8000"]
        .iter()
        .map(|url| worker_of(url, WorkerType::Regular))
        .collect();
    let request = HostRequest {
        model: "decisions-model",
        ..HostRequest::default()
    };
    decider.decide(&request, &workers, &[]);
    decider.decide(&request, &workers, &[]);
    // An empty pool: the filters leave nothing, a no-endpoints shed.
    decider.decide(&request, &[], &[]);

    let rendered = handle.render();
    for expected in [
        "smg_policy_decisions_total{model=\"decisions-model\",profile=\"default\",outcome=\"pick\",mode=\"embedded\"} 2",
        "smg_policy_decisions_total{model=\"decisions-model\",profile=\"default\",outcome=\"no_endpoints\",mode=\"embedded\"} 1",
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing {expected} in:\n{rendered}"
        );
    }
    assert!(
        rendered.contains("# HELP smg_policy_decisions_total "),
        "the family carries its help text:\n{rendered}"
    );
}
