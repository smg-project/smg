//! Building an application context applies the liveness budgets its
//! configuration carries, for the test harness as for the server: a context
//! built with non-default stall, wedge, stale and warm-up budgets runs under
//! them. One test in its own binary, as the budgets are process-wide and the
//! first context built sets them.

use std::time::Duration;

use smg::config::RouterConfig;
use smg_test_harness as common;

#[tokio::test]
async fn a_built_context_applies_its_configurations_liveness_budgets() {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .worker_stall_secs(5)
        .worker_wedge_secs(7)
        .worker_stale_secs(41)
        .build_unchecked();
    config.worker_warmup_secs = 90;
    config.worker_warmup_blocks = 4096;

    let _context = common::create_test_context_with_mock_tokenizer(config, "budget-test").await;

    let budgets = smg::worker::liveness_budgets();
    assert_eq!(
        (budgets.stall, budgets.wedge, budgets.stale),
        (
            Duration::from_secs(5),
            Duration::from_secs(7),
            Duration::from_secs(41)
        ),
        "{budgets:?}"
    );
    assert_eq!(
        (budgets.warmup_secs, budgets.warmup_blocks),
        (Duration::from_secs(90), 4096),
        "{budgets:?}"
    );
}
