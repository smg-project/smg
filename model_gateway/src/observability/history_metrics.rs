//! The in-memory history's size on `/metrics`.
//!
//! `smg_history_memory_records{kind}` is sampled every [`SAMPLE_EVERY`] from
//! the stores that can count their records without a query (the memory
//! backend); a store that cannot (Postgres, Redis, Oracle) sets nothing.

use std::{sync::Arc, time::Duration};

use metrics::{describe_gauge, gauge};

use crate::app_context::AppContext;

/// Records the in-memory history backend holds, by `kind`.
pub(crate) const RECORDS: &str = "smg_history_memory_records";

/// How often the stores are read.
const SAMPLE_EVERY: Duration = Duration::from_secs(10);

pub(crate) fn describe() {
    describe_gauge!(
        RECORDS,
        "Records the in-memory history backend holds, by kind (responses, conversations, items); sampled every 10 s, memory backend only"
    );
}

/// One sample of the three stores: a store that cannot count sets nothing.
pub(crate) fn sample(context: &AppContext) {
    let counts = [
        ("responses", context.response_storage.stored_record_count()),
        (
            "conversations",
            context.conversation_storage.stored_record_count(),
        ),
        (
            "items",
            context.conversation_item_storage.stored_record_count(),
        ),
    ];
    for (kind, count) in counts {
        if let Some(count) = count {
            gauge!(RECORDS, "kind" => kind).set(count as f64);
        }
    }
}

/// Samples for the life of the process.
#[expect(
    clippy::disallowed_methods,
    reason = "a sampler for the life of the process; nothing waits for it at shutdown"
)]
pub(crate) fn spawn_sampler(context: Arc<AppContext>) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(SAMPLE_EVERY);
        loop {
            tick.tick().await;
            sample(&context);
        }
    });
}
