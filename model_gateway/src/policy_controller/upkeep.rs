//! The upkeep of the policy engine's door: the timer that runs its clock and
//! shows its counts.
//!
//! The door ends nothing between two decisions by itself. A lease whose
//! dispatch the gateway never named stays on file, counted on its worker,
//! until a later decision at its file ends it. A pin of record past its end
//! stays for the life of the store. So do the session pins of a key that
//! stayed away past its absence. One pass of the door's clock ends all
//! three, and the timer makes that pass.
//!
//! The door also counts what no request's series says: the shadow decisions
//! beside its live ones, and the decisions that panicked. The timer sets a
//! series to each of those counts at every turn.

use std::{
    array,
    sync::Arc,
    time::{Duration, Instant},
};

use metrics::Counter;
use smg_policy_controller::engine::{ShadowKind, SHADOW_ROWS};
use tokio::{
    task::JoinHandle,
    time::{interval, MissedTickBehavior},
};

use super::engine::PolicyEngine;
use crate::observability::metrics::Metrics;

/// The time between two turns. A lease has seconds to name its dispatch,
/// and the store walks its pins only where a pass of theirs is due, so a
/// second is ample.
pub(crate) const EVERY: Duration = Duration::from_secs(1);

/// The `side` label of a single decision's shadow: the policy document's
/// own word for the one side of a `single` plan.
const SINGLE: &str = "single";

/// A binding's row of handles, one a kind.
type Row = [Counter; ShadowKind::ALL.len()];

/// The door's counts as series: handles the timer keeps and sets from the
/// door's own counts.
///
/// The door counts a shadow under its binding's position in the document
/// that decided, in [`SHADOW_ROWS`] rows. A binding behind the last row
/// counts in the last row, and a document applied later counts on in the
/// same rows. The series say what the door counts, and no more.
struct Series {
    /// The decisions that panicked.
    faults: Counter,
    /// The shadows of single decisions. A row is laid at its binding's
    /// first shadow, so a binding that shadows nothing shows no series.
    singles: [Option<Row>; SHADOW_ROWS],
}

impl Series {
    /// The handles of a timer that has shown nothing yet.
    fn new() -> Self {
        Self {
            faults: Metrics::policy_decision_faults(),
            singles: array::from_fn(|_| None),
        }
    }

    /// Set each series to the door's count of it.
    fn show(&mut self, policy_engine: &PolicyEngine) {
        self.faults.absolute(policy_engine.faults());
        for (binding, row) in (0..).zip(&mut self.singles) {
            let counts = ShadowKind::ALL.map(|kind| policy_engine.shadowed(binding, kind));
            if row.is_none() && counts.iter().all(|&count| count == 0) {
                continue;
            }
            let handles = row.get_or_insert_with(|| {
                ShadowKind::ALL
                    .map(|kind| Metrics::policy_shadow_decisions(binding, SINGLE, kind.as_str()))
            });
            for (handle, count) in handles.iter().zip(counts) {
                handle.absolute(count);
            }
        }
    }
}

/// One turn of the timer at `now`: a pass of the door's clock, then the
/// door's counts shown.
fn turn(policy_engine: &PolicyEngine, series: &mut Series, now: Instant) {
    policy_engine.sweep(now);
    series.show(policy_engine);
}

/// Start the timer of `policy_engine`: one turn every [`EVERY`], on no
/// request's path.
///
/// The task holds the policy engine weakly. It ends at the first tick behind
/// the policy engine's last holder: nothing stops it by hand, and it keeps no
/// context alive. A turn writes this process's store and series alone, so a
/// shutdown waits for none.
#[expect(
    clippy::disallowed_methods,
    reason = "a turn writes this process's store and series alone and the task ends behind its policy engine: a shutdown waits for neither"
)]
pub(crate) fn start(policy_engine: &Arc<PolicyEngine>) -> JoinHandle<()> {
    let policy_engine = Arc::downgrade(policy_engine);
    tokio::spawn(async move {
        let mut tick = interval(EVERY);
        // A late turn is one turn, never a burst of the turns missed.
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut series = Series::new();
        loop {
            tick.tick().await;
            let Some(policy_engine) = policy_engine.upgrade() else {
                break;
            };
            turn(&policy_engine, &mut series, Instant::now());
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use futures::FutureExt;
    use smg_policy_controller::{
        document::{AppliedSpec, PolicySpec, ShadowSpec},
        engine::ShadowKind,
        store::{GatewayId, LeasesStore, MemoryStore},
    };

    use super::{start, turn, Series, EVERY};
    use crate::{
        observability::metrics::test_support::render_with_recorder,
        policy_controller::engine::{
            testing::{bed, bed_with, MODEL},
            Ask, PolicyEngine,
        },
        routers::common::snapshot::WorkerRegistry,
    };

    /// A lease placed a minute ago, far past the time a gateway has to
    /// name its dispatch, is ended at the timer's next turn. Without a
    /// pass of the clock it stays live, counted on its worker.
    #[tokio::test(start_paused = true)]
    async fn the_timer_ends_a_lease_whose_dispatch_was_never_named() {
        let (policy_engine, store, _workers) = bed(&["http://w0:8000"]);
        let policy_engine = Arc::new(policy_engine);
        let _timer = start(&policy_engine);
        // The timer's task takes its first turn and waits for its tick.
        tokio::task::yield_now().await;

        let issued = Instant::now()
            .checked_sub(Duration::from_secs(60))
            .expect("a clock a minute old");
        let placed = policy_engine.place(&Ask::of(b"r1", MODEL), issued).await;
        assert!(placed.is_ok(), "the bed's one worker takes the request");
        assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 1);

        // Held time moves past the tick, and the timer makes its turn.
        tokio::time::advance(EVERY).await;
        tokio::task::yield_now().await;
        assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
    }

    /// The timer holds its policy engine weakly: it runs while the policy
    /// engine stands, and ends at the first tick behind the policy engine's
    /// last holder.
    #[tokio::test(start_paused = true)]
    async fn the_timer_ends_behind_its_policy_engines_last_holder() {
        let (policy_engine, _store, _workers) = bed(&["http://w0:8000"]);
        let policy_engine = Arc::new(policy_engine);
        let timer = start(&policy_engine);
        tokio::time::advance(EVERY).await;
        tokio::task::yield_now().await;
        assert!(!timer.is_finished());

        drop(policy_engine);
        let ended = tokio::time::timeout(2 * EVERY, timer).await;
        assert!(matches!(ended, Ok(Ok(()))), "the task ended: {ended:?}");
    }

    /// The policy engine of a bed of one worker whose one binding shadows a
    /// copy of its own profile: each decision has a shadow beside it. A shadow
    /// names a profile that no binding decides by.
    fn shadowing_policy_engine() -> PolicyEngine {
        let mut document = PolicySpec::preset("round_robin").expect("the preset");
        let live = document.profiles.get("round_robin").cloned();
        document
            .profiles
            .insert("next".to_owned(), live.expect("the preset's profile"));
        let binding = document
            .bindings
            .first_mut()
            .expect("the preset's one binding");
        binding.shadow = Some(ShadowSpec::Profile("next".to_owned()));
        let applied = AppliedSpec::validate(&document).expect("the document validates");
        let store = Arc::new(MemoryStore::new());
        store.apply(applied).expect("the store takes the document");
        let rows = [("http://w0:8000", Some(MODEL))];
        let (policy_engine, ..) = bed_with(store, Arc::new(WorkerRegistry::new()), rows, None);
        policy_engine
    }

    /// A turn sets the series to the door's counts: the one shadow beside
    /// the one decision, under its binding's position, the single side and
    /// its kind, with the binding's other kinds at zero, and the faults.
    /// A binding that shadowed nothing shows no row.
    #[test]
    fn a_turn_shows_the_doors_shadows_and_faults_as_series() {
        let policy_engine = shadowing_policy_engine();
        let rendered = render_with_recorder(|| {
            let mut series = Series::new();
            let placed = policy_engine
                .place(&Ask::of(b"r1", MODEL), Instant::now())
                .now_or_never()
                .expect("a decision at once");
            assert!(placed.is_ok(), "the bed's one worker takes the request");
            turn(&policy_engine, &mut series, Instant::now());
        });
        let counts = ShadowKind::ALL.map(|kind| policy_engine.shadowed(0, kind));
        assert_eq!(
            counts.iter().sum::<u64>(),
            1,
            "one shadow beside the one decision"
        );
        for (kind, count) in ShadowKind::ALL.into_iter().zip(counts) {
            let kind = kind.as_str();
            let line = format!(
                "smg_policy_shadow_decisions_total{{binding=\"0\",side=\"single\",kind=\"{kind}\"}} {count}"
            );
            assert!(rendered.contains(&line), "{line} is not in:\n{rendered}");
        }
        assert!(
            rendered.contains("smg_policy_decision_faults_total 0"),
            "{rendered}"
        );
        assert!(!rendered.contains("binding=\"1\""), "{rendered}");
    }
}
