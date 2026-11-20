//! The upkeep of the host's door: the timer that runs its clock.
//!
//! The door ends nothing between two decisions by itself. A lease whose
//! dispatch the gateway never named stays on file, counted on its worker,
//! until a later decision at its file ends it. A pin of record past its end
//! stays for the life of the store. So do the session pins of a key that
//! stayed away past its absence. One pass of the door's clock ends all
//! three, and the timer makes that pass.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::{
    task::JoinHandle,
    time::{interval, MissedTickBehavior},
};

use super::host::Host;

/// The time between two passes. A lease has seconds to name its dispatch,
/// and the store walks its pins only where a pass of theirs is due, so a
/// second is ample.
pub(crate) const EVERY: Duration = Duration::from_secs(1);

/// Start the timer of `host`: one pass of its door's clock every
/// [`EVERY`], on no request's path.
///
/// The task holds the host weakly. It ends at the first tick behind the
/// host's last holder: nothing stops it by hand, and it keeps no context
/// alive. A pass writes this process's store alone, so a shutdown waits
/// for none.
#[expect(
    clippy::disallowed_methods,
    reason = "a pass writes this process's store alone and the task ends behind its host: a shutdown waits for neither"
)]
pub(crate) fn start(host: &Arc<Host>) -> JoinHandle<()> {
    let host = Arc::downgrade(host);
    tokio::spawn(async move {
        let mut tick = interval(EVERY);
        // A late turn makes one pass, never a burst of the passes missed.
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let Some(host) = host.upgrade() else {
                break;
            };
            host.sweep(Instant::now());
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use smg_policy_controller::store::{GatewayId, LeasesStore};

    use super::{start, EVERY};
    use crate::policy_controller::host::{
        testing::{bed, MODEL},
        Ask,
    };

    /// A lease placed a minute ago, far past the time a gateway has to
    /// name its dispatch, is ended at the timer's next pass. Without a
    /// pass it stays live, counted on its worker.
    #[tokio::test(start_paused = true)]
    async fn the_timer_ends_a_lease_whose_dispatch_was_never_named() {
        let (host, store, _workers) = bed(&["http://w0:8000"]);
        let host = Arc::new(host);
        let _timer = start(&host);
        // The timer's task takes its first turn and waits for its tick.
        tokio::task::yield_now().await;

        let issued = Instant::now()
            .checked_sub(Duration::from_secs(60))
            .expect("a clock a minute old");
        let placed = host.place(&Ask::of(b"r1", MODEL), issued).await;
        assert!(placed.is_ok(), "the bed's one worker takes the request");
        assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 1);

        // Held time moves past the tick, and the timer makes its pass.
        tokio::time::advance(EVERY).await;
        tokio::task::yield_now().await;
        assert_eq!(LeasesStore::live(&*store, GatewayId::LOCAL), 0);
    }

    /// The timer holds its host weakly: it runs while the host stands,
    /// and ends at the first tick behind the host's last holder.
    #[tokio::test(start_paused = true)]
    async fn the_timer_ends_behind_its_hosts_last_holder() {
        let (host, _store, _workers) = bed(&["http://w0:8000"]);
        let host = Arc::new(host);
        let timer = start(&host);
        tokio::time::advance(EVERY).await;
        tokio::task::yield_now().await;
        assert!(!timer.is_finished());

        drop(host);
        let ended = tokio::time::timeout(2 * EVERY, timer).await;
        assert!(matches!(ended, Ok(Ok(()))), "the task ended: {ended:?}");
    }
}
