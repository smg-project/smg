//! The membership door writes the store. Its listener brings the worker
//! registry up to date and publishes membership for decisions in that order.
//! Each membership write asks for publication before it returns. A row
//! holds no credential, so its worker object carries none.
//!
//! Under a controller's address the follower writes the store in the door's
//! place, and settles the feed after each event it takes ([`following`]).
//!
//! The load collector writes the store through the feed's sink. Those
//! writes come from the collector's tasks and name no instant to a host, so
//! the feed runs the gateway's publication loop, which shows what waits
//! when the store's instant comes. The loop is the gateway's one hand on
//! the store's wake. It keeps the listener's order: the registry is brought
//! up to date, then the store publishes.
//!
//! The store's index is fed by the workers' cache events. The feed runs
//! the crate's follow of them for as long as its answer is kept. It is the
//! gateway's one collector of cache events: the classic one is not built
//! beside it. So the follow also carries what the classic one carried
//! beside the events: each worker's pushed load records, to the worker
//! monitor, which says them to the feed's load sink.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_policy_controller::{
    store::{LoadSink as StoreLoadSink, Maker, MemoryStore, WorkerFacts},
    worker_host::{
        Expect, HealthAdapter, KvFeed, LoadAdapter, Membership, Objects, Refused, Remover,
        SoleLead, Standing, StoreDoor, Wrote,
    },
};
use smg_worker::{
    connection::engine::EngineConnector,
    health::port::{HealthSink, ProbeAnswer},
    identity::basic::Worker,
    load::port::{GroupTick, LoadSink, PollMode, PolledMember, PolledRanks},
    registry::WorkerId,
    worker::{WorkerLoadResponse, WorkerStatus},
};
use tokio::{
    sync::{watch, Notify},
    time::sleep_until,
};
use tokio_util::task::AbortOnDropHandle;
use tracing::warn;

use crate::routers::common::snapshot::{ModelWorkerSnapshot, WorkerMonitor, WorkerRegistry};

/// The feed of a gateway that decides in its own process: the door its
/// worker workflows write at, and the holder that keeps the registry as
/// the store's rows say.
#[derive(Clone)]
pub(crate) struct Feed {
    /// The membership door: the store's own, for a host with no election.
    pub door: Arc<dyn Membership>,
    /// The one writer of the objects of rows where the controller was picked.
    pub objects: Arc<Objects<ModelWorkerSnapshot>>,
    /// The store the door writes and the holder follows.
    store: Arc<MemoryStore>,
    /// The feed's turn: one write with its listener, one settle, or one
    /// publication of the loop, at a time.
    turn: Arc<Mutex<()>>,
    /// The publication loop's wake: a load write through the feed's sink,
    /// or the store's word that a membership write left a build owed.
    woken: Arc<Notify>,
}

impl Feed {
    /// The health sink of a gateway that keeps its workers in the store.
    pub(crate) fn health_sink(&self) -> Arc<dyn HealthSink> {
        Arc::new(FeedHealth {
            objects: Arc::clone(&self.objects),
            adapter: HealthAdapter::new(
                Arc::clone(&self.objects),
                Arc::clone(&self.store),
                Arc::new(SoleLead),
                SoleLead::TERM,
            ),
            registry: Arc::clone(self.objects.registry()),
        })
    }

    /// Bring the registry to the store's rows, then publish them: the two
    /// calls the door's listener makes after each write. A follower that
    /// applies a served controller's events calls this after each event.
    /// It takes the feed's turn, so it never runs between a write at the
    /// door and that write's listener.
    pub(crate) fn settle(&self) {
        let _turn = self.turn.lock();
        settle(&self.objects, &self.store, None);
    }

    /// The store the door writes and the holder follows.
    pub(crate) fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }

    /// The load sink of a gateway that keeps its workers in the store: the
    /// crate's adapter under the sole lead. `period` is the interval in
    /// which the collector hears every ready rank. A worker no row stands
    /// for is passed over: nothing of it reaches the store.
    pub(crate) fn load_sink(&self, period: Duration) -> Arc<dyn LoadSink> {
        let adapter = LoadAdapter::new(
            Arc::clone(&self.objects),
            Arc::clone(&self.store),
            Arc::new(SoleLead),
            SoleLead::TERM,
            period,
        );
        if !adapter.period_taken() {
            warn!(
                period_secs = period.as_secs(),
                "The policy store keeps an earlier load report period, so this one is not in force"
            );
        }
        Arc::new(FeedLoads {
            adapter,
            woken: Arc::clone(&self.woken),
        })
    }

    /// Show the store's waiting writes as they come due, for as long as
    /// the answer is kept. The collector's sink writes from the collector's
    /// tasks and reads no instant, so without the loop a load write waits
    /// for the next one to be shown. The loop takes the store's one wake.
    pub(crate) fn publications(&self) -> AbortOnDropHandle<()> {
        self.store.wake_with(Box::new({
            let woken = Arc::clone(&self.woken);
            move || woken.notify_one()
        }));
        #[expect(
            clippy::disallowed_methods,
            reason = "the loop holds no request; its answer ends it, and a shutdown need not wait for it"
        )]
        let task = tokio::spawn(self.clone().publish_as_due());
        AbortOnDropHandle::new(task)
    }

    /// With no instant in hand the loop waits for a wake. With one in hand
    /// it sleeps to it and settles at it. A wake that comes while it sleeps
    /// is kept, and costs one more read of the instant.
    async fn publish_as_due(self) {
        loop {
            let Some(due) = self.store.publish_due() else {
                self.woken.notified().await;
                continue;
            };
            sleep_until(due.into()).await;
            self.settle_at(due);
        }
    }

    /// The loop's publication: a settle at the store's instant, under the
    /// feed's turn. It can pay a membership build, and the store may hold a
    /// row written past the door, so the registry is brought up to date
    /// first, as at every publication the feed makes.
    fn settle_at(&self, due: Instant) {
        let _turn = self.turn.lock();
        // The store shows nothing before its instant, so the loop never
        // hands it an earlier one.
        settle(&self.objects, &self.store, Some(Instant::now().max(due)));
    }

    /// Read the cache events of the store's workers into its index, and
    /// their pushed load records into `loads`, for as long as the answer is
    /// kept. The follow is the crate's own: as the registry changes it asks
    /// a stream of each gRPC worker, for its cache events where its row asks
    /// for them and for its load records alone where `loads` wants them.
    pub(crate) fn cache_events(&self, loads: Option<&Arc<WorkerMonitor>>) -> CacheEvents {
        let follow = KvFeed::new(Arc::clone(&self.store), Arc::clone(&self.objects));
        if let Some(monitor) = loads {
            follow.set_load_sink(monitor);
        }
        let (stop, stopped) = watch::channel(false);
        #[expect(
            clippy::disallowed_methods,
            reason = "the follow holds no request; its answer's drop stops it, and a shutdown need not wait for it"
        )]
        tokio::spawn(async move { follow.run(stopped).await });
        CacheEvents(stop)
    }
}

/// The feed's cache events, for as long as this is kept. Its drop tells the
/// follow to stop. The follow then ends every stream itself, so its task is
/// not aborted.
pub(crate) struct CacheEvents(watch::Sender<bool>);

impl Drop for CacheEvents {
    fn drop(&mut self) {
        self.0.send_replace(true);
    }
}

/// The crate's load adapter with the loop's wake. Every fact goes to the
/// adapter. A call that can leave a write waiting in the store wakes the
/// publication loop behind it. A reset is one: it drops the loads of rows
/// that are not `Ready`. A question leaves none.
struct FeedLoads {
    adapter: LoadAdapter<ModelWorkerSnapshot, MemoryStore>,
    woken: Arc<Notify>,
}

impl LoadSink for FeedLoads {
    fn record(
        &self,
        worker: &Arc<dyn Worker>,
        rank: i32,
        report: &WorkerLoadResponse,
        received_at: Instant,
        sampled_at: Instant,
    ) {
        self.adapter
            .record(worker, rank, report, received_at, sampled_at);
        self.woken.notify_one();
    }

    fn poll_mode(&self, worker: &Arc<dyn Worker>, now: Instant, interval: Duration) -> PollMode {
        self.adapter.poll_mode(worker, now, interval)
    }

    fn report(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> PolledRanks {
        let ranks = self.adapter.report(answer, polled_at);
        self.woken.notify_one();
        ranks
    }

    fn report_empty(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> Vec<i32> {
        let leaving = self.adapter.report_empty(answer, polled_at);
        self.woken.notify_one();
        leaving
    }

    fn fetch_failed(&self, members: &[Arc<dyn Worker>], polled_at: Instant) {
        self.adapter.fetch_failed(members, polled_at);
        self.woken.notify_one();
    }

    fn forget(&self, worker: &Arc<dyn Worker>) {
        self.adapter.forget(worker);
        self.woken.notify_one();
    }

    fn reset(&self) {
        self.adapter.reset();
        self.woken.notify_one();
    }

    fn needs_loads(&self) -> bool {
        self.adapter.needs_loads()
    }

    fn tick(&self, tick: &GroupTick<'_>) {
        self.adapter.tick(tick);
    }
}

/// The adapter writes a row's health and mirrors its status on the worker.
/// It writes a move of the worker's liveness veto as the row's stalled bit.
/// The registry writes the health of workers without a row.
/// The adapter answers stale for those workers, so it cannot promote them.
struct FeedHealth {
    objects: Arc<Objects<ModelWorkerSnapshot>>,
    adapter: HealthAdapter<ModelWorkerSnapshot, MemoryStore>,
    registry: Arc<WorkerRegistry>,
}

impl FeedHealth {
    fn sink(&self, worker_id: &WorkerId) -> &dyn HealthSink {
        if self.objects.standing(worker_id).is_some() {
            &self.adapter
        } else {
            self.registry.as_ref()
        }
    }
}

impl HealthSink for FeedHealth {
    fn probe_outcome(
        &self,
        worker_id: &WorkerId,
        passed: bool,
        revision: u64,
        launched: WorkerStatus,
    ) -> ProbeAnswer {
        self.sink(worker_id)
            .probe_outcome(worker_id, passed, revision, launched)
    }

    fn connected(
        &self,
        worker_id: &WorkerId,
        revision: u64,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        self.sink(worker_id).connected(worker_id, revision)
    }

    fn liveness_moved(&self, worker_id: &WorkerId, revision: u64) {
        self.sink(worker_id).liveness_moved(worker_id, revision);
    }
}

/// The order is the rule: the registry holds a row's object before a
/// decision can read the row. With an instant the store shows what is due
/// at it. With none it pays the build the membership writes left owed, at
/// once.
fn settle(objects: &Objects<ModelWorkerSnapshot>, store: &MemoryStore, at: Option<Instant>) {
    objects.follow(store);
    match at {
        Some(now) => store.publish(now),
        None => store.publish_now(),
    }
}

/// The feed over `store`, for the workers of `registry`. The holder lays
/// each object with `connector`, the one the gateway was started with.
pub(crate) fn feed(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
) -> Feed {
    over(store, holder(registry, connector))
}

/// The feed of a gateway that follows a controller at an address.
///
/// Such a gateway probes no worker of its own. The controller's verdict is
/// each row's status, and the holder copies it onto the row's object. The
/// follower writes the store and settles this feed after each event.
/// Nothing writes at its door.
pub(crate) fn following(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
) -> Feed {
    over(store, holder(registry, connector).with_status_followed())
}

/// A holder that writes `registry` and lays each object with `connector`.
fn holder(
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
) -> Objects<ModelWorkerSnapshot> {
    Objects::new(Arc::clone(registry)).with_connector(connector)
}

fn over(store: &Arc<MemoryStore>, objects: Objects<ModelWorkerSnapshot>) -> Feed {
    let objects = Arc::new(objects);
    let door = StoreDoor::new(Arc::clone(store)).telling({
        let objects = Arc::clone(&objects);
        let store = Arc::clone(store);
        move || settle(&objects, &store, None)
    });
    let turn = Arc::new(Mutex::new(()));
    Feed {
        door: Arc::new(FeedDoor {
            door,
            turn: Arc::clone(&turn),
        }),
        objects,
        store: Arc::clone(store),
        turn,
        woken: Arc::new(Notify::new()),
    }
}

/// The store door releases its turn before calling the listener. Hold the
/// feed's turn through that call, so neither a later write nor a settle
/// can publish a row the holder has yet to read.
struct FeedDoor {
    door: StoreDoor,
    turn: Arc<Mutex<()>>,
}

impl FeedDoor {
    fn write(
        &self,
        write: impl FnOnce(&StoreDoor) -> Result<Wrote, Refused>,
    ) -> Result<Wrote, Refused> {
        let _turn = self.turn.lock();
        write(&self.door)
    }
}

impl Membership for FeedDoor {
    fn found(
        &self,
        maker: &Maker,
        expect: Expect,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused> {
        self.write(|door| door.found(maker, expect, url, state))
    }

    fn drained(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.write(|door| door.drained(by, url, seen))
    }

    fn gone(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.write(|door| door.gone(by, url, seen))
    }
}

#[cfg(test)]
mod cache_events_tests;
#[cfg(test)]
mod tests;
