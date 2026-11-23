use std::{collections::BTreeMap, error::Error, fs, io, path::PathBuf, time::Duration};

use serde_json::json;
use smg_service_discovery::{
    testing::{Queued, RecordedJobs},
    FileDiscoveryConfig,
};
use tempfile::TempDir;
use tokio::{sync::oneshot, task::JoinSet};

use super::*;
use crate::{
    service::{
        controller::{
            config::tests::valid,
            worker_objects::{ObjectsWake, WorkerObjects},
            Controller,
        },
        elector::{ElectError, ElectionSettings, Elector},
    },
    store::{MemoryStore, WorkerFacts, WorkerRow, WorkersStore},
};

const WORKER: &str = "http://worker:8000";
const OTHER: &str = "http://other:8000";
const KEY: &str = "discovery-key";
const INTERVAL: Duration = Duration::from_secs(1);
const WAIT: Duration = Duration::from_secs(5);
const PASS: Duration = Duration::from_millis(1250);

struct ManifestFile(TempDir);

impl ManifestFile {
    fn path(&self) -> PathBuf {
        self.0.path().join("workers.json")
    }

    fn settings(&self) -> DiscoveryConfig {
        DiscoveryConfig::File(FileDiscoveryConfig {
            path: self.path(),
            check_interval_secs: INTERVAL.as_secs(),
        })
    }

    fn publish(&self, urls: &[&str]) -> io::Result<()> {
        let workers: Vec<_> = urls.iter().map(|url| json!({ "url": url })).collect();
        self.publish_text(&json!({ "version": 1, "workers": workers }).to_string())
    }

    fn publish_text(&self, text: &str) -> io::Result<()> {
        let incoming = self.0.path().join("incoming.json");
        fs::write(&incoming, text)?;
        fs::rename(incoming, self.path())
    }
}

struct Bed {
    file: ManifestFile,
    settings: RuntimeDiscoveryConfig,
    elector: Elector,
    standing: watch::Sender<ElectionObservation>,
    stop: watch::Sender<bool>,
    workers: WorkerObjects,
    store: Arc<MemoryStore>,
    host: Arc<Host>,
    jobs: Arc<RecordedJobs>,
    tasks: JoinSet<()>,
}

impl Bed {
    fn new() -> Result<Self, Box<dyn Error>> {
        Self::with_lease(ElectionSettings {
            duration: Duration::from_secs(60),
            renew_deadline: Duration::from_secs(30),
            ..ElectionSettings::default()
        })
    }

    /// A bed whose lead ends `lease.renew_deadline` after a take, unless
    /// the case takes it again.
    fn with_lease(lease: ElectionSettings) -> Result<Self, Box<dyn Error>> {
        let file = ManifestFile(tempfile::tempdir()?);
        let settings = RuntimeDiscoveryConfig::from_config(&file.settings(), Some(Roles::Regular))?;
        let elector = Elector::alone("controller", lease)?;
        let (standing, _) = watch::channel(ElectionObservation::default());
        let (stop, _) = watch::channel(false);
        let store = Arc::new(MemoryStore::new());
        let workers = WorkerObjects::new(Arc::clone(&store), ObjectsWake::default());
        let host = Arc::new(Host::new(
            Arc::clone(workers.objects().registry()),
            Some(Credential::for_test(KEY)),
        ));
        let jobs = Arc::new(RecordedJobs::default());
        Ok(Self {
            file,
            settings,
            elector,
            standing,
            stop,
            workers,
            store,
            host,
            jobs,
            tasks: JoinSet::new(),
        })
    }

    fn take(&mut self) -> Result<(), ElectError> {
        self.elector.tick(Instant::now())?;
        self.standing.send_replace(ElectionObservation::default());
        Ok(())
    }

    fn leave(&mut self) -> Result<(), ElectError> {
        self.elector.step_down(None, Instant::now())?;
        self.standing.send_replace(ElectionObservation::default());
        Ok(())
    }

    fn discovery(&self) -> Discovery {
        Discovery::new(
            self.settings.clone(),
            Arc::clone(&self.host),
            self.elector.fence(),
        )
    }

    fn run(&mut self) {
        let discovery = self.discovery();
        let standing = self.standing.subscribe();
        let stop = self.stop.subscribe();
        self.tasks
            .spawn(async move { discovery.run(standing, stop).await });
    }

    fn install_queue(&self) -> bool {
        self.host.set_jobs(self.jobs.clone())
    }

    async fn wait_for_job(&self, index: usize) -> Option<Queued> {
        until(WAIT, || self.jobs.queued().into_iter().nth(index)).await
    }

    async fn stays_at(&self, count: usize) -> bool {
        until(PASS, || (self.jobs.queued().len() > count).then_some(()))
            .await
            .is_none()
    }

    fn row(&self, url: &str, labels: BTreeMap<String, String>) -> Option<WorkerRow> {
        self.store.upsert(WorkerFacts {
            key: url.to_owned(),
            engine: url.to_owned(),
            models: vec!["model".to_owned()],
            dp_size: 1,
            labels,
            ..WorkerFacts::default()
        });
        self.workers.objects().follow(&*self.store);
        self.store.by_key(url)
    }

    async fn stop(&mut self) -> Result<(), Box<dyn Error>> {
        self.stop.send_replace(true);
        time::timeout(WAIT, self.tasks.join_next())
            .await?
            .ok_or_else(|| io::Error::other("no discovery task"))??;
        Ok(())
    }
}

async fn until<T>(bound: Duration, mut read: impl FnMut() -> Option<T>) -> Option<T> {
    time::timeout(bound, async {
        let mut tick = time::interval(Duration::from_millis(10));
        loop {
            if let Some(value) = read() {
                return value;
            }
            tick.tick().await;
        }
    })
    .await
    .ok()
}

#[tokio::test]
async fn a_replica_that_leads_runs_the_provider_and_its_worker_reaches_the_queue(
) -> Result<(), Box<dyn Error>> {
    use smg_worker::jobs::WorkerRegistrationMode;

    let file = ManifestFile(tempfile::tempdir()?);
    file.publish(&[WORKER])?;
    let mut config = valid();
    config.discovery = Some(DiscoverySettings::new(
        file.settings(),
        None,
        Some(Credential::for_test(KEY)),
    )?);
    let controller = Controller::bind(config).await?;
    let discovery = controller.discovery.as_ref().expect("configured discovery");
    assert!(Arc::ptr_eq(&discovery.fence, &controller.shared.fence));
    let jobs = Arc::new(RecordedJobs::default());
    assert!(discovery.host.set_jobs(jobs.clone()));

    let (stop, stopped) = oneshot::channel();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let registered = async {
        let added = until(WAIT, || jobs.queued().into_iter().next()).await;
        let _ = stop.send(());
        added
    };
    let (served, added) = tokio::join!(serve, registered);
    served?;
    let Some(Queued::Add { spec, mode, .. }) = added else {
        panic!("the leader did not queue the manifest worker");
    };
    assert_eq!(spec.url, WORKER);
    assert!(
        spec.api_key.as_deref() == Some(KEY),
        "the worker key was not stamped"
    );
    assert_eq!(mode, WorkerRegistrationMode::Upsert);
    assert_eq!(jobs.queued().len(), 1);
    Ok(())
}

#[tokio::test]
async fn a_replica_that_follows_starts_no_provider() -> Result<(), Box<dyn Error>> {
    let mut bed = Bed::new()?;
    bed.file.publish(&[WORKER])?;
    assert!(bed.install_queue());
    bed.standing.send_replace(ElectionObservation {
        holder: "controller".to_owned(),
        term: 1,
        looked: true,
    });
    bed.run();
    assert!(bed.stays_at(0).await, "the follower queued a worker");
    assert_eq!(
        Arc::strong_count(&bed.host),
        2,
        "the follower started a provider"
    );
    bed.stop().await
}

#[tokio::test]
async fn a_lost_lead_ends_the_provider() -> Result<(), Box<dyn Error>> {
    let mut bed = Bed::new()?;
    bed.file.publish(&[WORKER])?;
    assert!(bed.install_queue());
    bed.take()?;
    bed.run();
    assert!(bed.wait_for_job(0).await.is_some());
    bed.jobs.hold(WORKER);

    bed.leave()?;
    let ended = until(PASS, || (Arc::strong_count(&bed.host) == 2).then_some(())).await;
    bed.file.publish(&[OTHER])?;
    assert!(
        bed.stays_at(1).await,
        "a provider queued a worker after losing the lead"
    );
    assert!(
        ended.is_some(),
        "the provider held the host after losing the lead"
    );
    bed.stop().await
}

/// The election thread's end closes the watch of its words. The provider
/// ends there, and the runner stays until the stop: the serve reads a task
/// that ends before the stop as a stopped listener, and the election's end
/// is the serve's own to say.
#[tokio::test]
async fn a_closed_watch_ends_the_provider_and_the_runner_stays_until_the_stop(
) -> Result<(), Box<dyn Error>> {
    let mut bed = Bed::new()?;
    bed.file.publish(&[WORKER])?;
    assert!(bed.install_queue());
    bed.take()?;
    bed.run();
    assert!(bed.wait_for_job(0).await.is_some());
    bed.jobs.hold(WORKER);

    // The thread's end drops its sender. The provider's hold on the host
    // goes; the runner's own stays while it lives.
    let (gone, _) = watch::channel(ElectionObservation::default());
    drop(std::mem::replace(&mut bed.standing, gone));
    let ended = until(PASS, || (Arc::strong_count(&bed.host) <= 2).then_some(())).await;
    assert!(
        ended.is_some(),
        "the provider held the host after the watch closed"
    );
    assert!(
        time::timeout(PASS, bed.tasks.join_next()).await.is_err(),
        "the runner ended before the stop"
    );
    bed.stop().await
}

#[tokio::test]
async fn a_lead_that_ends_by_the_clock_ends_the_provider_with_no_word_from_the_election(
) -> Result<(), Box<dyn Error>> {
    // The election takes the lead once and is not heard again: no renewal
    // moves the fence's window and no word reaches the watch.
    let lead = Duration::from_secs(3);
    let mut bed = Bed::with_lease(ElectionSettings {
        duration: 2 * lead,
        renew_deadline: lead,
        ..ElectionSettings::default()
    })?;
    bed.file.publish(&[WORKER])?;
    assert!(bed.install_queue());
    bed.take()?;
    bed.run();
    assert!(bed.wait_for_job(0).await.is_some());
    bed.jobs.hold(WORKER);

    let ended = until(lead + PASS, || {
        (Arc::strong_count(&bed.host) == 2).then_some(())
    })
    .await;
    bed.file.publish(&[OTHER])?;
    assert!(
        bed.stays_at(1).await,
        "a provider queued a worker past the end of its lead"
    );
    assert!(
        ended.is_some(),
        "the provider held the host past the end of its lead"
    );
    bed.stop().await
}

#[tokio::test]
async fn a_second_take_starts_the_provider_again() -> Result<(), Box<dyn Error>> {
    let mut bed = Bed::new()?;
    bed.file.publish(&[WORKER])?;
    assert!(bed.install_queue());
    bed.take()?;
    bed.run();
    assert!(bed.wait_for_job(0).await.is_some());

    // The old provider would keep the last good manifest across a bad read.
    bed.file.publish_text("{")?;
    bed.leave()?;
    bed.take()?;
    assert!(bed.stays_at(1).await, "the old provider kept reconciling");
    bed.file.publish(&[OTHER])?;
    let Some(Queued::Add { spec, .. }) = bed.wait_for_job(1).await else {
        panic!("the next term did not start a provider");
    };
    assert_eq!(spec.url, OTHER);
    bed.stop().await
}

#[tokio::test]
async fn with_no_queue_a_pass_asks_nothing_and_asks_once_a_queue_stands(
) -> Result<(), Box<dyn Error>> {
    use smg_service_discovery::DiscoveryHost;

    let mut bed = Bed::new()?;
    bed.file.publish(&[WORKER])?;
    bed.take()?;
    bed.run();
    assert!(
        until(WAIT, || (Arc::strong_count(&bed.host) == 3).then_some(()))
            .await
            .is_some()
    );
    assert!(bed.stays_at(0).await);
    assert!(bed.host.jobs().is_none());
    assert!(bed.install_queue());
    let Some(Queued::Add { spec, .. }) = bed.wait_for_job(0).await else {
        panic!("the next pass did not use the installed queue");
    };
    assert_eq!(spec.url, WORKER);
    bed.stop().await?;
    assert_eq!(bed.jobs.queued().len(), 1);
    Ok(())
}

#[tokio::test]
async fn a_row_the_provider_registered_is_not_asked_again_and_one_it_no_longer_lists_is_removed(
) -> Result<(), Box<dyn Error>> {
    let mut bed = Bed::new()?;
    bed.file.publish(&[WORKER])?;
    assert!(bed.install_queue());
    bed.take()?;
    bed.run();
    let Some(Queued::Add { spec, .. }) = bed.wait_for_job(0).await else {
        panic!("expected the first pass to stamp the registration");
    };
    let row = bed
        .row(WORKER, spec.labels.into_iter().collect())
        .expect("owned row");
    let manual = bed.row(OTHER, BTreeMap::new()).expect("unlabelled row");
    let id = bed
        .workers
        .objects()
        .object(row.id)
        .expect("followed object");
    let revision = bed
        .workers
        .objects()
        .registry()
        .get_all_with_ids()
        .into_iter()
        .find_map(|(candidate, worker)| (candidate == id).then(|| worker.revision()))
        .expect("registry revision");

    assert!(
        bed.stays_at(1).await,
        "the same manifest queued another job"
    );
    // The queue holds the removal in flight once a pass has asked it, as a
    // host's queue does. The watch's pass and the tick's can both land
    // before the row is gone, and the second then asks nothing.
    bed.jobs.hold_after_the_next_read("worker:8000");
    bed.file.publish(&[])?;
    let Some(Queued::Remove { url, guards, .. }) = bed.wait_for_job(1).await else {
        panic!("the provider did not remove its departed worker");
    };
    assert_eq!(url, "worker:8000");
    assert_eq!(guards, [(id, revision)]);
    assert!(bed.store.remove(row.id, row.revision).is_some());
    bed.workers.objects().follow(&*bed.store);
    assert!(
        bed.stays_at(2).await,
        "a removal was queued for the unlabelled row"
    );
    assert!(bed.workers.objects().object(manual.id).is_some());
    bed.stop().await
}

#[tokio::test]
async fn stopping_waits_for_the_provider_to_release_the_host() -> Result<(), Box<dyn Error>> {
    let mut bed = Bed::new()?;
    bed.file.publish(&[WORKER])?;
    assert!(bed.install_queue());
    bed.take()?;
    let discovery = bed.discovery();
    let serve = async {
        discovery
            .run(bed.standing.subscribe(), bed.stop.subscribe())
            .await;
        Arc::strong_count(&bed.host)
    };
    let stop = async {
        assert!(bed.wait_for_job(0).await.is_some());
        bed.stop.send_replace(true);
    };
    let (remaining, ()) = time::timeout(WAIT, async { tokio::join!(serve, stop) }).await?;
    assert_eq!(remaining, 2);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn a_failed_start_retries_at_the_check_interval_only_while_leading(
) -> Result<(), Box<dyn Error>> {
    use tokio::sync::mpsc;
    use tracing::instrument::WithSubscriber as _;

    let mut bed = Bed::new()?;
    bed.take()?;
    let discovery = bed.discovery();
    let term = bed
        .elector
        .fence()
        .leader_term(Instant::now())
        .expect("leading term");
    let log = bed.file.0.path().join("starts.log");
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(fs::File::create(&log)?)
        .finish();
    let (attempted, mut attempts) = mpsc::unbounded_channel();
    let refused = || async {
        let _ = attempted.send(time::Instant::now());
        Err(io::Error::other("start refused").into())
    };
    let retry = discovery
        .start(term, refused, false)
        .with_subscriber(subscriber);
    let lead = async {
        let first = attempts.recv().await.expect("first attempt");
        assert!(
            time::timeout(INTERVAL - Duration::from_millis(5), attempts.recv())
                .await
                .is_err(),
            "retried early"
        );
        assert_eq!(attempts.recv().await, Some(first + INTERVAL));
        bed.leave()
    };
    let (task, left) = time::timeout(WAIT, async { tokio::join!(retry, lead) }).await?;
    left?;
    assert!(task.is_none());
    assert!(attempts.try_recv().is_err(), "retried after the lead ended");
    let text = fs::read_to_string(log)?;
    assert_eq!(text.matches("ERROR").count(), 2);
    assert_eq!(text.matches("provider=\"file\"").count(), 2);
    Ok(())
}

#[tokio::test]
async fn a_provider_that_ends_by_itself_is_said_and_starts_again_after_the_check_interval(
) -> Result<(), Box<dyn Error>> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::mpsc;
    use tracing::instrument::WithSubscriber as _;

    let mut bed = Bed::new()?;
    bed.take()?;
    let discovery = bed.discovery();
    let log = bed.file.0.path().join("ends.log");
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(fs::File::create(&log)?)
        .finish();
    let count = AtomicUsize::new(0);
    let (started, mut starts) = mpsc::unbounded_channel();
    // The first provider returns at once and the second panics.
    let start = || {
        let nth = count.fetch_add(1, Ordering::Relaxed);
        let _ = started.send(Instant::now());
        async move {
            #[expect(
                clippy::disallowed_methods,
                reason = "the runner holds the handle of the case's provider"
            )]
            let task = tokio::spawn(async move {
                assert_ne!(nth, 1, "the case's second provider panics");
            });
            Started::Ok(task)
        }
    };
    let serve = discovery
        .run_with(bed.standing.subscribe(), bed.stop.subscribe(), start)
        .with_subscriber(subscriber);
    let said = async {
        let said = until(WAIT, || {
            let text = fs::read_to_string(&log).ok()?;
            (text.matches("ERROR").count() >= 2).then_some(text)
        })
        .await;
        bed.stop.send_replace(true);
        said
    };
    let ((), said) = time::timeout(WAIT, async { tokio::join!(serve, said) }).await?;
    let text = said.expect("the two ends were not said");
    assert_eq!(text.matches("ended by itself").count(), 1);
    assert_eq!(text.matches("worker discovery failed").count(), 1);
    assert!(text.matches("provider=\"file\"").count() >= 2);
    let first = starts.recv().await.expect("the first start");
    let second = starts.recv().await.expect("the second start");
    assert!(
        second.duration_since(first) >= INTERVAL,
        "the provider started again before the check interval"
    );
    Ok(())
}
