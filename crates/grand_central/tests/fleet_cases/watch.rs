//! A watcher on the controller: it holds the fleet's rows as the controller
//! tells them, beside a link and past the link's end.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_grand_central::{
    engine::WorkerId,
    service::{
        client::{self, Client},
        proto::{gateway_frame, Hello},
        watcher::{WatchStream, WatchStreamEnd, Watcher},
    },
    store::{MemoryStore, WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::{Members, Objects},
};
use smg_worker::{registry::WorkerRegistry, worker::WorkerStatus as ObjectStatus};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::Notify,
    task::JoinSet,
};

use crate::{
    ask,
    fleet::{Harness, ROOM},
    linked_fleet, linked_pick, CASE, FLEET,
};

/// A watcher of `store` over worker objects of its own. The holder follows
/// no log of the store: it lays each status the watch tells.
fn watcher_of(store: Arc<MemoryStore>) -> Watcher<Members> {
    let objects = Objects::new(Arc::new(WorkerRegistry::new()));
    Watcher::new(store, Arc::new(objects))
}

/// What a watcher holds of its controller's workers, by id, in a first
/// copy. Each row's facts and revision are as the watcher's holder reads
/// them off its worker objects, and its status is the one the store told.
/// The watcher's table holds each row with the same facts at the same
/// revision, and no object stands that the table does not tell. A row the
/// table alone holds is not seen here.
fn watched(watcher: &Watcher<Members>) -> Vec<(WorkerId, WorkerFacts, WorkerStatus, u64)> {
    let mut rows = watcher.objects().checkpoint(watcher.store()).rows;
    rows.sort_by_key(|row| row.id.0);
    let objects = watcher.objects();
    assert_eq!(objects.len(), rows.len(), "one told row a worker object");
    let table = |id| WorkersStore::get(&**watcher.store(), id);
    rows.into_iter()
        .map(|row| {
            let held = table(row.id).map(|held| (held.facts, held.revision));
            let stated = (row.facts.clone(), row.revision);
            assert_eq!(held, Some(stated), "the table's row");
            (row.id, row.facts, row.status, row.revision)
        })
        .collect()
}

/// The status of the worker object a watcher holds for a row, if it holds
/// the row: what the watch told last, with nothing followed behind it.
fn status_of(watcher: &Watcher<Members>, id: WorkerId) -> Option<ObjectStatus> {
    let objects = watcher.objects();
    let object = objects.registry().get(&objects.object(id)?)?;
    Some(object.status())
}

/// Follows the stream until `stands` holds of its watcher, each event
/// inside the room of a wait.
async fn follows(
    following: &mut WatchStream<'_, Members>,
    mut stands: impl FnMut(&Watcher<Members>) -> bool,
) -> Result<(), String> {
    while !stands(following.watcher()) {
        let taken = tokio::time::timeout(ROOM, following.next())
            .await
            .map_err(|_| "the watch told nothing inside its room".to_owned())?
            .map_err(|ended| format!("the stream is followed no further: {ended:?}"))?;
        if !taken {
            return Err("the watch ended with no status".to_owned());
        }
    }
    Ok(())
}

/// A gateway that watches holds the controller's rows. A snapshot tells the
/// fleet as it registered. The tail tells each status the prober writes, an
/// engine that registers behind the snapshot and a dead engine's status. A
/// stream opened after the last place taken is told the tail alone. At one
/// place, the watcher holds what one that started from a later snapshot
/// holds. The later snapshot keeps a lease already on file in its store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_watcher_holds_the_fleets_rows_as_the_controller_tells_them() {
    use smg_grand_central::{
        store::{GatewayId, LeaseRecord, LeaseState, LeasesStore, PickRecord},
        tracking::{Lease, LeaseId},
    };

    let case = async {
        let mut harness = Harness::start().await.expect("the harness starts");
        let engines = harness
            .register(&FLEET[..3])
            .await
            .expect("the fleet registers");
        let ids: Vec<WorkerId> = engines
            .iter()
            .map(|engine| harness.worker(*engine).expect("the engine's row"))
            .collect();
        let dead = |watcher: &Watcher<Members>, id: WorkerId| {
            status_of(watcher, id).is_some_and(|status| status != ObjectStatus::Ready)
        };

        // The snapshot tells the fleet as it registered: the controller's
        // prober may have readied a row already.
        let mut watcher = watcher_of(Arc::new(MemoryStore::new()));
        let opened = watcher.open(harness.client()).await;
        let mut stream = opened.expect("the watch opens");
        let told = follows(&mut stream, |held| held.at().is_some()).await;
        told.expect("a snapshot is told");
        for id in &ids {
            let status = status_of(stream.watcher(), *id);
            assert!(
                status == Some(ObjectStatus::Pending) || status == Some(ObjectStatus::Ready),
                "a row of the snapshot reads {status:?}"
            );
        }

        // The prober starts: the tail tells each engine it heard.
        harness.feed();
        let ready = |watcher: &Watcher<Members>| {
            ids.iter()
                .all(|id| status_of(watcher, *id) == Some(ObjectStatus::Ready))
        };
        let told = follows(&mut stream, ready).await;
        told.expect("the fleet is told ready");

        // An engine that registers behind the snapshot.
        let late = harness.register(&FLEET[3..4]).await;
        let late = late.expect("one more engine registers")[0];
        let late = harness.worker(late).expect("the late engine's row");
        let ready =
            |watcher: &Watcher<Members>| status_of(watcher, late) == Some(ObjectStatus::Ready);
        let told = follows(&mut stream, ready).await;
        told.expect("the late engine is told ready");

        // A dead engine: the prober's word reaches the watcher.
        harness.kill(engines[0]).await.expect("an engine dies");
        let told = follows(&mut stream, |held| dead(held, ids[0])).await;
        told.expect("the death is told");

        // A stream opened after the last place taken: the tail alone.
        drop(stream);
        let opened = watcher.open(harness.client()).await;
        let mut stream = opened.expect("the watch opens again");
        harness
            .kill(engines[1])
            .await
            .expect("a second engine dies");
        let told = follows(&mut stream, |held| dead(held, ids[1])).await;
        told.expect("the second death is told");
        assert_eq!(
            stream.watcher().snapshots(),
            1,
            "no snapshot behind a place"
        );

        // A watcher that starts now takes a snapshot. Both are then told
        // one more death, and at its place the first holds what the second
        // holds.
        let store = Arc::new(MemoryStore::new());
        let leases: &dyn LeasesStore = &*store;
        let gateway = GatewayId(1);
        let now = Instant::now();
        leases
            .issue(LeaseRecord {
                lease: {
                    let mut lease = Lease::default();
                    lease.id = LeaseId(1);
                    lease
                },
                gateway,
                request: 1,
                target: late,
                picked: PickRecord::default(),
                decode: None,
                untold: false,
                term: 0,
                issued_at: now,
                dispatch_deadline: now + ROOM,
                state: LeaseState::ISSUED,
            })
            .expect("filed");
        assert_eq!(leases.live(gateway), 1);
        let mut fresh = watcher_of(Arc::clone(&store));
        let opened = fresh.open(harness.client()).await;
        let mut snapshot = opened.expect("a second watch opens");
        let told = follows(&mut snapshot, |held| held.at().is_some()).await;
        told.expect("a snapshot is told");
        assert_eq!(leases.live(gateway), 1);
        harness.kill(engines[2]).await.expect("a third engine dies");
        let told = follows(&mut stream, |held| dead(held, ids[2])).await;
        told.expect("the third death is told on the tail");
        let told = follows(&mut snapshot, |held| dead(held, ids[2])).await;
        told.expect("the third death is told behind the snapshot");
        let (first, second) = (stream.watcher(), snapshot.watcher());
        assert_eq!(first.at(), second.at(), "both stand at the death's place");
        assert_eq!(watched(first), watched(second));
        assert_eq!(watched(first).len(), 4, "four engines of one rank");
        assert_eq!((first.snapshots(), second.snapshots()), (1, 1));

        drop((stream, snapshot));
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_link_answers_beside_its_watch_and_the_watch_outlives_it() {
    const GATEWAY: &str = "fleet-link-watch";
    let case = async {
        let (mut harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let first = harness.worker(0).expect("the first worker's row");
        let mut watcher = watcher_of(Arc::new(MemoryStore::new()));
        let mut stream = watcher
            .open_on(&link)
            .await
            .expect("the link opens a watch");
        follows(&mut stream, |held| held.at().is_some())
            .await
            .expect("the watch takes a snapshot");
        assert_eq!(watched(stream.watcher()).len(), 1);
        assert_eq!(
            status_of(stream.watcher(), first),
            Some(ObjectStatus::Ready)
        );

        let tail = follows(&mut stream, |held| watched(held).len() == 2);
        let ask_and_register = async {
            linked_pick(&link, ask(b"watching"))
                .await
                .expect("the link answers while the watch waits");
            harness
                .register(&FLEET[1..2])
                .await
                .expect("a worker joins the fleet");
        };
        let (told, ()) = tokio::join!(biased; tail, ask_and_register);
        told.expect("the watch takes the new row on its tail");
        drop(stream);

        let mut stream = watcher
            .open_on(&link)
            .await
            .expect("the link resumes the watch");
        link.queue_frame(gateway_frame::Frame::Hello(Hello::default()));
        tokio::time::timeout(ROOM, link.closed())
            .await
            .expect("the second hello ends the link");
        drop(link);

        harness.kill(0).await.expect("the first worker dies");
        follows(&mut stream, |held| {
            status_of(held, first).is_some_and(|status| status != ObjectStatus::Ready)
        })
        .await
        .expect("the watch takes the status after the link ends");
        assert_eq!(stream.watcher().snapshots(), 1);
        drop(stream);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A watch whose controller goes silent, with its connection still open,
/// ends: the client keeps its channel alive, and a peer that answers no
/// ping ends the streams on it. A relay stands between the two. At a word
/// it stops relaying and closes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_watch_ends_when_its_controller_goes_silent() {
    let case = async {
        let harness = Harness::start().await.expect("the harness starts");
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let relay = listener.local_addr().expect("the relay's address");
        let controller = harness.gateway();
        let silent = Arc::new(Notify::new());
        let word = Arc::clone(&silent);
        let mut relayed = JoinSet::new();
        relayed.spawn(async move {
            let (mut near, _) = listener.accept().await.expect("the client dials");
            let far = TcpStream::connect(controller).await;
            let mut far = far.expect("the controller listens");
            tokio::select! {
                _ = tokio::io::copy_bidirectional(&mut near, &mut far) => {}
                () = word.notified() => std::future::pending::<()>().await,
            }
        });

        let address = format!("http://{relay}");
        let connected = Client::connect(&address, "silent-gateway").await;
        let mut client = connected.expect("the client connects through the relay");
        let mut watcher = watcher_of(Arc::new(MemoryStore::new()));
        let opened = watcher.open(&mut client).await;
        let mut stream = opened.expect("the watch opens");
        let first = tokio::time::timeout(ROOM, stream.next()).await;
        assert!(matches!(first, Ok(Ok(true))), "the snapshot: {first:?}");

        silent.notify_one();
        // A ping goes out one keepalive after the last frame read, and the
        // channel ends one answer's wait after a ping with no answer.
        let bound = client::KEEPALIVE + client::ANSWER_WAIT + Duration::from_secs(5);
        let ended = tokio::time::timeout(bound, stream.next())
            .await
            .expect("a silent controller ends the stream inside its bound");
        assert!(matches!(ended, Err(WatchStreamEnd::Client(_))), "{ended:?}");

        relayed.abort_all();
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}
