use std::{future::pending, io, net::SocketAddr};

use parking_lot::Mutex;
use tokio::{net::TcpListener, task::JoinSet};
use tonic::{
    transport::{server::TcpIncoming, Error as Transport, Server},
    Response, Streaming,
};

use super::*;
use crate::{
    document::spec::Pairing,
    engine::view::{Role, WireModes, WorkerId},
    service::{
        controller::{
            decision_slots::tests::{followed, frame_for, holder, lent_pick, preset, unserved},
            election::LeasePlace,
            replication::{
                journaled,
                tests::{journaled_from, parts},
                Replication,
            },
            worker_objects::WorkerObjects,
        },
        elector::{ElectionSettings, Elector, TickOutcome},
        proto::grand_central_server::{GrandCentral, GrandCentralServer},
        replication::{Entry, Event},
    },
    stages::registry,
    store::{WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::tests::lay_standing,
};

mod apply;
mod stream;

impl Lag {
    /// The age at `now` of the last frame at the head; `None` where the lag
    /// holds none.
    pub(in crate::service::controller) fn age(&self, now: Instant) -> Option<Duration> {
        self.read(now).map(|(age, _)| age)
    }

    /// Whether this replica is synced at `now`: its last frame at the head
    /// is younger than the bound.
    fn synced(&self, now: Instant) -> bool {
        self.read(now).is_some_and(|(_, lapse)| lapse.is_some())
    }

    /// The bound [`Lag::synced`] reads under.
    pub(in crate::service::controller) fn within(&self) -> Duration {
        self.within
    }
}

type Replica = (Arc<MemoryStore>, Arc<Log>);

/// A replica's store and the log it journals to under a shared lease:
/// the log's places start where this replica's run drew.
fn replica() -> Replica {
    let (store, feed) = journaled(&LeasePlace::File("shared".into()), 1);
    (store, feed.log)
}

/// The place before a log's first write, where a standby that applied
/// nothing is continued from.
fn start(log: &Log) -> Position {
    log.first().expect("every write is retained")
}

fn facts(key: &str, incarnation: u64) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation,
        models: vec!["m".to_owned()],
        dp_size: 1,
        wire_modes: WireModes::GRPC,
        ..WorkerFacts::default()
    }
}

/// Rows for a snapshot of two parts: one more than a page of rows
/// holds, which is 128.
fn a_page_and_one(leader: &MemoryStore) {
    for n in 0..129 {
        leader.upsert(facts(&format!("http://w{n:03}:8000"), 1));
    }
}

/// Every entry of `log` after `place`.
fn entries(log: &Log, mut place: Position) -> Vec<Entry> {
    let (mut all, mut out) = (Vec::new(), Vec::new());
    loop {
        log.read(place, 64, &mut out).expect("the place is held");
        let Some(last) = out.last() else {
            return all;
        };
        place = last.at;
        all.append(&mut out);
    }
}

/// One entry as the batch a leader sends it in, the last its log
/// holds.
fn batch(entry: &Entry) -> proto::ReplicationBatch {
    batch_of(std::slice::from_ref(entry), entry.at.seq)
}

/// `entries`, written under one term, as the batch a leader sends them
/// in while its log's head stands at `head`.
fn batch_of(entries: &[Entry], head: u64) -> proto::ReplicationBatch {
    proto::ReplicationBatch {
        first_seq: entries[0].at.seq,
        term: entries[0].at.term,
        events: entries.iter().map(worker_wire::replication).collect(),
        resync_required: false,
        snapshot: None,
        head_seq: head,
    }
}

/// The standby applies what the leader's log holds after its place, a
/// standby that applied nothing from the leader's first write.
fn follow(follower: &Follower, leader: &Log, now: Instant) {
    let place = match follower.place() {
        nothing if nothing == Position::default() => start(leader),
        place => place,
    };
    for entry in entries(leader, place) {
        assert_eq!(follower.apply(&batch(&entry), now), Ok(()), "{entry:?}");
    }
}

/// What a replica holds, without the instants: its place, its epoch
/// and its rows.
fn held(store: &MemoryStore) -> impl PartialEq + std::fmt::Debug {
    let cut = store.checkpoint();
    let rows: Vec<_> = cut
        .rows
        .into_iter()
        .map(|row| (row.id, row.facts, row.status, row.revision))
        .collect();
    (cut.at, cut.epoch, rows, store.term())
}

/// The parts of the leader's snapshot at `at`, as its replication scope
/// sends them under the term its store wrote.
fn snapshot((store, log): &Replica, at: Instant) -> Vec<proto::ReplicationBatch> {
    let mut elector = Elector::alone("controller-0@10.0.0.7:30100", ElectionSettings::default())
        .expect("the default settings are in order");
    let Ok(TickOutcome::Acquired { term }) = elector.tick(at) else {
        panic!("a free lease is taken at the first tick");
    };
    // The cases write the leader's rows through its store's own
    // entries: the holder its cut is taken from lays what stands.
    let cut_from = holder(store);
    lay_standing(&cut_from, store);
    let cut = Replication::new(
        Arc::clone(store),
        Arc::clone(log),
        elector.fence(),
        cut_from,
    )
    .cut(term)
    .expect("a cut");
    parts(cut, at)
}

/// A leader's store with its first worker, under term 1.
fn led() -> Replica {
    let leader = replica();
    leader.0.set_term(1);
    leader.0.upsert(facts("http://a:8000", 1));
    leader
}

/// The stock door over a replica's store.
fn door((store, _): &Replica) -> Arc<Generations> {
    let catalog = Arc::new(registry::StageRegistry::builtin());
    Arc::new(Generations::new(Arc::clone(store), catalog))
}
