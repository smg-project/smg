//! The worker load sink over the store.
//!
//! The collector holds its reports lock at a record, a poll answer or a
//! forget. The adapter takes its member map lock inside it, then reads the
//! registry. On a miss it also reads the holder, under the map's write
//! lock. It releases the map before calling the store, which takes its own
//! locks and may pay for a membership build on a poll answer. The adapter
//! calls no collector.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use smg_worker::{
    identity::basic::Worker,
    load::port::{LoadSink, PollMode, PolledMember, PolledRanks},
    registry::{Generation, WorkerId as ObjectId},
    worker::{ConnectionMode, WorkerLoadResponse},
};

use super::{object::role, Lead, Objects};
use crate::{
    load_report::load_row,
    store::{LoadSink as StoreLoadSink, LoadSource, PollMode as StorePollMode},
    view::{Role, SignalSource, WorkerId},
};

#[derive(Clone, Copy)]
struct Member {
    row: WorkerId,
    rank: u16,
    role: Role,
}

struct KeptMember {
    worker: Arc<dyn Worker>,
    id: ObjectId,
    member: Member,
}

/// The worker crate's load port over a store's load sink.
///
/// The overload verdict lives on the store's rows. The adapter leaves the
/// object's overload flag alone. A rank worker's row uses its own rank.
/// For any other worker the row covers the whole report, and a record is
/// written at the rank it came from. The store reads a worker that is no
/// rank at its own rank alone: where such a worker stands over an engine
/// of several ranks, the records of its other ranks are written and not
/// read, so its load under-reads until its registration states the width.
/// An unready rank counts as served when checking an empty
/// answer, so the engine's verdict stands until its next record or report.
/// Loads are a row's, not an object's: a forgotten object's row keeps them
/// while an object stands under it, the same or its replacement, and its
/// engine is judged again at its next record or report.
/// The host publishes the store's pending writes.
pub struct LoadAdapter<G, S: ?Sized> {
    objects: Arc<Objects<G>>,
    store: Arc<S>,
    lead: Arc<dyn Lead>,
    term: u64,
    /// One entry per URL heard or polled. Forget drops the object's entry
    /// and reset clears the map when registry events were lost.
    members: RwLock<FxHashMap<String, KeptMember>>,
    period_taken: bool,
}

impl<G: Generation, S: StoreLoadSink + ?Sized> LoadAdapter<G, S> {
    /// For the lead under `term`. Tells the store once how often the
    /// collector hears each ready rank, by its stream or a poll. The store
    /// keeps the first period it is told: [`Self::period_taken`] says
    /// whether this one stands.
    pub fn new(
        objects: Arc<Objects<G>>,
        store: Arc<S>,
        lead: Arc<dyn Lead>,
        term: u64,
        heard_every: Duration,
    ) -> Self {
        let period_taken = store.heard_every(heard_every);
        Self {
            objects,
            store,
            lead,
            term,
            members: RwLock::default(),
            period_taken,
        }
    }

    /// Whether the store took this adapter's period. `false` where another
    /// period stood first: the store keeps that one and ages a rank's
    /// sample by it.
    pub fn period_taken(&self) -> bool {
        self.period_taken
    }

    fn leads(&self) -> bool {
        self.lead.term() == Some(self.term)
    }

    fn is_current(&self, id: &ObjectId, worker: &Arc<dyn Worker>) -> bool {
        self.objects
            .registry()
            .get(id)
            .is_some_and(|current| Arc::ptr_eq(&current, worker))
    }

    fn read_member(&self, worker: &Arc<dyn Worker>) -> Option<KeptMember> {
        let id = self.objects.registry().get_id_by_url(worker.url())?;
        let standing = self.objects.standing_of(&id, worker)?;
        if !self.is_current(&id, worker) {
            return None;
        }
        Some(KeptMember {
            worker: Arc::clone(worker),
            id,
            member: Member {
                row: standing.row,
                rank: u16::try_from(worker.dp_rank().unwrap_or(0)).ok()?,
                role: role(*worker.worker_type()),
            },
        })
    }

    fn member(&self, worker: &Arc<dyn Worker>) -> Option<Member> {
        let url = worker.url();
        if let Some(kept) = self.members.read().get(url) {
            // The registry can retire the cached object before the collector
            // sees the event. The kept id lets this check borrow the key.
            if Arc::ptr_eq(&kept.worker, worker) && self.is_current(&kept.id, &kept.worker) {
                return Some(kept.member);
            }
        }
        let mut members = self.members.write();
        let kept = self.read_member(worker)?;
        let member = kept.member;
        members.insert(url.to_string(), kept);
        Some(member)
    }

    fn write_report<'a>(
        &self,
        answer: impl Iterator<Item = (&'a Arc<dyn Worker>, Option<&'a WorkerLoadResponse>)>,
        polled_at: Instant,
        received_at: Instant,
    ) -> Vec<i32> {
        if !self.leads() {
            return Vec::new();
        }
        let mut answer = answer.peekable();
        let Some((first, _)) = answer.peek() else {
            return Vec::new();
        };
        let width = first.dp_size().unwrap_or(1);
        let source = match first.connection_mode() {
            ConnectionMode::Grpc => LoadSource::Poll,
            ConnectionMode::Http | ConnectionMode::Zmq => LoadSource::Scrape,
        };
        let rows: Vec<_> = answer
            .filter_map(|(worker, report)| {
                let member = self.member(worker)?;
                let row = report
                    .filter(|report| !report.loads.is_empty())
                    .map(|report| {
                        load_row(report, member.role, received_at, received_at, source.into())
                    });
                Some((member.row, member.rank, row))
            })
            .collect();
        if rows.is_empty() || (rows.len() != width && rows.iter().all(|(_, _, row)| row.is_none()))
        {
            return Vec::new();
        }
        self.store
            .report(&rows, polled_at, source)
            .into_iter()
            .filter_map(|position| rows.get(position))
            .map(|&(_, rank, _)| i32::from(rank))
            .collect()
    }
}

impl<G: Generation, S: StoreLoadSink + ?Sized + 'static> LoadSink for LoadAdapter<G, S> {
    fn record(
        &self,
        worker: &Arc<dyn Worker>,
        rank: i32,
        report: &WorkerLoadResponse,
        received_at: Instant,
        sampled_at: Instant,
    ) {
        if !self.leads() {
            return;
        }
        let Ok(rank) = u16::try_from(rank) else {
            return;
        };
        let Some(member) = self.member(worker) else {
            return;
        };
        let row = load_row(
            report,
            member.role,
            received_at,
            sampled_at,
            SignalSource::Events,
        );
        self.store
            .record(member.row, rank, row, received_at, LoadSource::Stream);
    }

    fn poll_mode(&self, worker: &Arc<dyn Worker>, now: Instant, interval: Duration) -> PollMode {
        if !self.leads() {
            return PollMode::Suppressed;
        }
        let Some(member) = self.member(worker) else {
            return PollMode::Suppressed;
        };
        match self.store.poll_mode(member.row, now, interval) {
            StorePollMode::Poll => PollMode::Poll,
            StorePollMode::Fallback => PollMode::Fallback,
            StorePollMode::Suppressed => PollMode::Suppressed,
        }
    }

    fn report(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> PolledRanks {
        let newer = self.write_report(
            answer.iter().map(|member| (member.worker, member.answer)),
            polled_at,
            Instant::now(),
        );
        PolledRanks {
            newer,
            leaving: Vec::new(),
        }
    }

    fn report_empty(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> Vec<i32> {
        self.write_report(
            answer.iter().map(|member| (member.worker, None)),
            polled_at,
            polled_at,
        );
        Vec::new()
    }

    fn fetch_failed(&self, members: &[Arc<dyn Worker>], polled_at: Instant) {
        self.write_report(
            members.iter().map(|worker| (worker, None)),
            polled_at,
            polled_at,
        );
    }

    fn forget(&self, worker: &Arc<dyn Worker>) {
        // The entry goes whatever the term: it holds the worker object, and
        // with it the object's engine connection.
        let kept = {
            let mut members = self.members.write();
            if members
                .get(worker.url())
                .is_some_and(|kept| Arc::ptr_eq(&kept.worker, worker))
            {
                members.remove(worker.url())
            } else {
                None
            }
        };
        let Some(kept) = kept else {
            return;
        };
        // Loads are a row's. While an object stands under the row, the same
        // or its replacement, nothing is evicted: an eviction here could
        // wipe a replacement's first record. A row that went has no object.
        if self.leads() && self.objects.registry().get(&kept.id).is_none() {
            self.store.evict(kept.member.row);
        }
    }

    fn reset(&self) {
        // The store's rows stay. Each ends with the write that removes its
        // worker, so no row of a worker the collector lost stands there,
        // and a reset at a term's start would wipe the loads a standby
        // followed.
        self.members.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use std::slice;

    use parking_lot::Mutex;
    use smg_worker::{
        load::port::GroupTick,
        registry::WorkerRegistry,
        worker::{SchedulerLoadSnapshot, WorkerStatus as ObjectStatus},
    };

    use super::*;
    use crate::{
        store::{
            Ask, GatewayId, LoadSample, LoadSource, MemoryStore, Overload,
            PollMode as StorePollMode, ViewBuffers, WorkerFacts, WorkerPatch, WorkerStatus,
            WorkersStore,
        },
        view::{LoadRow, Role, SignalSource, WireModes, WorkerId},
        worker_host::{object, Members, SoleLead},
    };

    const URL: &str = "http://loads:8000";
    const MODEL: &str = "m";
    const PERIOD: Duration = Duration::from_secs(2);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct Calls {
        records: usize,
        reports: usize,
        polls: usize,
        evicts: usize,
        periods: usize,
        period: Option<Duration>,
    }

    /// A real store that also counts calls which might change no row.
    struct Store {
        memory: MemoryStore,
        calls: Mutex<Calls>,
    }

    impl StoreLoadSink for Store {
        fn record(
            &self,
            id: WorkerId,
            rank: u16,
            row: LoadRow,
            received_at: Instant,
            source: LoadSource,
        ) {
            self.calls.lock().records += 1;
            self.memory.record(id, rank, row, received_at, source);
        }

        fn report(
            &self,
            answer: &[(WorkerId, u16, Option<LoadRow>)],
            polled_at: Instant,
            source: LoadSource,
        ) -> Vec<usize> {
            self.calls.lock().reports += 1;
            self.memory.report(answer, polled_at, source)
        }

        fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> StorePollMode {
            self.calls.lock().polls += 1;
            self.memory.poll_mode(id, now, interval)
        }

        fn heard_every(&self, period: Duration) -> bool {
            let mut calls = self.calls.lock();
            calls.periods += 1;
            calls.period = Some(period);
            self.memory.heard_every(period)
        }

        fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample> {
            self.memory.load(id, rank)
        }

        fn ranks(&self, id: WorkerId) -> usize {
            self.memory.ranks(id)
        }

        fn evict(&self, id: WorkerId) {
            self.calls.lock().evicts += 1;
            self.memory.evict(id);
        }
    }

    struct Replica {
        objects: Arc<Objects<Members>>,
        store: Arc<Store>,
    }

    impl Replica {
        fn new() -> Self {
            Self {
                objects: Arc::new(Objects::new(Arc::new(WorkerRegistry::new()))),
                store: Arc::new(Store {
                    memory: MemoryStore::with_publish_window(Duration::ZERO),
                    calls: Mutex::default(),
                }),
            }
        }

        fn adapter(
            &self,
            lead: Arc<dyn Lead>,
            term: u64,
        ) -> LoadAdapter<Members, dyn StoreLoadSink> {
            LoadAdapter::new(
                Arc::clone(&self.objects),
                self.store.clone(),
                lead,
                term,
                PERIOD,
            )
        }

        fn lead(&self) -> LoadAdapter<Members, dyn StoreLoadSink> {
            self.adapter(Arc::new(SoleLead), SoleLead::TERM)
        }

        fn add(&self, facts: WorkerFacts) -> (WorkerId, Arc<dyn Worker>) {
            let id = self.store.memory.upsert(facts).id();
            let row = WorkersStore::get(&self.store.memory, id).expect("registered row");
            self.store
                .memory
                .set_status(id, WorkerStatus::Ready, row.revision);
            self.store.memory.publish_now();
            self.objects.follow(&self.store.memory);
            (id, self.at(&row.facts.key))
        }

        fn at(&self, url: &str) -> Arc<dyn Worker> {
            self.objects.registry().get_by_url(url).expect("an object")
        }

        fn replace(&self, id: WorkerId) -> Arc<dyn Worker> {
            let row = WorkersStore::get(&self.store.memory, id).expect("a row");
            let patch = WorkerPatch {
                priority: Some(20),
                ..Default::default()
            };
            assert!(self.store.memory.patch(id, patch, row.revision).is_some());
            self.objects.follow(&self.store.memory);
            self.at(&row.facts.key)
        }

        fn two_ranks(&self, waiting_requests: Option<u32>) -> [(WorkerId, Arc<dyn Worker>); 2] {
            [0, 1].map(|rank| {
                self.add(WorkerFacts {
                    key: format!("{URL}@{rank}"),
                    dp_size: 2,
                    dp_rank: rank,
                    overload: Overload {
                        waiting_requests,
                        ..Default::default()
                    },
                    ..facts(URL)
                })
            })
        }

        fn overloaded(&self, id: WorkerId, now: Instant) -> bool {
            let mut out = ViewBuffers::default();
            let ask = Ask {
                gateway: GatewayId(1),
                now,
                pinned: None,
                affinity: false,
            };
            self.store
                .memory
                .world()
                .build(MODEL, Role::Single, &ask, &mut out)
                .expect("the pool");
            out.candidates
                .iter()
                .find(|row| row.id == id)
                .expect("a candidate")
                .health
                .overloaded
        }
    }

    fn facts(url: &str) -> WorkerFacts {
        WorkerFacts {
            key: url.to_string(),
            engine: url.to_string(),
            incarnation: 1,
            models: vec![MODEL.to_string()],
            dp_size: 1,
            wire_modes: WireModes::GRPC,
            ..Default::default()
        }
    }

    fn report(rank: i32, running: i32) -> WorkerLoadResponse {
        WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: rank,
                num_running_reqs: running,
                num_waiting_reqs: running,
                gen_throughput: 30.0,
                prefill_throughput: Some(900.0),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn polled<'a>(
        worker: &'a Arc<dyn Worker>,
        answer: Option<&'a WorkerLoadResponse>,
    ) -> PolledMember<'a> {
        PolledMember {
            worker,
            answer,
            on_file: None,
        }
    }

    fn busy_pair(
        replica: &Replica,
        adapter: &impl LoadSink,
        now: Instant,
    ) -> [(WorkerId, Arc<dyn Worker>); 2] {
        let pair = replica.two_ranks(Some(5));
        adapter.record(&pair[0].1, 0, &report(0, 6), now, now);
        adapter.record(&pair[1].1, 1, &report(1, 1), now, now);
        for (id, worker) in &pair {
            assert!(replica.overloaded(*id, now));
            assert!(!worker.is_overloaded());
        }
        pair
    }

    #[test]
    fn a_record_writes_the_ranks_row_with_the_collectors_sample_instant() {
        for (role, rate) in [
            (Role::Single, 30),
            (Role::Prefill, 900),
            (Role::Decode, 30),
            (Role::Encode, 30),
        ] {
            let replica = Replica::new();
            let (id, worker) = replica.add(WorkerFacts {
                dp_rank: 1,
                dp_size: 2,
                role,
                ..facts(URL)
            });
            let adapter = replica.lead();
            let sampled = Instant::now();
            let received = sampled + Duration::from_millis(37);
            adapter.record(&worker, 1, &report(1, 5), received, sampled);
            let row = replica.store.load(id, 1).expect("the pushed row");
            assert_eq!((row.received_at, row.sampled_at), (received, sampled));
            assert_eq!(row.row.report_age_ms, 37);
            assert_eq!(
                (row.source, row.row.source),
                (LoadSource::Stream, SignalSource::Events)
            );
            assert_eq!((row.row.running, row.row.live_rate), (5, rate));
            assert_eq!(
                adapter.member(&worker).map(|member| member.role),
                Some(role)
            );
            assert_eq!(replica.store.ranks(id), 1);
        }
    }

    #[test]
    fn a_record_uses_its_given_rank_if_it_fits() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        let last = i32::from(u16::MAX);
        adapter.record(&worker, last, &report(last, 2), now, now);
        for rank in [-1, last + 1] {
            adapter.record(&worker, rank, &report(rank, 9), now, now);
        }
        assert_eq!(replica.store.calls.lock().records, 1);
        assert_eq!(
            replica
                .store
                .load(id, u16::MAX)
                .map(|sample| sample.row.running),
            Some(2)
        );
        assert_eq!(replica.store.ranks(id), 1);
    }

    #[test]
    fn a_record_of_a_replaced_object_is_passed_over() {
        let replica = Replica::new();
        let (id, old) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&old, 0, &report(0, 1), now, now);
        let before = replica.store.load(id, 0).expect("the old row");

        let replacement = replica.replace(id);
        adapter.record(&old, 0, &report(0, 9), now, now);
        assert_eq!(replica.store.load(id, 0), Some(before));

        adapter.record(&replacement, 0, &report(0, 2), now, now);
        adapter.record(&old, 0, &report(0, 9), now, now);
        assert_eq!(
            replica.store.load(id, 0).map(|sample| sample.row.running),
            Some(2)
        );
        assert_eq!(replica.store.calls.lock().records, 2);

        let (incarnation, _) = replica.add(WorkerFacts {
            incarnation: 2,
            ..facts(URL)
        });
        assert_ne!(incarnation, id);
        adapter.record(&replacement, 0, &report(0, 9), now, now);
        assert_eq!(
            adapter.poll_mode(&replacement, now, PERIOD),
            PollMode::Suppressed
        );
        assert_eq!(replica.store.calls.lock().records, 2);
        assert_eq!(replica.store.ranks(id), 0);
        assert_eq!(replica.store.ranks(incarnation), 0);
    }

    #[test]
    fn a_record_of_a_worker_with_no_row_is_passed_over() {
        let replica = Replica::new();
        let (id, paired) = replica.add(facts(URL));
        let row = WorkersStore::get(&replica.store.memory, id).expect("a row");
        let unpaired = object(&row);
        let adapter = replica.lead();
        let now = Instant::now();
        let load = report(0, 9);
        let before = *replica.store.calls.lock();
        adapter.record(&unpaired, 0, &load, now, now);
        assert_eq!(
            adapter.poll_mode(&unpaired, now, PERIOD),
            PollMode::Suppressed
        );
        assert_eq!(
            adapter.report(&[polled(&unpaired, Some(&load))], now),
            PolledRanks::default()
        );
        assert!(adapter
            .report_empty(&[polled(&unpaired, None)], now)
            .is_empty());
        adapter.fetch_failed(slice::from_ref(&unpaired), now);
        adapter.forget(&unpaired);
        assert_eq!(*replica.store.calls.lock(), before);
        assert_eq!(replica.store.ranks(id), 0);
        assert!(adapter.members.read().is_empty());

        let registry = replica.objects.registry();
        let object_id = registry.get_id_by_url(URL).expect("a registry id");
        assert!(registry.replace(&object_id, unpaired));
        adapter.record(&paired, 0, &load, now, now);
        assert_eq!(*replica.store.calls.lock(), before);
    }

    #[test]
    fn the_poll_question_is_the_stores() {
        let replica = Replica::new();
        let (_, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        assert_eq!(adapter.poll_mode(&worker, now, PERIOD), PollMode::Poll);
        adapter.record(&worker, 0, &report(0, 2), now, now);
        assert_eq!(
            adapter.poll_mode(&worker, now, PERIOD),
            PollMode::Suppressed
        );
        let later = now + PERIOD + Duration::from_millis(1);
        assert_eq!(
            adapter.poll_mode(&worker, later, PERIOD),
            PollMode::Fallback
        );
    }

    #[test]
    fn a_polls_answer_is_one_write_and_names_the_ranks_whose_stream_is_newer() {
        let replica = Replica::new();
        let [(zero, first), (one, second)] = replica.two_ranks(None);
        let adapter = replica.lead();
        let asked = Instant::now();
        let pushed = asked + Duration::from_millis(1);
        adapter.record(&second, 1, &report(1, 7), pushed, pushed);
        let stream = replica.store.load(one, 1).expect("stream row");
        let mut first_answer = report(0, 3);
        first_answer.sampled_at = Some(asked - PERIOD);
        let second_answer = report(1, 4);
        let unpaired = object(&WorkersStore::get(&replica.store.memory, zero).expect("a row"));
        let ranks = adapter.report(
            &[
                polled(&unpaired, None),
                polled(&second, Some(&second_answer)),
                polled(&first, Some(&first_answer)),
            ],
            asked,
        );
        assert_eq!(ranks.newer, [1]);
        assert!(ranks.leaving.is_empty());
        assert_eq!(replica.store.calls.lock().reports, 1);
        assert_eq!(replica.store.load(one, 1), Some(stream));
        let sample = replica.store.load(zero, 0).expect("the poll's row");
        assert_eq!((sample.row.running, sample.row.report_age_ms), (3, 0));
        assert_eq!((sample.sampled_at, sample.received_at), (asked, asked));
        assert_eq!(adapter.poll_mode(&first, asked, PERIOD), PollMode::Poll);
    }

    #[test]
    fn a_rank_the_answer_leaves_out_keeps_its_row_and_no_rank_leaves_the_file() {
        let replica = Replica::new();
        let [(zero, first), (one, second)] = replica.two_ranks(None);
        let adapter = replica.lead();
        let now = Instant::now();
        let old = report(1, 7);
        adapter.record(&second, 1, &old, now, now);
        let before = replica.store.load(one, 1).expect("row on file");
        let asked = now + PERIOD;
        let empty = WorkerLoadResponse::default();
        let absent = PolledMember {
            on_file: Some(&old),
            ..polled(&second, Some(&empty))
        };
        let ranks = adapter.report(&[polled(&first, Some(&report(0, 3))), absent], asked);
        assert_eq!(ranks, PolledRanks::default());
        assert_eq!(replica.store.load(one, 1), Some(before));
        assert!(!replica.overloaded(zero, asked));
        assert!(replica.overloaded(one, asked));
    }

    #[test]
    fn an_empty_answer_changes_nothing_on_file_and_answers_no_rank() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now();
        let [(zero, first), (one, second)] = busy_pair(&replica, &adapter, now);
        let rows = [replica.store.load(zero, 0), replica.store.load(one, 1)];
        let empty = WorkerLoadResponse::default();
        let asked = now + PERIOD;
        assert!(adapter
            .report_empty(
                &[polled(&first, Some(&empty)), polled(&second, None)],
                asked
            )
            .is_empty());
        assert_eq!(
            [replica.store.load(zero, 0), replica.store.load(one, 1)],
            rows
        );
        assert!(!replica.overloaded(zero, asked));
        assert!(!replica.overloaded(one, asked));
        assert_eq!(replica.store.calls.lock().reports, 1);
    }

    #[test]
    fn an_empty_answer_of_a_poll_that_left_a_rank_out_says_nothing() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now();
        let [(zero, first), (one, second)] = busy_pair(&replica, &adapter, now);
        let asked = now + PERIOD;
        assert!(adapter
            .report_empty(&[polled(&second, None)], asked)
            .is_empty());
        assert_eq!(replica.store.calls.lock().reports, 0);
        assert!(replica.overloaded(zero, asked));
        assert!(replica.overloaded(one, asked));
        assert_eq!(adapter.poll_mode(&first, asked, PERIOD), PollMode::Fallback);
    }

    #[test]
    fn a_failed_fetch_is_the_word_that_no_report_came() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now() - PERIOD;
        let [(zero, first), (one, second)] = busy_pair(&replica, &adapter, now);
        let rows = [replica.store.load(zero, 0), replica.store.load(one, 1)];
        let asked = now + Duration::from_millis(1);
        adapter.fetch_failed(slice::from_ref(&second), asked);
        assert_eq!(replica.store.calls.lock().reports, 0);
        assert!(replica.overloaded(one, Instant::now()));
        adapter.fetch_failed(&[first, second], asked);
        assert_eq!(replica.store.calls.lock().reports, 1);
        assert_eq!(
            [replica.store.load(zero, 0), replica.store.load(one, 1)],
            rows
        );
        assert!(!replica.overloaded(one, Instant::now()));
    }

    /// Loads are a row's, not an object's. A worker whose status left
    /// `Ready` is forgotten and its row stands: the adapter drops its entry
    /// and the row keeps its loads.
    #[test]
    fn a_failed_fetch_leaves_the_verdict_of_a_push_during_its_poll() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let asked = Instant::now() - PERIOD;
        let pushed = asked + Duration::from_millis(1);
        let [(zero, first), (one, second)] = busy_pair(&replica, &adapter, pushed);
        adapter.fetch_failed(&[first, second], asked);
        assert_eq!(replica.store.calls.lock().reports, 1);
        assert!(replica.overloaded(zero, pushed));
        assert!(replica.overloaded(one, pushed));
    }

    #[test]
    fn a_forgotten_worker_whose_object_stands_keeps_its_rows_loads() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 2), now, now);
        worker.set_status(ObjectStatus::NotReady);
        adapter.forget(&worker);
        assert_eq!(replica.store.calls.lock().evicts, 0);
        assert_eq!(replica.store.ranks(id), 1);
        assert!(adapter.members.read().is_empty());
    }

    /// A worker whose row went leaves the registry. Its forget evicts the
    /// row: nothing of it is on file.
    #[test]
    fn a_forgotten_worker_whose_row_went_has_nothing_on_file() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 2), now, now);
        let row = WorkersStore::get(&replica.store.memory, id).expect("a row");
        assert!(replica.store.memory.remove(id, row.revision).is_some());
        replica.objects.follow(&replica.store.memory);
        adapter.forget(&worker);
        assert_eq!(replica.store.calls.lock().evicts, 1);
        assert_eq!(replica.store.ranks(id), 0);
        assert!(adapter.members.read().is_empty());
    }

    /// The holder replaced the object under a row that stands. The row's
    /// loads are the replacement's now: the old object's forget drops its
    /// entry and evicts nothing, so no order of the two can wipe a record
    /// of the replacement.
    #[test]
    fn a_forget_of_a_replaced_object_leaves_the_rows_loads_to_its_replacement() {
        let replica = Replica::new();
        let (id, old) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&old, 0, &report(0, 3), now, now);
        let _replacement = replica.replace(id);
        adapter.forget(&old);
        assert_eq!(replica.store.calls.lock().evicts, 0);
        assert_eq!(
            replica.store.load(id, 0).map(|sample| sample.row.running),
            Some(3)
        );
        assert!(adapter.members.read().is_empty());
    }

    #[test]
    fn a_late_forget_of_an_old_object_keeps_the_replacements_load() {
        let replica = Replica::new();
        let (id, old) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&old, 0, &report(0, 1), now, now);
        let replacement = replica.replace(id);
        adapter.record(&replacement, 0, &report(0, 2), now, now);
        adapter.forget(&old);
        assert_eq!(
            replica.store.load(id, 0).map(|sample| sample.row.running),
            Some(2)
        );
        assert_eq!(adapter.members.read().len(), 1);
    }

    #[test]
    fn a_reset_drops_the_cache_and_leaves_the_stores_rows() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 5), now, now);
        assert_eq!(adapter.members.read().len(), 1);
        let before = replica.store.load(id, 0).expect("the row");
        adapter.reset();
        assert!(adapter.members.read().is_empty());
        assert_eq!(replica.store.load(id, 0), Some(before));
        assert_eq!(
            adapter.poll_mode(&worker, now, PERIOD),
            PollMode::Suppressed
        );
    }

    #[test]
    fn a_members_wire_picks_the_source() {
        for (wire_modes, source, signal) in [
            (WireModes::GRPC, LoadSource::Poll, SignalSource::Poll),
            (WireModes::HTTP, LoadSource::Scrape, SignalSource::Scrape),
            (WireModes::ZMQ, LoadSource::Scrape, SignalSource::Scrape),
        ] {
            let replica = Replica::new();
            let (id, worker) = replica.add(WorkerFacts {
                wire_modes,
                ..facts(URL)
            });
            let adapter = replica.lead();
            adapter.report(&[polled(&worker, Some(&report(0, 3)))], Instant::now());
            let load = replica.store.load(id, 0).expect("a sampled row");
            assert_eq!((load.source, load.row.source), (source, signal));
        }
    }

    struct TestLead(Mutex<Option<u64>>);

    impl Lead for TestLead {
        fn term(&self) -> Option<u64> {
            *self.0.lock()
        }
    }

    #[test]
    fn a_replica_that_follows_writes_nothing_and_asks_no_engine() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let lead = Arc::new(TestLead(Mutex::new(Some(1))));
        let adapter = replica.adapter(lead.clone(), 1);
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 1), now, now);
        let saved = replica.store.load(id, 0);
        let before = *replica.store.calls.lock();
        *lead.0.lock() = None;

        let load = report(0, 9);
        let later = now + PERIOD;
        assert_eq!(
            adapter.poll_mode(&worker, later, PERIOD),
            PollMode::Suppressed
        );
        adapter.record(&worker, 0, &load, later, later);
        assert_eq!(
            adapter.report(&[polled(&worker, Some(&load))], later),
            PolledRanks::default()
        );
        assert!(adapter
            .report_empty(&[polled(&worker, None)], later)
            .is_empty());
        adapter.fetch_failed(slice::from_ref(&worker), later);
        adapter.forget(&worker);
        assert!(adapter.members.read().is_empty());
        adapter.reset();
        assert!(adapter.members.read().is_empty());
        assert!(adapter.needs_loads());
        adapter.tick(&GroupTick {
            workers: slice::from_ref(&worker),
            reports: &[],
        });
        assert_eq!(*replica.store.calls.lock(), before);
        assert_eq!(replica.store.load(id, 0), saved);
    }

    #[test]
    fn an_adapter_of_an_earlier_term_writes_nothing() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let lead = Arc::new(TestLead(Mutex::new(Some(1))));
        let adapter = replica.adapter(lead.clone(), 1);
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 1), now, now);
        let before = replica
            .store
            .load(id, 0)
            .expect("the adapter wrote under term 1");
        *lead.0.lock() = Some(2);

        let later = now + PERIOD;
        adapter.record(&worker, 0, &report(0, 9), later, later);
        adapter.report(&[polled(&worker, Some(&report(0, 9)))], later);
        adapter.forget(&worker);
        assert_eq!(
            adapter.poll_mode(&worker, later, PERIOD),
            PollMode::Suppressed
        );
        assert_eq!(replica.store.load(id, 0), Some(before));
        assert_eq!(replica.store.calls.lock().records, 1);
    }

    /// An adapter whose term ended still drops a forgotten worker's entry:
    /// the entry holds the worker object, and with it the object's engine
    /// connection. It evicts nothing, since it writes nothing.
    #[test]
    fn a_forget_drops_the_entry_whatever_the_term() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let lead = Arc::new(TestLead(Mutex::new(Some(1))));
        let adapter = replica.adapter(lead.clone(), 1);
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 1), now, now);
        let held = Arc::strong_count(&worker);
        *lead.0.lock() = Some(2);

        adapter.forget(&worker);
        assert!(adapter.members.read().is_empty());
        assert_eq!(Arc::strong_count(&worker), held - 1);
        assert_eq!(replica.store.calls.lock().evicts, 0);
        assert_eq!(replica.store.ranks(id), 1);
    }

    /// The store keeps the first period it is told. An adapter built with
    /// another one reads that its period did not stand.
    #[test]
    fn an_adapter_says_whether_the_store_took_its_period() {
        let replica = Replica::new();
        assert!(replica.lead().period_taken());
        let other = LoadAdapter::new(
            Arc::clone(&replica.objects),
            replica.store.clone(),
            Arc::new(SoleLead),
            SoleLead::TERM,
            PERIOD * 2,
        );
        assert!(!other.period_taken());
        assert!(replica.lead().period_taken(), "the first period again");
    }

    #[test]
    fn the_store_is_told_once_how_often_a_rank_is_heard() {
        let replica = Replica::new();
        let (_, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        assert_eq!(replica.store.calls.lock().period, Some(PERIOD));
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 4), now, now);
        adapter.poll_mode(&worker, now, PERIOD);
        adapter.report(&[polled(&worker, Some(&report(0, 3)))], now + PERIOD);
        adapter.report_empty(&[polled(&worker, None)], now + PERIOD);
        adapter.fetch_failed(slice::from_ref(&worker), now + PERIOD);
        adapter.forget(&worker);
        adapter.reset();
        assert_eq!(replica.store.calls.lock().periods, 1);
    }
}
