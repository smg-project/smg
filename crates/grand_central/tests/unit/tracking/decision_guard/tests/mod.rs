use std::{
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::{identity::basic::Worker, test_support::ready_worker};

use super::DecisionGuard;
#[cfg(feature = "client")]
use crate::engine::decision_engine::{DecisionRefused, DecodeRefused};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, Target, Targets},
        decision_engine::{Request, RequestMaker, WorkerLists},
        framework::{Picker, TieBand},
        in_process::InProcess,
        list_cell::PublishedList,
        view::{Candidate, Constraints, RequestFacts, WorkerId},
    },
    policies::random::Random,
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::GatewayId,
    tracking::lease::{EventSet, Lease, Outcome, ReleaseReason, Side, TransportClass},
};

/// The model of the list's workers.
const MODEL: &str = "model-a";

/// How long a case waits for an answer it expects back. The cases of the
/// client's link alone read it, so it stands under their feature.
#[cfg(feature = "client")]
const PATIENCE: Duration = Duration::from_secs(20);

/// What a stage heard of a lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Issued,
    /// The target the dispatch named, its index on the lease told, and
    /// the microseconds since the decision.
    Dispatched(WorkerId, u8, u32),
    /// The served worker, the class and the microseconds the dispatch
    /// took.
    Completed(WorkerId, TransportClass, u32),
    Released(ReleaseReason),
}

type Heard = Arc<Mutex<Vec<Call>>>;

/// The `random` picker, keeping what the engine tells it of leases.
#[derive(Debug)]
struct Hears(Heard);

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        self.0.lock().push(Call::Issued);
    }

    fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, since: u32) {
        let call = Call::Dispatched(target.id, lease.target_index, since);
        self.0.lock().push(call);
    }

    fn on_completed(&self, _: &Lease, outcome: &Outcome) {
        let call = Call::Completed(outcome.served, outcome.class, outcome.duration_micros);
        self.0.lock().push(call);
    }

    fn on_released(&self, _: &Lease, reason: ReleaseReason) {
        self.0.lock().push(Call::Released(reason));
    }
}

impl Picker for Hears {
    fn name(&self) -> &'static str {
        "hears"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The engine over worker lists under the `random` preset with
/// [`Hears`] in its pick's place, the list of four ready workers its
/// host hands it, and what the picker heard.
struct Host {
    engine: InProcess,
    members: PublishedList,
    heard: Heard,
}

impl Host {
    /// One decision over the host's list for the request `id`, held.
    fn held(&self, id: &[u8], now: Instant) -> DecisionGuard<&InProcess> {
        let asks = Listed(stated(id), Some(&self.members));
        let decided = DecisionGuard::decide_under(&self.engine, GatewayId::LOCAL, &asks, now);
        decided.expect("a decision")
    }
}

/// A host with an engine, a list and a hearing picker of its own.
fn host() -> Host {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("hears", move || Box::new(Hears(Arc::clone(&kept))));
    added.expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = 1;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "hears".into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = InProcess::new(spec.expect("the stages hold the picker"));
    let list: Arc<[Arc<dyn Worker>]> = (0..4)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect();
    let members = PublishedList::new(list);
    Host {
        engine,
        members,
        heard,
    }
}

/// A request for [`MODEL`] under `id`, stated ahead.
fn stated(id: &[u8]) -> Request<'_> {
    Request {
        id,
        facts: RequestFacts {
            model: MODEL,
            seed: 7,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    }
}

/// A request stated ahead, with the list its host hands for it.
struct Listed<'a>(Request<'a>, Option<&'a PublishedList>);

impl RequestMaker for Listed<'_> {
    fn model(&self) -> &str {
        self.0.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.0
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.1
            .map_or(WorkerLists::None, |list| WorkerLists::Single(list.lent()))
    }
}

/// An outcome of `class` on `served`, as a host reads one.
fn outcome(served: WorkerId, class: TransportClass) -> Outcome {
    Outcome {
        side: Side::Single,
        served,
        status: 200,
        class,
        ttft_micros: 1_000,
        tokens_in: 12,
        tokens_out: 34,
        duration_micros: 5_000,
    }
}

/// What was heard, with a cancelled completion's time read as 0: a
/// clock made it, and it is `least` microseconds at least.
fn clocked(heard: &Heard, least: u32) -> Vec<Call> {
    let heard = heard.lock();
    let calls = heard.iter().map(|call| match *call {
        Call::Completed(served, TransportClass::Cancelled, took) => {
            assert!(took >= least, "a cancelled request took {took}");
            Call::Completed(served, TransportClass::Cancelled, 0)
        }
        said => said,
    });
    calls.collect()
}

/// A host embeds the handle in the context it allocates once a request,
/// and fits that context to its allocator's class. A handle is the size
/// it has in a build with no link: the link's arm rides behind a
/// pointer, and the door's own pointer tells the arms apart. A field
/// that grows the handle grows each such context, so its hosts are
/// read before this number moves.
#[test]
fn a_handle_is_the_size_it_has_in_a_build_with_no_link() {
    const HELD: usize = 504;
    assert_eq!(size_of::<DecisionGuard<Arc<InProcess>>>(), HELD);
    assert_eq!(size_of::<DecisionGuard<&InProcess>>(), HELD);
    // A placement is the handle and what its host reads of the answer:
    // two worker pointers, the routing key, the profile's label, the
    // label of a pair's decode side, one pointer to an encode request's
    // encoders, and what its pick states of the worker taken: six
    // bytes, in a word of their own.
    assert_eq!(size_of::<crate::tracking::Placement<'_>>(), HELD + 96);
    // A host boxes a single request's side once, and that side is its
    // placement with three words of its own: the box stays within the
    // 640 bytes of the allocation class it had.
    assert!(size_of::<crate::tracking::PlacementSide>() <= 640);
    // The lease the handle keeps states its request's text's two
    // lengths beside its token count: eight bytes more, the handle's
    // too. The second length sits in the padding the first left. The
    // handle keeps the two lengths the decode side's words state
    // beside that side's token count: eight bytes more.
    assert_eq!(size_of::<Lease>(), 72);
}

/// A request that is served: its lease is issued on the pick's first
/// target, dispatched at the host's word and ended at its completion.
/// Each is told once: what the host says after the end, and the drop,
/// say nothing.
#[test]
fn a_served_request_tells_its_dispatch_and_its_completion_once_each() {
    let host = host();
    let now = Instant::now();
    let mut handle = host.held(b"r-1", now);
    let issued = handle.lease().map(|leased| leased.lease.booked);
    let first = handle.take(0).expect("the pick's first target");
    assert_eq!(issued, Some(first.id), "issued on the pick's first target");
    handle.dispatched(now + Duration::from_micros(250));
    let ended = now + Duration::from_millis(5);
    handle.completed(&outcome(first.id, TransportClass::Ok), ended);
    let told = [
        Call::Issued,
        Call::Dispatched(first.id, 0, 250),
        Call::Completed(first.id, TransportClass::Ok, 5_000),
    ];
    assert_eq!(*host.heard.lock(), told);
    // Nothing more is owed: what its host says behind the end is told
    // to nobody.
    let later = ended + Duration::from_secs(1);
    handle.dispatched(later);
    handle.completed(&outcome(first.id, TransportClass::Ok), later);
    handle.released(ReleaseReason::ClientGone, Instant::now());
    drop(handle);
    assert_eq!(*host.heard.lock(), told, "one completion is told");
}

/// A host whose commit passes over the first target takes another: a
/// take says nothing, and the dispatch names the target taken under its
/// index in the pick. A take of no target of the pick, and one behind
/// the dispatch, change nothing.
#[test]
fn a_dispatch_on_an_alternate_names_the_target_taken() {
    let host = host();
    let now = Instant::now();
    let mut handle = host.held(b"r-1", now);
    let first = handle.take(0).expect("the pick's first target");
    let second = handle.take(1).expect("an alternate of the pick");
    assert_ne!(first.id, second.id);
    assert_eq!(handle.take(9), None, "no such target");
    assert_eq!(*host.heard.lock(), [Call::Issued], "a take says nothing");
    // Two hours behind the decision: more microseconds than the span's
    // field holds, so it says its most.
    handle.dispatched(now + Duration::from_secs(7_200));
    assert_eq!(handle.take(0), None, "the dispatch named its target");
    handle.completed(&outcome(second.id, TransportClass::Ok), now);
    let told = [
        Call::Issued,
        Call::Dispatched(second.id, 1, u32::MAX),
        Call::Completed(second.id, TransportClass::Ok, 5_000),
    ];
    assert_eq!(*host.heard.lock(), told);
}

/// A host whose commit refuses every target releases the lease under
/// that reason, once: nothing it says behind the release is told.
#[test]
fn a_commit_that_refuses_every_target_releases_the_lease_once() {
    let host = host();
    let now = Instant::now();
    let mut handle = host.held(b"r-1", now);
    let first = handle.take(0).expect("the pick's first target");
    handle.released(ReleaseReason::CommitFailed, Instant::now());
    let told = [Call::Issued, Call::Released(ReleaseReason::CommitFailed)];
    assert_eq!(*host.heard.lock(), told);
    handle.released(ReleaseReason::ClientGone, Instant::now());
    handle.completed(&outcome(first.id, TransportClass::Ok), now);
    drop(handle);
    assert_eq!(*host.heard.lock(), told, "no completion behind a release");
}

/// A handle dropped with no call knows no reason for its drop: before
/// any dispatch its lease is given up as abandoned, the word for a
/// lease nobody spoke for.
#[test]
fn a_handle_dropped_with_no_call_gives_its_lease_up_as_abandoned() {
    let host = host();
    let handle = host.held(b"r-1", Instant::now());
    assert_eq!(*host.heard.lock(), [Call::Issued]);
    drop(handle);
    let told = [Call::Issued, Call::Released(ReleaseReason::Abandoned)];
    assert_eq!(*host.heard.lock(), told);
}

/// A host that knows its client left before any dispatch says so, and
/// the handle's drop behind it says nothing more.
#[test]
fn a_host_that_knows_its_client_left_says_so_once() {
    let host = host();
    let mut handle = host.held(b"r-1", Instant::now());
    handle.released(ReleaseReason::ClientGone, Instant::now());
    let told = [Call::Issued, Call::Released(ReleaseReason::ClientGone)];
    assert_eq!(*host.heard.lock(), told);
    drop(handle);
    assert_eq!(*host.heard.lock(), told);
}

/// After a dispatch a lease lives until its completion: a handle
/// dropped with no outcome, and a release said there, are told as one
/// completion of class cancelled on the dispatched target, which says
/// the time since the dispatch.
#[test]
fn behind_a_dispatch_a_drop_and_a_release_are_a_cancelled_completion() {
    for released in [false, true] {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"r-1", now);
        let second = handle.take(1).expect("an alternate of the pick");
        handle.dispatched(now);
        thread::sleep(Duration::from_millis(1));
        let told = [
            Call::Issued,
            Call::Dispatched(second.id, 1, 0),
            Call::Completed(second.id, TransportClass::Cancelled, 0),
        ];
        if released {
            handle.released(ReleaseReason::ClientGone, Instant::now());
            assert_eq!(clocked(&host.heard, 1_000), told, "told where it was said");
        }
        drop(handle);
        assert_eq!(clocked(&host.heard, 1_000), told, "released: {released}");
    }
}

/// A request whose send failed was sent: a completion said with no
/// dispatch says the dispatch first, on the target taken and at the
/// completion's instant, so the engine hears the lease's events in
/// their order.
#[test]
fn a_completion_said_with_no_dispatch_says_the_dispatch_first() {
    let host = host();
    let now = Instant::now();
    let mut handle = host.held(b"r-1", now);
    let second = handle.take(1).expect("an alternate of the pick");
    let failed = outcome(second.id, TransportClass::ConnectFailed);
    handle.completed(&failed, now + Duration::from_micros(400));
    let told = [
        Call::Issued,
        Call::Dispatched(second.id, 1, 400),
        Call::Completed(second.id, TransportClass::ConnectFailed, 5_000),
    ];
    assert_eq!(*host.heard.lock(), told);
}

/// A decision that picks no worker has no lease: its handle takes no
/// target and tells nothing, whatever its host says.
#[test]
fn a_decision_that_picks_no_worker_owes_no_report() {
    let host = host();
    let now = Instant::now();
    // A worker of the list, by a decision that picks one.
    let mut picked = host.held(b"r-0", now);
    let member = picked.take(0).expect("the pick's first target").id;
    picked.released(ReleaseReason::CommitFailed, Instant::now());
    host.heard.lock().clear();
    // A request whose host hands no list is decided over no worker.
    let asks = Listed(stated(b"r-1"), None);
    let decided = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
    let mut handle = decided.expect("a decision");
    assert!(handle.decision().pick().is_none(), "no list, no pick");
    assert!(handle.lease().is_none());
    assert_eq!(handle.take(0), None);
    handle.dispatched(now);
    handle.completed(&outcome(member, TransportClass::Ok), now);
    handle.released(ReleaseReason::CommitFailed, Instant::now());
    drop(handle);
    assert_eq!(*host.heard.lock(), []);
}

/// A host that takes no target dispatches to the pick's first, and a
/// handle let go behind that dispatch completes the lease as cancelled
/// on that target. The completion's span counts from the dispatch: one
/// said ahead of the clock has no span yet.
#[test]
fn a_host_that_takes_no_target_dispatches_to_the_picks_first() {
    let host = host();
    let now = Instant::now();
    let mut handle = host.held(b"r-1", now);
    let pick = handle.decision().pick();
    let first = pick.and_then(|pick| pick.targets.first().copied());
    let first = first.expect("the pick's first target");
    handle.dispatched(now + Duration::from_secs(3_600));
    thread::sleep(Duration::from_millis(1));
    drop(handle);
    let told = [
        Call::Issued,
        Call::Dispatched(first.id, 0, 3_600_000_000),
        Call::Completed(first.id, TransportClass::Cancelled, 0),
    ];
    assert_eq!(*host.heard.lock(), told);
}

/// A handle that owns its engine travels with its request: dropped on
/// another thread behind its dispatch, it ends the lease from there.
#[test]
fn a_handle_that_owns_its_engine_ends_its_lease_from_another_thread() {
    let Host {
        engine,
        members,
        heard,
    } = host();
    let engine = Arc::new(engine);
    let now = Instant::now();
    let asks = Listed(stated(b"r-1"), Some(&members));
    let decided = DecisionGuard::decide_under(Arc::clone(&engine), GatewayId::LOCAL, &asks, now);
    let mut handle = decided.expect("a decision");
    let first = handle.take(0).expect("the pick's first target");
    handle.dispatched(now);
    let ended = thread::spawn(move || drop(handle));
    ended.join().expect("the request's thread");
    let told = [
        Call::Issued,
        Call::Dispatched(first.id, 0, 0),
        Call::Completed(first.id, TransportClass::Cancelled, 0),
    ];
    assert_eq!(clocked(&heard, 0), told);
}

/// Sixty-four hosts at once on one model, each holding its requests
/// from the decision to the end, by turns served, dropped behind the
/// dispatch, refused at the commit and dropped before any dispatch:
/// every lease is ended once, by its own path's word.
#[test]
fn sixty_four_hosts_at_once_end_every_lease_once() {
    let shared = host();
    let (hosts, each) = (64_u8, 32_u8);
    let start = Barrier::new(usize::from(hosts));
    thread::scope(|scope| {
        for host in 0..hosts {
            let (shared, start) = (&shared, &start);
            scope.spawn(move || {
                start.wait();
                for turn in 0..each {
                    let now = Instant::now();
                    let mut handle = shared.held(&[host, turn], now);
                    match turn % 4 {
                        0 => {
                            let first = handle.take(0).expect("the first target");
                            handle.dispatched(now);
                            handle.completed(&outcome(first.id, TransportClass::Ok), now);
                        }
                        1 => handle.dispatched(now),
                        2 => handle.released(ReleaseReason::CommitFailed, Instant::now()),
                        _ => {}
                    }
                }
            });
        }
    });
    let heard = shared.heard.lock();
    let told = |want: fn(&Call) -> bool| heard.iter().filter(|call| want(call)).count();
    let quarter = usize::from(hosts) * usize::from(each) / 4;
    let ok = |call: &Call| matches!(call, Call::Completed(_, TransportClass::Ok, _));
    let cancelled = |call: &Call| matches!(call, Call::Completed(_, TransportClass::Cancelled, _));
    let refused = |call: &Call| matches!(call, Call::Released(ReleaseReason::CommitFailed));
    let dropped = |call: &Call| matches!(call, Call::Released(ReleaseReason::Abandoned));
    assert_eq!(told(|call| matches!(call, Call::Issued)), 4 * quarter);
    assert_eq!(
        told(|call| matches!(call, Call::Dispatched(..))),
        2 * quarter
    );
    assert_eq!(
        (told(ok), told(cancelled), told(refused), told(dropped)),
        (quarter, quarter, quarter, quarter)
    );
}

/// A completion names the worker its host names, which need not be the
/// target the handle took: a stage reads the worker that served.
#[test]
fn a_completion_names_the_worker_its_host_names() {
    let host = host();
    let now = Instant::now();
    let mut handle = host.held(b"r-1", now);
    let other = handle.take(1).expect("the pick's second target");
    let taken = handle.take(0).expect("the pick's first target");
    assert_ne!(other.id, taken.id);
    handle.dispatched(now);
    handle.completed(&outcome(other.id, TransportClass::Ok), now);
    let told = [
        Call::Issued,
        Call::Dispatched(taken.id, 0, 0),
        Call::Completed(other.id, TransportClass::Ok, 5_000),
    ];
    assert_eq!(*host.heard.lock(), told);
}

/// A release behind a dispatch spans its host's two instants, the
/// dispatch's and its own: no clock is read.
#[test]
fn a_release_behind_a_dispatch_spans_its_hosts_instants() {
    let host = host();
    let now = Instant::now();
    let mut handle = host.held(b"r-1", now);
    let first = handle.take(0).expect("the pick's first target");
    let sent = now + Duration::from_secs(3_600);
    handle.dispatched(sent);
    handle.released(ReleaseReason::ClientGone, sent + Duration::from_millis(7));
    let told = [
        Call::Issued,
        Call::Dispatched(first.id, 0, 3_600_000_000),
        Call::Completed(first.id, TransportClass::Cancelled, 7_000),
    ];
    assert_eq!(*host.heard.lock(), told);
}

mod pairs;

/// A single's handle holds no decode side: a decode's take, dispatch
/// and release say nothing. Its lease ends under any side its host
/// names, a decode outcome among them: a decode phase can stand under
/// a single plan, and the outcome its host read reaches the engine.
#[test]
fn a_single_handle_ends_under_a_decode_outcome_and_says_nothing_else_of_a_decode_side() {
    let host = host();
    let now = Instant::now();
    let mut handle = host.held(b"single", now);
    let first = handle.take(0).expect("the first target");
    handle.dispatched_decode(now);
    assert_eq!(handle.take_decode(1), None);
    handle.released_decode(ReleaseReason::ClientGone, now);
    assert!(handle.lease().is_some(), "the single's lease stands");
    assert_eq!(*host.heard.lock(), [Call::Issued]);
    let decoded = Outcome {
        side: Side::Decode,
        ..outcome(first.id, TransportClass::Ok)
    };
    handle.completed(&decoded, now + Duration::from_micros(400));
    let told = [
        Call::Issued,
        Call::Dispatched(first.id, 0, 400),
        Call::Completed(first.id, TransportClass::Ok, 5_000),
    ];
    assert_eq!(*host.heard.lock(), told);
    drop(handle);
    assert_eq!(*host.heard.lock(), told, "the drop says nothing more");
}

#[cfg(feature = "client")]
mod linked {
    use std::pin::pin;

    use tokio::time::timeout;
    use tokio_stream::StreamExt as _;

    use super::{
        outcome, stated, DecisionGuard, DecisionRefused, DecodeRefused, Duration, InProcess,
        Instant, Request, Side, Target, TransportClass, WorkerId, PATIENCE,
    };
    use crate::{
        engine::{
            framework::{EncodeItems, EncodePlaced},
            view::{Identity, Phase},
            Demands, References,
        },
        service::{
            client::{poll, DecisionUnavailable, Error, Peer},
            proto::{
                self,
                completed::Class,
                controller_frame,
                decide_answer::Answer,
                decision::Outcome,
                gateway_frame::Frame,
                released::{Reason, What},
            },
        },
        tracking::ReleaseReason,
    };

    fn picked() -> proto::Decision {
        proto::Decision {
            leases: vec![proto::Lease {
                id: 73,
                kind: proto::LeaseKind::Credit.into(),
                side: proto::Side::Single.into(),
                dispatch_ttl_ms: 70,
                ..proto::Lease::default()
            }],
            outcome: Some(Outcome::Pick(proto::Pick {
                targets: [28, 19]
                    .into_iter()
                    .map(|id| proto::Target {
                        id,
                        dp_rank: 2,
                        pair: None,
                    })
                    .collect(),
            })),
            trace: None,
        }
    }

    async fn answered(
        peer: &mut Peer,
        told: proto::Decision,
        now: Instant,
    ) -> Result<DecisionGuard<&'static InProcess>, DecisionUnavailable> {
        let mut ask = stated(&[1]);
        ask.facts.budget_micros = 200_000;
        answered_to(peer, &ask, told, now).await
    }

    /// [`answered`] for the request `ask`, of id 1 and a budget of 200 ms.
    async fn answered_to(
        peer: &mut Peer,
        ask: &Request<'_>,
        told: proto::Decision,
        now: Instant,
    ) -> Result<DecisionGuard<&'static InProcess>, DecisionUnavailable> {
        peer.read_hello().await;
        let link = peer.link.clone();
        let exchange = async {
            let sent = peer.requests.next().await.expect("an ask went out");
            assert_eq!(sent.request_id, [1]);
            assert_eq!(sent.facts.expect("request facts").budget_micros, 200_000);
            peer.answer(1, link.term_floor(), Answer::Decision(told));
        };
        let both = async {
            let held = pin!(DecisionGuard::asked(&link, ask, now, PATIENCE));
            let (handle, ()) = tokio::join!(held, exchange);
            handle
        };
        timeout(PATIENCE, both).await.expect("an answer came back")
    }

    async fn reports(peer: &mut Peer, expected: impl IntoIterator<Item = Frame>) {
        for (index, frame) in expected.into_iter().enumerate() {
            let report = timeout(PATIENCE, peer.reports.next())
                .await
                .expect("a report was queued")
                .expect("the session stands");
            assert_eq!(
                report.seq,
                u64::try_from(index).expect("the sequence fits") + 2
            );
            assert_eq!(report.frame, Some(frame));
        }
        assert!(
            poll(pin!(peer.reports.next())).is_pending(),
            "no further report is owed"
        );
    }

    fn dispatch(index: u32, since: u32) -> Frame {
        Frame::Dispatched(proto::Dispatched {
            lease_id: 73,
            target_index: index,
            side: proto::Side::Single.into(),
            since_decision_micros: since,
        })
    }

    fn released(side: proto::Side, reason: Reason) -> Frame {
        Frame::Released(proto::Released {
            what: Some(What::LeaseId(73)),
            reason: reason.into(),
            side: side.into(),
        })
    }

    #[tokio::test(start_paused = true)]
    async fn a_served_link_request_reports_its_dispatch_and_completion_once() {
        let mut peer = Peer::new(1);
        let now = Instant::now();
        let mut handle = answered(&mut peer, picked(), now)
            .await
            .expect("the pick is held");
        assert_eq!(
            handle.lease().expect("a lease").due,
            now + Duration::from_millis(70)
        );
        let target = handle.take(1).expect("the pick's second target");
        assert_eq!(target.id, WorkerId(19));
        assert!(handle.take(2).is_none());
        let dispatched = now + Duration::from_micros(250);
        handle.dispatched(dispatched);
        assert!(handle.take(0).is_none());
        let complete = outcome(WorkerId(28), TransportClass::Ok);
        handle.completed(&complete, now + Duration::from_millis(5));
        handle.dispatched(now);
        handle.completed(&complete, now);
        handle.released(ReleaseReason::CommitFailed, now);
        drop(handle);
        reports(
            &mut peer,
            [
                dispatch(1, 250),
                Frame::Completed(proto::Completed {
                    lease_id: 73,
                    side: proto::Side::Single.into(),
                    served: 28,
                    status: 200,
                    class: Class::Ok.into(),
                    ttft_micros: 1_000,
                    tokens_in: 12,
                    tokens_out: 34,
                    duration_micros: 5_000,
                }),
            ],
        )
        .await;
    }

    /// A request can outlast the session its decision came over. Its
    /// handle holds the gateway's link, so what it reports then waits
    /// for the next session and rides it.
    #[tokio::test(start_paused = true)]
    async fn a_request_that_outlasts_its_session_reports_on_the_next_one() {
        let mut peer = Peer::patient(1);
        let now = Instant::now();
        let mut handle = answered(&mut peer, picked(), now)
            .await
            .expect("the pick is held");
        let target = handle.take(1).expect("the pick's second target");
        assert_eq!(target.id, WorkerId(19));
        peer.end_session().await;
        handle.dispatched(now + Duration::from_micros(250));

        drop(peer.redial(1));
        peer.read_hello().await;
        let complete = outcome(WorkerId(28), TransportClass::Ok);
        handle.completed(&complete, now + Duration::from_millis(5));
        drop(handle);
        reports(
            &mut peer,
            [
                dispatch(1, 250),
                Frame::Completed(proto::Completed {
                    lease_id: 73,
                    side: proto::Side::Single.into(),
                    served: 28,
                    status: 200,
                    class: Class::Ok.into(),
                    ttft_micros: 1_000,
                    tokens_in: 12,
                    tokens_out: 34,
                    duration_micros: 5_000,
                }),
            ],
        )
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_link_reports_when_every_commit_is_refused_or_a_client_leaves() {
        let now = Instant::now();
        for (reason, wire) in [
            (ReleaseReason::CommitFailed, Reason::CommitFailed),
            (ReleaseReason::ClientGone, Reason::ClientGone),
        ] {
            let mut peer = Peer::new(1);
            let mut handle = answered(&mut peer, picked(), now)
                .await
                .expect("the pick is held");
            handle.released(reason, now);
            handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
            drop(handle);
            reports(&mut peer, [released(proto::Side::Single, wire)]).await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_a_link_handle_before_dispatch_reports_abandonment() {
        let mut peer = Peer::new(1);
        let handle = answered(&mut peer, picked(), Instant::now())
            .await
            .expect("the pick is held");
        drop(handle);
        reports(
            &mut peer,
            [released(proto::Side::Single, Reason::Abandoned)],
        )
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_or_releasing_a_link_handle_after_dispatch_reports_cancellation() {
        for release in [false, true] {
            let mut peer = Peer::new(1);
            let now = Instant::now();
            let mut handle = answered(&mut peer, picked(), now)
                .await
                .expect("the pick is held");
            let sent = now + Duration::from_secs(3_600);
            handle.dispatched(sent);
            if release {
                handle.released(ReleaseReason::ClientGone, sent + Duration::from_millis(7));
            }
            drop(handle);
            reports(
                &mut peer,
                [
                    dispatch(0, 3_600_000_000),
                    Frame::Completed(proto::Completed {
                        lease_id: 73,
                        side: proto::Side::Single.into(),
                        served: 28,
                        status: 0,
                        class: Class::Cancelled.into(),
                        ttft_micros: 0,
                        tokens_in: 0,
                        tokens_out: 0,
                        duration_micros: if release { 7_000 } else { 0 },
                    }),
                ],
            )
            .await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_link_completion_supplies_the_dispatch_when_the_send_failed() {
        let mut peer = Peer::new(1);
        let now = Instant::now();
        let mut handle = answered(&mut peer, picked(), now)
            .await
            .expect("the pick is held");
        let target = handle.take(1).expect("the pick's second target");
        handle.completed(
            &outcome(target.id, TransportClass::ConnectFailed),
            now + Duration::from_micros(400),
        );
        drop(handle);
        reports(
            &mut peer,
            [
                dispatch(1, 400),
                Frame::Completed(proto::Completed {
                    lease_id: 73,
                    side: proto::Side::Single.into(),
                    served: 19,
                    status: 200,
                    class: Class::ConnectFailed.into(),
                    ttft_micros: 1_000,
                    tokens_in: 12,
                    tokens_out: 34,
                    duration_micros: 5_000,
                }),
            ],
        )
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_link_decision_without_a_lease_reports_nothing() {
        let mut peer = Peer::new(1);
        let now = Instant::now();
        let told = proto::Decision {
            outcome: Some(Outcome::Shed(proto::Shed {
                reason: proto::ShedReason::NoEndpoints.into(),
                ..proto::Shed::default()
            })),
            ..proto::Decision::default()
        };
        let mut handle = answered(&mut peer, told, now)
            .await
            .expect("the shed is held");
        assert!(handle.lease().is_none());
        assert!(handle.take(0).is_none());
        handle.dispatched(now);
        handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
        handle.released(ReleaseReason::CommitFailed, now);
        drop(handle);
        reports(&mut peer, []).await;
    }

    #[tokio::test(start_paused = true)]
    async fn an_unreadable_answer_is_unsent_and_its_lease_is_released_late() {
        let missing = proto::Decision {
            outcome: None,
            ..picked()
        };
        let mut unknown = picked();
        unknown.leases[0].side = i32::MAX;
        for (told, path, side) in [
            (missing, "outcome:", proto::Side::Single),
            (unknown, "leases.side:", proto::Side::Unspecified),
        ] {
            let mut peer = Peer::new(1);
            let answer = answered(&mut peer, told, Instant::now()).await;
            assert!(
                matches!(&answer, Err(DecisionUnavailable::Unsent(Error::NoDecision(why))) if why.starts_with(path)),
                "{answer:?}"
            );
            reports(&mut peer, [released(side, Reason::LateDecision)]).await;
        }
    }

    fn paired() -> proto::Decision {
        let mut told = picked();
        told.leases[0].side = proto::Side::Prefill.into();
        told.leases.push(proto::Lease {
            side: proto::Side::Decode.into(),
            ..told.leases[0]
        });
        let Some(Outcome::Pick(pick)) = &mut told.outcome else {
            panic!("a pick");
        };
        for (index, target) in pick.targets.iter_mut().enumerate() {
            target.id = 28;
            target.pair = Some(proto::Pair {
                decode_id: [47, 49][index],
                decode_dp_rank: 3,
            });
        }
        told
    }

    fn pair_dispatch(side: proto::Side, index: u32, since: u32) -> Frame {
        let Frame::Dispatched(mut said) = dispatch(index, since) else {
            panic!("a dispatch");
        };
        said.side = side.into();
        Frame::Dispatched(said)
    }

    fn pair_completion(side: proto::Side, served: u64) -> Frame {
        Frame::Completed(proto::Completed {
            lease_id: 73,
            side: side.into(),
            served,
            status: 200,
            class: Class::Ok.into(),
            ttft_micros: 1_000,
            tokens_in: 12,
            tokens_out: 34,
            duration_micros: 5_000,
        })
    }

    #[tokio::test(start_paused = true)]
    async fn a_linked_pair_reports_each_side_and_sends_no_encode_outcome() {
        let mut peer = Peer::new(1);
        let now = Instant::now();
        let mut handle = answered(&mut peer, paired(), now)
            .await
            .expect("the pair is held");
        assert_eq!(
            handle.lease().expect("prefill lease").lease.side,
            Side::Prefill
        );
        let first = handle.take(0).expect("the first pair");
        assert_eq!(first.pair.expect("a decode").decode_id, WorkerId(47));
        let sent = now + Duration::from_micros(100);
        handle.dispatched(sent);
        let alternate = handle
            .take_decode(1)
            .expect("another decode of that prefill");
        let decode = alternate.pair.expect("a decode").decode_id;
        assert_eq!(decode, WorkerId(49));
        handle.completed(
            &crate::tracking::Outcome {
                side: Side::Encode,
                ..outcome(WorkerId(47), TransportClass::Ok)
            },
            sent,
        );
        let prefill = outcome(first.id, TransportClass::Ok);
        handle.completed(&prefill, sent);
        handle.completed(&prefill, sent);
        let ended = now + Duration::from_micros(200);
        let decoded = crate::tracking::Outcome {
            side: Side::Decode,
            ..outcome(decode, TransportClass::Ok)
        };
        handle.completed(&decoded, ended);
        handle.completed(&decoded, ended);
        drop(handle);
        reports(
            &mut peer,
            [
                pair_dispatch(proto::Side::Prefill, 0, 100),
                pair_completion(proto::Side::Prefill, 28),
                pair_dispatch(proto::Side::Decode, 1, 200),
                pair_completion(proto::Side::Decode, 49),
            ],
        )
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_linked_pair_releases_both_waiting_sides_or_just_its_waiting_decode() {
        let now = Instant::now();
        let mut peer = Peer::new(1);
        let mut handle = answered(&mut peer, paired(), now)
            .await
            .expect("the pair is held");
        handle.released(ReleaseReason::CommitFailed, now);
        drop(handle);
        reports(
            &mut peer,
            [released(proto::Side::Unspecified, Reason::CommitFailed)],
        )
        .await;

        let mut peer = Peer::new(1);
        let mut handle = answered(&mut peer, paired(), now)
            .await
            .expect("another pair");
        handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
        drop(handle);
        reports(
            &mut peer,
            [
                pair_dispatch(proto::Side::Prefill, 0, 0),
                pair_completion(proto::Side::Prefill, 28),
                released(proto::Side::Decode, Reason::Abandoned),
            ],
        )
        .await;
    }

    /// No link asks a decode later: the link's request drops the word,
    /// so its controller issues both sides, and the handle owes the
    /// decode its dispatch or its release. A decode call on such a
    /// handle is refused as not chosen, and nothing crosses the link
    /// for it.
    #[tokio::test(start_paused = true)]
    async fn a_linked_pair_asked_with_its_decode_later_is_held_as_issued_and_takes_no_call() {
        let now = Instant::now();
        let mut peer = Peer::new(1);
        let mut ask = stated(&[1]);
        ask.facts.budget_micros = 200_000;
        ask.constraints.decode_later = true;
        let mut handle = answered_to(&mut peer, &ask, paired(), now)
            .await
            .expect("the pair is held");
        handle.dispatched(now);
        assert_eq!(
            handle.decide_decode(&ask, now).err(),
            Some(DecisionRefused::Decode(DecodeRefused::NotChosen))
        );
        handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
        drop(handle);
        reports(
            &mut peer,
            [
                pair_dispatch(proto::Side::Prefill, 0, 0),
                pair_completion(proto::Side::Prefill, 28),
                released(proto::Side::Decode, Reason::Abandoned),
            ],
        )
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn an_unreadable_pair_is_released_whole_whatever_its_labels() {
        let mut index = paired();
        index.leases[1].target_index = 2;
        // The prefill lease under the decode side's label: a release by
        // the labels would give the decode side back twice and keep the
        // prefill booking.
        let mut label = paired();
        label.leases[0].side = proto::Side::Decode.into();
        for (told, path) in [(index, "leases.target_index:"), (label, "leases")] {
            let mut peer = Peer::new(1);
            let answer = answered(&mut peer, told, Instant::now()).await;
            assert!(
                matches!(&answer, Err(DecisionUnavailable::Unsent(Error::NoDecision(why))) if why.starts_with(path)),
                "{answer:?}"
            );
            reports(
                &mut peer,
                [released(proto::Side::Unspecified, Reason::LateDecision)],
            )
            .await;
            assert!(peer.link.stands());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_handle_can_report_after_its_link_ends() {
        let mut peer = Peer::new(1);
        let now = Instant::now();
        let mut handle = answered(&mut peer, picked(), now)
            .await
            .expect("the pick is held");
        peer.controller(
            peer.link.term_floor(),
            Some(controller_frame::Frame::NotLeader(
                proto::NotLeader::default(),
            )),
        );
        assert!(timeout(PATIENCE, peer.reports.next())
            .await
            .expect("the session ended")
            .is_none());
        handle.dispatched(now);
        handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
        assert!(!peer.link.stands());
    }

    #[tokio::test(start_paused = true)]
    async fn a_handle_passes_the_wait_budget_to_the_link() {
        let peer = Peer::new(1);
        let mut ask = stated(&[1]);
        ask.facts.budget_micros = 200_000;
        let before = tokio::time::Instant::now();
        let answer = timeout(
            PATIENCE,
            pin!(DecisionGuard::<&InProcess>::asked(
                &peer.link,
                &ask,
                Instant::now(),
                Duration::ZERO
            )),
        )
        .await
        .expect("the ask returns");
        assert!(
            matches!(answer, Err(DecisionUnavailable::Late)),
            "{answer:?}"
        );
        assert_eq!(tokio::time::Instant::now(), before);
    }

    /// Where an answer puts two items: the workers 9 and 4, in that
    /// order.
    fn two_encoders() -> proto::EncodePlaced {
        let encoder = |id| proto::Target {
            id,
            dp_rank: 1,
            pair: None,
        };
        proto::EncodePlaced {
            encoders: vec![encoder(9), encoder(4)],
            one: false,
        }
    }

    /// [`answered`] for an encode request of `phase` with `items`,
    /// whose room is `placed`: the frame that went out, and the handle.
    /// The peer answers `told`, with `beside` for where the items go.
    async fn encode_answered(
        peer: &mut Peer,
        (phase, items): (Phase, EncodeItems<'_>),
        placed: &mut EncodePlaced,
        told: proto::Decision,
        beside: Option<proto::EncodePlaced>,
    ) -> (
        proto::DecideRequest,
        Result<DecisionGuard<&'static InProcess>, DecisionUnavailable>,
    ) {
        let mut ask = stated(&[1]);
        ask.facts.budget_micros = 200_000;
        ask.facts.phase = phase;
        peer.read_hello().await;
        let link = peer.link.clone();
        let exchange = async {
            let sent = peer.requests.next().await.expect("an ask went out");
            peer.answer_beside(1, link.term_floor(), Answer::Decision(told), beside);
            sent
        };
        let both = async {
            let now = Instant::now();
            let held = pin!(DecisionGuard::asked_encode(
                &link, &ask, items, placed, now, PATIENCE
            ));
            tokio::join!(exchange, held)
        };
        timeout(PATIENCE, both).await.expect("an answer came back")
    }

    /// An encode request over a link is its pair's one request. Its
    /// frame states the encode phase and its items, a key that is none
    /// at its place. The handle is a pair's, and the host's room takes
    /// each item's encoder in the keys' order, by its worker and its
    /// rank. The room is emptied at the ask: an earlier answer's
    /// encoders are gone.
    #[tokio::test(start_paused = true)]
    async fn a_linked_encode_request_states_its_items_and_takes_its_encoders() {
        let mut peer = Peer::new(1);
        let keys = [Identity::hashed(11), Identity::NONE];
        let items = EncodeItems {
            keys: &keys,
            references: true,
            ..EncodeItems::default()
        };
        let mut placed = EncodePlaced::new();
        placed.room(true).extend([Target::default(); 3]);
        let asked = (Phase::Prefill, items);
        let beside = Some(two_encoders());
        let (sent, handle) = encode_answered(&mut peer, asked, &mut placed, paired(), beside).await;
        let facts = sent.facts.expect("request facts");
        assert_eq!(facts.phase, i32::from(proto::Phase::Encode));
        let stated = sent.encode.expect("the frame states its items");
        let forms: Vec<_> = stated.keys.iter().map(|key| key.form.clone()).collect();
        assert_eq!(forms, [Some(proto::identity::Form::Hash(11)), None]);
        assert!(stated.references && !stated.decode_images && stated.one.is_none());

        let handle = handle.expect("a pair's handle");
        let lease = handle.lease().expect("the pair's lease");
        assert_eq!((lease.lease.id.0, lease.lease.side), (73, Side::Prefill));
        let named = |id| Target {
            id: WorkerId(id),
            dp_rank: 1,
            ..Target::default()
        };
        assert_eq!(placed.encoders(), [named(9), named(4)]);
        assert!(!placed.one());
    }

    /// The engine's entry takes an encode request at the prefill phase
    /// and refuses any other as a side sent on its own. The handle
    /// refuses it by the field before anything is sent, with what
    /// [`DecisionGuard::asked`] refuses: no credit goes, no time
    /// passes, and the room is emptied.
    #[tokio::test(start_paused = true)]
    async fn a_linked_encode_request_the_handle_refuses_sends_nothing() {
        let mut peer = Peer::new(1);
        let demand = Demands {
            runtime: 7,
            ..Demands::default()
        };
        let asks = [
            (Phase::Single, 200_000, Demands::default(), "facts.phase"),
            (Phase::Decode, 200_000, Demands::default(), "facts.phase"),
            (Phase::Encode, 200_000, Demands::default(), "facts.phase"),
            (Phase::Prefill, 200_000, demand, "constraints.demands"),
            (Phase::Prefill, 0, Demands::default(), ""),
        ];
        let before = tokio::time::Instant::now();
        for (phase, budget_micros, demands, field) in asks {
            let mut ask = stated(&[1]);
            ask.facts.budget_micros = budget_micros;
            ask.facts.phase = phase;
            ask.constraints.demands = demands;
            let mut placed = EncodePlaced::new();
            placed.room(true).push(Target::default());
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&InProcess>::asked_encode(
                    &peer.link,
                    &ask,
                    EncodeItems::default(),
                    &mut placed,
                    Instant::now(),
                    PATIENCE
                )),
            )
            .await
            .expect("the ask returns");
            let refused = match &answer {
                Err(DecisionUnavailable::Unsent(Error::OutOfRange(named))) => *named == field,
                Err(DecisionUnavailable::Late) => field.is_empty(),
                _ => false,
            };
            assert!(refused, "{phase:?} {field}: {answer:?}");
            assert!(placed.encoders().is_empty() && !placed.one());
        }
        assert_eq!(tokio::time::Instant::now(), before);
        assert!(
            poll(pin!(peer.requests.next())).is_pending(),
            "nothing went out"
        );
    }

    /// An answer whose encoders do not fit its request is one the
    /// handle cannot read: a pick that names no encoder for its keys,
    /// a count that is not one a key, an encoder that names a pair.
    /// Its pair's lease is released late, whole, as any unreadable
    /// pair's is, and the room is left empty.
    #[tokio::test(start_paused = true)]
    async fn an_encode_answer_the_handle_cannot_read_is_released_late() {
        let short = proto::EncodePlaced {
            encoders: two_encoders().encoders[..1].to_vec(),
            ..two_encoders()
        };
        let mut paired_encoder = two_encoders();
        paired_encoder.encoders[1].pair = Some(proto::Pair::default());
        for (beside, path) in [
            (None, "encode:"),
            (Some(short), "encode.encoders:"),
            (Some(paired_encoder), "encode.encoders.pair:"),
        ] {
            let mut peer = Peer::new(1);
            let keys = [Identity::hashed(11), Identity::hashed(12)];
            let items = EncodeItems {
                keys: &keys,
                ..EncodeItems::default()
            };
            let mut placed = EncodePlaced::new();
            let asked = (Phase::Prefill, items);
            let (_, answer) =
                encode_answered(&mut peer, asked, &mut placed, paired(), beside).await;
            assert!(
                matches!(&answer, Err(DecisionUnavailable::Unsent(Error::NoDecision(why))) if why.starts_with(path)),
                "{path} {answer:?}"
            );
            assert!(placed.encoders().is_empty() && !placed.one());
            reports(
                &mut peer,
                [released(proto::Side::Unspecified, Reason::LateDecision)],
            )
            .await;
            assert!(peer.link.stands());
        }
    }

    /// Over a link a budget of 0 says the request's time is spent, and
    /// the controller answers busy at its receipt. The handle answers
    /// such a request late before anything is sent: no credit goes, and
    /// no time passes.
    #[tokio::test(start_paused = true)]
    async fn a_request_with_no_budget_is_late_before_anything_is_sent() {
        let mut peer = Peer::new(1);
        let ask = stated(&[1]);
        assert_eq!(ask.facts.budget_micros, 0);
        let before = tokio::time::Instant::now();
        let answer = timeout(
            PATIENCE,
            pin!(DecisionGuard::<&InProcess>::asked(
                &peer.link,
                &ask,
                Instant::now(),
                PATIENCE
            )),
        )
        .await
        .expect("the ask returns");
        assert!(
            matches!(answer, Err(DecisionUnavailable::Late)),
            "{answer:?}"
        );
        assert_eq!(tokio::time::Instant::now(), before);
        assert!(
            poll(pin!(peer.requests.next())).is_pending(),
            "nothing went out"
        );
    }

    /// A controller on protocol 1 reads no hard demand, and would pick
    /// among workers the demand rules out. So the handle refuses a
    /// request that states a demand before anything is sent, over a
    /// session whose welcome chose that version or states none. It did
    /// so before a frame could state one. A welcome on a version the
    /// hello did not offer is refused so too. No credit goes, and no
    /// time passes.
    #[tokio::test(start_paused = true)]
    async fn a_request_that_states_a_demand_is_refused_before_anything_is_sent() {
        use crate::service::controller::PROTOCOL_VERSION;

        for protocol in [0, 1, PROTOCOL_VERSION + 1] {
            let mut peer = Peer::speaking(1, protocol);
            let mut ask = stated(&[1]);
            ask.facts.budget_micros = 200_000;
            ask.constraints.demands = Demands {
                runtime: 7,
                ..Demands::default()
            };
            let before = tokio::time::Instant::now();
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&InProcess>::asked(
                    &peer.link,
                    &ask,
                    Instant::now(),
                    PATIENCE
                )),
            )
            .await
            .expect("the ask returns");
            assert!(
                matches!(
                    answer,
                    Err(DecisionUnavailable::Unsent(Error::OutOfRange(
                        "constraints.demands"
                    )))
                ),
                "protocol {protocol}: {answer:?}"
            );
            assert_eq!(tokio::time::Instant::now(), before);
            assert!(
                poll(pin!(peer.requests.next())).is_pending(),
                "nothing went out"
            );
        }
    }

    /// Over a session whose welcome chose protocol 2, a single request
    /// that names no worker states its hard demands on its frame. Each
    /// part goes as the request states it, and the ask is answered as
    /// any ask is.
    #[tokio::test(start_paused = true)]
    async fn a_single_requests_demands_go_out_on_its_frame_over_a_session_on_protocol_two() {
        let mut peer = Peer::speaking(1, 2);
        let mut ask = stated(&[1]);
        ask.facts.budget_micros = 200_000;
        ask.constraints.demands = Demands {
            runtime: 7,
            references: References::Caller,
            input_window: 4_096,
            output_window: 512,
            decoding: 3,
            ..Demands::default()
        };
        peer.read_hello().await;
        let link = peer.link.clone();
        let exchange = async {
            let sent = peer.requests.next().await.expect("an ask went out");
            peer.answer(1, link.term_floor(), Answer::Decision(picked()));
            sent
        };
        let both = async {
            let now = Instant::now();
            let held = pin!(DecisionGuard::<&InProcess>::asked(
                &link, &ask, now, PATIENCE
            ));
            tokio::join!(held, exchange)
        };
        let (handle, sent) = timeout(PATIENCE, both).await.expect("an answer came back");
        assert!(handle.is_ok(), "the ask is answered: {:?}", handle.err());
        assert_eq!(
            sent.demands,
            Some(proto::Demands {
                runtime: 7,
                vision: false,
                references: proto::demands::References::Caller.into(),
                input_window: 4_096,
                output_window: 512,
                decoding: 3,
            })
        );
    }

    /// Over a session on protocol 2 the handle still refuses a demand
    /// that a controller holds on no frame of the request's kind yet,
    /// before anything is sent. The kinds are a request that names its
    /// worker, a request at a side's phase and an encode request. The
    /// part that demands vision is refused by its own name, on a single
    /// request too: no registration states whether a worker takes
    /// images.
    #[tokio::test(start_paused = true)]
    async fn a_demand_no_frame_of_its_requests_kind_holds_yet_is_refused_before_anything_is_sent() {
        let mut peer = Peer::speaking(1, 2);
        let demand = Demands {
            runtime: 7,
            ..Demands::default()
        };
        let vision = Demands {
            vision: true,
            ..Demands::default()
        };
        let both = Demands {
            vision: true,
            ..demand
        };
        let field = |answer: &Result<DecisionGuard<&InProcess>, DecisionUnavailable>| match answer {
            Err(DecisionUnavailable::Unsent(Error::OutOfRange(field))) => Some(*field),
            _ => None,
        };
        let before = tokio::time::Instant::now();
        let asks = [
            (Phase::Single, Some(Some(0)), demand, "constraints.demands"),
            (Phase::Single, Some(None), demand, "constraints.demands"),
            (Phase::Prefill, None, demand, "constraints.demands"),
            (Phase::Decode, None, demand, "constraints.demands"),
            (Phase::Single, None, vision, "constraints.demands.vision"),
            (Phase::Single, None, both, "constraints.demands.vision"),
        ];
        for (phase, named, demands, refused) in asks {
            let mut ask = stated(&[1]);
            ask.facts.budget_micros = 200_000;
            ask.facts.phase = phase;
            ask.constraints.demands = demands;
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&InProcess>::asked_naming(
                    &peer.link,
                    &ask,
                    named,
                    Instant::now(),
                    PATIENCE
                )),
            )
            .await
            .expect("the ask returns");
            assert_eq!(field(&answer), Some(refused), "{phase:?} {named:?}");
        }
        let mut ask = stated(&[1]);
        ask.facts.budget_micros = 200_000;
        ask.facts.phase = Phase::Prefill;
        ask.constraints.demands = demand;
        let mut placed = EncodePlaced::new();
        let answer = timeout(
            PATIENCE,
            pin!(DecisionGuard::<&InProcess>::asked_encode(
                &peer.link,
                &ask,
                EncodeItems::default(),
                &mut placed,
                Instant::now(),
                PATIENCE
            )),
        )
        .await
        .expect("the ask returns");
        assert_eq!(
            field(&answer),
            Some("constraints.demands"),
            "an encode request"
        );
        assert_eq!(tokio::time::Instant::now(), before);
        assert!(
            poll(pin!(peer.requests.next())).is_pending(),
            "nothing went out"
        );
    }
}
