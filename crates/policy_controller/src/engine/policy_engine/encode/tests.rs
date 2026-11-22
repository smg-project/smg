//! The cases of an encode request's decision over three lists of worker
//! objects.

use parking_lot::Mutex;
use smg_worker::{
    identity::basic::Worker, load::state::one_rank_load, test_support::ready_worker,
    worker::WorkerStatus,
};

use super::*;
use crate::{
    document::spec::PolicySpec,
    engine::{
        decision::{ShedReason, SUBSET_HINT},
        framework::{DecisionScope, Filter},
        pair_classes::PartnerClasses,
        policy_engine::{pair::PairAdmission, PolicyEngine},
        records::worker_id,
        view::{
            Candidate, Constraints, FixedFacts, Identity, MediaCounts, Phase, PoolFacts,
            RequestFacts, WorkerId,
        },
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
};

/// A host's reads of its own admission, as two tests of a place: whether
/// the prefill there has capacity, then whether the decode there has a room.
#[derive(Debug)]
struct Gates(fn(usize) -> bool, fn(usize) -> bool);

impl PairAdmission for Gates {
    fn has_capacity(&self, prefill: usize) -> bool {
        (self.0)(prefill)
    }

    fn has_room(&self, decode: usize) -> bool {
        (self.1)(decode)
    }
}

/// A host's three lists of ready workers with the engine's value of each:
/// the prefills (list `0`), the decodes (`1`), the encoders (`2`). Every
/// prefill pairs with every decode and every gate is open, until a case
/// says otherwise, under the stock encode document.
/// `loads` is the snapshot of load reports the host hands with a decision.
/// `images` says the request states an image. `excluded` are the workers
/// the host says it just failed to reach.
struct Host {
    engine: PolicyEngine,
    members: [PoolMembers; 3],
    partners: PartnerClasses,
    gates: Gates,
    loads: Option<Arc<LoadSnapshot>>,
    images: bool,
    excluded: Vec<WorkerId>,
}

/// Where the worker at `place` of a host's list `list` answers.
fn url(list: usize, place: usize) -> String {
    format!("http://list-{list}-worker-{place}:8000")
}

/// What an encoder of `class` states: no push, no references.
fn of(class: u32) -> EncoderFacts {
    EncoderFacts {
        class,
        ..EncoderFacts::default()
    }
}

/// The stock encode document: `power_of_two` on each of the three sides.
fn stock() -> PolicySpec {
    let preset = "power_of_two";
    PolicySpec::epd_presets(preset, preset, preset).expect("the presets")
}

/// Load reports of the first two workers of each of the three lists: the
/// worker at place `short` of each reports the shorter queue.
fn reports(short: usize) -> Arc<LoadSnapshot> {
    let report = |list: usize, place: usize| {
        let queued = if place == short { 0 } else { 90_000 };
        (url(list, place), one_rank_load(queued, 0.1, 1_000.0))
    };
    let reports = (0..3).flat_map(|list| [report(list, 0), report(list, 1)]);
    LoadSnapshot::from_loads_for_test(reports.collect())
}

/// The place of the decode a decision pairs its prefill with.
fn decode_of(decision: &Decision) -> Option<usize> {
    let pair = decision.pick().and_then(|pick| pick.targets.first()?.pair);
    pair.map(|pair| pair.decode_position as usize)
}

/// The stage a decision sheds under for want of a worker.
fn gap(decision: &Decision) -> Option<&'static str> {
    match decision {
        Decision::Shed(shed) if shed.reason == ShedReason::NoEndpoints => Some(shed.emptied_by),
        _ => None,
    }
}

impl Host {
    fn of(prefills: usize, decodes: usize, encoders: usize) -> Self {
        let spec = AppliedSpec::validate(&stock()).expect("a valid document");
        Self::under(spec, prefills, decodes, encoders)
    }

    /// [`Self::of`], under a document of the case's own.
    fn under(spec: AppliedSpec, prefills: usize, decodes: usize, encoders: usize) -> Self {
        let engine = PolicyEngine::new(spec);
        let counts = [prefills, decodes, encoders];
        // The host says of its workers that the first encoder takes images.
        let sees = url(2, 0);
        let members = [0, 1, 2].map(|list| {
            let ready = |place| ready_worker(&url(list, place), "model-a");
            let workers = (0..counts[list]).map(ready).collect();
            let facts_of = |worker: &Arc<dyn Worker>| FixedFacts {
                vision: worker.url() == sees,
                ..FixedFacts::default()
            };
            engine.pool_members_with_fixed_facts(&workers, facts_of)
        });
        let all: Vec<u32> = (0..decodes as u32).collect();
        let partners = PartnerClasses::number(decodes, (0..prefills).map(|_| all.as_slice()));
        Self {
            engine,
            members,
            partners,
            gates: Gates(|_| true, |_| true),
            loads: None,
            images: false,
            excluded: Vec::new(),
        }
    }

    /// The worker at `place` of list `list`.
    fn worker(&self, list: usize, place: usize) -> &Arc<dyn Worker> {
        &self.members[list].workers()[place]
    }

    /// One decision over the lists as they stand now, for a request of one
    /// item a key that states the subset hint `hint`. `said` is what the
    /// host says of its encoders and of its prefills. With the decision
    /// come the places of its prefill and of each key's encoder: a target
    /// names its worker by its place in its list and by its own id.
    fn decide(
        &self,
        said: (&[EncoderFacts], &[u32]),
        keys: &[Identity<'_>],
        hint: Option<u32>,
        placed: &mut EncodePlaced,
    ) -> (Decision, Option<usize>, Vec<usize>) {
        let [prefill, decode, encoders] = &self.members;
        let pair = PairLists {
            prefill,
            decode,
            classes: &self.partners,
            admission: &self.gates,
        };
        let lists = EncodeLists {
            pair,
            encoders,
            facts: said.0,
            prefill_classes: said.1,
        };
        let facts = RequestFacts {
            model: "model-a",
            phase: Phase::Prefill,
            media: MediaCounts {
                images: u16::from(self.images),
                ..MediaCounts::NONE
            },
            ..RequestFacts::default()
        };
        let constraints = Constraints {
            subset_hint: hint,
            excluded: &self.excluded,
            ..Constraints::default()
        };
        let mut request = Request {
            id: b"",
            facts,
            constraints,
        };
        let items = EncodeItems {
            keys,
            ..EncodeItems::default()
        };
        let document = self.engine.document.load();
        let (spec, clock, now) = (&document.spec, self.engine.clock(), Instant::now());
        let loads = self.loads.as_ref();
        let decision = lists.decide(spec, loads, clock, now, &mut request, items, placed);
        assert_eq!(request.facts.now_micros, clock.micros(now));
        let named = |list: usize, place: u32, id| {
            assert_eq!(worker_id(self.worker(list, place as usize).as_ref()), id);
            place as usize
        };
        let first = decision.pick().and_then(|pick| pick.targets.first());
        let prefill = first.map(|first| {
            let pair = first.pair.expect("a pair");
            named(1, pair.decode_position, pair.decode_id);
            named(0, first.position, first.id)
        });
        let encoders = placed.encoders().iter();
        let encoders = encoders.map(|target| named(2, target.position, target.id));
        (decision, prefill, encoders.collect())
    }
}

/// The pair is picked where an encoder is ready, and each key's encoder is
/// one of the class the chosen prefill names, as the three lists' workers
/// stand at the call. What the host says of an encoder and of a prefill is
/// read by the worker's place in its list.
#[test]
fn an_encode_request_is_decided_over_the_hosts_three_lists_as_they_stand_at_the_call() {
    let host = Host::of(2, 2, 3);
    // Encoders 0 and 1 hand to prefill 0, encoder 2 to prefill 1.
    let said = (&[of(1), of(1), of(2)][..], &[1, 2][..]);
    let keys = [Identity::plain("item-a"), Identity::plain("item-b")];
    let mut placed = EncodePlaced::new();
    let mut decide = || host.decide(said, &keys, None, &mut placed);
    let (_, prefill, encoders) = decide();
    let class = said.1[prefill.expect("a pick")];
    assert_eq!(encoders.len(), 2);
    assert!(encoders.iter().all(|&at| said.0[at].class == class));
    // A prefill that stopped being ready leaves the other, and its class's
    // one encoder, the last of the list, takes both keys.
    host.worker(0, 0).set_status(WorkerStatus::NotReady);
    let (_, prefill, encoders) = decide();
    assert_eq!((prefill, encoders), (Some(1), vec![2, 2]));
    // That encoder stops too: the pair is picked where an encoder is ready.
    host.worker(0, 0).set_status(WorkerStatus::Ready);
    host.worker(2, 2).set_status(WorkerStatus::NotReady);
    let (_, prefill, encoders) = decide();
    assert_eq!(prefill, Some(0));
    assert!(encoders.iter().all(|&at| at < 2), "{encoders:?}");
    // No encoder is ready: shed under the encode side's stage, no key placed.
    host.worker(2, 0).set_status(WorkerStatus::NotReady);
    host.worker(2, 1).set_status(WorkerStatus::NotReady);
    let (decision, _, encoders) = decide();
    assert_eq!((gap(&decision), encoders.len()), (Some("encoders"), 0));
    // A ready encoder and no decode at all: the shed is the pair's.
    let (alone, said) = (Host::of(1, 0, 1), (&[of(1)][..], &[1][..]));
    let (decision, ..) = alone.decide(said, &keys, None, &mut placed);
    assert_eq!(gap(&decision), Some("partnered"));
}

/// The records of the three lists read the load reports their host hands
/// with the decision: the stock document takes the shorter queue of each
/// list, whichever worker reports it.
#[test]
fn the_three_lists_records_read_the_load_reports_their_host_hands() {
    let mut host = Host::of(2, 2, 2);
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    for short in [0, 1] {
        host.loads = Some(reports(short));
        let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
        assert_eq!((prefill, encoders), (Some(short), vec![short]));
        assert_eq!(decode_of(&decision), Some(short));
    }
    // A request that states an image takes the encoder its host says takes
    // images, whichever reports the shorter queue.
    host.images = true;
    let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(encoders, vec![0]);
}

/// A pair's lists are narrowed by no subset hint: the pair's own decision
/// sheds a request that states one, and what an earlier decision placed is
/// emptied.
#[test]
fn a_request_that_states_a_subset_hint_is_shed_as_its_pair_and_places_no_key() {
    let host = Host::of(1, 1, 1);
    let said = (&[of(1)][..], &[1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    let (_, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, encoders), (Some(0), vec![0]));
    let (decision, _, encoders) = host.decide(said, &keys, Some(0), &mut placed);
    assert_eq!((gap(&decision), encoders.len()), (Some(SUBSET_HINT), 0));
}

/// The request's constraints reach each of the three sides: a worker its
/// host excludes steps aside in its own list while another is left, though
/// it reports the shorter queue. The classic router does the same on a
/// pair's two sides for the engines a failed attempt used
/// (`model_gateway/src/routers/common/placement.rs:1492-1493` and
/// `:1568-1569` on main ddd8a44cb). It names no encoder tried
/// (`model_gateway/src/routers/grpc/context.rs:1577-1581`) and places each
/// item with no word of that list
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2324-2338`),
/// so a host that excludes a pair's workers alone reads its answer.
#[test]
fn a_worker_the_requests_host_excludes_steps_aside_in_each_of_the_three_lists() {
    let mut host = Host::of(2, 2, 2);
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    host.loads = Some(reports(0));
    let (_, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, encoders), (Some(0), vec![0]));
    let first = |list| worker_id(host.worker(list, 0).as_ref());
    host.excluded = [0, 1, 2].map(first).to_vec();
    let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, encoders), (Some(1), vec![1]));
    assert_eq!(decode_of(&decision), Some(1));
}

/// The pair's records take their pair classes and their host's admission
/// by each worker's place in its own list, and the engine's call takes the
/// table of the classes that pair: a decode is its prefill's partner though
/// another reports the shorter queue, and a prefill its host says is at its
/// limit steps aside. The classic router picks an encode request's pair so
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2217-2229`
/// on main ddd8a44cb): it draws the decode from its prefill's partners
/// (`model_gateway/src/routers/common/placement.rs:1551-1553`) and, under
/// its prefill admission, takes a prefill at its limit out before its
/// policy picks (`:1470-1473`).
#[test]
fn the_pair_is_one_its_hosts_classes_pair_and_a_prefill_at_its_limit_steps_aside() {
    let mut host = Host::of(2, 2, 2);
    // Prefill 0 pairs with decode 1 alone, and prefill 1 with decode 0.
    host.partners = PartnerClasses::number(2, [&[1u32][..], &[0][..]]);
    host.loads = Some(reports(0));
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    let (decision, prefill, _) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, decode_of(&decision)), (Some(0), Some(1)));
    host.gates = Gates(|prefill| prefill != 0, |_| true);
    let (decision, prefill, _) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, decode_of(&decision)), (Some(1), Some(0)));
}

/// A decision asked while this thread's buffers are lent, as a stage asks
/// one from inside its own, takes buffers of its own and answers the same.
#[test]
fn a_decision_asked_inside_a_decision_takes_buffers_of_its_own() {
    let host = Host::of(1, 1, 1);
    let said = (&[of(1)][..], &[1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    let (_, prefill, encoders) = BUFFERS.with(|buffers| {
        let _lent = buffers.borrow_mut();
        host.decide(said, &keys, None, &mut placed)
    });
    assert_eq!((prefill, encoders), (Some(0), vec![0]));
}

/// The engine's call is shown the prefill list's pool facts, as a pair's
/// first decision is: a stage of the prefill profile that opens a scope
/// reads the prefill list's size beside its rows, and never the decodes'.
#[test]
fn the_engines_call_is_shown_the_prefill_lists_pool_facts() {
    #[derive(Debug)]
    struct Shown(Arc<Mutex<Vec<(u32, usize)>>>);

    impl Lifecycle for Shown {}

    impl Filter for Shown {
        fn name(&self) -> &'static str {
            "shown-pool"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    impl DecisionScope for Shown {
        fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], pool: &PoolFacts<'_>) {
            self.0.lock().push((pool.pool_size, rows.len()));
        }

        fn close(&self) {}
    }

    let shown = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&shown);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("shown-pool", move || Box::new(Shown(Arc::clone(&kept))))
        .expect("a free filter name");
    let mut document = stock();
    let prefill = document.profiles.get_mut("prefill");
    let filters = &mut prefill.expect("the prefill profile").filter;
    filters.push("shown-pool".into());
    let digest = document.digest().expect("a digest");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    // One prefill and two decodes: the two lists' sizes differ.
    let host = Host::under(spec.expect("a valid document"), 1, 2, 1);
    let said = (&[of(1)][..], &[1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    let (_, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, encoders), (Some(0), vec![0]));
    // The pool's size as its facts say it, and the rows read.
    let shown = shown.lock();
    assert!(
        !shown.is_empty() && shown.iter().all(|&seen| seen == (1, 1)),
        "{shown:?}"
    );
}
