//! The cases of an encode request's decision over three lists of worker
//! objects. No request here states an image: no record of these lists says
//! its worker takes one.

use smg_worker::{
    identity::basic::Worker, load::state::one_rank_load, test_support::ready_worker,
    worker::WorkerStatus,
};

use super::*;
use crate::{
    document::spec::PolicySpec,
    engine::{
        decision::{ShedReason, SUBSET_HINT},
        pair_classes::PartnerClasses,
        policy_engine::{pair::PairAdmission, PolicyEngine},
        records::worker_id,
        view::{Constraints, Identity, Phase, RequestFacts},
    },
};

/// Every prefill has capacity and every decode a room.
#[derive(Debug)]
struct Open;

impl PairAdmission for Open {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// A host's three lists of ready workers with the engine's value of each:
/// the prefills (list `0`), the decodes (`1`), the encoders (`2`). Every
/// prefill pairs with every decode, under the stock encode document.
/// `loads` is the snapshot of load reports the host hands with a decision.
struct Host {
    engine: PolicyEngine,
    members: [PoolMembers; 3],
    partners: PartnerClasses,
    loads: Option<Arc<LoadSnapshot>>,
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

/// The stage a decision sheds under for want of a worker.
fn gap(decision: &Decision) -> Option<&'static str> {
    match decision {
        Decision::Shed(shed) if shed.reason == ShedReason::NoEndpoints => Some(shed.emptied_by),
        _ => None,
    }
}

impl Host {
    fn of(prefills: usize, decodes: usize, encoders: usize) -> Self {
        let preset = "power_of_two";
        let document = PolicySpec::epd_presets(preset, preset, preset).expect("the presets");
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        let engine = PolicyEngine::new(spec);
        let counts = [prefills, decodes, encoders];
        let members = [0, 1, 2].map(|list| {
            let ready = |place| ready_worker(&url(list, place), "model-a");
            engine.pool_members(&(0..counts[list]).map(ready).collect())
        });
        let all: Vec<u32> = (0..decodes as u32).collect();
        let partners = PartnerClasses::number(decodes, (0..prefills).map(|_| all.as_slice()));
        Self {
            engine,
            members,
            partners,
            loads: None,
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
            admission: &Open,
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
            ..RequestFacts::default()
        };
        let constraints = Constraints {
            subset_hint: hint,
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
        let (spec, clock, now) = (self.engine.spec(), self.engine.clock(), Instant::now());
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
        let report = |list: usize, place: usize| {
            let queued = if place == short { 0 } else { 90_000 };
            (url(list, place), one_rank_load(queued, 0.1, 1_000.0))
        };
        let reports = (0..3).flat_map(|list| [report(list, 0), report(list, 1)]);
        host.loads = Some(LoadSnapshot::from_loads_for_test(reports.collect()));
        let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
        let pair = decision.pick().and_then(|pick| pick.targets.first()?.pair);
        assert_eq!((prefill, encoders), (Some(short), vec![short]));
        assert_eq!(pair.map(|pair| pair.decode_position as usize), Some(short));
    }
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
