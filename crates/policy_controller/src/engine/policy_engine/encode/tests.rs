//! The cases of an encode request's decision over three lists of worker
//! objects.

use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use parking_lot::Mutex;
use smg_worker::{
    identity::{
        basic::{ConnectionMode, Worker},
        builder::BasicWorkerBuilder,
    },
    load::state::{one_rank_load, TestFeed},
    model_card::ModelCard,
    test_support::{no_health_check, ready_worker},
    worker::{WorkerLoadResponse, WorkerSpec, WorkerStatus, WorkerType},
};

use super::*;
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{LiftedVeto, ShedReason, Targets, SUBSET_HINT},
        decision_engine::{DecisionEngine, HeldDecision, HeldDecode},
        framework::{DecisionScope, Filter, Picker, TieBand},
        list_cell::tests::PublishedList,
        pair_classes::PartnerClasses,
        policy_engine::pair::PairAdmission,
        records::worker_id,
        view::{
            Candidate, Constraints, FixedFacts, Identity, MediaCounts, PoolFacts, RequestFacts,
            WireModes, WorkerId,
        },
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry, Best},
    store::GatewayId,
    tracking::lease::{EventSet, Lease, Side},
};

mod signals;

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

/// A host's three lists of ready workers as it publishes them, each with
/// its cell: the prefills (list `0`), the decodes (`1`), the encoders (`2`). Every
/// prefill pairs with every decode and every gate is open, until a case
/// says otherwise, under the stock encode document.
/// `loads` is the snapshot of load reports the host hands with a decision.
/// `images` says the request states an image, and `video` a video.
/// `excluded` are the workers
/// the host says it just failed to reach. `places` are the places of the
/// prefill list and of the decode list the host holds the request to, and
/// `encoder_places` those of the encoder list.
/// `vetoes` are the places of each of the two its standing vetoes mask, and
/// those of them a hold stands on. `encoder_vetoes` are the same two of the
/// encoder list. `own` is a mask the request states itself: its stalled
/// places, and those a hold stands on. `tried` are the workers a retry says
/// its failed attempts were sent to, and `seed` is the request's seed.
struct Host {
    engine: PolicyEngine,
    lists: [PublishedList; 3],
    partners: PartnerClasses,
    gates: Gates,
    loads: Option<Arc<LoadSnapshot>>,
    images: bool,
    video: bool,
    excluded: Vec<WorkerId>,
    places: [Option<Vec<u32>>; 2],
    encoder_places: Option<Vec<u32>>,
    vetoes: [[Vec<u32>; 2]; 2],
    encoder_vetoes: [Vec<u32>; 2],
    own: [Vec<u32>; 2],
    tried: Vec<WorkerId>,
    seed: u64,
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
fn queues(short: usize) -> Vec<(String, WorkerLoadResponse)> {
    let report = |list: usize, place: usize| {
        let queued = if place == short { 0 } else { 90_000 };
        (url(list, place), one_rank_load(queued, 0.1, 1_000.0))
    };
    let reports = (0..3).flat_map(|list| [report(list, 0), report(list, 1)]);
    reports.collect()
}

/// A snapshot of those reports, as a host hands one with a decision.
fn reports(short: usize) -> Arc<LoadSnapshot> {
    LoadSnapshot::from_loads_for_test(queues(short))
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

/// The name a decision is declined under.
fn declined_under(decision: &Decision) -> Option<&'static str> {
    match decision {
        Decision::Shed(shed) if shed.reason == ShedReason::PolicyDeclined => Some(shed.emptied_by),
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
        let lists = [0, 1, 2].map(|list| {
            let ready = |place| ready_worker(&url(list, place), "model-a");
            let workers = (0..counts[list]).map(ready).collect();
            let facts_of = |worker: &Arc<dyn Worker>| FixedFacts {
                vision: worker.url() == sees,
                ..FixedFacts::default()
            };
            PublishedList::with_facts(workers, facts_of)
        });
        let all: Vec<u32> = (0..decodes as u32).collect();
        let partners = PartnerClasses::number(decodes, (0..prefills).map(|_| all.as_slice()));
        Self {
            engine,
            lists,
            partners,
            gates: Gates(|_| true, |_| true),
            loads: None,
            images: false,
            video: false,
            excluded: Vec::new(),
            places: [None, None],
            encoder_places: None,
            vetoes: Default::default(),
            encoder_vetoes: Default::default(),
            own: Default::default(),
            tried: Vec::new(),
            seed: 0,
        }
    }

    /// The worker at `place` of list `list`.
    fn worker(&self, list: usize, place: usize) -> &Arc<dyn Worker> {
        &self.lists[list].workers()[place]
    }

    /// The host says of the decode at `place` alone that it takes images.
    fn says_the_decode_takes_images(&mut self, place: usize) {
        let (decodes, sees) = (Arc::clone(self.lists[1].workers()), url(1, place));
        let facts_of = |worker: &Arc<dyn Worker>| FixedFacts {
            vision: worker.url() == sees,
            ..FixedFacts::default()
        };
        self.lists[1] = PublishedList::with_facts(decodes, facts_of);
    }

    /// The host's three lists, with what it says of its encoders and of its
    /// prefills.
    fn lists<'a>(&'a self, said: (&'a [EncoderFacts], &'a [u32])) -> EncodeLists<'a> {
        let [prefill, decode, encoders] = &self.lists;
        let pair = PairLists {
            prefill: prefill.lent(),
            decode: decode.lent(),
            classes: &self.partners,
            admission: &self.gates,
            prefill_places: self.places[0].as_deref(),
            decode_places: self.places[1].as_deref(),
            prefill_stalled: &self.vetoes[0][0],
            prefill_held: &self.vetoes[0][1],
            decode_stalled: &self.vetoes[1][0],
            decode_held: &self.vetoes[1][1],
        };
        EncodeLists {
            pair,
            encoders: encoders.lent(),
            encoders_places: self.encoder_places.as_deref(),
            encoders_stalled: &self.encoder_vetoes[0],
            encoders_held: &self.encoder_vetoes[1],
            facts: said.0,
            prefill_classes: said.1,
        }
    }

    /// The host's request as the engine's entries take it: made at the
    /// prefill phase over the three lists, held to no wire, its decode
    /// sent with its prefill, with the snapshot the host hands and the
    /// image and the video it states.
    fn asks<'a>(&'a self, said: (&'a [EncoderFacts], &'a [u32])) -> Asks<'a> {
        Asks {
            lists: WorkerLists::Encode(self.lists(said)),
            phase: Phase::Prefill,
            wires: WireModes::NONE,
            later: false,
            images: self.images,
            video: self.video,
            loads: self.loads.as_ref(),
        }
    }

    /// [`Self::asks`], over the pair's two lists alone.
    fn asks_as_a_pair<'a>(&'a self, said: (&'a [EncoderFacts], &'a [u32])) -> Asks<'a> {
        Asks {
            lists: WorkerLists::Pair(self.lists(said).pair),
            ..self.asks(said)
        }
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
        let lists = self.lists(said);
        let facts = RequestFacts {
            model: "model-a",
            phase: Phase::Prefill,
            media: MediaCounts {
                images: u16::from(self.images),
                video: u16::from(self.video),
                ..MediaCounts::NONE
            },
            attempt: u8::from(!self.tried.is_empty()),
            tried: &self.tried,
            seed: self.seed,
            ..RequestFacts::default()
        };
        let constraints = Constraints {
            subset_hint: hint,
            excluded: &self.excluded,
            stalled: &self.own[0],
            held: &self.own[1],
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
        let (engine, now) = (&self.engine, Instant::now());
        let loads = self.loads.as_ref();
        let decision = lists.decide(engine, loads, now, &mut request, items, placed);
        assert_eq!(request.facts.now_micros, engine.clock().micros(now));
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

/// A request whose media is video alone asks vision of its encoder as one
/// that states an image does: its item goes to the encoder its host says
/// takes images, whichever reports the shorter queue, and where the host
/// says it of no encoder the request is refused under the vision gap.
///
/// The classic router asks vision for any pixel item, an image or a video
/// (`is_pixel_modality` and `request_needs_vision`,
/// `model_gateway/src/routers/grpc/multimodal/capability.rs:325-344` on
/// main d6517a4fe), and holds each encoder to it
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:1530`).
#[test]
fn a_request_of_video_alone_takes_the_encoder_that_takes_images() {
    let mut host = Host::of(2, 2, 2);
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    // The second encoder reports the shorter queue, and the host says of
    // the first alone that it takes images.
    host.loads = Some(reports(1));
    let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(encoders, vec![1], "no pixel: the shorter queue");
    host.video = true;
    let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(encoders, vec![0], "a video: the encoder that takes images");
    // The host says of no encoder that it takes images.
    let encoders = Arc::clone(host.lists[2].workers());
    host.lists[2] = PublishedList::with_facts(encoders, |_| FixedFacts::default());
    let (decision, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((gap(&decision), encoders.len()), (Some("encode-vision"), 0));
}

/// An encode request's pair is made among first partners while one is
/// ready, as a pair's own request is: the entry hands the pair's decision
/// the first partners its host numbered
/// ([`PartnerClasses::number_first`]). The stock document takes the shorter
/// queue of each list, and the pair that goes first is taken past it.
#[test]
fn an_encode_requests_pair_is_made_among_its_first_partners() {
    let mut host = Host::of(2, 2, 2);
    // Each prefill has an encoder of its own class, and the first worker
    // of each list reports the shorter queue.
    let said = (&[of(1), of(2)][..], &[1, 2][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    host.loads = Some(reports(0));
    let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, encoders), (Some(0), vec![0]));
    assert_eq!(decode_of(&decision), Some(0));
    // Both prefills pair with both decodes, and prefill 1 with decode 1
    // first.
    let both: &[u32] = &[0, 1];
    host.partners = PartnerClasses::number_first(2, [(both, &[][..]), (both, &[1][..])]);
    let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, encoders), (Some(1), vec![1]));
    assert_eq!(decode_of(&decision), Some(1));
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

/// The encoders' cell is asked where an encoder's rows are read, and by no
/// answer that reads none. A request its pair alone answers and one with
/// no item that its wires decline each leave the cell of a list no
/// decision read empty: neither pays the fill of a list it does not read.
/// The first decision over the three lists fills it.
#[test]
fn an_answer_that_reads_no_encoder_fills_no_encoders_cell() {
    let said = (&[of(1)][..], &[1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    let host = Host::of(1, 1, 1);
    let (decision, _, _) = host.decide(said, &keys, Some(0), &mut placed);
    assert_eq!(gap(&decision), Some(SUBSET_HINT));
    assert!(!host.lists[2].filled(), "the pair's answer read no encoder");
    // The prefills' first worker is dialed over gRPC, and the request
    // states HTTP and no item.
    let (mut wired, _) = hearing(2, 2, 2);
    wired.lists[0] = first_dialed_over_grpc(&wired, 0);
    let of_two = (&[of(1), of(1)][..], &[1, 1][..]);
    let asks = Asks {
        wires: WireModes::HTTP,
        ..wired.asks(of_two)
    };
    let (entry, none, now) = (&wired.engine, EncodeItems::default(), Instant::now());
    let decided = entry.decide_encode_at(&asks, none, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert_eq!(declined_under(&decided.decision), Some(WIRES));
    assert!(
        !wired.lists[2].filled(),
        "the wires' refusal read no encoder"
    );
    let (_, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!((prefill, encoders), (Some(0), vec![0]));
    assert!(host.lists[2].filled(), "the decision read the encoders");
}

/// An encode request's pair is decided among the places its host states
/// for each of its two lists, as a pair's own decision is
/// ([`PairLists::decide`]), and each item's encoder is one at the encoders'
/// places. The worker at place 0 of each list reports the shorter queue, so
/// a list held to no place answers there, and a list held to its second
/// worker answers with that one. A request that states places of its own is
/// declined under the places' name still: one set of places holds no three
/// lists.
#[test]
fn an_encode_request_is_decided_among_the_places_of_each_of_its_lists() {
    let mut host = Host::of(2, 2, 2);
    host.loads = Some(reports(0));
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    let second = || Some(vec![1]);
    let stated = [
        ([None, None], None, (0, 0, 0)),
        ([second(), None], None, (1, 0, 0)),
        ([None, second()], None, (0, 1, 0)),
        ([None, None], second(), (0, 0, 1)),
        ([second(), second()], second(), (1, 1, 1)),
    ];
    for (places, encoder_places, (prefill, decode, encoder)) in stated {
        (host.places, host.encoder_places) = (places, encoder_places);
        let (decision, at, encoders) = host.decide(said, &keys, None, &mut placed);
        let answered = (at, decode_of(&decision), encoders);
        assert_eq!(answered, (Some(prefill), Some(decode), vec![encoder]));
    }
    // The request's own places name no one of the three lists.
    (host.places, host.encoder_places) = ([None, None], None);
    let lists = host.lists(said);
    let mut request = Request {
        id: b"",
        facts: RequestFacts {
            model: "model-a",
            phase: Phase::Prefill,
            ..RequestFacts::default()
        },
        constraints: Constraints {
            places: Some(&[0]),
            ..Constraints::default()
        },
    };
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let now = Instant::now();
    let decision = lists.decide(&host.engine, None, now, &mut request, items, &mut placed);
    assert_eq!(declined_under(&decision), Some("places"));
    assert!(placed.encoders().is_empty(), "no key stays placed");
}

/// A retry of an encode request is decided among the places of each list
/// on its wire, where its decode must take images. The first decode is on
/// the wire and takes no image, the second takes images off the wire, and
/// the third takes images on it: the retry's decode is the third. The
/// engine admits the decodes that take images and numbers the places with
/// them, so the place of the first decode opens no other.
///
/// The classic router decides a retry of an encode request as its pair
/// alone, held to the wire of the plan it kept
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:773-782`
/// and `:867-902` on main c0019b84c). It asks the decode's vision there
/// again where the encoders push (`:886-895`), and keeps the decode to the
/// workers on that wire that take images while a pair on the wire has one
/// (`:2431-2438`). Where the wire's decodes take none it hands the decode
/// pool as it is, and its dispatch refuses the side (`:2403-2407`). The
/// engine names that retry `places`. The name tells it from a retry whose
/// wire holds no decode, which is shed as its pair alone is.
#[test]
fn a_retry_of_an_encode_request_takes_a_decode_on_its_wire_that_takes_images() {
    let mut host = Host::of(2, 3, 1);
    // The host says of the second and the third decode that they take
    // images.
    let (decodes, blind) = (Arc::clone(host.lists[1].workers()), url(1, 0));
    host.lists[1] = PublishedList::with_facts(decodes, |worker| FixedFacts {
        vision: worker.url() != blind,
        ..FixedFacts::default()
    });
    let said = (&[of(1)][..], &[1, 1][..]);
    // No item: the first dispatch sent the encode jobs. The decode is held
    // to the ones that take images still.
    let retried = EncodeItems {
        decode_images: true,
        ..EncodeItems::default()
    };
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    let asked = || Request {
        id: b"",
        facts: RequestFacts {
            model: "model-a",
            phase: Phase::Prefill,
            attempt: 1,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    };
    let mut decode_under = |host: &Host| {
        let lists = host.lists(said);
        let decision = lists.decide(&host.engine, None, now, &mut asked(), retried, &mut placed);
        assert!(placed.encoders().is_empty(), "a retry places no item");
        (gap(&decision), decode_of(&decision))
    };
    // The wire holds the first and the third decode, and both prefills.
    host.places = [Some(vec![0, 1]), Some(vec![0, 2])];
    assert_eq!(decode_under(&host), (None, Some(2)));
    // It holds the first and the second.
    host.places = [Some(vec![0, 1]), Some(vec![0, 1])];
    assert_eq!(decode_under(&host), (None, Some(1)));
    // It holds the first alone: the request is held off every decode that
    // takes images, and no gap of the membership says so.
    host.places = [Some(vec![0, 1]), Some(vec![0])];
    assert_eq!(decode_under(&host), (Some("places"), None));
    // It holds no decode: the retry is shed as its pair alone is.
    host.places = [Some(vec![0, 1]), Some(Vec::new())];
    let pair = host.lists(said).pair;
    let alone = gap(&pair.decide(&host.engine, None, now, &mut asked(), None));
    assert!(alone.is_some_and(|stage| stage != "places"), "{alone:?}");
    assert_eq!(decode_under(&host), (alone, None));
}

/// A list's places that name no worker leave the request no worker of that
/// list. With every worker of the list admitted the pair's own pass reads
/// them, so the request is shed as its pair alone is shed
/// ([`PairLists::decide`]), and no key is placed.
#[test]
fn places_that_name_no_worker_shed_an_encode_request_as_they_shed_its_pair() {
    let mut host = Host::of(2, 2, 1);
    let said = (&[of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let now = Instant::now();
    let nobody = || Some(Vec::new());
    for held in [[nobody(), None], [None, nobody()], [nobody(), nobody()]] {
        let asked = || Request {
            id: b"",
            facts: RequestFacts {
                model: "model-a",
                phase: Phase::Prefill,
                seed: 41,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        };
        host.places = [None, None];
        let lists = host.lists(said);
        let decision = lists.decide(&host.engine, None, now, &mut asked(), items, &mut placed);
        assert!(decision.pick().is_some(), "not a pick: {decision:?}");
        host.places = held;
        let lists = host.lists(said);
        let decision = lists.decide(&host.engine, None, now, &mut asked(), items, &mut placed);
        let pair = lists
            .pair
            .decide(&host.engine, None, now, &mut asked(), None);
        let named = |decision: &Decision| match decision {
            Decision::Shed(shed) => (shed.reason, shed.emptied_by),
            other => panic!("not shed under {:?}: {other:?}", host.places),
        };
        assert_eq!(named(&decision), named(&pair));
        assert!(placed.encoders().is_empty(), "{:?}", host.places);
    }
}

/// An encode request's pair reads each side's standing vetoes beside that
/// side's list. The worker at one place of each list reports the shorter
/// queue, so with no veto the pair and the encoder are of that place. A
/// veto on the prefill there, with a hold or without, moves the prefill
/// alone: the decode and the encoder at that place still take the request.
/// A veto on the decode there moves the decode alone. Where the engine
/// admits a part of the prefills for the items, the mask is numbered with
/// the workers admitted: the third prefill is the second of the two an
/// encoder can hand to, and masking it leaves the first. A side masked
/// whole is paired by the pair's default, which lifts the liveness veto
/// and never takes a place a hold stands on: under a hold on every place
/// of a side no pair is made, and what was placed is emptied.
#[test]
fn an_encode_requests_pair_reads_each_sides_vetoes() {
    let mut host = Host::of(2, 2, 2);
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    for held in [false, true] {
        for masked in [0_usize, 1] {
            let other = 1 - masked;
            let on = |place: usize| {
                let mask = vec![place as u32];
                [mask.clone(), if held { mask } else { Vec::new() }]
            };
            host.loads = Some(reports(masked));
            host.vetoes = Default::default();
            let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
            let pair = (prefill, decode_of(&decision));
            assert_eq!(
                (pair, encoders),
                ((Some(masked), Some(masked)), vec![masked])
            );
            // The prefill at the place is vetoed, and no decode.
            host.vetoes = [on(masked), Default::default()];
            let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
            let pair = (prefill, decode_of(&decision));
            assert_eq!(pair, (Some(other), Some(masked)), "held {held}");
            assert_eq!(encoders, vec![masked], "held {held}");
            // The decode at the place is vetoed, and no prefill.
            host.vetoes = [Default::default(), on(masked)];
            let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
            let pair = (prefill, decode_of(&decision));
            assert_eq!(pair, (Some(masked), Some(other)), "held {held}");
            assert_eq!(encoders, vec![masked], "held {held}");
        }
    }
    // A side masked whole: the pair's default pairs past the mask, and the
    // key is placed. Under a hold on every place of the side no pair is
    // made, and what was placed is emptied.
    let whole = |held: bool| [vec![0, 1], if held { vec![0, 1] } else { Vec::new() }];
    for side in 0..2 {
        host.vetoes = Default::default();
        host.vetoes[side] = whole(false);
        let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
        assert_eq!(decision.trace().lifted, LiftedVeto::Liveness, "side {side}");
        assert!(prefill.is_some() && encoders.len() == 1, "side {side}");
        host.vetoes[side] = whole(true);
        let (decision, _, encoders) = host.decide(said, &keys, None, &mut placed);
        assert!(
            decision.pick().is_none() && encoders.is_empty(),
            "side {side}"
        );
    }
    host.vetoes = [whole(true), Default::default()];
    let (decision, _, _) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(gap(&decision), Some("healthy"));

    // The second prefill's class holds no encoder: the engine admits the
    // first and the third, and numbers the mask with them.
    let mut host = Host::of(3, 1, 1);
    let said = (&[of(1)][..], &[1, 2, 1][..]);
    for (masked, left) in [(2, 0), (0, 2)] {
        host.vetoes = [[vec![masked], Vec::new()], Default::default()];
        for _ in 0..8 {
            let (_, prefill, _) = host.decide(said, &keys, None, &mut placed);
            assert_eq!(prefill, Some(left), "masked {masked}");
        }
    }
    // A mask on the prefill left out masks neither of the two admitted.
    host.vetoes = [[vec![1], Vec::new()], Default::default()];
    let (_, prefill, _) = host.decide(said, &keys, None, &mut placed);
    assert!(matches!(prefill, Some(0 | 2)), "{prefill:?}");
}

/// The encoders read their own list's standing vetoes, stated beside it.
/// The worker at one place of each list reports the shorter queue, so with
/// no veto the pair and the encoder are of that place. A veto on the
/// encoder there, with a hold or without, moves the key alone: the pair
/// stays. A place past the list's end masks nothing. The encode side lifts
/// no veto: with every encoder masked the request is shed under the encode
/// side's stage, as with no encoder ready, and what was placed is emptied.
#[test]
fn an_encode_requests_encoders_read_their_own_lists_vetoes() {
    let mut host = Host::of(2, 2, 2);
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    for held in [false, true] {
        for masked in [0_usize, 1] {
            let other = 1 - masked;
            let at = Some(masked);
            host.loads = Some(reports(masked));
            for free in [[vec![], vec![]], [vec![7], vec![7]]] {
                host.encoder_vetoes = free;
                let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
                assert_eq!(
                    ((prefill, decode_of(&decision)), encoders),
                    ((at, at), vec![masked])
                );
            }
            // The encoder at the place is vetoed, and no worker of the pair.
            let mask = vec![masked as u32];
            host.encoder_vetoes = [mask.clone(), if held { mask } else { Vec::new() }];
            let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
            assert_eq!((prefill, decode_of(&decision)), (at, at), "held {held}");
            assert_eq!(encoders, vec![other], "held {held}");
        }
    }
    for holds in [vec![], vec![1], vec![0, 1]] {
        host.encoder_vetoes = [vec![0, 1], holds];
        let (decision, _, encoders) = host.decide(said, &keys, None, &mut placed);
        assert_eq!(
            gap(&decision),
            Some("encoders"),
            "{:?}",
            host.encoder_vetoes
        );
        assert!(encoders.is_empty(), "{:?}", host.encoder_vetoes);
    }
}

/// A mask names places of the encoder list, whatever part of the list a
/// key is offered. The second encoder is of another class than the prefill
/// names, so a key is offered the first and the third: masking the third
/// leaves the first, and masking the first leaves the third. A mask on the
/// encoder left out moves no key: the third encoder reports the shorter
/// queue and keeps the key, though it is the second of the two offered.
/// With both of the class masked the prefill is admitted no more than with
/// neither of them ready, and the two sheds name one stage.
#[test]
fn an_encoders_mask_names_places_of_the_list_under_a_class_that_holds_a_part_of_it() {
    let mut host = Host::of(1, 1, 3);
    let said = (&[of(1), of(2), of(1)][..], &[1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    for (masked, left) in [(2, 0), (0, 2)] {
        host.encoder_vetoes = [vec![masked], Vec::new()];
        for _ in 0..8 {
            let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
            assert_eq!(encoders, vec![left], "masked {masked}");
        }
    }
    // The encoder of the other class masked, with none and beside it: the
    // key stays on the third encoder, which reports the shorter queue.
    let queued = |place, queued| (url(2, place), one_rank_load(queued, 0.1, 1_000.0));
    host.loads = Some(LoadSnapshot::from_loads_for_test(vec![
        queued(0, 5),
        queued(2, 0),
    ]));
    for masked in [Vec::new(), vec![1]] {
        host.encoder_vetoes = [masked, Vec::new()];
        let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
        assert_eq!(encoders, vec![2], "{:?}", host.encoder_vetoes);
    }
    host.loads = None;
    // The prefill's class masked whole, beside an encoder of another
    // class that no veto holds back.
    host.encoder_vetoes = [vec![0, 2], Vec::new()];
    let (vetoed, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert!(vetoed.pick().is_none() && encoders.is_empty());
    host.encoder_vetoes = Default::default();
    for place in [0, 2] {
        host.worker(2, place).set_status(WorkerStatus::NotReady);
    }
    let (unready, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert!(unready.pick().is_none() && encoders.is_empty());
    assert!(gap(&vetoed).is_some());
    assert_eq!(gap(&vetoed), gap(&unready));
}

/// The pair is picked where an encoder is open, under the encoders' vetoes
/// as under their readiness: a prefill whose class is masked whole steps
/// aside for the prefill of a class that holds an open encoder, with a
/// hold or without.
#[test]
fn a_prefill_whose_encoders_are_all_vetoed_steps_aside() {
    let mut host = Host::of(2, 2, 3);
    // Encoders 0 and 1 hand to prefill 0, encoder 2 to prefill 1.
    let said = (&[of(1), of(1), of(2)][..], &[1, 2][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    for held in [false, true] {
        for (masked, prefill_left) in [(vec![0, 1], 1), (vec![2], 0)] {
            let holds = if held { masked.clone() } else { Vec::new() };
            host.encoder_vetoes = [masked.clone(), holds];
            for _ in 0..8 {
                let (_, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
                assert_eq!(prefill, Some(prefill_left), "{masked:?} held {held}");
                let class = said.1[prefill_left];
                assert_eq!(encoders.len(), 1);
                assert!(
                    encoders.iter().all(|&at| said.0[at].class == class),
                    "{masked:?} held {held}: {encoders:?}"
                );
                assert!(
                    encoders.iter().all(|&at| !masked.contains(&(at as u32))),
                    "{masked:?} held {held}: {encoders:?}"
                );
            }
        }
    }
}

/// The masks and the holds a stage of the encode profile was handed in its
/// constraints, each pair once, in the order it first met them.
type Handed = Arc<Mutex<Vec<(Vec<u32>, Vec<u32>)>>>;

/// A filter of a host's own build that keeps every row and notes the mask
/// and the hold its constraints state, and whether a row it was handed
/// carries a veto's bit of its own.
#[derive(Debug)]
struct ReadsVetoes(Handed, Arc<Mutex<bool>>);

impl Filter for ReadsVetoes {
    fn name(&self) -> &'static str {
        "reads-vetoes"
    }

    fn keep(&self, _: &RequestFacts<'_>, row: &Candidate, cx: &Constraints<'_>) -> bool {
        *self.1.lock() |= row.health.stalled || row.health.breaker_open;
        let handed = (cx.stalled.to_vec(), cx.held.to_vec());
        let mut seen = self.0.lock();
        if !seen.contains(&handed) {
            seen.push(handed);
        }
        true
    }
}

/// A host of one prefill and one decode under the stock encode document
/// whose encode profile holds the filter `filter` of the host's own build:
/// in its stock filters' stead where `alone`, and behind them else.
fn filtering(filter: &'static str, alone: bool, catalog: &StageRegistry, encoders: usize) -> Host {
    let mut document = stock();
    let encode = document.profiles.get_mut("encode");
    let filters = &mut encode.expect("the encode profile").filter;
    if alone {
        filters.clear();
    }
    filters.push(filter.into());
    let digest = document.digest().expect("a digest");
    let spec = AppliedSpec::validate_digested(&document, digest, catalog, &[]);
    Host::under(spec.expect("a valid document"), 1, 1, encoders)
}

/// A stage of the encode profile reads the encoders' mask and their hold
/// in its constraints. The first encoder is of another class than the
/// prefill names. Where the engine asks which encoders are open, the two
/// are numbered by the list. A key is offered the open encoders of the
/// class, and its stages read the two numbered with those: under a profile
/// whose filters keep a masked encoder open, the third encoder is the
/// second of the two offered. Under the stock filters the masked encoder is
/// not open, so a key is offered the second alone, under no mask. A hold
/// on a part of the mask is handed on as that part. A place past the
/// list's end is handed on as stated and names no encoder. The mask stays
/// beside the rows: no row a stage is handed carries a veto's bit.
#[test]
fn a_stage_of_the_encode_profile_reads_the_encoders_mask_and_hold() {
    let (handed, written) = (Handed::default(), Arc::new(Mutex::new(false)));
    let kept = (Arc::clone(&handed), Arc::clone(&written));
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("reads-vetoes", move || {
            Box::new(ReadsVetoes(Arc::clone(&kept.0), Arc::clone(&kept.1)))
        })
        .expect("a free filter name");
    let said = (&[of(2), of(1), of(1)][..], &[1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    let read = || std::mem::take(&mut *handed.lock());
    let none = || (Vec::new(), Vec::new());
    // The filter alone: no stage holds a masked encoder back.
    let mut host = filtering("reads-vetoes", true, &catalog, 3);
    host.encoder_vetoes = [vec![2], vec![2]];
    let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(encoders.len(), 1);
    assert_eq!(read(), [(vec![2], vec![2]), (vec![1], vec![1])]);
    host.encoder_vetoes = [vec![1, 9], Vec::new()];
    let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(encoders.len(), 1);
    assert_eq!(read(), [(vec![1, 9], Vec::new()), (vec![0], Vec::new())]);
    host.encoder_vetoes = [vec![1, 2], vec![2]];
    let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(encoders.len(), 1);
    assert_eq!(read(), [(vec![1, 2], vec![2]), (vec![0, 1], vec![1])]);
    host.encoder_vetoes = Default::default();
    let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(encoders.len(), 1);
    assert_eq!(read(), [none()]);
    // Behind the stock filters: the third encoder, masked under a hold,
    // is not open, and the second takes the key.
    let mut host = filtering("reads-vetoes", false, &catalog, 3);
    host.encoder_vetoes = [vec![2], vec![2]];
    let (_, _, encoders) = host.decide(said, &keys, None, &mut placed);
    assert_eq!(encoders, vec![1]);
    assert_eq!(read(), [(vec![2], vec![2]), none()]);
    assert!(!*written.lock());
}

/// A filter of a host's own build that opens a scope over the rows it is
/// handed, as a plugin's stage does, counts the scopes opened and keeps
/// every row.
#[derive(Debug)]
struct ShownRows(Arc<Mutex<u32>>);

impl Filter for ShownRows {
    fn name(&self) -> &'static str {
        "shown-rows"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self)
    }
}

impl DecisionScope for ShownRows {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        *self.0.lock() += 1;
    }

    fn close(&self) {}
}

/// An encode profile that shows its rows to a stage is declined for a
/// request with items under the stage `encode-scope`, with an encoder's
/// veto as with none: no scope is opened over the encoders and no key is
/// placed either way. So no mask need be written into an encoder's record.
#[test]
fn an_encode_profile_that_shows_its_rows_is_declined_under_a_veto_as_under_none() {
    let opened = Arc::new(Mutex::new(0));
    let counted = Arc::clone(&opened);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("shown-rows", move || {
            Box::new(ShownRows(Arc::clone(&counted)))
        })
        .expect("a free filter name");
    let mut host = filtering("shown-rows", false, &catalog, 2);
    let said = (&[of(1), of(1)][..], &[1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    for vetoes in [[vec![], vec![]], [vec![0], vec![]], [vec![0], vec![0]]] {
        host.encoder_vetoes = vetoes;
        let (decision, _, encoders) = host.decide(said, &keys, None, &mut placed);
        assert_eq!(
            declined_under(&decision),
            Some("encode-scope"),
            "{:?}",
            host.encoder_vetoes
        );
        assert!(encoders.is_empty(), "{:?}", host.encoder_vetoes);
        assert_eq!(*opened.lock(), 0, "{:?}", host.encoder_vetoes);
    }
}

/// An encode request that states a mask of its own is declined under the
/// stage `places`, as its pair's own decision declines it, and no key is
/// placed: the mask names places of one list, and each side's vetoes are
/// stated beside its list. So the request is answered alike at its first
/// decision and at its decode asked again, which is the pair's. What a
/// pair is not narrowed by is answered first: a hint of 0 sheds the same
/// request under its own stage. The same place stated beside the prefill
/// list is passed over, and the request is decided.
#[test]
fn an_encode_request_that_states_a_mask_of_its_own_is_declined_under_the_places_name() {
    let mut host = Host::of(2, 2, 2);
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    // A key is placed first, so a decline that left it would show.
    let (_, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
    assert!(prefill.is_some() && encoders.len() == 1);
    for own in [[vec![1], vec![]], [vec![1], vec![1]], [vec![], vec![1]]] {
        host.own = own;
        let (decision, _, encoders) = host.decide(said, &keys, None, &mut placed);
        assert_eq!(declined_under(&decision), Some("places"), "{:?}", host.own);
        assert!(encoders.is_empty(), "{:?}", host.own);
        let (decision, _, _) = host.decide(said, &keys, Some(0), &mut placed);
        assert_eq!(gap(&decision), Some("subset-hint"), "{:?}", host.own);
    }
    host.own = Default::default();
    host.vetoes = [[vec![1], Vec::new()], Default::default()];
    for _ in 0..8 {
        let (_, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
        assert_eq!((prefill, encoders.len()), (Some(0), 1));
    }
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

/// A host whose three lists are each two data-parallel engines of two
/// ranks, engine after engine: the worker at `place` of a list is a rank of
/// that list's engine `place / 2`.
fn ranked() -> Host {
    let mut host = Host::of(4, 4, 4);
    let types = [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode];
    for (list, worker_type) in types.into_iter().enumerate() {
        let rank = |place: usize| {
            let spec = WorkerSpec {
                worker_type,
                ..WorkerSpec::new(format!("grpc://list-{list}-engine-{}:9000", place / 2))
            };
            let built = <BasicWorkerBuilder>::from_spec(spec).dp_config(place % 2, 2);
            let worker = built.health_config(no_health_check()).build();
            worker.set_status(WorkerStatus::Ready);
            Arc::new(worker) as Arc<dyn Worker>
        };
        let ranks: Arc<[Arc<dyn Worker>]> = (0..4).map(rank).collect();
        host.lists[list] = PublishedList::with_facts(ranks, |_| FixedFacts::default());
    }
    host
}

/// An encode request's retry leaves every rank of the engines it tried,
/// each of its three lists by its own engines, as a pair's retry does over
/// its two. A tried prefill takes its engine's other rank out of the
/// prefills and leaves the decodes and the encoders as they were. A tried
/// decode and a tried encoder do the same on their lists, and the three
/// together move all three. Named the tried rank alone, each list kept its
/// engine's other rank.
#[test]
fn an_encode_requests_retry_leaves_every_rank_of_the_engines_it_tried() {
    use std::collections::BTreeSet;

    let mut host = ranked();
    let said = (&[of(1), of(1), of(1), of(1)][..], &[1, 1, 1, 1][..]);
    let keys = [Identity::plain("item")];
    // One rank of the first engine of each list.
    let rank = |list: usize, place: usize| worker_id(host.worker(list, place).as_ref());
    let (prefill, decode, encoder) = (rank(0, 1), rank(1, 0), rank(2, 1));
    let mut placed = EncodePlaced::new();
    // The engines a retry's prefill, decode and encoder stand on, over the
    // draws of the three lists.
    let mut landed = |tried: &[WorkerId]| {
        let mut on = [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()];
        host.tried = tried.to_vec();
        for seed in 0..32 {
            host.seed = seed;
            let (decision, prefill, encoders) = host.decide(said, &keys, None, &mut placed);
            on[0].insert(prefill.expect("a pick") / 2);
            on[1].insert(decode_of(&decision).expect("a pair") / 2);
            on[2].extend(encoders.iter().map(|&at| at / 2));
        }
        on
    };
    let (both, second) = (BTreeSet::from([0, 1]), BTreeSet::from([1]));
    let untried = [both.clone(), both.clone(), both.clone()];
    assert_eq!(landed(&[]), untried, "nobody tried");
    assert_eq!(
        landed(&[prefill]),
        [second.clone(), both.clone(), both.clone()],
        "the tried prefill's engine steps aside, and no other list moves"
    );
    assert_eq!(
        landed(&[decode]),
        [both.clone(), second.clone(), both.clone()],
        "the tried decode's engine steps aside, and no other list moves"
    );
    assert_eq!(
        landed(&[encoder]),
        [both.clone(), both, second.clone()],
        "the tried encoder's engine steps aside, and no other list moves"
    );
    assert_eq!(
        landed(&[prefill, decode, encoder]),
        [second.clone(), second.clone(), second],
        "three tried workers move all three lists"
    );
}

/// The workers an encode request's retry is decided to have tried stand in
/// the calling thread's own list, as a single request's do. A first
/// attempt takes no list, and neither does a retry over workers that each
/// stand alone. A retry over ranks grows the list once and hands it back
/// behind its decision: the retries behind it allocate nothing for it.
#[test]
fn an_encode_requests_retry_hands_the_threads_list_back() {
    let mut host = ranked();
    let said = (&[of(1), of(1), of(1), of(1)][..], &[1, 1, 1, 1][..]);
    let keys = [Identity::plain("item")];
    let mut placed = EncodePlaced::new();
    // Where the thread's list stands and what it holds room for.
    let kept = || {
        let list = single::TRIED.take();
        let stands = (list.as_ptr(), list.capacity());
        single::TRIED.set(list);
        stands
    };
    let mut asked = |host: &Host| {
        let (decision, ..) = host.decide(said, &keys, None, &mut placed);
        assert!(decision.pick().is_some(), "a pair");
    };
    asked(&host);
    assert_eq!(kept().1, 0, "a first attempt takes no list");
    let mut alone = Host::of(4, 4, 4);
    alone.tried = vec![worker_id(alone.worker(0, 1).as_ref())];
    asked(&alone);
    assert_eq!(kept().1, 0, "nor a retry over workers that stand alone");
    host.tried = vec![worker_id(host.worker(0, 1).as_ref())];
    asked(&host);
    let grown = kept();
    assert!(grown.1 >= 2, "the tried rank and its engine's other rank");
    asked(&host);
    assert_eq!(kept(), grown, "the list is kept as it grew");
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

/// A host's request as the engine's entries take it: made at `phase` over
/// `lists`, held to `wires`, with whether its decode is sent `later`,
/// whether it states an image (`images`) or a video (`video`), and the
/// snapshot of `loads` its host gives it, where it gives one.
struct Asks<'a> {
    lists: WorkerLists<'a>,
    phase: Phase,
    wires: WireModes,
    later: bool,
    images: bool,
    video: bool,
    loads: Option<&'a Arc<LoadSnapshot>>,
}

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        "model-a"
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        Request {
            id: b"",
            facts: RequestFacts {
                model: "model-a",
                phase: self.phase,
                media: MediaCounts {
                    images: u16::from(self.images),
                    video: u16::from(self.video),
                    ..MediaCounts::NONE
                },
                ..RequestFacts::default()
            },
            constraints: Constraints {
                wire_modes: self.wires,
                decode_later: self.later,
                ..Constraints::default()
            },
        }
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.lists
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads
    }
}

/// The side of each issue a stage heard.
type Heard = Arc<Mutex<Vec<Side>>>;

/// A picker that picks as [`Best`] and keeps the side of each issue it
/// hears.
#[derive(Debug)]
struct Hears(Heard);

impl Lifecycle for Hears {
    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push(lease.side);
    }

    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }
}

impl Picker for Hears {
    fn name(&self) -> &'static str {
        "hears"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        rows: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Best.pick(facts, rows, kept, scores, band, out);
    }
}

/// A host under the stock encode document whose encode side picks through
/// a stage that keeps the issues it hears, and those issues. The engine
/// tells every profile's picker each word, so one stage hears them all.
fn hearing(prefills: usize, decodes: usize, encoders: usize) -> (Host, Heard) {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("hears", move || Box::new(Hears(Arc::clone(&kept))))
        .expect("a free picker name");
    let mut document = stock();
    let encode = document.profiles.get_mut("encode");
    encode.expect("the encode profile").pick = "hears".into();
    let digest = document.digest().expect("a digest");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    let host = Host::under(spec.expect("a valid document"), prefills, decodes, encoders);
    (host, heard)
}

/// The places of the pair a decision's first target names.
fn pair_of(decision: &Decision) -> Option<(usize, usize)> {
    let first = decision.pick()?.targets.first()?;
    Some((first.position as usize, decode_of(decision)?))
}

/// The engine's entry answers as the three lists' own decision and issues
/// the pair's lease alone: of two engines made alike, the one asked through
/// the entry names, call by call, the pair and each key's encoder the other
/// names, takes one lease id a request and tells the prefill's issue and
/// the decode's. No word is told of an encoder. The entry that issues
/// nothing names the same and takes no id.
#[test]
fn the_engines_entry_answers_as_the_three_lists_and_issues_the_pairs_lease_alone() {
    let ((alone, unheard), (leasing, heard)) = (hearing(2, 2, 2), hearing(2, 2, 2));
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item-a"), Identity::plain("item-b")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    // Each engine's answer goes into a list of its own.
    let (mut placed, mut theirs) = (EncodePlaced::new(), EncodePlaced::new());
    let now = Instant::now();
    let places = |placed: &EncodePlaced| -> Vec<usize> {
        let encoders = placed.encoders().iter();
        encoders.map(|target| target.position as usize).collect()
    };
    for _ in 0..3 {
        let (decision, _, encoders) = alone.decide(said, &keys, None, &mut theirs);
        let asks = leasing.asks(said);
        let entry = &leasing.engine;
        let decided = entry.decide_encode_at(&asks, items, &mut placed, || now);
        let (decided, later) = decided.expect("a decision");
        assert!(
            decided.lease.is_some() && !later,
            "the pick's pair is leased"
        );
        assert_eq!(pair_of(&decided.decision), pair_of(&decision));
        assert_eq!((places(&placed), encoders.len()), (encoders, 2));
    }
    assert_eq!(leasing.engine.leases.0.load(Ordering::Relaxed), 3);
    assert_eq!(*heard.lock(), [Side::Prefill, Side::Decode].repeat(3));
    // The entry that issues nothing: the fourth decision of each engine,
    // into a list that holds nothing yet.
    let (decision, _, encoders) = alone.decide(said, &keys, None, &mut theirs);
    let asks = leasing.asks(said);
    let entry = &leasing.engine;
    let mut placed = EncodePlaced::new();
    let unleased = entry.decide_encode_without_lease(&asks, items, &mut placed, now);
    assert_eq!(pair_of(&unleased.expect("a decision")), pair_of(&decision));
    assert_eq!((places(&placed), encoders.len()), (encoders, 2));
    assert_eq!(leasing.engine.leases.0.load(Ordering::Relaxed), 3);
    assert_eq!((heard.lock().len(), unheard.lock().len()), (6, 0));
}

/// The engine's entries read the load reports a pair's first decision
/// reads: the snapshot the request gives for the call, and for a request
/// that gives none the newest the engine's own task keeps at hand. Under
/// the stock document each of the three lists gives its worker whose queue
/// is the shorter in the reports read, whichever it is, through the entry
/// that issues the pair's lease and through the one that issues nothing. A
/// snapshot the request gives is read in place of the engine's.
///
/// The classic router asks its encode policy for an item's encoder
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2338`
/// on main 7894b7595), and the stock policy scores the two workers it draws
/// by the reports of its own snapshot
/// (`model_gateway/src/policies/power_of_two.rs:136-137`,
/// `model_gateway/src/policies/least_load.rs:918` and `:450`).
#[tokio::test(start_paused = true)]
async fn the_engines_entries_read_the_load_reports_a_pairs_first_decision_reads() {
    let mut host = Host::of(2, 2, 2);
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let now = Instant::now();
    // The place both entries name in each of the three lists: the pair's
    // two, and the item's encoder.
    let named = |host: &Host| {
        let (asks, mut placed) = (host.asks(said), EncodePlaced::new());
        let leased = host
            .engine
            .decide_encode_at(&asks, items, &mut placed, || now);
        let (leased, _) = leased.expect("a decision");
        let leased = (pair_of(&leased.decision), placed.encoders()[0].position);
        let alone = host
            .engine
            .decide_encode_without_lease(&asks, items, &mut placed, now);
        let alone = pair_of(&alone.expect("a decision"));
        assert_eq!(leased, (alone, placed.encoders()[0].position));
        leased
    };
    let at = |short: usize| (Some((short, short)), short as u32);
    for short in [0, 1] {
        host.loads = Some(reports(short));
        assert_eq!(named(&host), at(short));
    }
    // No snapshot given: the newest the engine's task read from the channel
    // its host handed, turned once behind each publication.
    host.loads = None;
    let mut channel = None;
    let feed = TestFeed::attached_to(|loads| channel = loads);
    host.engine.set_loads(channel.expect("the feed's channel"));
    for short in [1, 0] {
        feed.publish(&queues(short).into_iter().collect());
        let step = Duration::from_millis(1);
        assert!(tokio::time::timeout(step, host.engine.run()).await.is_err());
        assert_eq!(named(&host), at(short));
    }
    host.loads = Some(reports(1));
    assert_eq!(named(&host), at(1));
}

/// A request the encode side sheds, and one its wires decline, issue
/// nothing: no lease, no id, no word, and no key stays placed. The pair's
/// two lists are their host's to hand on the request's wires. With one
/// worker of either dialed over another wire than the rest, a request that
/// states one wire alone is declined through both entries, with an item
/// and with none, and one that states both is decided.
///
/// The classic router tests the wire a worker at a time on a pair's two
/// sides (`model_gateway/src/routers/common/placement.rs:1301-1302` on main
/// 384b8fd6b): it passes over the one worker and serves a request held to
/// the other workers' wire by them. The engine declines that request, as it
/// declines a pair one of whose workers is dialed on a wire the request
/// does not state.
#[test]
fn a_request_shed_or_declined_by_its_wires_issues_nothing_and_places_no_key() {
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let now = Instant::now();
    // No encoder is ready: the encode side's shed, as the lists answer it.
    let (host, heard) = hearing(2, 2, 2);
    host.worker(2, 0).set_status(WorkerStatus::NotReady);
    host.worker(2, 1).set_status(WorkerStatus::NotReady);
    let asks = host.asks(said);
    let entry = &host.engine;
    let decided = entry.decide_encode_at(&asks, items, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert_eq!(gap(&decided.decision), Some("encoders"));
    assert!(decided.lease.is_none() && placed.encoders().is_empty());
    assert_eq!(entry.leases.0.load(Ordering::Relaxed), 0);
    assert!(heard.lock().is_empty());
    // In each of the pair's two lists in turn, the first worker is dialed
    // over gRPC. Every other worker of the three lists is dialed over HTTP.
    for list in 0..2 {
        let (mut host, heard) = hearing(2, 2, 2);
        host.lists[list] = first_dialed_over_grpc(&host, list);
        let held_to = |wires| Asks {
            wires,
            ..host.asks(said)
        };
        let entry = &host.engine;
        for wires in [WireModes::HTTP, WireModes::GRPC, WireModes::ZMQ] {
            // A request that states both wires is decided, and its key is
            // placed.
            let both = held_to(WireModes::HTTP.union(WireModes::GRPC));
            let decided = entry.decide_encode_at(&both, items, &mut placed, || now);
            let (decided, _) = decided.expect("a decision");
            assert!(decided.lease.is_some(), "list {list}");
            assert_eq!(placed.encoders().len(), 1, "list {list}");
            // One that states one wire is declined, and the key placed
            // before it is gone.
            let asks = held_to(wires);
            let leased = entry.decide_encode_at(&asks, items, &mut placed, || now);
            let (leased, _) = leased.expect("a decision");
            let named = declined_under(&leased.decision);
            assert_eq!(named, Some(WIRES), "list {list}, {wires:?}");
            assert!(leased.lease.is_none(), "list {list}, {wires:?}");
            assert!(placed.encoders().is_empty(), "list {list}, {wires:?}");
            let unleased = entry.decide_encode_without_lease(&asks, items, &mut placed, now);
            let named = declined_under(&unleased.expect("a decision"));
            assert_eq!(named, Some(WIRES), "list {list}, {wires:?}");
        }
        // A request with no item held to HTTP reads the same two lists.
        let (asks, none) = (held_to(WireModes::HTTP), EncodeItems::default());
        let decided = entry.decide_encode_at(&asks, none, &mut placed, || now);
        let (decided, _) = decided.expect("a decision");
        assert_eq!(declined_under(&decided.decision), Some(WIRES));
        assert!(decided.lease.is_none(), "list {list}");
        // The decided requests alone took an id and were told.
        assert_eq!(entry.leases.0.load(Ordering::Relaxed), 3);
        assert_eq!(*heard.lock(), [Side::Prefill, Side::Decode].repeat(3));
    }
}

/// The encoders are narrowed by the decision, where the pair's two lists
/// are their host's to narrow: an encoder is named by its place in its
/// list, and what its host says of it is laid by that place, so the host
/// has no narrowed list to hand. Of two encoders the first is dialed over
/// gRPC and reports the shorter queue, and the second is dialed over HTTP,
/// as the pair's lists are. A request with an item that states both wires
/// takes the first. One held to HTTP is decided through both entries, and
/// its item is placed on the second at every call. One held to a wire the
/// pair's lists are not dialed on is declined by them.
///
/// The classic router tests an encoder's wire with the rest of what the
/// request asks of it, and keeps the encoders that pass
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2138`
/// and `:2147` on main 384b8fd6b).
#[test]
fn an_item_is_placed_on_an_encoder_dialed_on_a_wire_its_request_states() {
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let now = Instant::now();
    let (mut host, heard) = hearing(2, 2, 2);
    host.lists[2] = first_dialed_over_grpc(&host, 2);
    let queue = |url: &str, queued| (url.to_owned(), one_rank_load(queued, 0.1, 1_000.0));
    let queues = vec![queue(host.worker(2, 0).url(), 0), queue(&url(2, 1), 90_000)];
    host.loads = Some(LoadSnapshot::from_loads_for_test(queues));
    let held_to = |wires| Asks {
        wires,
        ..host.asks(said)
    };
    let entry = &host.engine;
    let places = |placed: &EncodePlaced| -> Vec<u32> {
        let encoders = placed.encoders().iter();
        encoders.map(|target| target.position).collect()
    };
    let both = held_to(WireModes::HTTP.union(WireModes::GRPC));
    let decided = entry.decide_encode_at(&both, items, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert!(decided.lease.is_some());
    assert_eq!(places(&placed), [0], "the shorter queue");
    let http = held_to(WireModes::HTTP);
    for _ in 0..3 {
        let leased = entry.decide_encode_at(&http, items, &mut placed, || now);
        let (leased, _) = leased.expect("a decision");
        assert!(leased.lease.is_some());
        assert_eq!(places(&placed), [1]);
        let unleased = entry.decide_encode_without_lease(&http, items, &mut placed, now);
        assert!(unleased.expect("a decision").pick().is_some());
        assert_eq!(places(&placed), [1]);
    }
    for wires in [WireModes::GRPC, WireModes::ZMQ] {
        let asks = held_to(wires);
        let leased = entry.decide_encode_at(&asks, items, &mut placed, || now);
        let (leased, _) = leased.expect("a decision");
        let named = declined_under(&leased.decision);
        assert_eq!(named, Some(WIRES), "{wires:?}");
        assert!(leased.lease.is_none() && placed.encoders().is_empty());
    }
    // The decided requests alone took an id and were told.
    assert_eq!(entry.leases.0.load(Ordering::Relaxed), 4);
    assert_eq!(*heard.lock(), [Side::Prefill, Side::Decode].repeat(4));
}

/// Where the encoders' list holds encoders and none is dialed on a wire the
/// request states, a request with an item is declined under the wires'
/// name through both entries: it issues nothing and no key is placed. A
/// request with no item reads no encoder's wire, and is decided over its
/// pair. A list of no encoder is the encode side's own gap, whatever wire
/// the request states.
///
/// The classic router refuses the request with an item too, under a name
/// of its own, since no encoder passes its test of the wire
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2149-2176`
/// on main 384b8fd6b). It holds a request with no item to no encoder
/// (`:2149`, `:2187` and `:2215`).
#[test]
fn encoders_all_on_other_wires_decline_a_request_with_an_item_under_the_wires_name() {
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let now = Instant::now();
    let (mut host, heard) = hearing(2, 2, 2);
    let encoders: Arc<[Arc<dyn Worker>]> = Arc::new([over_grpc(2, 0), over_grpc(2, 1)]);
    host.lists[2] = PublishedList::new(encoders);
    let entry = &host.engine;
    let http = Asks {
        wires: WireModes::HTTP,
        ..host.asks(said)
    };
    let leased = entry.decide_encode_at(&http, items, &mut placed, || now);
    let (leased, _) = leased.expect("a decision");
    assert_eq!(declined_under(&leased.decision), Some(WIRES));
    assert!(leased.lease.is_none() && placed.encoders().is_empty());
    let unleased = entry.decide_encode_without_lease(&http, items, &mut placed, now);
    assert_eq!(declined_under(&unleased.expect("a decision")), Some(WIRES));
    assert_eq!(entry.leases.0.load(Ordering::Relaxed), 0);
    assert!(heard.lock().is_empty());
    // With no item the same request is its pair's, and takes no encoder.
    let none = EncodeItems::default();
    let decided = entry.decide_encode_at(&http, none, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert!(decided.lease.is_some() && placed.encoders().is_empty());
    // A request that states no wire takes an encoder of the list.
    let free = host.asks(said);
    let decided = entry.decide_encode_at(&free, items, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert!(decided.lease.is_some() && placed.encoders().len() == 1);
    // No encoder at all: the encode side's gap, by its own name.
    let (alone, _) = hearing(1, 1, 0);
    let asks = Asks {
        wires: WireModes::HTTP,
        ..alone.asks((&[], &[1]))
    };
    let decided = (alone.engine).decide_encode_at(&asks, items, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert_eq!(gap(&decided.decision), Some("encoders"));
}

/// A ready worker of list `list` at `place` that is dialed over gRPC.
fn over_grpc(list: usize, place: usize) -> Arc<dyn Worker> {
    let dialed = <BasicWorkerBuilder>::new(format!("grpc://list-{list}-worker-{place}:9000"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .model(ModelCard::new("model-a"))
        .health_config(no_health_check())
        .build();
    dialed.set_status(WorkerStatus::Ready);
    Arc::new(dialed)
}

/// `host`'s list `list` with its first worker dialed over gRPC, as the
/// host publishes a list: its other worker stays.
fn first_dialed_over_grpc(host: &Host, list: usize) -> PublishedList {
    let stays = Arc::clone(host.worker(list, 1));
    let workers: Arc<[Arc<dyn Worker>]> = Arc::new([over_grpc(list, 0), stays]);
    PublishedList::new(workers)
}

/// A decision that panics behind its first placed key leaves no key
/// placed: the encode side's picker places the first key and panics at the
/// second. The engine answers a fault, takes no lease id and tells no stage.
#[test]
fn a_decision_that_panics_behind_a_placed_key_leaves_no_key_placed() {
    #[derive(Debug)]
    struct Falls(AtomicUsize);

    impl Lifecycle for Falls {}

    impl Picker for Falls {
        fn name(&self) -> &'static str {
            "falls"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            rows: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            let picks = self.0.fetch_add(1, Ordering::Relaxed);
            assert!(picks == 0, "the second key's pick");
            Best.pick(facts, rows, kept, scores, band, out);
        }
    }

    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("falls", || Box::new(Falls(AtomicUsize::new(0))))
        .expect("a free picker name");
    let mut document = stock();
    let encode = document.profiles.get_mut("encode");
    encode.expect("the encode profile").pick = "falls".into();
    let digest = document.digest().expect("a digest");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    let host = Host::under(spec.expect("a valid document"), 1, 1, 2);
    let said = (&[of(1), of(1)][..], &[1][..]);
    let keys = [Identity::plain("item-a"), Identity::plain("item-b")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let asks = host.asks(said);
    let now = Instant::now();
    let decided = (host.engine).decide_encode_at(&asks, items, &mut placed, || now);
    assert_eq!(decided.err(), Some(DecisionRefused::Faulted));
    assert!(placed.encoders().is_empty(), "no key stays placed");
    assert_eq!(host.engine.leases.0.load(Ordering::Relaxed), 0);
}

/// The decode of `first`, a pair's first decision whose decode is sent
/// later, asked again of `host`'s engine over the lists `asks` hands, a
/// millisecond behind `now`.
fn decode_again(host: &Host, asks: &Asks<'_>, first: DecideOutcome, now: Instant) -> DecideOutcome {
    let lease = first.lease.expect("the pair's lease").lease;
    let picked = first.decision.pick().expect("a pick");
    let target = *picked.targets.first().expect("a target");
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    let call = now + Duration::from_millis(1);
    let decode = (host.engine).decide_decode_under(GatewayId::LOCAL, asks, &held, call);
    decode.expect("a decision")
}

/// An encode request whose decode is sent later: the entry says so with
/// its answer and the prefill's issue alone is told. Its decode asked again
/// over the three lists keeps the prefill and is issued under the pair's
/// lease.
#[test]
fn an_encode_requests_decode_is_asked_again_over_its_three_lists() {
    let (host, heard) = hearing(2, 2, 1);
    let said = (&[of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let now = Instant::now();
    let asks = Asks {
        later: true,
        ..host.asks(said)
    };
    let first = (host.engine).decide_encode_at(&asks, items, &mut placed, || now);
    let (first, later) = first.expect("a decision");
    assert!(later && first.lease.is_some(), "the prefill is issued");
    assert_eq!(*heard.lock(), [Side::Prefill]);
    let (pair, id) = (
        pair_of(&first.decision).expect("a pair"),
        first.lease.map(|l| l.lease.id),
    );
    let decode = decode_again(&host, &asks, first, now);
    assert_eq!(decode.lease.map(|l| l.lease.id), id, "the pair's lease");
    assert_eq!(pair_of(&decode.decision).map(|pair| pair.0), Some(pair.0));
    assert_eq!(*heard.lock(), [Side::Prefill, Side::Decode]);
    assert_eq!(host.engine.leases.0.load(Ordering::Relaxed), 1);
}

/// What a decode takes is its host's word alone: a request that asks images
/// of its decode is shed under the decodes' stage where its host says so of
/// no decode, and takes the one decode its host says so of.
///
/// The classic router holds a decode to its image label under one runtime
/// alone, and every other decode passes
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2192-2196`
/// on main 4cefb0de5). For a request that asks images of its decode it
/// refuses where a pair exists and no pair's decode passes (`:2202-2207`),
/// and picks among the decodes that pass (`:2211-2216`). A host that says
/// of each decode what that rule answers reads the same.
#[test]
fn a_decode_takes_images_where_its_host_says_so() {
    let mut host = Host::of(1, 2, 1);
    let said = (&[of(1)][..], &[1][..]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        decode_images: true,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let now = Instant::now();
    // The host says of no decode that it takes images.
    let asks = host.asks(said);
    let decided = (host.engine).decide_encode_at(&asks, items, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert_eq!(gap(&decided.decision), Some("decode-vision"));
    assert!(decided.lease.is_none() && placed.encoders().is_empty());
    // It says so of the second decode alone, and every request takes it.
    host.says_the_decode_takes_images(1);
    for _ in 0..4 {
        let asks = host.asks(said);
        let decided = (host.engine).decide_encode_at(&asks, items, &mut placed, || now);
        let (decided, _) = decided.expect("a decision");
        assert_eq!(decode_of(&decided.decision), Some(1));
        assert_eq!(placed.encoders().len(), 1);
    }
}

/// A request that states the later decode is declined under `decode-later`,
/// through both entries, where its decode is held to the ones that take
/// images: by its demand, or by image references beside an image. A decode
/// asked again takes no items, so that decision would hold it to no decode
/// that takes images. Nothing is issued, no id is taken, no word is told,
/// and the key an earlier decision placed is gone. The decline reads the
/// request's own facts: it stands where no decode of the lists takes
/// images. With its decode sent with its prefill the request is decided,
/// and so is one that states the later decode under no such hold.
///
/// The classic router picks a decode a second time in a retry alone, and
/// states there again the request's image demand of its decode and its
/// media references
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:827-838`
/// on main 3d0315f3d). It prefers a decode that takes images for
/// references beside an image only where a pair has one (`:2211-2212`).
#[test]
fn a_request_whose_decode_is_held_to_images_and_sent_later_is_declined() {
    let (mut host, heard) = hearing(1, 2, 1);
    host.says_the_decode_takes_images(1);
    // The one encoder takes references.
    let takes = [EncoderFacts {
        references: true,
        ..of(1)
    }];
    let said = (&takes[..], &[1][..]);
    let keys = [Identity::plain("item")];
    let plain = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let demands = EncodeItems {
        decode_images: true,
        ..plain
    };
    let refers = EncodeItems {
        references: true,
        ..plain
    };
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    let entry = &host.engine;
    let with_image = |later| Asks {
        later,
        images: true,
        ..host.asks(said)
    };
    for (held, asks) in [(demands, host.asks(said)), (refers, with_image(false))] {
        // Sent with its prefill: decided on the decode that takes images,
        // its key placed, its pair leased.
        let decided = entry.decide_encode_at(&asks, held, &mut placed, || now);
        let (decided, _) = decided.expect("a decision");
        assert_eq!(decode_of(&decided.decision), Some(1));
        assert!(decided.lease.is_some() && placed.encoders().len() == 1);
        // Sent later: declined whole by the entry that issues nothing, and
        // that key is gone.
        let later = Asks {
            later: true,
            ..asks
        };
        let unleased = entry.decide_encode_without_lease(&later, held, &mut placed, now);
        let unleased = unleased.expect("a decision");
        assert_eq!(declined_under(&unleased), Some("decode-later"));
        assert!(placed.encoders().is_empty(), "no key stays placed");
        // And by the entry that issues a pair's lease, behind a key placed.
        let decided = entry.decide_encode_at(&asks, held, &mut placed, || now);
        assert!(decided.is_ok() && placed.encoders().len() == 1);
        let leased = entry.decide_encode_at(&later, held, &mut placed, || now);
        let (leased, _) = leased.expect("a decision");
        assert_eq!(declined_under(&leased.decision), Some("decode-later"));
        assert!(leased.lease.is_none() && placed.encoders().is_empty());
    }
    // Two decisions a hold were issued, and no decline said a word.
    assert_eq!(entry.leases.0.load(Ordering::Relaxed), 4);
    let issued = [Side::Prefill, Side::Decode].repeat(4);
    assert_eq!(*heard.lock(), issued);
    // Under no hold the later decode is stated: a request that asks no
    // images of its decode, and references that state no image.
    for (free, asks) in [(plain, with_image(true)), (refers, host.asks(said))] {
        let later = Asks {
            later: true,
            ..asks
        };
        let decided = entry.decide_encode_at(&later, free, &mut placed, || now);
        let (decided, stated) = decided.expect("a decision");
        assert!(decided.lease.is_some() && stated);
    }
    // The decline reads no list: where no decode takes images, references
    // beside an image are decided with their prefill and declined later.
    let none = Host::of(1, 2, 1);
    let entry = &none.engine;
    let asks = Asks {
        images: true,
        ..none.asks(said)
    };
    let decided = entry.decide_encode_at(&asks, refers, &mut placed, || now);
    assert!(decided.expect("a decision").0.lease.is_some());
    let later = Asks {
        later: true,
        ..asks
    };
    let declined = entry.decide_encode_at(&later, refers, &mut placed, || now);
    let (declined, _) = declined.expect("a decision");
    assert_eq!(declined_under(&declined.decision), Some("decode-later"));
}

/// References beside a video hold the decode to the ones that take images
/// as references beside an image do: sent with its prefill the request is
/// decided on the decode its host says takes images, where the other
/// reports the shorter queue and takes the same references with no pixel
/// beside them. One that states
/// the later decode is declined under `decode-later`. The classic router
/// asks vision for an image or a video alike
/// (`model_gateway/src/routers/grpc/multimodal/capability.rs:325-344` on
/// main d6517a4fe).
#[test]
fn references_beside_a_video_hold_the_decode_as_beside_an_image() {
    let mut host = Host::of(1, 2, 1);
    host.says_the_decode_takes_images(1);
    let takes = [EncoderFacts {
        references: true,
        ..of(1)
    }];
    let said = (&takes[..], &[1][..]);
    let keys = [Identity::plain("item")];
    let refers = EncodeItems {
        keys: &keys,
        references: true,
        ..EncodeItems::default()
    };
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    // The first decode reports the shorter queue: under no hold it is the
    // pair's.
    host.loads = Some(reports(0));
    let entry = &host.engine;
    let free = entry.decide_encode_at(&host.asks(said), refers, &mut placed, || now);
    let (free, _) = free.expect("a decision");
    assert_eq!(decode_of(&free.decision), Some(0), "no pixel: no hold");
    let with_video = |later| Asks {
        later,
        video: true,
        ..host.asks(said)
    };
    let decided = entry.decide_encode_at(&with_video(false), refers, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert_eq!(decode_of(&decided.decision), Some(1));
    assert!(decided.lease.is_some() && placed.encoders().len() == 1);
    let later = entry.decide_encode_at(&with_video(true), refers, &mut placed, || now);
    let (later, _) = later.expect("a decision");
    assert_eq!(declined_under(&later.decision), Some("decode-later"));
    assert!(later.lease.is_none() && placed.encoders().is_empty());
}

/// Three lists through the engine's other entries are the pair's two: a
/// request that states no item is decided as its pair, and so is its decode
/// asked again, with a lease and without one. A maker of another phase, of
/// a pair's two lists or of none asks the encode entries for a side on its
/// own, and no key stays placed.
#[test]
fn three_lists_are_the_pairs_two_for_the_other_entries_and_other_lists_are_refused() {
    let (pairs, threes) = (Host::of(2, 2, 1), Host::of(2, 2, 1));
    let said = (&[of(1)][..], &[1, 1][..]);
    let now = Instant::now();
    let whole = |host: &Host, asks: &Asks<'_>| {
        let decided = host.engine.decide_under(GatewayId::LOCAL, asks, now);
        let decided = decided.expect("a decision").whole();
        decided.expect("a whole decision").0
    };
    for _ in 0..3 {
        // Two engines made alike, one handed the pair's two lists, one the
        // three.
        let pair = whole(&pairs, &pairs.asks_as_a_pair(said));
        let three = whole(&threes, &threes.asks(said));
        assert!(three.lease.is_some() && pair_of(&three.decision).is_some());
        assert_eq!(pair_of(&three.decision), pair_of(&pair.decision));
    }
    // The decode asked again, behind a first decision whose decode is sent
    // later: each engine decides it over the lists it was handed.
    let again = |host: &Host, asks: &Asks<'_>| {
        let decode = decode_again(host, asks, whole(host, asks), now);
        assert!(decode.lease.is_some(), "the decode is issued");
        pair_of(&decode.decision)
    };
    let two = Asks {
        later: true,
        ..pairs.asks_as_a_pair(said)
    };
    let three = Asks {
        later: true,
        ..threes.asks(said)
    };
    assert_eq!(again(&threes, &three), again(&pairs, &two));
    // The two entries that issue nothing answer alike: a request with no
    // item, then its decode again behind the target that decision named.
    let unleased = |host: &Host, asks: &Asks<'_>| {
        let first = host.engine.decide_without_lease(asks, now);
        let (first, _) = first.expect("a decision");
        let picked = first.pick().expect("a pick");
        let held = *picked.targets.first().expect("a target");
        let call = now + Duration::from_millis(1);
        let again = host.engine.decide_decode_without_lease(asks, &held, call);
        (pair_of(&first), pair_of(&again.expect("a decision")))
    };
    let (of_two, of_three) = (unleased(&pairs, &two), unleased(&threes, &three));
    assert!(of_three.0.is_some() && of_three.1.is_some());
    assert_eq!(of_three, of_two);
    // The encode entries take the three lists at the prefill phase alone.
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let entry = &threes.engine;
    let others = [
        threes.asks_as_a_pair(said),
        Asks {
            lists: WorkerLists::None,
            ..threes.asks(said)
        },
        Asks {
            phase: Phase::Single,
            ..threes.asks(said)
        },
        Asks {
            phase: Phase::Decode,
            ..threes.asks(said)
        },
        Asks {
            phase: Phase::Encode,
            ..threes.asks(said)
        },
    ];
    for other in &others {
        // A key placed by a decision is gone behind a refusal.
        let asks = threes.asks(said);
        let decided = entry.decide_encode_at(&asks, items, &mut placed, || now);
        assert!(decided.is_ok() && placed.encoders().len() == 1);
        let leased = entry.decide_encode_at(other, items, &mut placed, || now);
        assert_eq!(leased.err(), Some(DecisionRefused::PairSide));
        assert!(placed.encoders().is_empty(), "no key stays placed");
        let unleased = entry.decide_encode_without_lease(other, items, &mut placed, now);
        assert_eq!(unleased.err(), Some(DecisionRefused::PairSide));
    }
}

/// Equal item keys share an encoder under consistent hashing.
/// Different keys reach both encoders.
/// Excluding the first choice sends both copies to the other encoder.
#[test]
fn equal_keys_are_placed_on_one_encoder() {
    use std::collections::BTreeSet;

    let document = PolicySpec::epd_presets("consistent_hashing", "power_of_two", "power_of_two")
        .expect("the presets");
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let host = Host::under(spec, 1, 1, 2);
    let said = (&[of(1), of(1)][..], &[1][..]);
    let lists = host.lists(said);
    let asks = host.asks(said);
    let (engine, now) = (&host.engine, Instant::now());
    let spec = engine.spec();
    let ids = [0, 1].map(|at| worker_id(host.worker(2, at).as_ref()));
    let mut placed = EncodePlaced::new();
    let mut seen = BTreeSet::new();

    for n in 0..32_u64 {
        let hash = n << 59;
        let keys = [Identity::hashed(hash); 2];
        let items = EncodeItems {
            keys: &keys,
            ..EncodeItems::default()
        };
        let mut request = asks.make(spec);
        // Hash zero leaves the choice to the seeded picker.
        request.facts.seed = 7;
        let decision = lists.decide(engine, None, now, &mut request, items, &mut placed);
        let [first, second] = placed.encoders() else {
            panic!("expected two placements for hash {hash}, got {decision:?}");
        };
        assert_eq!(first.id, second.id, "duplicate hash {hash}");
        seen.insert(first.id);

        let excluded = [first.id];
        let other = if first.id == ids[0] { ids[1] } else { ids[0] };
        request.constraints.excluded = &excluded;
        let retry = lists.decide(engine, None, now, &mut request, items, &mut placed);
        let [first, second] = placed.encoders() else {
            panic!("expected two retry placements for hash {hash}, got {retry:?}");
        };
        assert_eq!([first.id, second.id], [other; 2], "retry for hash {hash}");
    }
    assert_eq!(
        seen,
        BTreeSet::from(ids),
        "initial picks reach both encoders"
    );
}

/// The build records the encode picker's need for text.
/// Two requests place their items without asking the picker for its needs again.
#[test]
fn an_encode_profiles_stage_is_asked_what_it_needs_at_the_build_alone() {
    use std::sync::atomic::AtomicBool;

    use crate::document::spec::Needs;

    #[derive(Debug)]
    struct BuildNeeds(Arc<AtomicBool>);

    impl Lifecycle for BuildNeeds {}

    impl Picker for BuildNeeds {
        fn name(&self) -> &'static str {
            "build-needs"
        }

        fn needs(&self) -> Needs {
            assert!(
                !self.0.load(Ordering::Relaxed),
                "needs called after the build"
            );
            Needs {
                text: true,
                ..Needs::default()
            }
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            rows: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Best.pick(facts, rows, kept, scores, band, out);
        }
    }

    let armed = Arc::new(AtomicBool::new(false));
    let stage = Arc::clone(&armed);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("build-needs", move || {
            Box::new(BuildNeeds(Arc::clone(&stage)))
        })
        .expect("a free picker name");
    let mut document = stock();
    let encode = document
        .profiles
        .get_mut("encode")
        .expect("the encode profile");
    encode.pick = "build-needs".into();
    let digest = document.digest().expect("a digest");
    let spec =
        AppliedSpec::validate_digested(&document, digest, &catalog, &[]).expect("a valid document");
    assert!(
        spec.needs().text,
        "the build records the encode picker's needs"
    );

    let host = Host::under(spec, 1, 1, 1);
    armed.store(true, Ordering::Relaxed);
    let said = (&[of(1)][..], &[1][..]);
    let asks = host.asks(said);
    let keys = [Identity::hashed(11), Identity::hashed(12)];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    for _ in 0..2 {
        host.engine
            .decide_encode_at(&asks, items, &mut placed, Instant::now)
            .expect("an encode decision");
        assert_eq!(
            placed.encoders().len(),
            keys.len(),
            "one placement per item"
        );
    }
    assert_eq!(host.engine.faults(), 0);
}
