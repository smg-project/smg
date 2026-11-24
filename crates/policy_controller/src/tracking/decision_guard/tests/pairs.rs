//! A handle of a pair: what it keeps of its two sides, and the words it
//! says of each. The engine is the one of a gateway's own process, over
//! its host's two lists. A door stands in front of it: it lends its host's
//! lists to each request and each decode call a handle passes it, and
//! keeps each word a handle says, with the index and the chosen side the
//! handle lent beside it.

use std::{
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    load::state::LoadSnapshot,
    test_support::{no_health_check, ready_worker},
    worker::{WorkerSpec, WorkerStatus, WorkerType},
};

use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, ShadowKind, Target},
        decision_engine::{
            DecideOutcome, Decided, DecisionEngine, DecisionRefused, DecodeRefused, HeldDecision,
            Plain, Request, RequestMaker, TokenKey, WorkerLists,
        },
        framework::{EncodeItems, EncodePlaced, EncoderFacts},
        list_cell::tests::PublishedList,
        prompt_hashes::{HashSeeds, PromptHashes},
        records::worker_id,
        view::{Constraints, Identity, Phase, RequestFacts, Session, WorkerId},
        EncodeLists, PairAdmission, PairLists, PartnerClasses, PolicyEngine,
    },
    store::GatewayId,
    tracking::{
        lease::{Dispatch, Outcome, ReleaseReason, Said, Side, TransportClass},
        DecisionGuard,
    },
};

const MODEL: &str = "model-a";

/// The gateway every case decides for: another than the one of a gateway's
/// own process, so a handle that names that one in the place of its own
/// shows.
const GATEWAY: GatewayId = GatewayId(11);

/// What the door keeps in a word's place where the word or the call names
/// another gateway than [`GATEWAY`].
const STRANGER: &str = "another gateway";

/// What a handle lent with a word or a decode call: the word, the index of
/// the target taken, and whether the decode side it lent stood chosen.
type Lent = Vec<(&'static str, u8, bool)>;

/// The engine with a door in front. The door lends each request and each
/// decode call its host's lists, so a request stated ahead is decided over
/// them as one its host makes is. An encode pair's request is passed on
/// with no list. It keeps each word a handle says, and the index and the
/// chosen side the handle lent with each word and each decode call.
///
/// A call under another gateway than [`GATEWAY`] is answered a fault and
/// asks no engine. A word under one is kept as [`STRANGER`] and told to
/// no engine. So a case that reads the handle, its words or what it lent
/// fails at its own line, and nothing fails inside a handle's drop.
struct Door<'a> {
    engine: &'a PolicyEngine,
    lists: WorkerLists<'a>,
    said: Mutex<Vec<Said>>,
    lent: Mutex<Lent>,
}

impl<'a> Door<'a> {
    fn new(engine: &'a PolicyEngine, lists: WorkerLists<'a>) -> Self {
        Self {
            engine,
            lists,
            said: Mutex::default(),
            lent: Mutex::default(),
        }
    }

    /// The words said since the last read.
    fn words(&self) -> Vec<Said> {
        mem::take(&mut *self.said.lock())
    }

    /// What was lent since the last read.
    fn lent(&self) -> Lent {
        mem::take(&mut *self.lent.lock())
    }

    /// Whether `gateway` is the one the handle was decided for, which a
    /// handle names with every call and every word. Another one is kept in
    /// the word's place, with what the handle lent.
    fn hears(&self, gateway: GatewayId, held: Option<&HeldDecision<'_>>) -> bool {
        if gateway != GATEWAY {
            let lent = held.map_or((0, false), |held| (held.index, held.decode.chosen));
            self.lent.lock().push((STRANGER, lent.0, lent.1));
        }
        gateway == GATEWAY
    }
}

/// A request's maker with its host's lists beside it.
struct Listed<'a, M>(&'a M, WorkerLists<'a>);

impl<M: RequestMaker> RequestMaker for Listed<'_, M> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.0.make(spec)
    }

    fn plain(&self) -> Option<Plain> {
        self.0.plain()
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.1
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.0.loads()
    }

    fn token_key(&self, page: usize) -> Option<TokenKey<'_>> {
        self.0.token_key(page)
    }
}

impl DecisionEngine for Door<'_> {
    fn decide_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<Decided, DecisionRefused> {
        if !self.hears(gateway, None) {
            return Err(DecisionRefused::Faulted);
        }
        let listed = Listed(maker, self.lists);
        DecisionEngine::decide_under(self.engine, gateway, &listed, now)
    }

    fn decide_decode_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        if !self.hears(gateway, Some(held)) {
            return Err(DecisionRefused::Faulted);
        }
        let lent = ("decode call", held.index, held.decode.chosen);
        self.lent.lock().push(lent);
        let listed = Listed(maker, self.lists);
        DecisionEngine::decide_decode_under(self.engine, gateway, &listed, held, now)
    }

    fn say(&self, gateway: GatewayId, held: &HeldDecision<'_>, said: Said, now: Instant) {
        if !self.hears(gateway, Some(held)) {
            return;
        }
        let word = match said {
            Said::Dispatched(dispatch, _) if dispatch.side == Side::Decode => "decode sent",
            Said::Dispatched(..) => "sent",
            Said::Completed(outcome) if outcome.side == Side::Decode => "decode ended",
            Said::Completed(_) => "ended",
            Said::Released(..) => "released",
        };
        let lent = (word, held.index, held.decode.chosen);
        self.lent.lock().push(lent);
        self.said.lock().push(said);
        DecisionEngine::say(self.engine, gateway, held, said, now);
    }

    fn sweep(&self, now: Instant) {
        DecisionEngine::sweep(self.engine, now);
    }

    fn with_spec<R>(&self, read: impl FnOnce(&AppliedSpec) -> R) -> R {
        DecisionEngine::with_spec(self.engine, read)
    }

    fn prompt_hashes(
        &self,
        model: &str,
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
    ) -> PromptHashes {
        DecisionEngine::prompt_hashes(self.engine, model, tokens, text, seeds)
    }

    fn faults(&self) -> u64 {
        DecisionEngine::faults(self.engine)
    }

    fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64 {
        DecisionEngine::shadowed(self.engine, binding, kind)
    }
}

/// The engine of the stock document `document`.
fn engine(document: &PolicySpec) -> PolicyEngine {
    let spec = AppliedSpec::validate(document).expect("a valid document");
    PolicyEngine::new(spec)
}

/// Workers of one side of a pair, `count` of them. Each is built with no
/// health checker, and so stands ready.
fn workers(name: &str, worker_type: WorkerType, count: usize) -> Arc<[Arc<dyn Worker>]> {
    let ready = |at: usize| -> Arc<dyn Worker> {
        let spec = WorkerSpec {
            worker_type,
            ..WorkerSpec::new(format!("grpc://{name}-{at}:9000"))
        };
        let built = <BasicWorkerBuilder>::from_spec(spec);
        Arc::new(built.health_config(no_health_check()).build())
    };
    (0..count).map(ready).collect()
}

/// A host whose admission is open on both sides.
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

/// A host's two lists of ready workers, every prefill a partner of every
/// decode, under the stock pair document with round robin on both sides,
/// or under a document of the case's own.
struct PairHost {
    engine: PolicyEngine,
    prefills: Arc<[Arc<dyn Worker>]>,
    decodes: Arc<[Arc<dyn Worker>]>,
    members: [PublishedList; 2],
    classes: PartnerClasses,
}

impl PairHost {
    fn of(prefills: usize, decodes: usize) -> Self {
        let document = PolicySpec::pair_presets("round_robin", "round_robin");
        Self::under(&document.expect("the pair presets"), prefills, decodes)
    }

    /// [`Self::of`], under `document`.
    fn under(document: &PolicySpec, prefills: usize, decodes: usize) -> Self {
        let engine = engine(document);
        let prefills = workers("prefill", WorkerType::Prefill, prefills);
        let decodes = workers("decode", WorkerType::Decode, decodes);
        let members = [
            PublishedList::new(Arc::clone(&prefills)),
            PublishedList::new(Arc::clone(&decodes)),
        ];
        let all: Vec<u32> = (0..decodes.len() as u32).collect();
        let partners = prefills.iter().map(|_| all.as_slice());
        let classes = PartnerClasses::number(decodes.len(), partners);
        Self {
            engine,
            prefills,
            decodes,
            members,
            classes,
        }
    }

    /// The host's two lists, as it lends them for one request.
    fn lists(&self) -> PairLists<'_> {
        let [prefill, decode] = &self.members;
        PairLists {
            prefill: prefill.lent(),
            decode: decode.lent(),
            classes: &self.classes,
            admission: &Open,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        }
    }

    /// The engine behind a door that lends this host's two lists.
    fn door(&self) -> Door<'_> {
        Door::new(&self.engine, WorkerLists::Pair(self.lists()))
    }

    /// The id of a prefill of the list other than `taken`.
    fn another_prefill(&self, taken: WorkerId) -> WorkerId {
        let ids = self
            .prefills
            .iter()
            .map(|worker| worker_id(worker.as_ref()));
        let mut others = ids.filter(|id| *id != taken);
        others.next().expect("another prefill")
    }

    /// The decode `target` names stops being ready. Answers the ids of the
    /// list's other decodes, which still stand ready.
    fn unready(&self, target: &Target) -> Vec<WorkerId> {
        let place = target.pair.expect("a pair's target").decode_position;
        self.decodes[place as usize].set_status(WorkerStatus::NotReady);
        let ids = self.decodes.iter().map(|worker| worker_id(worker.as_ref()));
        ids.filter(|id| *id != decode_of(target)).collect()
    }
}

/// A host's one list of three ready workers that serve [`MODEL`], under
/// the `least_load` preset. That document counts each send, so it takes a
/// lease's dispatch, and a single request is issued a lease: under a
/// document that takes no word of a lease a single's handle holds none.
struct ListHost {
    engine: PolicyEngine,
    members: PublishedList,
}

impl ListHost {
    fn new() -> Self {
        let document = PolicySpec::preset("least_load");
        let engine = engine(&document.expect("a preset of this build"));
        let serving = |at: usize| ready_worker(&format!("http://worker-{at}:8000"), MODEL);
        let list: Arc<[Arc<dyn Worker>]> = (0..3).map(serving).collect();
        let members = PublishedList::new(list);
        Self { engine, members }
    }

    /// The engine behind a door that lends this host's list.
    fn door(&self) -> Door<'_> {
        Door::new(&self.engine, WorkerLists::Single(self.members.lent()))
    }
}

/// A request for [`MODEL`] under `phase`.
fn stated(phase: Phase) -> Request<'static> {
    Request {
        id: b"request",
        facts: RequestFacts {
            model: MODEL,
            phase,
            seed: 7,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    }
}

/// A pair's request, both sides decided at once.
fn pair_request() -> Request<'static> {
    stated(Phase::Prefill)
}

/// A pair's request whose decode is sent later.
fn later_request() -> Request<'static> {
    let mut asked = pair_request();
    asked.constraints.decode_later = true;
    asked
}

/// One decision of `door` for the pair's request, both sides decided at
/// once, held.
fn paired<'d>(door: &'d Door<'d>, now: Instant) -> DecisionGuard<&'d Door<'d>> {
    DecisionGuard::decide(door, GATEWAY, &pair_request(), now).expect("a pair's handle")
}

/// One decision of `door` for the pair's request with its decode sent
/// later, held.
fn chosen<'d>(door: &'d Door<'d>, now: Instant) -> DecisionGuard<&'d Door<'d>> {
    DecisionGuard::decide(door, GATEWAY, &later_request(), now).expect("a pair's handle")
}

/// An outcome of `class` on `served` under `side`, as a host reads one.
fn outcome(side: Side, served: WorkerId, class: TransportClass) -> Outcome {
    Outcome {
        side,
        served,
        status: 200,
        class,
        ttft_micros: 1_000,
        tokens_in: 12,
        tokens_out: 34,
        duration_micros: 5_000,
    }
}

/// The dispatch of `side` to the target at `index`, said at `at`, `since`
/// microseconds behind the decision.
fn sent(side: Side, index: u8, since: u32, at: Instant) -> Said {
    let dispatch = Dispatch {
        target_index: index,
        side,
        since_decision_micros: since,
    };
    Said::Dispatched(dispatch, at)
}

/// The decode worker a pair's target names.
fn decode_of(target: &Target) -> WorkerId {
    target.pair.expect("a pair's target").decode_id
}

/// Each side of a pair is dispatched by its own word and ended by its own.
/// A decode's end said with no dispatch says the dispatch first, and a
/// prefill's end its host names as a single's is said under the prefill's
/// side. A side's dispatch and its end said twice are told once.
#[test]
fn each_side_of_a_pair_is_dispatched_and_ended_by_its_own_word() {
    for explicit in [true, false] {
        let host = PairHost::of(1, 2);
        let door = host.door();
        let now = Instant::now();
        let mut handle = paired(&door, now);
        let first = handle.take(0).expect("the first target");
        let second = handle.take(1).expect("the alternate target");
        let d1 = decode_of(&second);
        assert_ne!(decode_of(&first), d1);
        let prefill_at = now + Duration::from_micros(100);
        let decode_at = now + Duration::from_micros(200);

        handle.dispatched(prefill_at);
        if explicit {
            handle.dispatched_decode(decode_at);
            handle.dispatched_decode(decode_at);
        }
        let decoded = outcome(Side::Decode, d1, TransportClass::Ok);
        handle.completed(&decoded, decode_at);
        handle.completed(&decoded, decode_at);
        let prefilled = outcome(Side::Prefill, first.id, TransportClass::Ok);
        let said = if explicit {
            prefilled
        } else {
            outcome(Side::Single, first.id, TransportClass::Ok)
        };
        handle.completed(&said, decode_at);
        drop(handle);
        assert_eq!(
            door.words(),
            [
                sent(Side::Prefill, 1, 100, prefill_at),
                sent(Side::Decode, 1, 200, decode_at),
                Said::Completed(decoded),
                Said::Completed(prefilled),
            ],
            "explicit: {explicit}"
        );
    }
}

/// A decode sent later takes a call once its prefill went out: a call
/// ahead of that is refused by name, asks no door and leaves the side
/// chosen. Behind the prefill's dispatch the call's pick replaces the held
/// targets, takes its first and issues the side, under the prefill's
/// lease. The decode the handle held stopped being ready here, so the
/// call's pick names the list's two other decodes, and the handle moves to
/// its first. A side a call picked takes no second call, and its host may
/// still take another target's decode.
#[test]
fn a_handle_asks_for_its_decode_once_the_prefill_went_out() {
    let host = PairHost::of(1, 3);
    let door = host.door();
    let now = Instant::now();
    let asked = later_request();
    let mut handle = chosen(&door, now);
    let lease = handle.lease().expect("the prefill lease").lease.id;
    let first = handle.take(0).expect("the first target");
    let other = handle.take(1).expect("the alternate target");
    assert!(handle.decode_chosen());
    let original = *handle.decision();
    let not_held = DecisionRefused::Decode(DecodeRefused::PrefillNotHeld);
    assert_eq!(handle.decide_decode(&asked, now).err(), Some(not_held));
    assert_eq!(*handle.decision(), original);
    assert_eq!(handle.index, 1);
    assert!(
        handle.decode_chosen(),
        "a refused call leaves the side chosen"
    );
    assert_eq!(door.lent(), [], "a call refused by name asks no door");

    let sent_at = now + Duration::from_micros(100);
    handle.dispatched(sent_at);
    let ready = host.unready(&other);
    let call = sent_at + Duration::from_micros(50);
    let decision = *handle
        .decide_decode(&asked, call)
        .expect("the decode decision");
    let picked = decision.pick().expect("a pick").targets;
    let [new, second] = picked.as_slice() else {
        panic!("the two decodes that stand ready: {picked:?}");
    };
    // The host's list says which two are expected, each once.
    let named = [decode_of(new), decode_of(second)];
    assert!(named.iter().all(|decode| ready.contains(decode)));
    assert_ne!(named[0], named[1], "each of the two once: {ready:?}");
    assert_eq!((handle.index, handle.target), (0, *new));
    assert!(!handle.decode_chosen(), "the call's pick issued the side");
    let leased = handle
        .lease()
        .map(|leased| (leased.lease.id, leased.lease.side));
    assert_eq!(leased, Some((lease, Side::Prefill)));
    let stands = DecisionRefused::Decode(DecodeRefused::DecodeStands);
    assert_eq!(handle.decide_decode(&asked, call).err(), Some(stands));
    assert_eq!(handle.take_decode(1), Some(*second));
    let decode_at = call + Duration::from_micros(60);
    handle.dispatched_decode(decode_at);
    let decoded = outcome(Side::Decode, decode_of(second), TransportClass::Ok);
    let prefilled = outcome(Side::Prefill, first.id, TransportClass::Ok);
    handle.completed(&decoded, decode_at);
    handle.completed(&prefilled, decode_at);
    drop(handle);
    assert_eq!(
        door.words(),
        [
            sent(Side::Prefill, 1, 100, sent_at),
            // The span since the first decision: the side's lease is dated
            // by that decision's issue.
            sent(Side::Decode, 1, 210, decode_at),
            Said::Completed(decoded),
            Said::Completed(prefilled),
        ]
    );
    assert_eq!(
        door.lent(),
        [
            ("sent", 1, true),
            ("decode call", 1, true),
            ("decode sent", 1, false),
            ("decode ended", 1, false),
            ("ended", 1, false),
        ],
        "one call reached the door"
    );
}

/// A decode call the door sheds leaves the handle as it stood: its first
/// pick, the target its host took, and the side chosen. The side's
/// dispatch then goes to the decode of that target.
#[test]
fn a_handle_keeps_its_first_pick_when_the_decode_call_sheds() {
    let host = PairHost::of(1, 2);
    let door = host.door();
    let now = Instant::now();
    let mut handle = chosen(&door, now);
    let picked = handle.take(1).expect("an alternate target");
    let original = *handle.decision();
    handle.dispatched(now);
    // A pair is narrowed by no subset: a request that states one is shed.
    let mut narrowed = later_request();
    narrowed.constraints.subset_hint = Some(0);
    assert_eq!(handle.decide_decode(&narrowed, now).ok(), Some(&original));
    assert_eq!(*handle.decision(), original);
    assert_eq!((handle.index, handle.target), (1, picked));
    assert!(handle.decode_chosen(), "a shed leaves the side chosen");
    assert_eq!(
        door.lent(),
        [("sent", 1, true), ("decode call", 1, true)],
        "the door was asked, and shed the call"
    );
    handle.dispatched_decode(now);
    assert!(!handle.decode_chosen(), "the dispatch issued the side");
    assert_eq!(
        door.words(),
        [
            sent(Side::Prefill, 1, 0, now),
            sent(Side::Decode, 1, 0, now)
        ]
    );
}

/// A host that makes a pair's request: with a later decode, or with both
/// sides at once.
struct Made {
    later: bool,
}

/// The host whose request sends its decode later.
const LATER: Made = Made { later: true };

impl RequestMaker for Made {
    fn model(&self) -> &str {
        MODEL
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        if self.later {
            later_request()
        } else {
            pair_request()
        }
    }
}

/// A handle asks its decode again for a request its host makes, as for
/// one it stated: the call's pick issues the side.
#[test]
fn a_handle_asks_its_decode_again_for_a_request_its_host_makes() {
    let host = PairHost::of(1, 1);
    let door = host.door();
    let now = Instant::now();
    let mut handle =
        DecisionGuard::decide_under(&door, GATEWAY, &LATER, now).expect("a pair's handle");
    handle.dispatched(now);
    assert!(handle.decode_chosen());
    let decided = handle.decide_decode_under(&LATER, now);
    assert!(decided.is_ok_and(|decision| decision.pick().is_some()));
    assert!(!handle.decode_chosen(), "the call's pick issued the side");
    // An issued side takes no second call.
    let stands = DecisionRefused::Decode(DecodeRefused::DecodeStands);
    assert_eq!(handle.decide_decode_under(&LATER, now).err(), Some(stands));
}

/// A handle decided for a request its host makes holds the decode as the
/// made request said. Sent later it stands chosen and is owed no word: the
/// drop says the prefill's release alone. Decided at once it stands
/// issued, takes no decode call, and the drop gives both sides up in one
/// word.
#[test]
fn a_handle_decided_under_a_maker_holds_the_decode_as_the_made_request_said() {
    for later in [true, false] {
        let host = PairHost::of(1, 1);
        let door = host.door();
        let (made, now) = (Made { later }, Instant::now());
        let mut handle =
            DecisionGuard::decide_under(&door, GATEWAY, &made, now).expect("a pair's handle");
        assert_eq!(handle.decode_chosen(), later);
        // A decode call ahead of the prefill's dispatch is refused by name:
        // a decode issued with its prefill never stood chosen.
        let refused = if later {
            DecodeRefused::PrefillNotHeld
        } else {
            DecodeRefused::NotChosen
        };
        let refused = Some(DecisionRefused::Decode(refused));
        assert_eq!(handle.decide_decode_under(&made, now).err(), refused);
        assert_eq!(door.lent(), [], "later: {later}: no door was asked");
        drop(handle);
        let given_up = later.then_some(Side::Prefill);
        let abandoned = Said::Released(given_up, ReleaseReason::Abandoned);
        assert_eq!(door.words(), [abandoned], "later: {later}");
    }
}

/// The decode a call issued is dated by its own dispatch. The side's
/// lease keeps the first decision's issue instant, and the handle counts
/// the dispatch's span from that decision: a stage that dates a credit by
/// the two keeps it past a report sampled while the prefill ran. The
/// handle's lease shows the call's deadline, the lease's one.
#[test]
fn a_decode_asked_later_is_dated_by_its_own_dispatch() {
    let host = PairHost::of(1, 1);
    let door = host.door();
    let now = Instant::now();
    let asked = later_request();
    let mut handle = chosen(&door, now);
    let issued = handle
        .lease()
        .expect("the prefill lease")
        .lease
        .issued_micros;
    handle.dispatched(now + Duration::from_micros(100));
    // The prefill ran for eight seconds.
    let call = now + Duration::from_secs(8);
    let decided = handle.decide_decode(&asked, call);
    assert!(decided.is_ok_and(|decision| decision.pick().is_some()));
    let leased = handle.lease().expect("the prefill lease");
    assert_eq!(leased.due, call + Duration::from_secs(5));
    assert_eq!(leased.lease.issued_micros, issued);
    let sent_at = call + Duration::from_micros(60);
    handle.dispatched_decode(sent_at);
    let decode = door.words().into_iter().find_map(|said| match said {
        Said::Dispatched(dispatch, at) if dispatch.side == Side::Decode => {
            Some((dispatch.since_decision_micros, at))
        }
        _ => None,
    });
    assert_eq!(decode, Some((8_000_060, sent_at)));
}

/// What a handle lends its engine is read before the word moves the
/// handle. An engine with no file of its leases tells its stages from it:
/// the index names the target taken at an end too, where the word carries
/// none, and a decode that stood chosen is known at its dispatch, which is
/// its issue. A handle lends its decode side with every word: a single's
/// lends no session and a side that never stood chosen.
#[test]
fn a_handle_lends_the_index_taken_and_whether_the_decode_stood_chosen() {
    let host = PairHost::of(1, 2);
    let door = host.door();
    let now = Instant::now();
    let asked = later_request();
    // No decode call: the first pick's decode goes out, a side that stood
    // chosen until its dispatch.
    let mut handle = chosen(&door, now);
    let taken = handle.take(1).expect("the alternate target");
    handle.dispatched(now);
    handle.dispatched_decode(now);
    let decoded = outcome(Side::Decode, decode_of(&taken), TransportClass::Ok);
    let prefilled = outcome(Side::Prefill, taken.id, TransportClass::Ok);
    handle.completed(&decoded, now);
    handle.completed(&prefilled, now);
    drop(handle);
    assert_eq!(
        door.lent(),
        [
            ("sent", 1, true),
            ("decode sent", 1, true),
            ("decode ended", 1, false),
            ("ended", 1, false),
        ]
    );
    // A decode call's pick issues the side and takes the call's first
    // target: the side's dispatch is lent a side that stands issued. The
    // handle is dropped with both sides sent, and each ends.
    let mut handle = chosen(&door, now);
    handle.take(1).expect("the alternate target");
    handle.dispatched(now);
    let decided = handle.decide_decode(&asked, now);
    assert!(decided.is_ok_and(|decision| decision.pick().is_some()));
    handle.dispatched_decode(now);
    drop(handle);
    assert_eq!(
        door.lent(),
        [
            ("sent", 1, true),
            ("decode call", 1, true),
            ("decode sent", 0, false),
            ("ended", 0, false),
            ("decode ended", 0, false),
        ]
    );
    // A single's handle lends no session and a side that never stood
    // chosen: its words carry nothing of a pair's.
    let single = ListHost::new();
    let door = single.door();
    let asked = stated(Phase::default());
    let mut handle = DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a handle");
    let lent = handle.held().expect("a leased decision").decode;
    assert_eq!((lent.session, lent.chosen), (Session::NONE, false));
    handle.dispatched(now);
    drop(handle);
    assert_eq!(door.lent(), [("sent", 0, false), ("ended", 0, false)]);
}

/// A decode side given up while it stood chosen takes no call. The handle
/// owed no report of it, so a file of leases would still hold it chosen:
/// the handle refuses the call by name and asks no door. Nothing is issued
/// or said.
#[test]
fn a_decode_given_up_while_chosen_takes_no_call() {
    let host = PairHost::of(1, 1);
    let door = host.door();
    let now = Instant::now();
    let asked = later_request();
    let mut handle = chosen(&door, now);
    handle.dispatched(now);
    handle.released_decode(ReleaseReason::ClientGone, now);
    let stands = DecisionRefused::Decode(DecodeRefused::DecodeStands);
    assert_eq!(handle.decide_decode(&asked, now).err(), Some(stands));
    assert_eq!(door.words(), [sent(Side::Prefill, 0, 0, now)]);
    assert_eq!(door.lent(), [("sent", 0, true)], "no door was asked");
    // A single's handle answers by name too: its lease has no decode side.
    let single = ListHost::new();
    let door = single.door();
    let asked = stated(Phase::default());
    let mut alone = DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a handle");
    let not_chosen = DecisionRefused::Decode(DecodeRefused::NotChosen);
    assert_eq!(alone.decide_decode(&asked, now).err(), Some(not_chosen));
    assert_eq!(door.lent(), [], "no door was asked");
}

/// A decode that left with its prefill is a lease that is gone. The
/// prefill's end took the chosen side with it, so the handle owes neither
/// side a word: it refuses the call by name and asks no door.
#[test]
fn a_decode_that_left_with_its_prefill_is_a_lease_that_is_gone() {
    let host = PairHost::of(1, 1);
    let door = host.door();
    let now = Instant::now();
    let asked = later_request();
    let mut handle = chosen(&door, now);
    let first = handle.take(0).expect("the first target");
    handle.dispatched(now);
    let served = outcome(Side::Single, first.id, TransportClass::Ok);
    handle.completed(&served, now);
    assert!(!handle.owes());
    assert_eq!(door.lent(), [("sent", 0, true), ("ended", 0, true)]);
    let gone = DecisionRefused::Decode(DecodeRefused::NoLease);
    assert_eq!(handle.decide_decode(&asked, now).err(), Some(gone));
    drop(handle);
    assert_eq!(door.lent(), [], "no door was asked, and no word said");
}

/// A handle that never asks sends the decode its first pick named. The
/// side's target is taken behind the prefill's dispatch alone, and its own
/// dispatch is its issue, said once.
#[test]
fn a_handle_that_never_asks_sends_the_first_picks_decode() {
    let host = PairHost::of(1, 1);
    let door = host.door();
    let now = Instant::now();
    let mut handle = chosen(&door, now);
    let first = handle.take(0).expect("the first target");
    assert_eq!(handle.take_decode(0), None);
    handle.dispatched(now);
    assert_eq!(handle.take_decode(0), Some(first));
    let at = now + Duration::from_micros(80);
    handle.dispatched_decode(at);
    handle.dispatched_decode(at);
    assert_eq!(
        door.words(),
        [
            sent(Side::Prefill, 0, 0, now),
            sent(Side::Decode, 0, 80, at)
        ]
    );
}

/// A decode that still stands chosen is owed no word. A handle dropped
/// with it says the prefill's end alone: its release where nothing was
/// sent, and a cancelled completion behind its dispatch. A release the
/// host names for the chosen side says nothing, and the side takes no
/// target after it or after the prefill's end.
#[test]
fn a_handle_dropped_with_its_decode_still_chosen_says_nothing_of_it() {
    for was_sent in [false, true] {
        let host = PairHost::of(1, 1);
        let door = host.door();
        let now = Instant::now();
        let mut handle = chosen(&door, now);
        let first = handle.take(0).expect("the first target");
        if was_sent {
            handle.dispatched(now);
            assert_eq!(door.words().len(), 1);
        }
        drop(handle);
        let [report]: [Said; 1] = door.words().try_into().expect("one prefill report");
        if was_sent {
            let Said::Completed(cancelled) = report else {
                panic!("the prefill completed");
            };
            assert_eq!(
                (cancelled.side, cancelled.served, cancelled.class),
                (Side::Prefill, first.id, TransportClass::Cancelled)
            );
        } else {
            let abandoned = Said::Released(Some(Side::Prefill), ReleaseReason::Abandoned);
            assert_eq!(report, abandoned);
        }
    }

    let host = PairHost::of(1, 1);
    let door = host.door();
    let now = Instant::now();
    let mut handle = chosen(&door, now);
    let prefill = handle.take(0).expect("the first target").id;
    handle.dispatched(now);
    let _ = door.words();
    // A named release would be visible while the prefill is still held.
    handle.released_decode(ReleaseReason::ClientGone, now);
    assert_eq!(door.words(), []);
    assert_eq!(handle.take_decode(0), None);
    drop(handle);
    // The prefill's end alone, as a sent side's: a cancelled completion.
    let ended = door.words();
    let [Said::Completed(cancelled)] = ended.as_slice() else {
        panic!("the prefill's end alone: {ended:?}");
    };
    assert_eq!(
        (cancelled.side, cancelled.served, cancelled.class),
        (Side::Prefill, prefill, TransportClass::Cancelled)
    );

    let mut handle = chosen(&door, now);
    let prefill = handle.take(0).expect("the first target").id;
    handle.dispatched(now);
    let done = outcome(Side::Prefill, prefill, TransportClass::Ok);
    handle.completed(&done, now);
    assert_eq!(handle.take_decode(0), None);
    assert_eq!(
        door.words(),
        [sent(Side::Prefill, 0, 0, now), Said::Completed(done)],
        "the prefill's dispatch and its end"
    );
    drop(handle);
    assert_eq!(door.words(), [], "the drop says nothing more");
}

/// Behind the prefill's dispatch a take moves the decode alone: to a
/// target of the same prefill, and to none past the pick's end. A take
/// says nothing, and the side's dispatch names the target taken and goes
/// to its decode: the handle's end of the side names that worker.
#[test]
fn a_decode_taken_after_the_prefill_went_out_moves_the_decode_alone() {
    let host = PairHost::of(2, 2);
    let door = host.door();
    let now = Instant::now();
    let mut handle = paired(&door, now);
    let first = handle.take(0).expect("the first target");
    let alternate = handle.take(1).expect("the alternate target");
    assert_eq!(alternate.id, first.id, "the pick lists one prefill");
    assert_ne!(decode_of(&alternate), decode_of(&first));
    assert_eq!(handle.take(0), Some(first));
    assert_eq!(
        handle.take_decode(1),
        None,
        "the prefill still owes a dispatch"
    );
    // The pick lists one prefill. Add a target to exercise the handle's check.
    let other = host.another_prefill(first.id);
    let Decision::Pick(pick) = &mut handle.decided.decision else {
        panic!("a pair pick");
    };
    let different = pick.targets.as_slice().len();
    pick.targets.push(Target {
        id: other,
        ..alternate
    });

    handle.dispatched(now);
    assert_eq!(handle.take(1), None, "the prefill has dispatched");
    assert_eq!(handle.take_decode(different), None);
    assert_eq!(handle.take_decode(usize::MAX), None);
    assert_eq!(handle.take_decode(1), Some(alternate));
    assert_eq!(
        door.words(),
        [sent(Side::Prefill, 0, 0, now)],
        "a take says nothing"
    );
    handle.dispatched_decode(now);
    assert_eq!(handle.take_decode(0), None);
    assert_eq!(door.words(), [sent(Side::Decode, 1, 0, now)]);
    // Each side ends where it was sent: the prefill on the first target's
    // worker, the decode on the worker of the target taken.
    drop(handle);
    let ended = door.words().into_iter().map(|said| match said {
        Said::Completed(cancelled) => (cancelled.side, cancelled.served),
        other => panic!("a sent side completes: {other:?}"),
    });
    let ended: Vec<_> = ended.collect();
    let sent_to = [
        (Side::Prefill, first.id),
        (Side::Decode, decode_of(&alternate)),
    ];
    assert_eq!(ended, sent_to);
}

/// A pair decided whole issues both sides, and given up before any send
/// it ends both in one word.
#[test]
fn a_pair_given_up_before_any_send_ends_both_sides_in_one_word() {
    let host = PairHost::of(1, 1);
    let door = host.door();
    let now = Instant::now();
    let mut handle = paired(&door, now);
    assert!(!handle.decode_chosen(), "a pair decided whole issues both");
    let reason = ReleaseReason::CommitFailed;
    handle.released(reason, now);
    assert!(!handle.owes());
    drop(handle);
    assert_eq!(door.words(), [Said::Released(None, reason)]);
}

/// A pair dropped with one side sent completes that side as cancelled and
/// gives the other up. A decode its host releases by name ends alone,
/// once: as a cancelled completion behind its dispatch and as a release
/// ahead of it, and the prefill is given up at the drop.
#[test]
fn a_pair_dropped_with_one_side_sent_completes_it_and_gives_the_other_up() {
    for went in [Side::Prefill, Side::Decode] {
        let host = PairHost::of(1, 2);
        let door = host.door();
        let now = Instant::now();
        let mut handle = paired(&door, now);
        let taken = handle.take(1).expect("the alternate target");
        let decode = decode_of(&taken);
        let (worker, waiting) = if went == Side::Prefill {
            handle.dispatched(now);
            (taken.id, Side::Decode)
        } else {
            handle.dispatched_decode(now);
            assert!(handle.take(0).is_some());
            (decode, Side::Prefill)
        };
        assert_eq!(door.words().len(), 1);
        drop(handle);
        let [first, second]: [Said; 2] = door.words().try_into().expect("two reports");
        let (completed, released) = if went == Side::Prefill {
            (first, second)
        } else {
            (second, first)
        };
        let Said::Completed(cancelled) = completed else {
            panic!("the sent side completed");
        };
        assert_eq!(
            (cancelled.side, cancelled.served, cancelled.class),
            (went, worker, TransportClass::Cancelled)
        );
        let abandoned = ReleaseReason::Abandoned;
        assert_eq!(released, Said::Released(Some(waiting), abandoned));

        let mut handle = paired(&door, now);
        let first = handle.take(0).expect("the first target");
        let reason = ReleaseReason::ClientGone;
        let report = if went == Side::Decode {
            handle.dispatched_decode(now);
            assert_eq!(door.words().len(), 1);
            let cancelled = Outcome {
                side: Side::Decode,
                served: decode_of(&first),
                class: TransportClass::Cancelled,
                status: 0,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            };
            Said::Completed(cancelled)
        } else {
            Said::Released(Some(Side::Decode), reason)
        };
        handle.released_decode(reason, now);
        handle.released_decode(reason, now);
        assert_eq!(door.words(), [report]);
        assert!(handle.owes(), "the prefill is still owed");
        drop(handle);
        let given_up = Said::Released(Some(Side::Prefill), abandoned);
        assert_eq!(door.words(), [given_up]);
    }
}

/// A pair's lease holds no encode side: an encode outcome ends no side.
/// The engine hears the completion whole, under its own side, and the
/// handle still owes both of its own: the prefill ends at its own word,
/// and the decode is given up at the drop.
#[test]
fn an_encode_outcome_on_a_pairs_handle_ends_neither_side() {
    let host = PairHost::of(1, 1);
    let door = host.door();
    let now = Instant::now();
    let mut handle = paired(&door, now);
    let first = handle.take(0).expect("the first target");
    let encoded = outcome(Side::Encode, decode_of(&first), TransportClass::Ok);
    handle.completed(&encoded, now);
    assert_eq!(door.words(), [Said::Completed(encoded)]);
    assert!(handle.owes(), "neither side moved");
    let served = outcome(Side::Single, first.id, TransportClass::Ok);
    handle.completed(&served, now);
    drop(handle);
    let prefilled = outcome(Side::Prefill, first.id, TransportClass::Ok);
    assert_eq!(
        door.words(),
        [
            sent(Side::Prefill, 0, 0, now),
            Said::Completed(prefilled),
            Said::Released(Some(Side::Decode), ReleaseReason::Abandoned),
        ]
    );
}

/// A handle of an encode request's decision holds its pair's two sides,
/// under the prefill's lease, and nothing of an encoder. The decision is
/// the engine's over its host's three lists
/// ([`PolicyEngine::decide_encode_at`]), held as an answer its host asked
/// itself ([`DecisionGuard::holding`]). Its first target is the pair's,
/// both sides are issued, and the item's encoder is placed beside it. Each
/// word the handle says is of the prefill or of the decode: dropped with
/// its decode sent, it gives the prefill up and completes the decode as
/// cancelled.
#[test]
fn a_handle_of_an_encode_pair_holds_the_pairs_two_sides() {
    let preset = "round_robin";
    let document = PolicySpec::epd_presets(preset, preset, preset);
    let host = PairHost::under(&document.expect("the encode presets"), 1, 1);
    let encoders = PublishedList::new(workers("encode", WorkerType::Encode, 1));
    // The one encoder hands to the one prefill.
    let facts = [EncoderFacts {
        class: 1,
        ..EncoderFacts::default()
    }];
    let lists = WorkerLists::Encode(EncodeLists {
        pair: host.lists(),
        encoders: encoders.lent(),
        encoders_places: None,
        encoders_stalled: &[],
        encoders_held: &[],
        facts: &facts,
        prefill_classes: &[1],
    });
    let door = Door::new(&host.engine, lists);
    let now = Instant::now();
    let items = EncodeItems {
        keys: &[Identity::hashed(11)],
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let (request, engine) = (pair_request(), &host.engine);
    let asked = Listed(&request, lists);
    let decided = engine.decide_encode_at(&asked, items, &mut placed, || now);
    let (decided, later) = decided.expect("a decision");
    let mut handle = DecisionGuard::holding(&door, GATEWAY, decided, later, now);
    let side = handle.lease().map(|leased| leased.lease.side);
    assert_eq!(side, Some(Side::Prefill));
    assert!(!handle.decode_chosen(), "both sides are issued");
    let target = handle.take(0).expect("the first pair");
    let decode = decode_of(&target);
    let encoder = worker_id(encoders.workers()[0].as_ref());
    let named: Vec<WorkerId> = placed.encoders().iter().map(|at| at.id).collect();
    assert_eq!(named, [encoder]);
    let sent_at = now + Duration::from_micros(100);
    handle.dispatched_decode(sent_at);
    drop(handle);
    let words = door.words();
    let [first, released, Said::Completed(cancelled)] = words.as_slice() else {
        panic!("the decode's send, the prefill's release, the decode's end: {words:?}");
    };
    assert_eq!(*first, sent(Side::Decode, 0, 100, sent_at));
    let abandoned = ReleaseReason::Abandoned;
    assert_eq!(*released, Said::Released(Some(Side::Prefill), abandoned));
    assert_eq!(
        (cancelled.side, cancelled.served, cancelled.class),
        (Side::Decode, decode, TransportClass::Cancelled)
    );
}

/// A decode already sent names its prefill: a take behind it keeps the
/// prefill the decode was sent for.
#[test]
fn a_prefill_taken_behind_a_sent_decode_stays_on_its_prefill() {
    let host = PairHost::of(2, 2);
    let door = host.door();
    let now = Instant::now();
    let mut handle = paired(&door, now);
    let first = handle.take(0).expect("the first target");
    let alternate = handle.take(1).expect("the alternate target");
    assert_eq!(alternate.id, first.id, "the pick lists one prefill");
    assert_eq!(handle.take(0), Some(first));
    // The pick lists one prefill. Add a target to exercise the handle's check.
    let other = host.another_prefill(first.id);
    let Decision::Pick(pick) = &mut handle.decided.decision else {
        panic!("a pair pick");
    };
    let different = pick.targets.as_slice().len();
    pick.targets.push(Target {
        id: other,
        ..alternate
    });
    handle.dispatched_decode(now);
    assert_eq!(handle.take(different), None, "another prefill");
    assert_eq!(handle.take(1), Some(alternate), "the same prefill");
}
