//! An encode request's cells: three lists of worker objects under an encode
//! document.
//!
//! A cell named `encode_...` is run here ([`runs`]), by the bench's own
//! measuring. Its host hands the engine three lists: `--workers` prefills,
//! as many decodes, every pair open, and an eighth as many encoders, at
//! least one, each of which takes images. A request carries [`ITEMS`]
//! images, each under a key of its own. A caller asks the engine's entry
//! for an encode request ([`PolicyEngine::decide_encode_at`]). The pair's
//! decision is leased: its guard is told both dispatches and each side's
//! end. Nothing is said of an encoder, because nothing is owed of one.
//!
//! With `--entry place` a caller asks the one call a host makes for a
//! request ([`PolicyEngine::place`]) and says the same of the placement it
//! is answered. The result line's `entry=decide_at` names the engine's
//! entry as the switch does: for these cells that is the entry for an
//! encode request. The line also says a request's items and the list's
//! encoders (`items=`, `encoders=`).
//!
//! A cell named `encode_held_...` holds its request to places of each of
//! the three lists: every worker of the list, as a gateway states the
//! workers on a request's wire. The decision is then one among places.
//!
//! Every worker of the three lists has a load report, and a prefill states
//! its rate as its measured input rate too, as in a pair's cell. Under a
//! document that declares the bench's signal each decode's report states a
//! value of it, as in a pair's cell ([`signals::stated`]). The result's
//! `conversions=` counts the three lists together.

use std::{
    array,
    cell::{OnceCell, RefCell},
    ops::Deref,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        EncodeItems, EncodeLists, EncodePlaced, EncoderFacts, FixedFacts, Identity, LentList,
        MediaCounts, PairLists, PartnerClasses, PlaceRequest, Plain, PolicyEngine, Prompt, Request,
        RequestMaker, WorkerId, WorkerLists,
    },
    store::GatewayId,
    tracking::{DecisionGuard, Outcome, Placement, Side, TransportClass},
};
use smg_worker::{identity::basic::Worker, load::state::TestFeed, worker::WorkerType};
use tokio::runtime::Builder;

use super::{
    measure, mix,
    pair::{self, typed, Open},
    place, reports, signals, Cell, Host, Published, Run, Shadowed, Stated, MODEL,
};

/// The images one request carries, each an item of its own.
const ITEMS: usize = 4;

// An item's number stands in the four bits a turn's seed leaves free, in
// the items' keys.
const _: () = assert!(ITEMS <= 16, "an item's number stands in four bits");

/// The encoders of the run's list, for the result line.
static ENCODERS: AtomicUsize = AtomicUsize::new(0);

/// The encode document of the preset `name` on each of the three sides.
pub(super) fn of(name: &str) -> PolicySpec {
    PolicySpec::epd_presets(name, name, name).expect("a preset's name")
}

/// A pair's request that states [`ITEMS`] images.
pub(super) fn request(seed: u64) -> Request<'static> {
    let mut request = pair::request(seed);
    request.facts.media = images();
    request
}

/// The words an encode cell adds at the end of its result line.
pub(super) fn words() -> String {
    let encoders = ENCODERS.load(Ordering::Relaxed);
    format!(" items={ITEMS} encoders={encoders}")
}

/// The media of one request: [`ITEMS`] images.
fn images() -> MediaCounts {
    MediaCounts {
        images: ITEMS as u16,
        ..MediaCounts::default()
    }
}

/// The keys of one request's items by the turn's seed, each item's own:
/// the item's number stands in the four bits the seed leaves free.
fn keys(seed: u64) -> [Identity<'static>; ITEMS] {
    array::from_fn(|item| Identity::hashed(mix(seed.wrapping_mul(16) | item as u64) | 1))
}

/// The end of a served side, as its host says it.
fn end(side: Side, served: WorkerId) -> Outcome {
    let mut end = Outcome::reported(200, TransportClass::Ok).with_served(side, served);
    end.tokens_in = 1_024;
    end
}

/// What an encode request's host keeps beside its prefills (the host's own
/// list): its decodes with every worker's pair class, and its encoders with
/// what it says of each.
struct Sides {
    decodes: Published,
    classes: PartnerClasses,
    encoders: Published,
    /// What the host says of each encoder beside its object: it takes
    /// images.
    sees: Vec<FixedFacts>,
    /// Each encoder's encode facts: the one class, which hands to a prefill
    /// by no push.
    facts: Vec<EncoderFacts>,
    /// The encode class each prefill names: the one.
    prefill_classes: Vec<u32>,
    /// The places the host holds a request to, where the cell states them:
    /// the prefills', the decodes' and the encoders', every worker of each.
    places: Option<[Vec<u32>; 3]>,
}

/// One turn's encode request as its host states it: the request of
/// [`Stated`], over the host's three lists.
struct StatedEncode<'a> {
    stated: Stated<'a>,
    lists: EncodeLists<'a>,
}

impl RequestMaker for StatedEncode<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.stated.make(spec)
    }

    fn plain(&self) -> Option<Plain> {
        self.stated.plain()
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Encode(self.lists)
    }
}

/// The engine as an encode request's guard holds it: a type of its own, as
/// a pair's guard has one. That guard is then an instance of
/// [`DecisionGuard`] apart from a single request's and from a pair's, and
/// each of those keeps the one caller it has with no encode cell in the
/// binary.
struct EncodesEngine<'a>(&'a PolicyEngine);

impl Deref for EncodesEngine<'_> {
    type Target = PolicyEngine;

    fn deref(&self) -> &PolicyEngine {
        self.0
    }
}

thread_local! {
    /// Where a caller's decisions put each item's encoder. A host keeps it,
    /// so a decision allocates nothing for it after the first.
    static PLACED: RefCell<EncodePlaced> = RefCell::new(EncodePlaced::new());
}

impl Sides {
    /// The three lists as the host hands them for one request.
    fn lists<'a>(&'a self, host: &'a Host) -> EncodeLists<'a> {
        let places = |list: usize| Some(self.places.as_ref()?[list].as_slice());
        let pair = PairLists {
            prefill: host.list.lent(),
            decode: self.decodes.lent(),
            classes: &self.classes,
            admission: &Open,
            prefill_places: places(0),
            decode_places: places(1),
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        };
        let encoders = LentList {
            facts: Some(&self.sees),
            ..self.encoders.lent()
        };
        EncodeLists {
            pair,
            encoders,
            encoders_places: places(2),
            encoders_stalled: &[],
            encoders_held: &[],
            facts: &self.facts,
            prefill_classes: &self.prefill_classes,
        }
    }

    /// One encode request of `host` by the engine's entry: the prefill's
    /// place in its list, with both dispatches and each side's end told
    /// where the decision is leased. A turn that leaves an item with no
    /// encoder places nothing.
    fn decide(&self, host: &Host, seed: u64) -> Option<u32> {
        let stated = StatedEncode {
            stated: Stated {
                host,
                seed,
                prompt: None,
            },
            lists: self.lists(host),
        };
        let keys = keys(seed);
        let items = EncodeItems {
            keys: &keys,
            ..EncodeItems::default()
        };
        let dated = OnceCell::new();
        let clock = || *dated.get_or_init(Instant::now);
        let (decided, later) = PLACED.with_borrow_mut(|placed| {
            let answer = host.engine.decide_encode_at(&stated, items, placed, clock);
            answer.ok().filter(|_| placed.encoders().len() == ITEMS)
        })?;
        let target = *decided.decision.pick()?.targets.first()?;
        let decode = target.pair?.decode_id;
        if decided.lease.is_some() {
            let now = *dated.get()?;
            let engine = EncodesEngine(host.engine);
            let mut held = DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now);
            // The decode goes out with its prefill, and each side's end is
            // told under its own lease.
            held.dispatched(now);
            held.dispatched_decode(now);
            held.completed(&end(Side::Prefill, target.id), now);
            held.completed(&end(Side::Decode, decode), now);
        }
        Some(target.position)
    }

    /// One encode request of `host` by the one call: the model, the three
    /// lists, the items and a prompt of the plain request's size, with no
    /// entry in a token tree, and both sides' dispatch and end said of the
    /// placement. The call answers workers and no place of a list, so the
    /// place this says is the first. A turn that leaves an item with no
    /// encoder places nothing.
    fn place(&self, host: &Host, seed: u64) -> Option<u32> {
        let keys = keys(seed);
        let items = EncodeItems {
            keys: &keys,
            ..EncodeItems::default()
        };
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&place::PROMPT),
                ..Prompt::NONE
            },
            media: images(),
            encode: Some(items),
            ..PlaceRequest::of(MODEL, WorkerLists::Encode(self.lists(host)))
        };
        let mut answer = host.engine.place(&request);
        let placement = answer.as_mut().ok()?;
        if placement.encoders.workers().len() != ITEMS {
            return None;
        }
        if placement.owes_events() {
            said(placement);
        }
        Some(0)
    }
}

/// Both sides' dispatch and end, as a host says them of an encode request's
/// placement. Out of line, so a count by function reads it apart. Nothing
/// is said of an encoder.
#[inline(never)]
fn said(placement: &mut Placement<'_>) {
    let (prefill, decode) = (placement.id(), placement.decode_id());
    placement.sent(Side::Prefill);
    placement.sent(Side::Decode);
    placement.ended(Side::Prefill, end(Side::Prefill, prefill));
    if let Some(decode) = decode {
        placement.ended(Side::Decode, end(Side::Decode, decode));
    }
}

/// Runs the cell of `run` where it is an encode request's, one named
/// `encode_...`, and says whether it did. The cell's request reads nothing
/// the cell makes behind its engine, and none is made ahead of that.
pub(super) fn runs(run: &Run) -> bool {
    let Cell {
        behind,
        document,
        name,
        request,
        ..
    } = run.cell;
    if !name.starts_with("encode_") {
        return false;
    }
    // A shadow profile decides beside a single request's decision, and the
    // rule for a list over its thresholds is a pair's cell's to run.
    assert!(run.shadow == Shadowed::No, "an encode cell runs no shadow");
    assert!(!run.overloaded, "an encode cell's lists are open");
    let one_call = place::asked_with_lists(run);
    let side = |side: &str, count: usize, worker_type| -> Arc<[Arc<dyn Worker>]> {
        let url = |at: usize| format!("http://{side}-{at:04}:8000");
        (0..count).map(|at| typed(url(at), worker_type)).collect()
    };
    let prefills = side("prefill", run.workers, WorkerType::Prefill);
    let decodes = side("decode", run.workers, WorkerType::Decode);
    let encoders = side("encoder", (run.workers / 8).max(1), WorkerType::Encode);
    ENCODERS.store(encoders.len(), Ordering::Relaxed);
    let document = document();
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
    behind(engine);
    // Every prefill pairs with every decode and names the one encode class.
    let partners: Vec<u32> = (0..decodes.len() as u32).collect();
    let every = (0..prefills.len()).map(|_| partners.as_slice());
    let mut sees = FixedFacts::default();
    sees.vision = true;
    let class = EncoderFacts {
        class: 1,
        pushes: false,
        references: false,
    };
    let sides = Sides {
        classes: PartnerClasses::number(decodes.len(), every),
        decodes: Published::of(&decodes),
        encoders: Published::of(&encoders),
        sees: vec![sees; encoders.len()],
        facts: vec![class; encoders.len()],
        prefill_classes: vec![class.class; prefills.len()],
        places: name.starts_with("encode_held_").then(|| {
            [prefills.len(), decodes.len(), encoders.len()]
                .map(|workers| (0..workers as u32).collect())
        }),
    };
    let mut receiver = None;
    let feed = TestFeed::attached_to(|loads| receiver = loads);
    let mut loads = reports(&prefills);
    for report in loads.values_mut() {
        let rank = &mut report.loads[0];
        if rank.gen_throughput > 0.0 {
            rank.prefill_throughput = Some(rank.gen_throughput);
        }
    }
    loads.extend(reports(&decodes));
    loads.extend(reports(&encoders));
    let loads = signals::stated(&document, &decodes, loads);
    feed.publish(&loads);
    // What a host does once, as for one list: it hands its load monitor's
    // channel and spawns the engine's task.
    engine.set_loads(receiver.expect("the feed's channel"));
    let upkeep = Builder::new_current_thread().enable_time().build();
    let upkeep = upkeep.expect("a runtime");
    upkeep.spawn(engine.run());
    let host = Host {
        engine,
        list: Published::of(&prefills),
        request,
        plain: request(0).plain(),
    };
    // The reports are converted a list: the three lists' counts together.
    let converted = || {
        let (decodes, encoders) = (sides.decodes.lent(), sides.encoders.lent());
        host.conversions() + engine.conversions(decodes) + engine.conversions(encoders)
    };
    measure(
        run,
        &host,
        upkeep,
        || feed.publish(&loads),
        converted,
        |seed| {
            if one_call {
                sides.place(&host, seed)
            } else {
                sides.decide(&host, seed)
            }
        },
    );
    true
}
