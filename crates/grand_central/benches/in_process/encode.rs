//! An encode request's cells: three lists of worker objects under an encode
//! document.
//!
//! A cell named `encode_...` is run here ([`runs`]), by the bench's own
//! measuring. Its host hands the engine three lists: `--workers` prefills,
//! as many decodes, every pair open, and an eighth as many encoders, at
//! least one, each of which takes images. A request carries [`ITEMS`]
//! images, each under a key of its own. A caller asks the engine's entry
//! for an encode request ([`InProcess::decide_encode_at`]). The pair's
//! decision is leased: its guard is told both dispatches and each side's
//! end. Nothing is said of an encoder, because nothing is owed of one.
//!
//! With `--entry place` a caller asks the one call a host makes for a
//! request ([`InProcess::place`]) and says the same of the placement it
//! is answered. The result line's `entry=decide_at` names the engine's
//! entry as the switch does: for these cells that is the entry for an
//! encode request. The line also says a request's items and the list's
//! encoders (`items=`, `encoders=`). `--entry place_ruled` and `--entry
//! place_held` state a route's rule with the one call (the bench's `ruled`
//! file). Under `place_ruled` the rule takes every worker: the first
//! decision places, and the rule is asked of its picks alone. Under
//! `place_held` it is asked of the pair's two lists as well. It is asked
//! of no encoder under either.
//!
//! A cell named `encode_held_...` holds its request to places of each of
//! the three lists: every worker of the list, as a gateway states the
//! workers on a request's wire. The decision is then one among places.
//!
//! A cell named `encode_ring...` places its items by the plain ring on the
//! encode side ([`ringed`]), and each item's key carries its text, as a
//! host holds the hex of an item's content hash. In `encode_ring` the host
//! lends its ring for the items: the bench's stand-in for a ring a host
//! keeps, over the encoders in their list's order, which hashes nothing.
//! So the cell counts what the engine does with a ring lent for the items,
//! one ask and one walk an item, and no lookup of a host's ring. In
//! `encode_ring_own` no ring is lent, and the encode stage's own ring
//! places each item.
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

use smg_grand_central::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        EncodeItems, EncodeLists, EncodePlaced, EncoderFacts, FixedFacts, Identity, InProcess,
        MediaCounts, PairLists, PartnerClasses, PlaceRequest, Plain, Prompt, PublishedList,
        Request, RequestMaker, WorkerId, WorkerLists,
    },
    policies::consistent_hashing::LentRing,
    store::GatewayId,
    tracking::{DecisionGuard, Outcome, Placement, Side, TransportClass},
};
use smg_worker::{identity::basic::Worker, load::state::TestFeed, worker::WorkerType};
use tokio::runtime::Builder;

use super::{
    measure, mix,
    pair::{self, typed, Open},
    place, published, reports, ruled, signals, Cell, Host, ListOrder, Run, Shadowed, Stated, MODEL,
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

/// The encode document of the cells named `encode_ring...`: the encode side
/// places an item by the plain ring, and each side of the pair takes its
/// workers in turn.
pub(super) fn ringed() -> PolicySpec {
    let side = "round_robin";
    PolicySpec::epd_presets("consistent_hashing", side, side).expect("a preset's name")
}

/// The texts the items of an `encode_ring...` cell carry, in turn.
const TEXTS: usize = 1_024;

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
    let mut media = MediaCounts::default();
    media.images = ITEMS as u16;
    media
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
    decodes: PublishedList,
    classes: PartnerClasses,
    /// The encoders, published with what the host says of each beside its
    /// object: it takes images.
    encoders: PublishedList,
    /// Each encoder's encode facts: the one class, which hands to a prefill
    /// by no push.
    facts: Vec<EncoderFacts>,
    /// The encode class each prefill names: the one.
    prefill_classes: Vec<u32>,
    /// The places the host holds a request to, where the cell states them:
    /// the prefills', the decodes' and the encoders', every worker of each.
    places: Option<[Vec<u32>; 3]>,
    /// The texts an `encode_ring...` cell's items carry, each with its
    /// fingerprint, made once as a host holds the hex of an item's content
    /// hash beside the item; `None` in every other cell.
    texts: Option<Vec<(String, u64)>>,
    /// The ring the host lends for the items in the cell `encode_ring`: the
    /// encoders in their list's order.
    ring: Option<ListOrder>,
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
struct EncodesEngine<'a>(&'a InProcess);

impl Deref for EncodesEngine<'_> {
    type Target = InProcess;

    fn deref(&self) -> &InProcess {
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
        EncodeLists {
            pair,
            encoders: self.encoders.lent(),
            encoders_places: places(2),
            encoders_stalled: &[],
            encoders_held: &[],
            facts: &self.facts,
            prefill_classes: &self.prefill_classes,
        }
    }

    /// One request of an `encode_ring...` cell, by the engine's entry or by
    /// the one call: its items' keys in the clear, each with the text the
    /// turn's seed names, and the cell's ring lent for them. With `rule`
    /// the one call's request states the run's rule of its route.
    fn keyed(&self, host: &Host, seed: u64, one_call: bool, rule: bool) -> Option<u32> {
        let texts = self.texts.as_deref()?;
        let keys: [Identity<'_>; ITEMS] = array::from_fn(|item| {
            let turn = (seed as usize).wrapping_mul(ITEMS).wrapping_add(item);
            let (text, hash) = &texts[turn % texts.len()];
            // The key by its constructor, then its text: a bench stands
            // outside the crate, where the type may gain a field.
            let mut key = Identity::hashed(*hash);
            key.plain = Some(text);
            key
        });
        let ring = self.ring.as_ref().map(|ring| ring as &dyn LentRing);
        let mut items = EncodeItems::default();
        items.keys = &keys;
        items.ring = ring;
        if one_call {
            self.place_items(host, items, ring, rule)
        } else {
            self.decide_items::<true>(host, seed, items)
        }
    }

    /// One encode request of `host` by the engine's entry: the prefill's
    /// place in its list, with both dispatches and each side's end told
    /// where the decision is leased. A turn that leaves an item with no
    /// encoder places nothing.
    fn decide(&self, host: &Host, seed: u64) -> Option<u32> {
        let keys = keys(seed);
        let mut items = EncodeItems::default();
        items.keys = &keys;
        self.decide_items::<false>(host, seed, items)
    }

    /// [`Self::decide`] for `items`. Built into each of its two callers,
    /// so a cell's turn is one body. `KEYED` gives each caller an instance
    /// of its own: the closure that borrows the thread's placed room then
    /// has one caller in each, and stays built into that turn. With one
    /// instance for both it was compiled apart, and the turn of a cell
    /// that stood before read 243 instructions more.
    #[inline(always)]
    fn decide_items<const KEYED: bool>(
        &self,
        host: &Host,
        seed: u64,
        items: EncodeItems<'_>,
    ) -> Option<u32> {
        let stated = StatedEncode {
            stated: Stated {
                host,
                seed,
                prompt: None,
            },
            lists: self.lists(host),
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
    /// placement. With `rule` the request states the run's rule of its
    /// route ([`ruled::admits`]). The call answers workers and no place of
    /// a list, so the place this says is the first. A turn that leaves an
    /// item with no encoder places nothing.
    fn place(&self, host: &Host, seed: u64, rule: bool) -> Option<u32> {
        let keys = keys(seed);
        let mut items = EncodeItems::default();
        items.keys = &keys;
        self.place_items(host, items, None, rule)
    }

    /// [`Self::place`] for `items`, with the `ring` the host states on the
    /// request: the call lends it to the items. With `rule` the request
    /// states the run's rule of its route. Built into each of its two
    /// callers, as [`Self::decide_items`] is.
    #[inline(always)]
    fn place_items(
        &self,
        host: &Host,
        items: EncodeItems<'_>,
        ring: Option<&dyn LentRing>,
        rule: bool,
    ) -> Option<u32> {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&place::PROMPT),
                ..Prompt::NONE
            },
            media: images(),
            encode: Some(items),
            ring,
            admits: ruled::admits(rule),
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

/// Both sides' dispatch and end, as a host says them of the placement of
/// an encode request or of a pair. Out of line, so a count by function
/// reads it apart. Nothing is said of an encoder.
#[inline(never)]
pub(super) fn said(placement: &mut Placement<'_>) {
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
    let one_call = place::asked_with_lists(run, false);
    // The run's rule of its route, where its entry states one.
    let rule = one_call && ruled::stated(run);
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
    let engine: &'static InProcess = Box::leak(Box::new(InProcess::new(spec)));
    behind(engine);
    // Every prefill pairs with every decode and names the one encode class.
    let partners: Vec<u32> = (0..decodes.len() as u32).collect();
    let every = (0..prefills.len()).map(|_| partners.as_slice());
    let mut sees = FixedFacts::default();
    sees.vision = true;
    let mut class = EncoderFacts::default();
    class.class = 1;
    class.pushes = false;
    class.references = false;
    let sides = Sides {
        classes: PartnerClasses::number(decodes.len(), every),
        decodes: published(&decodes),
        encoders: PublishedList::with_facts(Arc::clone(&encoders), |_| sees),
        facts: vec![class; encoders.len()],
        prefill_classes: vec![class.class; prefills.len()],
        places: name.starts_with("encode_held_").then(|| {
            [prefills.len(), decodes.len(), encoders.len()]
                .map(|workers| (0..workers as u32).collect())
        }),
        texts: name.starts_with("encode_ring").then(|| {
            let text = |key: usize| format!("{:016x}", mix(key as u64));
            let keyed = |text: String| (Identity::plain(&text).hash, text);
            let texts = (0..TEXTS).map(text).map(keyed);
            texts.map(|(hash, text)| (text, hash)).collect()
        }),
        ring: (name == "encode_ring").then(|| ListOrder::of(&encoders)),
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
        list: published(&prefills),
        request,
        plain: request(0).plain(),
    };
    // The reports are converted a list: the three lists' counts together.
    let converted = || {
        let (decodes, encoders) = (sides.decodes.lent(), sides.encoders.lent());
        host.conversions() + engine.conversions(decodes) + engine.conversions(encoders)
    };
    // A cell whose items carry their text makes its own turn, so every
    // other cell's turn is the body it was.
    if sides.texts.is_some() {
        let turn = |seed| sides.keyed(&host, seed, one_call, rule);
        measure(run, &host, upkeep, || feed.publish(&loads), converted, turn);
        return true;
    }
    measure(
        run,
        &host,
        upkeep,
        || feed.publish(&loads),
        converted,
        |seed| {
            if one_call {
                sides.place(&host, seed, rule)
            } else {
                sides.decide(&host, seed)
            }
        },
    );
    true
}
