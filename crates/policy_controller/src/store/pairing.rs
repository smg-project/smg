//! The pair classes (design 4.1 and 5.1): which decodes a prefill may hand
//! its KV blocks to, decided where a membership write moves a member or an
//! offer and where a generation moves the pairing mode, and read by every
//! pair decision as a class on its rows and a table of classes.
//!
//! One invariant: **a worker has one class, and which classes pair is a
//! table of classes.** Two prefills that may pair with the same decodes
//! share one class, and so do two decodes that may pair with the same
//! prefills, whatever else tells them apart. A worker stands once in its
//! pool: a decode that reports no transport pairs, leniently, with prefills
//! of two transports, and is still one row of one class, which both
//! prefill classes list ([`PairTable`]). The table names classes, never
//! rows, so it grows with the classes that differ and a pool never does.
//!
//! What a pair is compared on is the rule of [`mismatch`] under one of
//! three modes ([`Pairing`]): a runtime, an operator's protocol, a KV
//! transport, four KV layout facts and an engine version, each a fact its
//! worker reports or leaves unknown. The mode judges the KV handoff alone.
//! What a wire cannot carry at all is that wire's fact and no mode switches
//! it off ([`runtime_specific`]): where a pair's requests are built for one
//! runtime, two known runtimes never pair.
//!
//! The derivation ([`PairClasses::derive`]) is a function of the membership
//! and the mode and of nothing else. It numbers each side's classes in the
//! order that side's workers are given, which the store makes the workers'
//! id order, so two stores that hold one membership number its classes
//! alike.
//!
//! A membership build of one worker's writes derives for that worker
//! alone: the store's writer keeps each model's offers and the rule's
//! verdicts on them from one build to the next, moves the written worker in
//! them and runs the rule again only on an offer no worker of its pool
//! held. The classes it hands back are the whole derivation's.
//!
//! Every value here is opaque text: a runtime, a transport and a layout
//! fact are compared for equality and never interpreted. Deriving a
//! descriptor from what an engine reports is the registration handshake's.

use std::{
    cmp::Ordering,
    collections::hash_map::Entry,
    fmt,
    hash::{BuildHasher, Hash},
};

use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::{
    document::spec::Pairing,
    engine::{
        framework::PairTable,
        view::{WireModes, WorkerId},
        PartnerClasses,
    },
};

#[cfg(test)]
mod tests;

/// The KV layout facts a descriptor may report, in the order they are
/// compared: the short name a pairing key spells, then the name a mismatch
/// reports.
pub const LAYOUT_FACTS: [(&str, &str); 4] = [
    ("dtype", "kv_cache_dtype"),
    ("page", "page_size"),
    ("attn", "attention_backend"),
    ("model", "model_dtype"),
];

/// What one prefill or decode worker offers a rendezvous partner, beside
/// its runtime (the row's engine kind). A field left `None` is one the
/// worker does not report. A blank value is no report either, and the rule
/// never meets one: the store's door hands it a normalized descriptor
/// ([`PairingDescriptor::normalize`]), as the gateway's own derive drops a
/// blank label where it makes a worker's descriptor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PairingDescriptor {
    /// The protocol an operator set. Two workers that both set one are
    /// compared on it whole.
    pub protocol: Option<String>,
    /// The KV transport.
    pub transport: Option<String>,
    /// The engine's version, compared under [`Pairing::Strict`] alone.
    pub version: Option<String>,
    /// The KV layout facts, in the order of [`LAYOUT_FACTS`].
    pub kv_layout: [Option<String>; 4],
}

impl PairingDescriptor {
    /// Make the descriptor what the rule reads, in place: every blank value
    /// dropped and every name read past the blanks around it. The store's
    /// door makes a worker's descriptor so, once a write.
    pub fn normalize(&mut self) {
        normalize(&mut self.protocol);
        normalize(&mut self.transport);
        normalize(&mut self.version);
        self.kv_layout.iter_mut().for_each(normalize);
    }

    /// Whether no value is blank or stands between blanks.
    pub(super) fn is_normalized(&self) -> bool {
        [&self.protocol, &self.transport, &self.version]
            .into_iter()
            .chain(&self.kv_layout)
            .flatten()
            .all(|value| !value.is_empty() && value.trim().len() == value.len())
    }
}

/// One stated value, normalized in place: a descriptor's, and the connector
/// of an encode offer.
pub(super) fn normalize(value: &mut Option<String>) {
    let Some(text) = value else {
        return;
    };
    let read = text.trim();
    if read.is_empty() {
        *value = None;
    } else if read.len() != text.len() {
        *text = read.to_string();
    }
}

/// The descriptor of a worker that has none: it reports nothing.
static NO_DESCRIPTOR: PairingDescriptor = PairingDescriptor {
    protocol: None,
    transport: None,
    version: None,
    kv_layout: [None, None, None, None],
};

/// One prefill or decode worker as the pairing rule reads it.
#[derive(Clone, Copy, Debug)]
pub struct Member<'a> {
    pub id: WorkerId,
    /// The wires the worker serves.
    pub wires: WireModes,
    /// The worker's runtime; empty when it is unknown.
    pub runtime: &'a str,
    pub descriptor: &'a PairingDescriptor,
}

/// What a worker offers a partner, whole: its side of the wire's fact
/// ([`runtime_specific`]), its runtime when it is known and its descriptor.
/// Two workers of one offer pair alike with every worker, so the rule runs
/// once a pair of offers and never once a pair of workers.
///
/// An unknown runtime is `None` here ([`known`]): the offers are numbered
/// with a compare a member.
type Offer<'a> = (bool, Option<&'a str>, &'a PairingDescriptor);

/// A runtime as an offer reads it: `None` when it is unknown, which a row
/// writes as the empty string. Two unknown runtimes are compared so and
/// never as strings: the C library's compare of two empty strings that
/// hold no buffer takes a slow path (about 200 ns by a probe, where two
/// strings of four bytes take 3). An engine kind is a kind's name, which
/// a worker object's builder parses as written too, so it is read as
/// written: a descriptor's free text is what the door normalizes.
pub(crate) fn known(runtime: &str) -> Option<&str> {
    (!runtime.is_empty()).then_some(runtime)
}

impl<'a> Member<'a> {
    /// A member from the facts of its row: a row with no descriptor reports
    /// nothing, and a row's descriptor comes normalized from the door.
    pub fn new(
        id: WorkerId,
        wires: WireModes,
        runtime: &'a str,
        descriptor: Option<&'a PairingDescriptor>,
    ) -> Self {
        let descriptor = descriptor.unwrap_or(&NO_DESCRIPTOR);
        debug_assert!(
            descriptor.is_normalized(),
            "a descriptor reaches the rule normalized"
        );
        Self {
            id,
            wires,
            runtime,
            descriptor,
        }
    }

    /// The runtime, when it is known.
    fn known_runtime(&self) -> Option<&'a str> {
        known(self.runtime)
    }

    fn offer(&self) -> Offer<'a> {
        (
            runtime_specific(self.wires),
            self.known_runtime(),
            self.descriptor,
        )
    }
}

/// The component two workers cannot pair on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mismatch {
    /// Two known runtimes differ.
    Runtime,
    /// Both sides set a protocol and the two differ.
    Explicit,
    /// Two known KV transports differ.
    Transport,
    /// Two reported engine versions differ, under [`Pairing::Strict`] alone.
    Version,
    /// Both sides report the named KV layout fact and disagree.
    KvLayout(&'static str),
    /// A component one side does not report, or a runtime or a transport
    /// that neither side names, refused under [`Pairing::Strict`] alone.
    Unknown(&'static str),
}

/// The component in the words a refusal reports it.
impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime => f.write_str("runtime"),
            Self::Explicit => f.write_str("pairing_protocol"),
            Self::Transport => f.write_str("transport"),
            Self::Version => f.write_str("version"),
            Self::KvLayout(fact) => f.write_str(fact),
            Self::Unknown(component) => write!(f, "unknown {component}"),
        }
    }
}

/// How the two sides of a would-be pair report one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Agreement {
    /// Both report it, the same.
    Agree,
    /// Both report it, differently.
    Differ,
    /// One side reports it, the other does not.
    OneSided,
    /// Neither reports it.
    Unreported,
}

/// What it means when neither side of a pair reports a component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unreported {
    /// The component always exists (a runtime, a transport): two workers
    /// that name neither leave it unknown, which only strict refuses.
    Unknown,
    /// The workers may not carry the fact (a KV layout fact): nothing to
    /// compare.
    NotAFact,
}

impl Agreement {
    fn of(mine: Option<&str>, theirs: Option<&str>) -> Self {
        match (mine, theirs) {
            (Some(a), Some(b)) if a == b => Self::Agree,
            (Some(_), Some(_)) => Self::Differ,
            (None, None) => Self::Unreported,
            _ => Self::OneSided,
        }
    }

    /// The verdict on a hard fact: a known difference is `differ`; a side
    /// that does not report the fact leaves it unknown, which only strict
    /// refuses; a fact nobody reports is what `unreported` says of the
    /// component.
    fn hard_fact(
        self,
        differ: Mismatch,
        name: &'static str,
        strict: bool,
        unreported: Unreported,
    ) -> Option<Mismatch> {
        let unknown = strict.then_some(Mismatch::Unknown(name));
        match self {
            Self::Differ => Some(differ),
            Self::OneSided => unknown,
            Self::Unreported if unreported == Unreported::Unknown => unknown,
            Self::Agree | Self::Unreported => None,
        }
    }
}

/// How the two workers' runtimes compare. An unknown runtime is not a
/// different one.
fn runtimes(prefill: &Member<'_>, decode: &Member<'_>) -> Agreement {
    Agreement::of(prefill.known_runtime(), decode.known_runtime())
}

/// The first component the KV handoff of `prefill` to `decode` fails on
/// under `mode`, if any.
///
/// Runtime, transport and each KV layout fact are hard facts: a known
/// difference refuses the pair in every mode but off, a fact one side
/// reports and the other does not only under strict, and a layout fact
/// neither side reports is not compared. Two protocols set by operators are
/// compared whole and settle the pair when equal; one set on one side alone
/// cannot be checked against reported facts, so strict refuses it and
/// lenient falls through to the facts. The version is a soft fact: workers
/// report it unevenly and a rolling upgrade mixes versions, so it counts
/// under strict alone, and only when both sides report one.
pub fn mismatch(prefill: &Member<'_>, decode: &Member<'_>, mode: Pairing) -> Option<Mismatch> {
    if mode == Pairing::Off {
        return None;
    }
    let strict = mode == Pairing::Strict;
    if let Some(mismatch) = runtimes(prefill, decode).hard_fact(
        Mismatch::Runtime,
        "runtime",
        strict,
        Unreported::Unknown,
    ) {
        return Some(mismatch);
    }
    let (mine, theirs) = (prefill.descriptor, decode.descriptor);
    match Agreement::of(mine.protocol.as_deref(), theirs.protocol.as_deref()) {
        Agreement::Agree => return None,
        Agreement::Differ => return Some(Mismatch::Explicit),
        Agreement::OneSided if strict => return Some(Mismatch::Unknown("pairing_protocol")),
        Agreement::OneSided | Agreement::Unreported => {}
    }
    if let Some(mismatch) = Agreement::of(mine.transport.as_deref(), theirs.transport.as_deref())
        .hard_fact(
            Mismatch::Transport,
            "transport",
            strict,
            Unreported::Unknown,
        )
    {
        return Some(mismatch);
    }
    let facts = mine.kv_layout.iter().zip(&theirs.kv_layout);
    for (&(_, name), (mine, theirs)) in LAYOUT_FACTS.iter().zip(facts) {
        if let Some(mismatch) = Agreement::of(mine.as_deref(), theirs.as_deref()).hard_fact(
            Mismatch::KvLayout(name),
            name,
            strict,
            Unreported::NotAFact,
        ) {
            return Some(mismatch);
        }
    }
    let versions = Agreement::of(mine.version.as_deref(), theirs.version.as_deref());
    (strict && versions == Agreement::Differ).then_some(Mismatch::Version)
}

/// Whether a pair drawn on `wires` must share a runtime: the wire's fact,
/// in its one home.
///
/// The HTTP wire forwards one body to both sides of a pair and has no such
/// fact. Every other wire builds a pair's requests in the message type of
/// one runtime, so a pair of two known runtimes cannot be dispatched on it
/// at all. A registration names one wire and a row's type holds a set, so
/// the function is total: a set that holds HTTP and one wire more is on the
/// runtime-specific side, as the empty set is, and a worker on that side
/// never pairs with an HTTP-only one.
pub const fn runtime_specific(wires: WireModes) -> bool {
    wires.bits() != WireModes::HTTP.bits()
}

/// Why `prefill` and `decode`, two workers on one side of
/// [`runtime_specific`], cannot pair under `mode`: the wire's own fact
/// first, then the mode's rule over the handoff ([`mismatch`]).
///
/// The mode says how much of the KV handoff is compared, and off compares
/// none of it. A known runtime difference on a runtime-specific wire is not
/// the handoff's fact but the wire's, so no mode switches it off. An
/// unknown runtime is not a different one and pairs as the mode says.
pub fn wire_mismatch(prefill: &Member<'_>, decode: &Member<'_>, mode: Pairing) -> Option<Mismatch> {
    if runtime_specific(prefill.wires) && runtimes(prefill, decode) == Agreement::Differ {
        return Some(Mismatch::Runtime);
    }
    mismatch(prefill, decode, mode)
}

/// The group a worker pairs in, as a refusal names it: the operator's
/// protocol, else `runtime/transport/layout`, with `?` for a transport or a
/// layout the worker does not report and `unspecified` for an unknown
/// runtime. The version is left out: it counts under strict alone, and a
/// rolling upgrade would show two keys for workers that pair.
pub fn key(member: &Member<'_>) -> String {
    let descriptor = member.descriptor;
    if let Some(protocol) = &descriptor.protocol {
        return protocol.clone();
    }
    let layout: Vec<String> = LAYOUT_FACTS
        .iter()
        .zip(&descriptor.kv_layout)
        .filter_map(|(&(short, _), value)| value.as_ref().map(|value| format!("{short}={value}")))
        .collect();
    format!(
        "{}/{}/{}",
        member.known_runtime().unwrap_or("unspecified"),
        descriptor.transport.as_deref().unwrap_or("?"),
        if layout.is_empty() {
            "?".to_string()
        } else {
            layout.join(",")
        },
    )
}

/// Why no prefill on one side of [`runtime_specific`] can pair with any
/// decode on it: each pool's distinct pairing keys and the components the
/// two pools disagree on, each list sorted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub prefill: Vec<String>,
    pub decode: Vec<String>,
    pub mismatches: Vec<String>,
}

/// The pair classes of one model's prefill and decode pools, and which of
/// them pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairClasses {
    /// Each prefill with its class, in the order given.
    prefills: Box<[(WorkerId, u32)]>,
    /// Each decode with its class, in the order given.
    decodes: Box<[(WorkerId, u32)]>,
    /// Where prefill class `k`'s list ends in `listed`, at `k - 1`.
    ends: Box<[u32]>,
    /// The lists one after another: the decode classes each prefill class
    /// pairs with, ascending.
    listed: Box<[u32]>,
    /// The refusal on each side of [`runtime_specific`]: the plain side
    /// first.
    refusals: [Option<Refusal>; 2],
}

impl PairClasses {
    /// Derive the pair classes of one model's `prefills` and `decodes`
    /// under `mode`.
    ///
    /// A prefill pairs with every decode on its side of
    /// [`runtime_specific`] that [`wire_mismatch`] does not refuse. The
    /// prefills that pair with the same decodes are one prefill class, and
    /// the decodes that pair with the same prefills are one decode class:
    /// workers of two descriptors that pair alike share a class. Each
    /// side's classes are numbered from `1` in the order their first worker
    /// is given, and a worker that pairs with nobody is class `0` and takes
    /// no number. The table lists, for each prefill class, the decode
    /// classes it pairs with: every decode class is named by some list, so
    /// the table numbers densely, as the engine asks of a host
    /// ([`PairTable::new`]). Where both pools hold workers on one side of
    /// the wire's fact and no pair is made there, that side's [`Refusal`]
    /// names the two pools' keys and what they disagree on.
    pub fn derive(prefills: &[Member<'_>], decodes: &[Member<'_>], mode: Pairing) -> Self {
        Derivation::default().whole(prefills, decodes, mode)
    }

    /// The classes as an engine over worker lists reads them
    /// ([`PartnerClasses`]): each worker's class by its place in the
    /// caller's list, with this derivation's table, which they take.
    ///
    /// `prefills` and `decodes` say, for each place of the caller's two
    /// lists in order, whether the worker there was among the members
    /// derived. The members derived stand in the lists' order, so each
    /// takes its class as derived. A worker that was not among them takes
    /// class `0`, which pairs with nobody.
    #[must_use]
    pub fn by_place(
        self,
        prefills: impl IntoIterator<Item = bool>,
        decodes: impl IntoIterator<Item = bool>,
    ) -> PartnerClasses {
        fn side(derived: &[(WorkerId, u32)], given: impl IntoIterator<Item = bool>) -> Box<[u32]> {
            let mut derived = derived.iter();
            let mut next = || derived.next().map_or(0, |&(_, class)| class);
            let class = |given: bool| if given { next() } else { 0 };
            given.into_iter().map(class).collect()
        }
        PartnerClasses::derived(
            side(&self.prefills, prefills),
            side(&self.decodes, decodes),
            self.ends,
            self.listed,
        )
    }

    /// Each prefill with its class, in the order given: `0` for a prefill
    /// no decode pairs with, else a class numbered from `1`.
    pub fn prefills(&self) -> &[(WorkerId, u32)] {
        &self.prefills
    }

    /// Each decode with its class, in the order given: `0` for a decode no
    /// prefill pairs with, else a class numbered from `1`.
    pub fn decodes(&self) -> &[(WorkerId, u32)] {
        &self.decodes
    }

    /// How many prefill classes there are: the distinct partner lists.
    pub fn classes(&self) -> u32 {
        self.ends.len() as u32
    }

    /// How many entries the lists hold together: with the classes'
    /// count, the table's size.
    pub fn entries(&self) -> usize {
        self.listed.len()
    }

    /// Which classes pair, as a pair decision reads it beside its rows.
    pub fn table(&self) -> PairTable<'_> {
        PairTable::new(&self.ends, &self.listed)
    }

    /// The partner list of prefill class `class`: the decodes of the
    /// classes it pairs with, as indexes into the decodes given, in that
    /// order; empty for class `0` and for a class past the last. Built on
    /// each call: what is held is the table, a list never.
    pub fn list(&self, class: u32) -> Vec<u32> {
        let listed = self.table().of(&class);
        let partner = |&(_, &(_, decode)): &(u32, &(WorkerId, u32))| {
            decode != 0 && listed.binary_search(&decode).is_ok()
        };
        let partners = (0u32..).zip(self.decodes.iter()).filter(partner);
        partners.map(|(at, _)| at).collect()
    }

    /// How many pairs the classes hold: for each prefill, the decodes of
    /// its list. A table that held one list a prefill would be four bytes
    /// a pair; what is held is a class a worker and the table of classes.
    pub fn pairs(&self) -> usize {
        self.prefills
            .iter()
            .map(|&(_, class)| self.list(class).len())
            .sum()
    }

    /// Why no pair was made among the workers on `wires`' side of
    /// [`runtime_specific`]; `None` when one was, and when a pool holds no
    /// worker on that side.
    pub fn refusal(&self, wires: WireModes) -> Option<&Refusal> {
        self.refusals[usize::from(runtime_specific(wires))].as_ref()
    }
}

/// A pool as a derivation reads it: its workers in id order, each by its
/// place.
pub(super) trait Pool<'a> {
    /// How many workers the pool holds.
    fn count(&self) -> usize;
    /// The id of the worker at `at`.
    fn id(&self, at: usize) -> WorkerId;
    /// The worker at `at` as the rule reads it.
    fn member(&self, at: usize) -> Member<'a>;
}

impl<'a> Pool<'a> for [Member<'a>] {
    fn count(&self) -> usize {
        self.len()
    }

    fn id(&self, at: usize) -> WorkerId {
        self[at].id
    }

    fn member(&self, at: usize) -> Member<'a> {
        self[at]
    }
}

/// Where the worker `id` stands in `pool`.
fn place<'a, P: Pool<'a> + ?Sized>(pool: &P, id: WorkerId) -> Option<usize> {
    let (mut low, mut high) = (0, pool.count());
    while low < high {
        let middle = low + (high - low) / 2;
        match pool.id(middle).cmp(&id) {
            Ordering::Less => low = middle + 1,
            Ordering::Greater => high = middle,
            Ordering::Equal => return Some(middle),
        }
    }
    None
}

/// The hash a [`Numbering`] is asked with.
fn hash(thing: impl Hash) -> u64 {
    FxBuildHasher.hash_one(thing)
}

/// What a hash steps by where two things share it.
const STEP: u64 = 0x9e37_79b9_7f4a_7c15;

/// Numbers distinct things as each first comes, by a hash the caller takes
/// and a compare the caller makes. It borrows nothing of what it numbers,
/// so a derivation keeps one and clears it: a map keyed by the things
/// themselves is built at every derivation.
#[derive(Default)]
struct Numbering {
    /// A hash, or a hash some steps on where things share one, and the
    /// number that took it.
    taken: FxHashMap<u64, u32>,
}

impl Numbering {
    fn clear(&mut self) {
        self.taken.clear();
    }

    /// The number of the thing of `hash`, and whether the thing is new:
    /// `same` says whether a number's thing is this one, and a new thing
    /// takes `next`.
    fn number(&mut self, mut hash: u64, next: u32, same: impl Fn(u32) -> bool) -> (u32, bool) {
        loop {
            match self.taken.entry(hash) {
                Entry::Vacant(vacant) => {
                    vacant.insert(next);
                    return (next, true);
                }
                Entry::Occupied(taken) if same(*taken.get()) => return (*taken.get(), false),
                // Another thing of this hash: this one stands a step on.
                Entry::Occupied(_) => hash = hash.wrapping_add(STEP),
            }
        }
    }
}

/// Numbers the distinct lists from `1` in the order each first comes; an
/// empty list is `0` and takes no number. Each list's number goes into
/// `of`, and where the first list of each number stands into `firsts`.
fn numbered(
    lists: &[Vec<u32>],
    numbering: &mut Numbering,
    of: &mut Vec<u32>,
    firsts: &mut Vec<usize>,
) {
    numbering.clear();
    of.clear();
    firsts.clear();
    for (at, list) in lists.iter().enumerate() {
        if list.is_empty() {
            of.push(0);
            continue;
        }
        let same = |number: u32| lists[firsts[number as usize]] == *list;
        let (number, new) = numbering.number(hash(list), firsts.len() as u32, same);
        if new {
            firsts.push(at);
        }
        of.push(number + 1);
    }
}

/// The first `count` of the kept `lists`, each cleared: a list keeps its
/// room from one derivation to the next.
fn cleared(lists: &mut Vec<Vec<u32>>, count: usize) -> &mut [Vec<u32>] {
    if lists.len() < count {
        lists.resize_with(count, Vec::new);
    }
    let lists = &mut lists[..count];
    lists.iter_mut().for_each(Vec::clear);
    lists
}

/// The class of a slot no derivation has classed yet.
const UNCLASSED: u32 = u32::MAX;

/// One distinct offer of a pool as a derivation keeps it.
#[derive(Clone, Copy)]
struct Slot {
    /// Where the offer's first worker stands in its pool. That worker's
    /// facts spell the offer, and the offers are numbered as their first
    /// workers come.
    first: usize,
    /// The offer's side of the wire's fact ([`runtime_specific`]).
    specific: bool,
    /// The class of the offer's workers as the last derivation left it.
    class: u32,
}

impl Slot {
    /// Write the offer's class. Whether that moved a class it had.
    fn classed(&mut self, class: u32) -> bool {
        let moved = self.class != UNCLASSED && self.class != class;
        self.class = class;
        moved
    }
}

/// One pool's offers as a derivation keeps them.
#[derive(Default)]
struct Offered {
    /// Each worker's offer, by slot, the workers in id order.
    of: Vec<u32>,
    /// A slot a distinct offer, and the slots that wait for one.
    slots: Vec<Slot>,
    /// The slots that hold an offer. A derivation that classes the offers
    /// puts them in the order their first workers come.
    order: Vec<u32>,
    /// The slots no worker holds. The next new offer takes one, so the
    /// slots never outnumber the most offers the pool held at once.
    free: Vec<u32>,
}

impl Offered {
    /// The worker of `pool` whose facts spell the offer of `slot`.
    fn first<'a, P: Pool<'a> + ?Sized>(&self, pool: &P, slot: u32) -> Member<'a> {
        pool.member(self.slots[slot as usize].first)
    }

    /// Number the offers of `pool` from nothing: a slot a distinct offer,
    /// as its first worker comes.
    fn number<'a, P: Pool<'a> + ?Sized>(&mut self, pool: &P, numbering: &mut Numbering) {
        self.of.clear();
        self.slots.clear();
        self.order.clear();
        self.free.clear();
        numbering.clear();
        // Neighbours in a pool mostly share an offer: a worker whose offer
        // is the one before it takes that slot by one compare, and the
        // numbering is asked where the two differ.
        let mut before: Option<(Offer<'a>, u32)> = None;
        for at in 0..pool.count() {
            let offer = pool.member(at).offer();
            let slot = match before {
                Some((same, slot)) if same == offer => slot,
                _ => {
                    let same = |slot| self.first(pool, slot).offer() == offer;
                    let next = self.slots.len() as u32;
                    let (slot, new) = numbering.number(hash(offer), next, same);
                    if new {
                        self.slots.push(Slot {
                            first: at,
                            specific: offer.0,
                            class: UNCLASSED,
                        });
                        self.order.push(slot);
                    }
                    before = Some((offer, slot));
                    slot
                }
            };
            self.of.push(slot);
        }
    }

    /// The worker at `at` leaves the pool: its place closes. Whether an
    /// offer went or an offer's first worker changed.
    fn leave(&mut self, at: usize) -> bool {
        let slot = self.of.remove(at);
        for &held in &self.order {
            let first = &mut self.slots[held as usize].first;
            if *first > at {
                *first -= 1;
            }
        }
        if self.slots[slot as usize].first != at {
            return false;
        }
        // It was its offer's first worker: the next of them is, and with
        // none left the slot is free.
        match self.of[at..].iter().position(|&of| of == slot) {
            Some(next) => self.slots[slot as usize].first = at + next,
            None => {
                self.order.retain(|&held| held != slot);
                self.free.push(slot);
            }
        }
        true
    }

    /// The worker at `at` of `pool` joins it: a place opens under the slot
    /// of its offer, a new slot where no worker of the pool holds that
    /// offer. The slot, whether it is new, and whether an offer came or an
    /// offer's first worker changed.
    fn join<'a, P: Pool<'a> + ?Sized>(&mut self, pool: &P, at: usize) -> (u32, bool, bool) {
        for &held in &self.order {
            let first = &mut self.slots[held as usize].first;
            if *first >= at {
                *first += 1;
            }
        }
        let offer = pool.member(at).offer();
        let holds = |&slot: &u32| self.first(pool, slot).offer() == offer;
        let held = self.order.iter().copied().find(holds);
        let (slot, new, moved) = if let Some(slot) = held {
            let first = &mut self.slots[slot as usize].first;
            let earlier = at < *first;
            if earlier {
                *first = at;
            }
            (slot, false, earlier)
        } else {
            let fresh = Slot {
                first: at,
                specific: offer.0,
                class: UNCLASSED,
            };
            let slot = if let Some(slot) = self.free.pop() {
                self.slots[slot as usize] = fresh;
                slot
            } else {
                self.slots.push(fresh);
                self.slots.len() as u32 - 1
            };
            self.order.push(slot);
            (slot, true, true)
        };
        self.of.insert(at, slot);
        (slot, new, moved)
    }
}

/// The working lists of one classing of the offers, kept from one to the
/// next: after the first of a size a derivation allocates the classes it
/// hands back and, where a side stands refused, that side's refusal,
/// written out to be compared with the last one. A whole derivation
/// allocates a list of the decode offers beside them.
#[derive(Default)]
struct Work {
    numbering: Numbering,
    /// A row a prefill offer: the decode offers it pairs with.
    rows: Vec<Vec<u32>>,
    /// A column a decode offer: the prefill classes it pairs with.
    columns: Vec<Vec<u32>>,
    /// Each row's class, and where the first row of each class stands.
    row_class: Vec<u32>,
    row_firsts: Vec<usize>,
    /// The same of the columns.
    column_class: Vec<u32>,
    column_firsts: Vec<usize>,
    /// One class's list on its way into the table.
    list: Vec<u32>,
    /// The table: where each prefill class's list ends, and the lists.
    ends: Vec<u32>,
    listed: Vec<u32>,
    /// What one side's refusal names, before it is sorted.
    refused: Vec<Mismatch>,
}

/// One model's derivation as the store's writer keeps it from one
/// membership build to the next: each pool's distinct offers, which worker
/// holds which, and the rule's verdict on every pair of offers. A write of
/// one worker moves that worker alone in it ([`Derivation::written`]), and
/// the rule runs again only on an offer no worker of its pool held. What is
/// kept grows with the workers and with the offers that differ, and a
/// model that loses a side takes its derivation with it.
#[derive(Default)]
pub(super) struct Derivation {
    /// The prefill pool's offers, then the decode pool's.
    sides: [Offered; 2],
    /// The rule's verdict on each prefill slot beside each decode slot.
    verdicts: Vec<Vec<Verdict>>,
    work: Work,
}

impl Derivation {
    /// Derive the classes of `prefills` and `decodes` under `mode` whole,
    /// as [`PairClasses::derive`] does: what is kept is made again from
    /// the two pools.
    pub(super) fn whole<'a, P: Pool<'a> + ?Sized>(
        &mut self,
        prefills: &P,
        decodes: &P,
        mode: Pairing,
    ) -> PairClasses {
        let Self {
            sides: [prefill, decode],
            verdicts,
            work,
            ..
        } = self;
        prefill.number(prefills, &mut work.numbering);
        decode.number(decodes, &mut work.numbering);
        let (rows, columns) = (prefill.slots.len(), decode.slots.len());
        if verdicts.len() < rows {
            verdicts.resize_with(rows, Vec::new);
        }
        // Each decode offer's first worker, read from its pool once: every
        // prefill offer is judged beside every one of them.
        let firsts: Vec<Member<'a>> = (0..columns as u32)
            .map(|slot| decode.first(decodes, slot))
            .collect();
        for (row, slot) in verdicts.iter_mut().zip(0..rows as u32) {
            let offer = prefill.first(prefills, slot);
            row.clear();
            let judged = |beside: &Member<'a>| Verdict::of(&offer, beside, mode);
            row.extend(firsts.iter().map(judged));
        }
        let (refusals, _) = self.class(prefills, decodes);
        let table = (&*self.work.ends, &*self.work.listed);
        self.classes(prefills, decodes, table, refusals)
    }

    /// Class the offers as they stand: each slot's class, the table into
    /// the working lists and each side's refusal. Whether a class moved
    /// that a slot had.
    fn class<'a, P: Pool<'a> + ?Sized>(
        &mut self,
        prefills: &P,
        decodes: &P,
    ) -> ([Option<Refusal>; 2], bool) {
        let Self {
            sides: [prefill, decode],
            verdicts,
            work,
            ..
        } = self;
        // The offers in the order their first workers come: the classes
        // are numbered in it.
        for Offered { order, slots, .. } in [&mut *prefill, &mut *decode] {
            order.sort_unstable_by_key(|&slot| slots[slot as usize].first);
        }
        // One row a prefill offer: the decode offers it pairs with, by
        // their place in that order. One prefill class a distinct row.
        let rows = cleared(&mut work.rows, prefill.order.len());
        for (row, &slot) in rows.iter_mut().zip(&prefill.order) {
            let verdicts = &verdicts[slot as usize];
            let pairs = |&(_, &beside): &(u32, &u32)| verdicts[beside as usize] == Verdict::Pairs;
            let offers = (0u32..).zip(&decode.order).filter(pairs);
            row.extend(offers.map(|(at, _)| at));
        }
        let numbering = &mut work.numbering;
        numbered(rows, numbering, &mut work.row_class, &mut work.row_firsts);
        // One column a decode offer: the prefill classes it pairs with,
        // ascending as the classes are walked. One decode class a distinct
        // column.
        let columns = cleared(&mut work.columns, decode.order.len());
        for (class, &first) in (1u32..).zip(&work.row_firsts) {
            for &offer in &rows[first] {
                columns[offer as usize].push(class);
            }
        }
        let firsts = &mut work.column_firsts;
        numbered(columns, numbering, &mut work.column_class, firsts);
        // The table: a prefill class lists the classes of the decode
        // offers its row names, each once.
        let (list, ends, listed) = (&mut work.list, &mut work.ends, &mut work.listed);
        ends.clear();
        listed.clear();
        for &first in &work.row_firsts {
            list.clear();
            list.extend(
                rows[first]
                    .iter()
                    .map(|&offer| work.column_class[offer as usize]),
            );
            list.sort_unstable();
            list.dedup();
            listed.extend_from_slice(list);
            ends.push(listed.len() as u32);
        }
        // The table is valid by construction, every decode class named by
        // some list: the question a host asks where it builds one.
        debug_assert_eq!(PairTable::new(ends, listed).validate(), Ok(()));
        let mut moved = false;
        let classed = [
            (&mut *prefill, &work.row_class),
            (&mut *decode, &work.column_class),
        ];
        for (Offered { order, slots, .. }, classes) in classed {
            for (&slot, &class) in order.iter().zip(classes) {
                moved |= slots[slot as usize].classed(class);
            }
        }
        let (pools, offered) = ([prefills, decodes], [&*prefill, &*decode]);
        let mut on = |side| refusal(side, pools, offered, verdicts, &mut work.refused);
        ([on(false), on(true)], moved)
    }

    /// Derive the classes for one written worker alone: `left` is the
    /// worker as the last write held it and `joined` the worker as it
    /// stands now, each where there is one, and `last` is the classes the
    /// last write left, which stand beside what is kept. `None` when those
    /// classes stand as they are.
    ///
    /// The worker's place closes in the pool it left and opens in the pool
    /// it joined, and no other worker's facts are read but those of an
    /// offer's first worker, which spell the offer. Where its offer is one
    /// its pool holds, it takes that offer's class. Where an offer came or
    /// went, or an offer's first worker changed, the rule runs on a new
    /// offer beside the other pool's and the offers are classed again from
    /// the verdicts kept. The answer is the whole derivation's.
    pub(super) fn written<'a, P: Pool<'a> + ?Sized>(
        &mut self,
        last: &PairClasses,
        prefills: &P,
        decodes: &P,
        mode: Pairing,
        (left, joined): (Option<WorkerId>, Option<WorkerId>),
    ) -> Option<PairClasses> {
        let pools = [prefills, decodes];
        let lasts = [&*last.prefills, &*last.decodes];
        let going = left.and_then(|id| {
            let at = |side: usize| lasts[side].binary_search_by_key(&id, |&(id, _)| id);
            (0..2).find_map(|side| Some((side, at(side).ok()?)))
        });
        let coming =
            joined.and_then(|id| (0..2).find_map(|side| Some((side, place(pools[side], id)?))));
        // What is kept stands beside the last classes, and the pools are
        // the last ones but for the written worker. Where a count says
        // otherwise the caller named the wrong worker, and the whole
        // derivation answers.
        let fits = (0..2).all(|side| {
            let kept = self.sides[side].of.len();
            let went = usize::from(going.is_some_and(|(from, _)| from == side));
            let came = usize::from(coming.is_some_and(|(to, _)| to == side));
            kept == lasts[side].len() && kept + came == pools[side].count() + went
        });
        if !fits {
            return Some(self.whole(prefills, decodes, mode));
        }
        let mut moved = false;
        let mut was = None;
        if let Some((side, at)) = going {
            moved |= self.sides[side].leave(at);
            was = Some((side, lasts[side][at].1));
        }
        let mut is = None;
        if let Some((side, at)) = coming {
            let (slot, new, came) = self.sides[side].join(pools[side], at);
            if new {
                self.judge(side, slot, prefills, decodes, mode);
            }
            moved |= came;
            is = Some((side, slot));
        }
        debug_assert!(
            self.sound(prefills, decodes, mode),
            "what is kept is the two pools'"
        );
        // Whether the worker stands where it stood, in the class it had.
        let stands = |kept: &Self| match (was, is) {
            (None, None) => true,
            (Some((from, class)), Some((to, slot))) => {
                left == joined && from == to && class == kept.sides[to].slots[slot as usize].class
            }
            _ => false,
        };
        if !moved {
            // No offer came or went: the table and the refusals stand, and
            // the worker took the class of an offer its pool held.
            if stands(self) {
                return None;
            }
            let table = (&*last.ends, &*last.listed);
            return Some(self.classes(prefills, decodes, table, last.refusals.clone()));
        }
        let (refusals, reclassed) = self.class(prefills, decodes);
        let table = (&*self.work.ends, &*self.work.listed);
        let same = !reclassed && table == (&*last.ends, &*last.listed) && refusals == last.refusals;
        if same && stands(self) {
            return None;
        }
        Some(self.classes(prefills, decodes, table, refusals))
    }

    /// Ask the rule about the new offer of `slot` on `side` (`0` the
    /// prefills') beside each offer the other pool holds.
    fn judge<'a, P: Pool<'a> + ?Sized>(
        &mut self,
        side: usize,
        slot: u32,
        prefills: &P,
        decodes: &P,
        mode: Pairing,
    ) {
        let Self {
            sides: [prefill, decode],
            verdicts,
            ..
        } = self;
        let at = slot as usize;
        if side == 0 {
            if verdicts.len() <= at {
                verdicts.resize_with(at + 1, Vec::new);
            }
            let (row, offer) = (&mut verdicts[at], prefill.first(prefills, slot));
            // Every held decode offer's verdict is written below, and no
            // other is read.
            row.resize(decode.slots.len(), Verdict::Apart);
            for &beside in &decode.order {
                let verdict = Verdict::of(&offer, &decode.first(decodes, beside), mode);
                row[beside as usize] = verdict;
            }
        } else {
            let offer = decode.first(decodes, slot);
            for &beside in &prefill.order {
                let row = &mut verdicts[beside as usize];
                if row.len() <= at {
                    row.resize(at + 1, Verdict::Apart);
                }
                row[at] = Verdict::of(&prefill.first(prefills, beside), &offer, mode);
            }
        }
    }

    /// The classes as they stand: each worker of the two pools with its
    /// offer's class, beside `table` and `refusals`.
    fn classes<'a, P: Pool<'a> + ?Sized>(
        &self,
        prefills: &P,
        decodes: &P,
        (ends, listed): (&[u32], &[u32]),
        refusals: [Option<Refusal>; 2],
    ) -> PairClasses {
        let classed = |offered: &Offered, pool: &P| {
            let of = |(at, &slot): (usize, &u32)| (pool.id(at), offered.slots[slot as usize].class);
            offered.of.iter().enumerate().map(of).collect()
        };
        PairClasses {
            prefills: classed(&self.sides[0], prefills),
            decodes: classed(&self.sides[1], decodes),
            ends: ends.into(),
            listed: listed.into(),
            refusals,
        }
    }

    /// Whether what is kept is the two pools': every worker under the slot
    /// of its offer, a slot a distinct offer with its first worker and its
    /// side, a slot held or free, and the rule's verdict on every pair of
    /// held slots. Asked in a debug build after a written worker's derive.
    fn sound<'a, P: Pool<'a> + ?Sized>(&self, prefills: &P, decodes: &P, mode: Pairing) -> bool {
        let pools = [prefills, decodes];
        let offers = (0..2).all(|side| {
            let (kept, pool) = (&self.sides[side], pools[side]);
            let offer = |at: usize| pool.member(at).offer();
            let held = |(at, &slot): (usize, &u32)| {
                let first = kept.slots[slot as usize].first;
                first <= at && offer(first) == offer(at) && kept.order.contains(&slot)
            };
            let distinct = |(nth, &slot): (usize, &u32)| {
                let Slot {
                    first, specific, ..
                } = kept.slots[slot as usize];
                let others = &kept.order[..nth];
                kept.of.get(first) == Some(&slot)
                    && specific == offer(first).0
                    && !kept.free.contains(&slot)
                    && others
                        .iter()
                        .all(|&other| kept.first(pool, other).offer() != offer(first))
            };
            kept.of.len() == pool.count()
                && kept.order.len() + kept.free.len() == kept.slots.len()
                && kept.of.iter().enumerate().all(held)
                && kept.order.iter().enumerate().all(distinct)
        });
        let [prefill, decode] = &self.sides;
        let judged = |&slot: &u32| {
            let offer = prefill.first(prefills, slot);
            let verdict = |&beside: &u32| {
                let asked = Verdict::of(&offer, &decode.first(decodes, beside), mode);
                self.verdicts[slot as usize].get(beside as usize) == Some(&asked)
            };
            decode.order.iter().all(verdict)
        };
        offers && prefill.order.iter().all(judged)
    }
}

/// The rule's verdict on one prefill offer beside one decode offer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Pairs,
    /// The two stand on two sides of the wire's fact: no pair is drawn
    /// across it, and nothing is compared.
    Apart,
    Refused(Mismatch),
}

impl Verdict {
    fn of(prefill: &Member<'_>, decode: &Member<'_>, mode: Pairing) -> Self {
        if runtime_specific(prefill.wires) != runtime_specific(decode.wires) {
            return Self::Apart;
        }
        wire_mismatch(prefill, decode, mode).map_or(Self::Pairs, Self::Refused)
    }
}

/// The held slots of `offered` on one `side` of the wire's fact.
fn on_side(offered: &Offered, side: bool) -> impl Iterator<Item = u32> + '_ {
    let there = move |&slot: &u32| offered.slots[slot as usize].specific == side;
    offered.order.iter().copied().filter(there)
}

/// Why the offers on one `side` of the wire's fact make no pair, each pool
/// given with the offers kept of it and the rule's `verdicts` on them: each
/// pool's distinct pairing keys there and the components the two pools
/// disagree on, each list sorted. `None` when a pair is made there, and
/// when a pool holds nobody on that side.
fn refusal<'a, P: Pool<'a> + ?Sized>(
    side: bool,
    pools: [&P; 2],
    offered: [&Offered; 2],
    verdicts: &[Vec<Verdict>],
    refused: &mut Vec<Mismatch>,
) -> Option<Refusal> {
    let [prefill, decode] = offered;
    if on_side(prefill, side).next().is_none() || on_side(decode, side).next().is_none() {
        return None;
    }
    refused.clear();
    for slot in on_side(prefill, side) {
        for &beside in &decode.order {
            match verdicts[slot as usize][beside as usize] {
                Verdict::Pairs => return None,
                Verdict::Refused(mismatch) => refused.push(mismatch),
                Verdict::Apart => {}
            }
        }
    }
    refused.sort_unstable();
    refused.dedup();
    let mut mismatches: Vec<String> = refused.iter().map(Mismatch::to_string).collect();
    mismatches.sort_unstable();
    let keys_of = |offered: &Offered, pool: &P| {
        keys(on_side(offered, side).map(|slot| offered.first(pool, slot)))
    };
    Some(Refusal {
        prefill: keys_of(prefill, pools[0]),
        decode: keys_of(decode, pools[1]),
        mismatches,
    })
}

/// The distinct pairing keys of some workers, sorted.
fn keys<'a>(members: impl Iterator<Item = Member<'a>>) -> Vec<String> {
    let mut keys: Vec<String> = members.map(|member| key(&member)).collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}
