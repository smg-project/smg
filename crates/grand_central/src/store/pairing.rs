//! The pair classes (design 4.1 and 5.1): which decodes a prefill may hand
//! its KV blocks to, derived by whoever holds the worker objects of a
//! model's two lists, and read by every pair decision as a class on its
//! rows and a table of classes.
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
//! On such a wire a pair one of whose workers names no runtime goes behind
//! every other pair. The classes state it beside their table, as the
//! decode classes each prefill class is paired with first
//! ([`FirstPartners`]): its partners in every other pair. A pair decision
//! then makes a pair with a worker of unknown runtime only while no first
//! pair is open. One table holds the pairs of both sides of the wire's
//! fact, so a pair of the plain side is a first pair too: that order
//! stands inside the runtime-specific side, and none between the two. Two
//! workers share a class where they pair alike and pair first alike.
//!
//! The derivation ([`PairClasses::derive`]) is a function of the membership
//! and the mode and of nothing else. It numbers each side's classes in the
//! order that side's workers are given, so two holders that give one
//! membership in one order number its classes alike. The store derives
//! none.
//!
//! Every value here is opaque text: a runtime, a transport and a layout
//! fact are compared for equality and never interpreted. Deriving a
//! descriptor from what an engine reports is the registration handshake's.

use std::{
    collections::hash_map::Entry,
    fmt,
    hash::{BuildHasher, Hash},
};

use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::{
    document::spec::Pairing,
    engine::{
        framework::{FirstPartners, PairTable},
        view::{WireModes, WorkerId},
        PartnerClasses,
    },
};

#[cfg(test)]
#[path = "../../tests/unit/store/pairing/tests.rs"]
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
///
/// The record gains fields: code outside this crate starts from the default
/// and assigns what it knows, and a pattern of it ends with `..`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
pub struct Refusal {
    pub prefill: Vec<String>,
    pub decode: Vec<String>,
    pub mismatches: Vec<String>,
}

/// The pair classes of one model's prefill and decode pools, which of them
/// pair, and which of those first.
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
    /// Where prefill class `k`'s first partners end in `first_listed`, at
    /// `k - 1`. Empty where no pair goes ahead of another.
    first_ends: Box<[u32]>,
    /// The first partners' lists one after another: the decode classes
    /// each prefill class is paired with first, ascending.
    first_listed: Box<[u32]>,
    /// The refusal on each side of [`runtime_specific`]: the plain side
    /// first.
    refusals: [Option<Refusal>; 2],
}

impl PairClasses {
    /// Derive the pair classes of one model's `prefills` and `decodes`
    /// under `mode`.
    ///
    /// A prefill pairs with every decode on its side of
    /// [`runtime_specific`] that [`wire_mismatch`] does not refuse. It
    /// pairs first with each of them but one that leaves the pair's
    /// runtime unknown on the runtime-specific side: there a first pair's
    /// two workers name one runtime. The prefills that pair with the same
    /// decodes, and first with the same of them, are one prefill class, and
    /// the decodes that pair so with the same prefills are one decode class:
    /// workers of two descriptors that pair alike share a class. Each
    /// side's classes are numbered from `1` in the order their first worker
    /// is given, and a worker that pairs with nobody is class `0` and takes
    /// no number. The table lists, for each prefill class, the decode
    /// classes it pairs with: every decode class is named by some list, so
    /// the table numbers densely, as the engine asks of a host
    /// ([`PairTable::new`]). The first partners list, for each prefill
    /// class, the decode classes it pairs with first: whole classes of its
    /// own list ([`FirstPartners::within`]). None is kept where every pair
    /// goes first and where none does: no pair then goes ahead of another.
    /// Where both pools hold workers on one side of the wire's fact and no
    /// pair is made there, that side's [`Refusal`] names the two pools'
    /// keys and what they disagree on.
    pub fn derive(prefills: &[Member<'_>], decodes: &[Member<'_>], mode: Pairing) -> Self {
        Derivation::default().whole(prefills, decodes, mode)
    }

    /// The classes as an engine over worker lists reads them
    /// ([`PartnerClasses`]): each worker's class by its place in the
    /// caller's list, with this derivation's table and its first partners,
    /// which they take.
    ///
    /// `prefills` and `decodes` say, for each place of the caller's two
    /// lists in order, whether the worker there was among the members
    /// derived. The members derived stand in the lists' order, so each
    /// takes its class as derived. A worker that was not among them takes
    /// class `0`, which pairs with nobody. A debug build holds the caller
    /// to its word on both ends: a member was derived for every place it
    /// says so of, and it says so of a place for every member derived. A
    /// release build answers a broken word and faults nothing: a place
    /// given with no member derived takes class `0`, and a member derived
    /// with no place given is passed over.
    #[must_use]
    pub fn by_place(
        self,
        prefills: impl IntoIterator<Item = bool>,
        decodes: impl IntoIterator<Item = bool>,
    ) -> PartnerClasses {
        fn side(derived: &[(WorkerId, u32)], given: impl IntoIterator<Item = bool>) -> Box<[u32]> {
            let mut derived = derived.iter();
            let mut next = || {
                let member = derived.next();
                debug_assert!(member.is_some(), "a member derived for every place given");
                member.map_or(0, |&(_, class)| class)
            };
            let class = |given: bool| if given { next() } else { 0 };
            let classes = given.into_iter().map(class).collect();
            let left = derived.as_slice();
            debug_assert!(left.is_empty(), "a place given for every member derived");
            classes
        }
        PartnerClasses::derived(
            side(&self.prefills, prefills),
            side(&self.decodes, decodes),
            [self.ends, self.listed],
            [self.first_ends, self.first_listed],
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

/// The hash a [`Numbering`] is asked with.
fn hash(thing: impl Hash) -> u64 {
    FxBuildHasher.hash_one(thing)
}

/// What a hash steps by where two things share it.
const STEP: u64 = 0x9e37_79b9_7f4a_7c15;

/// Numbers distinct things as each first comes, by a hash the caller takes
/// and a compare the caller makes. It borrows nothing of what it numbers,
/// so one serves every numbering of a derivation, cleared before each: a
/// map keyed by the things themselves is built at every numbering.
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

/// One distinct offer of a pool as a derivation keeps it.
#[derive(Clone, Copy)]
struct Slot {
    /// Where the offer's first worker stands in its pool. That worker's
    /// facts spell the offer, and the offers are numbered as their first
    /// workers come.
    first: usize,
    /// The offer's side of the wire's fact ([`runtime_specific`]).
    specific: bool,
}

/// One pool's offers as a derivation keeps them.
#[derive(Default)]
struct Offered {
    /// Each worker's offer, by slot, the workers in the order given.
    of: Vec<u32>,
    /// A slot a distinct offer, in the order their first workers come.
    slots: Vec<Slot>,
}

impl Offered {
    /// The worker of `pool` whose facts spell the offer of `slot`.
    fn first<'a>(&self, pool: &[Member<'a>], slot: u32) -> Member<'a> {
        pool[self.slots[slot as usize].first]
    }

    /// Number the offers of `pool` from nothing: a slot a distinct offer,
    /// as its first worker comes.
    fn number<'a>(&mut self, pool: &[Member<'a>], numbering: &mut Numbering) {
        numbering.clear();
        // Neighbours in a pool mostly share an offer: a worker whose offer
        // is the one before it takes that slot by one compare, and the
        // numbering is asked where the two differ.
        let mut before: Option<(Offer<'a>, u32)> = None;
        for (at, member) in pool.iter().enumerate() {
            let offer = member.offer();
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
                        });
                    }
                    before = Some((offer, slot));
                    slot
                }
            };
            self.of.push(slot);
        }
    }
}

/// The working lists of a derivation's one classing of the offers: new at
/// the derivation and dropped with it.
#[derive(Default)]
struct Work {
    numbering: Numbering,
    /// A row a prefill offer: the decode offers it pairs with, each beside
    /// whether the pair goes first ([`named`]).
    rows: Vec<Vec<u32>>,
    /// A column a decode offer: the prefill classes it pairs with, each
    /// beside the same.
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
    /// The first partners, laid out as the table is.
    first_ends: Vec<u32>,
    first_listed: Vec<u32>,
    /// What one side's refusal names, before it is sorted.
    refused: Vec<Mismatch>,
}

/// What a row names: the thing at `at` (a decode offer, or a prefill class
/// in a column), and whether its pair goes first. Two rows are then one
/// class only where they pair first alike.
fn named(at: u32, first: bool) -> u32 {
    at << 1 | u32::from(first)
}

/// One derivation's working state: each pool's distinct offers, which
/// worker holds which, and the rule's verdict on every pair of offers. It
/// grows with the workers and with the offers that differ.
#[derive(Default)]
struct Derivation {
    /// The prefill pool's offers, then the decode pool's.
    sides: [Offered; 2],
    /// The rule's verdict on each prefill slot beside each decode slot.
    verdicts: Vec<Vec<Verdict>>,
    work: Work,
}

impl Derivation {
    /// Derive the classes of `prefills` and `decodes` under `mode`, as
    /// [`PairClasses::derive`] states them. A derivation serves one call
    /// and is spent by it: every list it holds is new here.
    fn whole<'a>(
        mut self,
        prefills: &[Member<'a>],
        decodes: &[Member<'a>],
        mode: Pairing,
    ) -> PairClasses {
        let Self {
            sides: [prefill, decode],
            verdicts,
            work,
        } = &mut self;
        prefill.number(prefills, &mut work.numbering);
        decode.number(decodes, &mut work.numbering);
        let (rows, columns) = (prefill.slots.len(), decode.slots.len());
        verdicts.resize_with(rows, Vec::new);
        // Each decode offer's first worker, read from its pool once: every
        // prefill offer is judged beside every one of them.
        let firsts: Vec<Member<'a>> = (0..columns as u32)
            .map(|slot| decode.first(decodes, slot))
            .collect();
        for (row, slot) in verdicts.iter_mut().zip(0..rows as u32) {
            let offer = prefill.first(prefills, slot);
            let judged = |beside: &Member<'a>| Verdict::of(&offer, beside, mode);
            row.extend(firsts.iter().map(judged));
        }
        let refusals = self.class(prefills, decodes);
        self.classes(prefills, decodes, refusals)
    }

    /// Class the offers as they stand: each slot's class, the table and
    /// the first partners into the working lists, and each side's refusal.
    fn class(&mut self, prefills: &[Member<'_>], decodes: &[Member<'_>]) -> [Option<Refusal>; 2] {
        let Self {
            sides: [prefill, decode],
            verdicts,
            work,
        } = self;
        // One row a prefill offer: the decode offers it pairs with, each
        // beside whether the pair goes first. A side's offers stand in the
        // order their first workers come, and the classes are numbered in
        // it. One prefill class a distinct row.
        let rows = &mut work.rows;
        rows.resize_with(prefill.slots.len(), Vec::new);
        for (row, verdicts) in rows.iter_mut().zip(&*verdicts) {
            let paired = |(at, verdict): (u32, &Verdict)| match verdict {
                Verdict::Pairs => Some(named(at, true)),
                Verdict::Behind => Some(named(at, false)),
                Verdict::Apart | Verdict::Refused(_) => None,
            };
            row.extend((0u32..).zip(verdicts).filter_map(paired));
        }
        let numbering = &mut work.numbering;
        numbered(rows, numbering, &mut work.row_class, &mut work.row_firsts);
        // One column a decode offer: the prefill classes it pairs with,
        // ascending as the classes are walked, each beside whether it
        // pairs with the offer first. One decode class a distinct column:
        // the offers a class takes first stand apart from its others.
        let columns = &mut work.columns;
        columns.resize_with(decode.slots.len(), Vec::new);
        for (class, &first) in (1u32..).zip(&work.row_firsts) {
            for &offer in &rows[first] {
                columns[(offer >> 1) as usize].push(named(class, offer & 1 != 0));
            }
        }
        let firsts = &mut work.column_firsts;
        numbered(columns, numbering, &mut work.column_class, firsts);
        // The table: a prefill class lists the classes of the decode
        // offers its row names, each once. Then the first partners, laid
        // out so: the classes of the offers its row names first.
        let list = &mut work.list;
        let class_of = |offer: &u32| work.column_class[(offer >> 1) as usize];
        let tables = [
            (&mut work.ends, &mut work.listed, false),
            (&mut work.first_ends, &mut work.first_listed, true),
        ];
        for (ends, listed, first_alone) in tables {
            for &first in &work.row_firsts {
                let offers = rows[first].iter();
                let offers = offers.filter(|&&offer| !first_alone || offer & 1 != 0);
                list.clear();
                list.extend(offers.map(class_of));
                list.sort_unstable();
                list.dedup();
                listed.extend_from_slice(list);
                ends.push(listed.len() as u32);
            }
        }
        // Every pair first is no pair first, and neither is none.
        if work.first_listed.is_empty() || work.first_listed == work.listed {
            work.first_ends.clear();
            work.first_listed.clear();
        }
        // The table is valid by construction, every decode class named by
        // some list, and the first partners stand within it: the questions
        // a host asks where it builds the two.
        let table = PairTable::new(&work.ends, &work.listed);
        debug_assert_eq!(table.validate(), Ok(()));
        let partners = FirstPartners::new(&work.first_ends, &work.first_listed);
        debug_assert_eq!(partners.within(&table), Ok(()));
        let (pools, offered) = ([prefills, decodes], [&*prefill, &*decode]);
        let mut on = |side| refusal(side, pools, offered, verdicts, &mut work.refused);
        [on(false), on(true)]
    }

    /// The classes as they stand: each worker of the two pools with its
    /// offer's class, beside the table, the first partners and `refusals`.
    fn classes(
        &self,
        prefills: &[Member<'_>],
        decodes: &[Member<'_>],
        refusals: [Option<Refusal>; 2],
    ) -> PairClasses {
        let work = &self.work;
        let classed = |offered: &Offered, classes: &[u32], pool: &[Member<'_>]| {
            let of = |(at, &slot): (usize, &u32)| (pool[at].id, classes[slot as usize]);
            offered.of.iter().enumerate().map(of).collect()
        };
        PairClasses {
            prefills: classed(&self.sides[0], &work.row_class, prefills),
            decodes: classed(&self.sides[1], &work.column_class, decodes),
            ends: work.ends.as_slice().into(),
            listed: work.listed.as_slice().into(),
            first_ends: work.first_ends.as_slice().into(),
            first_listed: work.first_listed.as_slice().into(),
            refusals,
        }
    }
}

/// The rule's verdict on one prefill offer beside one decode offer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// The two pair, and no pair goes ahead of theirs: a first pair.
    Pairs,
    /// The two pair on the runtime-specific side, and one of them names no
    /// runtime: such a pair goes behind every first pair.
    ///
    /// The verdict marks the pair that goes behind and not the one that
    /// goes ahead, because one table holds both sides of the wire's fact
    /// and a pair decision makes every pair that is not first only while
    /// no first pair is open. A pair of the plain side, which builds no
    /// request for a runtime, is so a first pair, as a pair on one known
    /// runtime is: a gateway that speaks both wires is held to neither,
    /// and a model none of whose workers leaves its runtime unknown keeps
    /// no first partner at all.
    Behind,
    /// The two stand on two sides of the wire's fact: no pair is drawn
    /// across it, and nothing is compared.
    Apart,
    Refused(Mismatch),
}

impl Verdict {
    fn of(prefill: &Member<'_>, decode: &Member<'_>, mode: Pairing) -> Self {
        let specific = runtime_specific(prefill.wires);
        if specific != runtime_specific(decode.wires) {
            return Self::Apart;
        }
        // On the runtime-specific side the wire's fact has refused two known
        // runtimes that differ, so a pair that stands there goes behind
        // where either runtime is unknown: none is compared a second time.
        let unknown = || prefill.known_runtime().is_none() || decode.known_runtime().is_none();
        match wire_mismatch(prefill, decode, mode) {
            Some(mismatch) => Self::Refused(mismatch),
            None if specific && unknown() => Self::Behind,
            None => Self::Pairs,
        }
    }
}

/// The slots of `offered` on one `side` of the wire's fact.
fn on_side(offered: &Offered, side: bool) -> impl Iterator<Item = u32> + '_ {
    let there = move |&(_, slot): &(u32, &Slot)| slot.specific == side;
    (0u32..).zip(&offered.slots).filter(there).map(|(at, _)| at)
}

/// Why the offers on one `side` of the wire's fact make no pair, each pool
/// given with the offers kept of it and the rule's `verdicts` on them: each
/// pool's distinct pairing keys there and the components the two pools
/// disagree on, each list sorted. `None` when a pair is made there, and
/// when a pool holds nobody on that side.
fn refusal(
    side: bool,
    pools: [&[Member<'_>]; 2],
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
        for &verdict in &verdicts[slot as usize] {
            match verdict {
                Verdict::Pairs | Verdict::Behind => return None,
                Verdict::Refused(mismatch) => refused.push(mismatch),
                Verdict::Apart => {}
            }
        }
    }
    refused.sort_unstable();
    refused.dedup();
    let mut mismatches: Vec<String> = refused.iter().map(Mismatch::to_string).collect();
    mismatches.sort_unstable();
    let keys_of = |offered: &Offered, pool: &[Member<'_>]| {
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
