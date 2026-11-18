//! The encode side of a pair decision under the `epd` plan. A request that
//! carries encode items is decided once: its pair and each item's encoder
//! over one view, so no item is placed on an encoder its pair's prefill
//! takes nothing from.
//!
//! Which encoders hand to which prefill is the store's fact: every encoder
//! and every prefill carries its encode class
//! ([`store::encoding`](crate::store::encoding)), and an item is placed
//! over the encoders of the chosen prefill's class. Three sentences rule
//! the rest. The pair is picked among the prefills whose class holds an
//! open encoder that takes the request's media. A refusal's name comes
//! from membership: only the registered encoders prove a gap no retry
//! closes. The pick comes from what is open.
//!
//! The pair itself is the pair entry's ([`Engine::decide_pair`]), entered
//! unchanged over the prefills and decodes this side admits: a prefill of a
//! class with no open encoder is no candidate of the pair, and neither is a
//! decode without the vision the request asks of it. A request without
//! items reads no encoder, and where it asks no images of its decode it
//! reads no membership: it pays the pair entry's cost alone.

use super::{
    carries_a_column, column_fits, debug_assert_indexed, reason_of, run_profile, shed, Buffers,
    ClassBits, Engine, PairPools, Plan, Profile, Scratch, Set, AFFINITY_COLUMN,
};
use crate::{
    decision::{Decision, ShedReason, Target, Targets, Trace},
    view::{AffinityFacts, Candidate, Constraints, Identity, Phase, PoolFacts, RequestFacts},
    Draw,
};

#[cfg(test)]
mod tests;

/// The stage named when no encoder is open for the request though one that
/// takes it is registered, or when none is registered at all: a retry may
/// find one. Where the overload veto left none open, the veto's own stage,
/// `not-overloaded`, is named in its stead ([`saturation`]).
const ENCODERS: &str = "encoders";
/// The stage named when the request carries images and the registered
/// encoders that would take it otherwise take none.
const ENCODE_VISION: &str = "encode-vision";
/// The stage named when no registered encoder hands to a prefill of a pair:
/// a pushing encoder states no role, or no prefill states its end.
const EC_COMPATIBLE: &str = "ec-compatible";
/// The stage named when no registered encoder takes the request's media.
const ENCODE_MEDIA: &str = "encode-media";
/// The stage named when the decode must take images and no decode of a pair
/// an encoder serves does.
const DECODE_VISION: &str = "decode-vision";
/// The stage named when a column of the encode pool is not one entry a row.
const ENCODE_COLUMN: &str = "encode-column";
/// The stage named when a request with items meets an encode profile that
/// shows its rows to a plugin: no scope is opened over the encoders offered
/// to a key.
const ENCODE_SCOPE: &str = "encode-scope";

/// What the encode side reads of one encoder beside its candidate row, as
/// the store holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncoderFacts {
    /// The encoder's encode class; `0` when it states no end. The store
    /// numbers the classes from 1 over the whole pool as their first
    /// encoder stands, and a gateway's view keeps the numbers of its rows:
    /// a class may stand above the count of the rows a decision is handed.
    pub class: u32,
    /// Its registration states the pushing form, with its role or without.
    pub pushes: bool,
    /// It processes the references of an item itself.
    pub references: bool,
}

/// The encode pool of a pair decision: the model's encoders, open or not,
/// and the columns the encode side reads, each by a row's index in its
/// pool.
#[derive(Clone, Copy, Default)]
pub struct EncodePool<'a> {
    pub encoders: &'a [Candidate],
    /// One entry an encoder.
    pub facts: &'a [EncoderFacts],
    /// The encode class of each prefill of [`PairPools::prefills`]; `0` for
    /// a prefill no encoder hands to.
    pub prefill_classes: &'a [u32],
    /// The encoders' affinity facts: empty, or one entry an encoder.
    pub affinity: &'a [AffinityFacts],
}

/// The encode items of a request and what it asks for them.
#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeItems<'a> {
    /// Each item's key, the caller's content hash. Empty for a request
    /// without items.
    pub keys: &'a [Identity<'a>],
    /// The items taken as one: the caller's digest over them, opaque here.
    /// A class that pushes places it in the items' stead, so callers that
    /// share a fleet agree on it. `None` when each item is encoded on its
    /// own.
    pub one: Option<Identity<'a>>,
    /// The items are references an encoder processes itself.
    pub references: bool,
    /// The decode of the pair must take images.
    pub decode_images: bool,
}

/// Where an encode-pair decision puts each placed key's encoder. The host
/// keeps it beside its [`Scratch`], so a decision allocates nothing after
/// the first.
#[derive(Debug, Default)]
pub struct EncodePlaced {
    encoders: Vec<Target>,
    one: bool,
}

impl EncodePlaced {
    pub fn new() -> Self {
        Self::default()
    }

    /// The encoder of each placed key, in the keys' order: one for the one
    /// key, else one an item. A target's position is its row in the encode
    /// pool. Empty unless the decision is a pick.
    pub fn encoders(&self) -> &[Target] {
        &self.encoders
    }

    /// Whether the one key was placed in the items' stead. The answer says
    /// it: the list's length does not, since one item reads the same either
    /// way.
    pub fn one(&self) -> bool {
        self.one
    }

    fn clear(&mut self) {
        self.encoders.clear();
        self.one = false;
    }
}

/// Rows admitted from a pool, each standing at its index here, with where
/// it stood in the pool and its affinity facts.
#[derive(Debug, Default)]
struct Admitted {
    rows: Vec<Candidate>,
    /// Each row's index in the pool it was admitted from.
    from: Vec<u32>,
    affinity: Vec<AffinityFacts>,
    /// The rows here that a mask over the pool names, by their index here:
    /// empty until a mask is numbered anew ([`Admitted::renumber`]). Its
    /// room is taken with the rows ([`Admitted::take_room`]).
    masked: Vec<u32>,
}

impl Admitted {
    fn clear(&mut self) {
        self.rows.clear();
        self.from.clear();
        self.affinity.clear();
        self.masked.clear();
    }

    /// Numbers a mask over the pool's rows by the rows admitted, each by
    /// its index here: a masked row that was left out masks none. The mask
    /// and the rows both ascend by the pool's index.
    fn renumber(&mut self, stalled: &[u32]) {
        let mut stalled = stalled.iter().copied().peekable();
        for (index, &from) in (0u32..).zip(&self.from) {
            while stalled.next_if(|&at| at < from).is_some() {}
            if stalled.next_if_eq(&from).is_some() {
                self.masked.push(index);
            }
        }
    }

    /// Takes the room a mask numbered anew fills at most, one place a row
    /// admitted: a decision under a mask then allocates nothing that a
    /// decision under none did not.
    fn take_room(&mut self) {
        self.masked.reserve(self.rows.len());
    }

    /// Admits the pool's row `at`; its position becomes its index here.
    fn push(&mut self, row: &Candidate, at: usize, affinity: &[AffinityFacts]) {
        self.rows.push(Candidate {
            position: self.rows.len() as u32,
            ..*row
        });
        self.from.push(at as u32);
        if let Some(facts) = affinity.get(at) {
            self.affinity.push(*facts);
        }
    }

    /// The pool's index of the row at `position` here.
    fn origin(&self, position: u32) -> u32 {
        self.from
            .get(position as usize)
            .copied()
            .unwrap_or(position)
    }
}

/// The encode side's scratch, inside the host's [`Scratch`].
#[derive(Debug, Default)]
pub(super) struct EncodeScratch {
    /// The encode profile's runs.
    run: Buffers,
    /// The classes with an open encoder that takes the request's media: one
    /// bit a class, no wider than the rows handed.
    open: ClassBits,
    /// The classes with a registered encoder that takes it.
    member: ClassBits,
    /// The open classes whose number the bits do not hold, ascending.
    far_open: Vec<u32>,
    /// The same of the classes with a registered encoder.
    far_member: Vec<u32>,
    /// The open encoders that take the request's media, by their row,
    /// ascending. The pass reads an encoder's hard filters once, and the
    /// items are offered from this list.
    opened: Vec<u32>,
    /// The decode classes with a registered decode: one bit a class, no
    /// wider than the fleet.
    decodes_of: ClassBits,
    /// The decode classes with a registered decode that takes images.
    vision_of: ClassBits,
    /// Under no table, the classes of the decodes whose number the bits do
    /// not hold, ascending.
    far: Vec<u32>,
    /// The same of the decodes that take images.
    far_vision: Vec<u32>,
    /// The prefills the encode side admits, when it does not admit all.
    prefills: Admitted,
    /// The decodes it admits, when it does not admit all.
    decodes: Admitted,
    /// The chosen class's open encoders that take the request's media.
    offered: Admitted,
}

/// The name of the gap when no encoder is open for the request, read from
/// membership: `any_member` says a registered encoder takes the request and
/// states an end.
fn encode_gap(
    encode: &EncodePool<'_>,
    items: &EncodeItems<'_>,
    images: bool,
    any_member: bool,
) -> &'static str {
    if any_member {
        return ENCODERS;
    }
    let references = |of: &EncoderFacts| !items.references || of.references;
    if images
        && encode
            .facts
            .iter()
            .any(|of| of.class != 0 && references(of))
    {
        return ENCODE_VISION;
    }
    // A pushing encoder that takes the media and states no end lacks its
    // role: one that stated an end is counted above.
    let rows = encode.encoders.iter().zip(encode.facts);
    if rows
        .into_iter()
        .any(|(row, of)| of.pushes && references(of) && (!images || row.vision))
    {
        return EC_COMPATIBLE;
    }
    if !encode.encoders.is_empty() && (items.references || images) {
        return ENCODE_MEDIA;
    }
    ENCODERS
}

/// Whether an encoder takes the request's media: references when its items
/// are references, an image when it carries one. Inlined by force: each
/// pass asks it once an encoder, and its two tests cost less than a call.
#[inline(always)]
fn takes_media(items: &EncodeItems<'_>, images: bool, row: &Candidate, of: &EncoderFacts) -> bool {
    (!items.references || of.references) && (!images || row.vision)
}

/// The stage that left no encoder open, when its reason is saturation:
/// the overload veto's stage. The encoders that take the
/// request are put to the profile's hard filters in their order, as a
/// pair's side puts its rows: the stage that refuses the last of them left
/// none. A stage ahead of the veto that leaves none is no saturation, and
/// neither is one behind it that refuses an encoder the veto kept.
///
/// Cold and out of line: a request an encoder is open for never comes here.
#[cold]
#[inline(never)]
fn saturation(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
    encode: &EncodePool<'_>,
    items: &EncodeItems<'_>,
    images: bool,
) -> Option<&'static str> {
    let hard = || profile.filters.iter().filter(|f| !f.soft());
    // Where each of them is refused, counted over the hard filters: the
    // farthest of those places is the stage that left none.
    let last = encode
        .encoders
        .iter()
        .zip(encode.facts)
        .filter(|(row, of)| of.class != 0 && takes_media(items, images, row, of))
        .filter_map(|(row, _)| hard().position(|f| !f.keep(facts, row, cx)))
        .max()?;
    let stage = hard().nth(last)?.name();
    (reason_of(stage) == ShedReason::Saturated).then_some(stage)
}

/// Whether a set holds `class`: by its bit, or by `far`, the ascending list
/// of the numbers the bits do not hold. The bit's read is inlined into each
/// pass; the list's stays a call.
#[inline(always)]
fn holds(classes: &ClassBits, far: &[u32], class: u32) -> bool {
    classes.has(class) || (!far.is_empty() && far_holds(far, class))
}

/// Whether `far` holds `class`. A numbering that is dense over the rows
/// handed leaves the list empty, so the read stands apart from the pass
/// over the rows.
#[cold]
#[inline(never)]
fn far_holds(far: &[u32], class: u32) -> bool {
    far.binary_search(&class).is_ok()
}

/// A shed of the encode entry. Cold and out of line: the entry declines
/// ahead of its passes, and built in place at each of its early returns a
/// shed left the passes' helpers out of line.
#[cold]
#[inline(never)]
fn turned_away(reason: ShedReason, stage: &'static str, bare: Trace) -> Decision {
    shed(reason, stage, bare)
}

impl Engine {
    /// The `epd` plan in one decision: the pair, then each encode item over
    /// the encoders of the chosen prefill's encode class.
    ///
    /// The pair is the pair entry's over the rows the encode side admits: a
    /// prefill whose class holds an open encoder that takes the request's
    /// media, and every decode, or the decodes that take images when the
    /// request asks that of its decode (and when its items are references
    /// with images and a pair has such a decode). With every row admitted
    /// the pools are handed on as they are. A shed of the pair is returned
    /// as the pair gave it.
    ///
    /// Before the pair, the gaps no retry closes are named by membership:
    /// no registered encoder takes the request (`encode-vision`,
    /// `encode-media`), none hands to a prefill of a pair (`ec-compatible`),
    /// no decode of a served pair takes the images asked of it
    /// (`decode-vision`). With an encoder registered and none open the stage
    /// is `encoders`. Where the overload veto is the stage that left none
    /// open, the encoders are saturated: the shed names `not-overloaded` and
    /// its reason is saturation, as a pair's side sheds its own rows. This
    /// entry does not enter the all-vetoed rule, as the pair entry does not.
    ///
    /// The items are placed by the binding's encode profile over the open
    /// encoders of the chosen prefill's class that take the media, each
    /// item's key as the decision's key. Under a class that pushes, the one
    /// key is placed in the items' stead when the request carries it.
    /// `placed` holds each placed key's encoder; a request without items
    /// reads no encoder and places none.
    ///
    /// What this side cannot read truly is declined. A request with items
    /// under an encode profile that shows its rows to a plugin names
    /// `encode-scope`: no scope is opened over the encoders offered to a
    /// key. A pair column of another length than its pool names
    /// `affinity-column` before a cut, as the pair entry names it. So does
    /// a column on the request, or rows named on it, before a stage of the
    /// encode side is asked. The encode side's stages are handed no pin
    /// ([`RequestFacts::pinned_at`]): a pin is a position in another pool.
    ///
    /// A list this side hands on in a pool's stead (the prefills or the
    /// decodes it admits, the encoders it offers a key) is numbered anew. A
    /// mask ([`Constraints::stalled`]) numbers the rows of the host's pool,
    /// so beside such a list it is numbered with the rows: a masked row that
    /// was left out masks none. Each side's run and each key's is a whole
    /// pass, where a rotation is turned by the rows kept: two such lists of
    /// one size turn rotations apart, and none takes a name of its own
    /// ([`PoolFacts::reach`]).
    #[expect(
        clippy::too_many_arguments,
        reason = "the pair entry's inputs and the encode side's"
    )]
    pub fn decide_encode_pair(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        encode: EncodePool<'_>,
        items: EncodeItems<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        encode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
        placed: &mut EncodePlaced,
    ) -> Decision {
        placed.clear();
        // The encode side's buffers leave the scratch for the decision: the
        // pair entry takes the scratch whole while they hold its pools.
        let mut side = scratch.encode.take().unwrap_or_default();
        let decision = self.encode_pair(
            facts, pool, pools, encode, items, cx, decode_cx, encode_cx, scratch, &mut side, placed,
        );
        scratch.encode = Some(side);
        if decision.pick().is_none() {
            placed.clear();
        }
        decision
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the pair entry's inputs and the encode side's"
    )]
    fn encode_pair(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        encode: EncodePool<'_>,
        items: EncodeItems<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        encode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
        side: &mut EncodeScratch,
        placed: &mut EncodePlaced,
    ) -> Decision {
        let bare = Trace {
            seed: facts.seed,
            generation: pool.generation,
            ..Trace::default()
        };
        let declined = |stage| turned_away(ShedReason::PolicyDeclined, stage, bare);
        let gap = |stage| turned_away(ShedReason::NoEndpoints, stage, bare);
        // The prefill's lookup names the plan: a binding of another plan
        // declines, as a pair decision under a single binding is declined.
        let epd = self
            .binding(facts.model, Phase::Prefill)
            .and_then(|binding| match binding.plan {
                Plan::Epd { encode } => Some(encode),
                Plan::Single | Plan::PdPair => None,
            });
        let Some(index) = epd else {
            return declined("binding");
        };
        let Some(profile) = self.profiles.get(usize::from(index)) else {
            return declined("binding");
        };
        if encode.facts.len() != encode.encoders.len()
            || encode.prefill_classes.len() != pools.prefills.len()
            || !column_fits(encode.affinity, encode.encoders)
        {
            return declined(ENCODE_COLUMN);
        }
        // A cut copies the entries of the rows it keeps, so a column of
        // another length could fit the rows a cut leaves: the pair's columns
        // are refused before one, as the pair entry refuses them.
        if !column_fits(pools.prefill_affinity, pools.prefills)
            || !column_fits(pools.decode_affinity, pools.decodes)
        {
            return declined(AFFINITY_COLUMN);
        }
        // A column on the request, or rows named on it, name neither pool.
        // The pair entry refuses them: here they are refused before a stage
        // of the encode side is asked.
        if carries_a_column(facts) {
            return declined(AFFINITY_COLUMN);
        }
        debug_assert_indexed(encode.encoders);
        let needs = !items.keys.is_empty();
        // A profile that shows its rows to a plugin answers inside a scope
        // opened over the rows of one decision. None is opened over the
        // encoders offered to a key, so no stage of such a profile is asked.
        if needs && self.marks[usize::from(index)].scoped() {
            return declined(ENCODE_SCOPE);
        }
        let images = facts.media.images != 0;
        // A request with no item that asks no images of its decode is the
        // pair's alone: the pair entry answers it over the pools as handed,
        // and no pass of the encode side is laid for it.
        if !needs && !items.decode_images && !(items.references && images) {
            return self.decide_pair(facts, pool, pools, cx, decode_cx, scratch);
        }
        let encode_facts = RequestFacts {
            phase: Phase::Encode,
            affinity: encode.affinity,
            // A pin is a position in another pool: the encode side takes none.
            pinned_at: 0,
            ..*facts
        };
        let takes = |row: &Candidate, of: &EncoderFacts| takes_media(&items, images, row, of);
        let open = |row: &Candidate| {
            let hard = profile.filters.iter().filter(|f| !f.soft());
            hard.into_iter()
                .all(|f| f.keep(&encode_facts, row, encode_cx))
        };
        if needs {
            // An encode class is numbered over the whole pool, and a view
            // hands fewer rows than the pool holds: one bit a number up to
            // the count of the rows handed, and a list for the numbers
            // above it. No number is dropped and no set is wider than the
            // fleet.
            let width = encode.encoders.len() + 1;
            side.open.reset(width);
            side.member.reset(width);
            side.far_open.clear();
            side.far_member.clear();
            let mut any_open = false;
            let mut any_member = false;
            side.opened.clear();
            for (at, (row, of)) in (0u32..).zip(encode.encoders.iter().zip(encode.facts)) {
                if of.class != 0 && takes(row, of) {
                    any_member = true;
                    if !side.member.insert(of.class) {
                        side.far_member.push(of.class);
                    }
                    if open(row) {
                        any_open = true;
                        side.opened.push(at);
                        if !side.open.insert(of.class) {
                            side.far_open.push(of.class);
                        }
                    }
                }
            }
            if !any_open {
                // Encoders the overload veto left none of are a saturated
                // fleet, and the shed says so as a pair's side says it of
                // its own rows.
                if let Some(stage) =
                    saturation(profile, &encode_facts, encode_cx, &encode, &items, images)
                {
                    return turned_away(ShedReason::Saturated, stage, bare);
                }
                return gap(encode_gap(&encode, &items, images, any_member));
            }
            side.far_open.sort_unstable();
            side.far_member.sort_unstable();
        }
        // The pairs by membership: one at all, one an encoder serves, and
        // one whose decode takes images. Every request that comes here asks
        // one of them: the request that asks none left by the early answer.
        //
        // The decodes' classes stand in sets of bits no wider than the
        // fleet. Under a table the width is the table's own, as the pair
        // entry keeps it, and a number above it is no class of the table.
        // Under none a class pairs by its number, whatever the number: one
        // bit a number up to the count of the decodes, and a list for the
        // numbers above it, which a dense numbering never names.
        let class_of = |at: usize| encode.prefill_classes.get(at).copied().unwrap_or(0);
        let (mut any_pair, mut any_served, mut any_vision) = (false, false, false);
        let tabled = !pools.table.is_empty();
        let width = if tabled {
            pools.table.width()
        } else {
            pools.decodes.len() + 1
        };
        side.decodes_of.reset(width);
        side.vision_of.reset(width);
        side.far.clear();
        side.far_vision.clear();
        for row in pools.decodes.iter().filter(|row| row.pair_class != 0) {
            if side.decodes_of.insert(row.pair_class) {
                if row.vision {
                    side.vision_of.insert(row.pair_class);
                }
            } else if !tabled {
                side.far.push(row.pair_class);
                if row.vision {
                    side.far_vision.push(row.pair_class);
                }
            }
        }
        side.far.sort_unstable();
        side.far_vision.sort_unstable();
        // The three answers only rise: the pass ends at the prefill
        // that gives the last of them.
        for (at, row) in pools.prefills.iter().enumerate() {
            let listed = pools.table.of(&row.pair_class);
            if listed
                .iter()
                .any(|&class| holds(&side.decodes_of, &side.far, class))
            {
                any_pair = true;
                if !needs || holds(&side.member, &side.far_member, class_of(at)) {
                    any_served = true;
                    if listed
                        .iter()
                        .any(|&class| holds(&side.vision_of, &side.far_vision, class))
                    {
                        any_vision = true;
                        break;
                    }
                }
            }
        }
        if needs && any_pair && !any_served {
            return gap(EC_COMPATIBLE);
        }
        if items.decode_images && any_served && !any_vision {
            return gap(DECODE_VISION);
        }
        let vision = items.decode_images || (items.references && images && any_vision);
        // The rows the pair is picked among. With every row admitted the
        // pools are the host's own; else each side's admitted rows stand in
        // a pool of their own and the targets are read back by where a row
        // stood.
        let mut admitted = pools;
        let open_class = |class: u32| holds(&side.open, &side.far_open, class);
        let every_prefill = !needs
            || encode
                .prefill_classes
                .iter()
                .all(|&class| open_class(class));
        // A list handed on in a pool's stead is numbered anew, and a mask
        // numbers the rows of the host's pool: beside such a list the mask
        // is numbered with the rows.
        let (prefill_mask, decode_mask);
        let (mut cx, mut decode_cx) = (cx, decode_cx);
        if !every_prefill {
            side.prefills.clear();
            for (at, row) in pools.prefills.iter().enumerate() {
                if open_class(class_of(at)) {
                    side.prefills.push(row, at, pools.prefill_affinity);
                }
            }
            side.prefills.take_room();
            if !cx.stalled.is_empty() {
                side.prefills.renumber(cx.stalled);
                prefill_mask = Constraints {
                    stalled: &side.prefills.masked,
                    ..*cx
                };
                cx = &prefill_mask;
            }
            admitted.prefills = &side.prefills.rows;
            admitted.prefill_affinity = &side.prefills.affinity;
        }
        let every_decode = !vision || pools.decodes.iter().all(|row| row.vision);
        if !every_decode {
            side.decodes.clear();
            for (at, row) in pools.decodes.iter().enumerate() {
                if row.vision {
                    side.decodes.push(row, at, pools.decode_affinity);
                }
            }
            side.decodes.take_room();
            if !decode_cx.stalled.is_empty() {
                side.decodes.renumber(decode_cx.stalled);
                decode_mask = Constraints {
                    stalled: &side.decodes.masked,
                    ..*decode_cx
                };
                decode_cx = &decode_mask;
            }
            admitted.decodes = &side.decodes.rows;
            admitted.decode_affinity = &side.decodes.affinity;
        }
        let pair = self.decide_pair(facts, pool, admitted, cx, decode_cx, scratch);
        let Decision::Pick(mut pick) = pair else {
            return pair;
        };
        if !every_prefill || !every_decode {
            let mut targets = Targets::new();
            for target in pick.targets.iter() {
                let mut target = *target;
                if !every_prefill {
                    target.position = side.prefills.origin(target.position);
                }
                if let Some(decode) = target.pair.as_mut().filter(|_| !every_decode) {
                    decode.decode_position = side.decodes.origin(decode.decode_position);
                }
                targets.push(target);
            }
            pick.targets = targets;
        }
        if !needs {
            return Decision::Pick(pick);
        }
        // The alternates of a pair keep its prefill, so the first target's
        // class is every target's.
        let chosen = pick.targets.first().map(|target| target.position as usize);
        let class = chosen.map_or(0, class_of);
        side.offered.clear();
        let mut pushes = false;
        for &at in &side.opened {
            let at = at as usize;
            let (Some(row), Some(of)) = (encode.encoders.get(at), encode.facts.get(at)) else {
                continue;
            };
            if of.class == class {
                pushes = of.pushes;
                side.offered.push(row, at, encode.affinity);
            }
        }
        side.offered.take_room();
        // The offered encoders are numbered anew unless they are the whole
        // pool, and the encode side's mask is numbered with them.
        let offered_mask;
        let mut encode_cx = encode_cx;
        if !encode_cx.stalled.is_empty() && side.offered.rows.len() != encode.encoders.len() {
            side.offered.renumber(encode_cx.stalled);
            offered_mask = Constraints {
                stalled: &side.offered.masked,
                ..*encode_cx
            };
            encode_cx = &offered_mask;
        }
        let one = items.one.filter(|_| pushes);
        placed.one = one.is_some();
        let keys = one.as_ref().map_or(items.keys, std::slice::from_ref);
        // Each key draws its own sequence, derived from the request's seed.
        let mut seeds = Draw::new(Draw::new(facts.seed).next_u64());
        for key in keys {
            let key_facts = RequestFacts {
                session_key: *key,
                seed: seeds.next_u64(),
                affinity: &side.offered.affinity,
                ..encode_facts
            };
            let of_key = run_profile(
                profile,
                index,
                &key_facts,
                pool,
                Set::side(&side.offered.rows, None, None),
                encode_cx,
                &mut side.run,
            );
            match of_key {
                Decision::Pick(of_key) => {
                    let Some(target) = of_key.targets.first() else {
                        return gap(ENCODERS);
                    };
                    placed.encoders.push(Target {
                        position: side.offered.origin(target.position),
                        pair: None,
                        ..*target
                    });
                }
                Decision::Shed(_) | Decision::Queue(_) => return of_key,
            }
        }
        Decision::Pick(pick)
    }
}
