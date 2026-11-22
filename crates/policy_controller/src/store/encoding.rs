//! The encode classes: which encoders hand an encoded item to which
//! prefills, decided where a membership write moves a member or an offer
//! and again where a generation moves the pairing mode. A decision that
//! carries encode items reads them as the class its chosen prefill names
//! and that class's list of encoders.
//!
//! One invariant: **a worker states its own end of the encode handoff, and
//! an encoder hands to the prefills that state the same end.** An end is
//! the worker's runtime, the form its registration states for the handoff
//! ([`EncodeHandoff`]) and, under a push, the connector it states its role on.
//! The encoders of one end are one class, a prefill names the class of its
//! end, and no item is handed across two classes.
//!
//! Nothing is read off another fact. A worker that states no form has no
//! end, which is how a registration that never heard of the form reads. A
//! runtime's name does not say its form and a role does not either: under a
//! push an encoder states the producer role and a prefill the consumer
//! role, each on a connector, and a rendezvous asks for neither.
//!
//! The derivation ([`EncodeClasses::derive`]) is a function of the
//! membership and of nothing else. A worker stands in it at its place in
//! the order given, and the classes are numbered as their first encoder
//! comes. The store gives each side as the rows of its pool: a membership
//! write puts a side in id order before it derives a class and builds the
//! pool's rows in that order, so a place here is a row there, and two
//! stores that hold one membership number the classes alike.
//!
//! **A class's number is a name inside one publication.** A write that
//! takes a class's first encoder away numbers the classes again, so a
//! number is read with its list under one borrow of the publication, never
//! compared across two and never kept.
//!
//! Every value here is opaque text: a runtime and a connector are compared
//! for equality and never interpreted. Deriving an offer from what an
//! engine reports is the registration handshake's.
//!
//! What a stated value is follows the pairing rule
//! ([`pairing`](super::pairing)): a runtime is a kind's name, read as
//! written, and an empty one is unknown; a connector is free text, which
//! the store's door normalizes once a write ([`EncodeOffer::normalized`]),
//! so a blank one states no connector.

use rustc_hash::FxHashMap;

use super::pairing::{known, normalize};
use crate::engine::EncoderFacts;

#[cfg(test)]
mod tests;

/// The form a worker's registration states for the encode handoff.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EncodeHandoff {
    /// No form is stated: the worker has no end of an encode handoff.
    #[default]
    None,
    /// The encoder pushes an encoded item to the prefill over a connector.
    Push,
    /// The encoder and the prefill meet at a rendezvous.
    Rendezvous,
}

/// A worker's end of the encode handoff, as its registration states it. The
/// default states nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EncodeOffer {
    pub handoff: EncodeHandoff,
    /// The worker states the producer role: an encoder's, under a push.
    pub producer: bool,
    /// The worker states the consumer role: a prefill's, under a push.
    pub consumer: bool,
    /// The connector the role is stated on.
    pub connector: Option<String>,
    /// The worker, an encoder, processes an item's references itself.
    pub references: bool,
}

impl EncodeOffer {
    /// The offer as the rule reads it: a blank connector dropped and a name
    /// read past the blanks around it, by the one function that normalizes
    /// a pairing descriptor's values. The store's door makes a worker's
    /// offer so, once a write.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        normalize(&mut self.connector);
        self
    }
}

/// One encoder or prefill as the derivation reads it.
#[derive(Clone, Copy, Debug)]
pub struct EncodeMember<'a> {
    /// The worker's runtime; empty when it is unknown.
    pub runtime: &'a str,
    pub offer: &'a EncodeOffer,
}

/// The side of the handoff a worker stands on.
#[derive(Clone, Copy)]
enum Part {
    Encoder,
    Prefill,
}

/// A worker's end: its runtime, the form and, under a push, the connector.
/// No end holds an empty name, and a rendezvous has no connector at all: a
/// key is compared once a worker, and two empty names compare slowly in
/// some C libraries' vectorized `memcmp`.
type End<'a> = (&'a str, EncodeHandoff, Option<&'a str>);

/// The end `member` states as `part`, or `None` when it states none: no
/// form, an unknown runtime, or under a push no role of that part on a
/// connector.
fn end<'a>(member: &EncodeMember<'a>, part: Part) -> Option<End<'a>> {
    let runtime = known(member.runtime)?;
    let offer = member.offer;
    match offer.handoff {
        EncodeHandoff::None => None,
        EncodeHandoff::Rendezvous => Some((runtime, EncodeHandoff::Rendezvous, None)),
        EncodeHandoff::Push => {
            let stated = match part {
                Part::Encoder => offer.producer,
                Part::Prefill => offer.consumer,
            };
            let connector = offer.connector.as_deref().filter(|name| !name.is_empty())?;
            stated.then_some((runtime, EncodeHandoff::Push, Some(connector)))
        }
    }
}

/// Whether `encoder` hands an encoded item to `prefill`: the rule, worker by
/// worker. Each states an end and the two ends are one.
pub fn hands_to(encoder: &EncodeMember<'_>, prefill: &EncodeMember<'_>) -> bool {
    match (end(encoder, Part::Encoder), end(prefill, Part::Prefill)) {
        (Some(ours), Some(theirs)) => ours == theirs,
        _ => false,
    }
}

/// The encode classes of one model's encoders and prefills: each worker's
/// class at its place in the order the derivation was given, and each
/// class's encoders.
///
/// A class's number is a name inside the publication that holds these
/// classes. A later write may number the classes again, so a number is read
/// with its list under one borrow, never compared across two publications
/// and never kept.
///
/// The default holds no class and no table, and names class `0` at every
/// place.
///
/// Beside the replay of the recorded rows and the cases, nothing reads a
/// class's list or form yet.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EncodeClasses {
    /// Each encoder's class, in the order given.
    encoders: Box<[u32]>,
    /// The class each prefill names, in the order given.
    prefills: Box<[u32]>,
    /// Where class `k`'s list ends in `listed`, at `k - 1`.
    ends: Box<[u32]>,
    /// The lists one after another: each class's encoders, as places in the
    /// order given, ascending.
    listed: Box<[u32]>,
    /// Each class's form, at `k - 1`.
    forms: Box<[EncodeHandoff]>,
}

impl EncodeClasses {
    /// Derive the encode classes of one model's `encoders` and `prefills`.
    ///
    /// The encoders that state one end are one class. The classes are
    /// numbered from `1` in the order their first encoder is given; an
    /// encoder that states no end is class `0` and takes no number. A
    /// prefill names the class of the end it states, and `0` when it states
    /// none or no encoder states the same.
    ///
    /// When no encoder states an end there is no class: the prefills are
    /// not read and the classes hold no table.
    pub fn derive<'a>(
        encoders: impl IntoIterator<Item = EncodeMember<'a>>,
        prefills: impl IntoIterator<Item = EncodeMember<'a>>,
    ) -> Self {
        let encoders = encoders.into_iter();
        let mut numbers: FxHashMap<End<'a>, u32> = FxHashMap::default();
        let mut forms = Vec::new();
        let mut classed = Vec::with_capacity(encoders.size_hint().0);
        for member in encoders {
            let class = end(&member, Part::Encoder).map_or(0, |end| {
                let next = forms.len() as u32 + 1;
                let class = *numbers.entry(end).or_insert(next);
                if class == next {
                    forms.push(end.1);
                }
                class
            });
            classed.push(class);
        }
        if forms.is_empty() {
            return Self::default();
        }
        let naming = prefills
            .into_iter()
            .map(|member| {
                let named = end(&member, Part::Prefill).and_then(|end| numbers.get(&end));
                named.copied().unwrap_or(0)
            })
            .collect();
        // The lists, one after another, each in the order its encoders are
        // given: every class's count first, then where each list ends, then
        // each encoder at its list's next place.
        let mut ends = vec![0u32; forms.len()];
        for &class in &classed {
            if let Some(count) = (class as usize).checked_sub(1) {
                ends[count] += 1;
            }
        }
        let mut next = Vec::with_capacity(ends.len());
        let mut total = 0;
        for end in &mut ends {
            next.push(total);
            total += *end;
            *end = total;
        }
        let mut listed = vec![0u32; total as usize];
        for (at, &class) in (0u32..).zip(&classed) {
            if let Some(list) = (class as usize).checked_sub(1) {
                listed[next[list] as usize] = at;
                next[list] += 1;
            }
        }
        Self {
            encoders: classed.into_boxed_slice(),
            prefills: naming,
            ends: ends.into_boxed_slice(),
            listed: listed.into_boxed_slice(),
            forms: forms.into_boxed_slice(),
        }
    }

    /// Each encoder's class, in the order given: `0` for an encoder that
    /// states no end, else a class numbered from `1`. Empty when there is
    /// no class.
    pub fn encoders(&self) -> &[u32] {
        &self.encoders
    }

    /// The class each prefill names, in the order given: `0` for a prefill
    /// no encoder hands to. Empty when there is no class.
    pub fn prefills(&self) -> &[u32] {
        &self.prefills
    }

    /// The class the prefill at `place` names: `0` when no encoder hands to
    /// it, and at every place when there is no class.
    pub fn prefill_class(&self, place: usize) -> u32 {
        self.prefills.get(place).copied().unwrap_or(0)
    }

    /// What a decision reads of each encoder beside its row
    /// ([`EncoderFacts`]), one for each of `offers`: the offers the encoders
    /// state, in the order the classes were derived in.
    ///
    /// An encoder's class is the one it has here: `0` where it states no
    /// end, where there is no class, and for an offer past the encoders
    /// derived. It pushes where its registration states the pushing form,
    /// with its role or without. It takes an item's references where its
    /// offer says so.
    ///
    /// The store writes these on a model's encode pool. A host that holds
    /// its own lists of workers writes them for its encoders by this one
    /// rule.
    pub fn encoder_facts<'a, C>(&self, offers: impl IntoIterator<Item = &'a EncodeOffer>) -> C
    where
        C: FromIterator<EncoderFacts>,
    {
        let stated = offers
            .into_iter()
            .enumerate()
            .map(|(at, offer)| EncoderFacts {
                class: self.encoders.get(at).copied().unwrap_or(0),
                pushes: offer.handoff == EncodeHandoff::Push,
                references: offer.references,
            });
        stated.collect()
    }

    /// How many classes there are: the distinct ends the encoders state.
    pub fn classes(&self) -> u32 {
        self.ends.len() as u32
    }

    /// The encoders of class `class`, as places in the order given,
    /// ascending; empty for class `0` and for a class past the last.
    pub fn list(&self, class: u32) -> &[u32] {
        let Some(at) = (class as usize).checked_sub(1) else {
            return &[];
        };
        let Some(&end) = self.ends.get(at) else {
            return &[];
        };
        let start = at.checked_sub(1).map_or(0, |before| self.ends[before]);
        &self.listed[start as usize..end as usize]
    }

    /// The form of class `class`: [`EncodeHandoff::None`] for class `0` and for a
    /// class past the last.
    pub fn form(&self, class: u32) -> EncodeHandoff {
        let form = (class as usize)
            .checked_sub(1)
            .and_then(|at| self.forms.get(at));
        form.copied().unwrap_or_default()
    }

    /// The bytes the classes hold: a class a worker, and each class's list
    /// and form.
    pub fn bytes(&self) -> usize {
        size_of_val(&*self.encoders)
            + size_of_val(&*self.prefills)
            + size_of_val(&*self.ends)
            + size_of_val(&*self.listed)
            + size_of_val(&*self.forms)
    }
}
