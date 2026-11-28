//! The request of one call ([`InProcess::place`]): what a caller states of
//! it, the request as the engine makes it of that, and what one ask of the
//! call hands on to the next.
//!
//! A caller states a [`PlaceRequest`]: its prompt ([`Prompt`]), its headers
//! ([`HeaderRead`]) and its route's own rule ([`WorkerRule`]) are among its
//! fields. The engine makes a [`Made`] of it for one decision: the
//! request's keys, where it stands on a ring its caller lends
//! ([`Made::stated`]), and its prompt's block hashes where a lookup asks
//! them late ([`Made::hashed_whole`]). A [`Call`] is what one ask of a call
//! hands on to the next.
//!
//! The request's making for a decision, `impl RequestMaker for Made`, is
//! written in the placing call's module (`place`): it reads a pair's lists
//! and an encode request's, which the deciders' modules declare, and this
//! module names no decider. A decider is handed the made request for its
//! prompt's late hashes.
//!
//! Whether a worker is ready, healthy and let through by its circuit
//! breaker ([`ready`]), stands here because the placing call and a
//! decider's short path both read it. Whether its route's own rule admits
//! it is another question ([`PlaceRequest::admits`]).
//!
//! The fields of [`Made`] and of [`Call`] are open to the modules of the
//! in-process engine (`pub(super)`) for the placing call, which builds a
//! made request by its fields and reads what a decision left in them. A
//! module beside the call goes by the methods.

#![expect(
    clippy::exhaustive_structs,
    reason = "a caller still builds or matches these types whole: each mark comes with its call"
)]

use std::{
    any::Any,
    cell::{Cell, OnceCell, RefCell},
    fmt,
    time::Instant,
};

use smg_worker::{identity::basic::Worker, runtime::RoutingState};

use super::{InProcess, Loaded};
// Named by the doc links of this module alone.
#[cfg(doc)]
use crate::engine::{
    decision_engine::RequestMaker,
    view::{Constraints, RequestFacts},
};
use crate::{
    document::{spec::AppliedSpec, DerivedKeys, KeySource},
    engine::{
        decision::{PairSide, MAX_TARGETS},
        decision_engine::{TextKey, TokenKey, WorkerLists},
        demands::Demands,
        framework::EncodeItems,
        prompt_hashes::{HashSeeds, PromptHashes},
        view::{Header, MediaCounts, Phase, WorkerId},
    },
    policies::{
        consistent_hashing::{Head, LentRing},
        prefix_hash::HeadRule,
    },
};

/// A request's headers, as its caller holds them.
///
/// The engine reads the headers the document names: the sources of a
/// request's routing value and of its client's value, and the headers its
/// view shows a stage ([`AppliedSpec::view_headers`]). It reads one more
/// where the request's host names it, for the id a kept decision is found
/// by ([`PlaceRequest::explain_id_header`]), and no other.
pub trait HeaderRead {
    /// The value of the header `name`, where the request carries it. `name`
    /// is lowercase, as a document writes it.
    fn header(&self, name: &str) -> Option<&[u8]>;
}

/// A request that carries no header.
impl HeaderRead for () {
    fn header(&self, _: &str) -> Option<&[u8]> {
        None
    }
}

/// Headers as pairs of a lowercase name and a value. Where a name stands
/// twice, the first pair is the header.
impl<const N: usize> HeaderRead for [(&str, &str); N] {
    fn header(&self, name: &str) -> Option<&[u8]> {
        let (_, value) = self.iter().find(|(held, _)| *held == name)?;
        Some(value.as_bytes())
    }
}

/// A request's headers as an HTTP server holds them. The map keeps its
/// names lowercase, and where a name stands twice the first value is the
/// header. A value that is not text is handed as its bytes: a source that
/// reads it passes it over.
impl HeaderRead for http::HeaderMap {
    fn header(&self, name: &str) -> Option<&[u8]> {
        self.get(name).map(http::HeaderValue::as_bytes)
    }
}

/// A prompt's entry in a tree its caller lends, as the caller names it: the
/// one question a decision asks where it reads the caller's token tree
/// ([`CacheIndexes::token_tree`]), and the one it asks where it reads the
/// caller's string tree ([`CacheIndexes::string_tree`]). A caller of
/// [`InProcess::place`] answers them for its request
/// ([`Prompt::entry`]), as a maker written by hand answers
/// [`RequestMaker::token_key`] and [`RequestMaker::text_key`].
///
/// A caller that keeps no string tree answers the first alone.
///
/// [`CacheIndexes::token_tree`]: crate::engine::CacheIndexes::token_tree
/// [`CacheIndexes::string_tree`]: crate::engine::CacheIndexes::string_tree
pub trait PromptEntry {
    /// The entry by token ids for a tree whose page is `page` tokens
    /// ([`TokenKey`]): `None` where the caller holds no token ids of the
    /// prompt. The marker of a cache partition fills whole pages, which is
    /// why the page is asked with.
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>>;

    /// The entry by text for the caller's string tree ([`TextKey`]):
    /// `None`, the default, where the caller holds the prompt's token ids
    /// or no text of it, or lends no string tree.
    fn text_key(&self) -> Option<TextKey<'_>> {
        None
    }
}

/// A request's prompt, as its caller holds it: what the stages that place a
/// prompt on the worker that holds its beginning read.
#[derive(Clone, Copy)]
pub struct Prompt<'a> {
    /// The prompt's token ids, where the caller tokenized it.
    pub tokens: Option<&'a [u32]>,
    /// The prompt's text, where the caller holds no token ids. Token ids
    /// win where both are stated.
    pub text: Option<&'a str>,
    /// The two seeds of the prompt's cache partition, and
    /// [`HashSeeds::PLAIN`] for a request with none. The caller's own rule
    /// keeps two partitions apart.
    pub seeds: HashSeeds,
    /// The prompt's entry in the tree the caller lends: its token tree
    /// ([`CacheIndexes::token_tree`]), or its string tree for a prompt it
    /// holds as text ([`CacheIndexes::string_tree`]). The caller's own
    /// rule names it as the seeds are: the same cache partition, for a
    /// tree's lookup. `None` from a caller that lends no tree: the
    /// request is then named its holders from the record of hashed
    /// prefixes. One entry answers for every tree the request is decided
    /// in, a pair's two sides too when a pair reads one: the token tree's
    /// page is that question's argument.
    ///
    /// It is asked once by each decision of the call that reads the tree,
    /// and never where no stage of the request's place reads the tree. The
    /// tree then stores the prompt for the worker the call can send to:
    /// the first target of that decision that is ready and that the
    /// route's rule admits ([`PlaceRequest::admits`]), and no worker where
    /// no target is.
    ///
    /// The object is borrowed as its caller holds it and is not asked to
    /// be [`Sync`], so a prompt is neither [`Send`] nor [`Sync`], whether
    /// it states an entry or not: its caller makes it with the request,
    /// between two awaits.
    ///
    /// [`CacheIndexes::token_tree`]: crate::engine::CacheIndexes::token_tree
    /// [`CacheIndexes::string_tree`]: crate::engine::CacheIndexes::string_tree
    pub entry: Option<&'a dyn PromptEntry>,
}

impl Prompt<'_> {
    /// A request that states no prompt.
    pub const NONE: Self = Self {
        tokens: None,
        text: None,
        seeds: HashSeeds::PLAIN,
        entry: None,
    };
}

/// Sizes alone: a client's tokens and text reach no log line.
impl fmt::Debug for Prompt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Prompt")
            .field("tokens", &self.tokens.map(<[u32]>::len))
            .field("text", &self.text.map(str::len))
            .field("seeds", &self.seeds)
            .field("entry", &self.entry.is_some())
            .finish()
    }
}

/// A route's own rule for a worker: whether the worker can take the
/// request, by what the route alone knows of the two
/// ([`PlaceRequest::admits`]).
pub type WorkerRule<'a> = &'a dyn Fn(&dyn Worker) -> bool;

/// What a caller states of one request.
///
/// Every field is the caller's own reading of its request. A caller names
/// the fields it holds over [`PlaceRequest::of`], which states nothing
/// else:
/// `PlaceRequest { streaming: true, tried: &tried, ..PlaceRequest::of(model, lists) }`.
///
/// A request is not [`Send`]: its headers, its route's rule and its
/// prompt's entry in a lent tree are borrowed as their caller holds them,
/// and none is asked to be [`Sync`]. A caller makes one for each call,
/// between two awaits.
pub struct PlaceRequest<'a> {
    /// The model the request is for, as its caller resolved it.
    pub model: &'a str,
    /// The workers the request may be sent to, as the caller holds them
    /// for this request: one list, a pair's two with their pair classes,
    /// or an encode request's three.
    pub lists: WorkerLists<'a>,
    /// The request's prompt.
    pub prompt: Prompt<'a>,
    /// The request's headers.
    pub headers: &'a dyn HeaderRead,
    /// The request's own id, where it has one: a source of its routing
    /// value where the document names it, and what a decision the engine
    /// kept of the request is found by ([`InProcess::explained`]). A
    /// request with none is found by its `explain_id`, or else by the
    /// header its `explain_id_header` names, and is not kept where it
    /// states none of the three.
    pub request_id: Option<&'a str>,
    /// The id a decision the engine kept of the request is found by where
    /// the request states no id of its own: its host's name for it. It is
    /// no source of a routing value and no lease holds it. The engine
    /// reads it under a document that keeps a sample, and nowhere else.
    pub explain_id: Option<&'a str>,
    /// The name of the header that holds its host's id of the request,
    /// lowercase: for a host that holds that id among the request's
    /// headers and states no `explain_id`, which wins. Under a document
    /// that keeps a sample the engine reads the header's value off
    /// `headers`, for a request that states no id of its own. Under a
    /// document that keeps none, no request pays a read for it.
    pub explain_id_header: Option<&'a str>,
    /// The tenant the request is of.
    pub tenant: Option<&'a str>,
    /// The request's priority band.
    pub priority: i32,
    /// Whether the answer is streamed.
    pub streaming: bool,
    /// The most tokens the answer may hold. Zero states none.
    pub max_output_tokens: u32,
    /// The media the request carries, by count.
    pub media: MediaCounts,
    /// What the request demands of the worker that takes it.
    pub demands: Demands,
    /// The items of an encode request, which hands its three lists
    /// ([`WorkerLists::Encode`]), and `None` for every other. Three lists
    /// with `None` are placed as their pair. The call lends the items the
    /// ring this request states ([`Self::ring`]): a ring written into the
    /// items themselves ([`EncodeItems::ring`]) is not read.
    pub encode: Option<EncodeItems<'a>>,
    /// The retry number, `0` for the first attempt. It is stated and not
    /// counted: a failed attempt of a pair names two workers as tried.
    pub attempt: u8,
    /// The workers the request's failed attempts were sent to, a pair's
    /// two sides each by its own id.
    pub tried: &'a [WorkerId],
    /// The time left for the decision, in microseconds. Zero leaves it
    /// unbounded. A single request's first decision is given all of it,
    /// and each later one what its call has left: the call reads the
    /// clock once ahead of its first decision, for a request that states
    /// a budget and for no other. Each decision of a pair is given all of
    /// it, and each of an encode request.
    pub budget_micros: u32,
    /// Whether the caller asks a pair's decode again behind its prefill's
    /// answer. Where the document picks the decode after the prefill, such
    /// a request's first decision books the prefill alone. A caller that
    /// sends both sides at once states nothing here and keeps one decision
    /// under every document. The word is a pair's: over one list, or over
    /// none, it asks nothing. The request made states no later decode
    /// then, and a first attempt that carries it is plain still. An
    /// encode request whose decode is held to the workers that take images
    /// ([`EncodeItems::decode_images`], or references beside an image or a
    /// video) books its decode with its prefill whatever it states here: a
    /// decode asked again reads no image fact. So does one with items
    /// that demands vision or a way of processing its references
    /// ([`Self::demands`]): its encoders meet that part of the demand, and
    /// a decode asked again would hold the decode to it. So does a pair
    /// that names its decode, on both sides or on the decode's alone: a
    /// decode asked again would be picked by the pair's routing value.
    pub decode_again: bool,
    /// Whether the caller counts this request on the decode its first
    /// decision named while it asks that decode again: one of that
    /// decode's requests in flight, as a guard the caller holds on that
    /// decode across the call counts it. The decode asked again then reads
    /// that decode with one request less in flight
    /// ([`Constraints::counted_on_decode`]), and every other decode as it
    /// stands. That decode is the worker of its id in the list handed at
    /// this call, and its count of session keys is read as the worker
    /// states it. Where a stage reads the count of the request's key on a
    /// worker, that decode is read with one of the key's requests less
    /// too. A caller that states nothing here is read as it counts:
    /// a request it counted on that decode stands among that decode's
    /// requests. A first decision reads nothing here.
    pub counted_on_decode: bool,
    /// The hash ring the caller keeps of the model's workers, lent to the
    /// request's decision ([`LentRing`]): one ring for a pair's two lists,
    /// and no part of a list. Under a document whose ring stage places a
    /// routing value by its own text, the value goes to the worker this
    /// ring's walk reaches, where the caller's own ring policy puts it:
    /// the engine reads the value's place on the ring with the request's
    /// keys. Under the prefix form of the ring stage the engine asks the
    /// ring where each head of the request stands
    /// ([`LentRing::head_position`]). A ring that answers nothing for a
    /// head is asked about no further head and is not lent, and neither
    /// is a ring to a request that has no head. A pair whose two sides
    /// read the ring by two words is asked for each side's own places: a
    /// key's own place for one side and its heads' for the other, or the
    /// heads two rules cut. Its ring stays lent while either side has a
    /// place, and a side the ring places nothing for goes by its stage's
    /// own ring. Under any other document the ring is asked nothing.
    /// `None` from a caller that keeps no ring: the stage places by its
    /// own.
    ///
    /// An encode request's items are lent this ring too
    /// ([`EncodeItems::ring`]). An item that carries its text goes to the
    /// encoder the ring's walk reaches from where that text stands, where
    /// the caller's own ring policy puts an item. The encode profile's
    /// word alone says whether the ring is asked for an item and by which
    /// of its two questions: a request whose own profile reads no ring
    /// lends its items one all the same.
    pub ring: Option<&'a dyn LentRing>,
    /// The route's own rule for a worker its request may be sent to, and
    /// `None` for a route that has none. It is asked of the workers the
    /// engine picked: a request whose first pick it admits pays one call
    /// of it, and a pair's request one for each of its two workers. A
    /// request whose decision stores where its prompt went, in a token
    /// tree its caller lends ([`Prompt::entry`]) or in the record of sent
    /// prefixes a `cache_aware` profile keeps, has the rule asked inside
    /// that decision, ahead of the write, so the prompt is stored for a
    /// worker the rule takes. The pick takes that answer and asks the rule
    /// of that worker no second time. A rule asked there runs inside a
    /// decision: it asks the engine nothing, and where it panics the prompt
    /// is stored for no worker and the panic reaches the caller behind the
    /// decision.
    /// Behind the first decision none of whose picks can be sent to,
    /// whether it refused them or they were not ready, it is asked of
    /// every worker of the list, once a call, and the call decides again
    /// among the workers it takes ([`InProcess::place`]): those of a
    /// single request's list, and those of each of a pair's two, among the
    /// places its caller states for that side. A first call may ask the
    /// rule of any worker
    /// of its lists, picked or not. A pick it is asked of is ready. A
    /// worker of a list it is asked of may not be: it answers for a worker
    /// in any state. No request is sent to a worker it refuses when it is
    /// asked of that worker as a pick, whatever it said of the worker
    /// before. A rule that holds for every request of a route belongs to
    /// the list its caller hands. For a single request that names its
    /// worker ([`InProcess::place`]) it is asked ahead of the call's
    /// decision too, of each available worker the request's number counts.
    /// For a pair that names its worker it is asked there too, in each
    /// list's order, of workers that meet the request's hard demands at
    /// places the caller states for the list and does not mask. Where the
    /// prefill's side reads the name, they are the available workers of
    /// the two lists and, where the rule takes no available decode under
    /// a binding that steers a pool all over its thresholds, the decodes
    /// that pass every veto but the overload one. Where the decode's side
    /// alone reads the name, they are the available decodes, and the
    /// available prefills as far as the first that opens the pair: the
    /// rule is asked of no prefill behind that one. Where none of them
    /// opens it, under a profile that places among prefills over their
    /// thresholds, the prefills that pass every veto but the overload one
    /// are asked so too, an available one a second time. Where a prefill
    /// opens the pair, the rule is then asked of every prefill at the
    /// places the caller states, ready or not.
    ///
    /// Where a pair's decode is decided again on its decode side
    /// ([`PlacementSide::decode_again`](crate::tracking::PlacementSide::decode_again)),
    /// the rule is asked of that decision's picks and of no other worker,
    /// under the lock the pair's two sides share. It must leave both sides
    /// alone: the send or the drop of one, from inside the rule, waits on
    /// that lock and never returns. That decision asks the request's
    /// headers, its ring, its lists' admission and their workers' own calls
    /// under the same lock, and each is held to the same.
    ///
    /// A rule that takes something of a worker on a yes is sound only
    /// where a pair's decode is decided again, which asks it of picks
    /// alone, best first, and of none behind the first it admits: at a
    /// first placement a rule answers by reading alone, because a request
    /// that names its worker asks it of workers the request is never sent
    /// to.
    ///
    /// For an encode request the rule is its pair's: it is asked of the
    /// pair's two workers and of the pair's two lists, and of no encoder.
    pub admits: Option<WorkerRule<'a>>,
}

impl<'a> PlaceRequest<'a> {
    /// A first attempt for `model` over `lists` that states no other fact:
    /// no prompt, no header, no id, no tenant, no media, no demand, no
    /// encode item, nothing tried, no budget, no second ask of a decode, no
    /// count on a decode, no ring and no rule of its route.
    #[must_use]
    pub fn of(model: &'a str, lists: WorkerLists<'a>) -> Self {
        Self {
            model,
            lists,
            prompt: Prompt::NONE,
            headers: &(),
            request_id: None,
            explain_id: None,
            explain_id_header: None,
            tenant: None,
            priority: 0,
            streaming: false,
            max_output_tokens: 0,
            media: MediaCounts::NONE,
            demands: Demands::default(),
            encode: None,
            attempt: 0,
            tried: &[],
            budget_micros: 0,
            decode_again: false,
            counted_on_decode: false,
            ring: None,
            admits: None,
        }
    }

    /// The phase the request is decided at: a pair is asked at its
    /// prefill, over its two lists or an encode request's three, and every
    /// other request is single.
    pub(super) const fn phase(&self) -> Phase {
        match self.lists {
            WorkerLists::Pair(_) | WorkerLists::Encode(_) => Phase::Prefill,
            WorkerLists::None | WorkerLists::Single(_) => Phase::Single,
        }
    }

    /// Whether the request asks a pair's decode again
    /// ([`Self::decode_again`]). The word is a pair's, over its two lists
    /// or an encode request's three: over one list, and over none, it asks
    /// nothing. The request made and its plain test both read this.
    pub(super) const fn asks_decode_again(&self) -> bool {
        self.decode_again && matches!(self.lists, WorkerLists::Pair(_) | WorkerLists::Encode(_))
    }
}

/// One request as the engine makes it of what its caller stated: the one
/// place a fact of a request is made.
pub(super) struct Made<'r, 'a> {
    /// The engine that decides: the size of a cache block of the request's
    /// model is read of its record of cache events. The document is the
    /// one each making is handed.
    pub(super) engine: &'a InProcess,
    pub(super) request: &'r PlaceRequest<'a>,
    /// The targets earlier decisions of this call named that could not be
    /// sent to.
    pub(super) missed: &'r [WorkerId],
    /// The lists the decision is lent ([`RequestMaker::workers`]): the
    /// request's own. A pair's ask, or one of an encode request that
    /// carries items, behind the decision that had the rule asked of the
    /// pair's two lists is lent the same lists with each side's places
    /// beside it (`place::pair`).
    pub(super) lists: &'r WorkerLists<'r>,
    /// The places of a single request's list the decision is held to: the
    /// workers the route's rule takes, for an ask behind the decision that
    /// had the rule asked of the list ([`Call::admitted`]). `None`: every
    /// place, and for a pair, whose places stand beside its lent lists.
    pub(super) places: Option<&'r [u32]>,
    /// The time left for the decision, in microseconds: the request's own
    /// budget, or what its call has left of it ([`Call::left`]).
    pub(super) budget_micros: u32,
    /// The prompt's block hashes, made at the request's first making and
    /// borrowed by each one after it.
    pub(super) hashes: OnceCell<PromptHashes>,
    /// The call the request is asked in ([`Call`]): the hashes are then
    /// made in its cell, once a call, and `hashes` stays empty. Where the
    /// call says so, the request is made with no routing value and no
    /// client's value, whatever its sources hold ([`Call::unkeyed`]).
    pub(super) of_call: Option<&'r Call>,
    /// The headers of the document's view that the request carries, made
    /// at the request's first making like the hashes.
    pub(super) headers: OnceCell<Vec<Header<'a>>>,
    /// The request's routing value in the clear, as the last making
    /// derived it ([`Made::key`]).
    pub(super) key: Cell<Option<&'a str>>,
    /// Where the request's key stands on the ring its caller lends, taken
    /// at the request's first making and borrowed by each one after it:
    /// none where the request carries no key or is lent no ring.
    pub(super) place: OnceCell<Option<u64>>,
    /// Where the request's heads stand on that ring under the prefix form
    /// of the ring stage: asked of the lender at the request's first
    /// making, like the key's place.
    pub(super) heads: OnceCell<Places>,
    /// Where the request's heads stand on that ring for a pair's decode
    /// side, where the two sides cut a prompt by two rules
    /// ([`Made::stated_apart`]) or the two sides' shadows do (`shadow_heads`):
    /// `heads` holds the prefill side's then.
    pub(super) decode_heads: OnceCell<Places>,
    /// The first shadow profile whose heads' places this maker was asked
    /// for ([`RequestMaker::shadow_ring`]): its rule cut `heads`, and a
    /// shadow of another rule is stated its own in `decode_heads`.
    pub(super) shadow_heads: OnceCell<u16>,
    /// The panic of the route's rule where it was asked inside a decision
    /// ([`RequestMaker::sends_to`]): kept for the call, which raises it
    /// again outside the decision (`place::raise`).
    pub(super) panicked: OnceCell<Box<dyn Any + Send>>,
    /// What a decision was answered on the targets it asked
    /// ([`RequestMaker::sends_to`]): the call's pick takes these answers,
    /// and reads none of those targets again.
    pub(super) sent: Cell<Sent>,
    /// The same on the decodes of a pair's targets, by their places in the
    /// decode list: `sent` holds the prefills' then.
    pub(super) decodes: Cell<Sent>,
    /// What a decision was answered on the encoders it placed
    /// ([`RequestMaker::sends_to_encoder`]): the call takes these answers,
    /// and reads none of those encoders again.
    pub(super) encoders: RefCell<Answered>,
    /// Whether the request is a pair's decode decided again: a target's
    /// prefill was sent already, and its decode alone is asked.
    pub(super) decode_alone: bool,
    /// Whether a stage of the document the request's decisions run under
    /// reads a request's size: a plain pick's lease sizes the text by it
    /// ([`RequestMaker::prompt_chars`]). A call says it ([`Made::asked`]).
    /// False outside one: the decode entry leases no plain pick.
    pub(super) sized: bool,
    /// The number a pair's request names its decode by, where the decision
    /// counts it behind the prefill it picks ([`Call::named_decode`]): the
    /// request made states it ([`Constraints::target_worker`]). `None` for
    /// every other request.
    pub(super) named_decode: Option<u32>,
}

/// What a call's maker answered on a decision's targets where the decision
/// asked which of them a prompt is stored for, in a lent tree or in
/// the record of sent prefixes: each place asked, and whether the call
/// sends to it. A pick's targets are asked best first up to the first one
/// the call sends to, so a pick's targets fit.
#[derive(Clone, Copy)]
pub(super) struct Sent {
    /// The places asked, in the order asked.
    places: [u32; MAX_TARGETS],
    /// How many were asked and kept.
    asked: u8,
    /// A bit a place kept, in that order: set where the call sends to it.
    sendable: u8,
}

impl Sent {
    /// No target asked.
    const NONE: Self = Self {
        places: [0; MAX_TARGETS],
        asked: 0,
        sendable: 0,
    };

    /// The answer on the target at `place`: whether the call sends to it.
    /// `None` where the lookup did not ask it.
    pub(super) fn answer(&self, place: u32) -> Option<bool> {
        let asked = &self.places[..usize::from(self.asked)];
        let at = asked.iter().position(|&asked| asked == place)?;
        Some(self.sendable & (1 << at) != 0)
    }

    /// Keeps the answer on the target at `place`. A target past the room
    /// is not kept, and the pick reads it itself.
    pub(super) fn keep(&mut self, place: u32, sendable: bool) {
        let at = usize::from(self.asked);
        if let Some(slot) = self.places.get_mut(at) {
            *slot = place;
            self.sendable |= u8::from(sendable) << at;
            self.asked += 1;
        }
    }
}

/// What a call's maker answered on the encoders an encode request's
/// decision placed, by their places in the encoders' list: each place
/// asked, and whether the call sends to it. A request's items have no
/// ceiling, so the list grows with the encoders asked of.
#[derive(Default)]
pub(super) struct Answered(pub(super) Vec<(u32, bool)>);

impl Answered {
    /// The answer on the encoder at `place`: whether the call sends to it.
    /// `None` where the decision did not ask of it.
    pub(super) fn answer(&self, place: u32) -> Option<bool> {
        let kept = self.0.iter().find(|(asked, _)| *asked == place);
        kept.map(|&(_, sendable)| sendable)
    }
}

impl<'r, 'a> Made<'r, 'a> {
    /// `request` for a decision of `engine` that leaves `missed` out.
    pub(super) const fn new(
        engine: &'a InProcess,
        request: &'r PlaceRequest<'a>,
        missed: &'r [WorkerId],
    ) -> Self {
        Self {
            engine,
            request,
            missed,
            lists: &request.lists,
            places: None,
            budget_micros: request.budget_micros,
            hashes: OnceCell::new(),
            of_call: None,
            headers: OnceCell::new(),
            key: Cell::new(None),
            place: OnceCell::new(),
            heads: OnceCell::new(),
            decode_heads: OnceCell::new(),
            shadow_heads: OnceCell::new(),
            panicked: OnceCell::new(),
            sent: Cell::new(Sent::NONE),
            decodes: Cell::new(Sent::NONE),
            encoders: RefCell::new(Answered(Vec::new())),
            decode_alone: false,
            sized: false,
            named_decode: None,
        }
    }

    /// `request` for one ask of `call`: its decision leaves out the targets
    /// the call missed so far, is held to the places the call keeps
    /// ([`Call::admitted`]), and borrows the prompt's hashes where an
    /// earlier ask of the call made them. The headers of the document's
    /// view and the key's place on a lent ring, or the heads' places, are
    /// made again by each ask that makes the request: a later ask is the
    /// call's cold path. `engine` is the engine as the call's decisions
    /// read it: whether its document sizes a request is kept of it, so
    /// the maker reads no document the engine holds in force later.
    pub(super) fn asked(
        engine: Loaded<'a, '_>,
        request: &'r PlaceRequest<'a>,
        call: &'r Call,
    ) -> Self {
        Self {
            of_call: Some(call),
            places: call.admitted.as_deref(),
            sized: engine.document.spec.needs().prompt_size,
            ..Self::new(engine.object, request, call.missed.as_slice())
        }
    }

    /// The request's routing value in the clear, where a making derived
    /// one that leaves so: what the request's placement names. None ahead
    /// of a making, and for a request a place answers, which is not made.
    pub(super) fn key(&self) -> Option<&'a str> {
        self.key.get()
    }

    /// Whether the request's making leaves the prompt's block hashes to the
    /// lookup that names the request's holders ([`Self::hashed_whole`]): a
    /// call's request of token ids that names its entry in a token tree,
    /// over one list, under a document none of whose stages reads the
    /// hashes itself ([`AppliedSpec::hashes_named_alone`]). That lookup
    /// reads the list by an index of cache events and by the record of sent
    /// prefixes, and reads none of it for a request it decides inside a
    /// lent tree's match. Every other request is hashed at its making: a
    /// pair's sides read the list in their columns, and a prompt of more
    /// token ids than a request's size counts is sized by its list. The
    /// asks of a call answer alike: its decisions are one document's.
    pub(super) fn leaves_hashes(&self, spec: &AppliedSpec) -> bool {
        let prompt = self.request.prompt;
        self.of_call.is_some()
            && prompt
                .tokens
                .is_some_and(|tokens| u32::try_from(tokens.len()).is_ok())
            && prompt.entry.is_some()
            && matches!(self.request.lists, WorkerLists::Single(_))
            && spec.hashes_named_alone()
    }

    /// The prompt's block hashes as deep as `spec` reads them, for the
    /// lookup of a request whose making left them out
    /// ([`Self::leaves_hashes`]), on a road that reads the list: hashed at
    /// the call's first such ask, by the block size its making read. `None`
    /// for a request hashed at its making: its facts hold the list. `None`
    /// too under a document that reads no block: there is no list.
    pub(super) fn hashed_whole(&self, spec: &AppliedSpec) -> Option<&PromptHashes> {
        let call = self.of_call.filter(|_| self.leaves_hashes(spec))?;
        let (tokens, made) = (self.request.prompt.tokens?, call.hashes.get()?);
        let seeds = self.request.prompt.seeds;
        let whole = call.whole.get_or_init(|| made.whole(spec, tokens, seeds));
        whole.as_ref()
    }

    /// The request's routing value and its client's value, by the sources
    /// its binding names in `spec`, the decision's document, with where the
    /// value a ring places it by stands on a ring, as `place` answers for
    /// that value's text and its source, `None` for the client's value
    /// ([`KeySources::derive_keys`](crate::document::KeySources::derive_keys)).
    /// The routing value is the first source that holds one; none where
    /// the binding names no source or none holds a value. The client's
    /// value is read only where the request has no routing value: it
    /// places where no session does. A request its call makes with no key
    /// has neither ([`Call::unkeyed`]).
    fn keys(
        &self,
        spec: &AppliedSpec,
        place: impl FnOnce(&str, Option<KeySource<'a>>) -> Option<u64>,
    ) -> DerivedKeys<'a> {
        let request = self.request;
        let sources = spec.key_sources_for(request.model, request.phase());
        let unkeyed = || self.of_call.is_some_and(|call| call.unkeyed);
        let Some(sources) = sources.filter(|_| !unkeyed()) else {
            return DerivedKeys::default();
        };
        let headers = request.headers;
        let header = move |name: &str| headers.header(name);
        sources.derive_keys_by(header, request.request_id, place)
    }

    /// The request's keys, the ring its decision is lent and where the
    /// request stands on it, by the word of the profile that decides the
    /// request. A stage that places a routing value by its own text
    /// ([`Engine::states_key_place`](crate::engine::Engine::states_key_place))
    /// is stated that text's place, read with the keys. The prefix form of
    /// the ring
    /// ([`Engine::states_head_places`](crate::engine::Engine::states_head_places))
    /// is stated where the lender puts each head of the request
    /// ([`LentRing::head_position`]): the routing value where a header
    /// handed the request one, else the prompt's heads, the deepest first.
    /// A value the request's own id handed names the session alone
    /// ([`RequestFacts::session_key_from_rid`]). A lender
    /// that places no head lends no ring, a request that has no head is
    /// lent none, and a request under any other profile lends none: its
    /// stage places by its own. The lender is asked at the first making
    /// of one ask of the call, and a later making of that ask borrows
    /// what it answered. A call that asks again makes its request anew,
    /// and asks the lender again.
    ///
    /// A pair's two profiles answer together, and their one list is both
    /// sides'. Where they read the ring by two words no one list is, and
    /// each side is stated its own ([`Self::stated_apart`]). The keys of
    /// such a pair are derived here, once, as every other request's are,
    /// and the lender is asked with them for each word a side reads
    /// ([`Self::words_apart`]).
    pub(super) fn stated<'w>(&'w self, spec: &'w AppliedSpec) -> Stated<'a, 'w> {
        let request = self.request;
        let (model, phase) = (request.model, request.phase());
        let Some(ring) = request.ring else {
            return (self.keys(spec, |_, _| None), None, &[], None);
        };
        // The two questions are asked of one lookup of the binding.
        let engine = spec.engine();
        let profiles = engine.ring_profiles(model, phase);
        if engine.key_place_of(profiles) {
            return self.stated_key::<false>(spec, ring);
        }
        let Some(rule) = engine.head_rule_of(profiles) else {
            let (Some(own), Some(decode)) = profiles else {
                return (self.keys(spec, |_, _| None), None, &[], None);
            };
            // A pair whose two sides read the ring by two words: the
            // lender is asked for each word a side reads, with the keys'
            // one derivation.
            let words = self.words_apart(spec, (own, decode));
            let (first_key, first_heads) = (words.first_key, words.first_heads);
            let key_at = Cell::new(None);
            let keys = self.keys(spec, |text, source| {
                if first_heads && source.is_some_and(leads_a_prompt) {
                    key_at.set(Some(ring.head_position(Head::Text(text))));
                }
                first_key.then(|| ring.position(text))
            });
            let (places, decode_places) =
                self.stated_apart(ring, &words, keys.ring_position, key_at.take());
            // The ring is lent where either side has a place.
            let lent = !(places.is_empty() && decode_places.is_empty());
            return (
                keys,
                lent.then_some(ring),
                places,
                lent.then_some(decode_places),
            );
        };
        self.stated_heads::<false>(spec, ring, rule, &self.heads)
    }

    /// [`Self::stated`] by a stage that places a routing value by its own
    /// text: that text's place on `ring`. `SHADOW` parts the copy a
    /// shadow's statement compiles ([`RequestMaker::shadow_ring`]) from
    /// the live one: a closure called from one place is built into its
    /// caller, so the live statement stays whole inside [`Self::stated`].
    #[inline(always)]
    pub(super) fn stated_key<'w, const SHADOW: bool>(
        &'w self,
        spec: &AppliedSpec,
        ring: &'a dyn LentRing,
    ) -> Stated<'a, 'w> {
        let first = self.place.get().is_none();
        let keys = self.keys(spec, |text, _| first.then(|| ring.position(text)));
        let place = self.place.get_or_init(|| keys.ring_position);
        (keys, Some(ring), place.as_slice(), None)
    }

    /// [`Self::stated`] by the prefix form of the ring, which cuts the
    /// request by `rule`: the places of its heads on `ring`, kept in
    /// `heads`. `SHADOW` as in [`Self::stated_key`].
    #[inline(always)]
    pub(super) fn stated_heads<'w, const SHADOW: bool>(
        &'w self,
        spec: &AppliedSpec,
        ring: &'a dyn LentRing,
        rule: HeadRule<'_>,
        heads: &'w OnceCell<Places>,
    ) -> Stated<'a, 'w> {
        let request = self.request;
        // A routing value a header handed leads the prompt: its text is the
        // one head. A client's value places nothing under this form, and
        // neither does a value the request's own id handed: that request
        // is placed by its prompt's heads.
        let key_at = Cell::new(None);
        let first = heads.get().is_none();
        let keys = self.keys(spec, |text, source| {
            if first && source.is_some_and(leads_a_prompt) {
                key_at.set(Some(ring.head_position(Head::Text(text))));
            }
            None
        });
        let heads = heads.get_or_init(|| {
            let prompt = (request.prompt.tokens, request.prompt.text);
            Places::of_heads::<SHADOW>(ring, rule, prompt, key_at.take())
        });
        // The ring is lent with a place: a request that has no head, and
        // one whose lender places none, state nothing and lend nothing.
        let places = heads.as_slice();
        let ring = (!places.is_empty()).then_some(ring);
        (keys, ring, places, None)
    }

    /// The word each side of a pair reads the lent ring by, where the two
    /// sides read it by two words: `own` and `decode` are the pair's two
    /// profiles
    /// ([`Engine::ring_profiles`](crate::engine::Engine::ring_profiles)).
    /// A side's word is the one a single request of its own profile is
    /// stated by. A side that places a routing value by its own text
    /// reads that text's place. A side of the prefix form reads the
    /// places of the request's heads by its own rule. A side that reads
    /// no ring, or reads it by two words itself, reads nothing.
    ///
    /// With the words, whether this making is the first to ask the lender
    /// where the key's own text stands, and where the key stands as a
    /// head: the lender's answers are kept at the first making, as a
    /// single request's are.
    #[inline(never)]
    fn words_apart<'e>(&self, spec: &'e AppliedSpec, (own, decode): (usize, usize)) -> Words<'e> {
        let engine = spec.engine();
        let word = |at: usize| {
            let alone = (Some(at), None);
            if engine.key_place_of(alone) {
                Word::Key
            } else {
                engine.head_rule_of(alone).map_or(Word::Unread, Word::Heads)
            }
        };
        let (prefill, decode) = (word(own), word(decode));
        let by_key = matches!(prefill, Word::Key) || matches!(decode, Word::Key);
        let by_heads = matches!(prefill, Word::Heads(_)) || matches!(decode, Word::Heads(_));
        Words {
            prefill,
            decode,
            first_key: by_key && self.place.get().is_none(),
            first_heads: by_heads && self.heads.get().is_none(),
        }
    }

    /// [`Self::stated`] for a pair whose two sides read the lent ring by
    /// two words (`words`), so that no one list is both sides': each
    /// side's own places, the prefill side's first. Each side is stated
    /// what a single request of its own profile is stated. `key_place` is
    /// where the lender put the routing value's own text, and `key_at`
    /// its answer for that value as a head, where this making asked.
    ///
    /// A routing value that leads the prompt is the one head under every
    /// rule, so two sides of the prefix form then stand at one place. A
    /// side that has no place places by its stage's own ring: its list
    /// is empty ([`RequestFacts::decode_ring_positions`]). So a lender
    /// that places no head leaves the side of the prefix form alone to
    /// its own ring, and under two rules is asked about no head behind
    /// the one it declined.
    ///
    /// The keys are derived in [`Self::stated`] and not here: a statement
    /// that an out-of-line function returns whole passes every request's
    /// statement through memory, a single request's too.
    #[inline(never)]
    fn stated_apart<'w>(
        &'w self,
        ring: &'a dyn LentRing,
        words: &Words<'_>,
        key_place: Option<u64>,
        key_at: Option<Option<u64>>,
    ) -> (&'w [u64], &'w [u64]) {
        let request = self.request;
        let prompt = (request.prompt.tokens, request.prompt.text);
        let side = |word: Word<'_>, heads: &'w OnceCell<Places>| -> &'w [u64] {
            match word {
                Word::Unread => &[],
                Word::Key => self.place.get_or_init(|| key_place).as_slice(),
                Word::Heads(rule) => heads
                    .get_or_init(|| Places::of_heads::<true>(ring, rule, prompt, key_at))
                    .as_slice(),
            }
        };
        let (prefill, decode) = (words.prefill, words.decode);
        let places = side(prefill, &self.heads);
        // Two rules cut two lists: the decode side's stands in its own
        // cell where the prefill side's fills the first. A lender that
        // placed no head for the first is asked about none for the second.
        let second = matches!((prefill, decode), (Word::Heads(_), Word::Heads(_)));
        let declined = second && self.heads.get().is_some_and(|heads| heads.unplaced);
        let decode_places: &[u64] = match (declined, second) {
            (true, _) => &[],
            (false, true) => side(decode, &self.decode_heads),
            (false, false) => side(decode, &self.heads),
        };
        (places, decode_places)
    }
}

/// What a request states of a lent ring ([`Made::stated`]): its keys, the
/// ring its decision is lent, its places on it, and a pair's decode side's
/// own places where the two sides are stated apart
/// ([`RequestFacts::decode_ring_positions`]). A statement of one list
/// ([`Made::stated_key`], [`Made::stated_heads`]) holds `None` there.
type Stated<'a, 'w> = (
    DerivedKeys<'a>,
    Option<&'a dyn LentRing>,
    &'w [u64],
    Option<&'w [u64]>,
);

/// The words a pair's two sides read a lent ring by
/// ([`Made::words_apart`]), and what the lender is still to be asked.
#[derive(Clone, Copy)]
struct Words<'e> {
    /// The prefill side's word.
    prefill: Word<'e>,
    /// The decode side's word.
    decode: Word<'e>,
    /// A side reads the key's own place, and no making took it yet.
    first_key: bool,
    /// A side reads the heads' places, and no making took them yet.
    first_heads: bool,
}

/// What one profile reads of a ring its request is lent
/// ([`Made::words_apart`]).
#[derive(Clone, Copy)]
enum Word<'e> {
    /// No place of the request's: the profile reads no ring, or reads it
    /// by two words itself.
    Unread,
    /// The place of the routing value's own text.
    Key,
    /// The places of the request's heads, cut by this rule.
    Heads(HeadRule<'e>),
}

/// Whether a routing value from `source` leads a request's prompt under a
/// ring's prefix form: every source but the request's own id, whose value
/// names the session alone. One test for the place the one call states on
/// a lent ring ([`Made::stated`]) and for the word its request carries
/// ([`RequestFacts::session_key_from_rid`]), so the stated place and the
/// stage's own key cannot part where a source gains a kind.
pub(super) fn leads_a_prompt(source: KeySource<'_>) -> bool {
    source != KeySource::Rid
}

/// How many of a request's places on a lent ring its maker keeps in the
/// list itself ([`Places`]).
const HEADS_INLINE: usize = 8;

/// Where a request's heads stand on the ring its caller lends, as the
/// lender answered for each ([`Made::stated`]). As many as [`HEADS_INLINE`]
/// stand in the list itself. A prompt that reaches more boundaries has its
/// list on the heap, made once an ask of the call: the gateway's own
/// policy lists every boundary a prompt reaches
/// (`model_gateway/src/policies/prefix_hash.rs`, lines 172 to 180 at
/// 882be6f4f).
#[derive(Default)]
pub(super) struct Places {
    inline: [u64; HEADS_INLINE],
    /// How many places `inline` holds, while `spilled` holds none.
    len: usize,
    spilled: Vec<u64>,
    /// The lender places no head: this list states no place, and a
    /// request with no other place on the ring is lent none.
    pub(super) unplaced: bool,
}

impl Places {
    /// Where a request's heads stand on `ring`, the deepest first: the one
    /// place of `key_at`, the lender's answer for a routing value that
    /// leads the prompt, where the request has one, else its answers for
    /// the heads `rule` cuts of `prompt`, its token ids and its text. A
    /// lender that places no head is asked nothing more, and the list is
    /// then none. `APART` parts the copy a pair's sides stated apart
    /// compile ([`Made::stated_apart`]), and a shadow's own statement with
    /// them ([`RequestMaker::shadow_ring`], by [`Made::stated_heads`]), from
    /// the live one list's: a closure called from one place is built into
    /// its caller, so that statement stays whole inside [`Made::stated`].
    #[inline(always)]
    fn of_heads<const APART: bool>(
        ring: &dyn LentRing,
        rule: HeadRule<'_>,
        (tokens, text): (Option<&[u32]>, Option<&str>),
        key_at: Option<Option<u64>>,
    ) -> Self {
        let mut places = Self::default();
        match key_at {
            Some(answer) => places.state(answer),
            None => {
                rule.heads(tokens, text, |head| {
                    // A lender that places no head is asked nothing more.
                    if !places.unplaced {
                        places.state(ring.head_position(head));
                    }
                });
                places.turn();
            }
        }
        places
    }

    /// The lender's answer for one head, behind those stated so far.
    fn state(&mut self, answer: Option<u64>) {
        let Some(place) = answer else {
            self.unplaced = true;
            return;
        };
        if self.spilled.is_empty() && self.len < HEADS_INLINE {
            self.inline[self.len] = place;
            self.len += 1;
            return;
        }
        if self.spilled.is_empty() {
            self.spilled.extend_from_slice(&self.inline);
        }
        self.spilled.push(place);
    }

    /// The list turned round: its heads were stated the shallowest first.
    fn turn(&mut self) {
        if self.spilled.is_empty() {
            self.inline[..self.len].reverse();
        } else {
            self.spilled.reverse();
        }
    }

    /// The places, and none where the lender places no head.
    fn as_slice(&self) -> &[u64] {
        if self.unplaced {
            &[]
        } else if self.spilled.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.spilled
        }
    }
}

/// Whether `worker` can be sent to now: ready, and let through by its
/// circuit breaker, on one read of its state. What a caller alone knows of
/// a worker at the instant of a send. The overload veto and the liveness
/// veto are the decision's own, and a pick that lifted one stands.
pub(super) fn ready(worker: &dyn Worker) -> bool {
    ready_by(worker.routing_state())
}

/// [`ready`] by a state its caller read: one read of a worker says its
/// vetoes too.
pub(super) const fn ready_by(state: RoutingState) -> bool {
    state.healthy && state.can_execute
}

/// What one ask of a call hands on to the next.
#[derive(Default)]
pub(super) struct Call {
    /// The workers the call's asks named that could not be sent to, each
    /// once, by the id the list's records name it by: a single request's
    /// targets, and each side of a pair's by its own. The next ask's
    /// decision excludes them ([`Constraints::excluded`]): the profile's
    /// `not-tried` filter passes them over while another worker remains.
    pub(super) missed: Vec<WorkerId>,
    /// What the call's first making states of the prompt's blocks: their
    /// hashes, or their count alone where the making left the list to the
    /// lookup (`whole`). A prompt is hashed once a call, however many asks
    /// the call takes.
    pub(super) hashes: OnceCell<PromptHashes>,
    /// The side of a pair that the call's last walk with no target could
    /// not send to, written by that walk (`place::walk_naming_side`): the
    /// decode side where a target of the walk lost its decode alone, its
    /// prefill being one that could be sent to, and the prefill side
    /// otherwise. A pair that names its worker and ends with no worker
    /// for its asks is shed on it (`place::target`).
    ///
    /// The order is that of the gateway's own pair selection, which asks
    /// the prefill first. Its hash policy reads the health of the worker
    /// at a named place once more at its pick, and one that is not
    /// healthy and eligible then is that policy's miss: the pair is
    /// declined for the prefill's role, and no decode is picked
    /// (`target_among` in
    /// `model_gateway/src/policies/consistent_hashing.rs`, and the prefill
    /// policy's decline in `pair_pass`,
    /// `model_gateway/src/routers/common/placement.rs`). Behind the
    /// prefill's pick that selection reads the pick's partners a second
    /// time: nobody open there is the decode's role, and so is a number
    /// that reaches none of them (`pair_pass`: its read of the pick's
    /// open partners, and the decode policy's decline behind it).
    ///
    /// `None` ahead of such a walk, and for the call of a single request
    /// and of an encode request that carries items: their walks write
    /// none.
    pub(super) unsent: Option<PairSide>,
    /// The prompt's whole list of block hashes, where the call's making
    /// left the list to the lookup that reads it ([`Made::hashed_whole`]):
    /// made where that lookup first asks, once a call, and `None` under a
    /// document that reads no block.
    pub(super) whole: OnceCell<Option<PromptHashes>>,
    /// The asks of the call that took no target.
    pub(super) spent: usize,
    /// The places of the list whose worker the route's rule takes
    /// ([`PlaceRequest::admits`]), ascending, each once, from the first
    /// decision none of whose targets could be sent to
    /// (`place::hold_to_rule`). The call's later asks are held to them
    /// ([`Constraints::places`]). `None` ahead of that decision, and for a
    /// request with no rule. For a request that names its worker they are
    /// that worker's one place, from ahead of the call's first ask
    /// (`place::target`), and the rule is not asked of the list again. A single
    /// request's alone: the places of a pair's two lists stand beside its
    /// later asks (`place::pair`).
    pub(super) admitted: Option<Vec<u32>>,
    /// Whether the call's requests are made with no routing value and no
    /// client's value: a request that names its worker over a list none of
    /// whose workers can be sent to (`place::target`). The worker a request
    /// names outranks its routing value, so that value keeps no worker
    /// over a veto.
    pub(super) unkeyed: bool,
    /// Whether the call's pair is held to the two workers its request
    /// names (`place::target::pair`). Such a call makes one decision: it books
    /// the decode with its prefill under every document, and no ask
    /// follows it (`place::pair`). The worker a request names is that
    /// decision's, so no later one moves it.
    pub(super) named_pair: bool,
    /// The number the call's pair names its decode by, where the decode's
    /// side alone reads the request's target source (`place::target`): the
    /// prefill is each decision's own pick, so each ask of the call states
    /// the number beside the decodes' places, and its decision counts it
    /// behind the prefill it picks ([`Constraints::target_worker`]). A text
    /// that is no number is stated as one that reaches no decode. `None`
    /// for every other call. Each of its decisions books the decode with
    /// its prefill under every document, as a pair held on both sides
    /// does: a decode asked again would count no number.
    pub(super) named_decode: Option<u32>,
    /// The instant of the call's first ask, for a request that states a
    /// budget: the budget is counted from it ([`Call::left`]). `None` for
    /// a request that states no budget: no clock is read for its budget.
    pub(super) began: Option<Instant>,
}
