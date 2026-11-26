//! One call for one request ([`InProcess::place`]): what a caller states
//! of the request, the request's facts made of that in one place, and the
//! worker the request goes to.
//!
//! A caller fills a [`PlaceRequest`] from what it already holds: the model,
//! its worker lists, the prompt, the headers. It names no policy, derives
//! nothing and looks inside no decision. The engine makes the request of it
//! ([`Made`]), under the document that decides:
//!
//! - the routing value and the client's value come from the sources the
//!   document's binding names, read in the caller's headers and in the
//!   request's own id ([`AppliedSpec::key_sources_for`]), with where the
//!   value stands on the hash ring the caller lends
//!   ([`PlaceRequest::ring`]);
//! - the seed is one no other request of the thread was given
//!   ([`fresh_seed`]);
//! - the prompt's blocks are hashed under its cache partition, at the first
//!   making, or where the lookup of the request's holders reads the list
//!   when no stage reads it itself ([`Made::hashed_whole`]), and never for
//!   a request a place answers ([`RequestMaker::plain`]): such a request is
//!   not made at all.
//!
//! One thing of the request the engine does not make: the prompt's entry in
//! a tree its caller lends. The caller names it ([`Prompt::entry`]),
//! by its own rule for a cache partition, and a decision that reads the
//! tree asks it. That decision stores the prompt in the tree for the first
//! of its targets this call can send to ([`RequestMaker::sends_to`]), and
//! for none where it can send to none.
//!
//! The engine then decides, and takes the first target of the decision that
//! can be sent to: ready ([`ready`]), and admitted by its route's own rule
//! ([`PlaceRequest::admits`]). Where no target of a decision can be sent
//! to, the engine decides again with the workers it could not send to
//! excluded, three decisions a call ([`ASKS`]). Behind the first decision
//! none of whose targets can be sent to, the later decisions of a request
//! that states a rule are held to the workers the rule takes
//! ([`hold_to_rule`]): those of a single request's list, and those of each
//! of a pair's two, whatever kept its targets from being sent to, and
//! where a pair's first decision shed the request as saturated. The
//! caller is answered a [`Placement`], which it keeps with the request, or
//! the reason there is none ([`Unplaced`]).
//!
//! A single request that names its worker, under a document whose binding
//! says which header names it, is held to that one worker and goes to no
//! other ([`target`]). A pair that names its worker is held so on each
//! side whose profile is the hash ring keyed by no prefix. Where its
//! prefill's is another, each decision counts the decode's number behind
//! the prefill it picks.
//!
//! A request with a pair's two lists is placed a target at a time over both
//! its workers ([`pair`]). Its later decisions are held to the workers the
//! rule takes on each list, as a single request's are to those of its one.
//!
//! A request with an encode request's three lists is placed the same way:
//! a decision names its pair and each item's encoder, and the request is
//! placed where its pair's two workers and every encoder can be sent to
//! ([`encode`]), three decisions a call at most. Its later decisions are
//! held to the workers the rule takes on each of its pair's two lists, as
//! a pair's are, with the items it carries or with none.
//!
//! A pair's decode decided again behind its prefill is placed by the same
//! read of a worker, over the decodes that decision names
//! ([`decode_again`]). Its caller states the request again, and the
//! request is made of that anew: its routing value derived and its prompt's
//! blocks hashed a second time.

#![expect(
    clippy::exhaustive_structs,
    reason = "a caller still builds or matches these types whole: each mark comes with its call"
)]

mod decode_again;
mod encode;
mod pair;
pub(crate) mod target;

use std::{
    any::Any,
    cell::{Cell, OnceCell, RefCell},
    fmt,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    sync::Arc,
    time::Instant,
};

use smg_worker::{identity::basic::Worker, runtime::RoutingState};

use super::{
    encode::{splits, EncodeLists},
    InProcess, Loaded,
};
use crate::{
    document::{
        spec::{AppliedSpec, DecodePick},
        DerivedKeys, KeySource,
    },
    engine::{
        decision::{
            Decision, PairSide, Pick, ShedReason, Target, ADMITS, BUDGET, ENCODE_ITEMS, MAX_TARGETS,
        },
        decision_engine::{
            AlonePick, DecideOutcome, DecisionRefused, Plain, Request, RequestMaker, TextKey,
            TokenKey, WorkerLists,
        },
        demands::Demands,
        framework::EncodeItems,
        list_cell::LentList,
        prompt_hashes::{
            prompt_bytes_of, prompt_chars_of, prompt_tokens_of, HashSeeds, PromptHashes,
        },
        records::worker_id,
        view::{
            Constraints, Header, Identity, MediaCounts, Phase, RequestFacts, WireModes, WorkerId,
        },
    },
    fresh_seed,
    policies::{
        consistent_hashing::{Head, LentRing},
        prefix_hash::HeadRule,
    },
    store::GatewayId,
    tracking::{lease::ReleaseReason, DecisionGuard, Placement},
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
    const fn phase(&self) -> Phase {
        match self.lists {
            WorkerLists::Pair(_) | WorkerLists::Encode(_) => Phase::Prefill,
            WorkerLists::None | WorkerLists::Single(_) => Phase::Single,
        }
    }

    /// Whether the request asks a pair's decode again
    /// ([`Self::decode_again`]). The word is a pair's, over its two lists
    /// or an encode request's three: over one list, and over none, it asks
    /// nothing. The request made and its plain test both read this.
    const fn asks_decode_again(&self) -> bool {
        self.decode_again && matches!(self.lists, WorkerLists::Pair(_) | WorkerLists::Encode(_))
    }
}

/// One request as the engine makes it of what its caller stated: the one
/// place a fact of a request is made.
pub(super) struct Made<'r, 'a> {
    /// The engine that decides: the size of a cache block of the request's
    /// model is read of its record of cache events. The document is the
    /// one each making is handed.
    engine: &'a InProcess,
    request: &'r PlaceRequest<'a>,
    /// The targets earlier decisions of this call named that could not be
    /// sent to.
    missed: &'r [WorkerId],
    /// The lists the decision is lent ([`RequestMaker::workers`]): the
    /// request's own. A pair's ask, or one of an encode request that
    /// carries items, behind the decision that had the rule asked of the
    /// pair's two lists is lent the same lists with each side's places
    /// beside it ([`pair`]).
    lists: &'r WorkerLists<'r>,
    /// The places of a single request's list the decision is held to: the
    /// workers the route's rule takes, for an ask behind the decision that
    /// had the rule asked of the list ([`Call::admitted`]). `None`: every
    /// place, and for a pair, whose places stand beside its lent lists.
    places: Option<&'r [u32]>,
    /// The time left for the decision, in microseconds: the request's own
    /// budget, or what its call has left of it ([`Call::left`]).
    budget_micros: u32,
    /// The prompt's block hashes, made at the request's first making and
    /// borrowed by each one after it.
    hashes: OnceCell<PromptHashes>,
    /// The call the request is asked in ([`Call`]): the hashes are then
    /// made in its cell, once a call, and `hashes` stays empty. Where the
    /// call says so, the request is made with no routing value and no
    /// client's value, whatever its sources hold ([`Call::unkeyed`]).
    of_call: Option<&'r Call>,
    /// The headers of the document's view that the request carries, made
    /// at the request's first making like the hashes.
    headers: OnceCell<Vec<Header<'a>>>,
    /// The request's routing value in the clear, as the last making
    /// derived it ([`Made::key`]).
    key: Cell<Option<&'a str>>,
    /// Where the request's key stands on the ring its caller lends, taken
    /// at the request's first making and borrowed by each one after it:
    /// none where the request carries no key or is lent no ring.
    place: OnceCell<Option<u64>>,
    /// Where the request's heads stand on that ring under the prefix form
    /// of the ring stage: asked of the lender at the request's first
    /// making, like the key's place.
    heads: OnceCell<Places>,
    /// Where the request's heads stand on that ring for a pair's decode
    /// side, where the two sides cut a prompt by two rules
    /// ([`Made::stated_apart`]) or the two sides' shadows do (`shadow_heads`):
    /// `heads` holds the prefill side's then.
    decode_heads: OnceCell<Places>,
    /// The first shadow profile whose heads' places this maker was asked
    /// for ([`RequestMaker::shadow_ring`]): its rule cut `heads`, and a
    /// shadow of another rule is stated its own in `decode_heads`.
    shadow_heads: OnceCell<u16>,
    /// The panic of the route's rule where it was asked inside a decision
    /// ([`RequestMaker::sends_to`]): kept for the call, which raises it
    /// again outside the decision ([`raise`]).
    panicked: OnceCell<Box<dyn Any + Send>>,
    /// What a decision was answered on the targets it asked
    /// ([`RequestMaker::sends_to`]): the call's pick takes these answers,
    /// and reads none of those targets again.
    sent: Cell<Sent>,
    /// The same on the decodes of a pair's targets, by their places in the
    /// decode list: `sent` holds the prefills' then.
    decodes: Cell<Sent>,
    /// What a decision was answered on the encoders it placed
    /// ([`RequestMaker::sends_to_encoder`]): the call takes these answers,
    /// and reads none of those encoders again.
    encoders: RefCell<Answered>,
    /// Whether the request is a pair's decode decided again: a target's
    /// prefill was sent already, and its decode alone is asked.
    decode_alone: bool,
    /// Whether a stage of the document the request's decisions run under
    /// reads a request's size: a plain pick's lease sizes the text by it
    /// ([`RequestMaker::prompt_chars`]). A call says it ([`Made::asked`]).
    /// False outside one: the decode entry leases no plain pick.
    sized: bool,
    /// The number a pair's request names its decode by, where the decision
    /// counts it behind the prefill it picks ([`Call::named_decode`]): the
    /// request made states it ([`Constraints::target_worker`]). `None` for
    /// every other request.
    named_decode: Option<u32>,
}

/// What a call's maker answered on a decision's targets where the decision
/// asked which of them a prompt is stored for, in a lent tree or in
/// the record of sent prefixes: each place asked, and whether the call
/// sends to it. A pick's targets are asked best first up to the first one
/// the call sends to, so a pick's targets fit.
#[derive(Clone, Copy)]
struct Sent {
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
    fn answer(&self, place: u32) -> Option<bool> {
        let asked = &self.places[..usize::from(self.asked)];
        let at = asked.iter().position(|&asked| asked == place)?;
        Some(self.sendable & (1 << at) != 0)
    }

    /// Keeps the answer on the target at `place`. A target past the room
    /// is not kept, and the pick reads it itself.
    fn keep(&mut self, place: u32, sendable: bool) {
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
struct Answered(Vec<(u32, bool)>);

impl Answered {
    /// The answer on the encoder at `place`: whether the call sends to it.
    /// `None` where the decision did not ask of it.
    fn answer(&self, place: u32) -> Option<bool> {
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
    fn leaves_hashes(&self, spec: &AppliedSpec) -> bool {
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
    fn stated<'w>(&'w self, spec: &'w AppliedSpec) -> Stated<'a, 'w> {
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
    fn stated_key<'w, const SHADOW: bool>(
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
    fn stated_heads<'w, const SHADOW: bool>(
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
fn leads_a_prompt(source: KeySource<'_>) -> bool {
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
struct Places {
    inline: [u64; HEADS_INLINE],
    /// How many places `inline` holds, while `spilled` holds none.
    len: usize,
    spilled: Vec<u64>,
    /// The lender places no head: this list states no place, and a
    /// request with no other place on the ring is lent none.
    unplaced: bool,
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

impl RequestMaker for Made<'_, '_> {
    fn model(&self) -> &str {
        self.request.model
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        let request = self.request;
        // The request's place is taken at the first making: a later one
        // asks the ring nothing, and nothing is hashed again.
        let (keys, ring, places, decode_places) = self.stated(spec);
        let session_key = keys.session.map_or(Identity::NONE, |derived| derived.key);
        self.key.set(session_key.plain);
        // The headers the document's view shows a stage, each named by its
        // place in that list. A value that is not text is passed over.
        let headers = self.headers.get_or_init(|| {
            let read = request.headers;
            let stated = |(name, at): (&str, u8)| {
                let value = std::str::from_utf8(read.header(name)?).ok()?;
                Some(Header { name: at, value })
            };
            // A view lists 256 names at most: the range ends where they do.
            let listed = spec.view_headers().zip(0..=u8::MAX);
            listed.filter_map(stated).collect()
        });
        let mut facts = RequestFacts {
            model: request.model,
            // The partition's, with a prompt or with none: a plugin's view
            // states the seed of a request that carries no block too.
            namespace_seed: request.prompt.seeds.partition,
            phase: request.phase(),
            tenant: request.tenant.map_or(Identity::NONE, Identity::plain),
            priority: request.priority,
            streaming: request.streaming,
            max_output_tokens: request.max_output_tokens,
            // A key that leaves in the clear is the one the placement names,
            // and its caller counts the request under it on its worker: the
            // count a capped sessions rule reads off the list's records.
            session_counted: session_key.plain.is_some(),
            session_key,
            // The one place the word is set: the prefix form of a ring
            // reads it, and places such a request by its prompt.
            session_key_from_rid: keys
                .session
                .is_some_and(|derived| !leads_a_prompt(derived.source)),
            client_key: keys.client_key,
            ring,
            ring_positions: places,
            decode_ring_positions: decode_places,
            attempt: request.attempt,
            tried: request.tried,
            budget_micros: self.budget_micros,
            headers: headers.as_slice(),
            media: request.media,
            seed: fresh_seed(),
            ..RequestFacts::default()
        };
        let (tokens, text) = (request.prompt.tokens, request.prompt.text);
        if tokens.is_some() || text.is_some() {
            // Under the decision's own document, as the keys are.
            let cell = self.of_call.map_or(&self.hashes, |call| &call.hashes);
            let hashes = cell.get_or_init(|| {
                // By the id the list's workers registered under, as their
                // table finds its index: a request may name the model by
                // another.
                let listed = match &request.lists {
                    WorkerLists::None => None,
                    WorkerLists::Single(list) => list.workers.first(),
                    WorkerLists::Pair(lists) => lists.prefill.workers.first(),
                    WorkerLists::Encode(lists) => lists.pair.prefill.workers.first(),
                };
                let model = listed.map_or(request.model, |worker| worker.model_id());
                let size = self.engine.affinity.block_size(model);
                let seeds = request.prompt.seeds;
                // A list the lookup alone reads is hashed where it is read.
                match tokens.filter(|_| self.leaves_hashes(spec)) {
                    Some(tokens) => PromptHashes::unhashed(spec, size, tokens, seeds),
                    None => PromptHashes::under_document(spec, size, tokens, text, seeds),
                }
            });
            hashes.write_to(&mut facts);
        }
        // The document's word, like the keys: read from the generation
        // that decides. A request over one list, or over none, states no
        // later decode whatever its caller says: the word is a pair's, as
        // the plain test reads it. An encode request whose decode is held
        // to the ones that take images, by its demand or by the preference
        // of references beside pixels, states none either: a decode asked
        // again reads no image fact. Nor does one that leaves a part of
        // its hard demand to its encoders: a decode asked again takes no
        // items and would hold the decode to the whole. A pair held to the
        // two workers it names states none, and neither does one whose
        // decision counts the number its decode is named by: a decode asked
        // again would be picked by the pair's routing value. The holds are
        // read last, so a request that asks no decode again reads none of
        // them.
        let later = request.asks_decode_again()
            && spec.decode_pick() == DecodePick::AfterPrefill
            && !self.of_call.is_some_and(|call| call.named_pair)
            && self.named_decode.is_none()
            && !request.encode.is_some_and(|items| {
                items.holds_decode_to_images(request.media.holds_pixels())
                    || splits(&items, request.demands)
            });
        Request {
            id: request.request_id.map_or(&[], str::as_bytes),
            facts,
            constraints: Constraints {
                excluded: self.missed,
                places: self.places,
                decode_later: later,
                counted_on_decode: request.counted_on_decode,
                demands: request.demands,
                target_worker: self.named_decode,
                ..Constraints::default()
            },
        }
    }

    fn shadow_ring<'w>(
        &'w self,
        spec: &'w AppliedSpec,
        shadow: u16,
    ) -> Option<(&'w dyn LentRing, &'w [u64])> {
        let ring = self.request.ring?;
        // The shadow profile alone is asked what a deciding profile is.
        let (engine, profiles) = (spec.engine(), (Some(usize::from(shadow)), None));
        let (_, ring, places, _) = if engine.key_place_of(profiles) {
            self.stated_key::<true>(spec, ring)
        } else {
            let rule = engine.head_rule_of(profiles)?;
            // A pair's two sides' shadows may cut the request by two
            // rules: two sides, so two rules at most. The first to ask
            // fills `heads`; one of another rule fills the decode side's
            // list, which no live side fills where a shadow asks. A
            // routing value that leads the prompt is asked of the lender
            // once a rule.
            let first = usize::from(*self.shadow_heads.get_or_init(|| shadow));
            let heads = match engine.head_rule_of((Some(first), profiles.0)) {
                Some(_) => &self.heads,
                // A lender that placed no head under the first rule is
                // asked about none under the second.
                None if self.heads.get().is_some_and(|heads| heads.unplaced) => return None,
                None => &self.decode_heads,
            };
            self.stated_heads::<true>(spec, ring, rule, heads)
        };
        Some((ring?, places))
    }

    // A first attempt that names nothing tried or missed, is held to no
    // place of its list, demands nothing, carries no encode item and asks
    // no decode again ([`PlaceRequest::asks_decode_again`]) is plain. Its
    // size is the one the request made would state.
    fn plain(&self) -> Option<Plain> {
        let request = self.request;
        let first = request.attempt == 0 && request.tried.is_empty() && self.missed.is_empty();
        let first = first && self.places.is_none();
        let again = request.asks_decode_again();
        let asks_more = !request.demands.is_none() || request.encode.is_some() || again;
        (first && !asks_more).then(|| Plain {
            phase: request.phase(),
            wires: WireModes::NONE,
            prompt_tokens: prompt_tokens_of(request.prompt.tokens),
        })
    }

    // The id the request made states, read where its caller stated it: no
    // request is made for it.
    fn request_id<'w>(&'w self, _: &'w AppliedSpec) -> &'w [u8] {
        self.request.request_id.map_or(&[], str::as_bytes)
    }

    // The id its host states for a kept decision alone: the word it
    // stated, or else the value of the header it named, read here and
    // nowhere ahead of the call.
    fn explain_id(&self) -> &[u8] {
        let request = self.request;
        match (request.explain_id, request.explain_id_header) {
            (Some(id), _) => id.as_bytes(),
            (None, Some(name)) => request.headers.header(name).unwrap_or(&[]),
            (None, None) => &[],
        }
    }

    // Asked where a plain request's pick is leased, ahead of its making:
    // the hashing counts the text of a request that is made, so a text is
    // passed over once. The lease is its decision's document's, and so is
    // the flag: the engine may hold another document in force by now.
    fn prompt_chars(&self) -> u32 {
        let prompt = self.request.prompt;
        prompt_chars_of(prompt.tokens, prompt.text, self.sized)
    }

    // The text's bytes, for the same lease: its length is at hand.
    fn prompt_bytes(&self) -> u32 {
        let prompt = self.request.prompt;
        prompt_bytes_of(prompt.tokens, prompt.text)
    }

    // The lists the ask is lent: the request's own, but for an ask of a
    // pair or of an encode request that is held to its route's rule
    // ([`pair`]).
    fn workers(&self) -> WorkerLists<'_> {
        *self.lists
    }

    // The caller's own naming, handed on: nothing else reads the field.
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>> {
        self.request.prompt.entry?.token_key(page)
    }

    fn text_key(&self) -> Option<TextKey<'_>> {
        self.request.prompt.entry?.text_key()
    }

    // The walk's own read of a target, made ahead of the walk for the
    // write of where the prompt went, a lent tree's or the record's: each
    // worker of the target ready, and admitted by the route's rule, which
    // is asked of a worker that is ready and of no other. A single
    // request's target is its one worker, and a pair's is its prefill and
    // its decode, in a pair's two lists or an encode request's.
    //
    // A rule that panics here panics inside a decision, where the engine's
    // own catch would answer it as a decision that faulted. It is the
    // caller's rule, and its panic is the caller's: the panic is kept, no
    // later target is asked or stored, and the call raises it again behind
    // the decision ([`raise`]). The walk catches the rule's panic the same
    // way where it asks ([`walk`]).
    //
    // Each answer is kept ([`Sent`]), a pair's by its side: the call's pick
    // takes it, so a worker is read and its rule asked once a decision. A
    // pair's targets share workers, their prefill first of all: one that
    // an earlier target was asked of answers as it did.
    fn sends_to(&self, target: &Target) -> bool {
        match self.request.lists {
            WorkerLists::None => true,
            WorkerLists::Single(list) => self.asks(list, target.position, &self.sent),
            WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }) => {
                let side = |list: LentList<'_>, place: u32, kept: &Cell<Sent>| {
                    let held = kept.get().answer(place);
                    held.unwrap_or_else(|| self.asks(list, place, kept))
                };
                target.pair.is_some_and(|pair| {
                    (self.decode_alone || side(lists.prefill, target.position, &self.sent))
                        && side(lists.decode, pair.decode_position, &self.decodes)
                })
            }
        }
    }

    // The call's own read of an encoder a decision placed, made ahead of
    // that read for the write of where the prompt went: ready, on one read
    // of its state. The route's rule is asked of no encoder. The answer is
    // kept ([`Answered`]) and the call takes it, so an encoder is read once
    // a decision, however many the request's items are placed on.
    fn sends_to_encoder(&self, position: u32) -> bool {
        let WorkerLists::Encode(lists) = &self.request.lists else {
            return true;
        };
        let kept = self.encoders.borrow().answer(position);
        kept.unwrap_or_else(|| {
            let worker = lists.encoders.workers.get(position as usize);
            let sendable = worker.is_some_and(|worker| ready(worker.as_ref()));
            self.encoders.borrow_mut().0.push((position, sendable));
            sendable
        })
    }
}

impl Made<'_, '_> {
    /// Whether the call sends to the worker at `place` of `list`, one
    /// worker of a target ([`RequestMaker::sends_to`]). The answer is kept
    /// in `kept`.
    fn asks(&self, list: LentList<'_>, place: u32, kept: &Cell<Sent>) -> bool {
        let Some(worker) = list.workers.get(place as usize) else {
            return false;
        };
        let worker = worker.as_ref();
        if self.panicked.get().is_some() {
            return false;
        }
        let mut sent = kept.get();
        let sendable = ready(worker)
            && self.request.admits.is_none_or(|admits| {
                match catch_unwind(AssertUnwindSafe(|| admits(worker))) {
                    Ok(admitted) => admitted,
                    Err(panic) => {
                        drop(self.panicked.set(panic));
                        false
                    }
                }
            });
        sent.keep(place, sendable);
        kept.set(sent);
        sendable
    }
}

/// Why a request has no worker. The engine says why, and its caller answers
/// its client by it: a status and its words are the caller's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unplaced {
    /// The engine shed the request. A caller answers by `reason`: the
    /// model's workers are overloaded ([`ShedReason::Saturated`]), it has
    /// no worker for the request ([`ShedReason::NoEndpoints`]), a queue is
    /// at its limit ([`ShedReason::FlowLimit`]), the decision's time
    /// passed ([`ShedReason::Deadline`]), or the policy declined the
    /// request ([`ShedReason::PolicyDeclined`]).
    Shed {
        /// Why.
        reason: ShedReason,
        /// The stage that left no worker, what of the request the engine
        /// does not decide, or what of the request ended its call: a
        /// route's rule that takes no worker of the list (`admits`), a
        /// budget the call's earlier decisions used up (`budget`), a
        /// worker the request names that cannot take it
        /// (`target-worker`). Under [`ShedReason::PolicyDeclined`] the
        /// name `binding` says that the spec in force binds no plan of
        /// the request's kind for its model: a caller answers that no
        /// retry changes it. Two sheds of a pair over whole lists name a
        /// stage by the list judged, which may be another than the one
        /// that ended a pass
        /// ([`Shed::emptied_by`](crate::engine::decision::Shed::emptied_by)).
        /// A caller answers by two of these names, each a constant of
        /// the crate: [`BINDING`](crate::engine::decision::BINDING), and
        /// [`TARGET_WORKER`](crate::engine::decision::TARGET_WORKER),
        /// under which a shed is a decline whatever its reason. A
        /// caller's refusal line can name it.
        emptied_by: &'static str,
        /// The seconds after which a retry may find room: a caller's
        /// retry hint.
        retry_after_secs: u16,
        /// The side of a pair this refusal is for, by the one definition
        /// at the decision's field
        /// ([`Shed::side`](crate::engine::decision::Shed::side)): a
        /// pair's caller answers for that role. A decision's own side
        /// passes as it is. A pair shed under the worker it names
        /// (`target-worker`) names the side that worker was missed on:
        /// the side its number reached nobody on, or the side its
        /// decision was left with nobody on. Where no pair the call named
        /// could be sent to, that is the decode side where a target of
        /// the call's last ask lost its decode alone, its prefill being
        /// one that could be sent to, and the prefill side otherwise. A
        /// first decision's shed as saturated that stands behind a rule
        /// names the side of the list the gateway's own pair selection
        /// judges there.
        /// `None` for a single request, for a pair refused
        /// before either side was read, and for a refusal behind the
        /// route's rule that is made of no one list its caller holds: the
        /// shed under the rule's name (`admits`), and a first decision's
        /// shed with no endpoint that stands where the rule takes no
        /// worker of a list, since the rule narrowed the lists that
        /// decision judged. A pair held to its named prefill alone
        /// answers both under `target-worker`, on no side. One made where
        /// no prefill of the lists pairs with a decode keeps its side: the
        /// rule is not asked behind it.
        /// `None` too under the budget, and for what the call does not
        /// place.
        side: Option<PairSide>,
    },
    /// The engine queued the request. A caller that holds no request
    /// answers as for overloaded workers.
    Queued,
    /// The caller's admission limit alone left a pair's request no prefill
    /// ([`PairAdmission::has_capacity`](crate::engine::PairAdmission::has_capacity)):
    /// a prefill with an open partner stands but for its limit, and no
    /// such prefill is under its limit. A partner is open where the decode
    /// side's hard filters keep it, and where the binding's all-vetoed
    /// rule lifts the overload veto for a decode pool that is over its
    /// thresholds whole. The pair's default answers this too, where the
    /// one prefill it took by load is at its limit. The request is not
    /// refused: its caller waits for a prefill's room and places it again.
    /// Outside that default, a prefill at its limit with no open partner
    /// makes no request wait: a request with no other prefill is answered
    /// its shed. A document with no capacity filter never answers this.
    PrefillsFull,
    /// The call's decisions named workers and none of their targets could
    /// be sent to when the call came to it: a target holds where its
    /// worker, and a pair's decode beside it, is ready, let through by its
    /// circuit breaker and admitted by its route's rule. A caller answers
    /// that it has no worker for the request.
    ///
    /// For a single request, for a pair and for an encode request's pair,
    /// a rule that answers each worker one way within a call is not the
    /// cause: a request whose rule takes no worker of a list is shed,
    /// under the rule's name (`admits`), by its first decision's shed
    /// where that stands, or by a pair's one decision more over the rule's
    /// prefills, and one whose rule takes some is decided among
    /// them. Under a profile that filters by health a list
    /// with no ready worker is shed by that filter. So this answer is left
    /// for a worker that stopped being ready between a decision's read of
    /// it and the send, for a profile that lists no such filter, and for a
    /// rule that refused as a pick a worker it took when it was asked of
    /// the list.
    ///
    /// An encode request has no worker also where each decision of its
    /// call named an encoder that could not be sent to.
    NoWorkers,
    /// The engine refused the call's last decision, for the reason it
    /// carries: it made none in its place. Where the call decided again,
    /// the decisions it gave up stand behind this one. The fault is the
    /// deciding side's own and not the fleet's state: a caller answers
    /// that the decision failed, and gives no retry hint.
    Refused(DecisionRefused),
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

/// Whether the route's own rule admits `worker` ([`PlaceRequest::admits`]):
/// every worker, for a request that states none.
fn admitted(request: &PlaceRequest<'_>, worker: &dyn Worker) -> bool {
    request.admits.is_none_or(|admits| admits(worker))
}

/// Whether `request` can be sent to `worker` now: the worker is ready
/// ([`ready`]) and its route's own rule admits it. The rule is asked of a
/// worker that is ready, and of no other.
pub(super) fn sendable(request: &PlaceRequest<'_>, worker: &dyn Worker) -> bool {
    ready(worker) && admitted(request, worker)
}

/// The refusal of a request this call does not place, by name: the shed a
/// decision answers for what no stage of it runs.
const fn declined(stated: &'static str) -> Unplaced {
    Unplaced::Shed {
        reason: ShedReason::PolicyDeclined,
        emptied_by: stated,
        retry_after_secs: 1,
        side: None,
    }
}

/// The engine's one decision for `made` ([`InProcess::decide_at`]): the
/// place it answers, or `None` with its whole decision written into
/// `whole`, the room its caller keeps ([`Loaded::decide_into`]); and the
/// instant it was dated at. A place dates nothing: no clock is read for
/// it, but for the record of one that is kept as a sample
/// ([`InProcess::explained`]). A single's ask and a pair's decide here,
/// three lists with no item
/// stated among the pairs. An encode request's items are decided with its
/// pair in one call ([`Loaded::decide_encode_at`]), and a decode decided
/// again is asked through its guard ([`decode_again`]).
#[inline]
pub(super) fn decide(
    engine: Loaded<'_, '_>,
    made: &Made<'_, '_>,
    whole: &mut Option<(DecideOutcome, bool)>,
) -> Result<(Option<AlonePick>, Option<Instant>), DecisionRefused> {
    let dated = Cell::new(None);
    let clock = || {
        let now = Instant::now();
        dated.set(Some(now));
        now
    };
    let alone = engine.decide_into(made, Some(made), clock, whole)?;
    Ok((alone, dated.get()))
}

/// The pick of a whole decision, or why the decision placed nothing: its
/// shed, with the reason and the stage that left nobody, or its queue.
pub(super) fn picked(decision: &Decision) -> Result<&Pick, Unplaced> {
    match decision {
        Decision::Pick(pick) => Ok(pick),
        Decision::Shed(shed) => Err(Unplaced::Shed {
            reason: shed.reason,
            emptied_by: shed.emptied_by,
            retry_after_secs: shed.retry_after_secs,
            side: shed.side,
        }),
        Decision::Queue(_) => Err(Unplaced::Queued),
    }
}

/// The workers of one target, as its caller's lists hold them: the worker
/// the request goes to, and a pair's decode.
pub(super) type Sides = (Arc<dyn Worker>, Option<Arc<dyn Worker>>);

/// One walk of a whole decision's targets, best first: the first target
/// that holds is taken. `sides` answers a target's workers where each can
/// be sent to, or the ids of those that cannot: a single request's worker,
/// or each side of a pair by its own id.
///
/// `None` where no target holds. The decision is then given up by name
/// ([`give_up`]), and the ids `sides` named join `missed`, each once: the
/// next decision is told to leave them out. A document's `not-tried`
/// filter does so while another worker is left, and steps aside where
/// none is.
///
/// What `sides` asks can panic, a route's own rule among it. A decision
/// that is leased is then given up by name, and the panic goes on to the
/// caller: a walk leaves no lease behind. A decision that is not leased
/// owes nothing, and its targets are asked with nothing between.
///
/// # Errors
///
/// The decision's shed or its queue ([`picked`]).
#[inline]
pub(super) fn walk<'a>(
    engine: Loaded<'static, '_>,
    decided: DecideOutcome,
    later: bool,
    now: Instant,
    missed: &mut Vec<WorkerId>,
    sides: impl Fn(&Target) -> Result<Sides, [Option<WorkerId>; 2]>,
) -> Result<Option<Placement<'a>>, Unplaced> {
    walk_naming_side(engine, decided, later, now, missed, None, sides)
}

/// [`walk`], which names a side too. Where no target holds, `unsent` is
/// told the side the walk could not send to ([`Call::unsent`]): the decode
/// side where a target lost its decode alone, its prefill being one that
/// could be sent to, and the prefill side otherwise. A pair's ask hands
/// it. A walk that takes a target writes nothing, and a walk handed no
/// `unsent` names no side.
///
/// # Errors
///
/// The decision's shed or its queue ([`picked`]).
#[inline]
pub(super) fn walk_naming_side<'a>(
    engine: Loaded<'static, '_>,
    decided: DecideOutcome,
    later: bool,
    now: Instant,
    missed: &mut Vec<WorkerId>,
    unsent: Option<&mut Option<PairSide>>,
    sides: impl Fn(&Target) -> Result<Sides, [Option<WorkerId>; 2]>,
) -> Result<Option<Placement<'a>>, Unplaced> {
    let pick = picked(&decided.decision)?;
    let mut failed = [[None; 2]; MAX_TARGETS];
    let mut ask = || {
        for (index, target) in pick.targets.iter().enumerate() {
            match sides(target) {
                Ok(sides) => return Some((index, sides)),
                Err(ids) => failed[index] = ids,
            }
        }
        None
    };
    let held = if decided.lease.is_some() {
        match catch_unwind(AssertUnwindSafe(ask)) {
            Ok(held) => held,
            Err(panic) => {
                give_up(engine.object, decided, later, now);
                resume_unwind(panic)
            }
        }
    } else {
        ask()
    };
    let Some((index, (worker, decode))) = held else {
        give_up(engine.object, decided, later, now);
        // A target that lost its decode alone had a prefill that could be
        // sent to. The rows are read once: for the side, and for the ids
        // the next decision is told.
        let mut decode_alone = false;
        for ids in failed {
            decode_alone |= matches!(ids, [None, Some(_)]);
            for id in ids.into_iter().flatten() {
                if !missed.contains(&id) {
                    missed.push(id);
                }
            }
        }
        if let Some(unsent) = unsent {
            *unsent = Some(if decode_alone {
                PairSide::Decode
            } else {
                PairSide::Prefill
            });
        }
        return Ok(None);
    };
    Ok(Placement::taken(
        engine, decided, later, now, index, worker, decode,
    ))
}

/// Gives up a whole decision none of whose targets can be sent to. A lease
/// its pick was issued is released by name, and the stages that take a
/// release hear it. A decision that is not leased owes nothing.
pub(super) fn give_up(
    engine: &'static InProcess,
    decided: DecideOutcome,
    later: bool,
    now: Instant,
) {
    if decided.lease.is_some() {
        DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now)
            .released(ReleaseReason::CommitFailed, now);
    }
}

/// Holds the later asks of a call to the workers of `list` its route's
/// rule takes: a single request's one list, or one of a pair's two, a
/// list at a time. An encode request's pair is a pair here, with the items
/// the request carries or with none, and its encoders' list never comes
/// here. The rule is asked of every worker of `list`, ready or not, in the
/// list's order, among the places its caller states for the list
/// (`stated`, and every place for `None`). The places it takes are kept in
/// `held`, ascending, each once: a single request's in its call
/// ([`Call::admitted`]), and each of a pair's two beside the asks that are
/// lent them ([`pair`]). It is asked of a list once a call, while the call
/// holds no decision: behind the call's first decision none of whose
/// targets can be sent to, whether it refused them or they were not ready,
/// and behind a pair's or an encode request's first decision that sheds
/// the request: as saturated, or with no endpoint where the pair's own
/// test sends it to the rule ([`pair::goes_to_the_rule`]). A request that
/// states no rule keeps no places: its later asks are made over its lists
/// whole. A rule that panics here leaves no lease behind. Out of line: a
/// request whose first decision takes a target runs none of it.
///
/// A worker that is not ready is asked too. A rule that takes no worker
/// is so told from a rule whose workers are not ready: the first is the
/// rule's own answer, and the second is the decision's, by the filter
/// that left nobody.
///
/// The rule's workers are found behind the call's first ask that takes
/// no target, so the two asks the call has left are made among them
/// ([`ASKS`]).
///
/// # Errors
///
/// The shed of a request whose rule takes no worker of the list
/// ([`ShedReason::NoEndpoints`], under `admits`): a single request is
/// asked no further decision. A pair's routine answers its first
/// decision's shed in its place where that shed stands, or the shed of
/// one decision more over the rule's prefills ([`pair::again`]).
#[cold]
#[inline(never)]
fn hold_to_rule(
    request: &PlaceRequest<'_>,
    list: Option<LentList<'_>>,
    stated: Option<&[u32]>,
    held: &mut Option<Vec<u32>>,
) -> Result<(), Unplaced> {
    if held.is_some() || request.admits.is_none() {
        return Ok(());
    }
    let workers = list.map_or(&[][..], |list| &list.workers[..]);
    let takes = |worker: &Arc<dyn Worker>| admitted(request, worker.as_ref());
    let places: Vec<u32> = match stated {
        None => (0..)
            .zip(workers)
            .filter_map(|(place, worker)| takes(worker).then_some(place))
            .collect(),
        // A place past the list's end names no worker: the caller's
        // places may be those of an older list.
        Some(stated) => stated
            .iter()
            .copied()
            .filter(|&place| workers.get(place as usize).is_some_and(takes))
            .collect(),
    };
    if places.is_empty() {
        return Err(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: ADMITS,
            retry_after_secs: 1,
            side: None,
        });
    }
    *held = Some(places);
    Ok(())
}

/// The asks of one call: the first, and two more that leave out the
/// workers the call could not send to. An ask is one decision of the call
/// and the walk of its targets. The two later asks of a request that
/// states a rule are made among the workers the rule takes
/// ([`hold_to_rule`]).
const ASKS: usize = 3;

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
    hashes: OnceCell<PromptHashes>,
    /// The side of a pair that the call's last walk with no target could
    /// not send to, written by that walk ([`walk_naming_side`]): the
    /// decode side where a target of the walk lost its decode alone, its
    /// prefill being one that could be sent to, and the prefill side
    /// otherwise. A pair that names its worker and ends with no worker
    /// for its asks is shed on it ([`target`]).
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
    unsent: Option<PairSide>,
    /// The prompt's whole list of block hashes, where the call's making
    /// left the list to the lookup that reads it ([`Made::hashed_whole`]):
    /// made where that lookup first asks, once a call, and `None` under a
    /// document that reads no block.
    whole: OnceCell<Option<PromptHashes>>,
    /// The asks of the call that took no target.
    spent: usize,
    /// The places of the list whose worker the route's rule takes
    /// ([`PlaceRequest::admits`]), ascending, each once, from the first
    /// decision none of whose targets could be sent to
    /// ([`hold_to_rule`]). The call's later asks are held to them
    /// ([`Constraints::places`]). `None` ahead of that decision, and for a
    /// request with no rule. For a request that names its worker they are
    /// that worker's one place, from ahead of the call's first ask
    /// ([`target`]), and the rule is not asked of the list again. A single
    /// request's alone: the places of a pair's two lists stand beside its
    /// later asks ([`pair`]).
    admitted: Option<Vec<u32>>,
    /// Whether the call's requests are made with no routing value and no
    /// client's value: a request that names its worker over a list none of
    /// whose workers can be sent to ([`target`]). The worker a request
    /// names outranks its routing value, so that value keeps no worker
    /// over a veto.
    unkeyed: bool,
    /// Whether the call's pair is held to the two workers its request
    /// names ([`target::pair`]). Such a call makes one decision: it books
    /// the decode with its prefill under every document, and no ask
    /// follows it ([`pair`]). The worker a request names is that
    /// decision's, so no later one moves it.
    named_pair: bool,
    /// The number the call's pair names its decode by, where the decode's
    /// side alone reads the request's target source ([`target`]): the
    /// prefill is each decision's own pick, so each ask of the call states
    /// the number beside the decodes' places, and its decision counts it
    /// behind the prefill it picks ([`Constraints::target_worker`]). A text
    /// that is no number is stated as one that reaches no decode. `None`
    /// for every other call. Each of its decisions books the decode with
    /// its prefill under every document, as a pair held on both sides
    /// does: a decode asked again would count no number.
    named_decode: Option<u32>,
    /// The instant of the call's first ask, for a request that states a
    /// budget: the budget is counted from it ([`Call::left`]). `None` for
    /// a request that states no budget: no clock is read for its budget.
    began: Option<Instant>,
}

impl Call {
    /// What the call's next decision is given of the request's budget, in
    /// microseconds: all of it for the first decision, whose instant is
    /// kept, and the budget less the time since for a later one. A
    /// request that states no budget (`0`) is given none, and no clock is
    /// read for its budget.
    ///
    /// # Errors
    ///
    /// The shed of a request none of whose budget is left
    /// ([`ShedReason::Deadline`], under `budget`): no further decision is
    /// made.
    fn left(&mut self, request: &PlaceRequest<'_>) -> Result<u32, Unplaced> {
        let budget = request.budget_micros;
        if budget == 0 {
            return Ok(0);
        }
        let Some(began) = self.began else {
            self.began = Some(Instant::now());
            return Ok(budget);
        };
        let spent = u32::try_from(began.elapsed().as_micros()).unwrap_or(u32::MAX);
        match budget.saturating_sub(spent) {
            0 => Err(Unplaced::Shed {
                reason: ShedReason::Deadline,
                emptied_by: BUDGET,
                retry_after_secs: 1,
                side: None,
            }),
            left => Ok(left),
        }
    }

    /// What an ask answers where no target of its decision can be sent to:
    /// the answer of `ask`, the call's next ask, while the call has an ask
    /// left, and no worker for the request ([`Unplaced::NoWorkers`]) behind
    /// its last: [`ASKS`] asks a call. The next ask answers in this one's
    /// place, and a request whose first ask takes a target pays for no
    /// other.
    #[cold]
    pub(super) fn again<'a>(
        &mut self,
        ask: impl FnOnce(&mut Self) -> Result<Placement<'a>, Unplaced>,
    ) -> Result<Placement<'a>, Unplaced> {
        self.spent += 1;
        if self.spent < ASKS {
            ask(self)
        } else {
            Err(Unplaced::NoWorkers)
        }
    }
}

/// The id the list's records name the worker at `place` by: the id a
/// decision's filters compare. The engine's value of the list holds the
/// records, and the ask's decision filled the list's cell, so this ask of
/// the cell fills nothing.
fn named(engine: Loaded<'_, '_>, list: Option<LentList<'_>>, place: u32) -> Option<WorkerId> {
    let records = engine
        .members_of(list?)
        .records(&engine.document.spec, &engine.clock);
    records.ids().get(place as usize).copied()
}

/// Raises again the panic of a route's rule that was asked inside a
/// decision ([`RequestMaker::sends_to`]), behind the decision that asked
/// it. That decision stored the prompt for no worker. A lease it was
/// issued is given up by name first, as the walk gives one up behind a
/// rule that panics there. Out of line: a rule that panics is no request's
/// path.
#[cold]
#[inline(never)]
fn raise(
    engine: &'static InProcess,
    panic: Box<dyn Any + Send>,
    whole: Option<(DecideOutcome, bool)>,
    decided: Result<(Option<AlonePick>, Option<Instant>), DecisionRefused>,
) -> ! {
    if let (Some((outcome, later)), Ok((_, dated))) = (whole, decided) {
        give_up(engine, outcome, later, dated.unwrap_or_else(Instant::now));
    }
    resume_unwind(panic)
}

/// One ask for a single request over `list`, the list its caller lent:
/// one decision, and the first of its targets that can be sent to
/// ([`walk`]): ready ([`ready`]) and admitted by its route's rule. At a
/// pick the rule is asked of a worker that is ready, and of no other. A
/// target the decision asked is taken by the answer it was given there
/// ([`Sent`]): it is read once a decision, and its rule asked once.
/// Where none can be sent to, the decision is given up by name, its
/// targets join the ones the call's next ask leaves out
/// ([`Call::missed`]), and the call's next ask answers ([`Call::again`]).
/// The later asks of a request that states a rule are held to the
/// workers the rule takes ([`hold_to_rule`]).
fn single<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    list: Option<LentList<'_>>,
    call: &mut Call,
) -> Result<Placement<'a>, Unplaced> {
    let mut made = Made {
        budget_micros: call.left(request)?,
        ..Made::asked(engine, request, call)
    };
    let mut whole = None;
    let decided = decide(engine, &made, &mut whole);
    if let Some(panic) = made.panicked.take() {
        raise(engine.object, panic, whole, decided);
    }
    let (alone, dated) = decided.map_err(Unplaced::Refused)?;
    let at = |place: u32| list?.workers.get(place as usize);
    // Each arm answers its own placement: a place's holds no decision, and
    // is written without the room of one.
    match alone {
        // The engine picked the place on the read of the worker that
        // found it eligible, and under a document whose session table
        // alone takes a word on a lease read it ready a second time: the
        // worker is not read again here, and its route's rule is asked of
        // it.
        Some(pick) => {
            let worker = at(pick.place).ok_or(Unplaced::NoWorkers)?;
            // A caller of this call states no worker id of its own: where
            // its list's records stand, they name the worker by its
            // address, as the placement of a place does. The decision
            // filled the list's cell, so a debug build's ask fills nothing.
            debug_assert!(
                list.and_then(|list| engine.members_of(list).records.get())
                    .is_none_or(|rows| rows.ids()[pick.place as usize] == worker_id(&**worker)),
                "a caller of `place` states no worker id of its own"
            );
            if !admitted(request, worker.as_ref()) {
                let id = named(engine, list, pick.place).ok_or(Unplaced::NoWorkers)?;
                call.missed.push(id);
                // The pick read the worker ready, and a place holds no
                // decision: the rule is asked of the list at once.
                hold_to_rule(request, list, None, &mut call.admitted)?;
                return call.again(|call| single(engine, request, list, call));
            }
            let mut placement = Placement::alone(Arc::clone(worker), engine.label_at(pick.profile));
            placement.routing_key = made.key();
            Ok(placement)
        }
        None => {
            // The pass writes its decision ahead of every answer but a
            // place.
            let Some((decided, later)) = whole else {
                return Err(Unplaced::Refused(DecisionRefused::Faulted));
            };
            let (key, sent) = (made.key(), made.sent.get());
            let now = dated.unwrap_or_else(Instant::now);
            let taken = walk(engine, decided, later, now, &mut call.missed, |target| {
                let worker = at(target.position);
                // The lookup's answer stands for a target it asked.
                let sendable = sent.answer(target.position).unwrap_or_else(|| {
                    worker.is_some_and(|worker| sendable(request, worker.as_ref()))
                });
                match worker {
                    Some(worker) if sendable => Ok((Arc::clone(worker), None)),
                    _ => Err([Some(target.id), None]),
                }
            });
            // One match takes the placement out of the walk's answer. A
            // `?` ahead of a `let else` takes it out in two steps, and a
            // build copies it at each.
            match taken {
                Ok(Some(mut placement)) => {
                    placement.routing_key = key;
                    Ok(placement)
                }
                Ok(None) => {
                    // The walk gave the decision up: a rule is asked of
                    // the list while the call holds none.
                    hold_to_rule(request, list, None, &mut call.admitted)?;
                    call.again(|call| single(engine, request, list, call))
                }
                Err(unplaced) => Err(unplaced),
            }
        }
    }
}

impl InProcess {
    /// Places one request: decides it, and answers the worker it goes to
    /// as a [`Placement`] its caller keeps with the request.
    ///
    /// The engine makes the request of what `request` states, decides it,
    /// and takes the first target of the decision that can be sent to:
    /// ready, let through by its circuit breaker, and admitted by its
    /// route's own rule ([`PlaceRequest::admits`]). A request a place
    /// answers is placed on the worker at that place, which the pick itself
    /// read, and under a document whose session table alone takes a word
    /// on a lease read ready a second time: the rule alone is asked of it.
    /// The placement names the request's routing value where the document
    /// derived one that leaves in the clear, and none for a request a place
    /// answers: no fact of such a request is made.
    ///
    /// Where no target of a decision can be sent to, the engine decides
    /// again with their workers excluded ([`Constraints::excluded`], which
    /// a profile's `not-tried` filter honours while another worker
    /// remains), three decisions a call. The first is given the request's
    /// whole budget ([`PlaceRequest::budget_micros`]), and each later one
    /// what the call has left of it: a request none of whose budget is
    /// left when a decision is due is shed ([`ShedReason::Deadline`],
    /// under `budget`). Once the excluded workers and the tried ones cover
    /// the list, the filter keeps them all, and a later decision can name
    /// an excluded worker again.
    ///
    /// A worker the rule refuses is left out like one that is not ready.
    /// The rule is asked of a decision's picks first: a request whose
    /// first pick it admits pays one call of it, inside the decision where
    /// that stores the prompt for a worker. Behind the first decision
    /// none of whose picks can be sent to, whether the rule refused them
    /// or they were not ready, it is given up and the rule is asked of
    /// every worker of the list, ready or not, once a call: of a single
    /// request's list, and of each of a pair's two, among the places its
    /// caller states for that side. The call's later decisions are held
    /// to the places of the workers it takes ([`Constraints::places`], and
    /// a pair's a side each,
    /// [`PairLists::prefill_places`](crate::engine::PairLists::prefill_places)
    /// and [`PairLists::decode_places`](crate::engine::PairLists::decode_places)):
    /// the one such worker of a long list is
    /// found, one over its overload thresholds is answered by the
    /// binding's rule for a list wholly over them, and one an earlier
    /// attempt tried is handed again where no other is left. The call's
    /// two later decisions are made among them: three decisions a call.
    /// A request whose rule takes no worker of a list is shed
    /// with no endpoint, under `admits`. A pair whose first decision shed
    /// it with no endpoint keeps that shed there but in the last case
    /// below, under the stage that
    /// decision named and with no side; where no prefill of its lists
    /// pairs with a decode, or the pair states a hard demand, that shed
    /// answers with its side and the rule is not asked behind it. Behind
    /// a hard demand a pair's decision is handed the workers that meet
    /// it, a part of each list, and its shed is not judged by the count
    /// of a list: it is saturated where the overload veto left a pass
    /// nobody, and else has no endpoint. A pair that states no hard demand,
    /// and whose host states no places of the prefills or the decodes
    /// ([`PairLists::prefill_places`](crate::engine::PairLists::prefill_places),
    /// [`PairLists::decode_places`](crate::engine::PairLists::decode_places)),
    /// is answered otherwise in three cases, under a document that sheds
    /// a list wholly over its thresholds. Two stand behind a first
    /// decision that shed it as saturated and said the list it shed over.
    /// That shed is its answer where the rule takes no prefill
    /// and the decision shed over the prefills. It is its answer too where
    /// the rule takes a prefill and no decode: where one of its prefills
    /// can take the request and the decision shed over the decodes, and
    /// where none can, one is ready and every ready one is over its
    /// thresholds. In the third the rule takes a prefill and no decode,
    /// and no such shed stands: one decision more is held to the rule's
    /// prefills, with no decode lent, and its shed as saturated over
    /// those of them that pair with a decode is the answer, for the
    /// prefill side. One whose rule takes workers
    /// none of which is ready, or
    /// workers of a pair's two lists no two of which pair, is answered by
    /// the decision held to them, which names what left nobody. Behind a
    /// shed made over the prefills, a rule whose every prefill pairs with
    /// no decode is answered as one that takes no prefill. An encode
    /// request is held so over its pair's two lists, with the items it
    /// carries or with none: the rule is asked of no encoder, and the
    /// encoders' list is decided as its caller lent it. One that states an
    /// item, or holds its decode to the ones that take images, is not
    /// answered so behind a rule whose every prefill pairs with no decode:
    /// its decodes are asked too, and where the rule takes one a decision
    /// among the rule's workers answers. Nor is the third case's decision
    /// made for it: with no such shed standing it keeps the answer it had,
    /// under `admits` or by its first decision's shed with no endpoint.
    ///
    /// A decision that is given up has written what its stages write as
    /// they decide, and its release takes none of that back: the tokens a
    /// `bucket` profile books on a worker a prefill decision listed, a new
    /// session's claim of its worker, the move of a session's pin at its
    /// cap. A decision whose later target is taken has written for its
    /// first target all the same.
    ///
    /// Where a prompt went is written otherwise, in the record of sent
    /// prefixes a `cache_aware` profile keeps and in a tree the
    /// caller lends ([`Prompt::entry`]). A decision that writes either
    /// asks, ahead of the write, which of its targets this call can send
    /// to ([`RequestMaker::sends_to`]): ready, and admitted by the route's
    /// rule. The prompt is stored for the first such target, and the call
    /// takes that target by the same answer: a worker is read, and its
    /// rule asked, once a decision. Where there is none the prompt is
    /// stored for no worker: a decision that is given up leaves no holder
    /// behind.
    ///
    /// A request with a pair's two lists is placed over both, a target at
    /// a time. The first target both of whose workers can be sent to is
    /// taken, the rule asked of each of the two: the placement names the
    /// prefill as its worker and the decode beside it. Where no target
    /// holds, the sides that could not be sent to are excluded, each by
    /// its own id, and a side that held stays a candidate of the next
    /// decision, each given the request's whole budget. Behind the first
    /// decision no target of which holds, or that sheds the request as
    /// saturated, the later decisions of a pair that states a rule are
    /// held to the workers the rule takes on each of its two lists, as a
    /// single request's are to those of its one: three decisions a call.
    /// Which workers pair is the caller's word, in the classes its lists
    /// carry: no pair is tested here again. Where the prefill side's
    /// filters leave its run nobody and a prefill with an open partner
    /// stands but for the caller's limit, the answer is
    /// [`Unplaced::PrefillsFull`], on which the caller waits for room and
    /// places the request again. So it is where the pair's default takes a
    /// prefill at that limit, under a profile that lists it as a hard
    /// filter.
    ///
    /// A request with an encode request's three lists is decided over the
    /// three ([`InProcess::decide_encode_at`]), with the items it
    /// carries. Its pair's targets are tried as a pair's are, and the
    /// placement names each item's encoder beside the pair
    /// ([`Placement::encoders`]). An item has one encoder: where it cannot
    /// be sent to, the decision is given up and the engine decides again
    /// with that encoder excluded, and with each side of the pair that
    /// could not be sent to either. Behind a decision given up its call
    /// makes three decisions at most, each given the request's whole
    /// budget. Behind the first of them, and behind a first decision that
    /// sheds the request as saturated over a list of its pair, the rule
    /// is asked of the pair's two lists, and the later decisions are
    /// made among the workers it takes on each, as a pair's are. A
    /// request with no item whose decode is held to no images is its
    /// pair's alone: its first shed with no endpoint goes to the rule as
    /// a pair's does. One that states an item, or holds its decode to
    /// the ones that take images, keeps such a shed, and its pair's
    /// decision asks no question of a shed: it is answered as it was
    /// before the question stood. A shed made for the encoders
    /// stands, and no rule is asked behind it: the rule narrows nothing of
    /// them. An encoder is read by its state alone:
    /// the route's rule is asked of the pair's two workers and of its two
    /// lists. With no item the three lists are placed as their pair.
    ///
    /// A single request may name its worker. Under a document whose
    /// binding names a target source
    /// ([`KeySourceSpec::target`](crate::document::spec::KeySourceSpec::target)),
    /// the header's number counts, from 0 and in the list's order, the
    /// workers of the request's list it can be sent to when the call reads
    /// them: available by the worker's own test, and taken by the route's
    /// rule, which is asked of each available worker up to the one the
    /// number reaches. So a number names another worker while one ahead of
    /// it is down, and a worker over its overload thresholds is not
    /// counted. The call's one decision is held to the place of the worker
    /// the number reaches ([`Constraints::places`]), ahead of the
    /// request's routing value and of every other rule of the document:
    /// its filters read the worker, and a retry is handed the worker it
    /// tried. The request goes to that worker or to none. It is shed with
    /// no endpoint, under `target-worker`, where the header's text is no
    /// number, fewer workers can be sent to, the decision names no worker
    /// because of the worker, or the worker cannot be sent to after all.
    /// Where no worker of the list can be sent to, there is none to count:
    /// the request is placed as one that names no worker, by the binding's
    /// rule for workers that are all vetoed, and with no routing value,
    /// which a named worker outranks. A header that is absent, empty or
    /// not text names no worker.
    ///
    /// A pair may name its worker too, a side at a time. A pair's binding
    /// names one list of sources for its two sides, and its target source
    /// is the side's whose profile is the hash ring keyed by no prefix.
    /// The prefill's number counts, in the prefill list's order, the
    /// prefills that can be sent to and have a partner that can, at the
    /// places the caller states for each side: those with such a partner
    /// among their first partners alone, where there is one. Where the
    /// decode's side is the ring's too, its number counts those partners
    /// of the prefill reached, and the call's one decision is held to the
    /// two places
    /// ([`PairLists::prefill_places`](crate::engine::PairLists::prefill_places),
    /// [`PairLists::decode_places`](crate::engine::PairLists::decode_places)).
    /// Where it is not, the decision is
    /// held to the prefill's place and the decode is its own profile's
    /// pick. A named prefill at the caller's admission limit is waited
    /// for where that decision says so ([`Unplaced::PrefillsFull`]), a
    /// named decode with no free room is taken, and a retry is handed
    /// the pair again. A number that
    /// reaches no worker on a side, and a pair the held decision cannot
    /// place, shed the request under `target-worker`. Where no decode can
    /// be sent to and the binding steers decodes that are all over their
    /// thresholds, one over them alone counts as a partner, and the decode
    /// goes by load. Under a binding that sheds them no decode counts so.
    /// Where no prefill is counted, the pair is placed as one that names
    /// no worker, with its routing value. A pair held to the two workers
    /// it names books its decode with its prefill under every document
    /// ([`PlaceRequest::decode_again`]): a decode asked again behind its
    /// prefill ([`Placement::decode_again`]) is refused as one that never
    /// stood chosen, and that decode stands.
    ///
    /// Where the decode's side alone is such a ring, the prefill is its
    /// own profile's pick, which a decision alone knows. The call lends
    /// the places of the decodes that can be sent to, and each of its
    /// decisions counts the decode's number among the partners there of
    /// the prefill it picks: its first partners where the pair is made
    /// among those. A number that reaches none sheds the request under
    /// `target-worker`, and so does a decode the decision cannot name or
    /// the call cannot send to. Such a pair books its decode with its
    /// prefill too. Where no prefill opens the pair, the header is not
    /// read and the pair is placed as one that names no worker. A prefill
    /// opens it where a decode that can be sent to partners it, and it
    /// can be sent to itself or, under a profile that weighs load itself
    /// and a binding that steers, is over its thresholds alone.
    ///
    /// An encode request is not read for a target: it is placed by the
    /// rest of the document.
    ///
    /// Declined by name ([`ShedReason::PolicyDeclined`]): a request that
    /// carries encode items over other lists than an encode request's
    /// three (`encode-items`). A request that states a demand is held to
    /// the workers of each list that meet it ([`PlaceRequest::demands`]):
    /// a list the demand leaves nobody sheds it with no endpoint
    /// ([`ShedReason::NoEndpoints`]), under that demand's stage.
    ///
    /// Its caller counts each request it sends under the placement's
    /// routing value on the request's worker, as that worker's load guard
    /// does. The decision is told that the count stands
    /// ([`RequestFacts::session_counted`]), so a sessions rule with an
    /// in-flight cap keeps a session on its worker below the cap. A stage
    /// is shown the headers of the document's view that the request
    /// carries ([`AppliedSpec::view_headers`]), each named by its place in
    /// that list.
    ///
    /// Its caller lends each list with the cell beside it and asks the
    /// engine for no value of a list. It lends no load snapshot: a
    /// decision that reads load reports reads the newest snapshot the
    /// engine's own task keeps at hand ([`InProcess::run`]), of the
    /// channel its host handed ([`InProcess::set_loads`]), and no report
    /// until its host has done both.
    ///
    /// The overload test is the engine's. The decision leaves out a worker
    /// over its overload thresholds (the `not-overloaded` filter), and the
    /// binding's own rule answers a list whose every worker is over them.
    /// This call asks it of no worker a decision picked again, so a caller
    /// that asks this call for every request keeps no overload test of its
    /// own. Behind a pair's saturated shed the call reads the rule's own
    /// prefills once more, to say which shed is its answer.
    ///
    /// A program that asks this call states no worker id of its own
    /// ([`FixedFacts::id`](crate::engine::FixedFacts::id)): a placement
    /// names its worker by the fingerprint of the worker's address
    /// ([`Placement::id`]), the id a next attempt states as tried. A debug
    /// build fails a request placed on a worker its list's records name
    /// otherwise.
    ///
    /// Every decision of one call runs under the document in force where
    /// the call began. A document put in force during the call decides
    /// the next call ([`InProcess::apply`]).
    ///
    /// # Errors
    ///
    /// [`Unplaced`]: why the request has no worker. Where a decision
    /// named workers and none can be sent to, a lease it was issued is
    /// given up at once, and the stages that take a release hear it, once
    /// a decision. A route's rule that panics passes its panic to the
    /// caller, behind the same release.
    pub fn place<'a>(&'static self, request: &PlaceRequest<'a>) -> Result<Placement<'a>, Unplaced> {
        if request.encode.is_some() && !matches!(request.lists, WorkerLists::Encode(_)) {
            return Err(declined(ENCODE_ITEMS));
        }
        let document = self.in_force();
        let own = self.under(&document);
        match request.lists {
            // Whether a binding names a target source was decided when the
            // document was applied: under one that names none, a request
            // pays this test and no read of a header.
            WorkerLists::Pair(ref lists) if own.document.spec.names_target() => {
                target::pair(own, request, lists)
            }
            WorkerLists::Pair(ref lists) => pair::place(own, request, lists, &mut Call::default()),
            WorkerLists::Single(list) if own.document.spec.names_target() => {
                target::single(own, request, list)
            }
            WorkerLists::Single(list) => single(own, request, Some(list), &mut Call::default()),
            WorkerLists::None => single(own, request, None, &mut Call::default()),
            // A pair's lists and an encode request's are read where the
            // request holds them: a copy of either here costs every request
            // a larger frame.
            WorkerLists::Encode(ref lists) => encode::three(own, request, lists),
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/in_process/place/tests/mod.rs"]
mod tests;
