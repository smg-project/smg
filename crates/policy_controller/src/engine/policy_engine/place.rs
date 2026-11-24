//! One call for one request ([`PolicyEngine::place`]): what a caller states
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
//!   making, and never for a request a place answers
//!   ([`RequestMaker::plain`]): such a request is not made at all.
//!
//! One thing of the request the engine does not make: the prompt's entry in
//! a token tree its caller lends. The caller names it ([`Prompt::entry`]),
//! by its own rule for a cache partition, and a decision that reads the
//! tree asks it. That decision stores the prompt in the tree for the first
//! of its targets this call can send to ([`RequestMaker::sends_to`]), and
//! for none where it can send to none.
//!
//! The engine then decides, and takes the first target of the decision that
//! can be sent to: ready ([`ready`]), and admitted by its route's own rule
//! ([`PlaceRequest::admits`]). Where no target of a decision can be sent
//! to, the engine decides again with the workers it could not send to
//! excluded, three decisions a call ([`ASKS`]). Behind a single request's
//! decision none of whose targets can be sent to, where the rule refused
//! one that was ready, the later decisions are held to the workers of the
//! list the rule takes ([`hold_to_rule`]), and the call makes one decision
//! among them at least: a fourth where its third found them. The caller is
//! answered a [`Placement`], which it keeps with the request, or the reason
//! there is none ([`Unplaced`]).
//!
//! A single request that names its worker, under a document whose binding
//! says which header names it, is held to that one worker and goes to no
//! other ([`target`]).
//!
//! A request with a pair's two lists is placed a target at a time over both
//! its workers ([`pair`]), three decisions a call at most. Its later
//! decisions are not held to the workers the rule takes.
//!
//! A request with an encode request's three lists is placed the same way:
//! a decision names its pair and each item's encoder, and the request is
//! placed where its pair's two workers and every encoder can be sent to
//! ([`encode`]).
//!
//! A pair's decode decided again behind its prefill is placed by the same
//! read of a worker, over the decodes that decision names
//! ([`decode_again`]). Its caller states the request again, and the
//! request is made of that anew: its routing value derived and its prompt's
//! blocks hashed a second time.

mod decode_again;
mod encode;
mod pair;
mod target;

use std::{
    any::Any,
    cell::{Cell, OnceCell},
    fmt,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    sync::Arc,
    time::Instant,
};

use smg_worker::identity::basic::Worker;

use super::{encode::EncodeLists, Loaded, PolicyEngine};
use crate::{
    document::{
        spec::{AppliedSpec, DecodePick},
        DerivedKeys, KeySource,
    },
    engine::{
        decision::{Decision, Pick, ShedReason, Target, MAX_TARGETS},
        decision_engine::{
            AlonePick, DecideOutcome, DecisionEngine, DecisionRefused, Plain, Request,
            RequestMaker, TokenKey, WorkerLists,
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
    policies::consistent_hashing::{Head, LentRing},
    store::GatewayId,
    tracking::{lease::ReleaseReason, DecisionGuard, Placement},
};

/// A request's headers, as its caller holds them.
///
/// The engine reads the headers the document names and no other: the
/// sources of a request's routing value and of its client's value, and the
/// headers its view shows a stage ([`AppliedSpec::view_headers`]).
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

/// A prompt's entry in a token tree, as the request's caller names it: the
/// one question a decision asks where it reads the tree the caller lends
/// ([`CacheIndexes::token_tree`]). A caller of [`PolicyEngine::place`]
/// answers it for its request ([`Prompt::entry`]), as a maker written by
/// hand answers [`RequestMaker::token_key`].
///
/// [`CacheIndexes::token_tree`]: crate::engine::CacheIndexes::token_tree
pub trait TokenEntry {
    /// The entry by token ids for a tree whose page is `page` tokens
    /// ([`TokenKey`]): `None` where the caller holds no token ids of the
    /// prompt. The marker of a cache partition fills whole pages, which is
    /// why the page is asked with.
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>>;
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
    /// The prompt's entry in the token tree the caller lends
    /// ([`CacheIndexes::token_tree`]), named by the caller's own rule as
    /// the seeds are: the same cache partition, for a tree's lookup.
    /// `None` from a caller that lends no tree or holds no token ids: the
    /// request is then named its holders from the record of hashed
    /// prefixes. One entry answers for every tree the request is decided
    /// in, a pair's two sides too when a pair reads one: the tree's page
    /// is the question's argument.
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
    pub entry: Option<&'a dyn TokenEntry>,
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
/// prompt's entry in a token tree are borrowed as their caller holds them,
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
    /// value where the document names it.
    pub request_id: Option<&'a str>,
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
    /// with `None` are placed as their pair.
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
    /// decode asked again reads no image fact.
    pub decode_again: bool,
    /// The hash ring the caller keeps of the model's workers, lent to the
    /// request's decision ([`LentRing`]): one ring for a pair's two lists,
    /// and no part of a list. Under a document whose ring stage places a
    /// routing value by its own text, the value goes to the worker this
    /// ring's walk reaches, where the caller's own ring policy puts it:
    /// the engine reads the value's place on the ring with the request's
    /// keys. Under the prefix form of the ring stage the engine asks the
    /// ring where each head of the request stands
    /// ([`LentRing::head_position`]). A ring that answers nothing for a
    /// head is asked nothing more and is not lent, and neither is a ring
    /// to a request that has no head. Under any other document the ring
    /// is asked nothing. `None` from a caller that keeps no ring: the
    /// stage places by its own.
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
    /// Behind
    /// a single request's decision none of whose picks can be sent to,
    /// where it refused one that was ready, it is asked of every worker of
    /// the list, once a call, and the call decides again among the workers
    /// it takes ([`PolicyEngine::place`]). A pick it is asked of is ready.
    /// A worker of the list it is asked of may not be: it answers for a
    /// worker in any state. No request is sent to a worker it refuses when
    /// it is asked of that worker as a pick, whatever it said of the
    /// worker before. It is asked of no pair's list: a pair's worker it
    /// refuses is passed over like one that is not ready. A rule that
    /// holds for every request of a route belongs to the list its caller
    /// hands. For a single request that names its worker
    /// ([`PolicyEngine::place`]) it is asked ahead of the call's decision
    /// too, of each available worker the request's number counts.
    ///
    /// Where a pair's decode is decided again on its decode side
    /// ([`PlacementSide::decode_again`](crate::tracking::PlacementSide::decode_again)),
    /// the rule is asked under the lock the pair's two sides share. It
    /// must leave both sides alone: the send or the drop of one, from
    /// inside the rule, waits on that lock and never returns. That
    /// decision asks the request's headers, its ring and its lists'
    /// admission under the same lock, and each is held to the same.
    ///
    /// For an encode request the rule is its pair's: it is asked of the
    /// pair's two workers and of no encoder.
    pub admits: Option<WorkerRule<'a>>,
}

impl<'a> PlaceRequest<'a> {
    /// A first attempt for `model` over `lists` that states no other fact:
    /// no prompt, no header, no id, no tenant, no media, no demand, no
    /// encode item, nothing tried, no budget, no second ask of a decode, no
    /// ring and no rule of its route.
    #[must_use]
    pub fn of(model: &'a str, lists: WorkerLists<'a>) -> Self {
        Self {
            model,
            lists,
            prompt: Prompt::NONE,
            headers: &(),
            request_id: None,
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
    /// The engine that decides. Its one document names the sources of the
    /// request's routing value.
    engine: &'a PolicyEngine,
    request: &'r PlaceRequest<'a>,
    /// The targets earlier decisions of this call named that could not be
    /// sent to.
    missed: &'r [WorkerId],
    /// The places of the list the decision is held to: the workers the
    /// route's rule takes, for an ask behind the decision that had the
    /// rule asked of the list ([`Call::admitted`]). `None`: every place.
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
    /// Whether the request is a pair's decode decided again: a target's
    /// prefill was sent already, and its decode alone is asked.
    decode_alone: bool,
}

/// What a call's maker answered on a decision's targets where the decision
/// asked which of them a prompt is stored for, in a lent token tree or in
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
    /// Whether the route's rule refused a target that was ready.
    refused: bool,
}

impl Sent {
    /// No target asked.
    const NONE: Self = Self {
        places: [0; MAX_TARGETS],
        asked: 0,
        sendable: 0,
        refused: false,
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

impl<'r, 'a> Made<'r, 'a> {
    /// `request` for a decision of `engine` that leaves `missed` out.
    pub(super) const fn new(
        engine: &'a PolicyEngine,
        request: &'r PlaceRequest<'a>,
        missed: &'r [WorkerId],
    ) -> Self {
        Self {
            engine,
            request,
            missed,
            places: None,
            budget_micros: request.budget_micros,
            hashes: OnceCell::new(),
            of_call: None,
            headers: OnceCell::new(),
            key: Cell::new(None),
            place: OnceCell::new(),
            heads: OnceCell::new(),
            panicked: OnceCell::new(),
            sent: Cell::new(Sent::NONE),
            decodes: Cell::new(Sent::NONE),
            decode_alone: false,
        }
    }

    /// `request` for one ask of `call`: its decision leaves out the targets
    /// the call missed so far, is held to the places the call keeps
    /// ([`Call::admitted`]), and borrows the prompt's hashes where an
    /// earlier ask of the call made them. The headers of the document's
    /// view and the key's place on a lent ring, or the heads' places, are
    /// made again by each ask that makes the request: a later ask is the
    /// call's cold path.
    pub(super) fn asked(
        engine: &'a PolicyEngine,
        request: &'r PlaceRequest<'a>,
        call: &'r Call,
    ) -> Self {
        Self {
            of_call: Some(call),
            places: call.admitted.as_deref(),
            ..Self::new(engine, request, call.missed.as_slice())
        }
    }

    /// The request's routing value in the clear, where a making derived
    /// one that leaves so: what the request's placement names. None ahead
    /// of a making, and for a request a place answers, which is not made.
    pub(super) fn key(&self) -> Option<&'a str> {
        self.key.get()
    }

    /// The request's routing value and its client's value, by the sources
    /// its binding names in the engine's document, with where the value a
    /// ring places it by stands on a ring, as `place` answers for that
    /// value's text and its source, `None` for the client's value
    /// ([`KeySources::derive_keys`](crate::document::KeySources::derive_keys)).
    /// The routing value is the first source that holds one; none where
    /// the binding names no source or none holds a value. The client's
    /// value is read only where the request has no routing value: it
    /// places where no session does. A request its call makes with no key
    /// has neither ([`Call::unkeyed`]).
    fn keys(
        &self,
        place: impl FnOnce(&str, Option<KeySource<'a>>) -> Option<u64>,
    ) -> DerivedKeys<'a> {
        let request = self.request;
        let spec = self.engine.spec();
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
    fn stated<'w>(
        &'w self,
        spec: &'w AppliedSpec,
    ) -> (DerivedKeys<'a>, Option<&'a dyn LentRing>, &'w [u64]) {
        let request = self.request;
        let (model, phase) = (request.model, request.phase());
        let Some(ring) = request.ring else {
            return (self.keys(|_, _| None), None, &[]);
        };
        // The two questions are asked of one lookup of the binding.
        let engine = spec.engine();
        let profiles = engine.ring_profiles(model, phase);
        if engine.key_place_of(profiles) {
            let first = self.place.get().is_none();
            let keys = self.keys(|text, _| first.then(|| ring.position(text)));
            let place = self.place.get_or_init(|| keys.ring_position);
            return (keys, Some(ring), place.as_slice());
        }
        let Some(rule) = engine.head_rule_of(profiles) else {
            return (self.keys(|_, _| None), None, &[]);
        };
        // A routing value a header handed leads the prompt: its text is the
        // one head. A client's value places nothing under this form, and
        // neither does a value the request's own id handed: that request
        // is placed by its prompt's heads.
        let key_at = Cell::new(None);
        let first = self.heads.get().is_none();
        let keys = self.keys(|text, source| {
            if first && source.is_some_and(leads_a_prompt) {
                key_at.set(Some(ring.head_position(Head::Text(text))));
            }
            None
        });
        let heads = self.heads.get_or_init(|| {
            let mut places = Places::default();
            match key_at.take() {
                Some(answer) => places.state(answer),
                None => {
                    let (tokens, text) = (request.prompt.tokens, request.prompt.text);
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
        });
        // The ring is lent with a place: a request that has no head, and
        // one whose lender places none, state nothing and lend nothing.
        let places = heads.as_slice();
        let ring = (!places.is_empty()).then_some(ring);
        (keys, ring, places)
    }
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
    /// The lender places no head: the request is lent no ring and states
    /// no place.
    unplaced: bool,
}

impl Places {
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
        // The engine decides under its one document, the one it hands here.
        debug_assert!(std::ptr::eq(spec, self.engine.spec()));
        // The request's place is taken at the first making: a later one
        // asks the ring nothing, and nothing is hashed again.
        let (keys, ring, places) = self.stated(spec);
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
                self.engine
                    .prompt_hashes(model, tokens, text, request.prompt.seeds)
            });
            hashes.write_to(&mut facts);
        }
        // The document's word, like the keys: read from the generation
        // that decides. A request over one list, or over none, states no
        // later decode whatever its caller says: the word is a pair's, as
        // the plain test reads it. An encode request whose decode is held
        // to the ones that take images, by its demand or by the preference
        // of references beside pixels, states none either: a decode asked
        // again reads no image fact. The hold is read last, so a request
        // that asks no decode again reads none of it.
        let later = request.asks_decode_again()
            && spec.decode_pick() == DecodePick::AfterPrefill
            && !request
                .encode
                .is_some_and(|items| items.holds_decode_to_images(request.media.holds_pixels()));
        Request {
            id: request.request_id.map_or(&[], str::as_bytes),
            facts,
            constraints: Constraints {
                excluded: self.missed,
                places: self.places,
                decode_later: later,
                demands: request.demands,
                ..Constraints::default()
            },
        }
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

    // Asked where a plain request's pick is leased, ahead of its making:
    // the hashing counts the text of a request that is made, so a text is
    // passed over once.
    fn prompt_chars(&self) -> u32 {
        let prompt = self.request.prompt;
        let sized = self.engine.spec().needs().prompt_size;
        prompt_chars_of(prompt.tokens, prompt.text, sized)
    }

    // The text's bytes, for the same lease: its length is at hand.
    fn prompt_bytes(&self) -> u32 {
        let prompt = self.request.prompt;
        prompt_bytes_of(prompt.tokens, prompt.text)
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.request.lists
    }

    // The caller's own naming, handed on: nothing else reads the field.
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>> {
        self.request.prompt.entry?.token_key(page)
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
                    Ok(admitted) => {
                        sent.refused |= !admitted;
                        admitted
                    }
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
        /// (`target-worker`). A caller's refusal line names it.
        emptied_by: &'static str,
        /// The seconds after which a retry may find room: a caller's
        /// retry hint.
        retry_after_secs: u16,
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
    /// For a single request, a rule that answers each worker one way
    /// within a call is not the cause: a request whose rule takes no
    /// worker of its list is shed under the rule's name (`admits`), and
    /// one whose rule takes some is decided among them. Under a profile
    /// that filters by health a list with no ready worker is shed by that
    /// filter. So this answer is left for a worker that stopped being
    /// ready between a decision's read of it and the send, for a profile
    /// that lists no such filter, and for a rule that refused as a pick a
    /// worker it took when it was asked of the list.
    ///
    /// For a pair, the answer does not say whether the rule refused the
    /// picks or they were not ready: a caller that answers the two apart
    /// reads its lists itself.
    ///
    /// An encode request has no worker also where each decision of its
    /// call named an encoder that could not be sent to.
    NoWorkers,
    /// The engine refused the call's last decision, for the reason it
    /// carries: it made none in its place. Where the call decided again,
    /// the decisions it gave up stand behind this one. A caller answers
    /// that it cannot place the request now.
    Refused(DecisionRefused),
}

/// Whether `worker` can be sent to now: ready, and let through by its
/// circuit breaker, on one read of its state. What a caller alone knows of
/// a worker at the instant of a send. The overload veto and the liveness
/// veto are the decision's own, and a pick that lifted one stands.
pub(super) fn ready(worker: &dyn Worker) -> bool {
    let state = worker.routing_state();
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

/// What a request that carries encode items over other lists than an
/// encode request's three is declined under: no list of it holds an encoder.
const ENCODE_ITEMS: &str = "encode-items";

/// What a request is shed under where its route's rule takes no worker of
/// its list: the rule's own name on a request ([`PlaceRequest::admits`]).
const ADMITS: &str = "admits";

/// What a request is shed under where its call's earlier decisions left
/// none of its budget ([`PlaceRequest::budget_micros`]).
const BUDGET: &str = "budget";

/// The refusal of a request this call does not place, by name: the shed a
/// decision answers for what no stage of it runs.
const fn declined(stated: &'static str) -> Unplaced {
    Unplaced::Shed {
        reason: ShedReason::PolicyDeclined,
        emptied_by: stated,
        retry_after_secs: 1,
    }
}

/// The engine's one decision for `made` ([`PolicyEngine::decide_at`]): the
/// place it answers, or `None` with its whole decision written into
/// `whole`, the room its caller keeps ([`Loaded::decide_into`]); and the
/// instant it was dated at. A place dates nothing: no clock is read for
/// it. A single's ask and a pair's decide here, three lists with no item
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
    let alone = engine.decide_into(made, clock, whole)?;
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
        for id in failed.into_iter().flatten().flatten() {
            if !missed.contains(&id) {
                missed.push(id);
            }
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
    engine: &'static PolicyEngine,
    decided: DecideOutcome,
    later: bool,
    now: Instant,
) {
    if decided.lease.is_some() {
        DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now)
            .released(ReleaseReason::CommitFailed, now);
    }
}

/// Holds the later asks of `call` to the workers of the list its route's
/// rule takes. The rule is asked of every worker of `list`, ready or
/// not, in the list's order, and the places it takes are kept, ascending,
/// each once ([`Call::admitted`]). It is asked of the list once a call:
/// behind the first decision none of whose targets can be sent to, where
/// it refused one that was ready, and while the call holds no decision. A
/// rule that panics here leaves no lease behind. Out of line: a request
/// whose picks the rule admits runs none of it. A single request's ask
/// alone comes here: a pair's later decisions are made over its two lists
/// whole.
///
/// A worker that is not ready is asked too. A rule that takes no worker
/// is so told from a rule whose workers are not ready: the first is the
/// rule's own answer, and the second is the decision's, by the filter
/// that left nobody.
///
/// The ask that finds the rule's workers is followed by one decision
/// among them at least. Where it is the call's last ([`ASKS`]), the call
/// is given one more: no ask behind that one finds them again, so a call
/// makes four decisions at most.
///
/// # Errors
///
/// The shed of a request whose rule takes no worker of the list
/// ([`ShedReason::NoEndpoints`], under `admits`): no further decision is
/// made.
#[cold]
#[inline(never)]
fn hold_to_rule(
    request: &PlaceRequest<'_>,
    list: Option<LentList<'_>>,
    call: &mut Call,
) -> Result<(), Unplaced> {
    if call.admitted.is_some() {
        return Ok(());
    }
    let workers = list.map_or(&[][..], |list| &list.workers[..]);
    let places: Vec<u32> = (0..)
        .zip(workers)
        .filter_map(|(place, worker)| admitted(request, worker.as_ref()).then_some(place))
        .collect();
    if places.is_empty() {
        return Err(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: ADMITS,
            retry_after_secs: 1,
        });
    }
    call.admitted = Some(places);
    // One ask is left behind this one, whichever of the call's it is.
    call.spent = call.spent.min(ASKS - 2);
    Ok(())
}

/// The asks of one call: the first, and two more that leave out the
/// workers the call could not send to. An ask is one decision of the call
/// and the walk of its targets. A single request's call whose third ask
/// found the workers its route's rule takes is given a fourth among them
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
    /// The prompt's block hashes, made at the call's first making: a prompt
    /// is hashed once a call, however many asks the call takes.
    hashes: OnceCell<PromptHashes>,
    /// The asks of the call that took no target.
    spent: usize,
    /// The places of the list whose worker the route's rule takes
    /// ([`PlaceRequest::admits`]), ascending, each once, from the first
    /// decision none of whose targets could be sent to where the rule
    /// refused one that was ready ([`hold_to_rule`]). The call's later
    /// asks are held to them ([`Constraints::places`]). `None` ahead of
    /// that decision, and for a request with no rule. For a request that
    /// names its worker they are that worker's one place, from ahead of
    /// the call's first ask ([`target`]), and the rule is not asked of
    /// the list again.
    admitted: Option<Vec<u32>>,
    /// Whether the call's requests are made with no routing value and no
    /// client's value: a request that names its worker over a list none of
    /// whose workers can be sent to ([`target`]). The worker a request
    /// names outranks its routing value, so that value keeps no worker
    /// over a veto.
    unkeyed: bool,
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
            }),
            left => Ok(left),
        }
    }

    /// What an ask answers where no target of its decision can be sent to:
    /// the answer of `ask`, the call's next ask, while the call has an ask
    /// left, and no worker for the request ([`Unplaced::NoWorkers`]) behind
    /// its last: [`ASKS`] asks a call, and one more where the last of them
    /// found the workers the route's rule takes ([`hold_to_rule`]). The
    /// next ask answers in this one's place, and a request whose first ask
    /// takes a target pays for no other.
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
    engine: &'static PolicyEngine,
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
/// Where the rule refused one of them that was ready, the call's later
/// asks are held to the workers the rule takes ([`hold_to_rule`]).
fn single<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    list: Option<LentList<'_>>,
    call: &mut Call,
) -> Result<Placement<'a>, Unplaced> {
    let mut made = Made {
        budget_micros: call.left(request)?,
        ..Made::asked(engine.object, request, call)
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
        // found it eligible: the worker is read no second time, and its
        // route's rule is asked of it.
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
                hold_to_rule(request, list, call)?;
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
            // Whether the rule refused a target that was ready: inside
            // the decision, or here.
            let refused = Cell::new(sent.refused);
            let taken = walk(engine, decided, later, now, &mut call.missed, |target| {
                let worker = at(target.position);
                // The lookup's answer stands for a target it asked.
                let sendable = sent.answer(target.position).unwrap_or_else(|| {
                    worker.is_some_and(|worker| {
                        ready(worker.as_ref()) && {
                            let admitted = admitted(request, worker.as_ref());
                            refused.set(refused.get() || !admitted);
                            admitted
                        }
                    })
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
                    // The walk gave the decision up: the rule is asked of
                    // the list while the call holds none.
                    if refused.get() {
                        hold_to_rule(request, list, call)?;
                    }
                    call.again(|call| single(engine, request, list, call))
                }
                Err(unplaced) => Err(unplaced),
            }
        }
    }
}

impl PolicyEngine {
    /// Places one request: decides it, and answers the worker it goes to
    /// as a [`Placement`] its caller keeps with the request.
    ///
    /// The engine makes the request of what `request` states, decides it,
    /// and takes the first target of the decision that can be sent to:
    /// ready, let through by its circuit breaker, and admitted by its
    /// route's own rule ([`PlaceRequest::admits`]). A request a place
    /// answers is placed on the worker at that place, which the pick itself
    /// read: the rule alone is asked of it. The placement names the
    /// request's routing value where the document derived one that leaves
    /// in the clear, and none for a request a place answers: no fact of
    /// such a request is made.
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
    /// none of whose picks can be sent to, where the rule refused one that
    /// was ready, the decision is given up and the rule is asked of every
    /// worker of the list, ready or not, once a call. The call's later
    /// decisions are held to the places of the workers it takes
    /// ([`Constraints::places`]): the one such worker of a long list is
    /// found, one over its overload thresholds is answered by the
    /// binding's rule for a list wholly over them, and one an earlier
    /// attempt tried is handed again where no other is left. The call
    /// makes one decision among them at least, a fourth where its third
    /// found them. A request whose rule takes no worker of the list is
    /// shed with no endpoint, under `admits`. One whose rule takes workers
    /// none of which is ready is answered by the decision held to them,
    /// which names the filter that left nobody.
    ///
    /// A decision that is given up has written what its stages write as
    /// they decide, and its release takes none of that back: the tokens a
    /// `bucket` profile books on a worker a prefill decision listed, a new
    /// session's claim of its worker, the move of a session's pin at its
    /// cap. A decision whose later target is taken has written for its
    /// first target all the same.
    ///
    /// Where a prompt went is written otherwise, in the record of sent
    /// prefixes a `cache_aware` profile keeps and in a token tree the
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
    /// decision: three decisions a call at most, each given the request's
    /// whole budget. The rule is asked of a pair's picks and never of its
    /// lists, so a pair can be answered no worker while its lists hold
    /// workers the rule takes: where three decisions' picks are all
    /// refused, where such a worker is overloaded and a worker the rule
    /// refuses is not, and where an earlier attempt of the request tried
    /// it.
    /// Which workers pair is the caller's word, in the classes its lists
    /// carry: no pair is tested here again. Prefills with an open partner
    /// that are all at the caller's limit answer
    /// [`Unplaced::PrefillsFull`], on which the caller waits for room and
    /// places the request again.
    ///
    /// A request with an encode request's three lists is decided over the
    /// three ([`PolicyEngine::decide_encode_at`]), with the items it
    /// carries. Its pair's targets are tried as a pair's are, and the
    /// placement names each item's encoder beside the pair
    /// ([`Placement::encoders`]). An item has one encoder: where it cannot
    /// be sent to, the decision is given up and the engine decides again
    /// with that encoder excluded, and with each side of the pair that
    /// could not be sent to either. Its call is a pair's in what follows a
    /// decision given up: three decisions at most, each given the
    /// request's whole budget, and the rule asked of no list. An encoder
    /// is read by its state alone: the route's rule is asked of the pair's
    /// two workers. With no item the three lists are placed as their pair.
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
    /// not text names no worker. A pair's request and an encode request
    /// are not read for a target: they are placed by the rest of the
    /// document.
    ///
    /// Declined by name ([`ShedReason::PolicyDeclined`]): a request that
    /// carries encode items over other lists than an encode request's
    /// three (`encode-items`). A pair that states a demand is shed with no
    /// worker for it ([`ShedReason::NoEndpoints`], under `wires`) where
    /// each of its lists holds a worker: a pair's decision narrows neither
    /// list by a demand.
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
    /// engine's own task keeps at hand ([`PolicyEngine::run`]), of the
    /// channel its host handed ([`PolicyEngine::set_loads`]), and no report
    /// until its host has done both.
    ///
    /// The overload test is the engine's. The decision leaves out a worker
    /// over its overload thresholds (the `not-overloaded` filter), and the
    /// binding's own rule answers a list whose every worker is over them.
    /// This call asks it of no worker again, so a caller that asks this
    /// call for every request keeps no overload test of its own.
    ///
    /// A program that asks this call states no worker id of its own
    /// ([`FixedFacts::id`](crate::engine::FixedFacts::id)): a placement
    /// names its worker by the fingerprint of the worker's address
    /// ([`Placement::id`]), the id a next attempt states as tried. A debug
    /// build fails a request placed on a worker its list's records name
    /// otherwise.
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
        let own = self.loaded();
        match request.lists {
            WorkerLists::Pair(ref lists) => pair::place(own, request, lists, &mut Call::default()),
            // Whether a binding names a target source was decided when the
            // document was applied: under one that names none, a request
            // pays this test and no read of a header.
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
mod tests;
