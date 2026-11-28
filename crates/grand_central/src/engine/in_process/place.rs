//! One call for one request ([`InProcess::place`]): the request's facts made
//! for a decision, and the worker the request goes to. What a caller states
//! of the request ([`PlaceRequest`]) and the request as the engine keeps it
//! ([`Made`]) are declared beside this module ([`super::request`]); its
//! implementation of [`RequestMaker`], by which a decision asks for those
//! facts, is written here.
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

mod decode_again;
mod encode;
mod pair;
pub(crate) mod target;

use std::{
    any::Any,
    cell::Cell,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    sync::Arc,
    time::Instant,
};

use smg_worker::identity::basic::Worker;

// Named by the doc links of this module alone.
#[cfg(doc)]
use super::request::Prompt;
use super::{
    encode::{splits, EncodeLists},
    request::{leads_a_prompt, ready, Call, Made, PlaceRequest, Sent},
    InProcess, Loaded,
};
use crate::{
    document::spec::{AppliedSpec, DecodePick},
    engine::{
        decision::{
            Decision, PairSide, Pick, ShedReason, Target, ADMITS, BUDGET, ENCODE_ITEMS, MAX_TARGETS,
        },
        decision_engine::{
            AlonePick, DecideOutcome, DecisionRefused, Plain, Request, RequestMaker, TextKey,
            TokenKey, WorkerLists,
        },
        list_cell::LentList,
        prompt_hashes::{prompt_bytes_of, prompt_chars_of, prompt_tokens_of, PromptHashes},
        records::worker_id,
        view::{Constraints, Header, Identity, RequestFacts, WireModes, WorkerId},
    },
    fresh_seed,
    policies::consistent_hashing::LentRing,
    store::GatewayId,
    tracking::{lease::ReleaseReason, DecisionGuard, Placement},
};

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
        /// A caller's refusal line names it.
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
    /// the decisions it gave up stand behind this one. A caller answers
    /// that it cannot place the request now.
    Refused(DecisionRefused),
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
