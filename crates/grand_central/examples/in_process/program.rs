//! The program: what it holds, how it starts the engine, and the one way it
//! serves a request.
//!
//! # What it holds
//!
//! It holds what a program that routes requests holds already, in the worker
//! crate's own types. That is the worker registry, and the program's value of
//! each membership generation ([`Lists`]).
//!
//! That value keeps the members the registry published. At its first lending
//! it cuts them into one list for each kind of worker the program routes to,
//! each with an empty cell of its own ([`PublishedList`]). The kinds are the
//! single workers, the prefills and the decodes of a pair, and the encoders of
//! an encode request.
//!
//! The program writes routing settings, as a configuration file would, and no
//! document. Its document is the one the settings mean
//! ([`RoutingSettings::policy_document`]). The settings' mode says which kinds
//! of worker the program routes to ([`Plan`]):
//!
//! - single workers ([`settings`])
//! - pairs of a prefill worker and a decode worker ([`pair_settings`])
//! - such pairs behind encoders that take a request's items
//!   ([`encode_settings`])
//!
//! # How it starts the engine
//!
//! The engine takes the registry and the load monitor's channel once, at its
//! start ([`InProcess::start`]). It reads both in place, so the program tells
//! it nothing of a worker that joins or leaves.
//!
//! The program hands its lists by model beside the start ([`PublishedLists`]).
//! The engine's task fills the cells of a model's newest lists behind each
//! membership change.
//!
//! # What a request lends and states
//!
//! Of the engine the program asks what [`Engine`] lists, and nothing else: one
//! call, which serves every request. With it the program lends and states:
//!
//! - the lists its plan names, each with its cell ([`Lists`])
//! - its own hash ring of the generation ([`Ring`]): where the document places
//!   a routing value by a ring, the value goes where this ring puts it
//! - whether its route sends a pair's decode behind the prefill's answer,
//!   which no setting holds ([`Program::in_turn`])
//! - a retry's wire, so that a retry stays on the wire of its first worker
//!   ([`wire`])
//! - the route's own rule for a retry, by what it alone knows of the request
//!   and a worker ([`OWN_ANSWER`]). A worker that answered the request itself
//!   takes no further attempt of it.
//! - the prompt's token ids, and its entry in the token tree the program lends
//!   ([`Asked::tokens`], [`Trees`])
//!
//! # What it leaves out
//!
//! - It keeps no circuit breaker's count and chooses no rank inside a worker.
//!   Both are a route's own, behind the engine's answer.
//! - It keeps no limit of its own on a pair's workers ([`NoLimit`]).
//! - An encode request carries items and no more. Beyond its wire the program
//!   states no fact of a worker, so it serves no image, no other demand and no
//!   item handed as a reference.

use std::{
    borrow::Cow,
    future::Future,
    sync::{Arc, OnceLock},
};

use http::HeaderMap;
use kv_index::{ModelTrees, TenantId, TokenTree};
use smg_grand_central::{
    document::{AppliedSpec, ModeSettings, PolicySettings, RoutingSettings},
    engine::{
        CacheIndex, CacheIndexes, Demands, EncodeClasses, EncodeItems, EncodeLists, EncodeMember,
        EncoderFacts, FixedFacts, HeaderRead, Identity, InProcess, LentList, ModelLists,
        PairAdmission, PairLists, PartnerClasses, Phase, PlaceRequest, Prompt, PromptEntry,
        PublishedList, TokenKey, Unplaced, WorkerData, WorkerId, WorkerLists, WorkerRule,
    },
    policies::consistent_hashing::LentRing,
    store::{EncodeHandoff, EncodeOffer},
    tracking::{Outcome, Placement, Side, TransportClass},
};
use smg_worker::{
    identity::basic::Worker,
    load::{inflight::WorkerLoadGuard, state::LoadReceiver},
    registry::{Generation, WorkerRegistry},
    worker::WorkerType,
};

use crate::ring::Ring;

/// The attempts one client request gets at most: the first and two more.
pub const ATTEMPTS: usize = 3;

/// The status the program answers a request the engine named no worker
/// for, whatever the reason, where no worker's own answer stands
/// ([`OWN_ANSWER`]).
pub const NO_WORKER: u16 = 503;

/// The statuses a send is asked again for: the set a gateway's retry layer
/// asks again for.
pub const ASKED_AGAIN: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// The status of an answer a worker gave a request itself: it took the
/// request, ran it and failed on it. The same request sent there again
/// costs the same work for the same answer, so that worker takes no further
/// attempt of it. The other statuses of [`ASKED_AGAIN`] say the request was
/// not served now, and the worker that gave one may be sent it again.
pub const OWN_ANSWER: u16 = 500;

/// What a send answers where its worker answered nothing: the worker was
/// not reached. No status is the worker's then, so no such send is a
/// worker's own answer.
pub const UNANSWERED: u16 = 0;

/// The status the client reads for a send no worker answered
/// ([`UNANSWERED`]). It is the program's own word, and a send is asked
/// again for it ([`ASKED_AGAIN`]).
pub const NOT_REACHED: u16 = 502;

/// What the engine sheds a request under where its route's rule takes no
/// worker of the request's list ([`Unplaced::Shed`]).
const RULED_OUT: &str = "admits";

/// What a send answered: the status of its worker's reply, or no reply
/// where no worker was reached. A send hands the program one number, a
/// status or [`UNANSWERED`], and one place tells the two apart ([`From`]):
/// past it the program holds no number that could be either.
#[derive(Clone, Copy)]
pub enum Reply {
    /// The worker answered this status.
    Status(u16),
    /// No worker answered: the send reached none.
    Unanswered,
}

impl From<u16> for Reply {
    fn from(answered: u16) -> Self {
        if answered == UNANSWERED {
            Self::Unanswered
        } else {
            Self::Status(answered)
        }
    }
}

impl Reply {
    /// The status the client reads for the send: the worker's, or the
    /// program's own where no worker answered ([`NOT_REACHED`]).
    const fn status(self) -> u16 {
        match self {
            Self::Status(status) => status,
            Self::Unanswered => NOT_REACHED,
        }
    }

    /// Whether the worker answered the request itself ([`OWN_ANSWER`]). It
    /// reads what the worker answered, never the status the client reads. So a
    /// send no worker answered is nobody's own answer, whatever the client
    /// reads for it.
    const fn own(self) -> bool {
        matches!(self, Self::Status(OWN_ANSWER))
    }

    /// How the send ended, as the engine hears it. A 408 and a 504 say the
    /// worker ran out of time, whichever side of 500 they stand on. A send
    /// no worker answered reached no worker.
    const fn class(self) -> TransportClass {
        match self {
            Self::Unanswered => TransportClass::ConnectFailed,
            Self::Status(..400) => TransportClass::Ok,
            Self::Status(408 | 504) => TransportClass::Timeout,
            Self::Status(400..500) => TransportClass::ClientError,
            Self::Status(_) => TransportClass::ServerError,
        }
    }
}

/// Notes in `own_answers` the engine of `worker` where `reply`, what its
/// send answered, is the worker's own answer to the request
/// ([`Reply::own`]). An engine is known by its base address, as a gateway
/// knows it: the answer of one rank stands for the engine's other ranks.
fn note(own_answers: &mut Vec<String>, worker: &dyn Worker, reply: Reply) {
    if reply.own() {
        own_answers.push(worker.base_url().to_owned());
    }
}

/// The runtime the program says each of its workers runs, for the encode
/// handoff ([`Lists::cut`]). It is opaque text: the rule compares it and
/// reads nothing in it.
const RUNTIME: &str = "example";

/// The number the program knows a worker's wire by: the runtime it runs and
/// the transport it is dialed over, each by its place in the worker crate's
/// own list. It is never zero, which states no wire.
///
/// The numbering is the program's own. The engine compares two numbers and
/// reads nothing else of one ([`DemandFacts::runtime`]), so each host numbers
/// its wires as it likes.
///
/// A retry demands the wire of its first worker ([`Demands::runtime`]), since
/// the request a route keeps for a retry was built for one runtime and
/// transport. A host whose retries are bound to no wire states no demand.
///
/// [`DemandFacts::runtime`]: smg_grand_central::engine::DemandFacts::runtime
fn wire(worker: &dyn Worker) -> u32 {
    let runtime = worker.metadata().spec.runtime_type as u32;
    let transport = *worker.connection_mode() as u32;
    ((runtime + 1) << 8) | (transport + 1)
}

/// What the program states of `worker` beside its object: its wire, and
/// nothing else.
fn facts_of(worker: &Arc<dyn Worker>) -> FixedFacts {
    let mut facts = FixedFacts::default();
    facts.demands.runtime = wire(worker.as_ref());
    facts
}

/// The routing settings of a gateway that sets `policy` and nothing else:
/// each other setting at the value a configuration that leaves it out gets
/// ([`RoutingSettings::default`]). Such a gateway routes to single workers.
pub fn settings(policy: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        policy,
        ..RoutingSettings::default()
    }
}

/// The routing settings of a gateway that routes to pairs of a prefill
/// worker and a decode worker, both sides under `policy`, and sets nothing
/// else ([`settings`]).
pub fn pair_settings(policy: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: None,
            decode_policy: None,
        },
        ..settings(policy)
    }
}

/// The routing settings of a gateway that routes to such pairs behind
/// encoders, and sets nothing else ([`settings`]). Every side is under
/// `policy` but the encode side, which keeps the policy a configuration that
/// names none gets.
pub fn encode_settings(policy: PolicySettings) -> RoutingSettings {
    RoutingSettings {
        mode: ModeSettings::EncodePrefillDecode {
            encode_policy: None,
            prefill_policy: None,
            decode_policy: None,
        },
        ..settings(policy)
    }
}

/// The kinds of worker the program routes to: what each of its requests
/// lends the engine. It is the settings' word, as a gateway's mode is, and
/// is not read off a model's workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Single workers: a request lends their one list.
    Single,
    /// Pairs of a prefill worker and a decode worker: a request lends the
    /// two lists.
    Pair,
    /// Such pairs behind encoders: a request lends the three lists, and
    /// one that carries items states them.
    Encode,
}

impl Plan {
    /// The plan `settings` mean by their mode. `None` for a mode this
    /// program does not serve.
    pub fn of(settings: &RoutingSettings) -> Option<Self> {
        match settings.mode {
            ModeSettings::Regular => Some(Self::Single),
            ModeSettings::PrefillDecode { .. } => Some(Self::Pair),
            ModeSettings::EncodePrefillDecode { .. } => Some(Self::Encode),
            _ => None,
        }
    }
}

/// The program's value of one membership generation of a model. It holds the
/// members the registry published. Cut from them at the value's first lending,
/// it holds the list of each kind of worker, with the cell the program keeps
/// beside it ([`Cut`]).
///
/// Of a worker the program states its wire beside its object ([`wire`]) and no
/// other fact. Of an encoder it states none.
#[derive(Debug)]
pub struct Lists {
    members: Arc<[Arc<dyn Worker>]>,
    cut: OnceLock<Cut>,
}

/// A generation's members by the kind of worker each is:
///
/// - each kind's list in the registry's order, with an empty cell beside it
/// - the wire of each worker a retry can go to ([`facts_of`])
/// - the pair classes of the prefills and the decodes
/// - what the encode side reads of the encoders and of the prefills
/// - the generation's ring, made of every member's address
///
/// Which workers pair is the program's own word: here every prefill pairs with
/// every decode.
///
/// Which encoder hands an item to which prefill is derived by the crate's one
/// rule ([`EncodeClasses::derive`]), from what the program says each worker
/// states. Here that is a rendezvous under one runtime, so every encoder hands
/// to every prefill.
#[derive(Debug)]
struct Cut {
    single: PublishedList,
    prefill: PublishedList,
    decode: PublishedList,
    encode: PublishedList,
    classes: PartnerClasses,
    /// What the encode side reads of each encoder, by its place in the
    /// encoders' list.
    encoder_facts: Box<[EncoderFacts]>,
    /// The encode class each prefill names, by its place in the prefills'
    /// list.
    prefill_classes: Box<[u32]>,
    /// The program's ring of the generation's workers, of every kind: one
    /// ring serves every list.
    ring: Ring,
}

impl Generation for Lists {
    /// Made under the registry's guard: no worker is read and no lock
    /// taken. The lists are cut behind it ([`Lists::cut`]).
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            members,
            cut: OnceLock::new(),
        }
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.members
    }
}

impl Lists {
    /// The lists, cut at the first ask. Each member is read once: for the kind
    /// of worker it is, for the address the ring holds it by, and for its wire
    /// unless it is an encoder.
    ///
    /// Asks that meet there wait for the one that cuts. The engine's task asks
    /// behind each membership change, so a request cuts only where it comes
    /// ahead of the task.
    fn cut(&self) -> &Cut {
        self.cut.get_or_init(|| {
            let of = |kind: WorkerType| -> Arc<[Arc<dyn Worker>]> {
                let members = self.members.iter();
                let of_kind = members.filter(|worker| *worker.worker_type() == kind);
                of_kind.cloned().collect()
            };
            let (prefills, decodes) = (of(WorkerType::Prefill), of(WorkerType::Decode));
            let encoders = of(WorkerType::Encode);
            // Each prefill states every place of the decode list.
            let every: Vec<u32> = (0u32..).take(decodes.len()).collect();
            let partners = prefills.iter().map(|_| every.as_slice());
            // What each worker states of the encode handoff: a rendezvous
            // under the one runtime.
            let mut offer = EncodeOffer::default();
            offer.handoff = EncodeHandoff::Rendezvous;
            let member = EncodeMember::of(RUNTIME, &offer);
            let ends = EncodeClasses::derive(
                encoders.iter().map(|_| member),
                prefills.iter().map(|_| member),
            );
            let classed = (0..prefills.len()).map(|place| ends.prefill_class(place));
            Cut {
                classes: PartnerClasses::number(decodes.len(), partners),
                encoder_facts: ends.encoder_facts(encoders.iter().map(|_| &offer)),
                prefill_classes: classed.collect(),
                ring: Ring::of(self.members.iter().map(|worker| worker.url())),
                single: PublishedList::with_facts(of(WorkerType::Regular), facts_of),
                prefill: PublishedList::with_facts(prefills, facts_of),
                decode: PublishedList::with_facts(decodes, facts_of),
                // A retry states no item, so no demand meets an encoder.
                encode: PublishedList::new(encoders),
            }
        })
    }

    /// The program's ring of this generation: every worker of the model
    /// stands on it, whatever its kind.
    pub fn ring(&self) -> &Ring {
        &self.cut().ring
    }

    /// The single workers' list lent with its cell and its workers' wires.
    fn lent(&self) -> LentList<'_> {
        self.cut().single.lent()
    }

    /// A pair's two lists lent, each with its cell, and their pair classes:
    /// every place of each list, under no veto of the program's own and no
    /// limit ([`NoLimit`]).
    fn pair(&self) -> PairLists<'_> {
        let cut = self.cut();
        PairLists {
            prefill: cut.prefill.lent(),
            decode: cut.decode.lent(),
            classes: &cut.classes,
            admission: &NoLimit,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        }
    }

    /// An encode request's three lists lent: the pair's two as a pair lends
    /// them, and the encoders with their cell. With them goes what the encode
    /// side reads of the encoders and of the prefills. The program adds no
    /// veto of its own.
    fn encode(&self) -> EncodeLists<'_> {
        let cut = self.cut();
        EncodeLists {
            pair: self.pair(),
            encoders: cut.encode.lent(),
            encoders_places: None,
            encoders_stalled: &[],
            encoders_held: &[],
            facts: &cut.encoder_facts,
            prefill_classes: &cut.prefill_classes,
        }
    }
}

/// The program's reads of its own admission for a pair. It keeps no limit
/// on a prefill and no rooms on a decode, so every prefill has capacity and
/// every decode a room. A gateway answers from its prefill queue and its
/// decode rooms here.
#[derive(Debug)]
struct NoLimit;

impl PairAdmission for NoLimit {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// The program's lists by model, as it hands them to the engine beside its
/// start. For a model it answers the generation the registry holds now: each
/// list of it that holds a worker, lent as a request lends it.
///
/// The engine's task asks them behind each membership word of the registry,
/// and fills a cell it finds empty. So a list's first request reads a filled
/// cell where the task came first.
#[derive(Debug)]
struct PublishedLists(Arc<WorkerRegistry<Lists>>);

impl ModelLists for PublishedLists {
    fn lists_of(&self, model: &str, hand: &mut dyn FnMut(LentList<'_>)) {
        // The generation is taken out of the registry first: the engine
        // reads each worker of a list it fills, and no guard of the
        // registry stands meanwhile.
        let Some(lists) = self.0.model_generation(model) else {
            return;
        };
        let cut = lists.cut();
        for list in [&cut.single, &cut.prefill, &cut.decode, &cut.encode] {
            if !list.workers().is_empty() {
                hand(list.lent());
            }
        }
    }
}

/// The program's token trees of the prompts it sent, one a model. It lends
/// them to the engine at its start ([`WorkerData::cache_indexes`]).
///
/// A model's tree is made where the engine first asks for it, with the
/// document's block as its page. The engine asks under a document that reads
/// the prefixes sent to each worker, and under no other. It then decides a
/// single request inside the tree's match, and the same descent writes the
/// prompt for the worker the request goes to.
///
/// The engine calls the tree's match and nothing else. The bound and the
/// workers that leave are the program's ([`Program::trim`],
/// [`Program::leave`]).
///
/// A host that has such a tree lends that one, and makes none for the engine
/// alone. This program has no other use of a tree: it keeps one, in the index
/// crate's store, to show what a lent tree asks of its host.
///
/// The store is empty until the engine's first ask, so the program makes it
/// always. A host whose trees cost it a task of its own makes them only where
/// its document reads them. The document's needs say so
/// (`Needs::sent_prefixes`), and a gateway asks them before it makes its
/// trees.
///
/// The engine decides no request but a single one inside a lent tree, so the
/// program lends its tree at no other phase. It keeps no index of its workers'
/// cache events, since its workers read as publishing none. It keeps no tree
/// of prompts held as text, since it holds every prompt as token ids.
struct Trees(Arc<ModelTrees>);

impl CacheIndexes for Trees {
    fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
        None
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        None
    }

    fn token_tree(&self, model: &str, phase: Phase) -> Option<Arc<TokenTree>> {
        match phase {
            Phase::Single => Some(self.0.token_tree_or_new(model)),
            Phase::Prefill | Phase::Decode | Phase::Encode => None,
        }
    }
}

/// Every call the program makes of the engine once it is started: one.
///
/// [`Engine::place`] is the host contract's one call for a request: a
/// single one, a pair and an encode request alike. The start
/// ([`Program::start`]) is the one other place that names the engine. A
/// pair's decode asked again is asked of the placement that call answered,
/// where it says that its decode still stands chosen
/// ([`Placement::decode_chosen`], [`Placement::decode_again`]): neither
/// names the engine.
struct Engine(&'static InProcess);

impl Engine {
    /// The one call for a request.
    fn place<'a>(&self, request: &PlaceRequest<'a>) -> Result<Placement<'a>, Unplaced> {
        self.0.place(request)
    }
}

/// What one attempt of a request states beyond what its client asked:
///
/// - its number
/// - the workers the request's failed attempts named
/// - the wire the request stays on ([`wire`])
/// - the route's own rule for a worker ([`Program::serve`])
///
/// A pair's failed attempt names both its workers. It names its decode too
/// where the prefill failed ahead of it and the decode was sent nothing.
///
/// A first attempt states the default: the number `0`, no worker tried, `0`
/// for no wire, and no rule. An attempt states no rule until a worker answered
/// the request itself.
#[derive(Clone, Copy, Default)]
pub struct Stated<'a> {
    number: u8,
    tried: &'a [WorkerId],
    wire: u32,
    admits: Option<WorkerRule<'a>>,
}

/// What a client's request states. It prints nothing of itself: a client's
/// tokens reach no log line.
#[derive(Clone, Copy)]
pub struct Asked<'a> {
    /// The model the request is for.
    pub model: &'a str,
    /// The request's headers, where it carries any.
    pub headers: Option<&'a HeaderMap>,
    /// The keys of the request's encode items, its caller's content hash of
    /// each. Empty for a request with none. A program that routes to no
    /// encoder reads none.
    pub items: &'a [Identity<'a>],
    /// The prompt's token ids, where the program tokenized it. `None` for a
    /// request it read no prompt of: such a request states no prompt and
    /// names no entry in the program's tree.
    pub tokens: Option<&'a [u32]>,
}

/// A request's entry in the program's token tree ([`Trees`]): its prompt's
/// token ids as they are. The program's workers share one cache partition,
/// so the entry holds no marker and the prompt is hashed under the plain
/// seeds. A host that parts its workers' caches hands the ids behind the
/// partition's marker, in whole pages, and states the partition's seeds
/// with the prompt.
impl PromptEntry for Asked<'_> {
    fn token_key(&self, _: usize) -> Option<TokenKey<'_>> {
        Some(TokenKey::of(Cow::Borrowed(self.tokens?), 0))
    }
}

/// One attempt of a request on its way: the placement, and the load guard of
/// each worker the request goes to. A pair's decode has a guard as its prefill
/// has. A route keeps them to the end of the response, streamed or not.
///
/// A route that sends a pair in turn lets the prefill's guard go once the
/// prefill answered ([`Self::prefill_ended`]). That worker holds the request
/// no more while the decode runs.
///
/// A pair's decode is counted from the placement, whether it is sent at once
/// or behind its prefill's answer. A gateway makes that guard behind its
/// selection too. Where a second ask names another decode, the guard moves
/// there ([`Self::follow_decode`]).
///
/// No guard of an encoder stands here. An item is counted on its encoder while
/// its own send runs ([`Program::serve`]).
pub struct InFlight {
    /// The placement, kept: it owns all it holds.
    pub placement: Placement<'static>,
    /// The load guard of the worker the request goes to first: a single
    /// request's one worker, or a pair's prefill.
    first: Option<WorkerLoadGuard>,
    /// The load guard of a pair's decode.
    decode: Option<WorkerLoadGuard>,
}

impl InFlight {
    /// The sides of the attempt, in the order they are sent: the one side
    /// of a single request, or a pair's prefill and then its decode.
    pub fn sides(&self) -> &'static [Side] {
        if self.placement.decode.is_some() {
            &[Side::Prefill, Side::Decode]
        } else {
            &[Side::Single]
        }
    }

    /// The worker `side` of the attempt goes to.
    pub fn worker(&self, side: Side) -> &Arc<dyn Worker> {
        match (side, &self.placement.decode) {
            (Side::Decode, Some(decode)) => decode,
            _ => &self.placement.worker,
        }
    }

    /// Says the attempt sent: its one side, or both sides of a pair in one
    /// word, ahead of the first send.
    pub fn sent(&mut self) {
        for &side in self.sides() {
            self.placement.sent(side);
        }
    }

    /// Moves the decode's load guard to the decode the placement names
    /// now, under the routing key it held: behind a second ask that took
    /// another decode. The request is counted there before the first
    /// decode counts it no more. A guard that counts that decode already
    /// stays as it is.
    fn follow_decode(&mut self) {
        let (Some(decode), Some(guard)) = (&self.placement.decode, &self.decode) else {
            return;
        };
        if !guard.counts(decode) {
            self.decode = Some(guard.onto(Arc::clone(decode)));
        }
    }

    /// `side` of the attempt ended with `reply`, what its send answered:
    /// the engine hears the status the client reads for it and how the
    /// send ended ([`Reply::status`], [`Reply::class`]). A caller hands a
    /// reply, or the number its send answered ([`From`]).
    pub fn ended(&mut self, side: Side, reply: impl Into<Reply>) {
        let reply = reply.into();
        let outcome = Outcome::reported(reply.status(), reply.class());
        self.placement.ended(side, outcome);
    }

    /// The prefill of a pair sent in turn ended with `reply`, what its
    /// send answered: its end is said, and its load guard goes with it. A
    /// prefill that answered holds the request no more while its decode
    /// runs, as a gateway's sequential dispatch frees the prefill once it
    /// is drained. A pair sent at once keeps both guards to the end of the
    /// response.
    ///
    /// The guard goes where the end is said, and this program's one
    /// placement says that end behind its second ask. So the prefill still
    /// counts the request while the engine decides the decode. A host that
    /// keeps a pair's two sides may let the prefill go ahead of that ask.
    fn prefill_ended(&mut self, reply: Reply) {
        self.ended(Side::Prefill, reply);
        drop(self.first.take());
    }
}

/// How the program answered one client request.
#[derive(Debug, PartialEq, Eq)]
pub struct Answer {
    /// The address of the worker each send went to, in order. A pair's
    /// attempt is two sends: its prefill, then its decode. A pair sent in
    /// turn whose prefill failed is one send. An encode request's first
    /// attempt sends each item to its encoder ahead of them.
    pub sent: Vec<String>,
    /// The status the client reads: the last attempt's. Where the engine
    /// named no worker it is [`NO_WORKER`], or a worker's own answer where
    /// the route's rule left the request no worker ([`OWN_ANSWER`]).
    pub status: u16,
    /// Why the engine named no worker, where it named none.
    pub unplaced: Option<Unplaced>,
}

/// A send: handed the worker a request goes to, it answers the status the
/// worker answered, or [`UNANSWERED`] where the worker answered nothing.
type Sends = Box<dyn Fn(&dyn Worker) -> u16 + Send + Sync>;

/// The program.
pub struct Program {
    /// The program's registry. The engine holds it too, since its start.
    registry: Arc<WorkerRegistry<Lists>>,
    engine: Engine,
    plan: Plan,
    sends: Sends,
    /// Whether the program's route sends a pair's decode behind its
    /// prefill's answer ([`Self::in_turn`]).
    in_turn: bool,
    /// The program's token trees ([`Trees`]). The engine holds them too,
    /// since its start.
    trees: Arc<ModelTrees>,
}

impl Program {
    /// The program's start: the engine's one call ([`InProcess::start`]), made
    /// over the program's own data.
    ///
    /// The engine is built from `spec` for the life of the process. It takes
    /// the program's registry, and its load monitor's channel where the
    /// program has a monitor. The program lends its token trees with them
    /// ([`Trees`]), a tree's page the block of `spec`, and keeps no index of
    /// cache events. Beside the start it hands the engine its lists by model
    /// ([`PublishedLists`]).
    ///
    /// `plan` is the kinds of worker the settings behind `spec` route to
    /// ([`Plan::of`]). `sends` are the program's sends.
    ///
    /// It answers the program with the engine's own task, which a program
    /// spawns on its runtime ahead of its first request. The task keeps the
    /// newest load snapshot at hand and runs the engine's upkeep. It also
    /// follows the registry: it tells the document's stages a worker that
    /// joined or left, and fills the cells of the model's lists that stand
    /// then.
    pub fn start(
        plan: Plan,
        spec: AppliedSpec,
        loads: Option<LoadReceiver>,
        sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
    ) -> (Self, impl Future<Output = ()> + Send + 'static) {
        let registry = Arc::new(WorkerRegistry::new());
        let handed: Arc<WorkerRegistry<Lists>> = Arc::clone(&registry);
        let trees = Arc::new(ModelTrees::new(spec.block_size() as usize));
        let lent: Arc<dyn CacheIndexes> = Arc::new(Trees(Arc::clone(&trees)));
        let data = WorkerData {
            registry: handed,
            loads,
            cache_indexes: Some(lent),
        };
        let (engine, task) = InProcess::start(spec, data);
        engine.set_lists(Arc::new(PublishedLists(Arc::clone(&registry))));
        let program = Self {
            registry,
            engine: Engine(engine),
            plan,
            sends: Box::new(sends),
            in_turn: false,
            trees,
        };
        (program, task)
    }

    /// The program with a route that sends a pair's decode behind its
    /// prefill's answer, as a route with sequential dispatch does. Each pair's
    /// request then says that its decode is asked again
    /// ([`PlaceRequest::decode_again`]). [`Self::serve`] asks it behind the
    /// prefill's answer, where the decode still stands chosen.
    ///
    /// It is the route's word and no setting: a program that sends both sides
    /// at once never says it. The decode stands chosen under a document that
    /// picks a pair's decode behind its prefill, and no setting says that
    /// either. A host whose route asks again writes it into the document its
    /// settings mean (`plans.pd-pair.decode`).
    #[must_use]
    pub fn in_turn(mut self) -> Self {
        self.in_turn = true;
        self
    }

    /// A worker joins: the registry publishes the next generation of its
    /// model, with empty cells. `false` where the registry holds a worker
    /// at its address already. The engine's task hears the join from the
    /// registry, and fills the cells of the model's lists that stand then.
    pub fn join(&self, worker: Arc<dyn Worker>) -> bool {
        self.registry.register(worker).is_some()
    }

    /// The worker at `address` leaves. `false` where the registry holds none
    /// there. The registry finds a worker by an address written without its
    /// scheme too.
    ///
    /// The engine's task hears the removal with the worker's own object, so
    /// the stages are told the worker under the address it states.
    ///
    /// The worker is taken out of the program's trees too, under that address:
    /// the engine writes a prompt for a worker by it. A tree keeps what it
    /// holds for a worker that is gone until its host takes it out. The size
    /// pass alone would keep it for good.
    ///
    /// A worker that joins at the address again starts with no prompt, unless
    /// a decision under way at the leave wrote one behind the purge, as in a
    /// gateway.
    pub fn leave(&self, address: &str) -> bool {
        let Some(left) = self.registry.remove_by_url(address) else {
            return false;
        };
        self.trees.remove_tenant(&TenantId::from(left.url()));
        true
    }

    /// The program's size pass: each model's tree is held to `tokens`
    /// tokens ([`ModelTrees::evict_token_trees`]). The bound is the host's,
    /// as the purge of a worker that left is: the engine calls a tree's
    /// match and nothing else. A program runs the pass on a timer of its
    /// own, with its settings' number. This one spawns nothing, and its
    /// builder calls the pass.
    pub fn trim(&self, tokens: usize) {
        self.trees.evict_token_trees(tokens, |_, _| {});
    }

    /// The program's token tree of `model`: `None` until the engine asked
    /// for it, and so under a document that reads no sent prefix.
    pub fn tree(&self, model: &str) -> Option<Arc<TokenTree>> {
        self.trees.token_tree(model)
    }

    /// The model's generation as the registry holds it now.
    pub fn published(&self, model: &str) -> Option<Arc<Lists>> {
        self.registry.model_generation(model)
    }

    /// The request of one attempt, as each call for that attempt states it. It
    /// states what the attempt states ([`Stated`]) of the request its client
    /// asked, over `lists`: the model's generation as its caller holds it at
    /// that call.
    ///
    /// - The attempt's number is stated, and not counted off the tried
    ///   workers: a failed attempt of a pair names two workers.
    /// - A retry states the wire it stays on as its demand
    ///   ([`Demands::runtime`]). A first attempt demands nothing.
    /// - An attempt behind a worker's own answer states the route's rule
    ///   ([`PlaceRequest::admits`]).
    /// - The request lends what the program's plan says: the single workers'
    ///   list, a pair's two, or an encode request's three. A model that has no
    ///   generation lends no list and states no item.
    /// - The items a request carries are stated with those three lists, at its
    ///   first attempt alone. A retry states none and is placed as the pair.
    /// - The generation's ring is lent with its lists ([`Lists::ring`]).
    /// - A program whose route sends a pair's decode behind its prefill's
    ///   answer says so here.
    /// - A request that holds its prompt's token ids states them and its entry
    ///   in the program's tree, at each attempt alike ([`Asked::tokens`]).
    fn request<'a>(
        &self,
        asked: &'a Asked<'a>,
        lists: Option<&'a Lists>,
        stated: Stated<'a>,
    ) -> PlaceRequest<'a> {
        let lent = match (lists, self.plan) {
            (None, _) => WorkerLists::None,
            (Some(lists), Plan::Single) => WorkerLists::Single(lists.lent()),
            (Some(lists), Plan::Pair) => WorkerLists::Pair(lists.pair()),
            (Some(lists), Plan::Encode) => WorkerLists::Encode(lists.encode()),
        };
        let headers: &dyn HeaderRead = match asked.headers {
            Some(headers) => headers,
            None => &(),
        };
        // Items go with an encode request's three lists and with no other:
        // those lists with none stated are placed as their pair, and a model
        // that lends no list states none. They go out once a request, so the
        // first attempt alone states them. The type takes no literal outside
        // the library, so the keys are written over its default.
        let mut items = EncodeItems::default();
        items.keys = asked.items;
        let lends = matches!(lent, WorkerLists::Encode(_));
        let carries = stated.number == 0 && lends && !asked.items.is_empty();
        // The generation's ring goes with every request. The engine asks it
        // where the document's ring stage places by a routing value's text,
        // and for an encode request's items.
        let ring = lists.map(|lists| lists.ring() as &dyn LentRing);
        // A retry stays on its first worker's wire: the engine holds it to
        // the workers that state the same number, ahead of every stage.
        let demands = Demands {
            runtime: stated.wire,
            ..Demands::default()
        };
        // The prompt as the program holds it: its token ids, and the
        // request itself as its entry in the program's tree. The engine
        // asks the entry where a decision reads that tree, and of no other
        // request.
        let mut prompt = Prompt::NONE;
        if asked.tokens.is_some() {
            prompt.tokens = asked.tokens;
            prompt.entry = Some(asked);
        }
        PlaceRequest {
            prompt,
            headers,
            demands,
            encode: carries.then_some(items),
            attempt: stated.number,
            tried: stated.tried,
            decode_again: self.in_turn,
            ring,
            admits: stated.admits,
            ..PlaceRequest::of(asked.model, lent)
        }
    }

    /// One attempt placed. The engine is asked once, over the generation that
    /// stands, for the request of that attempt ([`Self::request`]), which
    /// states `stated`. A caller that keeps its own attempt states a first
    /// attempt's ([`Stated::default`]).
    ///
    /// The answer is the placement, kept with the load guard of each worker
    /// the request goes to, under the routing key the placement names. A
    /// pair's decode is counted from here, sent at once or later. An item's
    /// encoder takes its guard at the item's send, and no sooner
    /// ([`Self::serve`]).
    ///
    /// # Errors
    ///
    /// Why the engine named no worker.
    pub fn place(&self, asked: &Asked<'_>, stated: Stated<'_>) -> Result<InFlight, Unplaced> {
        let lists = self.published(asked.model);
        let request = self.request(asked, lists.as_deref(), stated);
        let placement = self.engine.place(&request)?;
        let key = placement.routing_key;
        let guard = WorkerLoadGuard::with_key(Arc::clone(&placement.worker), key);
        let decode = placement.decode.as_ref();
        let decode = decode.map(|decode| WorkerLoadGuard::with_key(Arc::clone(decode), key));
        Ok(InFlight {
            placement: placement.kept(),
            first: Some(guard),
            decode,
        })
    }

    /// One send of the program to `worker`: what it answered ([`Reply`]).
    fn send(&self, worker: &dyn Worker) -> Reply {
        (self.sends)(worker).into()
    }

    /// Serves one client request, in up to [`ATTEMPTS`] attempts. For each
    /// attempt it places the attempt, says it sent, sends each side of it and
    /// says each side's end. Where the attempt was answered a status a gateway
    /// asks again for ([`ASKED_AGAIN`]), it places the next attempt past the
    /// workers tried.
    ///
    /// Both sides of a pair sent at once go out, whatever the first answered.
    /// The attempt's status is its prefill's where that one failed, and its
    /// decode's otherwise. A route that sends a pair's decode behind its
    /// prefill's answer ([`Self::in_turn`]) sends the pair in turn
    /// ([`Self::sent_in_turn`]): a prefill that failed sends no decode.
    ///
    /// A route does more than this: it waits between two attempts, and a
    /// response can say that it is not to be asked again.
    ///
    /// # A retry's wire
    ///
    /// A retry stays on the wire of the worker the first attempt went to
    /// ([`wire`]): each later attempt states that number as its demand. A wire
    /// that holds one worker keeps it for the retry, and a wire that holds
    /// none any more leaves the retry no endpoint.
    ///
    /// # The route's own rule
    ///
    /// A worker that answered the request itself ([`OWN_ANSWER`]) takes no
    /// further attempt of it. From then on each attempt states the route's own
    /// rule ([`PlaceRequest::admits`]). The rule refuses the engines that
    /// answered so, and reads the program's list of them alone.
    ///
    /// - A send no worker answered refuses nobody ([`UNANSWERED`]).
    /// - A tried worker steps aside while another is left. A worker the rule
    ///   refuses is sent nothing.
    /// - For a pair the rule is the plain one: the side whose own answer it
    ///   was is refused, and its partner is not.
    /// - An attempt reads one side's answer, the side that gave it its status.
    ///   Sent at once, that is the first that failed in the order of the
    ///   sends. Behind a prefill that failed, a decode's answer is not its own
    ///   word on the request.
    ///
    /// A gateway's pair route may go further, and refuse the partner where no
    /// pair is left without the worker that answered.
    ///
    /// Where the rule left the request no worker, the engine says so by its
    /// shed under `admits` ([`Unplaced::Shed`]). The worker's own answer is
    /// then the client's, as it is: nobody is left to give another. A shed for
    /// any other cause is answered [`NO_WORKER`], at a retry as at a first
    /// attempt.
    ///
    /// A gateway that keeps such a rule may end the request ahead of that
    /// call. This program asks the engine once more and reads its word.
    ///
    /// # An encode request
    ///
    /// The items go out once a request, with its first attempt: each to the
    /// encoder the placement names for it, ahead of the pair. That order is
    /// this program's own. A gateway whose workers state a rendezvous launches
    /// the items beside the pair.
    ///
    /// An item is counted on its encoder while its send runs, under no routing
    /// key, as a gateway's encode stage counts it. The key is the pair's, and
    /// an encoder that answered holds the request no more. Nothing is said of
    /// an encoder, since no lease stands for one.
    ///
    /// An encoder's failure is the request's answer. Neither side of the pair
    /// is sent, the placement's drop gives the pair up, and no attempt
    /// follows.
    ///
    /// A retry behind a failed side of the pair states no item and sends none.
    /// It is placed as the pair alone, with the pair's two workers as tried
    /// and no encoder, as a gateway's retry is.
    pub fn serve(&self, asked: &Asked<'_>) -> Answer {
        let (mut tried, mut sent, mut attempts) = (Vec::new(), Vec::new(), 0u8);
        // The wire the request stays on, from its first attempt's worker.
        let mut pinned = 0;
        // The engines that answered the request themselves.
        let mut own_answers = Vec::new();
        loop {
            // The route's own rule. It answers by reading alone, of a
            // worker in any state.
            let rule: WorkerRule<'_> =
                &|worker: &dyn Worker| !own_answers.iter().any(|noted| noted == worker.base_url());
            let stated = Stated {
                number: attempts,
                tried: &tried,
                wire: pinned,
                admits: (!own_answers.is_empty()).then_some(rule),
            };
            let placed = self.place(asked, stated);
            attempts += 1;
            let mut attempt = match placed {
                Ok(attempt) => attempt,
                Err(unplaced) => {
                    // The engine's word that the rule left no worker: the
                    // workers left answered the request themselves, and
                    // that answer is the client's.
                    let ruled_out = matches!(
                        unplaced,
                        Unplaced::Shed {
                            emptied_by: RULED_OUT,
                            ..
                        }
                    );
                    return Answer {
                        sent,
                        status: if ruled_out { OWN_ANSWER } else { NO_WORKER },
                        unplaced: Some(unplaced),
                    };
                }
            };
            // The first send that failed is the attempt's answer. Every item
            // goes out, whatever an earlier one's encoder answered.
            let mut status = 0;
            for encoder in attempt.placement.encoders.workers() {
                let reply = {
                    // Counted on its encoder until the send ends.
                    let _counted = WorkerLoadGuard::with_key(Arc::clone(encoder), None);
                    self.send(encoder.as_ref())
                };
                sent.push(encoder.url().to_owned());
                if status < 400 {
                    status = reply.status();
                }
            }
            if status >= 400 {
                // An item goes out once a request: its encoder's failure is
                // the request's answer, and neither side of the pair is sent.
                return Answer {
                    sent,
                    status,
                    unplaced: None,
                };
            }
            if self.in_turn && attempt.placement.decode.is_some() {
                let (side, reply) = self.sent_in_turn(asked, &mut attempt, stated, &mut sent);
                note(&mut own_answers, attempt.worker(side).as_ref(), reply);
                status = reply.status();
            } else {
                attempt.sent();
                for &side in attempt.sides() {
                    let worker = Arc::clone(attempt.worker(side));
                    let reply = self.send(worker.as_ref());
                    attempt.ended(side, reply);
                    sent.push(worker.url().to_owned());
                    // The first side that failed gives the attempt its
                    // status, and that side's answer alone is noted.
                    if status < 400 {
                        status = reply.status();
                        note(&mut own_answers, worker.as_ref(), reply);
                    }
                }
            }
            if !ASKED_AGAIN.contains(&status) || usize::from(attempts) == ATTEMPTS {
                return Answer {
                    sent,
                    status,
                    unplaced: None,
                };
            }
            // The request kept for a retry was built for the wire of the
            // worker it went to first: a pair's is its prefill's.
            if pinned == 0 {
                pinned = wire(attempt.placement.worker.as_ref());
            }
            tried.push(attempt.placement.id());
            tried.extend(attempt.placement.decode_id());
        }
    }

    /// A pair's attempt sent in turn, for the attempt that states `stated`:
    /// the prefill first, and behind its good answer the decode, which the
    /// engine is asked for again. The answer is the side that gave the attempt
    /// its status, and what that side's send answered: the prefill where it
    /// failed, else the decode.
    ///
    /// The order is the one the second ask stands on
    /// ([`Placement::decode_again`]):
    ///
    /// 1. The prefill's send is said and made.
    /// 2. Behind a good answer the program asks the placement whether its
    ///    decode still stands chosen ([`Placement::decode_chosen`]), as a
    ///    gateway does ahead of its second ask.
    /// 3. Where it does, the request is stated again, over the generation that
    ///    stands then, and the decode is asked ahead of the prefill's end.
    ///
    /// The decode's guard stands on the first decode across the second ask, so
    /// the request says so (`PlaceRequest::counted_on_decode`). The engine
    /// then reads that decode without this request.
    ///
    /// That ask ends one of two ways:
    ///
    /// - Where it takes a decode, the prefill's end is said, the guard follows
    ///   the decode, and the decode's send is said behind it.
    /// - Where it takes none, the first decision's decode is sent under the
    ///   guard it has, and its send is said ahead of the prefill's end.
    ///
    /// A decode that does not stand chosen was booked with its prefill. Either
    /// the document books both sides at once, or the engine books the request
    /// whole whatever its route says ([`PlaceRequest::decode_again`]). The
    /// engine would refuse a second ask with no decision made. So the program
    /// states no request again and sends that decode, in the same order.
    ///
    /// The engine hears the decode's dispatch where its send is said
    /// ([`Placement::sent`]), and not where its guard was made. The prefill's
    /// own guard goes where its end is said ([`InFlight::prefill_ended`]).
    ///
    /// A prefill that failed sends no decode. The decode's guard ends with the
    /// attempt, and a decode the engine had booked is given up by the
    /// placement's drop.
    fn sent_in_turn(
        &self,
        asked: &Asked<'_>,
        attempt: &mut InFlight,
        stated: Stated<'_>,
        sent: &mut Vec<String>,
    ) -> (Side, Reply) {
        attempt.placement.sent(Side::Prefill);
        let prefill = Arc::clone(attempt.worker(Side::Prefill));
        let reply = self.send(prefill.as_ref());
        sent.push(prefill.url().to_owned());
        if reply.status() >= 400 {
            attempt.prefill_ended(reply);
            return (Side::Prefill, reply);
        }
        // A decode booked with its prefill does not stand chosen: it is the
        // decode to send, and no request is stated again for it.
        let taken = attempt.placement.decode_chosen() && {
            let lists = self.published(asked.model);
            let request = PlaceRequest {
                counted_on_decode: true,
                ..self.request(asked, lists.as_deref(), stated)
            };
            attempt.placement.decode_again(&request).is_ok()
        };
        if taken {
            attempt.prefill_ended(reply);
            attempt.follow_decode();
            attempt.placement.sent(Side::Decode);
        } else {
            attempt.placement.sent(Side::Decode);
            attempt.prefill_ended(reply);
        }
        let decode = Arc::clone(attempt.worker(Side::Decode));
        let reply = self.send(decode.as_ref());
        attempt.ended(Side::Decode, reply);
        sent.push(decode.url().to_owned());
        (Side::Decode, reply)
    }
}
