//! The program: what it holds, how it starts the engine, and the one way it
//! serves a request.
//!
//! It holds what a program that routes requests holds already, in the
//! worker crate's own types: the worker registry, with the program's value
//! of each membership generation ([`Lists`]). That value keeps the members
//! the registry published. At its first lending it cuts them into one list
//! for each kind of worker the program routes to, each with an empty cell
//! of its own ([`PublishedList`]): the single workers, the prefills and the
//! decodes of a pair, and the encoders of an encode request.
//!
//! The engine takes the registry and the load monitor's channel once, at
//! its start ([`InProcess::start`]), and reads both in place: the program
//! tells it nothing of a worker that joins or leaves. The program hands its
//! lists by model beside the start ([`PublishedLists`]), and the engine's
//! task fills the cells of a model's newest lists behind each membership
//! change. Its sends are a function its builder gives ([`Program::start`]):
//! handed the worker a request goes to, it answers the status that worker
//! answered.
//!
//! The document is the one its routing settings mean
//! ([`RoutingSettings::policy_document`]): the program writes settings as a
//! configuration file would, and no document. The settings' mode says the
//! kinds of worker the program routes to as well ([`Plan`]): single workers
//! ([`settings`]), pairs of a prefill worker and a decode worker
//! ([`pair_settings`]), or such pairs behind encoders that take a request's
//! items ([`encode_settings`]).
//!
//! A route says one thing more of a pair, which no setting holds: whether it
//! sends the decode at once or behind the prefill's answer
//! ([`Program::in_turn`]). A route of the second kind asks the engine for
//! its decode again, behind that answer, where the placement says that the
//! decode still stands chosen.
//!
//! Of the engine the program asks what [`Engine`] lists, and nothing else.
//! It keeps no circuit breaker's count and chooses no rank inside a worker:
//! both are a route's own, behind the engine's answer. It keeps no limit of
//! its own on a pair's workers either ([`NoLimit`]).
//!
//! An encode request here carries items and no more: the program states no
//! fact of a worker, so it serves no image, no demand and no item handed as
//! a reference.

use std::{
    future::Future,
    sync::{Arc, OnceLock},
};

use http::HeaderMap;
use smg_grand_central::{
    document::{AppliedSpec, ModeSettings, PolicySettings, RoutingSettings},
    engine::{
        EncodeClasses, EncodeItems, EncodeLists, EncodeMember, EncoderFacts, HeaderRead, Identity,
        InProcess, LentList, ModelLists, PairAdmission, PairLists, PartnerClasses, PlaceRequest,
        PublishedList, Unplaced, WorkerData, WorkerId, WorkerLists,
    },
    store::{EncodeHandoff, EncodeOffer},
    tracking::{Outcome, Placement, Side, TransportClass},
};
use smg_worker::{
    identity::basic::Worker,
    load::{inflight::WorkerLoadGuard, state::LoadReceiver},
    registry::{Generation, WorkerRegistry},
    worker::WorkerType,
};

/// The attempts one client request gets at most: the first and two more.
pub const ATTEMPTS: usize = 3;

/// The status the program answers a request the engine named no worker
/// for, whatever the reason.
pub const NO_WORKER: u16 = 503;

/// The statuses a send is asked again for: the set a gateway's retry layer
/// asks again for.
pub const ASKED_AGAIN: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// The runtime the program says each of its workers runs, for the encode
/// handoff ([`Lists::cut`]). It is opaque text: the rule compares it and
/// reads nothing in it.
const RUNTIME: &str = "example";

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
/// encoders, every side under `policy` but the encode side, which keeps the
/// policy a configuration that names none gets, and sets nothing else
/// ([`settings`]).
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

/// The program's value of one membership generation of a model: the
/// members the registry published and, cut from them at the value's first
/// lending, the list of each kind of worker with the cell the program keeps
/// beside it ([`Cut`]). The program knows nothing of a worker beside its
/// object, so it states no fact of one.
#[derive(Debug)]
pub struct Lists {
    members: Arc<[Arc<dyn Worker>]>,
    cut: OnceLock<Cut>,
}

/// A generation's members by the kind of worker each is: each kind's list
/// in the registry's order with an empty cell beside it, the pair classes
/// of the prefills and the decodes, and what the encode side reads of the
/// encoders and of the prefills. Which workers pair is the program's own
/// word: here every prefill pairs with every decode. Which encoder hands an
/// item to which prefill is derived by the crate's one rule
/// ([`EncodeClasses::derive`]) from what the program says each worker
/// states: here a rendezvous under one runtime, so every encoder hands to
/// every prefill.
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
    /// The lists, cut at the first ask: each member is read once, for the
    /// kind of worker it is. Asks that meet there wait for the one that
    /// cuts. The engine's task asks behind each membership change, so a
    /// request cuts only where it comes ahead of the task.
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
            let offer = EncodeOffer {
                handoff: EncodeHandoff::Rendezvous,
                ..EncodeOffer::default()
            };
            let member = EncodeMember {
                runtime: RUNTIME,
                offer: &offer,
            };
            let ends = EncodeClasses::derive(
                encoders.iter().map(|_| member),
                prefills.iter().map(|_| member),
            );
            let classed = (0..prefills.len()).map(|place| ends.prefill_class(place));
            Cut {
                classes: PartnerClasses::number(decodes.len(), partners),
                encoder_facts: ends.encoder_facts(encoders.iter().map(|_| &offer)),
                prefill_classes: classed.collect(),
                single: PublishedList::new(of(WorkerType::Regular)),
                prefill: PublishedList::new(prefills),
                decode: PublishedList::new(decodes),
                encode: PublishedList::new(encoders),
            }
        })
    }

    /// The single workers' list lent with its cell, and with no fact.
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
    /// them, the encoders with their cell, and what the encode side reads
    /// of the encoders and of the prefills, under no veto of the program's
    /// own.
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
/// start: the generation the registry holds of a model now, each list of
/// it that holds a worker lent as a request lends it. The engine's task
/// asks them behind each membership word of the registry and fills a cell
/// it finds empty, so a list's first request reads a filled cell where the
/// task came first.
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

/// What a client's request states.
#[derive(Clone, Copy, Debug)]
pub struct Asked<'a> {
    /// The model the request is for.
    pub model: &'a str,
    /// The request's headers, where it carries any.
    pub headers: Option<&'a HeaderMap>,
    /// The keys of the request's encode items, its caller's content hash of
    /// each. Empty for a request with none. A program that routes to no
    /// encoder reads none.
    pub items: &'a [Identity<'a>],
}

/// One attempt of a request on its way: the placement, and the load guard
/// of each worker the request goes to, a pair's decode as well as its
/// prefill. A route keeps them to the end of the response, streamed or not.
/// A route that sends a pair in turn lets the prefill's guard go once the
/// prefill answered ([`Self::prefill_ended`]): that worker holds the request
/// no more while the decode runs.
/// A pair's decode is counted from the placement, whether it is sent at
/// once or behind its prefill's answer: a gateway makes that guard behind
/// its selection too. Where a second ask names another decode, the guard
/// moves there ([`Self::follow_decode`]). No guard of an encoder stands
/// here: an item is counted on its encoder while its own send runs
/// ([`Program::serve`]).
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

    /// `side` of the attempt ended with `status`. A 408 and a 504 say the
    /// worker ran out of time, whichever side of 500 they stand on.
    pub fn ended(&mut self, side: Side, status: u16) {
        let class = match status {
            ..400 => TransportClass::Ok,
            408 | 504 => TransportClass::Timeout,
            400..500 => TransportClass::ClientError,
            _ => TransportClass::ServerError,
        };
        let outcome = Outcome::reported(status, class);
        self.placement.ended(side, outcome);
    }

    /// The prefill of a pair sent in turn ended with `status`: its end is
    /// said, and its load guard goes with it. A prefill that answered
    /// holds the request no more while its decode runs, as a gateway's
    /// sequential dispatch frees the prefill once it is drained. A pair
    /// sent at once keeps both guards to the end of the response.
    fn prefill_ended(&mut self, status: u16) {
        self.ended(Side::Prefill, status);
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
    /// The status the client reads: the last attempt's, or [`NO_WORKER`]
    /// where the engine named no worker.
    pub status: u16,
    /// Why the engine named no worker, where it named none.
    pub unplaced: Option<Unplaced>,
}

/// A send: handed the worker a request goes to, it answers the status the
/// worker answered.
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
}

impl Program {
    /// The program's start: the engine's one call
    /// ([`InProcess::start`]), made over the program's own data. The
    /// engine is built from `spec` for the life of the process and takes
    /// the program's registry, and its load monitor's channel where the
    /// program has a monitor. This program keeps no index of cache events.
    /// Beside the start it hands the engine its lists by model
    /// ([`PublishedLists`]). `plan` is the kinds of worker the settings
    /// behind `spec` route to ([`Plan::of`]).
    ///
    /// It answers the program with the engine's own task, which a program
    /// spawns on its runtime ahead of its first request. The task keeps the
    /// newest load snapshot at hand, runs the engine's upkeep, and follows
    /// the registry: it tells the document's stages a worker that joined or
    /// left, and fills the cells of the model's lists that stand then.
    /// `sends` are the program's sends.
    pub fn start(
        plan: Plan,
        spec: AppliedSpec,
        loads: Option<LoadReceiver>,
        sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
    ) -> (Self, impl Future<Output = ()> + Send + 'static) {
        let registry = Arc::new(WorkerRegistry::new());
        let handed: Arc<WorkerRegistry<Lists>> = Arc::clone(&registry);
        let data = WorkerData {
            registry: handed,
            loads,
            cache_indexes: None,
        };
        let (engine, task) = InProcess::start(spec, data);
        engine.set_lists(Arc::new(PublishedLists(Arc::clone(&registry))));
        let program = Self {
            registry,
            engine: Engine(engine),
            plan,
            sends: Box::new(sends),
            in_turn: false,
        };
        (program, task)
    }

    /// The program with a route that sends a pair's decode behind its
    /// prefill's answer, as a route with sequential dispatch does. Each
    /// pair's request then says that its decode is asked again
    /// ([`PlaceRequest::decode_again`]), and [`Self::serve`] asks it behind
    /// the prefill's answer, where the decode still stands chosen. It is
    /// the route's word and no setting: a program that sends both sides at
    /// once never says it.
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

    /// The worker at `address` leaves. `false` where the registry holds
    /// none there. The registry finds a worker by an address written
    /// without its scheme too. The engine's task hears the removal with
    /// the worker's own object, so the stages are told the worker under
    /// the address it states.
    pub fn leave(&self, address: &str) -> bool {
        self.registry.remove_by_url(address).is_some()
    }

    /// The model's generation as the registry holds it now.
    pub fn published(&self, model: &str) -> Option<Arc<Lists>> {
        self.registry.model_generation(model)
    }

    /// The request of one attempt, as each call for that attempt states it:
    /// the attempt of number `attempt`, `0` the first, of a request that
    /// already tried `tried`, over `lists`, the model's generation as its
    /// caller holds it at that call. The number is stated and not counted
    /// off `tried`: a failed attempt of a pair names two workers. The
    /// request lends what the program's plan says: the single workers'
    /// list, a pair's two, or an encode request's three. The items a
    /// request carries are stated with those three lists, at its first
    /// attempt alone: a retry states none and is placed as the pair. A
    /// model that has no generation lends no list and states no item. A
    /// program whose route sends a pair's decode behind its prefill's
    /// answer says so here.
    fn request<'a>(
        &self,
        asked: &Asked<'a>,
        lists: Option<&'a Lists>,
        attempt: u8,
        tried: &'a [WorkerId],
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
        let carries = attempt == 0 && lends && !asked.items.is_empty();
        PlaceRequest {
            headers,
            encode: carries.then_some(items),
            attempt,
            tried,
            decode_again: self.in_turn,
            ..PlaceRequest::of(asked.model, lent)
        }
    }

    /// One attempt placed: the engine asked once over the generation that
    /// stands, for the request of that attempt ([`Self::request`]). The
    /// answer is the placement kept with the load guard of each worker the
    /// request goes to, under the routing key the placement names: a
    /// pair's decode is counted from here, sent at once or later. An item's
    /// encoder takes its guard at the item's send, and no sooner
    /// ([`Self::serve`]).
    ///
    /// # Errors
    ///
    /// Why the engine named no worker.
    pub fn place(
        &self,
        asked: &Asked<'_>,
        attempt: u8,
        tried: &[WorkerId],
    ) -> Result<InFlight, Unplaced> {
        let lists = self.published(asked.model);
        let request = self.request(asked, lists.as_deref(), attempt, tried);
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

    /// Serves one client request: places an attempt, says it sent, sends
    /// each side of it, says each side's end, and where the attempt was
    /// answered a status a gateway asks again for ([`ASKED_AGAIN`]) places
    /// the next attempt past the workers tried, up to [`ATTEMPTS`]
    /// attempts. Both sides of a pair sent at once go out, whatever the
    /// first answered: the attempt's status is its prefill's where that one
    /// failed, and its decode's otherwise.
    ///
    /// An encode request's items go out once a request, with its first
    /// attempt: each to the encoder the placement names for it, ahead of
    /// the pair. That order is this program's own: a gateway whose workers
    /// state a rendezvous launches the items beside the pair. An item is
    /// counted on its encoder while its send runs, under no routing key,
    /// as a gateway's encode stage counts it: the key is the pair's, and an
    /// encoder that answered holds the request no more. Nothing is said of
    /// an encoder, since no lease stands for one.
    ///
    /// An encoder's failure is the request's answer: neither side of the
    /// pair is sent, the placement's drop gives the pair up, and no attempt
    /// follows. A retry behind a failed side of the pair states no item and
    /// sends none: it is placed as the pair alone, with the pair's two
    /// workers as tried and no encoder, as a gateway's retry is.
    ///
    /// A route that sends a pair's decode behind its prefill's answer
    /// ([`Self::in_turn`]) sends the pair in turn ([`Self::sent_in_turn`]):
    /// a prefill that failed sends no decode.
    ///
    /// A route does more than this: it waits between two attempts, and a
    /// response can say that it is not to be asked again.
    pub fn serve(&self, asked: &Asked<'_>) -> Answer {
        let (mut tried, mut sent, mut attempts) = (Vec::new(), Vec::new(), 0u8);
        loop {
            let placed = self.place(asked, attempts, &tried);
            attempts += 1;
            let mut attempt = match placed {
                Ok(attempt) => attempt,
                Err(unplaced) => {
                    return Answer {
                        sent,
                        status: NO_WORKER,
                        unplaced: Some(unplaced),
                    };
                }
            };
            // The first send that failed is the attempt's answer. Every item
            // goes out, whatever an earlier one's encoder answered.
            let mut status = 0;
            for encoder in attempt.placement.encoders.workers() {
                let answered = {
                    // Counted on its encoder until the send ends.
                    let _counted = WorkerLoadGuard::with_key(Arc::clone(encoder), None);
                    (self.sends)(encoder.as_ref())
                };
                sent.push(encoder.url().to_owned());
                if status < 400 {
                    status = answered;
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
                status = self.sent_in_turn(asked, &mut attempt, attempts - 1, &tried, &mut sent);
            } else {
                attempt.sent();
                for &side in attempt.sides() {
                    let worker = Arc::clone(attempt.worker(side));
                    let answered = (self.sends)(worker.as_ref());
                    attempt.ended(side, answered);
                    sent.push(worker.url().to_owned());
                    if status < 400 {
                        status = answered;
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
            tried.push(attempt.placement.id());
            tried.extend(attempt.placement.decode_id());
        }
    }

    /// A pair's attempt sent in turn, for the attempt of number `number`:
    /// the prefill first, and behind its good answer the decode, which the
    /// engine is asked for again. The answer is the attempt's status: the
    /// prefill's where it failed, else the decode's.
    ///
    /// The order is the one the second ask stands on
    /// ([`Placement::decode_again`]). The prefill's send is said and made.
    /// Behind a good answer the program asks the placement whether its
    /// decode still stands chosen ([`Placement::decode_chosen`]), as a
    /// gateway does ahead of its second ask. Where it does, the request is
    /// stated again, over the generation that stands then, and the decode
    /// is asked ahead of the prefill's end. The decode's guard stands on
    /// the first decode across that ask, so the request says so
    /// (`PlaceRequest::counted_on_decode`): the engine then reads that
    /// decode without this request. Where the ask takes a decode, the
    /// prefill's end is said, the guard follows the decode, and the
    /// decode's send is said behind it. Where it takes none, the first
    /// decision's decode is sent under the guard it has, and its send is
    /// said ahead of the prefill's end.
    ///
    /// A decode that does not stand chosen was booked with its prefill:
    /// by a document that books both sides at once, or for a request the
    /// engine books whole whatever its route says
    /// ([`PlaceRequest::decode_again`]). The engine would refuse its second
    /// ask with no decision made, so the program states no request again
    /// and sends that decode, in the same order.
    ///
    /// The engine hears the decode's dispatch where its send is said
    /// ([`Placement::sent`]), and not where its guard was made. The
    /// prefill's own guard goes where its end is said
    /// ([`InFlight::prefill_ended`]). A prefill that failed sends no decode:
    /// the decode's guard ends with the attempt, and a decode the engine
    /// had booked is given up by the placement's drop.
    fn sent_in_turn(
        &self,
        asked: &Asked<'_>,
        attempt: &mut InFlight,
        number: u8,
        tried: &[WorkerId],
        sent: &mut Vec<String>,
    ) -> u16 {
        attempt.placement.sent(Side::Prefill);
        let prefill = Arc::clone(attempt.worker(Side::Prefill));
        let answered = (self.sends)(prefill.as_ref());
        sent.push(prefill.url().to_owned());
        if answered >= 400 {
            attempt.prefill_ended(answered);
            return answered;
        }
        // A decode booked with its prefill does not stand chosen: it is the
        // decode to send, and no request is stated again for it.
        let taken = attempt.placement.decode_chosen() && {
            let lists = self.published(asked.model);
            let request = PlaceRequest {
                counted_on_decode: true,
                ..self.request(asked, lists.as_deref(), number, tried)
            };
            attempt.placement.decode_again(&request).is_ok()
        };
        if taken {
            attempt.prefill_ended(answered);
            attempt.follow_decode();
            attempt.placement.sent(Side::Decode);
        } else {
            attempt.placement.sent(Side::Decode);
            attempt.prefill_ended(answered);
        }
        let decode = Arc::clone(attempt.worker(Side::Decode));
        let answered = (self.sends)(decode.as_ref());
        attempt.ended(Side::Decode, answered);
        sent.push(decode.url().to_owned());
        answered
    }
}
