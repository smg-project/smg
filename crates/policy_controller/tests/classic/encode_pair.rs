//! The classic encode-pair rows: a fleet of encoders, prefills and decodes,
//! what the gateway reads of each from its registration, a list of requests
//! and what the gateway's selection stage answered each: the pair, the
//! encoders it offered the request's items and each item's encoder, or a
//! refusal by its name. Nothing is drawn.
//!
//! A row replays in two parts. The store's part is which encoders hand to
//! which prefill ([`EncodeClasses`]). A placed request with items was
//! offered the encoders of its prefill's class, those that were ready and
//! take the request's media, and no other. A refusal that names a permanent
//! gap of the encode side says no class serves the request: no prefill with
//! a partner names a class that holds an encoder taking its media. A
//! request without items reads no encoder.
//!
//! The decision's part is every answer whole, through the engine's entry
//! for an encode request ([`PolicyEngine::decide_encode_at`]) over the
//! row's fleet as a host of the engine keeps it: the prefills, the decodes
//! and the encoders as three lists of worker objects, each lent with its
//! cell and with what the host says of its workers. A worker's state and
//! the requests the router holds on it are its object's own. The pair
//! replays pick for pick, and with it whether the items were placed as one
//! key, each key's encoder, and a refusal by its name. A ring's owner is
//! not one identity in the gateway's ring and this crate's, so under
//! consistent hashing a key is held to the encoders it was offered and
//! equal keys to one encoder. The pair's own refusal is replayed by its
//! name alone.
//!
//! A row records the handoff's form on its encoders. The gateway reads a
//! prefill's form as it reads an encoder's, from the runtime, so a prefill
//! here states the form its runtime's encoders state in the row.

use std::{sync::Arc, time::Instant};

use serde::Deserialize;
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        fingerprint, Constraints, Decision, EncodeItems, EncodeLists, EncodePlaced, EncoderFacts,
        FixedFacts, Identity, LentList, ListCell, MediaCounts, PairAdmission, PairLists,
        PartnerClasses, Phase, PolicyEngine, Request as HostRequest, RequestFacts, RequestMaker,
        WorkerLists,
    },
    store::{EncodeClasses, EncodeHandoff, EncodeMember, EncodeOffer},
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus, WorkerType},
};

/// The model every worker of a row serves.
const MODEL: &str = "m";

/// The classic policy these rows record, as a sheet's head names it.
pub const POLICY: &str = "encode_pair";

/// The refusals that name a permanent gap of the encode side: membership
/// proves one, so no class serves the request.
const PERMANENT: [&str; 3] = [
    "ec_incompatible",
    "encode_not_supported",
    "encode_vision_unsupported",
];

/// The form a row records for a runtime's encode handoff.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Form {
    Push,
    Rendezvous,
}

/// An encoder as the gateway reads it from its registration, and whether
/// it is ready.
#[derive(Debug, Deserialize)]
struct Encoder {
    runtime: String,
    /// `None` when its runtime states no form.
    handoff: Option<Form>,
    ec_producer: bool,
    connector: Option<String>,
    images: bool,
    references: bool,
    open: bool,
    in_flight: u32,
}

#[derive(Debug, Deserialize)]
struct Prefill {
    runtime: String,
    ec_consumer: bool,
    connector: Option<String>,
    open: bool,
    in_flight: u32,
    /// The decodes the gateway pairs it with, by index.
    partners: Vec<u32>,
}

/// A decode as the gateway reads it: whether its registration says it
/// takes images, and whether it is ready.
#[derive(Debug, Deserialize)]
struct Decode {
    runtime: String,
    images: bool,
    open: bool,
    in_flight: u32,
}

/// What a request asks of an encoder.
#[derive(Debug, Deserialize)]
struct Request {
    /// The keys of its encode items.
    items: Vec<String>,
    references: bool,
    images: bool,
    /// The decode of the pair must take images.
    decode_images: bool,
    /// Each item is encoded on its own; without it a pushing runtime
    /// places the items as one key.
    fanout: bool,
}

impl Request {
    /// Whether `encoder` takes the request's media: references only when
    /// it processes them itself, images only when it takes them.
    fn taken_by(&self, encoder: &Encoder) -> bool {
        (!self.references || encoder.references) && (!self.images || encoder.images)
    }
}

/// The gateway's answer to one request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Answer {
    Placed {
        prefill: usize,
        decode: usize,
        /// The encoders the request's items were offered, by index.
        offered: Vec<u32>,
        /// The keys the items were placed by: each item's, or the one key.
        keys: Vec<String>,
        /// Each key's encoder, by index.
        encoders: Vec<u32>,
    },
    Refused {
        name: String,
    },
}

#[derive(Debug, Deserialize)]
struct Outcome {
    sequence: Vec<Answer>,
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    prefill_policy: String,
    decode_policy: String,
    encode_policy: String,
    encoders: Vec<Encoder>,
    prefills: Vec<Prefill>,
    decodes: Vec<Decode>,
    requests: Vec<Request>,
    outcome: Outcome,
}

#[derive(Debug, Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

fn handoff(form: Option<Form>) -> EncodeHandoff {
    match form {
        Some(Form::Push) => EncodeHandoff::Push,
        Some(Form::Rendezvous) => EncodeHandoff::Rendezvous,
        None => EncodeHandoff::None,
    }
}

/// Each worker's end of the encode handoff in this crate's shape: the
/// encoders', then the prefills'. `None` when the encoders of one runtime
/// state two forms, which leaves its prefills no one form.
fn offers(row: &Row) -> Option<(Vec<EncodeOffer>, Vec<EncodeOffer>)> {
    let form_of = |runtime: &str| {
        let mut stated = row
            .encoders
            .iter()
            .filter(|encoder| encoder.runtime == runtime)
            .map(|encoder| encoder.handoff);
        let first = stated.next();
        stated
            .all(|form| Some(form) == first)
            .then_some(first.flatten())
    };
    let encoders = row
        .encoders
        .iter()
        .map(|encoder| EncodeOffer {
            handoff: handoff(encoder.handoff),
            producer: encoder.ec_producer,
            consumer: false,
            connector: encoder.connector.clone(),
            references: encoder.references,
        })
        .collect();
    let prefills = row
        .prefills
        .iter()
        .map(|prefill| {
            Some(EncodeOffer {
                handoff: handoff(form_of(&prefill.runtime)?),
                producer: false,
                consumer: prefill.ec_consumer,
                connector: prefill.connector.clone(),
                references: false,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some((encoders, prefills))
}

/// What of one row's store part does not replay, request by request.
fn replay(row: &Row) -> Vec<String> {
    let Some((of_encoders, of_prefills)) = offers(row) else {
        return vec!["the encoders of one runtime state two forms".to_string()];
    };
    let classes = EncodeClasses::derive(
        row.encoders
            .iter()
            .zip(&of_encoders)
            .map(|(worker, offer)| EncodeMember {
                runtime: &worker.runtime,
                offer,
            }),
        row.prefills
            .iter()
            .zip(&of_prefills)
            .map(|(worker, offer)| EncodeMember {
                runtime: &worker.runtime,
                offer,
            }),
    );
    let mut found = Vec::new();
    if row.requests.len() != row.outcome.sequence.len() {
        found.push(format!(
            "{} requests and {} answers",
            row.requests.len(),
            row.outcome.sequence.len()
        ));
    }
    let answered = row.requests.iter().zip(&row.outcome.sequence);
    for (at, (request, answer)) in answered.enumerate() {
        if request.items.is_empty() {
            // A request without items reads no encoder: an answer that
            // offers it one is a defect, whatever the classes hold.
            let offered: &[u32] = match answer {
                Answer::Placed { offered, .. } => offered,
                Answer::Refused { .. } => &[],
            };
            if !offered.is_empty() {
                found.push(format!(
                    "request {at}: it carries no item, yet it was offered the encoders {offered:?}"
                ));
            }
            continue;
        }
        let encoder = |place: u32| row.encoders.get(place as usize);
        let takes = |place: u32| encoder(place).is_some_and(|encoder| request.taken_by(encoder));
        match answer {
            Answer::Placed {
                prefill, offered, ..
            } => {
                let class = classes.list(classes.prefill_class(*prefill));
                let here: Vec<u32> = class
                    .iter()
                    .copied()
                    .filter(|&place| {
                        takes(place) && encoder(place).is_some_and(|encoder| encoder.open)
                    })
                    .collect();
                if here != *offered {
                    found.push(format!(
                        "request {at}: prefill {prefill} was offered the encoders {offered:?}; its class holds {class:?}, of which {here:?} are ready and take the request's media"
                    ));
                }
            }
            Answer::Refused { name } if PERMANENT.contains(&name.as_str()) => {
                let partnered = row
                    .prefills
                    .iter()
                    .enumerate()
                    .filter(|(_, prefill)| !prefill.partners.is_empty());
                for (prefill, _) in partnered {
                    let class = classes.list(classes.prefill_class(prefill));
                    if let Some(place) = class.iter().copied().find(|&place| takes(place)) {
                        found.push(format!(
                            "request {at}: refused as {name}, yet the class of prefill {prefill} holds encoder {place}, which takes the request's media"
                        ));
                    }
                }
            }
            Answer::Refused { .. } => {}
        }
    }
    found.extend(decided(row, &classes, &of_encoders));
    found
}

/// The pair policy every row records, and the one an item's encoder is
/// picked by where no ring places it.
const FEWEST: &str = "power_of_two:least_requests";

/// The engine of a row's three policies under the `epd` plan, as a host of
/// worker lists holds it: each side of the pair takes the worker that holds
/// fewer requests, and so does an item's encoder unless the row places
/// items on the ring. `None` for a policy this replay does not build.
fn engine(row: &Row) -> Option<PolicyEngine> {
    if row.prefill_policy != FEWEST || row.decode_policy != FEWEST {
        return None;
    }
    let encode = match row.encode_policy.as_str() {
        FEWEST => "power_of_two",
        "consistent_hashing" => "consistent_hashing",
        _ => return None,
    };
    let mut spec = PolicySpec::epd_presets(encode, "power_of_two", "power_of_two").ok()?;
    for (name, profile) in &mut spec.profiles {
        if name != "encode" || encode == "power_of_two" {
            profile.pick = "count-first".into();
        }
    }
    let applied = AppliedSpec::validate(&spec).ok()?;
    Some(PolicyEngine::new(applied))
}

/// One worker object of a row's list: ready or not, with the requests the
/// router holds on it as its object's own count.
fn object(list: &str, at: usize, role: WorkerType, open: bool, in_flight: u32) -> Arc<dyn Worker> {
    let object = <BasicWorkerBuilder>::new(format!("http://{list}-{at}:8000"))
        .worker_type(role)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        })
        .build();
    object.set_status(if open {
        WorkerStatus::Ready
    } else {
        WorkerStatus::NotReady
    });
    for _ in 0..in_flight {
        object.increment_load();
    }
    Arc::new(object)
}

/// What a host says of a worker beside its object: whether it takes images.
fn takes_images(vision: bool) -> FixedFacts {
    let mut said = FixedFacts::default();
    said.vision = vision;
    said
}

/// A pair's host with no limit of its own: every prefill has capacity and
/// every decode a room.
#[derive(Debug)]
struct Open;

impl PairAdmission for Open {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// A row's fleet as a host of the engine keeps it: the prefills, the
/// decodes and the encoders as three lists of worker objects, what the host
/// says of each list's workers, the cell kept beside each list, and the
/// pair classes of the row's partner lists.
struct Fleet {
    lists: [Arc<[Arc<dyn Worker>]>; 3],
    said: [Vec<FixedFacts>; 3],
    cells: [ListCell; 3],
    partners: PartnerClasses,
}

impl Fleet {
    fn of(row: &Row) -> Self {
        let pushes = |runtime: &str| {
            let mut of_runtime = row.encoders.iter().filter(|w| w.runtime == runtime);
            of_runtime.any(|w| w.handoff == Some(Form::Push))
        };
        let prefills = (row.prefills.iter().enumerate())
            .map(|(at, w)| object("prefill", at, WorkerType::Prefill, w.open, w.in_flight));
        let decodes = (row.decodes.iter().enumerate())
            .map(|(at, w)| object("decode", at, WorkerType::Decode, w.open, w.in_flight));
        let encoders = (row.encoders.iter().enumerate())
            .map(|(at, w)| object("encode", at, WorkerType::Encode, w.open, w.in_flight));
        // The gateway holds the decodes of a pushing runtime to what their
        // registration says of images, and no other runtime's.
        let said = [
            row.prefills.iter().map(|_| takes_images(false)).collect(),
            (row.decodes.iter())
                .map(|w| takes_images(w.images || !pushes(&w.runtime)))
                .collect(),
            (row.encoders.iter())
                .map(|w| takes_images(w.images))
                .collect(),
        ];
        // Each prefill's partners as places of the decode list, ascending,
        // each once.
        let partners: Vec<Vec<u32>> = (row.prefills.iter())
            .map(|prefill| {
                let mut list = prefill.partners.clone();
                list.sort_unstable();
                list.dedup();
                list
            })
            .collect();
        Self {
            lists: [prefills.collect(), decodes.collect(), encoders.collect()],
            said,
            cells: Default::default(),
            partners: PartnerClasses::number(row.decodes.len(), partners.iter().map(Vec::as_slice)),
        }
    }

    /// The list at `list` as its host lends it for one decision.
    fn lent(&self, list: usize) -> LentList<'_> {
        LentList {
            workers: &self.lists[list],
            cell: &self.cells[list],
            facts: Some(&self.said[list]),
        }
    }
}

/// One request of a row, over the fleet's three lists as they are lent.
struct Asked<'a> {
    request: HostRequest<'a>,
    lists: EncodeLists<'a>,
}

impl RequestMaker for Asked<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> HostRequest<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Encode(self.lists)
    }
}

/// The recorded name of a refusal, read from the stage a shed names: the
/// encode side's stages by their names, the pair's own refusal for every
/// other.
fn refusal(stage: &str) -> &'static str {
    match stage {
        "encoders" => "unavailable",
        "encode-vision" => "encode_vision_unsupported",
        "ec-compatible" => "ec_incompatible",
        "encode-media" => "encode_not_supported",
        "decode-vision" => "decode_vision_unsupported",
        _ => "placement",
    }
}

/// What of one row's answers the decision does not replay, request by
/// request. `offers` are the ends its encoders state, in the row's order.
fn decided(row: &Row, classes: &EncodeClasses, offers: &[EncodeOffer]) -> Vec<String> {
    let Some(engine) = engine(row) else {
        return vec!["a policy this replay does not build".to_string()];
    };
    let ring = row.encode_policy != FEWEST;
    let fleet = Fleet::of(row);
    // What the decision reads of each encoder, by the rule the store
    // writes it by.
    let of_encoders: Vec<EncoderFacts> = classes.encoder_facts(offers);
    let prefill_classes: Vec<u32> = (0..row.prefills.len())
        .map(|at| classes.prefill_class(at))
        .collect();
    let lists = EncodeLists {
        pair: PairLists {
            prefill: fleet.lent(0),
            decode: fleet.lent(1),
            classes: &fleet.partners,
            admission: &Open,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        },
        encoders: fleet.lent(2),
        encoders_places: None,
        encoders_stalled: &[],
        encoders_held: &[],
        facts: &of_encoders,
        prefill_classes: &prefill_classes,
    };
    let now = Instant::now();
    let mut placed = EncodePlaced::new();
    let mut found = Vec::new();
    let answered = row.requests.iter().zip(&row.outcome.sequence);
    for (at, (request, answer)) in answered.enumerate() {
        let keys: Vec<Identity<'_>> = (request.items.iter())
            .map(|item| Identity::plain(item))
            .collect();
        // The items taken as one: the caller's digest, any value its
        // callers agree on.
        let digest = Identity::hashed(fingerprint(request.items.concat().as_bytes()));
        let items = EncodeItems {
            keys: &keys,
            one: (!request.fanout && !keys.is_empty()).then_some(digest),
            references: request.references,
            decode_images: request.decode_images,
        };
        let seed = at as u64 + 1;
        let id = seed.to_le_bytes();
        // The request as its host makes it for the prefill phase: held to
        // no wire, its decode sent with its prefill.
        let asked = Asked {
            request: HostRequest {
                id: &id,
                facts: RequestFacts {
                    model: MODEL,
                    phase: Phase::Prefill,
                    media: MediaCounts {
                        images: u16::from(request.images),
                        ..MediaCounts::default()
                    },
                    seed,
                    ..RequestFacts::default()
                },
                constraints: Constraints::default(),
            },
            lists,
        };
        let decision = match engine.decide_encode_at(&asked, items, &mut placed, || now) {
            Ok((decided, _)) => decided.decision,
            Err(refused) => {
                found.push(format!("request {at}: no decision: {refused:?}"));
                continue;
            }
        };
        match (answer, &decision) {
            (
                Answer::Placed {
                    prefill,
                    decode,
                    offered,
                    keys: placed_by,
                    encoders: recorded,
                },
                Decision::Pick(pick),
            ) => {
                let target = pick.targets.first();
                let pair = target.map(|t| {
                    let decode = t.pair.map(|pair| pair.decode_position as usize);
                    (t.position as usize, decode)
                });
                if pair != Some((*prefill, Some(*decode))) {
                    found.push(format!(
                        "request {at}: the pair of prefill {prefill} and decode {decode}, replayed {pair:?}"
                    ));
                }
                let one = *placed_by != request.items;
                if placed.one() != one {
                    found.push(format!(
                        "request {at}: the items placed as one key: {one}, replayed {}",
                        placed.one()
                    ));
                }
                let here: Vec<u32> = placed.encoders().iter().map(|t| t.position).collect();
                let same = if ring {
                    let of_keys = || placed_by.iter().zip(&here);
                    here.len() == recorded.len()
                        && here.iter().all(|encoder| offered.contains(encoder))
                        && of_keys().all(|(key, encoder)| {
                            of_keys().all(|(other, of_other)| key != other || encoder == of_other)
                        })
                } else {
                    here == *recorded
                };
                if !same {
                    found.push(format!(
                        "request {at}: the encoders {recorded:?} of the offered {offered:?}, replayed {here:?}"
                    ));
                }
            }
            (Answer::Refused { name }, Decision::Shed(shed)) => {
                let here = refusal(shed.emptied_by);
                if here != name {
                    found.push(format!(
                        "request {at}: refused as {name}, replayed {here} (the stage {})",
                        shed.emptied_by
                    ));
                }
            }
            (Answer::Placed { .. }, other) => {
                found.push(format!("request {at}: placed, replayed {other:?}"));
            }
            (Answer::Refused { name }, other) => {
                found.push(format!(
                    "request {at}: refused as {name}, replayed {other:?}"
                ));
            }
        }
    }
    found
}

/// The rows of a sheet replayed: how many, and what did not replay.
pub fn replay_rows(text: &str, source: &str) -> (usize, Vec<String>) {
    let sheet: Sheet = match serde_json::from_str(text) {
        Ok(sheet) => sheet,
        Err(err) => {
            return (
                0,
                vec![format!("{source} is not a sheet of {POLICY} rows: {err}")],
            )
        }
    };
    let defects = sheet
        .rows
        .iter()
        .flat_map(|row| {
            let name = &row.name;
            replay(row)
                .into_iter()
                .map(move |why| format!("{source}: {name}: {why}"))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// Rows copied from the recorded sheet, one or more of each form the
/// store's part reads, with every runtime replaced by a token. The rule
/// compares runtimes for equality and reads none, and the tokens keep the
/// names' order, so a copied row is its sheet's row under one renaming: the
/// pair index's rule for a copied row holds these rows to that too.
pub const INLINE: &str = r#"{
  "recorded_from": "4ba2a6431357be5411a2383cdc4916b41471d722",
  "policy": "encode_pair",
  "rows": [
    {
      "name": "two-runtimes-the-items-go-to-the-encoders-of-the-pushing-pair",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": true, "references": true, "open": true, "in_flight": 3},
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": true, "references": true, "open": true, "in_flight": 1},
        {"runtime": "sigma", "handoff": "rendezvous", "ec_producer": false, "connector": null, "images": true, "references": false, "open": true, "in_flight": 2},
        {"runtime": "sigma", "handoff": "rendezvous", "ec_producer": false, "connector": null, "images": true, "references": false, "open": true, "in_flight": 0}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": true, "connector": "c1", "open": true, "in_flight": 0, "partners": [0]},
        {"runtime": "sigma", "ec_consumer": false, "connector": null, "open": true, "in_flight": 4, "partners": [1]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0},
        {"runtime": "sigma", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01", "02", "03"], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"placed": {"prefill": 0, "decode": 0, "offered": [0, 1], "keys": ["01", "02", "03"], "encoders": [1, 1, 1]}}
      ]}
    },
    {
      "name": "two-runtimes-the-pair-is-picked-where-an-encoder-is-ready-the-rendezvous-encoders-are-not",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": true, "references": true, "open": true, "in_flight": 0},
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": true, "references": true, "open": true, "in_flight": 1},
        {"runtime": "sigma", "handoff": "rendezvous", "ec_producer": false, "connector": null, "images": true, "references": false, "open": false, "in_flight": 3},
        {"runtime": "sigma", "handoff": "rendezvous", "ec_producer": false, "connector": null, "images": true, "references": false, "open": false, "in_flight": 2}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": true, "connector": "c1", "open": true, "in_flight": 4, "partners": [0]},
        {"runtime": "sigma", "ec_consumer": false, "connector": null, "open": true, "in_flight": 0, "partners": [1]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0},
        {"runtime": "sigma", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01", "02"], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"placed": {"prefill": 0, "decode": 0, "offered": [0, 1], "keys": ["01", "02"], "encoders": [0, 0]}}
      ]}
    },
    {
      "name": "push-a-prefill-takes-the-encoders-of-its-connector-the-one-that-holds-fewer",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": true, "references": true, "open": true, "in_flight": 2},
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c2", "images": true, "references": true, "open": true, "in_flight": 0},
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c2", "images": true, "references": true, "open": true, "in_flight": 1}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": true, "connector": "c1", "open": true, "in_flight": 4, "partners": [0, 1]},
        {"runtime": "zeta", "ec_consumer": true, "connector": "c2", "open": true, "in_flight": 0, "partners": [0, 1]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0},
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 1}
      ],
      "requests": [
        {"items": ["01", "02", "03"], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"placed": {"prefill": 1, "decode": 0, "offered": [1, 2], "keys": ["01", "02", "03"], "encoders": [1, 1, 1]}}
      ]}
    },
    {
      "name": "push-a-producer-and-a-consumer-on-two-connectors-are-refused",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": true, "references": true, "open": true, "in_flight": 0}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": true, "connector": "c2", "open": true, "in_flight": 0, "partners": [0]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01"], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"refused": {"name": "ec_incompatible"}}
      ]}
    },
    {
      "name": "push-an-encoder-without-the-producer-role-is-refused",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": false, "connector": "c1", "images": true, "references": true, "open": true, "in_flight": 0}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": true, "connector": "c1", "open": true, "in_flight": 0, "partners": [0]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01"], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"refused": {"name": "ec_incompatible"}}
      ]}
    },
    {
      "name": "push-a-role-without-a-connector-is-no-role",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": false, "connector": null, "images": true, "references": true, "open": true, "in_flight": 0}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": false, "connector": null, "open": true, "in_flight": 0, "partners": [0]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01"], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"refused": {"name": "ec_incompatible"}}
      ]}
    },
    {
      "name": "push-a-role-on-a-connector-without-a-name-is-no-role",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": false, "connector": null, "images": true, "references": true, "open": true, "in_flight": 0}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": false, "connector": null, "open": true, "in_flight": 0, "partners": [0]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01"], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"refused": {"name": "ec_incompatible"}}
      ]}
    },
    {
      "name": "rendezvous-references-are-not-supported",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "sigma", "handoff": "rendezvous", "ec_producer": false, "connector": null, "images": true, "references": false, "open": true, "in_flight": 2},
        {"runtime": "sigma", "handoff": "rendezvous", "ec_producer": false, "connector": null, "images": true, "references": false, "open": true, "in_flight": 1}
      ],
      "prefills": [
        {"runtime": "sigma", "ec_consumer": false, "connector": null, "open": true, "in_flight": 0, "partners": [0]}
      ],
      "decodes": [
        {"runtime": "sigma", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01"], "references": true, "images": true, "decode_images": false, "fanout": true},
        {"items": ["01"], "references": true, "images": false, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"refused": {"name": "encode_not_supported"}},
        {"refused": {"name": "encode_not_supported"}}
      ]}
    },
    {
      "name": "push-references-go-to-the-encoders-that-process-them",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": true, "references": false, "open": true, "in_flight": 0},
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": true, "references": true, "open": true, "in_flight": 3}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": true, "connector": "c1", "open": true, "in_flight": 0, "partners": [0]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01", "02"], "references": true, "images": true, "decode_images": false, "fanout": true},
        {"items": ["01", "02"], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"placed": {"prefill": 0, "decode": 0, "offered": [1], "keys": ["01", "02"], "encoders": [1, 1]}},
        {"placed": {"prefill": 0, "decode": 0, "offered": [0, 1], "keys": ["01", "02"], "encoders": [0, 0]}}
      ]}
    },
    {
      "name": "images-without-an-encoder-that-takes-them-are-refused",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "zeta", "handoff": "push", "ec_producer": true, "connector": "c1", "images": false, "references": true, "open": true, "in_flight": 0}
      ],
      "prefills": [
        {"runtime": "zeta", "ec_consumer": true, "connector": "c1", "open": true, "in_flight": 0, "partners": [0]}
      ],
      "decodes": [
        {"runtime": "zeta", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01"], "references": false, "images": true, "decode_images": false, "fanout": true},
        {"items": ["01"], "references": false, "images": false, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"refused": {"name": "encode_vision_unsupported"}},
        {"placed": {"prefill": 0, "decode": 0, "offered": [0], "keys": ["01"], "encoders": [0]}}
      ]}
    },
    {
      "name": "an-encoder-whose-runtime-states-no-handoff-serves-no-item",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "encode_policy": "power_of_two:least_requests",
      "encoders": [
        {"runtime": "kappa", "handoff": null, "ec_producer": false, "connector": null, "images": true, "references": false, "open": true, "in_flight": 0}
      ],
      "prefills": [
        {"runtime": "kappa", "ec_consumer": false, "connector": null, "open": true, "in_flight": 0, "partners": [0]}
      ],
      "decodes": [
        {"runtime": "kappa", "images": true, "open": true, "in_flight": 0}
      ],
      "requests": [
        {"items": ["01"], "references": false, "images": true, "decode_images": false, "fanout": true},
        {"items": ["01"], "references": false, "images": false, "decode_images": false, "fanout": true},
        {"items": [], "references": false, "images": true, "decode_images": false, "fanout": true}
      ],
      "outcome": {"sequence": [
        {"refused": {"name": "encode_not_supported"}},
        {"refused": {"name": "unavailable"}},
        {"placed": {"prefill": 0, "decode": 0, "offered": [], "keys": [], "encoders": []}}
      ]}
    }
  ]
}"#;
