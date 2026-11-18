//! The classic encode-pair rows: a fleet of encoders, prefills and decodes,
//! what the gateway reads of each from its registration, a list of requests
//! and what the gateway's selection stage answered each: the pair, the
//! encoders it offered the request's items and each item's encoder, or a
//! refusal by its name. Nothing is drawn.
//!
//! This module replays the store's part of a row: which encoders hand to
//! which prefill ([`EncodeClasses`]). A placed request with items was
//! offered the encoders of its prefill's class, those that were ready and
//! take the request's media, and no other. A refusal that names a permanent
//! gap of the encode side says no class serves the request: no prefill with
//! a partner names a class that holds an encoder taking its media. A
//! request without items reads no encoder. The pair, each item's encoder
//! and the other refusals are the decision's: the rule that places the
//! items replays them.
//!
//! A row records the handoff's form on its encoders. The gateway reads a
//! prefill's form as it reads an encoder's, from the runtime, so a prefill
//! here states the form its runtime's encoders state in the row.

use serde::Deserialize;
use smg_policy_controller::store::encoding::{
    EncodeClasses, EncodeHandoff, EncodeMember, EncodeOffer,
};

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
}

#[derive(Debug, Deserialize)]
struct Prefill {
    runtime: String,
    ec_consumer: bool,
    connector: Option<String>,
    /// The decodes the gateway pairs it with, by index.
    partners: Vec<u32>,
}

/// What a request asks of an encoder.
#[derive(Debug, Deserialize)]
struct Request {
    /// The keys of its encode items.
    items: Vec<String>,
    references: bool,
    images: bool,
}

impl Request {
    /// Whether `encoder` takes the request's media: references only when
    /// it processes them itself, images only when it takes them.
    fn taken_by(&self, encoder: &Encoder) -> bool {
        (!self.references || encoder.references) && (!self.images || encoder.images)
    }
}

/// The part of an answer the store's lists answer for.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Answer {
    Placed {
        prefill: usize,
        /// The encoders the request's items were offered, by index.
        offered: Vec<u32>,
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
    encoders: Vec<Encoder>,
    prefills: Vec<Prefill>,
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
            continue;
        }
        let encoder = |place: u32| row.encoders.get(place as usize);
        let takes = |place: u32| encoder(place).is_some_and(|encoder| request.taken_by(encoder));
        match answer {
            Answer::Placed { prefill, offered } => {
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
