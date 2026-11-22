//! The classic pair-index rows: a prefill pool and a decode pool, what each
//! worker reports of its KV handoff, and what the gateway's pair index made
//! of the two pools on each of its two wires under each pairing mode: the
//! rule's verdict on every pair, each prefill's partner list and the refusal
//! where no pair exists. Nothing is drawn: a row replays whole.

use serde::Deserialize;
use serde_json::Value;
use smg_policy_controller::{
    document::Pairing,
    engine::{WireModes, WorkerId},
    store::pairing::{key, wire_mismatch, Member, PairClasses, PairingDescriptor, LAYOUT_FACTS},
};

/// The classic policy these rows record, as a sheet's head names it.
pub const POLICY: &str = "pair-index";

/// One worker of a row: its runtime (`None`: undetected), what it reports
/// and the pairing key the gateway names its group by.
#[derive(Debug, Deserialize)]
struct Worker {
    runtime: Option<String>,
    protocol: Option<String>,
    transport: Option<String>,
    version: Option<String>,
    /// The layout facts it reports: each fact's name and its value.
    kv_layout: Vec<(String, String)>,
    key: String,
}

/// Why no pair exists on a wire, as the gateway reports it.
#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Refusal {
    prefill: Vec<String>,
    decode: Vec<String>,
    mismatches: Vec<String>,
}

/// What the pair index made of the two pools on one wire under one mode,
/// and what this crate makes of them in the same shape.
#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Outcome {
    /// Per prefill, per decode: the component the pair is refused on.
    verdicts: Vec<Vec<Option<String>>>,
    /// Per prefill: its partners, by index in the decode pool.
    partners: Vec<Vec<u32>>,
    refusal: Option<Refusal>,
}

#[derive(Debug, Deserialize)]
struct Modes {
    strict: Outcome,
    lenient: Outcome,
    off: Outcome,
}

/// The gateway's two pair wires: the one whose pairs share a runtime, and
/// the plain one.
#[derive(Debug, Deserialize)]
struct Wires {
    grpc: Modes,
    http: Modes,
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    prefills: Vec<Worker>,
    decodes: Vec<Worker>,
    outcome: Wires,
}

#[derive(Debug, Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

/// Each worker's descriptor in this crate's shape; `None` when one reports
/// a layout fact this crate does not compare.
fn descriptors(pool: &[Worker]) -> Option<Vec<PairingDescriptor>> {
    pool.iter()
        .map(|worker| {
            let mut kv_layout: [Option<String>; 4] = Default::default();
            for (name, value) in &worker.kv_layout {
                let at = LAYOUT_FACTS.iter().position(|&(_, fact)| fact == name)?;
                kv_layout[at] = Some(value.clone());
            }
            Some(PairingDescriptor {
                protocol: worker.protocol.clone(),
                transport: worker.transport.clone(),
                version: worker.version.clone(),
                kv_layout,
            })
        })
        .collect()
}

/// A pool as the pairing rule reads it on `wires`, numbered from `first`.
fn members<'a>(
    pool: &'a [Worker],
    descriptors: &'a [PairingDescriptor],
    wires: WireModes,
    first: u64,
) -> Vec<Member<'a>> {
    pool.iter()
        .zip(descriptors)
        .zip(first..)
        .map(|((worker, descriptor), id)| {
            let runtime = worker.runtime.as_deref().unwrap_or("");
            Member::new(WorkerId(id), wires, runtime, Some(descriptor))
        })
        .collect()
}

/// What this crate makes of the two pools under `mode`: the rule on every
/// pair, the derived class of each prefill read back as its partner list,
/// and the refusal of the pools' wire.
fn outcome(prefills: &[Member<'_>], decodes: &[Member<'_>], mode: Pairing) -> Outcome {
    let classes = PairClasses::derive(prefills, decodes, mode);
    let verdict = |prefill, decode| wire_mismatch(prefill, decode, mode).map(|m| m.to_string());
    Outcome {
        verdicts: prefills
            .iter()
            .map(|prefill| {
                decodes
                    .iter()
                    .map(|decode| verdict(prefill, decode))
                    .collect()
            })
            .collect(),
        partners: classes
            .prefills()
            .iter()
            .map(|&(_, class)| classes.list(class).to_vec())
            .collect(),
        refusal: prefills
            .first()
            .and_then(|prefill| classes.refusal(prefill.wires))
            .map(|refusal| Refusal {
                prefill: refusal.prefill.clone(),
                decode: refusal.decode.clone(),
                mismatches: refusal.mismatches.clone(),
            }),
    }
}

/// What of one row does not replay: a key, or a wire's outcome under a
/// mode, each with the two sides.
fn replay(row: &Row) -> Vec<String> {
    let (Some(of_prefills), Some(of_decodes)) =
        (descriptors(&row.prefills), descriptors(&row.decodes))
    else {
        return vec!["a layout fact this crate does not compare".to_string()];
    };
    let mut found = Vec::new();
    let wires = [
        ("grpc", WireModes::GRPC, &row.outcome.grpc),
        ("http", WireModes::HTTP, &row.outcome.http),
    ];
    for (wire, on, recorded) in wires {
        let prefills = members(&row.prefills, &of_prefills, on, 1);
        let decodes = members(&row.decodes, &of_decodes, on, 1 + row.prefills.len() as u64);
        let workers = row.prefills.iter().chain(&row.decodes);
        for (worker, member) in workers.zip(prefills.iter().chain(&decodes)) {
            let here = key(member);
            if here != worker.key {
                found.push(format!("{wire}: key {}, replayed {here}", worker.key));
            }
        }
        let modes = [
            ("strict", Pairing::Strict, &recorded.strict),
            ("lenient", Pairing::Lenient, &recorded.lenient),
            ("off", Pairing::Off, &recorded.off),
        ];
        for (name, mode, recorded) in modes {
            let here = outcome(&prefills, &decodes, mode);
            if here != *recorded {
                found.push(format!(
                    "{wire}/{name}: recorded {recorded:?}, replayed {here:?}"
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

/// The names of one copied row beside its sheet's row: for each kind of
/// name (a runtime, a protocol, a transport, a version, one layout fact's
/// values, a pairing key), the sheet's name and the copy's token for it.
#[derive(Default)]
struct Renaming {
    names: Vec<(String, String, String)>,
}

impl Renaming {
    /// Whether `copied` is `held` but for its names: one shape, every
    /// number, flag and null the same, every word of the rule the same (a
    /// verdict, a refused component, a layout fact), and each name noted
    /// under its kind; `kind` is the kind of the names at this place.
    fn same_but_names(&mut self, held: &Value, copied: &Value, kind: Option<&str>) -> bool {
        match (held, copied) {
            (Value::String(held), Value::String(copied)) => match kind {
                Some(kind) => {
                    self.names
                        .push((kind.to_string(), held.clone(), copied.clone()));
                    true
                }
                None => held == copied,
            },
            (Value::Array(held), Value::Array(copied)) => {
                held.len() == copied.len()
                    && held
                        .iter()
                        .zip(copied)
                        .all(|(held, copied)| self.same_but_names(held, copied, kind))
            }
            (Value::Object(held), Value::Object(copied)) => {
                held.len() == copied.len()
                    && held.iter().all(|(key, held)| {
                        copied.get(key).is_some_and(|copied| match key.as_str() {
                            "kv_layout" => self.same_layout(held, copied),
                            "runtime" | "protocol" | "transport" | "version" | "key" => {
                                self.same_but_names(held, copied, Some(key))
                            }
                            // A refusal lists each side's pairing keys.
                            "prefill" | "decode" => self.same_but_names(held, copied, Some("key")),
                            _ => self.same_but_names(held, copied, None),
                        })
                    })
            }
            _ => held == copied,
        }
    }

    /// A worker's layout facts: the same facts in the same order, each
    /// fact's value a name of that fact's own kind.
    fn same_layout(&mut self, held: &Value, copied: &Value) -> bool {
        let (Some(held), Some(copied)) = (held.as_array(), copied.as_array()) else {
            return false;
        };
        held.len() == copied.len()
            && held.iter().zip(copied).all(|(held, copied)| {
                let facts = (
                    held.as_array().map(Vec::as_slice),
                    copied.as_array().map(Vec::as_slice),
                );
                match facts {
                    (Some([Value::String(fact), held]), Some([Value::String(same), copied]))
                        if fact == same =>
                    {
                        self.same_but_names(held, copied, Some(&format!("layout {fact}")))
                    }
                    _ => false,
                }
            })
    }

    /// Whether each kind's names are renamed one to one and in the sheet's
    /// order: any two names of a kind compare as their tokens do.
    fn one_to_one_in_order(&self) -> bool {
        self.names.iter().all(|(kind, held, copied)| {
            self.names.iter().filter(|(other, ..)| other == kind).all(
                |(_, other_held, other_copied)| held.cmp(other_held) == copied.cmp(other_copied),
            )
        })
    }
}

/// What separates the rows copied into this module from the sheet they were
/// copied from: a copied row the sheet does not hold, or one that is not
/// the sheet's row of its name under one renaming of its names. Nothing in
/// a pair row is drawn, so nothing else may differ.
pub fn copied_row_defects(inline: &str, sheet: &str, source: &str) -> Vec<String> {
    let rows = |text: &str, source: &str| {
        let rows = serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|mut sheet| sheet.get_mut("rows").map(Value::take));
        match rows {
            Some(Value::Array(rows)) => Ok(rows),
            _ => Err(format!("{source} is not a sheet of rows")),
        }
    };
    let (copied, held) = match (rows(inline, "the inline rows"), rows(sheet, source)) {
        (Ok(copied), Ok(held)) => (copied, held),
        (Err(why), _) | (_, Err(why)) => return vec![why],
    };
    copied
        .iter()
        .filter_map(|row| {
            let name = row.get("name").and_then(Value::as_str).unwrap_or("?");
            let Some(sheet_row) = held.iter().find(|held| held.get("name") == row.get("name"))
            else {
                return Some(format!("{source}: the inline row {name} is on no sheet"));
            };
            let mut names = Renaming::default();
            let same = names.same_but_names(sheet_row, row, None) && names.one_to_one_in_order();
            (!same).then(|| {
                format!("{source}: the inline row {name} is not the sheet's row under one renaming")
            })
        })
        .collect()
}

/// Rows copied from the recorded sheet, one or more of each form, with
/// every runtime, transport and layout value replaced by a token. The rule
/// compares names for equality and reads none, and the tokens keep the
/// names' order, so a row's verdicts, lists and sorted keys are the sheet's
/// ([`copied_row_defects`] holds each copied row to that).
pub const INLINE: &str = r#"{
  "recorded_from": "64ff5a437b5007af5079f0492b46f4e5a712b66f",
  "policy": "pair-index",
  "rows": [
    {
      "name": "pair-two-transports-list-apart",
      "prefills": [
        {"runtime": "zeta", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "zeta/tx-b/?"},
        {"runtime": "zeta", "protocol": null, "transport": "tx-a", "version": null, "kv_layout": [], "key": "zeta/tx-a/?"}
      ],
      "decodes": [
        {"runtime": "zeta", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "zeta/tx-b/?"},
        {"runtime": "zeta", "protocol": null, "transport": "tx-a", "version": null, "kv_layout": [], "key": "zeta/tx-a/?"},
        {"runtime": "zeta", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "zeta/tx-b/?"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [[null, "transport", null], ["transport", null, "transport"]], "partners": [[0, 2], [1]], "refusal": null},
          "lenient": {"verdicts": [[null, "transport", null], ["transport", null, "transport"]], "partners": [[0, 2], [1]], "refusal": null},
          "off": {"verdicts": [[null, null, null], [null, null, null]], "partners": [[0, 1, 2], [0, 1, 2]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [[null, "transport", null], ["transport", null, "transport"]], "partners": [[0, 2], [1]], "refusal": null},
          "lenient": {"verdicts": [[null, "transport", null], ["transport", null, "transport"]], "partners": [[0, 2], [1]], "refusal": null},
          "off": {"verdicts": [[null, null, null], [null, null, null]], "partners": [[0, 1, 2], [0, 1, 2]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-a-decode-that-names-no-transport-is-on-two-lists",
      "prefills": [
        {"runtime": "zeta", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "zeta/tx-b/?"},
        {"runtime": "zeta", "protocol": null, "transport": "tx-a", "version": null, "kv_layout": [], "key": "zeta/tx-a/?"}
      ],
      "decodes": [
        {"runtime": "zeta", "protocol": null, "transport": null, "version": null, "kv_layout": [], "key": "zeta/?/?"},
        {"runtime": "zeta", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "zeta/tx-b/?"},
        {"runtime": "zeta", "protocol": null, "transport": "tx-a", "version": null, "kv_layout": [], "key": "zeta/tx-a/?"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [["unknown transport", null, "transport"], ["unknown transport", "transport", null]], "partners": [[1], [2]], "refusal": null},
          "lenient": {"verdicts": [[null, null, "transport"], [null, "transport", null]], "partners": [[0, 1], [0, 2]], "refusal": null},
          "off": {"verdicts": [[null, null, null], [null, null, null]], "partners": [[0, 1, 2], [0, 1, 2]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [["unknown transport", null, "transport"], ["unknown transport", "transport", null]], "partners": [[1], [2]], "refusal": null},
          "lenient": {"verdicts": [[null, null, "transport"], [null, "transport", null]], "partners": [[0, 1], [0, 2]], "refusal": null},
          "off": {"verdicts": [[null, null, null], [null, null, null]], "partners": [[0, 1, 2], [0, 1, 2]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-no-shared-transport-is-refused",
      "prefills": [
        {"runtime": "zeta", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "zeta/tx-b/?"}
      ],
      "decodes": [
        {"runtime": "zeta", "protocol": null, "transport": "tx-a", "version": null, "kv_layout": [], "key": "zeta/tx-a/?"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [["transport"]], "partners": [[]], "refusal": {"prefill": ["zeta/tx-b/?"], "decode": ["zeta/tx-a/?"], "mismatches": ["transport"]}},
          "lenient": {"verdicts": [["transport"]], "partners": [[]], "refusal": {"prefill": ["zeta/tx-b/?"], "decode": ["zeta/tx-a/?"], "mismatches": ["transport"]}},
          "off": {"verdicts": [[null]], "partners": [[0]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [["transport"]], "partners": [[]], "refusal": {"prefill": ["zeta/tx-b/?"], "decode": ["zeta/tx-a/?"], "mismatches": ["transport"]}},
          "lenient": {"verdicts": [["transport"]], "partners": [[]], "refusal": {"prefill": ["zeta/tx-b/?"], "decode": ["zeta/tx-a/?"], "mismatches": ["transport"]}},
          "off": {"verdicts": [[null]], "partners": [[0]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-two-runtimes-each-list-their-own",
      "prefills": [
        {"runtime": "zeta", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "zeta/tx-b/?"},
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "rho/tx-b/?"}
      ],
      "decodes": [
        {"runtime": "zeta", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "zeta/tx-b/?"},
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "rho/tx-b/?"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [[null, "runtime"], ["runtime", null]], "partners": [[0], [1]], "refusal": null},
          "lenient": {"verdicts": [[null, "runtime"], ["runtime", null]], "partners": [[0], [1]], "refusal": null},
          "off": {"verdicts": [[null, "runtime"], ["runtime", null]], "partners": [[0], [1]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [[null, "runtime"], ["runtime", null]], "partners": [[0], [1]], "refusal": null},
          "lenient": {"verdicts": [[null, "runtime"], ["runtime", null]], "partners": [[0], [1]], "refusal": null},
          "off": {"verdicts": [[null, null], [null, null]], "partners": [[0, 1], [0, 1]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-an-undetected-runtime-on-one-side",
      "prefills": [
        {"runtime": "zeta", "protocol": null, "transport": null, "version": null, "kv_layout": [], "key": "zeta/?/?"}
      ],
      "decodes": [
        {"runtime": null, "protocol": null, "transport": null, "version": null, "kv_layout": [], "key": "unspecified/?/?"},
        {"runtime": "zeta", "protocol": null, "transport": null, "version": null, "kv_layout": [], "key": "zeta/?/?"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [["unknown runtime", "unknown transport"]], "partners": [[]], "refusal": {"prefill": ["zeta/?/?"], "decode": ["unspecified/?/?", "zeta/?/?"], "mismatches": ["unknown runtime", "unknown transport"]}},
          "lenient": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null},
          "off": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [["unknown runtime", "unknown transport"]], "partners": [[]], "refusal": {"prefill": ["zeta/?/?"], "decode": ["unspecified/?/?", "zeta/?/?"], "mismatches": ["unknown runtime", "unknown transport"]}},
          "lenient": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null},
          "off": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-equal-protocols-settle-the-pair-over-the-transport",
      "prefills": [
        {"runtime": "zeta", "protocol": "blue", "transport": "tx-b", "version": null, "kv_layout": [], "key": "blue"}
      ],
      "decodes": [
        {"runtime": "zeta", "protocol": "blue", "transport": "tx-a", "version": null, "kv_layout": [], "key": "blue"},
        {"runtime": "zeta", "protocol": "red", "transport": "tx-b", "version": null, "kv_layout": [], "key": "red"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [[null, "pairing_protocol"]], "partners": [[0]], "refusal": null},
          "lenient": {"verdicts": [[null, "pairing_protocol"]], "partners": [[0]], "refusal": null},
          "off": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [[null, "pairing_protocol"]], "partners": [[0]], "refusal": null},
          "lenient": {"verdicts": [[null, "pairing_protocol"]], "partners": [[0]], "refusal": null},
          "off": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-a-layout-fact-one-side-reports",
      "prefills": [
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [["kv_cache_dtype", "a"], ["attention_backend", "a"]], "key": "rho/tx-b/dtype=a,attn=a"}
      ],
      "decodes": [
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [["kv_cache_dtype", "a"]], "key": "rho/tx-b/dtype=a"},
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [["kv_cache_dtype", "a"], ["attention_backend", "a"]], "key": "rho/tx-b/dtype=a,attn=a"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [["unknown attention_backend", null]], "partners": [[1]], "refusal": null},
          "lenient": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null},
          "off": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [["unknown attention_backend", null]], "partners": [[1]], "refusal": null},
          "lenient": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null},
          "off": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-two-layout-facts-differ-and-one-is-named",
      "prefills": [
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [["kv_cache_dtype", "a"], ["page_size", "16"], ["attention_backend", "a"], ["model_dtype", "a"]], "key": "rho/tx-b/dtype=a,page=16,attn=a,model=a"}
      ],
      "decodes": [
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [["kv_cache_dtype", "b"], ["page_size", "16"], ["attention_backend", "b"], ["model_dtype", "a"]], "key": "rho/tx-b/dtype=b,page=16,attn=b,model=a"},
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [["kv_cache_dtype", "a"], ["page_size", "64"], ["attention_backend", "a"], ["model_dtype", "b"]], "key": "rho/tx-b/dtype=a,page=64,attn=a,model=b"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [["kv_cache_dtype", "page_size"]], "partners": [[]], "refusal": {"prefill": ["rho/tx-b/dtype=a,page=16,attn=a,model=a"], "decode": ["rho/tx-b/dtype=a,page=64,attn=a,model=b", "rho/tx-b/dtype=b,page=16,attn=b,model=a"], "mismatches": ["kv_cache_dtype", "page_size"]}},
          "lenient": {"verdicts": [["kv_cache_dtype", "page_size"]], "partners": [[]], "refusal": {"prefill": ["rho/tx-b/dtype=a,page=16,attn=a,model=a"], "decode": ["rho/tx-b/dtype=a,page=64,attn=a,model=b", "rho/tx-b/dtype=b,page=16,attn=b,model=a"], "mismatches": ["kv_cache_dtype", "page_size"]}},
          "off": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [["kv_cache_dtype", "page_size"]], "partners": [[]], "refusal": {"prefill": ["rho/tx-b/dtype=a,page=16,attn=a,model=a"], "decode": ["rho/tx-b/dtype=a,page=64,attn=a,model=b", "rho/tx-b/dtype=b,page=16,attn=b,model=a"], "mismatches": ["kv_cache_dtype", "page_size"]}},
          "lenient": {"verdicts": [["kv_cache_dtype", "page_size"]], "partners": [[]], "refusal": {"prefill": ["rho/tx-b/dtype=a,page=16,attn=a,model=a"], "decode": ["rho/tx-b/dtype=a,page=64,attn=a,model=b", "rho/tx-b/dtype=b,page=16,attn=b,model=a"], "mismatches": ["kv_cache_dtype", "page_size"]}},
          "off": {"verdicts": [[null, null]], "partners": [[0, 1]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-a-version-counts-strictly-where-both-report-one",
      "prefills": [
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": "1", "kv_layout": [], "key": "rho/tx-b/?"}
      ],
      "decodes": [
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": "1", "kv_layout": [], "key": "rho/tx-b/?"},
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": "0", "kv_layout": [], "key": "rho/tx-b/?"},
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "rho/tx-b/?"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [[null, "version", null]], "partners": [[0, 2]], "refusal": null},
          "lenient": {"verdicts": [[null, null, null]], "partners": [[0, 1, 2]], "refusal": null},
          "off": {"verdicts": [[null, null, null]], "partners": [[0, 1, 2]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [[null, "version", null]], "partners": [[0, 2]], "refusal": null},
          "lenient": {"verdicts": [[null, null, null]], "partners": [[0, 1, 2]], "refusal": null},
          "off": {"verdicts": [[null, null, null]], "partners": [[0, 1, 2]], "refusal": null}
        }
      }
    },
    {
      "name": "pair-two-versions-of-prefill-list-the-same-decodes",
      "prefills": [
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": "1", "kv_layout": [], "key": "rho/tx-b/?"},
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": "2", "kv_layout": [], "key": "rho/tx-b/?"}
      ],
      "decodes": [
        {"runtime": "rho", "protocol": null, "transport": "tx-b", "version": null, "kv_layout": [], "key": "rho/tx-b/?"}
      ],
      "outcome": {
        "grpc": {
          "strict": {"verdicts": [[null], [null]], "partners": [[0], [0]], "refusal": null},
          "lenient": {"verdicts": [[null], [null]], "partners": [[0], [0]], "refusal": null},
          "off": {"verdicts": [[null], [null]], "partners": [[0], [0]], "refusal": null}
        },
        "http": {
          "strict": {"verdicts": [[null], [null]], "partners": [[0], [0]], "refusal": null},
          "lenient": {"verdicts": [[null], [null]], "partners": [[0], [0]], "refusal": null},
          "off": {"verdicts": [[null], [null]], "partners": [[0], [0]], "refusal": null}
        }
      }
    }
  ]
}"#;
