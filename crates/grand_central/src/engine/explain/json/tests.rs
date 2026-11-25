//! The cases of a kept decision's JSON form.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::*;
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Pair, Pick, Queue, ShadowKind, Shed, Survivors, Targets, Trace},
        view::fingerprint,
    },
    store::GatewayId,
};

/// The preset `round_robin` under generation 7: the document a record's
/// names are read from.
fn standing() -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.generation = 7;
    AppliedSpec::validate(&document).expect("the document validates")
}

fn record(decision: Decision, binding: Option<u32>) -> DecisionRecord {
    DecisionRecord {
        decision,
        binding,
        at: Instant::now(),
        shadow: None,
        shadow_target: None,
        decode_shadow: None,
        decode_shadow_target: None,
        gateway: GatewayId::LOCAL,
        request: fingerprint(b"request-1"),
    }
}

/// A pair's pick of two targets under generation 7: the first names a
/// decode, the trace counts four survivors and scores the two.
fn pick() -> Decision {
    let mut targets = Targets::new();
    targets.push(Target {
        position: 2,
        id: WorkerId(u64::MAX),
        dp_rank: 3,
        pair: Some(Pair {
            decode_position: 1,
            decode_id: WorkerId(22),
            decode_dp_rank: 1,
            ..Pair::default()
        }),
    });
    targets.push(Target {
        position: 0,
        id: WorkerId(11),
        ..Target::default()
    });
    let mut top = [(WorkerId(0), 0.0); 3];
    top[0] = (WorkerId(u64::MAX), 0.75);
    top[1] = (WorkerId(11), 0.5);
    Decision::Pick(Pick {
        targets,
        trace: Trace {
            profile: 0,
            survivors: Survivors::counted(4),
            top,
            top_len: 2,
            seed: u64::MAX - 1,
            generation: 7,
            lifted: LiftedVeto::Overload,
        },
        ..Pick::default()
    })
}

/// A shed made ahead of every profile, under a generation the standing
/// document is not.
fn shed() -> Decision {
    Decision::Shed(Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 2,
        wait_allowed: true,
        trace: Trace {
            survivors: Survivors::NOT_COUNTED,
            generation: 6,
            ..Trace::default()
        },
    })
}

/// The host's address for the workers 11 and 22, and none for any other.
fn url(id: WorkerId) -> Option<String> {
    matches!(id.0, 11 | 22).then(|| format!("http://worker-{}:8000", id.0))
}

/// `record` as JSON, with its instant taken out: the instant, a decimal
/// string, checked to lie between `before` and now.
fn written(record: &DecisionRecord, spec: &AppliedSpec, before: SystemTime) -> Value {
    let mut json = record.json(spec.engine(), &url);
    let fields = json.as_object_mut().expect("an object");
    let at = fields.remove("at_unix_micros").expect("its instant");
    let micros = |at: SystemTime| {
        at.duration_since(UNIX_EPOCH)
            .expect("past the epoch")
            .as_micros()
    };
    let at: u128 = at
        .as_str()
        .expect("a decimal string")
        .parse()
        .expect("digits");
    assert!(micros(before) <= at && at <= micros(SystemTime::now()));
    json
}

/// A pick is written whole: its targets with the decode side of a pair, its
/// trace, its names under the standing generation, what its shadow said
/// and, in the trace, the worker that shadow picked. A 64-bit id, a seed
/// and a generation are decimal strings. A worker its host holds no more
/// has no address. No id of the request stands in it.
#[test]
fn a_pick_is_written_with_its_targets_its_trace_and_its_names() {
    let spec = standing();
    let before = SystemTime::now();
    let mut record = record(pick(), Some(0));
    record.shadow = Some(ShadowKind::OtherTarget);
    record.shadow_target = Some(WorkerId(u64::MAX - 2));
    let names = record.names(spec.engine()).expect("the standing names");
    assert_eq!(
        (names.plan, names.profile.as_str()),
        ("single", "round_robin")
    );
    let wanted = json!({
        "generation": "7",
        "binding": 0,
        "decision": {
            "pick": { "targets": [
                {
                    "id": "18446744073709551615",
                    "dp_rank": 3,
                    "pair": {
                        "decode_id": "22",
                        "decode_dp_rank": 1,
                        "decode_url": "http://worker-22:8000",
                    },
                },
                { "id": "11", "dp_rank": 0, "url": "http://worker-11:8000" },
            ] },
            "trace": {
                "profile": 0,
                "no_profile": false,
                "survivors": 4,
                "drew_first": false,
                "top": [
                    { "worker_id": "18446744073709551615", "score": 0.75 },
                    { "worker_id": "11", "score": 0.5, "url": "http://worker-11:8000" },
                ],
                "seed": "18446744073709551614",
                "generation": "7",
                "lifted": "LIFTED_VETO_OVERLOAD",
                "shadow_pick": "18446744073709551613",
            },
        },
        "plan_name": "single",
        "profile_name": "round_robin",
        "picker_name": names.picker,
        "filter_names": names.filters,
        "scorer_names": names.scorers,
        "shadow": "other-target",
    });
    assert_eq!(written(&record, &spec, before), wanted);
}

/// A shed is written with its reason in the contract's word. An option
/// that holds no value is left out: no binding matched, the generation
/// that decided is not the standing one, no survivor was counted, no
/// shadow was due and none picked a worker. A zero and an empty list are
/// written. A queue ticket is a decimal string.
#[test]
fn a_shed_is_written_with_its_reason_and_no_unset_option() {
    let spec = standing();
    let before = SystemTime::now();
    let wanted = json!({
        "generation": "6",
        "decision": {
            "shed": {
                "reason": "SHED_REASON_NO_ENDPOINTS",
                "emptied_by": "healthy",
                "retry_after_secs": 2,
                "wait_allowed": true,
            },
            "trace": {
                "profile": 0,
                "no_profile": true,
                "drew_first": true,
                "top": [],
                "seed": "0",
                "generation": "6",
                "lifted": "LIFTED_VETO_NONE",
            },
        },
    });
    assert_eq!(written(&record(shed(), None), &spec, before), wanted);

    let queue = Decision::Queue(Queue {
        ticket: u64::MAX,
        max_wait_micros: 9,
        trace: Trace::default(),
    });
    let json = written(&record(queue, None), &spec, before);
    let wanted = json!({ "ticket": "18446744073709551615", "max_wait_micros": 9 });
    assert_eq!(json["decision"]["queue"], wanted);
}

/// A score that is no number has no JSON form: its row is written without
/// the name, beside a row whose score is one.
#[test]
fn a_score_that_is_no_number_is_left_out_of_its_row() {
    let spec = standing();
    for unwritten in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let Decision::Pick(mut picked) = pick() else {
            panic!("the fixture is a pick");
        };
        picked.trace.top[1].1 = unwritten;
        let json = record(Decision::Pick(picked), Some(0)).json(spec.engine(), &url);
        let rows = &json["decision"]["trace"]["top"];
        assert_eq!(rows[0]["score"], json!(0.75), "{unwritten}");
        let wanted = json!({ "worker_id": "11", "url": "http://worker-11:8000" });
        assert_eq!(rows[1], wanted, "{unwritten}");
    }
}

/// The names of a wire message's fields, read from the print of a value
/// that holds none: `Name { a: 0, b: [] }`.
#[cfg(feature = "controller")]
fn wire_names(message: &impl std::fmt::Debug) -> Vec<String> {
    let print = format!("{message:?}");
    let fields = print.split_once('{').map_or("", |(_, fields)| fields);
    let fields = fields.rsplit_once('}').map_or("", |(fields, _)| fields);
    let names = fields
        .split(", ")
        .filter_map(|field| field.split_once(": "));
    names.map(|(name, _)| name.trim().to_owned()).collect()
}

/// Every name of `json`, an object, is a field of the wire message `wire`
/// or one of the names `beside`, the ones this form adds.
#[cfg(feature = "controller")]
fn held_to(json: &Value, wire: &impl std::fmt::Debug, beside: &[&str], at: &str) {
    let names = wire_names(wire);
    assert!(!names.is_empty(), "{at}: the wire message prints no field");
    for name in json.as_object().expect(at).keys() {
        let known = names.contains(name) || beside.contains(&name.as_str());
        assert!(
            known,
            "{at}: `{name}` is no field of the contract's message"
        );
    }
}

/// [`held_to`], and the other way: every field of the wire message `wire`
/// is a name of `json` but the ones of `left_out`, given in their order.
/// `json` is one that holds every option.
#[cfg(feature = "controller")]
fn closed(
    json: &Value,
    wire: &impl std::fmt::Debug,
    (beside, left_out): (&[&str], &[&str]),
    at: &str,
) {
    held_to(json, wire, beside, at);
    let fields = json.as_object().expect(at);
    let mut absent = wire_names(wire);
    absent.retain(|name| !fields.contains_key(name));
    absent.sort_unstable();
    assert_eq!(absent, left_out, "{at}: the contract's fields left out");
}

/// The JSON's names are the contract's at every level, for a pick and for a
/// shed, so that the two forms cannot part unseen. Two lists are closed.
/// The names this form adds: `shadow` at the record, `url` at a target and
/// at a top row, `decode_url` at a pair. And the contract's fields it
/// leaves out, held for a pick that holds every option. At the record
/// those are the two ids of the request, which the record does not hold,
/// and six fields that no kept record holds and the control call leaves
/// unset. Below it they are a decision's leases, five fields of a trace and
/// two of a top row, which no trace holds. A decision's outcome stands
/// under the contract's word for it, and so does each reason and each
/// lifted veto.
#[cfg(feature = "controller")]
#[test]
fn every_name_is_the_contracts_but_two_closed_lists() {
    use crate::service::proto::{self, decision::Outcome};

    let spec = standing();
    let named = |outcome: &Outcome| {
        let print = format!("{outcome:?}");
        print.split('(').next().expect("a variant").to_lowercase()
    };
    let mut kept = record(pick(), Some(0));
    kept.shadow = Some(ShadowKind::Agreed);
    kept.shadow_target = Some(WorkerId(11));
    let outcomes = [
        (kept, Outcome::Pick(proto::Pick::default())),
        (record(shed(), None), Outcome::Shed(proto::Shed::default())),
    ];
    for (record, outcome) in outcomes {
        let json = record.json(spec.engine(), &url);
        held_to(
            &json,
            &proto::DecisionRecord::default(),
            &["shadow"],
            "the record",
        );
        let decision = &json["decision"];
        let word = named(&outcome);
        let fields = decision.as_object().expect("a decision");
        let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
        names.sort_unstable();
        let mut wanted = [word.as_str(), "trace"];
        wanted.sort_unstable();
        assert_eq!(names, wanted, "a decision's outcome and its trace");
        let mut wire = wire_names(&proto::Decision::default());
        wire.sort_unstable();
        assert_eq!(wire, ["leases", "outcome", "trace"], "a decision");
        let trace = &decision["trace"];
        match outcome {
            Outcome::Pick(pick) => {
                let unheld = [
                    "candidates",
                    "decision_id",
                    "gateway_id",
                    "outcome",
                    "projections",
                    "request",
                    "request_id",
                    "view_digest",
                ];
                let whole = &proto::DecisionRecord::default();
                closed(&json, whole, (&["shadow"], &unheld), "the record");
                closed(&decision[&word], &pick, (&[], &[]), "a pick");
                let targets = decision[&word]["targets"].as_array().expect("its targets");
                assert_eq!(targets.len(), 2);
                let target = &proto::Target::default();
                closed(&targets[0], target, (&[], &[]), "a pair's target");
                held_to(&targets[1], target, &["url"], "a target");
                let pair = &targets[0]["pair"];
                let sides = &proto::Pair::default();
                closed(pair, sides, (&["decode_url"], &[]), "a pair");
                let untraced = ["affinity_source", "drops", "fallback", "picker", "plan"];
                let traced = &proto::Trace::default();
                closed(trace, traced, (&[], &untraced), "a trace");
                let rows = trace["top"].as_array().expect("its top rows");
                let row = &proto::Scored::default();
                let unscored = ["contributions", "dp_rank"];
                closed(&rows[1], row, (&["url"], &unscored), "a top row");
            }
            Outcome::Shed(shed) => closed(&decision[&word], &shed, (&[], &[]), "a shed"),
            Outcome::Queue(_) => panic!("no queue is asked here"),
        }
        held_to(trace, &proto::Trace::default(), &[], "a trace");
        for row in trace["top"].as_array().expect("its top rows") {
            held_to(row, &proto::Scored::default(), &["url"], "a top row");
        }
    }

    let reasons = [
        (ShedReason::Saturated, proto::ShedReason::Saturated),
        (ShedReason::FlowLimit, proto::ShedReason::FlowLimit),
        (ShedReason::NoEndpoints, proto::ShedReason::NoEndpoints),
        (
            ShedReason::PolicyDeclined,
            proto::ShedReason::PolicyDeclined,
        ),
        (ShedReason::Deadline, proto::ShedReason::Deadline),
    ];
    for (reason, wire) in reasons {
        assert_eq!(shed_reason(reason), wire.as_str_name());
    }
    for veto in [LiftedVeto::None, LiftedVeto::Overload, LiftedVeto::Liveness] {
        assert_eq!(lifted(veto), proto::LiftedVeto::from(veto).as_str_name());
    }
}

/// Whether a field of a wire message is a number of 64 bits, by its type.
#[cfg(feature = "controller")]
fn wide<T: 'static>(_: &T) -> bool {
    use std::any::TypeId;

    let of = TypeId::of::<T>();
    of == TypeId::of::<u64>() || of == TypeId::of::<i64>()
}

/// A number the contract carries in 64 bits is a decimal string in the
/// JSON, and a narrower one is a JSON number. Each number's form is held to
/// the type of the wire message's own field, so a field whose width changes
/// turns the case red.
#[cfg(feature = "controller")]
#[test]
fn a_number_is_written_by_the_width_the_contract_carries_it_in() {
    use crate::service::proto;

    let spec = standing();
    let waits = Decision::Queue(Queue {
        ticket: 7,
        max_wait_micros: 9,
        trace: Trace::default(),
    });
    let mut picked = record(pick(), Some(0));
    picked.shadow_target = Some(WorkerId(11));
    let picked = picked.json(spec.engine(), &url);
    let refused = record(shed(), None).json(spec.engine(), &url);
    let queued = record(waits, None).json(spec.engine(), &url);
    let (kept, trace) = (proto::DecisionRecord::default(), proto::Trace::default());
    let (target, pair) = (proto::Target::default(), proto::Pair::default());
    let (row, shedding) = (proto::Scored::default(), proto::Shed::default());
    let queue = proto::Queue::default();
    let first = &picked["decision"]["pick"]["targets"][0];
    let traced = &picked["decision"]["trace"];
    let stopped = &refused["decision"]["shed"];
    let held = &queued["decision"]["queue"];
    let numbers = [
        ("generation", &picked["generation"], wide(&kept.generation)),
        ("binding", &picked["binding"], wide(&kept.binding)),
        (
            "at_unix_micros",
            &picked["at_unix_micros"],
            wide(&kept.at_unix_micros),
        ),
        ("id", &first["id"], wide(&target.id)),
        ("dp_rank", &first["dp_rank"], wide(&target.dp_rank)),
        (
            "decode_id",
            &first["pair"]["decode_id"],
            wide(&pair.decode_id),
        ),
        (
            "decode_dp_rank",
            &first["pair"]["decode_dp_rank"],
            wide(&pair.decode_dp_rank),
        ),
        ("profile", &traced["profile"], wide(&trace.profile)),
        (
            "survivors",
            &traced["survivors"],
            wide(&trace.survivors.unwrap_or_default()),
        ),
        ("seed", &traced["seed"], wide(&trace.seed)),
        (
            "a trace's generation",
            &traced["generation"],
            wide(&trace.generation),
        ),
        (
            "shadow_pick",
            &traced["shadow_pick"],
            wide(&trace.shadow_pick),
        ),
        (
            "worker_id",
            &traced["top"][0]["worker_id"],
            wide(&row.worker_id),
        ),
        (
            "retry_after_secs",
            &stopped["retry_after_secs"],
            wide(&shedding.retry_after_secs),
        ),
        ("ticket", &held["ticket"], wide(&queue.ticket)),
        (
            "max_wait_micros",
            &held["max_wait_micros"],
            wide(&queue.max_wait_micros),
        ),
    ];
    for (name, written, wide) in numbers {
        let form = (written.is_string(), written.is_number());
        assert_eq!(form, (wide, !wide), "{name}: {written}");
    }
}

/// A trace's values are the ones its wire form states, field by field: the
/// JSON is held to the trace's own mapping to the contract, for a pick and
/// for a shed.
#[cfg(feature = "controller")]
#[test]
fn a_traces_values_are_the_ones_its_wire_form_states() {
    use crate::service::proto;

    let spec = standing();
    for decision in [pick(), shed()] {
        let wire = decision.trace().wire();
        let json = record(decision, Some(0)).json(spec.engine(), &url);
        let trace = &json["decision"]["trace"];
        let lifted = proto::LiftedVeto::try_from(wire.lifted).expect("a veto");
        let wanted = json!({
            "profile": wire.profile,
            "no_profile": wire.no_profile,
            "drew_first": wire.drew_first,
            "seed": wire.seed.to_string(),
            "generation": wire.generation.to_string(),
            "lifted": lifted.as_str_name(),
        });
        for (name, value) in wanted.as_object().expect("an object") {
            assert_eq!(&trace[name], value, "{name}");
        }
        let survivors = trace.get("survivors").and_then(Value::as_u64);
        assert_eq!(survivors, wire.survivors.map(u64::from));
        let rows = trace["top"].as_array().expect("its top rows");
        assert_eq!(rows.len(), wire.top.len());
        for (written, row) in rows.iter().zip(&wire.top) {
            assert_eq!(written["worker_id"], row.worker_id.to_string());
            assert_eq!(written["score"], json!(row.score));
        }
    }
}
