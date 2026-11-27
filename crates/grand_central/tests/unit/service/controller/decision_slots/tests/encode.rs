//! The cases of an encode request through the deciding threads, and the
//! fleet and the frames the frame decision's encode cases share with them.

use std::sync::atomic::Ordering;

use super::{
    bed, booked, busy, frame_for, leading, outcome, pair_of, phased, written_ready, GATEWAY, MODEL,
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        view::{Role, WireModes, WorkerId},
        DemandFacts,
    },
    service::proto::{self, busy::Reason, decision::Outcome},
    stages::registry::StageRegistry,
    store::{
        EncodeHandoff, EncodeOffer, GatewayId, LeasesStore, MemoryStore, Reachability, WorkerFacts,
    },
    worker_host::{Members, Objects},
};

/// The model of the encode request's cases.
pub(crate) const ENCODED: &str = "model-encoded";

/// The encode plan's document as `generation`, with the stages of
/// `catalog`: an encode request's three sides, each among the workers its
/// health filter keeps, by its own picker: the encoders' first of `picks`,
/// then the prefills', then the decodes'.
pub(crate) fn encode_plan_in(
    catalog: &StageRegistry,
    generation: u64,
    picks: [&str; 3],
) -> AppliedSpec {
    let [encoders, prefills, decodes] = picks;
    let document = format!(
        r#"{{
          "apiVersion": "smg.policy/v1",
          "generation": {generation},
          "profiles": {{
            "encoders": {{ "filter": ["healthy"], "pick": "{encoders}" }},
            "prefills": {{ "filter": ["healthy"], "pick": "{prefills}" }},
            "decodes": {{ "filter": ["healthy"], "pick": "{decodes}" }}
          }},
          "bindings": [
            {{ "plan": "epd",
              "profiles": {{ "encode": "encoders", "prefill": "prefills", "decode": "decodes" }} }}
          ]
        }}"#
    );
    let document: PolicySpec = serde_json::from_str(&document).expect("a document of this build");
    let digest = document.digest().expect("the document digests");
    AppliedSpec::validate_digested(&document, digest, catalog, &[])
        .expect("the catalog holds every stage the document names")
}

/// The encode plan's document of this build's own stages as generation 1:
/// each side by a seeded draw.
pub(crate) fn encode_plan() -> AppliedSpec {
    encode_plan_in(&StageRegistry::builtin(), 1, ["random"; 3])
}

/// What a member of the cases states beside its role: the wires it is
/// dialed on, the one gateway that dials it where one alone does, and
/// whether it states its end of the encode handoff.
#[derive(Clone, Copy)]
pub(crate) struct Stated {
    pub(crate) wires: WireModes,
    pub(crate) only: Option<GatewayId>,
    pub(crate) pushes: bool,
}

/// A member every gateway dials on the HTTP wire, with its end stated.
pub(crate) const PUSHING: Stated = Stated {
    wires: WireModes::HTTP,
    only: None,
    pushes: true,
};

/// A ready member `name` of `model` in `role` that states `stated`,
/// written through `objects`. An encoder and a prefill that state their
/// end state one push between them, under one runtime: one encode class,
/// which every such prefill names.
pub(crate) fn member_of(
    store: &MemoryStore,
    objects: &Objects<Members>,
    model: &str,
    name: &str,
    role: Role,
    stated: Stated,
) -> WorkerId {
    member_stating(
        store,
        objects,
        model,
        (name, role),
        stated,
        DemandFacts::default(),
    )
}

/// [`member_of`] for a worker that states `demands` for a request's hard
/// demands.
pub(crate) fn member_stating(
    store: &MemoryStore,
    objects: &Objects<Members>,
    model: &str,
    (name, role): (&str, Role),
    stated: Stated,
    demands: DemandFacts,
) -> WorkerId {
    let key = format!("http://{name}:8000");
    let encode = match role {
        Role::Encode | Role::Prefill if stated.pushes => EncodeOffer {
            handoff: EncodeHandoff::Push,
            producer: role == Role::Encode,
            consumer: role == Role::Prefill,
            connector: Some("a".to_owned()),
            references: false,
        },
        _ => EncodeOffer::default(),
    };
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        engine_kind: "one".to_owned(),
        incarnation: 1,
        role,
        wire_modes: stated.wires,
        models: vec![model.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        reachable_from: stated.only.map_or(Reachability::Any, Reachability::Gateway),
        encode,
        demand_facts: demands,
        ..WorkerFacts::default()
    };
    written_ready(store, objects, facts)
}

/// The `n`th encode request's frame for `model`: a frame of the encode
/// phase that carries `keys` keys, each a hash of its own, and what its
/// gateway states of itself.
pub(crate) fn frame_of(
    model: &str,
    n: u8,
    keys: u64,
    wires: WireModes,
    hint: Option<u32>,
) -> proto::DecideRequest {
    let mut frame = phased(frame_for(model, n, u64::from(n)), proto::Phase::Encode);
    frame.constraints = Some(proto::Constraints {
        wire_modes: u32::from(wires.bits()),
        subset_hint: hint,
        ..proto::Constraints::default()
    });
    let key = |key: u64| proto::Identity {
        form: Some(proto::identity::Form::Hash((key << 8) | u64::from(n))),
    };
    frame.encode = Some(proto::EncodeItems {
        keys: (1..=keys).map(key).collect(),
        ..proto::EncodeItems::default()
    });
    frame
}

/// An encode request that carries its items is decided by the deciding
/// threads, as its pair's one request. Its answer carries the pick, a lease
/// a side of the pair under one id, and each item's encoder beside them. No
/// lease stands for an encoder and none books one. A frame that states the
/// later decode is answered the prefill side's lease alone, as a pair's
/// frame is. A frame of the encode phase that does not state
/// `DecideRequest.encode` is still answered `Busy` with no reason and
/// counted with the sides, and its answer names no encoder.
#[tokio::test]
async fn a_served_encode_request_is_answered_its_pairs_leases_and_each_items_encoder() {
    let mut bed = bed(Some(encode_plan()), true, (2, 8), []);
    let member =
        |name: &str, role| member_of(&bed.store, &bed.reads.objects, ENCODED, name, role, PUSHING);
    let p = member("ep0", Role::Prefill);
    let d = member("ed0", Role::Decode);
    let e = member("ee0", Role::Encode);
    let whole = frame_of(ENCODED, 1, 2, WireModes::NONE, None);
    let answer = bed.ask(whole).await;
    assert_eq!(answer.request_id, [1]);
    let (targets, sides) = pair_of(&answer);
    assert_eq!(targets, [(p.0, Some(d.0))]);
    let [(prefill, proto::Side::Prefill, 0), (decode, proto::Side::Decode, 0)] = sides[..] else {
        panic!("a lease a side of the pair: {sides:?}");
    };
    assert_eq!(prefill, decode, "one id for the pair");
    let placed = answer.encode.as_ref().expect("where the items go");
    let encoders: Vec<u64> = placed.encoders.iter().map(|encoder| encoder.id).collect();
    assert_eq!((encoders, placed.one), (vec![e.0, e.0], false));
    assert_eq!(
        booked(&bed, [p, d, e]),
        [1, 1, 0],
        "no lease books an encoder"
    );
    assert_eq!(bed.store.live(GATEWAY), 1);

    let mut later = frame_of(ENCODED, 2, 1, WireModes::NONE, None);
    later.decode_later = true;
    let answer = bed.ask(later).await;
    let (_, sides) = pair_of(&answer);
    assert!(
        matches!(sides[..], [(_, proto::Side::Prefill, 0)]),
        "the prefill side's lease alone: {sides:?}"
    );
    let placed = answer.encode.as_ref().expect("where the item goes");
    assert_eq!(placed.encoders.len(), 1);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);

    let mut bare = frame_of(ENCODED, 3, 1, WireModes::NONE, None);
    bare.encode = None;
    let answer = bed.ask(bare).await;
    assert_eq!(busy(&answer), Reason::Unspecified);
    assert_eq!(answer.encode, None);
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 1);
    assert_eq!(bed.store.live(GATEWAY), 2);
}

/// A frame with items never stands alone. Under a plan that pairs no side
/// a prefill frame is decided alone, over the prefills, and is filed a
/// single's lease. The same model's frame with items is handed to the
/// engine with them, which declines it by its binding: nothing is filed,
/// and its answer names no encoder.
#[tokio::test]
async fn a_served_frame_with_items_is_not_decided_alone_under_a_plan_that_pairs_no_side() {
    let mut bed = leading([]);
    let alone = phased(frame_for(MODEL, 1, 1), proto::Phase::Prefill);
    let answer = bed.ask(alone).await;
    assert!(matches!(outcome(&answer), Outcome::Pick(_)), "{answer:?}");
    assert_eq!(bed.store.live(GATEWAY), 1);
    for keys in [0, 1] {
        let answer = bed
            .ask(frame_of(MODEL, 2, keys, WireModes::NONE, None))
            .await;
        let Outcome::Shed(shed) = outcome(&answer) else {
            panic!("{keys} keys: a shed: {answer:?}");
        };
        let declined = (proto::ShedReason::PolicyDeclined, "binding");
        assert_eq!((shed.reason(), shed.emptied_by.as_str()), declined);
        assert_eq!(answer.encode, None, "{keys} keys");
    }
    assert_eq!(bed.store.live(GATEWAY), 1, "nothing more is filed");
    assert_eq!(bed.counts.pair_sides.load(Ordering::Relaxed), 0);
}
