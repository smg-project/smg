//! The Explain call: what the leader answers for a decision's key, and
//! what it refuses.

use super::*;

const EXPLAIN_GATEWAY: &str = "gateway-a";

fn explain_request(id: u8) -> Request<proto::ExplainRequest> {
    from(
        "operator-a",
        proto::ExplainRequest {
            key: Some(proto::explain_request::Key::RequestId(vec![id])),
            gateway_id: EXPLAIN_GATEWAY.to_owned(),
        },
    )
}

fn explain_spec(service: &Service, profile: &str, sample: u32) {
    let document = format!(
            "apiVersion: smg.policy/v1\nprofiles:\n  a-other: {{pick: random}}\n  {profile}: {{filter: [healthy], score: [least-load], pick: best}}\nbindings:\n  - match: {{model: other}}\n    profile: a-other\n  - profile: {profile}\nexplain: {{sample: {sample}}}"
        );
    service
        .shared
        .generations
        .apply(document.into_bytes(), Format::Yaml)
        .expect("a valid document");
}

async fn decide_for_explain(service: &Service, request: proto::DecideRequest) -> proto::Decision {
    let (answers, mut received) = mpsc::channel(1);
    let job = Job {
        request: Checked::new(request).expect("a frame"),
        gateway: gateway_id(EXPLAIN_GATEWAY),
        received: Instant::now(),
        answers,
    };
    assert!(service.shared.deciders.submit(job).is_none());
    let answered = tokio::time::timeout(Duration::from_secs(5), received.recv())
        .await
        .expect("an answer in time")
        .expect("an answer")
        .frame();
    match answered.answer {
        Some(proto::decide_answer::Answer::Decision(decision)) => decision,
        other => panic!("a decision: {other:?}"),
    }
}

#[tokio::test]
async fn explain_echoes_the_key_and_decision_and_names_only_the_standing_generation() {
    let (_elector, _standing, control) = electing(true);
    let _ = bed::fleet(&control.shared.store, &control.shared.objects);
    explain_spec(&control, "z-first", 1);

    for (model, id) in [(bed::MODEL, 1), ("empty", 2)] {
        let frame = bed::frame_for(model, id, u64::from(id));
        let mut decided = decide_for_explain(&control, frame).await;
        assert_eq!(!decided.leases.is_empty(), model == bed::MODEL);
        decided.leases.clear();
        let record = control
            .explain(explain_request(id))
            .await
            .expect("a hit")
            .into_inner();
        assert_eq!(record.request_id, [id]);
        assert_eq!(record.gateway_id, EXPLAIN_GATEWAY);
        assert_eq!((record.generation, record.binding), (1, 1));
        assert_eq!(record.decision, Some(decided));
        assert!(record.at_unix_micros > 0);
        // A model no worker serves is shed before a profile is chosen:
        // its record names none.
        let served = model == bed::MODEL;
        let (profile, picker): (&str, &str) = if served {
            ("z-first", "best")
        } else {
            ("", "")
        };
        let (filters, scorers): (&[&str], &[&str]) = if served {
            (&["healthy"], &["least-load"])
        } else {
            (&[], &[])
        };
        assert_eq!(record.profile_name, profile);
        assert_eq!(record.picker_name, picker);
        assert_eq!(record.filter_names, filters);
        assert_eq!(record.scorer_names, scorers);
        if served {
            assert_eq!(record.plan_name, "single");
        }
        assert_eq!((record.decision_id, record.view_digest), (0, 0));
        assert!(record.candidates.is_empty() && record.projections.is_empty());
        assert!(record.request.is_none() && record.outcome.is_none());
    }

    let mut wrong_gateway = explain_request(1);
    "gateway-b".clone_into(&mut wrong_gateway.get_mut().gateway_id);
    let miss = control
        .explain(wrong_gateway)
        .await
        .expect_err("a different gateway");
    assert_eq!(miss.code(), Code::NotFound);

    explain_spec(&control, "z-second", 0);
    let _ = decide_for_explain(&control, bed::frame(3, 3)).await;
    let miss = control
        .explain(explain_request(3))
        .await
        .expect_err("sampling off");
    assert_eq!(miss.code(), Code::NotFound);
    let old = control
        .explain(explain_request(1))
        .await
        .expect("an older hit")
        .into_inner();
    assert_eq!((old.generation, old.binding), (1, 1));
    assert_eq!(
        old.decision
            .expect("the decision")
            .trace
            .expect("the trace")
            .profile,
        1
    );
    assert!(old.profile_name.is_empty() && old.plan_name.is_empty() && old.picker_name.is_empty());
    assert!(old.filter_names.is_empty() && old.scorer_names.is_empty());
}

#[tokio::test]
async fn explain_names_only_the_plan_for_a_shed_ahead_of_the_first_profile() {
    let (_elector, _standing, control) = electing(true);
    let _ = bed::fleet(&control.shared.store, &control.shared.objects);
    explain_spec(&control, "a-first", 1);

    let mut frame = bed::frame(1, 1);
    frame.constraints = Some(proto::Constraints {
        subset_hint: Some(0),
        ..proto::Constraints::default()
    });
    let decision = decide_for_explain(&control, frame).await;
    assert!(matches!(
        decision.outcome,
        Some(proto::decision::Outcome::Shed(shed)) if shed.emptied_by == "subset-hint"
    ));

    let record = control
        .explain(explain_request(1))
        .await
        .expect("the sampled shed")
        .into_inner();
    assert_eq!(record.plan_name, "single");
    assert!(record.profile_name.is_empty(), "{}", record.profile_name);
    assert!(record.picker_name.is_empty());
    assert!(record.filter_names.is_empty());
    assert!(record.scorer_names.is_empty());
    let trace = record
        .decision
        .and_then(|decision| decision.trace)
        .expect("the trace");
    assert_eq!((trace.profile, trace.no_profile), (0, true));
}

#[tokio::test]
async fn explain_refuses_unsupported_or_incomplete_keys_by_field() {
    use proto::explain_request::Key;

    let (_elector, _standing, control) = electing(true);
    for (key, gateway, field) in [
        (Some(Key::DecisionId(0)), "", "decision_id"),
        (None, EXPLAIN_GATEWAY, "request_id"),
        (Some(Key::RequestId(vec![])), EXPLAIN_GATEWAY, "request_id"),
        (
            Some(Key::RequestId(vec![0; 257])),
            EXPLAIN_GATEWAY,
            "request_id",
        ),
        (Some(Key::RequestId(vec![1])), "", "gateway_id"),
    ] {
        let request = from(
            "operator-a",
            proto::ExplainRequest {
                key,
                gateway_id: gateway.to_owned(),
            },
        );
        let refused = control.explain(request).await.expect_err("an invalid key");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert_eq!(refused.message().split(':').next(), Some(field));
    }
}

#[tokio::test]
async fn explain_is_for_the_leader_and_a_new_term_starts_empty() {
    let (mut elector, standing, control) = electing(true);
    explain_spec(&control, "z-first", 1);
    let _ = decide_for_explain(&control, bed::frame_for("empty", 1, 1)).await;
    assert!(control.explain(explain_request(1)).await.is_ok());
    let no_principal = control
        .explain(Request::new(explain_request(1).into_inner()))
        .await;
    assert_eq!(
        no_principal.expect_err("the call needs a principal").code(),
        Code::Unauthenticated
    );
    let gateway = Service::new(vec![Scope::Gateway], Arc::clone(&control.shared));
    assert_eq!(
        gateway
            .explain(explain_request(1))
            .await
            .expect_err("control scope only")
            .code(),
        Code::PermissionDenied
    );

    assert!(elector.release().expect("the release"));
    standing.send_replace(saw("controller-1@10.0.0.8:30100", 2));
    let mut invalid = explain_request(1);
    invalid.get_mut().key = None;
    for request in [explain_request(1), invalid] {
        let lost = control
            .explain(request)
            .await
            .expect_err("this replica follows");
        assert_eq!(lost.code(), Code::FailedPrecondition);
        let details = RpcStatus::decode(lost.details()).expect("a google.rpc.Status");
        let [detail] = details.details.as_slice() else {
            panic!("one detail: {details:?}");
        };
        let named = proto::NotLeader::decode(detail.value.as_slice()).expect("a NotLeader");
        assert_eq!(named.leader_address, "10.0.0.8:30100");
    }

    elector.lead_at(Instant::now());
    control.shared.store.set_term(2);
    let miss = control
        .explain(explain_request(1))
        .await
        .expect_err("a new lead");
    assert_eq!(miss.code(), Code::NotFound);
    let _ = decide_for_explain(&control, bed::frame_for("empty", 2, 2)).await;
    assert!(control.explain(explain_request(2)).await.is_ok());
    let miss = control
        .explain(explain_request(1))
        .await
        .expect_err("the old record stayed gone");
    assert_eq!(miss.code(), Code::NotFound);
}
