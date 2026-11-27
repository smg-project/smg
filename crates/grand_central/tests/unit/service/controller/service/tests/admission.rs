//! Which calls a listener admits: the scopes it mounts and the principal
//! its layer placed, for every method of the contract.

use super::*;

impl Call {
    /// Every call, in the contract's order.
    const ALL: [Call; 14] = [
        Call::Session,
        Call::Decide,
        Call::WatchWorkers,
        Call::RegisterWorker,
        Call::PatchWorker,
        Call::DeregisterWorker,
        Call::GetSnapshot,
        Call::Explain,
        Call::ApplyPolicySpec,
        Call::GetPolicySpec,
        Call::RollbackPolicySpec,
        Call::StepDown,
        Call::WatchRegime,
        Call::Replicate,
    ];
}

/// The table a second time, copied by hand from the contract's header:
/// a change of a call's scopes is made in both.
#[test]
fn every_call_keeps_the_scopes_copied_here_from_the_contracts_header() {
    let expected: [(&str, &[Scope]); 14] = [
        ("Session", &[Scope::Gateway]),
        ("Decide", &[Scope::Gateway]),
        ("WatchWorkers", &[Scope::Gateway]),
        ("RegisterWorker", &[Scope::Control]),
        ("PatchWorker", &[Scope::Control]),
        ("DeregisterWorker", &[Scope::Control]),
        ("GetSnapshot", &[Scope::Control]),
        ("Explain", &[Scope::Control]),
        ("ApplyPolicySpec", &[Scope::Control]),
        ("GetPolicySpec", &[Scope::Control]),
        ("RollbackPolicySpec", &[Scope::Control]),
        ("StepDown", &[Scope::Control]),
        ("WatchRegime", &[Scope::Control]),
        ("Replicate", &[Scope::Replication]),
    ];
    assert_eq!(Call::ALL.map(|call| (call.name(), call.scopes())), expected);
}

/// Admission hands back the principal the layer placed, on the scopes
/// the listener mounts. Off them the call is denied whoever asks; on
/// them a call with no principal is refused as unauthenticated.
#[test]
fn a_listener_admits_the_calls_of_its_scopes_from_a_caller_with_a_principal() {
    let listeners: [&[Scope]; 4] = [
        &[Scope::Gateway],
        &[Scope::Control],
        &[Scope::Replication],
        &Scope::ALL,
    ];
    for scopes in listeners {
        let (_elector, _standing, service) = service(scopes);
        for call in Call::ALL {
            let mounted = call.scopes().iter().any(|scope| scopes.contains(scope));
            let placed = from("gateway-a", ());
            match service.admit(call, &placed) {
                Ok(principal) => {
                    assert!(mounted, "{} admitted on {scopes:?}", call.name());
                    assert_eq!(principal, &told("gateway-a"));
                }
                Err(status) => {
                    assert!(!mounted, "{} refused on {scopes:?}", call.name());
                    assert_eq!(status.code(), Code::PermissionDenied);
                    assert_eq!(
                        status.message(),
                        format!("{} is not a call of this listener's scopes", call.name())
                    );
                }
            }
            let status = service
                .admit(call, &Request::new(()))
                .expect_err("no principal stands on the call");
            let code = if mounted {
                Code::Unauthenticated
            } else {
                Code::PermissionDenied
            };
            assert_eq!(status.code(), code, "{} on {scopes:?}", call.name());
            if mounted {
                assert_eq!(status.message(), NO_PRINCIPAL);
            }
        }
    }
}

#[tokio::test]
async fn an_unserved_call_is_denied_off_its_scope_and_unimplemented_to_a_principal_on_it() {
    let (_elector, _standing, gateway) = service(&[Scope::Gateway]);
    let (_elector, _standing, control) = service(&[Scope::Control]);
    let (_elector, _standing, replication) = service(&[Scope::Replication]);

    let read = || from("operator-a", proto::GetPolicySpecRequest::default());
    let denied = gateway.get_policy_spec(read()).await.expect_err("refused");
    assert_eq!(denied.code(), Code::PermissionDenied);
    let unserved = control.get_policy_spec(read()).await.expect_err("refused");
    assert_eq!(unserved.code(), Code::Unimplemented);
    assert_eq!(
        unserved.message(),
        "GetPolicySpec is not served by a controller started with no spec source"
    );
    // A caller no layer placed is refused before the call is named
    // unserved: no call body is reached without a principal.
    let unplaced = control
        .get_policy_spec(Request::new(proto::GetPolicySpecRequest::default()))
        .await
        .expect_err("refused");
    assert_eq!(unplaced.code(), Code::Unauthenticated);
    assert_eq!(unplaced.message(), NO_PRINCIPAL);

    // The snapshot is the operators' read and the leader's to serve:
    // this replica does not lead. A standby is restored on its
    // Replicate stream: the replication scope does not make the call.
    let snapshot = || from("standby-a", proto::SnapshotRequest::default());
    for (listener, code) in [
        (&gateway, Code::PermissionDenied),
        (&control, Code::Unavailable),
        (&replication, Code::PermissionDenied),
    ] {
        let status = listener
            .get_snapshot(snapshot())
            .await
            .expect_err("refused");
        assert_eq!(status.code(), code);
    }
}

/// The path the transport routes `Decide` by is the contract's: its
/// service's name, then its method's.
#[test]
fn the_decide_calls_path_is_the_contracts_services_and_its_methods() {
    assert!(methods().contains(&"Decide"));
    assert_eq!(
        super::super::DECIDE_PATH,
        format!("/{}/Decide", controller::SERVICE_NAME)
    );
}

/// The methods of the contract's service, by the contract's own text.
fn methods() -> Vec<&'static str> {
    const CONTRACT: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/contracts/grand_central.proto"
    ));
    CONTRACT
        .lines()
        .skip_while(|line| !line.starts_with("service GrandCentral"))
        .take_while(|line| !line.starts_with('}'))
        .filter_map(|line| line.trim_start().strip_prefix("rpc "))
        .filter_map(|rest| rest.split('(').next())
        .collect()
}

/// Every method of the contract, called by its own name over a socket
/// on a listener of each scope set, three ways: a name stated to a
/// listener with no layer, a name stated to one that takes identities
/// as told, and no name stated to that one. `Leader` is served every
/// way; any other answers in its own name: denied off the listener's
/// scopes, and on them unauthenticated to a caller the layer did not
/// place. To the caller it placed, a call no step serves is
/// unimplemented, `Session` is reached (the empty message is no
/// `Hello`), and registration, watching rows, snapshots and Explain
/// are refused by a replica that does not lead. The names are read
/// from the contract, so a method added there joins the case. A name
/// the contract dropped is no method: whoever calls it, the transport
/// answers unimplemented and no body admits the call first.
#[tokio::test]
async fn every_method_of_the_contract_answers_by_its_calls_scopes_and_its_callers_principal() {
    /// The relays' two calls, which left the contract.
    const DROPPED: [&str; 2] = ["PushLoad", "PushKvEvents"];
    let methods = methods();
    assert_eq!(
        methods.len(),
        Call::ALL.len() + 1,
        "one call a method, and Leader: {methods:?}"
    );
    let listeners = [
        &[Scope::Gateway][..],
        &[Scope::Control],
        &[Scope::Replication],
    ];
    let ways = [
        (Layer::Off, true),
        (Layer::AsTold, true),
        (Layer::AsTold, false),
    ];
    for (scopes, (layer, stated)) in listeners
        .into_iter()
        .flat_map(|scopes| ways.map(|way| (scopes, way)))
    {
        let (_elector, _standing, service) = service(scopes);
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a loopback port");
        let addr = listener.local_addr().expect("the bound address");
        // Aborted at its drop, at the end of this listener's round.
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(service.routes(layer))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("a uri")
            .connect()
            .await
            .expect("the listener accepts");
        let mut client = Grpc::new(channel);
        for method in methods.iter().chain(&DROPPED) {
            let path = PathAndQuery::try_from(format!("/smg.policy.v1.GrandCentral/{method}"))
                .expect("a path");
            client.ready().await.expect("the channel is ready");
            // An empty message is every request's default.
            let request = if stated {
                stating("gateway-a")
            } else {
                Request::new(())
            };
            let answered = client
                .unary(request, path, ProstCodec::<(), ()>::default())
                .await;
            let round = format!("{method} on {scopes:?}, {layer:?}, a name stated: {stated}");
            if DROPPED.contains(method) {
                let status = answered.expect_err("the contract holds no such method");
                assert_eq!(
                    (status.code(), status.message()),
                    (Code::Unimplemented, ""),
                    "{round}"
                );
                continue;
            }
            if *method == "Leader" {
                assert!(answered.is_ok(), "{round}: {answered:?}");
                continue;
            }
            let call = Call::ALL.into_iter().find(|call| call.name() == *method);
            assert!(call.is_some(), "{method} is no call of the table");
            let mounted = call
                .into_iter()
                .flat_map(Call::scopes)
                .any(|scope| scopes.contains(scope));
            let expected = if !mounted {
                (
                    Code::PermissionDenied,
                    format!("{method} is not a call of this listener's scopes"),
                )
            } else if layer == Layer::AsTold && stated && *method == "Session" {
                (
                        Code::FailedPrecondition,
                        "no session stands on this stream: Hello is the first frame, a frame of no kind came first"
                            .to_owned(),
                    )
            } else if layer == Layer::AsTold
                && stated
                && matches!(
                    *method,
                    "ApplyPolicySpec" | "GetPolicySpec" | "RollbackPolicySpec"
                )
            {
                (
                    Code::Unimplemented,
                    format!("{method} is not served by a controller started with no spec source"),
                )
            } else if layer == Layer::AsTold && stated && *method == "Decide" {
                // The stream opens, and the call itself refuses the
                // case's one frame: it carries no id.
                (
                    Code::InvalidArgument,
                    "request_id is out of its range in this protocol version".to_owned(),
                )
            } else if layer == Layer::AsTold && stated {
                match call.and_then(refused_off_the_leader) {
                    Some((code, why)) => (code, why.to_owned()),
                    None => (
                        Code::Unimplemented,
                        format!("{method} is not served by this build"),
                    ),
                }
            } else {
                (Code::Unauthenticated, NO_PRINCIPAL.to_owned())
            };
            let status = answered.expect_err("no listener here serves the call");
            assert_eq!(
                (status.code(), status.message().to_owned()),
                expected,
                "{round}"
            );
        }
    }
}
