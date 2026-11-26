//! The exposure: what a row reveals to its plugin, and what it withholds.

use smg_grand_central::{
    document::Needs,
    engine::{Header, Identity},
    plugins::{
        grpc::{GrpcPluginSpec, Plugin, Tls, Transport},
        shared::{Exposure, HeaderSet, Reveal},
    },
    service::proto,
    stages::Lifecycle,
    tracking::Lease,
};

use crate::{
    decide_with, facts, lead, pool, profile, runtime, single_engine, spec,
    support::grpc_plugin::{eventually, TestPlugin, ALL},
};

#[test]
fn the_exposure_withholds_what_the_row_does_not_reveal() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "narrow-view");
    spec.view = Exposure {
        tenant: Reveal::None,
        session_key: Reveal::Plain,
        headers: HeaderSet::of([2]),
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let headers = [
        {
            let mut header = Header::default();
            header.name = 2;
            header.value = "exposed";
            header
        },
        {
            let mut header = Header::default();
            header.name = 9;
            header.value = "hidden";
            header
        },
    ];
    let mut request = facts(1);
    request.tenant = Identity::plain("acme");
    request.session_key = Identity::plain("s-1");
    request.headers = &headers;
    let candidates = pool(4);
    assert_eq!(
        lead(&decide_with(&engine, &request, &candidates)),
        Some(103)
    );
    let seen = served.seen();
    let sent = seen.frames[0].request.as_ref().expect("the request");
    assert_eq!(sent.tenant, None, "withheld");
    assert_eq!(
        sent.session_key.as_ref().and_then(|id| id.form.clone()),
        Some(proto::identity::Form::Plain("s-1".to_owned())),
        "in the clear, as the row asks and the host holds it"
    );
    assert_eq!(
        sent.headers,
        [proto::Header {
            name: 2,
            value: "exposed".to_owned(),
        }],
        "the allow-listed header alone"
    );
}

/// A frame carries the request's block hashes to a plugin whose row
/// declares them, in either form, and none to one whose row declares
/// neither: a prompt's length sizes no frame of a plugin that reads none.
#[test]
fn a_frame_carries_block_hashes_to_a_row_that_declares_them_alone() {
    let runtime = runtime();
    let hashes: Vec<u64> = (1..=300).collect();
    let mut request = facts(1);
    request.block_hashes = &hashes;
    let candidates = pool(4);
    let sent = |name: &str, needs: Needs| {
        let served = TestPlugin::serve();
        let mut spec = spec(&served, name);
        spec.needs = needs;
        let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
        let engine = single_engine(profile(&plugin, ALL));
        assert_eq!(
            lead(&decide_with(&engine, &request, &candidates)),
            Some(103),
            "{name}: the plugin answers"
        );
        let seen = served.seen();
        let sent = seen.frames[0].request.as_ref().expect("the request");
        sent.block_hashes.clone()
    };
    assert!(
        sent("reads-none", Needs::default()).is_empty(),
        "a row that declares no hashes is sent none"
    );
    let tokens = Needs::NONE.with_token_hashes();
    assert_eq!(sent("reads-tokens", tokens), hashes);
    let text = Needs::NONE.with_text_hashes();
    assert_eq!(sent("reads-text", text), hashes);
}

/// A lease on the lifecycle carries the session's value, and a request
/// shows the same value in the hash form, so a plugin matches the two: a
/// row that withholds the session key withholds both.
#[test]
fn a_lease_and_its_request_show_one_session_value_or_none() {
    let mut keyed = facts(1);
    keyed.session_key = Identity::plain("s-1");
    let mut lease = Lease::default();
    lease.session = keyed.session();
    let value = lease.session.get();
    assert_ne!(value, 0, "a keyed request has a session");
    let runtime = runtime();
    // The session as the request's frame and as the lease's event show it.
    let shown_under = |name: &str, session_key: Reveal| {
        let served = TestPlugin::serve();
        let mut spec = spec(&served, name);
        spec.view = Exposure {
            session_key,
            ..Exposure::default()
        };
        let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
        let engine = single_engine(profile(&plugin, ALL));
        let decision = decide_with(&engine, &keyed, &pool(4));
        engine.on_decision_issued(&lease, &decision);
        assert!(eventually(|| served.seen().events.len() == 1));
        let seen = served.seen();
        let request = seen.frames[0].request.as_ref().expect("the request");
        let in_the_frame = request.session_key.as_ref().and_then(|id| id.form.clone());
        let on_the_lease = match &seen.events[0].event {
            Some(proto::lifecycle_event::Event::Issued(issued)) => {
                issued.decision.as_ref().expect("the decision").leases[0].session_key
            }
            other => panic!("issued, not {other:?}"),
        };
        (in_the_frame, on_the_lease)
    };
    assert_eq!(
        shown_under("session-withheld", Reveal::None),
        (None, 0),
        "withheld from both"
    );
    assert_eq!(
        shown_under("session-hashed", Reveal::Hash),
        (Some(proto::identity::Form::Hash(value)), value),
        "one value in both, under a row that reveals the key's hash"
    );
}

/// A row printed in its debug form writes none of its PEMs: one of them
/// is the private key.
#[test]
fn a_printed_row_withholds_its_pems() {
    let pems: [&[u8]; 3] = [b"AUTHORITY", b"CERTIFICATE", b"PRIVATE"];
    let row = GrpcPluginSpec::new(
        "printed",
        Transport::Tls {
            address: "plugin.example:9000".to_owned(),
            tls: Tls {
                server_name: "plugin.example".to_owned(),
                ca_pem: pems[0].to_vec(),
                cert_pem: pems[1].to_vec(),
                key_pem: pems[2].to_vec(),
            },
        },
    );
    let printed = format!("{row:?}");
    assert!(printed.contains("plugin.example"), "{printed}");
    for pem in pems {
        // A byte vector prints as its numbers between brackets.
        let numbers = format!("{pem:?}");
        assert!(
            !printed.contains(&numbers[1..numbers.len() - 1]),
            "a PEM's bytes are printed: {printed}"
        );
    }
}
