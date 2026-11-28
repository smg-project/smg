use super::*;

/// Each provider's kind stamps its own name on a worker and on a series.
#[test]
fn each_kind_has_its_own_labels() {
    let kinds = [
        DiscoveryKind::Kubernetes,
        DiscoveryKind::File,
        DiscoveryKind::Slurm,
        DiscoveryKind::Consul,
    ];
    let labels = kinds.map(|kind| (kind.as_label(), kind.metric_label()));
    assert_eq!(
        labels,
        [
            ("kubernetes", "kubernetes"),
            ("file", "file"),
            ("slurm", "slurm"),
            ("consul", "consul"),
        ]
    );
}

fn worker_at(url: &str) -> DiscoveredWorker {
    DiscoveredWorker {
        discovery_id: "id-1".to_string(),
        endpoint: Endpoint::parse_with_rank(url).expect(url).0,
        worker_type: WorkerType::Regular,
        bootstrap_port: None,
        model_id_override: None,
        kv_connector: None,
        kv_role: None,
        kv_engine_id: None,
        pairing_protocol: None,
        compat_labels: BTreeMap::new(),
    }
}

#[test]
fn fingerprint_is_deterministic_and_spelling_insensitive() {
    let a = worker_at("10.0.0.1:8080");
    assert_eq!(a.fingerprint(), a.clone().fingerprint());
    // Two spellings that parse to one endpoint register identically.
    assert_eq!(
        worker_at("[::1]:8080").fingerprint(),
        worker_at("[0:0:0:0:0:0:0:1]:8080").fingerprint()
    );
}

/// Every configuration field moves the fingerprint, so a change to any of
/// them is a change the source made to this worker.
#[test]
fn every_configuration_field_moves_the_fingerprint() {
    let base = worker_at("10.0.0.1:8080");
    let variants = [
        DiscoveredWorker {
            endpoint: Endpoint::parse("grpc://10.0.0.1:8080").unwrap(),
            ..base.clone()
        },
        DiscoveredWorker {
            worker_type: WorkerType::Prefill,
            ..base.clone()
        },
        DiscoveredWorker {
            bootstrap_port: Some(9000),
            ..base.clone()
        },
        DiscoveredWorker {
            model_id_override: Some("m".to_string()),
            ..base.clone()
        },
        DiscoveredWorker {
            kv_connector: Some("c".to_string()),
            ..base.clone()
        },
        DiscoveredWorker {
            kv_role: Some("r".to_string()),
            ..base.clone()
        },
        DiscoveredWorker {
            kv_engine_id: Some("e".to_string()),
            ..base.clone()
        },
        DiscoveredWorker {
            pairing_protocol: Some("p".to_string()),
            ..base.clone()
        },
        // A change of the value, not only its arrival: a Pod relabelled
        // in the middle of a rollout updates its worker in place.
        DiscoveredWorker {
            pairing_protocol: Some("q".to_string()),
            ..base.clone()
        },
        // Stated and empty is not the same answer as not stated.
        DiscoveredWorker {
            pairing_protocol: Some(String::new()),
            ..base.clone()
        },
    ];
    let mut seen = vec![base.fingerprint()];
    for variant in &variants {
        let hash = variant.fingerprint();
        assert!(!seen.contains(&hash), "{variant:?} collided");
        seen.push(hash);
    }
}

/// A worker that states no pairing protocol keeps the fingerprint it had
/// before the field was read: these two are the fingerprints of the same
/// two records on the commit before it. Reading the field therefore
/// re-registers no worker that does not use it.
#[test]
fn a_worker_with_no_pairing_protocol_keeps_its_earlier_fingerprint() {
    assert_eq!(
        worker_at("10.0.0.1:8080").fingerprint(),
        "3a04468c1f7c7f257b3ff4794b6ec558a281805bd5d103b6d71bf6ff85444e8e"
    );
    let full = DiscoveredWorker {
        worker_type: WorkerType::Decode,
        bootstrap_port: Some(9000),
        model_id_override: Some("m".to_string()),
        kv_connector: Some("c".to_string()),
        kv_role: Some("r".to_string()),
        kv_engine_id: Some("e".to_string()),
        ..worker_at("grpc://10.0.0.1:8080")
    };
    assert_eq!(
        full.fingerprint(),
        "d3c6de55f4e2dcf4ac367bb58751b33a2f3b2bc74cf42c6b87a4f30c2870d88c"
    );
}

/// Identity and compatibility labels are set aside in the hash. Otherwise a
/// new instance at the same address and an edit to the configuration would
/// be indistinguishable, and a provider's bookkeeping labels could force
/// replacements.
#[test]
fn identity_and_compat_labels_do_not_move_the_fingerprint() {
    let base = worker_at("10.0.0.1:8080");
    let renamed = DiscoveredWorker {
        discovery_id: "id-2".to_string(),
        compat_labels: BTreeMap::from([("k".to_string(), "v".to_string())]),
        ..base.clone()
    };
    assert_eq!(base.fingerprint(), renamed.fingerprint());
}

/// The length prefix is what keeps a value from forging a field boundary.
/// The presence byte alone separates `"ab"|"c"` from `"a"|"bc"`, but not a
/// value that contains the presence byte itself: without lengths, both of
/// these encode as `\x01 a \x01 b \x01 c`. Provider input is operator-written
/// text, so a control byte in it is not hypothetical.
#[test]
fn a_value_cannot_forge_a_field_boundary() {
    let left = DiscoveredWorker {
        kv_connector: Some("a\u{1}b".to_string()),
        kv_role: Some("c".to_string()),
        ..worker_at("10.0.0.1:8080")
    };
    let right = DiscoveredWorker {
        kv_connector: Some("a".to_string()),
        kv_role: Some("b\u{1}c".to_string()),
        ..worker_at("10.0.0.1:8080")
    };
    assert_ne!(left.fingerprint(), right.fingerprint());

    // Absent and empty are different answers, which the presence byte
    // (not the length) guarantees.
    let absent = worker_at("10.0.0.1:8080");
    let empty = DiscoveredWorker {
        kv_role: Some(String::new()),
        ..absent.clone()
    };
    assert_ne!(absent.fingerprint(), empty.fingerprint());
}
