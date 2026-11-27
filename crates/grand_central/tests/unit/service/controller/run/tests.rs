use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{mpsc, Arc, Mutex},
    thread,
    time::Duration,
};

use clap::{error::ErrorKind, Parser};
use tokio::{runtime::Handle, sync::oneshot};

use super::{bind, color_for, set, truthy, Cli, OverloadThresholds, SpecSource};
use crate::{
    document::spec::SpecError,
    engine::{Candidate, Constraints, Filter, RequestFacts},
    service::{
        controller::{
            membership::tests::Logged, spec_store::SpecStore, Config, ConfigError, Controller,
            ElectionConfig, Error, LeasePlace, Scope, SpecRejected, StoreRefused,
        },
        elector::ElectionSettings,
    },
    stages::registry::StageRegistry,
};

/// A spec source for a command line that names none: a start names one,
/// and the parser reads this one's flag and opens nothing.
const SOURCE: [&str; 2] = ["--spec-store", "/var/lib/smg/spec"];

fn parse(args: &[&str]) -> Cli {
    let mut argv = vec!["smg-grand-central", "--controller-id", "controller-0"];
    if !args
        .iter()
        .any(|arg| ["--spec-file", "--spec-store"].contains(arg))
    {
        argv.extend(SOURCE);
    }
    argv.extend_from_slice(args);
    Cli::try_parse_from(argv).expect("the flags parse")
}

/// Whether the parser refuses `flags` beside an id and a spec source:
/// the refusal is then the flags' own.
fn refuses(flags: &[&str]) -> bool {
    let mut argv = vec!["smg-grand-central", "--controller-id", "c"];
    argv.extend(SOURCE);
    argv.extend_from_slice(flags);
    Cli::try_parse_from(argv).is_err()
}

/// The configuration of `args` under an empty environment: a case
/// reads nothing of the process's own.
fn config(args: &[&str]) -> Result<Config, ConfigError> {
    parse(args).config_with(|_| None)
}

#[cfg(feature = "discovery")]
#[test]
fn the_discovery_flags_reach_the_configuration() {
    use smg_service_discovery::RuntimeDiscoveryConfig;

    let args = [
        "--discovery-provider",
        "file",
        "--discovery-file",
        "/run/workers.json",
        "--discovery-check-interval-secs",
        "13",
        "--api-key",
        "worker-key",
    ];
    let configured = config(&args).expect("discovery configuration");
    let settings = configured.discovery.as_ref().expect("selected provider");
    let RuntimeDiscoveryConfig::File(file) = settings.runtime() else {
        panic!("expected the file provider");
    };
    assert_eq!(file.path, Path::new("/run/workers.json"));
    assert_eq!(file.check_interval, Duration::from_secs(13));
    assert!(settings.credential().is_some());
    assert_eq!(configured, config(&args).expect("the same settings"));
    assert_ne!(configured, config(&args[..6]).expect("without a key"));
    let mut other_file = args;
    other_file[3] = "/run/other.json";
    assert_ne!(configured, config(&other_file).expect("another manifest"));
    let disaggregated = [&args[..], &["--prefill-selector", "role=prefill"]].concat();
    assert_ne!(
        configured,
        config(&disaggregated).expect("another set of roles")
    );
    assert!(config(&[]).expect("the defaults").discovery.is_none());
    assert!(matches!(
        config(&["--discovery-provider", "file"]),
        Err(ConfigError::DiscoveryFlag {
            flag: "--discovery-file",
            ..
        })
    ));
}

#[cfg(feature = "discovery")]
#[test]
fn the_worker_key_is_read_only_for_a_selected_provider() {
    use std::ffi::OsString;

    let file = [
        "--discovery-provider",
        "file",
        "--discovery-file",
        "/run/workers.json",
    ];
    let key = |name: &str| (name == "SMG_API_KEY").then(|| OsString::from("worker-key"));
    let keyed = parse(&file).config_with(key).expect("a key to send");
    assert!(keyed
        .discovery
        .as_ref()
        .is_some_and(|settings| settings.credential().is_some()));
    assert_ne!(keyed, config(&file).expect("no key to send"));

    // A key file that is not there fails every read of the key.
    let missing = |name: &str| (name == "SMG_API_KEY_FILE").then(|| OsString::from("/missing/key"));
    assert!(matches!(
        parse(&file).config_with(missing),
        Err(ConfigError::DiscoveryFlag {
            flag: "--api-key-file",
            ..
        })
    ));
    let unread = parse(&[]).config_with(missing).expect("no key is read");
    assert!(unread.discovery.is_none());

    for flag in [
        ["--api-key", "worker-key"],
        ["--api-key-file", "/missing/key"],
    ] {
        assert!(
            matches!(
                config(&flag),
                Err(ConfigError::DiscoveryFlag { flag: refused, .. }) if refused == flag[0]
            ),
            "{flag:?}"
        );
    }
}

#[cfg(feature = "discovery")]
#[test]
fn a_slurm_job_name_under_two_roles_is_refused_at_the_start() {
    let shared = [
        "--discovery-provider",
        "slurm",
        "--slurm-prefill-job-name",
        "shared",
        "--slurm-decode-job-name",
        "shared",
        "--slurm-port",
        "8000",
    ];
    let Err(ConfigError::DiscoveryFlag { flag, reason }) = config(&shared) else {
        panic!("one job name under two roles was taken");
    };
    assert_eq!(flag, "--discovery-provider");
    assert!(
        reason.contains("'shared' maps to both prefill and decode workers"),
        "{reason}"
    );
    let apart = [&shared[..5], &["other", "--slurm-port", "8000"]].concat();
    assert!(config(&apart).expect("two names").discovery.is_some());
}

#[cfg(not(feature = "discovery"))]
#[test]
fn a_build_without_discovery_refuses_its_flags() {
    for flags in [
        &["--service-discovery"][..],
        &["--discovery-provider", "file"],
        &["--discovery-file", "workers.json"],
        &["--api-key", "worker-key"],
        &["--api-key-file", "worker-key.txt"],
    ] {
        assert!(refuses(flags), "{flags:?}");
    }
}

const IDENTITY: [&str; 4] = ["--tls-cert", "server.pem", "--tls-key", "server.key"];

fn with_identity<'a>(args: &[&'a str]) -> Vec<&'a str> {
    let mut all = IDENTITY.to_vec();
    all.extend_from_slice(args);
    all
}

fn authorities(config: &Config) -> Vec<Option<PathBuf>> {
    config
        .listeners
        .iter()
        .map(|l| l.tls.as_ref().and_then(|t| t.client_ca.clone()))
        .collect()
}

#[test]
fn the_defaults_mount_each_scope_on_a_listener_of_its_own() {
    let config = config(&[]).expect("the defaults");
    assert_eq!(config.controller_id, "controller-0");
    let mounts: Vec<(&str, &[Scope])> = config
        .listeners
        .iter()
        .map(|l| (l.name.as_str(), l.scopes.as_slice()))
        .collect();
    assert_eq!(
        mounts,
        vec![
            ("gateway", &[Scope::Gateway][..]),
            ("control", &[Scope::Control][..]),
            ("replication", &[Scope::Replication][..]),
        ]
    );
    assert_eq!(config.listeners[0].addr.port(), 30100);
    assert_eq!(config.listeners[1].addr.port(), 30101);
    assert_eq!(config.listeners[2].addr.port(), 30102);
    assert_eq!(config.admin.port(), 30103);
    assert!(config.listeners.iter().all(|l| l.tls.is_none()));
    assert_eq!(config.shutdown_settle, Duration::from_secs(5));
    assert_eq!(config.shutdown_grace, Duration::from_secs(30));
    assert_eq!(config.validate(), Ok(()));
}

/// The relays' listener left with their two calls, and the ingestion
/// scope's authority with the scope. A command line that still names
/// either flag is refused by the parser as a usage error, with exit
/// code 2, before a listener opens.
#[test]
fn the_parser_refuses_the_flags_of_the_ingestion_scope() {
    for flag in [
        ["--ingestion-listen", "127.0.0.1:30104"],
        ["--ingestion-client-ca", "relays.pem"],
    ] {
        let mut argv = vec!["smg-grand-central", "--controller-id", "c"];
        argv.extend(SOURCE);
        argv.extend(flag);
        let refused = Cli::try_parse_from(argv).expect_err("the flag is gone");
        assert_eq!(refused.kind(), ErrorKind::UnknownArgument, "{flag:?}");
        assert_eq!(refused.exit_code(), 2, "{flag:?}");
    }
}

#[test]
fn one_identity_serves_every_listener_and_the_authorities_are_per_scope() {
    let config = config(&with_identity(&[
        "--gateway-client-ca",
        "gateways.pem",
        "--replication-client-ca",
        "replicas.pem",
    ]))
    .expect("tls on every listener");
    assert_eq!(
        authorities(&config),
        [
            Some(PathBuf::from("gateways.pem")),
            None,
            Some(PathBuf::from("replicas.pem")),
        ],
        "the operators stay server-TLS"
    );
    assert!(config.listeners.iter().all(|l| l
        .tls
        .as_ref()
        .is_some_and(|t| t.cert == Path::new("server.pem"))));
    assert_eq!(config.validate(), Ok(()));
}

#[test]
fn the_tls_refusals_are_named() {
    assert_eq!(
        config(&["--tls-cert", "server.pem"]),
        Err(ConfigError::TlsHalfConfigured {
            given: "cert",
            missing: "key"
        })
    );
    assert_eq!(
        config(&["--tls-key", "server.key"]),
        Err(ConfigError::TlsHalfConfigured {
            given: "key",
            missing: "cert"
        })
    );
    assert_eq!(
        config(&["--control-client-ca", "operators.pem"]),
        Err(ConfigError::ClientCaWithoutTls(Scope::Control))
    );
    // The gateways' authority alone is in order: their listener
    // mounts no second scope to share its rule.
    let alone = config(&with_identity(&["--gateway-client-ca", "gateways.pem"]))
        .expect("one scope, one authority");
    assert_eq!(
        authorities(&alone),
        [Some(PathBuf::from("gateways.pem")), None, None]
    );
}

#[test]
fn the_shutdown_flags_reach_the_configuration() {
    let bounds = config(&[
        "--shutdown-settle-secs",
        "2",
        "--shutdown-grace-period-secs",
        "10",
    ])
    .expect("bounds");
    assert_eq!(bounds.shutdown_settle, Duration::from_secs(2));
    assert_eq!(bounds.shutdown_grace, Duration::from_secs(10));
    let zero = config(&["--shutdown-grace-period-secs", "0"]).expect("parsed");
    assert_eq!(zero.validate(), Err(ConfigError::ZeroShutdownGrace));
}

#[test]
fn the_load_poll_interval_is_the_gateways_default_and_refuses_zero() {
    let default = config(&[]).expect("parsed");
    assert_eq!(default.load_poll_interval, Duration::from_secs(10));
    let two = config(&["--load-monitor-interval", "2"]).expect("parsed");
    assert_eq!(two.load_poll_interval, Duration::from_secs(2));
    let zero = config(&["--load-monitor-interval", "0"]).expect("parsed");
    assert_eq!(zero.validate(), Err(ConfigError::ZeroLoadPollInterval));
}

/// The two overload flags are the gateway's by name and default, and by
/// bounds but for one: a count over the most a load report states is
/// refused here, which the gateway's flag takes. A start that types
/// neither states 8 queued requests and a usage of 0.8, and one that
/// types them states what it typed. A value out of the configuration's
/// bounds is refused where the command line is read, by the flag's name
/// beside the value, typed after an equals sign or as the next word: a
/// count or a usage of zero, a usage over one, a negative number and
/// one that is no number. The ends are taken.
#[test]
fn the_overload_flags_are_the_gateways_and_refuse_a_value_out_of_bounds() {
    let stating = |waiting, usage| OverloadThresholds {
        waiting_requests: Some(waiting),
        token_usage: Some(usage),
    };
    let default = config(&[]).expect("parsed");
    assert_eq!(default.worker_overload, Config::DEFAULT_WORKER_OVERLOAD);
    assert_eq!(default.worker_overload, stating(8, 0.8));
    let typed = config(&[
        "--worker-overload-waiting-requests",
        "4",
        "--worker-overload-token-usage",
        "0.625",
    ])
    .expect("parsed");
    assert_eq!(typed.worker_overload, stating(4, 0.625));
    assert_eq!(typed.validate(), Ok(()));
    let ends = config(&[
        "--worker-overload-waiting-requests",
        "1",
        "--worker-overload-token-usage",
        "1",
    ])
    .expect("parsed");
    assert_eq!(ends.worker_overload, stating(1, 1.0));
    assert_eq!(ends.validate(), Ok(()));

    // What the parser says of `flag` typed with `value`, after an equals
    // sign and as the next word: the first of the two.
    let refusal = |flag: &str, value: &str| {
        let joined = format!("{flag}={value}");
        let said = [vec![joined.as_str()], vec![flag, value]].map(|typed| {
            let mut argv = vec!["smg-grand-central", "--controller-id", "c"];
            argv.extend(SOURCE);
            argv.extend(typed);
            let refused = Cli::try_parse_from(argv).expect_err("a value out of bounds");
            refused.to_string()
        });
        let named = format!("invalid value '{value}' for '{flag} <");
        assert!(said.iter().all(|one| one.contains(&named)), "{said:?}");
        let [after_equals, _] = said;
        after_equals
    };
    let count = "--worker-overload-waiting-requests";
    let bounds = ConfigError::OverloadWaitingRequests(0).to_string();
    let bounds = bounds.split_once(": ").map_or("", |(_, bounds)| bounds);
    assert!(bounds.starts_with("the count is from 1 to "), "{bounds}");
    assert!(refusal(count, "0").contains(bounds));
    // The most queued requests a load report states, where a count holds
    // it: taken, and the first count over it is out of bounds.
    if let Ok(most) = usize::try_from(i64::MAX) {
        let top = config(&[count, &most.to_string()]).expect("parsed");
        assert_eq!(top.worker_overload.waiting_requests, Some(most));
        assert_eq!(top.validate(), Ok(()));
        assert!(refusal(count, "9223372036854775808").contains(bounds));
    }
    for typed in ["-1", "4e0"] {
        refusal(count, typed);
    }
    let usage = "--worker-overload-token-usage";
    let bounds = ConfigError::OverloadTokenUsageOutOfRange.to_string();
    for typed in ["0", "1.5", "-0.5", "inf", "NaN"] {
        assert!(refusal(usage, typed).contains(&bounds), "{typed}");
    }
    refusal(usage, "half");
}

#[test]
fn with_no_lease_flag_one_replica_leads_itself_on_the_designs_settings() {
    let alone = config(&[]).expect("parsed");
    assert_eq!(alone.election, ElectionConfig::default());
    assert_eq!(alone.election.lease, LeasePlace::Process);
    assert_eq!(alone.election.settings, ElectionSettings::default());
    assert_eq!(
        alone.validate(),
        Ok(()),
        "a wildcard listener is no refusal"
    );
}

/// A host that starts a controller in its own process makes its
/// configuration by [`Config::of`]. Each of the seven fields that call does
/// not take reads there what this command line states where none of their
/// flags is typed: a default that moves in one place moves in both, or this
/// case stops.
#[test]
fn the_start_of_a_configuration_states_what_the_command_line_states_by_default() {
    let typed = config(&[]).expect("the defaults");
    let started = Config::of(
        typed.controller_id.clone(),
        typed.listeners.clone(),
        typed.admin,
    );
    assert_eq!(started, typed);
}

#[test]
fn a_lease_file_elects_under_an_address_a_client_can_dial() {
    // The default gateway listener binds every address: a shared lease
    // behind it is refused until an address is advertised.
    let file = config(&["--lease-file", "/run/smg/lease"]).expect("parsed");
    assert_eq!(
        file.election.lease,
        LeasePlace::File(PathBuf::from("/run/smg/lease"))
    );
    assert_eq!(file.validate(), Err(ConfigError::AdvertiseAddressRequired));

    let advertised = config(&[
        "--lease-file",
        "/run/smg/lease",
        "--advertise-address",
        "10.0.0.7",
    ])
    .expect("parsed");
    assert_eq!(advertised.election.advertise.as_deref(), Some("10.0.0.7"));
    assert_eq!(advertised.validate(), Ok(()));

    // A gateway listener that binds one address is the address.
    let bound = config(&[
        "--lease-file",
        "/run/smg/lease",
        "--gateway-listen",
        "127.0.0.1:30100",
    ])
    .expect("parsed");
    assert_eq!(bound.validate(), Ok(()));

    let malformed = config(&["--advertise-address", "controller@host"]).expect("parsed");
    assert_eq!(
        malformed.validate(),
        Err(ConfigError::MalformedAdvertiseAddress(
            "controller@host".to_owned()
        ))
    );
}

#[test]
fn the_lease_timings_reach_the_configuration_and_keep_their_order() {
    let slow = config(&[
        "--lease-duration-ms",
        "15000",
        "--lease-renew-period-ms",
        "2000",
        "--lease-renew-deadline-ms",
        "10000",
    ])
    .expect("parsed");
    assert_eq!(
        slow.election.settings,
        ElectionSettings {
            duration: Duration::from_secs(15),
            renew_period: Duration::from_secs(2),
            renew_deadline: Duration::from_secs(10),
        }
    );
    assert_eq!(slow.validate(), Ok(()));

    let inverted = config(&["--lease-renew-deadline-ms", "3000"]).expect("parsed");
    assert_eq!(
        inverted.validate(),
        Err(ConfigError::ElectionSettings(
            "the duration is not longer than the renew deadline"
        ))
    );
}

#[cfg(feature = "kubernetes")]
#[test]
fn a_lease_object_is_named_and_takes_the_pods_namespace_unless_one_is_given() {
    let args = [
        "--advertise-address",
        "10.0.0.7",
        "--lease-name",
        "smg-policy",
    ];
    let object = config(&args).expect("parsed");
    assert_eq!(
        object.election.lease,
        LeasePlace::Kubernetes {
            namespace: None,
            name: "smg-policy".to_owned(),
        }
    );
    assert_eq!(object.validate(), Ok(()));

    let mut named = args.to_vec();
    named.extend(["--lease-namespace", "policy"]);
    assert_eq!(
        config(&named).expect("parsed").election.lease,
        LeasePlace::Kubernetes {
            namespace: Some("policy".to_owned()),
            name: "smg-policy".to_owned(),
        }
    );

    // The object stores whole seconds.
    let mut fraction = args.to_vec();
    fraction.extend(["--lease-duration-ms", "2500"]);
    assert_eq!(
        config(&fraction).expect("parsed").validate(),
        Err(ConfigError::LeaseDurationNotWholeSeconds(
            Duration::from_millis(2500)
        ))
    );

    // One lease, one place; a namespace names an object's.
    for refused in [
        &[
            "--lease-name",
            "smg-policy",
            "--lease-file",
            "/run/smg/lease",
        ][..],
        &["--lease-namespace", "policy"],
        &[
            "--lease-namespace",
            "policy",
            "--lease-file",
            "/run/smg/lease",
        ],
    ] {
        assert!(refuses(refused), "{refused:?}");
    }
}

#[test]
fn no_listener_takes_identities_as_told_unless_it_is_named() {
    assert!(parse(&[]).identity_as_told.is_empty());
    assert_eq!(
        parse(&["--identity-as-told", "gateway,control"]).identity_as_told,
        ["gateway", "control"]
    );
    assert_eq!(
        parse(&[
            "--identity-as-told",
            "gateway",
            "--identity-as-told",
            "replication"
        ])
        .identity_as_told,
        ["gateway", "replication"]
    );
}

/// Every listener on loopback, at a port the host picks.
const EPHEMERAL: [&str; 8] = [
    "--gateway-listen",
    "127.0.0.1:0",
    "--control-listen",
    "127.0.0.1:0",
    "--replication-listen",
    "127.0.0.1:0",
    "--admin-listen",
    "127.0.0.1:0",
];

/// Keeps every worker: a stage of a host's own.
#[derive(Debug)]
struct Any;

impl Filter for Any {
    fn name(&self) -> &'static str {
        "any"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }
}

/// The stages the run is handed are the ones the bound controller holds.
#[tokio::test]
async fn the_bound_controller_holds_the_stages_the_run_is_handed() {
    let config = parse(&EPHEMERAL).config().expect("the flags are in order");
    let mut stages = StageRegistry::builtin();
    stages.filter("any", || Box::new(Any)).expect("a free name");
    let dir = tempfile::tempdir().expect("a temp dir");
    let source = SpecSource::Store(dir.path().to_owned());
    let controller = bind(config, stages, &[], source).await.expect("binds");
    assert!(controller
        .stage_registry()
        .filters()
        .any(|name| name == "any"));
    // The stock build holds no such stage.
    assert!(StageRegistry::builtin().filters().all(|name| name != "any"));
}

/// A spec file's document names the stages the run is handed: a stage
/// of a host's own starts the controller on it, and the stock build
/// refuses the same file by the stage's name.
#[tokio::test]
async fn a_spec_file_names_the_stages_the_run_is_handed() {
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("policy.yaml");
    let document =
        "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [any]\n    pick: random\n";
    std::fs::write(&path, document).expect("the file");
    let start = |stages: StageRegistry| {
        let config = parse(&EPHEMERAL).config().expect("the flags are in order");
        bind(config, stages, &[], SpecSource::File(path.clone()))
    };

    let mut stages = StageRegistry::builtin();
    stages.filter("any", || Box::new(Any)).expect("a free name");
    let controller = start(stages).await.expect("a host's stage starts");
    assert_eq!(controller.store().snapshot().spec().generation(), 1);
    drop(controller);

    match start(StageRegistry::builtin()).await {
        Err(Error::SpecFile {
            source: SpecRejected::Invalid(SpecError::UnknownFilter { profile, name }),
            ..
        }) => assert_eq!((profile.as_str(), name.as_str()), ("default", "any")),
        Err(other) => panic!("the stock build refused it as {other}"),
        Ok(_) => panic!("the stock build holds no such stage"),
    }
}

/// Scores every worker alike, and says whether the lifecycle's calls
/// keep its state: a stage of a host's own.
#[derive(Debug)]
struct Level(bool);

impl crate::stages::Lifecycle for Level {
    fn needs_lifecycle(&self) -> bool {
        self.0
    }
}

impl crate::engine::Scorer for Level {
    fn name(&self) -> &'static str {
        "level"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        out: &mut [f32],
    ) -> crate::engine::TieBand {
        out.fill(1.0);
        crate::engine::TieBand::NORMALIZED
    }
}

/// The controller makes the membership call and a lease's calls but
/// the end of a lease whose gateway was let go: a spec file that names
/// a stage in need of them refuses the start by the stage's field, and
/// the same stage with no such need starts.
#[tokio::test]
async fn a_spec_file_that_names_a_stage_in_need_of_the_lifecycle_refuses_the_start() {
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("policy.yaml");
    let document =
        "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    score: [level]\n    pick: best\n";
    std::fs::write(&path, document).expect("the file");
    let start = |needs: bool| {
        let config = parse(&EPHEMERAL).config().expect("the flags are in order");
        let mut stages = StageRegistry::builtin();
        stages
            .scorer("level", move || Box::new(Level(needs)))
            .expect("a free name");
        bind(config, stages, &[], SpecSource::File(path.clone()))
    };

    let controller = start(false).await.expect("no need of the calls starts");
    assert_eq!(controller.store().snapshot().spec().generation(), 1);
    drop(controller);

    match start(true).await {
        Err(Error::SpecFile {
            source: SpecRejected::Invalid(SpecError::NotInBuild { path: field, what }),
            ..
        }) => assert_eq!(
            (field.as_str(), what),
            (
                "profiles.default.score[0].level",
                "the lifecycle this stage needs"
            )
        ),
        Err(other) => panic!("the start was refused as {other}"),
        Ok(_) => panic!("no served line makes the calls the stage needs"),
    }
}

/// The controller's file of leases keeps each side's own session
/// verdict of a pair. So its own start takes a spec file that binds a
/// sessions rule under the pair plan, as it takes the same file with
/// no rule.
#[tokio::test]
async fn a_spec_file_that_binds_a_sessions_rule_under_a_pair_starts() {
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("policy.yaml");
    let pair = "apiVersion: smg.policy/v1\nview:\n  headers: [x-session-id]\nprofiles:\n  q:\n    pick: random\nbindings:\n  - plan: pd-pair\n    profiles:\n      prefill: q\n      decode: q\n";
    let rule = "    sessions:\n      sticky_header: x-session-id\n      inflight_cap: 65535\n";
    let start = |document: String| {
        std::fs::write(&path, document).expect("the file");
        let config = parse(&EPHEMERAL).config().expect("the flags are in order");
        let stages = StageRegistry::builtin();
        bind(config, stages, &[], SpecSource::File(path.clone()))
    };
    for document in [pair.to_string(), format!("{pair}{rule}")] {
        let started = start(document).await;
        let controller = started.expect("a pair starts, under a rule and under none");
        assert_eq!(controller.store().snapshot().spec().generation(), 1);
    }
}

/// A spec file starts under each source of prefix affinity, and one that
/// names none reads the prefixes sent, as `merged` does. A served
/// decision that reads a column writes their record at its lease's
/// issue: a pair's side, and a single decision handed its column whole.
#[tokio::test]
async fn a_spec_file_starts_under_each_source_of_prefix_affinity() {
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("policy.yaml");
    let start = |affinity: &str| {
        let document = format!(
                "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    score:\n      - prefix-affinity: {affinity}\n    pick: best\n"
            );
        std::fs::write(&path, document).expect("the file");
        let config = parse(&EPHEMERAL).config().expect("the flags are in order");
        bind(
            config,
            StageRegistry::builtin(),
            &[],
            SpecSource::File(path.clone()),
        )
    };
    for (affinity, sent) in [
        ("{source: index}", false),
        ("{source: predictor}", true),
        ("{source: merged}", true),
        ("{}", true),
    ] {
        let controller = start(affinity).await.expect(affinity);
        let needs = controller.store().snapshot().spec().needs();
        assert_eq!(needs.sent_prefixes, sent, "{affinity}");
    }
}

/// The flag's names reach the bound controller: a name the flags mount
/// no listener under refuses the start, by that name.
#[tokio::test]
async fn a_listener_named_as_told_is_one_the_flags_mount() {
    // The start binds: its source is a store that is there.
    let dir = tempfile::tempdir().expect("a directory");
    let store = dir.path().to_str().expect("a path in UTF-8");
    let start = |flags: &[&str]| {
        let mut args = EPHEMERAL.to_vec();
        args.extend_from_slice(&["--spec-store", store]);
        args.extend_from_slice(flags);
        let cli = parse(&args);
        let config = cli.config().expect("the flags are in order");
        async move {
            bind(
                config,
                StageRegistry::builtin(),
                &cli.identity_as_told,
                cli.spec_source().expect("the flags name a store"),
            )
            .await
            .map(drop)
        }
    };
    // A name the flags mount no listener under: the ingestion scope's
    // left with the scope.
    let refused = start(&["--identity-as-told", "ingestion"]).await;
    assert!(
        matches!(
            &refused,
            Err(Error::Config(ConfigError::UnknownListener(name))) if name == "ingestion"
        ),
        "{refused:?}"
    );
    start(&["--identity-as-told", "gateway,replication"])
        .await
        .expect("both are listeners the flags mount");
}

#[test]
fn the_command_line_refuses_what_it_cannot_read() {
    assert!(refuses(&["--gateway-listen", "nowhere"]));
    assert!(refuses(&["--log-level", "loud"]));
}

/// A blocking call that has not returned does not hold the run's end:
/// the serve's answer is back inside five seconds, where a dropped
/// runtime waits on the call without end.
#[test]
fn the_runs_end_waits_on_no_blocked_call() {
    let runtime = super::runtime(Some(1)).expect("a runtime");
    let (release, blocked) = mpsc::channel::<()>();
    let (ended, end) = mpsc::channel();
    thread::spawn(move || {
        let answer = super::served(runtime, async move {
            let (started, start) = oneshot::channel::<()>();
            tokio::task::spawn_blocking(move || {
                let _ = started.send(());
                // Returns when `release` goes, at the case's end.
                let _ = blocked.recv();
            });
            start.await.expect("the blocking call starts");
            7
        });
        let _ = ended.send(answer);
    });
    let waited = end.recv_timeout(Duration::from_secs(5));
    drop(release);
    assert_eq!(waited, Ok(7), "the end returns with a call still blocked");
}

/// The run's own call ends it so: a start refused after a stage of the
/// host's left a blocking call gives its code inside five seconds.
#[test]
fn a_refused_start_ends_the_run_with_a_call_still_blocked() {
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("policy.yaml");
    // The first filter is built, then the second is refused by name.
    let document =
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [any, nowhere]\n    pick: random\n";
    std::fs::write(&path, document).expect("the file");
    let (release, blocked) = mpsc::channel::<()>();
    let blocked = Arc::new(Mutex::new(blocked));
    let (left, leaves) = mpsc::channel::<()>();
    let mut stages = StageRegistry::builtin();
    stages
        .filter("any", move || {
            // On the run's runtime alone: a registration builds the
            // stage once too, for its name.
            if let Ok(runtime) = Handle::try_current() {
                let (begun, begin) = mpsc::channel::<()>();
                let blocked = Arc::clone(&blocked);
                runtime.spawn_blocking(move || {
                    let _ = begun.send(());
                    // Returns when `release` goes, at the case's end.
                    if let Ok(blocked) = blocked.lock() {
                        let _ = blocked.recv();
                    }
                });
                // The call is on its thread before the stage is handed
                // out: an end leaves a call that has not begun unrun.
                if begin.recv().is_ok() {
                    let _ = left.send(());
                }
            }
            Box::new(Any)
        })
        .expect("a free name");
    let mut args: Vec<&str> = EPHEMERAL.to_vec();
    args.extend(["--spec-file", path.to_str().expect("a path in UTF-8")]);
    let cli = parse(&args);

    let (ended, end) = mpsc::channel();
    thread::spawn(move || {
        let _ = ended.send(super::run_with(cli, stages));
    });
    let waited = end.recv_timeout(Duration::from_secs(5));
    drop(release);
    leaves.try_recv().expect("the stage left a blocking call");
    assert_eq!(
        waited,
        Ok(ExitCode::FAILURE),
        "the run returns with a call still blocked"
    );
}

#[test]
fn the_deciding_threads_take_the_operators_count_inside_their_range() {
    let default = config(&[]).expect("parsed");
    assert_eq!(default.decide_shards, Config::default_decide_shards());
    assert_eq!(default.validate(), Ok(()));
    let three = config(&["--decide-shards", "3"]).expect("parsed");
    assert_eq!(three.decide_shards, 3);
    for bad in ["0", "257", "many"] {
        assert!(refuses(&["--decide-shards", bad]), "{bad}");
    }
}

#[test]
fn the_runtime_thread_count_is_bounded() {
    assert_eq!(
        parse(&["--runtime-worker-threads", "4"]).runtime_worker_threads,
        Some(4)
    );
    assert_eq!(parse(&[]).runtime_worker_threads, None);
    for bad in ["0", "4097", "many"] {
        assert!(refuses(&["--runtime-worker-threads", bad]), "{bad}");
    }
}

#[test]
fn a_terminal_is_colored_unless_refused_and_a_pipe_only_when_forced() {
    // (terminal, refused, forced)
    assert!(color_for(true, false, false));
    assert!(!color_for(true, true, false));
    assert!(color_for(true, true, true));
    assert!(!color_for(false, false, false));
    assert!(!color_for(false, true, false));
    assert!(color_for(false, false, true));
    assert!(color_for(false, true, true));
}

/// A force variable set to `0` does not force (clicolors); `NO_COLOR` set
/// to anything non-empty, `0` included, refuses (no-color.org); an empty
/// value is unset for both. The readers are pure: the process environment
/// is not touched.
#[test]
fn a_force_variable_set_to_zero_does_not_force_and_an_empty_one_is_unset() {
    let zero = Some(OsStr::new("0"));
    let empty = Some(OsStr::new(""));
    let one = Some(OsStr::new("1"));
    assert!(set(zero));
    assert!(!truthy(zero));
    assert!(!set(empty));
    assert!(!truthy(empty));
    assert!(set(one));
    assert!(truthy(one));
    assert!(!set(None));
    assert!(!truthy(None));
}

/// The flag's long help says how YAML reads: an operator meets the rule
/// before a start refuses a number where a field takes text.
#[test]
fn the_spec_files_help_says_how_yaml_reads() {
    let help = <Cli as clap::CommandFactory>::command()
        .render_long_help()
        .to_string();
    // The help wraps where the terminal does.
    let words = help.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(words.contains("quoted where a field takes text"), "{help}");
    assert!(words.contains("a key with no value is null"), "{help}");
}

#[test]
fn the_spec_file_is_a_flag_with_no_default() {
    assert_eq!(parse(&with_identity(&[])).spec_file, None);
    assert_eq!(
        parse(&with_identity(&[
            "--spec-file",
            "/etc/smg/policy/policy.yaml"
        ]))
        .spec_file,
        Some(PathBuf::from("/etc/smg/policy/policy.yaml"))
    );
}

/// A start names one source of its spec. The parser refuses a command
/// line that names none, and one that names both, by both flags' names.
#[test]
fn a_start_names_one_spec_source_and_the_parser_refuses_none_and_both() {
    use clap::error::ErrorKind;

    let dir = PathBuf::from("/var/lib/smg/spec");
    let stored = parse(&with_identity(&["--spec-store", "/var/lib/smg/spec"]));
    assert_eq!(stored.spec_source(), Some(SpecSource::Store(dir)));
    let file = PathBuf::from("/etc/smg/policy/policy.yaml");
    let filed = parse(&with_identity(&[
        "--spec-file",
        "/etc/smg/policy/policy.yaml",
    ]));
    assert_eq!(filed.spec_source(), Some(SpecSource::File(file)));

    let mut argv = vec!["smg-grand-central", "--controller-id", "controller-0"];
    argv.extend(with_identity(&[]));
    let neither = Cli::try_parse_from(argv.clone()).expect_err("a source is named");
    argv.extend(["--spec-file", "/a.yaml", "--spec-store", "/b"]);
    let both = Cli::try_parse_from(argv).expect_err("one source");
    for (refused, kind) in [
        (neither, ErrorKind::MissingRequiredArgument),
        (both, ErrorKind::ArgumentConflict),
    ] {
        assert_eq!(refused.kind(), kind);
        let told = refused.to_string();
        assert!(
            told.contains("--spec-file") && told.contains("--spec-store"),
            "{told}"
        );
    }
}

/// A command line that names neither source is the state the parser
/// rules out. Past the parser it has no source, and the start refuses
/// it before a runtime is built, in the refusal's own words: the case
/// reads the line the start wrote, on its own thread.
#[test]
fn a_command_line_past_the_parser_with_no_spec_source_is_refused() {
    let mut cli = parse(&[]);
    cli.spec_store = None;
    assert_eq!(cli.spec_source(), None);
    // The case's own subscriber keeps this thread's lines, a second one
    // registered beside it, as in the door's audit case.
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let code = tracing::subscriber::with_default(subscriber, || {
        super::run_with(cli, StageRegistry::builtin())
    });
    assert_eq!(code, ExitCode::FAILURE);
    let said = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    assert!(
        said.contains("start refused")
            && said.contains("no spec source is named: a start takes --spec-file or --spec-store"),
        "{said}"
    );
}

/// A caller that drops the spec store call while its start waits on a
/// held directory leaves the task to bind the store once the holder
/// lets go: its series are bound with it, and a second call is refused
/// as a second source.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_spec_store_call_that_binds_later_leaves_its_series() {
    let dir = tempfile::tempdir().expect("a directory");
    let holder = SpecStore::open(dir.path().to_owned()).expect("the holder");
    let config = parse(&EPHEMERAL).config().expect("the flags are in order");
    let mut controller = Controller::bind_with(config, StageRegistry::builtin())
        .await
        .expect("the controller binds");
    let call = controller.spec_store(dir.path().to_owned());
    let dropped = tokio::time::timeout(Duration::from_millis(200), call).await;
    assert!(dropped.is_err(), "the call waits on the held directory");
    drop(holder);
    let bound = async {
        while !controller.metrics.spec_bound() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), bound)
        .await
        .expect("the task binds the store with its series");
    assert!(controller.shared.generations.bound());
    match controller.spec_store(dir.path().to_owned()).await {
        Err(Error::SpecStore {
            source: StoreRefused::SecondSource,
            ..
        }) => {}
        other => panic!("a second source: {other:?}"),
    }
}

/// A spec store starts the controller on the newest generation on file,
/// under that file's number; a path that leads to no directory refuses
/// the start by name; and a controller that took a store takes no file
/// and no second store, refused at once on the directory it holds.
#[tokio::test]
async fn a_spec_store_starts_on_its_newest_generation() {
    let dir = tempfile::tempdir().expect("a directory");
    let document =
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";
    let store = dir.path().join("store");
    std::fs::create_dir(&store).expect("the directory");
    for name in ["00000000000000000003.yaml", "00000000000000000004.yaml"] {
        std::fs::write(store.join(name), document).expect("the file");
    }
    let start = |path: PathBuf| {
        let config = parse(&EPHEMERAL).config().expect("the flags are in order");
        bind(
            config,
            StageRegistry::builtin(),
            &[],
            SpecSource::Store(path),
        )
    };

    let mut controller = start(store.clone()).await.expect("the store starts");
    assert_eq!(controller.store().snapshot().spec().generation(), 4);
    let file = store.join("00000000000000000004.yaml");
    match controller.spec_file(file.clone()).await {
        Err(Error::SpecFile {
            source: SpecRejected::SecondSource,
            ..
        }) => {}
        other => panic!("a second source: {other:?}"),
    }
    match controller.spec_store(store.clone()).await {
        Err(Error::SpecStore {
            source: StoreRefused::SecondSource,
            ..
        }) => {}
        other => panic!("a second source: {other:?}"),
    }
    drop(controller);

    match start(file).await {
        Err(
            refused @ Error::SpecStore {
                source: StoreRefused::NotADirectory,
                ..
            },
        ) => assert!(
            refused
                .to_string()
                .ends_with(": the path leads to no directory"),
            "{refused}"
        ),
        Err(other) => panic!("a file was refused as {other}"),
        Ok(_) => panic!("a file is no directory"),
    }
}

/// A start ends on the refusal's words where its document's bindings
/// differ on the overload switch: a spec store whose newest generation
/// on file is such a document does not bind, and a spec file that holds
/// one does not start. Nothing serves until the document is edited.
#[tokio::test]
async fn a_document_whose_bindings_differ_on_the_overload_switch_ends_the_start() {
    const WORDS: &str = "bindings[1].overload.enabled = true is not the false bindings[0] says, by its block or the block's default: a controller builds its workers with one default set of overload thresholds, so the bindings of its document say one word";
    let dir = tempfile::tempdir().expect("a directory");
    let head = "apiVersion: smg.policy/v1\nprofiles:\n  p:\n    pick: random\nbindings:\n";
    let first =
        "  - match:\n      model: chat/*\n    profile: p\n    overload:\n      enabled: false\n";
    let differs = format!("{head}{first}  - profile: p\n");
    let store = dir.path().join("store");
    std::fs::create_dir(&store).expect("the directory");
    std::fs::write(store.join("00000000000000000004.yaml"), &differs).expect("the file");
    let file = dir.path().join("spec.yaml");
    std::fs::write(&file, &differs).expect("the file");
    let start = |source: SpecSource| {
        let config = parse(&EPHEMERAL).config().expect("the flags are in order");
        bind(config, StageRegistry::builtin(), &[], source)
    };

    match start(SpecSource::Store(store.clone())).await {
        Err(
            refused @ Error::SpecStore {
                source: StoreRefused::Rejected { generation: 4, .. },
                ..
            },
        ) => assert_eq!(
            refused.to_string(),
            format!("the spec store {}: generation 4: {WORDS}", store.display())
        ),
        Err(other) => panic!("the store was refused as {other}"),
        Ok(_) => panic!("the store's newest generation does not run"),
    }
    match start(SpecSource::File(file.clone())).await {
        Err(
            refused @ Error::SpecFile {
                source: SpecRejected::Invalid(_),
                ..
            },
        ) => assert_eq!(
            refused.to_string(),
            format!("the spec file {}: {WORDS}", file.display())
        ),
        Err(other) => panic!("the file was refused as {other}"),
        Ok(_) => panic!("the file's document does not run"),
    }
}

/// Beside a lease another replica can take, a spec store refuses the
/// start by name: a standby follows a file's generations alone.
#[tokio::test]
async fn a_spec_store_refuses_the_start_beside_a_lease_another_replica_can_take() {
    let dir = tempfile::tempdir().expect("a directory");
    let store = dir.path().join("store");
    std::fs::create_dir(&store).expect("the directory");
    let lease = dir.path().join("lease");
    let flags = ["--lease-file", lease.to_str().expect("a path in UTF-8")];
    let mut args = EPHEMERAL.to_vec();
    args.extend_from_slice(&flags);
    let config = parse(&args).config().expect("the flags are in order");
    assert!(matches!(config.election.lease, LeasePlace::File(_)));

    let source = SpecSource::Store(store.clone());
    match bind(config, StageRegistry::builtin(), &[], source).await {
            Err(refused @ Error::SpecStore {
                source: StoreRefused::BesideASharedLease,
                ..
            }) => assert_eq!(
                refused.to_string(),
                format!(
                    "the spec store {}: another replica can take this controller's lease, and a standby follows a file's generations alone in this build: keep the lease in the process, or hand the spec as a file",
                    store.display()
                )
            ),
            Err(other) => panic!("refused as {other}"),
            Ok(_) => panic!("a shared lease takes no spec store"),
        }
}

/// Beside a lease another replica can take, a spec file is passed at
/// the start and not applied: the replica runs no number of its own
/// until it leads, so it follows its leader's generation whatever the
/// number. The file is its source all the same, and writes while the
/// replica leads. One that does not run refuses the start. Under a
/// lease in the process the file is applied at the start, and writes
/// always.
#[tokio::test]
async fn a_spec_file_is_applied_at_the_start_under_a_lease_in_the_process_alone() {
    let dir = tempfile::tempdir().expect("a directory");
    let document =
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";
    let file = dir.path().join("policy.yaml");
    std::fs::write(&file, document).expect("the file");
    let lease = dir.path().join("lease");
    let shared = ["--lease-file", lease.to_str().expect("a path in UTF-8")];
    let start = |flags: &[&str]| {
        let mut args = EPHEMERAL.to_vec();
        args.extend_from_slice(flags);
        let config = parse(&args).config().expect("the flags are in order");
        bind(
            config,
            StageRegistry::builtin(),
            &[],
            SpecSource::File(file.clone()),
        )
    };

    let alone = start(&[]).await.expect("the file starts");
    assert_eq!(alone.store().snapshot().spec().generation(), 1);
    assert!(alone.shared.generations.source_writes());
    drop(alone);

    let paired = start(&shared).await.expect("the file starts");
    assert_eq!(
        paired.store().snapshot().spec().generation(),
        0,
        "passed, not applied"
    );
    assert!(paired.shared.generations.bound(), "a source all the same");
    assert!(
        !paired.shared.generations.source_writes(),
        "its source writes while it leads, and it has not led"
    );
    drop(paired);

    std::fs::write(&file, document.replace("healthy", "no-such-stage")).expect("the file");
    match start(&shared).await {
        Err(Error::SpecFile { .. }) => {}
        Err(other) => panic!("refused as {other}"),
        Ok(_) => panic!("a document that does not run refuses the start"),
    }
}
