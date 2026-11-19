use std::sync::Mutex;

use super::*;

const SQUEUE_23_11: &str = include_str!("../testdata/slurm/squeue-23.11.json");
const SQUEUE_24_11: &str = include_str!("../testdata/slurm/squeue-24.11.json");
const SQUEUE_21_08: &str = include_str!("../testdata/slurm/squeue-21.08.json");
const NODES_23_11: &str = include_str!("../testdata/slurm/scontrol-nodes-23.11.json");
const NODES_24_11: &str = include_str!("../testdata/slurm/scontrol-nodes-24.11.json");

fn discovery_config(job_names: &[&str]) -> SlurmDiscoveryConfig {
    SlurmDiscoveryConfig {
        job_names: job_names.iter().map(|name| name.to_string()).collect(),
        encode_job_names: vec![],
        prefill_job_names: vec![],
        decode_job_names: vec![],
        port: 8000,
        bootstrap_port: None,
        user: None,
        account: None,
        partition: None,
        check_interval_secs: 30,
    }
}

fn config(job_names: &[&str]) -> SlurmProviderConfig {
    SlurmProviderConfig::new(&discovery_config(job_names), Some(Roles::Regular)).unwrap()
}

fn job(jobs: &Jobs, id: u64) -> &Job {
    jobs.jobs.iter().find(|job| job.id == id).unwrap()
}

// ---- What the recordings say. ----

/// Both supported releases read into the same jobs: the requeued one with
/// its restart count, the two array tasks under their own ids, the
/// pending one, alice's, and the user `squeue` ran as.
#[test]
fn the_recorded_job_lists_read_alike() {
    for (recording, alice, pending, second_task) in
        [(SQUEUE_23_11, 8, 6, 7), (SQUEUE_24_11, 6, 7, 8)]
    {
        let jobs = parse_jobs(recording.as_bytes()).unwrap();
        assert_eq!(jobs.cluster.as_deref(), Some("smgtest"));
        assert_eq!(jobs.requested_by.as_deref(), Some("root"));
        assert_eq!(jobs.jobs.len(), 8);

        let requeued = job(&jobs, 4);
        assert_eq!(requeued.name, "smg-decode");
        assert_eq!(requeued.restart_count, 1);
        assert_eq!(requeued.states, ["RUNNING"]);
        assert_eq!(requeued.nodes, "slurmnode");
        assert_eq!(requeued.cluster, "smgtest");
        assert!(requeued.start_time.is_some_and(|start| start > 0));

        for id in [3, second_task] {
            assert_eq!(job(&jobs, id).name, "smg-prefill");
            assert_eq!(job(&jobs, id).restart_count, 0);
        }
        assert_eq!(job(&jobs, alice).user_name, "alice");
        assert_eq!(job(&jobs, pending).states, ["PENDING"]);
        assert_eq!(job(&jobs, 2).account, "acct-b");
        assert_eq!(job(&jobs, 2).partition, "gpu");
        assert_eq!(job(&jobs, 1).account, "");
    }
}

/// 21.08 prints a different shape and has no `scontrol --json`: it is
/// refused by its release, not by whatever field it lacks.
#[test]
fn an_older_release_is_refused_by_name() {
    let reason = parse_jobs(SQUEUE_21_08.as_bytes()).unwrap_err();
    assert!(reason.contains("Slurm 21.08.5"), "{reason}");
    assert!(reason.contains("23.11 and later"), "{reason}");

    assert!(check_release("23.11.4").is_ok());
    assert!(check_release("24.11.5").is_ok());
    assert!(check_release("25.05.0").is_ok());
    assert!(check_release("23.02.7").is_err());
    assert!(check_release("garbage").is_err());
}

#[test]
fn an_unset_start_time_reads_as_none() {
    let mut answer: serde_json::Value = serde_json::from_str(SQUEUE_24_11).unwrap();
    answer["jobs"][0]["start_time"] =
        serde_json::json!({"set": false, "infinite": false, "number": 0});
    let jobs = parse_jobs(answer.to_string().as_bytes()).unwrap();
    assert_eq!(jobs.jobs[0].start_time, None);
}

#[test]
fn errors_in_an_answer_reject_it() {
    let mut answer: serde_json::Value = serde_json::from_str(SQUEUE_24_11).unwrap();
    answer["errors"] = serde_json::json!([{"description": "controller unreachable"}]);
    let reason = parse_jobs(answer.to_string().as_bytes()).unwrap_err();
    assert!(reason.contains("controller unreachable"), "{reason}");

    assert!(parse_jobs(b"squeue: error: Invalid user").is_err());
}

/// `NodeAddr` is reported whether set or not: unset, Slurm reports
/// `NodeHostname`, and unset too, the node's name.
#[test]
fn the_recorded_nodes_report_an_address_for_each() {
    for recording in [NODES_23_11, NODES_24_11] {
        let nodes = parse_nodes(recording.as_bytes()).unwrap();
        let addresses: Vec<(&str, &str, &str)> = nodes
            .iter()
            .map(|n| (n.name.as_str(), n.address.as_str(), n.hostname.as_str()))
            .collect();
        assert_eq!(
            addresses,
            [
                ("slurmnode", "127.0.0.1", "slurmnode"),
                ("gpu01", "10.0.0.11", "gpu01.cluster"),
                ("gpu02", "gpu02.cluster", "gpu02.cluster"),
                ("gpu03", "gpu03", "gpu03"),
            ]
        );
    }
}

/// `scontrol show hostnames` exits 0 on a malformed hostlist: 23.11 prints
/// nothing, and 21.08 printed `gpu]`. Both are refused.
#[test]
fn hostname_expansion_refuses_what_is_not_a_node_list() {
    assert_eq!(
        parse_hostnames(b"gpu01\ngpu02\nnode-10\n").unwrap(),
        ["gpu01", "gpu02", "node-10"]
    );
    assert!(parse_hostnames(b"").is_err());
    assert!(parse_hostnames(b"gpu]\n").is_err());
    assert!(parse_hostnames(b"a b\n").is_err());
}

// ---- A poll, against a scripted Slurm. ----

/// Answers from fixed data, and records the node lookups it was asked.
struct FakeSlurm {
    jobs: Jobs,
    hostlists: HashMap<String, Vec<String>>,
    nodes: Vec<Node>,
    node_lookups: Mutex<Vec<Vec<String>>>,
    /// `scontrol show hostnames` failing as a command, not answering badly.
    expansion_fails: bool,
}

impl FakeSlurm {
    /// The 23.11 recording, which ignored every filter but `--user`, with
    /// each job given its own node `n<id>` at `10.0.1.<id>`.
    fn recorded() -> Self {
        let mut jobs = parse_jobs(SQUEUE_23_11.as_bytes()).unwrap();
        let mut hostlists = HashMap::new();
        let mut nodes = Vec::new();
        for job in &mut jobs.jobs {
            job.nodes = format!("n{}", job.id);
            hostlists.insert(job.nodes.clone(), vec![job.nodes.clone()]);
            nodes.push(Node {
                name: job.nodes.clone(),
                address: format!("10.0.1.{}", job.id),
                hostname: format!("{}.cluster", job.nodes),
            });
        }
        Self {
            jobs,
            hostlists,
            nodes,
            node_lookups: Mutex::new(Vec::new()),
            expansion_fails: false,
        }
    }

    fn job_mut(&mut self, id: u64) -> &mut Job {
        self.jobs.jobs.iter_mut().find(|job| job.id == id).unwrap()
    }
}

#[async_trait]
impl SlurmClient for FakeSlurm {
    async fn running_jobs(&self, _: &SlurmProviderConfig) -> Result<Jobs, SlurmError> {
        Ok(self.jobs.clone())
    }

    async fn expand_hostlist(&self, hostlist: &str) -> Result<Vec<String>, SlurmError> {
        if self.expansion_fails {
            return Err(SlurmError::Command {
                command: format!("scontrol show hostnames {hostlist}"),
                reason: "timed out after 30s".to_string(),
            });
        }
        self.hostlists
            .get(hostlist)
            .cloned()
            .ok_or_else(|| invalid(format!("unknown hostlist {hostlist}")))
    }

    async fn nodes(&self, names: &[String]) -> Result<Vec<Node>, SlurmError> {
        self.node_lookups.lock().unwrap().push(names.to_vec());
        Ok(self
            .nodes
            .iter()
            .filter(|node| names.contains(&node.name))
            .cloned()
            .collect())
    }
}

fn urls_of(outcome: &PollOutcome) -> Vec<String> {
    let mut urls: Vec<String> = outcome
        .workers
        .iter()
        .map(|worker| worker.endpoint.render())
        .collect();
    urls.sort();
    urls
}

async fn urls(slurm: &FakeSlurm, config: &SlurmProviderConfig) -> Vec<String> {
    let mut urls: Vec<String> = poll(slurm, config)
        .await
        .unwrap()
        .workers
        .iter()
        .map(|worker| worker.endpoint.render())
        .collect();
    urls.sort();
    urls
}

/// The recorded list holds a pending job, an unrelated name and another
/// user's job beside the two running `smg-worker` jobs of `root`; only
/// those two become workers, because the answer is filtered here.
#[tokio::test]
async fn the_answer_is_filtered_by_state_name_and_user() {
    let slurm = FakeSlurm::recorded();
    assert_eq!(
        urls(&slurm, &config(&["smg-worker"])).await,
        ["10.0.1.1:8000", "10.0.1.2:8000"]
    );

    let alice = SlurmProviderConfig {
        user: Some("alice".to_string()),
        ..config(&["smg-worker"])
    };
    assert_eq!(urls(&slurm, &alice).await, ["10.0.1.8:8000"]);
    // A numeric user is a uid; root's is 0.
    let uid = SlurmProviderConfig {
        user: Some("0".to_string()),
        ..config(&["smg-worker"])
    };
    assert_eq!(urls(&slurm, &uid).await, ["10.0.1.1:8000", "10.0.1.2:8000"]);
}

/// Slurm stores a job's account lowercased (checked against Slurm 24.11:
/// `-A ACCT-B` runs as `acct-b`), so a configured account matches in any
/// case.
#[tokio::test]
async fn the_account_matches_in_any_case() {
    let slurm = FakeSlurm::recorded();
    let upper = SlurmProviderConfig {
        account: Some("ACCT-B".to_string()),
        ..config(&["smg-worker"])
    };
    assert_eq!(urls(&slurm, &upper).await, ["10.0.1.2:8000"]);
}

/// 23.11 lists every job of the user, filters or not: one it would have
/// filtered out may have no start time, and must not reject the poll. A
/// job that is placed needs one, as its identity carries it.
#[tokio::test]
async fn only_a_placed_job_needs_a_start_time() {
    let mut slurm = FakeSlurm::recorded();
    // The pending job, and one with a name no mapping lists.
    slurm.job_mut(6).start_time = None;
    slurm.job_mut(5).start_time = None;
    assert_eq!(
        urls(&slurm, &config(&["smg-worker"])).await,
        ["10.0.1.1:8000", "10.0.1.2:8000"]
    );

    // A selected one is left out alone.
    slurm.job_mut(1).start_time = None;
    let outcome = poll(&slurm, &config(&["smg-worker"])).await.unwrap();
    assert_eq!(urls_of(&outcome), ["10.0.1.2:8000"]);
    assert_eq!(
        outcome.skipped,
        [Skip {
            reason: series::DISCOVERY_SKIP_INVALID,
            message: "job 1 (smg-worker): running with no start time".to_string(),
        }]
    );
}

#[tokio::test]
async fn the_answer_is_filtered_by_account_and_partition() {
    let slurm = FakeSlurm::recorded();
    for scoped in [
        SlurmProviderConfig {
            account: Some("acct-b".to_string()),
            ..config(&["smg-worker"])
        },
        SlurmProviderConfig {
            partition: Some("gpu".to_string()),
            ..config(&["smg-worker"])
        },
    ] {
        assert_eq!(urls(&slurm, &scoped).await, ["10.0.1.2:8000"]);
    }
}

/// Each mapped name gives its jobs' workers that type, and only encode and
/// prefill workers carry the bootstrap port.
#[tokio::test]
async fn job_names_map_to_worker_types() {
    let slurm = FakeSlurm::recorded();
    let config = SlurmProviderConfig::new(
        &SlurmDiscoveryConfig {
            job_names: vec![],
            encode_job_names: vec!["smg-worker".to_string()],
            prefill_job_names: vec!["smg-prefill".to_string()],
            decode_job_names: vec!["smg-decode".to_string()],
            bootstrap_port: Some(8998),
            ..discovery_config(&[])
        },
        Some(Roles::EncodePrefillDecode),
    )
    .unwrap();
    let mut workers: Vec<(String, WorkerType, Option<u16>)> = poll(&slurm, &config)
        .await
        .unwrap()
        .workers
        .into_iter()
        .map(|w| (w.endpoint.render(), w.worker_type, w.bootstrap_port))
        .collect();
    workers.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        workers,
        [
            ("10.0.1.1:8000".to_string(), WorkerType::Encode, Some(8998)),
            ("10.0.1.2:8000".to_string(), WorkerType::Encode, Some(8998)),
            ("10.0.1.3:8000".to_string(), WorkerType::Prefill, Some(8998)),
            ("10.0.1.4:8000".to_string(), WorkerType::Decode, None),
            ("10.0.1.7:8000".to_string(), WorkerType::Prefill, Some(8998)),
        ]
    );
}

/// A multi-node allocation places a worker on each node: at `NodeAddr`
/// (bracketed when IPv6), else `NodeHostname`. One lookup names every node
/// once.
#[tokio::test]
async fn every_node_of_an_allocation_gets_a_worker() {
    let mut slurm = FakeSlurm::recorded();
    slurm.job_mut(1).nodes = "gpu[01-04]".to_string();
    slurm.hostlists.insert(
        "gpu[01-04]".to_string(),
        ["gpu01", "gpu02", "gpu03", "gpu04"]
            .map(String::from)
            .to_vec(),
    );
    // The recorded gpu01 (NodeAddr set) and gpu02 (unset, so Slurm reports
    // NodeHostname), then an IPv6 NodeAddr and a record with only a
    // hostname.
    let recorded = parse_nodes(NODES_24_11.as_bytes()).unwrap();
    slurm.nodes.extend(
        recorded
            .into_iter()
            .filter(|node| node.name == "gpu01" || node.name == "gpu02"),
    );
    slurm.nodes.push(Node {
        name: "gpu03".to_string(),
        address: "fd00::13".to_string(),
        hostname: "gpu03".to_string(),
    });
    slurm.nodes.push(Node {
        name: "gpu04".to_string(),
        address: String::new(),
        hostname: "gpu04.cluster".to_string(),
    });

    let mut found: Vec<(String, String)> = poll(&slurm, &config(&["smg-worker"]))
        .await
        .unwrap()
        .workers
        .into_iter()
        .map(|worker| (worker.endpoint.render(), worker.discovery_id))
        .collect();
    found.sort();
    let start = |id| job(&slurm.jobs, id).start_time.unwrap();
    let on_job_1 = |node| format!("smgtest/1/{}/0/{node}/regular", start(1));
    let mut expected = vec![
        ("10.0.0.11:8000".to_string(), on_job_1("gpu01")),
        ("gpu02.cluster:8000".to_string(), on_job_1("gpu02")),
        ("[fd00::13]:8000".to_string(), on_job_1("gpu03")),
        ("gpu04.cluster:8000".to_string(), on_job_1("gpu04")),
        (
            "10.0.1.2:8000".to_string(),
            format!("smgtest/2/{}/0/n2/regular", start(2)),
        ),
    ];
    expected.sort();
    assert_eq!(found, expected);
    assert_eq!(
        *slurm.node_lookups.lock().unwrap(),
        [vec!["gpu01", "gpu02", "gpu03", "gpu04", "n2"]]
    );
}

/// A requeued job keeps its id but starts again: its worker is a new
/// instance at the same address.
#[tokio::test]
async fn a_requeue_is_a_new_instance_at_the_same_address() {
    let mut slurm = FakeSlurm::recorded();
    let config = config(&["smg-decode"]);
    let before = poll(&slurm, &config).await.unwrap().workers;
    let job = slurm.job_mut(4);
    job.restart_count += 1;
    job.start_time = job.start_time.map(|start| start + 60);
    let after = poll(&slurm, &config).await.unwrap().workers;

    assert_eq!(before.len(), 1);
    assert_eq!(after.len(), 1);
    assert_eq!(before[0].endpoint, after[0].endpoint);
    assert_ne!(before[0].discovery_id, after[0].discovery_id);
    assert!(
        after[0].discovery_id.starts_with("smgtest/4/"),
        "{}",
        after[0].discovery_id
    );
}

/// A job with a node `scontrol` does not know is left out whole; the rest
/// of the poll stands.
#[tokio::test]
async fn a_job_on_a_node_scontrol_does_not_know_is_left_out() {
    let mut slurm = FakeSlurm::recorded();
    slurm.nodes.retain(|node| node.name != "n2");
    let outcome = poll(&slurm, &config(&["smg-worker"])).await.unwrap();
    assert_eq!(urls_of(&outcome), ["10.0.1.1:8000"]);
    assert_eq!(
        outcome.skipped,
        [Skip {
            reason: series::DISCOVERY_SKIP_INVALID,
            message: "job 2 (smg-worker): node n2 is missing from scontrol's answer".to_string(),
        }]
    );
}

/// Slurm shares a node between jobs unless they run `--exclusive`, but
/// only one server can listen on the port: the one of the job that
/// started first, as a later server cannot bind a port in use. That job
/// keeps the address, a later one is left out there, and every other
/// placement stands, the later job's other nodes included.
#[tokio::test]
async fn two_jobs_on_one_node_leave_it_to_the_first_started() {
    let mut slurm = FakeSlurm::recorded();
    // Job 2 shares job 1's node, and has a node of its own too.
    slurm.job_mut(2).nodes = "n1,n2".to_string();
    slurm
        .hostlists
        .insert("n1,n2".to_string(), vec!["n1".into(), "n2".into()]);
    let mut config = config(&["smg-worker", "smg-decode"]);
    config
        .roles
        .insert("smg-decode".to_string(), WorkerType::Regular);
    let on_n1 = |outcome: &PollOutcome| {
        let worker = outcome
            .workers
            .iter()
            .find(|worker| worker.endpoint.render() == "10.0.1.1:8000");
        worker.unwrap().discovery_id.clone()
    };
    let conflict = |outcome: &PollOutcome| {
        assert_eq!(outcome.skipped.len(), 1, "{:?}", outcome.skipped);
        let skip = outcome.skipped[0].clone();
        assert_eq!(skip.reason, series::DISCOVERY_SKIP_CONFLICT);
        skip.message
    };

    // Started in the same second as job 1, job 2 is taken as the later.
    let outcome = poll(&slurm, &config).await.unwrap();
    let urls = ["10.0.1.1:8000", "10.0.1.2:8000", "10.0.1.4:8000"];
    assert_eq!(urls_of(&outcome), urls);
    assert!(on_n1(&outcome).starts_with("smgtest/1/"));
    let message = conflict(&outcome);
    assert!(
        message.starts_with("smgtest/2/")
            && message.contains("places a worker at 10.0.1.1:8000, where smgtest/1/"),
        "{message}"
    );

    // Started a second before job 1, job 2 keeps the node.
    let first = slurm.job_mut(1).start_time.unwrap();
    slurm.job_mut(2).start_time = Some(first - 1);
    let outcome = poll(&slurm, &config).await.unwrap();
    assert_eq!(urls_of(&outcome), urls);
    assert!(on_n1(&outcome).starts_with("smgtest/2/"));
    let message = conflict(&outcome);
    assert!(
        message.starts_with("smgtest/1/") && message.contains("where smgtest/2/"),
        "{message}"
    );
}

/// With no job to place, `scontrol` is not asked about nodes at all: asked
/// about none, it lists every node.
#[tokio::test]
async fn no_selected_job_asks_about_no_node() {
    let slurm = FakeSlurm::recorded();
    assert!(poll(&slurm, &config(&["no-such-job"]))
        .await
        .unwrap()
        .workers
        .is_empty());
    assert!(slurm.node_lookups.lock().unwrap().is_empty());
}

/// A federation can list its other clusters' jobs (an older release may
/// ignore `--local` under `--json`, as 23.11 ignores most filters there).
/// `scontrol` resolves node names in the local cluster only, so a remote
/// job on a node named like a local one must not take that node's
/// address, nor a remote-only node reject the poll: both are left out.
#[tokio::test]
async fn jobs_of_another_cluster_are_left_out() {
    let mut slurm = FakeSlurm::recorded();
    // Job 2 runs in a remote cluster, on a node named like job 1's.
    let remote = slurm.job_mut(2);
    remote.cluster = "remote".to_string();
    remote.nodes = "n1".to_string();
    // Alice's job 8, as root's, runs remotely on a node the local cluster
    // does not have.
    let only_remote = slurm.job_mut(8);
    only_remote.cluster = "remote".to_string();
    only_remote.user_name = "root".to_string();
    only_remote.nodes = "remote-only".to_string();

    assert_eq!(
        urls(&slurm, &config(&["smg-worker"])).await,
        ["10.0.1.1:8000"]
    );
    assert_eq!(*slurm.node_lookups.lock().unwrap(), [vec!["n1"]]);
}

#[tokio::test]
async fn the_local_cluster_must_be_named() {
    let mut slurm = FakeSlurm::recorded();
    slurm.jobs.cluster = None;
    let error = poll(&slurm, &config(&["smg-worker"])).await.unwrap_err();
    assert!(error.to_string().contains("cluster it ran in"), "{error}");
}

#[tokio::test]
async fn the_default_scope_needs_squeue_to_name_its_user() {
    let mut slurm = FakeSlurm::recorded();
    slurm.jobs.requested_by = None;
    let error = poll(&slurm, &config(&["smg-worker"])).await.unwrap_err();
    assert!(
        error.to_string().contains("configure the Slurm user"),
        "{error}"
    );

    let named = SlurmProviderConfig {
        user: Some("root".to_string()),
        ..config(&["smg-worker"])
    };
    assert_eq!(urls(&slurm, &named).await.len(), 2);
}

/// A hostlist `scontrol` cannot expand leaves its job out; `scontrol`
/// failing as a command leaves the answer unread, and rejects the poll.
#[tokio::test]
async fn an_expansion_that_fails_leaves_its_job_out_and_a_failed_command_the_poll() {
    let mut slurm = FakeSlurm::recorded();
    slurm.hostlists.remove("n1");
    let outcome = poll(&slurm, &config(&["smg-worker"])).await.unwrap();
    assert_eq!(urls_of(&outcome), ["10.0.1.2:8000"]);
    assert_eq!(outcome.skipped.len(), 1);
    assert!(outcome.skipped[0]
        .message
        .starts_with("job 1 (smg-worker): "));

    slurm.expansion_fails = true;
    let error = poll(&slurm, &config(&["smg-worker"])).await.unwrap_err();
    assert!(matches!(error, SlurmError::Command { .. }), "{error}");
}

#[test]
fn squeue_is_asked_for_the_mapped_names_in_one_scope() {
    assert_eq!(
        config(&["smg-worker", "a-model"]).squeue_args(),
        [
            "--json",
            "--local",
            "--states=RUNNING",
            "--name=a-model,smg-worker",
            "--me"
        ]
    );
    let scoped = SlurmProviderConfig {
        user: Some("svc".to_string()),
        account: Some("acct".to_string()),
        partition: Some("gpu".to_string()),
        ..config(&["w"])
    };
    assert_eq!(
        scoped.squeue_args(),
        [
            "--json",
            "--local",
            "--states=RUNNING",
            "--name=w",
            "--user=svc",
            "--account=acct",
            "--partition=gpu"
        ]
    );
}

#[test]
fn modes_without_worker_pools_cannot_use_slurm_discovery() {
    assert!(SlurmProviderConfig::new(&discovery_config(&["w"]), None).is_err());
}

// ---- The commands, against scripts standing in for them. ----

#[cfg(unix)]
fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;

    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

#[cfg(unix)]
fn cli(squeue: PathBuf, scontrol: PathBuf) -> SlurmCli {
    SlurmCli {
        squeue,
        scontrol,
        timeout: Duration::from_secs(5),
    }
}

/// The start refuses an image without the commands: the first one missing
/// is named, looked up on the given `PATH` for a bare name and as the
/// file it names otherwise, and the error names the image that has them.
#[cfg(unix)]
#[test]
fn a_missing_command_is_named_before_the_first_poll() {
    let dir = tempfile::tempdir().unwrap();
    let on_path = Some(dir.path().as_os_str());
    let commands = SlurmCli::default();
    assert_eq!(commands.missing_command(None), Some(Path::new("squeue")));
    script(dir.path(), "squeue", "exit 0");
    assert_eq!(
        commands.missing_command(on_path),
        Some(Path::new("scontrol"))
    );
    script(dir.path(), "scontrol", "exit 0");
    assert_eq!(commands.missing_command(on_path), None);

    let absent = dir.path().join("absent");
    let by_path = cli(dir.path().join("squeue"), absent.clone());
    assert_eq!(by_path.missing_command(None), Some(absent.as_path()));
    assert_eq!(
        cli(dir.path().join("squeue"), dir.path().join("scontrol")).missing_command(None),
        None
    );

    let error = missing_command_error(Path::new("squeue"));
    assert!(error.starts_with("`squeue` is not on PATH"), "{error}");
    assert!(error.contains("`-slurm` variant"), "{error}");
}

/// The real command path end to end: the arguments `squeue` and `scontrol`
/// are given, and their recorded answers read back.
#[cfg(unix)]
#[tokio::test]
async fn the_commands_are_run_with_their_arguments_and_read() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("squeue.json"), SQUEUE_24_11).unwrap();
    std::fs::write(dir.path().join("nodes.json"), NODES_24_11).unwrap();
    let log = dir.path().join("calls");
    let squeue = script(
        dir.path(),
        "squeue",
        &format!(
            r#"echo "squeue $*" >> {log}; cat {}"#,
            dir.path().join("squeue.json").display(),
            log = log.display()
        ),
    );
    let scontrol = script(
        dir.path(),
        "scontrol",
        &format!(
            r#"echo "scontrol $*" >> {log}
if [ "$1" = show ]; then echo slurmnode; else cat {}; fi"#,
            dir.path().join("nodes.json").display(),
            log = log.display()
        ),
    );
    let slurm = cli(squeue, scontrol);

    let workers = poll(&slurm, &config(&["smg-decode"]))
        .await
        .unwrap()
        .workers;
    assert_eq!(workers.len(), 1);
    assert_eq!(workers[0].endpoint.render(), "127.0.0.1:8000");
    assert_eq!(
        std::fs::read_to_string(&log).unwrap(),
        "squeue --json --local --states=RUNNING --name=smg-decode --me\n\
             scontrol show hostnames slurmnode\n\
             scontrol --json show nodes slurmnode\n"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_command_that_fails_says_how() {
    let dir = tempfile::tempdir().unwrap();
    let failing = script(
        dir.path(),
        "squeue",
        "echo 'squeue: error: Unable to contact slurm controller' >&2; exit 1",
    );
    let slow = script(dir.path(), "slow", "sleep 10");
    let missing = dir.path().join("absent");

    let error = cli(failing, missing.clone())
        .running_jobs(&config(&["w"]))
        .await
        .unwrap_err();
    assert!(matches!(error, SlurmError::Command { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("Unable to contact slurm controller"),
        "{error}"
    );
    assert_eq!(error.metric_reason(), series::SNAPSHOT_ERROR_UNREADABLE);

    let slurm = SlurmCli {
        timeout: Duration::from_millis(200),
        ..cli(slow, missing.clone())
    };
    let started = time::Instant::now();
    let error = slurm.running_jobs(&config(&["w"])).await.unwrap_err();
    assert!(error.to_string().contains("timed out"), "{error}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the command was not killed"
    );

    let error = cli(missing.clone(), missing)
        .expand_hostlist("n1")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("could not start"), "{error}");
}

/// An answer that exits 0 but cannot be used is invalid, and carries what
/// the command wrote to stderr.
#[cfg(unix)]
#[tokio::test]
async fn an_unusable_answer_is_invalid() {
    let dir = tempfile::tempdir().unwrap();
    let scontrol = script(
        dir.path(),
        "scontrol",
        "echo 'Invalid hostlist: gpu[01-' >&2",
    );
    let error = cli(dir.path().join("absent"), scontrol)
        .expand_hostlist("gpu[01-")
        .await
        .unwrap_err();
    assert!(matches!(error, SlurmError::Invalid { .. }), "{error}");
    assert!(error.to_string().contains("Invalid hostlist"), "{error}");
    assert_eq!(error.metric_reason(), series::SNAPSHOT_ERROR_INVALID);
}
