//! Controller replicas as real processes: this package's binary started on
//! ephemeral loopback ports over one lease (a file, or a `Lease` object
//! behind a client configuration of the test's own), each one's bound ports
//! read from its own log, and the generated client asking a replica who
//! leads.
//! Every answer goes through one [`Record`], which holds the election's
//! promise over the whole run: once a replica has answered as the leader
//! under a term, no call sent after that answer returned is answered as the
//! leader under a lower one. What the replicas case and the failover cells
//! share, included by path.

use std::{
    ffi::OsStr,
    io::{BufRead, BufReader, Write},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use prost::Message as _;
use smg_policy_controller::service::{
    elector::ElectionSettings,
    proto::{
        controller_frame, gateway_frame, policy_controller_client::PolicyControllerClient,
        GatewayFrame, Hello, LeaderInfo, LeaderRequest, NotLeader,
    },
};
use tokio::sync::mpsc;
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream, metadata::AsciiMetadataValue,
    transport::Channel, Request, Status,
};

const BIN: &str = env!("CARGO_BIN_EXE_smg-policy-controller");

/// `google.rpc.Status`, as a client of any gRPC stack decodes a status's
/// details.
#[derive(Clone, PartialEq, prost::Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<prost_types::Any>,
}

/// The `NotLeader` in a refusal's details, read the way every gRPC client
/// reads them: a `google.rpc.Status` that says the refusal's code and
/// message again and holds the one message under its type's URL.
pub fn not_leader(refused: &Status) -> NotLeader {
    let details = RpcStatus::decode(refused.details()).expect("a google.rpc.Status");
    assert_eq!(
        (details.code, details.message.as_str()),
        (refused.code() as i32, refused.message())
    );
    assert_eq!(details.details.len(), 1, "one detail: {details:?}");
    let detail = details.details.first().expect("one detail");
    assert_eq!(
        detail.type_url,
        "type.googleapis.com/smg.policy.v1.NotLeader"
    );
    NotLeader::decode(detail.value.as_slice()).expect("a NotLeader")
}

/// How long a start may take before its listeners are in its log.
const START: Duration = Duration::from_secs(30);

/// How long one `Leader` call may take: a killed replica refuses at once,
/// a stopped one answers until its listeners close.
const CALL: Duration = Duration::from_millis(500);

/// Room for a loaded host on top of a lease's own timing.
pub const ROOM: Duration = Duration::from_secs(30);

/// Between two samples.
const PACE: Duration = Duration::from_millis(5);

/// What a replica is never left to find in the test's environment: a
/// cluster's configuration and a proxy. A replica reaches the test's own
/// addresses alone.
const AMBIENT: [&str; 9] = [
    "KUBECONFIG",
    "KUBERNETES_SERVICE_HOST",
    "KUBERNETES_SERVICE_PORT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
];

/// Where the replicas' lease lives.
#[derive(Clone, Copy, Debug)]
pub enum Lease<'a> {
    /// A lease file.
    File(&'a Path),
    /// A `Lease` object of this name, reached through the client
    /// configuration at `kubeconfig`, which names its API server and its
    /// namespace. The replica is given that file and no other.
    Object {
        /// The object's name.
        name: &'a str,
        /// The client configuration.
        kubeconfig: &'a Path,
    },
    /// The lease in the replica's own process, as a controller alone runs
    /// by default: no place is passed.
    Own,
}

/// A measurement's timings: `default`, or the three of
/// `SMG_REPLICAS_TIMINGS_MS` (`duration,renew period,renew deadline`, in
/// milliseconds), printed first so a table names what it measured.
pub fn timings(default: ElectionSettings, out: &mut impl Write) -> ElectionSettings {
    let settings = match std::env::var("SMG_REPLICAS_TIMINGS_MS") {
        Err(_) => default,
        Ok(given) => {
            let counts: Vec<u64> = given
                .split(',')
                .map(|part| part.trim().parse().expect("a count of milliseconds"))
                .collect();
            let [duration, renew_period, renew_deadline] =
                <[u64; 3]>::try_from(counts).expect("SMG_REPLICAS_TIMINGS_MS takes three counts");
            ElectionSettings {
                duration: Duration::from_millis(duration),
                renew_period: Duration::from_millis(renew_period),
                renew_deadline: Duration::from_millis(renew_deadline),
            }
        }
    };
    writeln!(
        out,
        "TIMINGS_MS duration={} renew_period={} renew_deadline={}",
        settings.duration.as_millis(),
        settings.renew_period.as_millis(),
        settings.renew_deadline.as_millis()
    )
    .expect("stdout");
    settings
}

/// The time of day of an RFC 3339 stamp in UTC (`...T08:45:24.089815Z`),
/// in microseconds.
fn micros_of_day(stamp: &str) -> Option<u64> {
    let time = stamp.split_once('T')?.1.trim_end_matches('Z');
    let mut parts = time.splitn(3, ':');
    let hours: u64 = parts.next()?.parse().ok()?;
    let minutes: u64 = parts.next()?.parse().ok()?;
    let seconds = parts.next()?;
    let (seconds, fraction) = seconds.split_once('.').unwrap_or((seconds, "0"));
    let seconds: u64 = seconds.parse().ok()?;
    // The fraction's first six digits, padded: microseconds.
    let micros: u64 = format!("{fraction:0<6}").get(..6)?.parse().ok()?;
    Some(((hours * 60 + minutes) * 60 + seconds) * 1_000_000 + micros)
}

/// One controller process.
pub struct Replica {
    /// Its controller id.
    pub id: String,
    /// The address it advertises, when it was started under one.
    advertised: Option<String>,
    child: Child,
    lines: Arc<Mutex<Vec<String>>>,
    /// The thread that reads the process's log into `lines`, until joined.
    reader: Option<thread::JoinHandle<()>>,
    /// Where its gateway and ingestion scopes listen.
    pub gateway: SocketAddr,
    /// Where its control scope listens.
    pub control: SocketAddr,
    /// Where its replication scope listens.
    pub replication: SocketAddr,
    /// Where its admin endpoints listen.
    pub admin: SocketAddr,
    client: Option<PolicyControllerClient<Channel>>,
    /// The directory of the spec source it was given where its flags named
    /// none, kept as long as the process.
    _spec: Option<tempfile::TempDir>,
    /// It was given a spec file beside a lease another replica can take:
    /// its start's document lands a moment behind the start of its first
    /// lead, and [`Record::agreed`] waits for it.
    filed: bool,
}

/// Where a replica listens unless its start names a place: the loopback
/// address, every port the system's pick.
const LOOPBACK: (IpAddr, u16) = (IpAddr::V4(Ipv4Addr::LOCALHOST), 0);

/// A document of one plain profile: what a replica runs where its flags
/// name no spec source and a store is refused.
const PLAIN_SPEC: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";

impl Replica {
    /// Starts the binary as `id` over `lease`, every listener on an
    /// ephemeral loopback port, and returns once its log names the four
    /// bound addresses. The holder identity is the gateway listener's own
    /// address: no advertised address is passed. Its listeners stay open
    /// `settle_secs` after a stop's signal; its grace period is five seconds,
    /// or twice a settle window that needs more: the binary asks that much.
    pub fn start(
        id: &str,
        lease: Lease<'_>,
        settings: ElectionSettings,
        settle_secs: u64,
    ) -> Result<Self, String> {
        Self::start_as(id, None, lease, settings, settle_secs, &[], LOOPBACK)
    }

    /// Starts the binary as [`Replica::start`] does, under an advertised
    /// `host:port` of its own: the holder identity is `id` at that address
    /// whatever the listeners bind, so a second process started the same way
    /// is the first one's restart to the election.
    pub fn start_at(
        id: &str,
        advertise: &str,
        lease: Lease<'_>,
        settings: ElectionSettings,
        settle_secs: u64,
    ) -> Result<Self, String> {
        Self::start_as(
            id,
            Some(advertise),
            lease,
            settings,
            settle_secs,
            &[],
            LOOPBACK,
        )
    }

    /// [`Replica::start`] with `flags` after its own on the command line.
    pub fn start_with(
        id: &str,
        lease: Lease<'_>,
        settings: ElectionSettings,
        settle_secs: u64,
        flags: &[&OsStr],
    ) -> Result<Self, String> {
        Self::start_as(id, None, lease, settings, settle_secs, flags, LOOPBACK)
    }

    /// [`Replica::start_with`] with every listener at the address `ip` and
    /// the replication listener at `port` there (`0` takes one), its
    /// listeners closed at once on a stop's signal. A standby dials its
    /// leader's host at its own replication port: the replicas of a pair
    /// listen on one replication port, each at an address of its own.
    pub fn start_on(
        id: &str,
        place: (IpAddr, u16),
        lease: Lease<'_>,
        settings: ElectionSettings,
        flags: &[&OsStr],
    ) -> Result<Self, String> {
        Self::start_as(id, None, lease, settings, 0, flags, place)
    }

    fn start_as(
        id: &str,
        advertise: Option<&str>,
        lease: Lease<'_>,
        settings: ElectionSettings,
        settle_secs: u64,
        flags: &[&OsStr],
        (ip, replication): (IpAddr, u16),
    ) -> Result<Self, String> {
        let at = |port: u16| SocketAddr::new(ip, port).to_string();
        let millis = |duration: Duration| duration.as_millis().to_string();
        let grace_secs = (2 * settle_secs).max(5);
        let mut command = Command::new(BIN);
        for variable in AMBIENT {
            command.env_remove(variable);
        }
        let alone = matches!(lease, Lease::Own);
        match lease {
            Lease::File(path) => {
                command.arg("--lease-file").arg(path);
            }
            Lease::Object { name, kubeconfig } => {
                command
                    .args(["--lease-name", name])
                    .env("KUBECONFIG", kubeconfig);
            }
            Lease::Own => {}
        }
        if let Some(address) = advertise {
            command.args(["--advertise-address", address]);
        }
        // A start names its spec's source. A replica whose flags name none
        // takes one that changes nothing it runs where it can: a store on an
        // empty directory, alone. Beside a lease another replica can take a
        // store is refused, so there it takes a file of one plain profile.
        let named = flags.iter().any(|flag| {
            ["--spec-file", "--spec-store"]
                .map(OsStr::new)
                .contains(flag)
        });
        let spec = if named {
            None
        } else {
            let dir =
                tempfile::tempdir().map_err(|error| format!("{id}: a spec directory: {error}"))?;
            if alone {
                command.arg("--spec-store").arg(dir.path());
            } else {
                let file = dir.path().join("policy.yaml");
                std::fs::write(&file, PLAIN_SPEC)
                    .map_err(|error| format!("{id}: a spec file: {error}"))?;
                command.arg("--spec-file").arg(file);
            }
            Some(dir)
        };
        command
            .args(["--controller-id", id])
            .args(["--gateway-listen", &at(0)])
            .args(["--control-listen", &at(0)])
            .args(["--replication-listen", &at(replication)])
            .args(["--admin-listen", &at(0)])
            .args(["--lease-duration-ms", &millis(settings.duration)])
            .args(["--lease-renew-period-ms", &millis(settings.renew_period)])
            .args([
                "--lease-renew-deadline-ms",
                &millis(settings.renew_deadline),
            ])
            .args(["--shutdown-settle-secs", &settle_secs.to_string()])
            .args(["--shutdown-grace-period-secs", &grace_secs.to_string()])
            .args(["--runtime-worker-threads", "2"])
            // A `Session` needs a principal and a replica here has no
            // certificate: its gateway listener takes the name as told.
            .args(["--identity-as-told", "gateway"])
            .args(["--log-level", "info", "--log-json"])
            .args(flags)
            .env_remove("RUST_LOG")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command
            .spawn()
            .map_err(|error| format!("start {id}: {error}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| format!("{id}: no stdout pipe"))?;
        let lines = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&lines);
        let reader = thread::Builder::new()
            .name(format!("log-{id}"))
            .spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    sink.lock().push(line);
                }
            })
            .map_err(|error| format!("{id}: the log reader: {error}"))?;

        let mut replica = Self {
            id: id.to_owned(),
            advertised: advertise.map(str::to_owned),
            child,
            lines,
            reader: Some(reader),
            gateway: unbound(),
            control: unbound(),
            replication: unbound(),
            admin: unbound(),
            client: None,
            filed: spec.is_some() && !alone,
            _spec: spec,
        };
        let started = Instant::now();
        loop {
            let bound = replica.bound();
            if let [Some(gateway), Some(control), Some(replication), Some(admin)] = bound {
                replica.gateway = gateway;
                replica.control = control;
                replica.replication = replication;
                replica.admin = admin;
                return Ok(replica);
            }
            if let Ok(Some(status)) = replica.child.try_wait() {
                return Err(format!(
                    "{id} exited at its start ({status}): {}",
                    replica.log()
                ));
            }
            if started.elapsed() > START {
                return Err(format!(
                    "{id} bound nothing in {START:?}: {}",
                    replica.log()
                ));
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// The gateway, control, replication and admin addresses the log names
    /// so far.
    fn bound(&self) -> [Option<SocketAddr>; 4] {
        let mut bound = [None; 4];
        for line in self.lines.lock().iter() {
            let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let fields = &entry["fields"];
            let addr = fields["addr"].as_str().and_then(|addr| addr.parse().ok());
            let slot = match (fields["message"].as_str(), fields["listener"].as_str()) {
                (Some("listener bound"), Some("gateway")) => 0,
                (Some("listener bound"), Some("control")) => 1,
                (Some("listener bound"), Some("replication")) => 2,
                (Some("admin listener bound"), _) => 3,
                _ => continue,
            };
            bound[slot] = addr;
        }
        bound
    }

    /// The identity this replica holds the lease under.
    pub fn identity(&self) -> String {
        match &self.advertised {
            Some(address) => format!("{}@{address}", self.id),
            None => format!("{}@{}", self.id, self.gateway),
        }
    }

    /// The log so far, one line a line.
    pub fn log(&self) -> String {
        self.lines.lock().join("\n")
    }

    /// The terms of this replica's takes, in order, from its "leading"
    /// lines.
    pub fn takes(&self) -> Vec<u64> {
        self.lines
            .lock()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|entry| entry["fields"]["message"] == "leading")
            .filter_map(|entry| entry["fields"]["term"].as_u64())
            .collect()
    }

    /// Whether the log holds a line whose message is `message`.
    pub fn logged(&self, message: &str) -> bool {
        self.lines
            .lock()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .any(|entry| entry["fields"]["message"] == message)
    }

    /// Whether this replica lost the lease after its last take, by its own
    /// log. A leader stalled past the lease's duration finds the lease
    /// moved, and has none to release at its stop.
    pub fn lost_its_lease(&self) -> bool {
        let lines = self.lines.lock();
        let last = |message: &str| {
            lines.iter().rposition(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .is_ok_and(|entry| entry["fields"]["message"] == message)
            })
        };
        last("the lease moved; the fence is closed") > last("leading")
    }

    /// The time of day, in microseconds, of the first log line whose
    /// message is `message`, by the replica's own clock: two of them give
    /// the time between two of its lines.
    pub fn logged_at_micros(&self, message: &str) -> Option<u64> {
        self.lines
            .lock()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|entry| entry["fields"]["message"] == message)
            .and_then(|entry| entry["timestamp"].as_str().and_then(micros_of_day))
    }

    /// Waits, within the kit's room, until `met` reads true of this
    /// replica's log. A live replica's line reaches the log through a pipe
    /// and a reader thread, a moment after the replica acted on it: a case
    /// reads such a line behind this wait. A stopped replica's log is whole
    /// once [`Replica::exited`] has returned.
    pub async fn logs(&self, met: impl Fn(&Self) -> bool) -> bool {
        let started = Instant::now();
        while !met(self) {
            if started.elapsed() > ROOM {
                return false;
            }
            tokio::time::sleep(PACE).await;
        }
        true
    }

    /// One `Leader` call on the gateway listener; `None` when the replica
    /// did not answer within the call's bound (killed, or gone).
    pub async fn leader(&mut self) -> Option<LeaderInfo> {
        if self.client.is_none() {
            let endpoint = Channel::from_shared(format!("http://{}", self.gateway))
                .ok()?
                .connect_timeout(CALL);
            self.client = Some(PolicyControllerClient::new(
                tokio::time::timeout(CALL, endpoint.connect())
                    .await
                    .ok()?
                    .ok()?,
            ));
        }
        let client = self.client.as_mut()?;
        match tokio::time::timeout(CALL, client.leader(LeaderRequest::default())).await {
            Ok(Ok(answer)) => Some(answer.into_inner()),
            _ => {
                // A broken channel is dialed again at the next call.
                self.client = None;
                None
            }
        }
    }

    /// One `Hello` as `gateway` on a `Session` of its own over the gateway
    /// listener: the term of the `Welcome` that answers it, or the name of
    /// what answered in its place.
    pub async fn hello(&self, gateway: &str) -> Result<u64, String> {
        self.hello_under(gateway, 0).await
    }

    /// The same with the gateway's term floor on the frame.
    pub async fn hello_under(&self, gateway: &str, term_floor: u64) -> Result<u64, String> {
        let endpoint = Channel::from_shared(format!("http://{}", self.gateway))
            .map_err(|error| error.to_string())?
            .connect_timeout(CALL);
        let channel = tokio::time::timeout(CALL, endpoint.connect())
            .await
            .map_err(|_| "no connection".to_owned())?
            .map_err(|_| "no connection".to_owned())?;
        let (frames, sent) = mpsc::channel(1);
        let mut request = Request::new(ReceiverStream::new(sent));
        let name = AsciiMetadataValue::try_from(gateway).map_err(|error| error.to_string())?;
        request.metadata_mut().insert("x-smg-principal", name);
        let mut answers =
            tokio::time::timeout(CALL, PolicyControllerClient::new(channel).session(request))
                .await
                .map_err(|_| "no answer".to_owned())?
                .map_err(|status| format!("{:?}", status.code()))?
                .into_inner();
        let hello = Hello {
            gateway_id: gateway.to_owned(),
            protocol_min: 1,
            protocol_max: 1,
            ..Hello::default()
        };
        frames
            .send(GatewayFrame {
                seq: 0,
                term_floor,
                frame: Some(gateway_frame::Frame::Hello(hello)),
            })
            .await
            .map_err(|_| "the stream closed".to_owned())?;
        let answer = tokio::time::timeout(CALL, answers.message())
            .await
            .map_err(|_| "no answer".to_owned())?
            .map_err(|status| format!("{:?}", status.code()))?
            .ok_or_else(|| "the stream's end".to_owned())?;
        match answer.frame {
            Some(controller_frame::Frame::Welcome(_)) => Ok(answer.term),
            Some(controller_frame::Frame::NotLeader(_)) => Err("NotLeader".to_owned()),
            _ => Err("another frame".to_owned()),
        }
    }

    /// SIGKILL: the process is gone with its lease unreleased.
    pub fn kill(&mut self) -> Result<(), String> {
        self.child
            .kill()
            .map_err(|error| format!("kill {}: {error}", self.id))
    }

    /// SIGTERM, the way an orchestrator stops a pod.
    pub fn terminate(&self) -> Result<(), String> {
        let status = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .map_err(|error| format!("signal {}: {error}", self.id))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("kill -TERM {}: {status}", self.id))
        }
    }

    /// The exit status, waited for under the kit's own bound, with the log
    /// read to its end: every line the process wrote on its way out is in
    /// the log from here. A replica that has not exited inside [`ROOM`] is
    /// killed and named, and so is a log that does not end: a process that
    /// hangs at its exit fails its case instead of holding the job.
    pub fn exited(&mut self) -> Result<ExitStatus, String> {
        let waited = Instant::now();
        let status = loop {
            let exited = self
                .child
                .try_wait()
                .map_err(|error| format!("wait {}: {error}", self.id))?;
            if let Some(status) = exited {
                break status;
            }
            if waited.elapsed() >= ROOM {
                // Best effort: the case fails by the name below either way.
                let _ = self.child.kill();
                let _ = self.child.wait();
                return Err(format!("{} did not exit in {ROOM:?}", self.id));
            }
            thread::sleep(PACE);
        };
        if let Some(reader) = self.reader.take() {
            while !reader.is_finished() {
                if waited.elapsed() >= ROOM {
                    return Err(format!(
                        "{}: its log did not end in {ROOM:?} of its exit's wait",
                        self.id
                    ));
                }
                thread::sleep(PACE);
            }
            reader
                .join()
                .map_err(|_| format!("{}: the log reader panicked", self.id))?;
        }
        Ok(status)
    }

    /// The CPU time the process has run, every thread, in nanoseconds
    /// (`/proc/<pid>/task/*/schedstat`); `0` where `/proc` has no answer.
    pub fn cpu_nanos(&self) -> u64 {
        let tasks = format!("/proc/{}/task", self.child.id());
        let Ok(entries) = std::fs::read_dir(tasks) else {
            return 0;
        };
        entries
            .filter_map(Result::ok)
            .filter_map(|task| std::fs::read_to_string(task.path().join("schedstat")).ok())
            .filter_map(|stat| {
                stat.split(' ')
                    .next()
                    .and_then(|ran| ran.parse::<u64>().ok())
            })
            .sum()
    }

    /// The process's resident size in KiB, from `/proc/<pid>/status`; `0`
    /// where `/proc` has no answer.
    pub fn rss_kib(&self) -> u64 {
        std::fs::read_to_string(format!("/proc/{}/status", self.child.id()))
            .ok()
            .and_then(|status| {
                status.lines().find_map(|line| {
                    line.strip_prefix("VmRSS:")
                        .and_then(|rest| rest.trim().split(' ').next())
                        .and_then(|kib| kib.parse().ok())
                })
            })
            .unwrap_or(0)
    }
}

impl Drop for Replica {
    fn drop(&mut self) {
        // A replica a test left running is not left behind it.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn unbound() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 0))
}

/// One `Leader` answer with the instants its call was sent and returned.
#[derive(Clone, Debug)]
pub struct Answer {
    /// The controller id of the replica that answered.
    pub replica: String,
    /// When the call was sent.
    pub sent: Instant,
    /// When its answer returned.
    pub returned: Instant,
    /// The answer.
    pub info: LeaderInfo,
}

/// Every answer of a run, in the order the calls returned.
#[derive(Default)]
pub struct Record {
    answers: Vec<Answer>,
}

impl Record {
    /// Asks `Leader` of each replica in turn and records the answers; a
    /// replica that did not answer has no entry. Each call is its own
    /// instant: [`Record::check`] reads the sent and returned instants, not
    /// the sample.
    pub async fn sample(&mut self, replicas: &mut [Replica]) -> Vec<Answer> {
        let mut answers = Vec::with_capacity(replicas.len());
        for replica in replicas.iter_mut() {
            let sent = Instant::now();
            if let Some(info) = replica.leader().await {
                answers.push(Answer {
                    replica: replica.id.clone(),
                    sent,
                    returned: Instant::now(),
                    info,
                });
            }
        }
        self.answers.extend(answers.iter().cloned());
        answers
    }

    /// Samples until one replica answers as the leader and every replica
    /// names it under its term: the leader's place in `replicas` and the
    /// term. A leader this helper gave a file is agreed once its log says
    /// a document was applied: its start's lands a moment behind the start
    /// of its lead, and until then it has no decision for a request.
    pub async fn agreed(&mut self, replicas: &mut [Replica]) -> Result<(usize, u64), String> {
        let started = Instant::now();
        loop {
            let answers = self.sample(replicas).await;
            let leaders: Vec<&Answer> = answers.iter().filter(|a| a.info.is_leader).collect();
            if let ([leader], true) = (&leaders[..], answers.len() == replicas.len()) {
                let named = answers.iter().all(|answer| {
                    answer.info.controller_id == leader.info.controller_id
                        && answer.info.address == leader.info.address
                        && answer.info.term == leader.info.term
                });
                let place = replicas.iter().position(|r| r.id == leader.replica);
                if let (true, Some(place)) = (named, place) {
                    let leads = &replicas[place];
                    if !leads.filed || leads.logged("policy spec applied") {
                        return Ok((place, leader.info.term));
                    }
                }
            }
            if started.elapsed() > ROOM {
                return Err(format!(
                    "the replicas did not agree on a leader in {ROOM:?}: {answers:?}"
                ));
            }
            tokio::time::sleep(PACE).await;
        }
    }

    /// Samples until a replica answers as the leader under a term above
    /// `term`: that answer.
    pub async fn promoted(
        &mut self,
        replicas: &mut [Replica],
        term: u64,
        within: Duration,
    ) -> Result<Answer, String> {
        let started = Instant::now();
        loop {
            let answers = self.sample(replicas).await;
            if let Some(answer) = answers
                .into_iter()
                .find(|answer| answer.info.is_leader && answer.info.term > term)
            {
                return Ok(answer);
            }
            if started.elapsed() > within {
                return Err(format!("no replica led above term {term} in {within:?}"));
            }
            tokio::time::sleep(PACE).await;
        }
    }

    /// How many answers the record holds.
    pub fn len(&self) -> usize {
        self.answers.len()
    }

    /// The election's promise over every answer recorded, or the first
    /// pair of answers that breaks it:
    /// - once a replica has answered as the leader under a term, no call
    ///   sent after that answer returned is answered as the leader under a
    ///   lower term (two such answers are two open fences);
    /// - one replica's answers never fall in term.
    pub fn check(&self) -> Result<(), String> {
        let leaders: Vec<&Answer> = self
            .answers
            .iter()
            .filter(|answer| answer.info.is_leader)
            .collect();
        for later in &leaders {
            for earlier in &leaders {
                if earlier.info.term > later.info.term && later.sent > earlier.returned {
                    return Err(format!(
                        "{} answered as the leader under term {} to a call sent after {} had answered as the leader under term {}",
                        later.replica, later.info.term, earlier.replica, earlier.info.term
                    ));
                }
                if earlier.info.term == later.info.term && earlier.replica != later.replica {
                    return Err(format!(
                        "{} and {} both answered as the leader under term {}",
                        earlier.replica, later.replica, later.info.term
                    ));
                }
            }
        }
        let mut last: Vec<(&str, u64)> = Vec::new();
        for answer in &self.answers {
            match last
                .iter_mut()
                .find(|(replica, _)| *replica == answer.replica)
            {
                Some((replica, term)) => {
                    if answer.info.term < *term {
                        return Err(format!(
                            "{replica} answered term {} after term {term}",
                            answer.info.term
                        ));
                    }
                    *term = answer.info.term;
                }
                None => last.push((&answer.replica, answer.info.term)),
            }
        }
        Ok(())
    }
}
