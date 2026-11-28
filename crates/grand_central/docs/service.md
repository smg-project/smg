# Run the service

Grand Central's standalone service is one binary. A gateway you host asks it for a worker over gRPC. The stock gateway decides in its own process and does not call it. This page covers its flags, its worker discovery, the calls it serves and the client a host calls it with.

The [README](../README.md#as-a-standalone-service) has the short start. [Embed the engine](embedding.md) covers a host that decides in its own process.

## Start the service

Build from a checkout with `protoc` available. The service feature uses the worker protocol files from the checkout.

- The stock binary requires the `controller` feature, which enables `worker-host`.
- Without `controller`, cargo builds no binary and says nothing.

Save a policy spec as `policy.yaml`, then run the following from the checkout root. The [README's first spec](../README.md#try-it) will do.

```sh
cargo build --release --locked -p smg-grand-central --features controller
target/release/smg-grand-central --controller-id controller-0 --spec-file policy.yaml \
  --gateway-listen 127.0.0.1:30100 --control-listen 127.0.0.1:30101 \
  --replication-listen 127.0.0.1:30102 --admin-listen 127.0.0.1:30103 \
  --identity-as-told gateway,control
```

Then use the service:

- Register workers with `RegisterWorker` on the control listener.
- Use `Session` and `Decide` on the gateway listener to ask for a worker.
- The service returns targets and leases for the caller to dispatch and report.
- [The controller's wire test](../tests/controller_cases/decisions.rs) is a complete local example of registration and the decision stream.
- The [deployment chart](../../../deploy/helm/smg/README.md#grand-central) can mount `grandCentral.spec` as a file or `grandCentral.specStoreClaim` as a store.

The tables below list the flags by purpose. [Worker discovery](#worker-discovery) has its own flags.

### The spec and the controller's name

| Flag | Default | Effect |
|---|---|---|
| `--help`, `--version` | none | Show help or the version, and exit. |
| `--controller-id` | the value of `HOSTNAME` | The controller's name. Use letters, digits, `.`, `_` and `-`. |
| `--spec-file` | none | The policy spec as a watched YAML or JSON file. |
| `--spec-store` | none | The policy spec as an existing directory, managed through the spec calls. |

Choose exactly one of `--spec-file` and `--spec-store`.

- A spec store serves one replica, its lease in the process. Beside a shared lease the start is refused.
- A store with no generation on file is a source too. The controller starts, reads ready and declines every decision until the first apply.

### Listeners

| Flag | Default | Effect |
|---|---|---|
| `--gateway-listen` | `0.0.0.0:30100` | The gateway scope's listener. |
| `--control-listen` | `0.0.0.0:30101` | The control scope's listener. |
| `--replication-listen` | `0.0.0.0:30102` | The replication scope's listener. |
| `--admin-listen` | `0.0.0.0:30103` | The admin listener. It uses HTTP. |
| `--identity-as-told` | empty | Listener names among `gateway`, `control` and `replication`, separated by commas. |

- By default the four listeners bind every IPv4 address.
- A listener named by `--identity-as-told` accepts one `x-smg-principal` value as the caller's identity. Use it on loopback or behind a trusted proxy.
- The [table of calls](#the-calls) names each scope's calls.

### TLS and identity

| Flag | Default | Effect |
|---|---|---|
| `--tls-cert`, `--tls-key` | none | The PEM server chain and its key, for the gRPC listeners. Give the two together. |
| `--gateway-client-ca`, `--control-client-ca`, `--replication-client-ca` | none | A PEM client CA for one scope. |

- A client CA requires server TLS. It conflicts with `--identity-as-told` on that listener.
- A client CA requires a verified certificate at transport level. This build has no certificate-to-principal mapping.
- Calls requiring a principal use a listener configured with `--identity-as-told`.
- A stated subject is a visible ASCII token of at most 253 bytes without commas.
- The service checks the scope before the principal.

### Leader election

| Flag | Default | Effect |
|---|---|---|
| `--lease-file` | none | A file lease, for replicas on one host. Its parent directory must exist. |
| `--lease-name` | none | A named `Lease`. It needs the `kubernetes` feature. |
| `--lease-namespace` | the client's namespace | The namespace of the named `Lease`. It needs `--lease-name`. |
| `--advertise-address` | the gateway listener's address, unless that is a wildcard | A dialable host or host:port. A bare host uses the gateway port. |
| `--lease-duration-ms` | 3000 | Milliseconds a follower sees a lease stand unrenewed before it takes it. |
| `--lease-renew-deadline-ms` | 2000 | Milliseconds a leader serves on, counted from the start of its last renewal. Past them it fences itself. |
| `--lease-renew-period-ms` | 1000 | Milliseconds between two renewals of the leader, and between two looks of a follower. |

- With no lease flag the lease is local to the process.
- The two shared lease types, the file and the named `Lease`, are exclusive.
- A shared lease requires `--advertise-address` when the gateway listener binds a wildcard address.
- The duration must exceed the deadline, which must exceed the positive period.
- A named `Lease` takes a duration of whole seconds.

### Load monitoring

| Flag | Default | Effect |
|---|---|---|
| `--load-monitor-interval` | 10 | Seconds between load polls, positive. Also the freshness window for a stream report to stand in for a poll. |
| `--worker-overload-waiting-requests` | 8 | A default overload threshold: a count of queued requests, from 1. |
| `--worker-overload-token-usage` | 0.8 | A default overload threshold: a KV usage above 0 and at most 1. |

- The two overload flags are the gateway's flags of the same names.
- They are the default thresholds of an engine's worker. A threshold its registration states stands over the default.
- A spec whose bindings all say `overload.enabled: false` switches the defaults off. See [Overload](policy-spec.md#overload).
- A count over 2^63 - 1 is refused.
- A worker of the `external` runtime takes no default.
- Start every replica with the same two: each judges by its own while it leads.

### Shutdown, threads and logs

| Flag | Default | Effect |
|---|---|---|
| `--shutdown-settle-secs` | 5 | The settle window, in seconds. At most half of the grace period. |
| `--shutdown-grace-period-secs` | 30 | The grace period, in seconds. It is positive and includes the settle window. |
| `--decide-shards` | available parallelism, at most 8 | From 1 to 256. Each model maps to one shard with room for 128 jobs. |
| `--runtime-worker-threads` | the async runtime's default | From 1 to 4096 when set. |
| `--log-level` | `info` | Also accepts `debug`, `warn` and `error`. |
| `--log-json` | off | Write logs as JSON. Text output is the default. |

- Readiness drops before the listeners close.
- `RUST_LOG` can set the log filter.
- `NO_COLOR`, `FORCE_COLOR` and `CLICOLOR_FORCE` control text color, with force taking precedence.

### What a start refuses

Startup refuses these, among others:

- missing or duplicate scopes;
- duplicate listener names;
- two listeners on overlapping addresses, the admin listener among them, unless the port is 0;
- unusable TLS material;
- lease timings out of order;
- an advertised address that is the wildcard, or whose port is no number from 1 to 65535;
- an advertised host that is empty, or that is no IP address and has a character outside ASCII letters, digits, `.`, `-` and `_`;
- an advertised host that is no IP address and whose last label is a number, like `10.1` or `0xcafe`.

The last label is read with one closing dot set aside. It is a number in two forms. One is one or more digits and nothing else. The other is `0x`, in either case, and then hexadecimal digits or none. Every other host that is no IP address is taken as given, one with an empty label among them.

[Flags](../src/service/controller/run.rs) · [Listener configuration](../src/service/controller/config.rs)

## Worker discovery

A controller built with the `discovery` feature looks for its workers by itself. The replica that leads registers each worker a pass finds, and removes each one its source lists no more. The feature is off by default, and the stock image does not build it yet.

```sh
cargo build --release --locked -p smg-grand-central --features discovery
```

The binary then takes the gateway's worker discovery flags, under the same names and with the same defaults. `--discovery-provider` selects one of `kubernetes`, `file`, `slurm` and `consul`. `--service-discovery` still selects Kubernetes.

### Flags of every provider

| Flag | Default | Effect |
|---|---|---|
| `--discovery-provider` | unset | One of `kubernetes`, `file`, `slurm` and `consul`. With neither this nor `--service-discovery`, the controller runs no provider. |
| `--service-discovery` | off | The older spelling of `--discovery-provider kubernetes`. Give one of the two, not both. |
| `--discovery-check-interval-secs` | 30 | Seconds between periodic passes, positive. A start that failed is tried again after it. |
| `--api-key`, `--api-key-file` | `SMG_API_KEY`, `SMG_API_KEY_FILE` | A key for discovered workers, or a file that holds it alone. Refused today: see [the worker key](#the-worker-key). |
| `--encode-selector`, `--prefill-selector`, `--decode-selector` | none | Kubernetes: the label selectors of encode, prefill and decode pods. Under any provider they select the roles too. |

- The check interval is refused with no provider. So is a key flag.
- The file and Consul providers refuse a record outside the selected roles. Slurm refuses job names that do not fit them.

### Kubernetes

| Flag | Default | Effect |
|---|---|---|
| `--selector` | none | The label selector of regular worker pods, as `key=value`. |
| `--service-discovery-port` | 80 | The port of a discovered pod that has no `smg.ai/worker-ports` annotation. |
| `--service-discovery-namespace` | every namespace | The namespace to watch. |
| `--model-id-from` | none | A source of a worker's model id in pod metadata: `namespace`, `label:<key>` or `annotation:<key>`. |
| `--kv-connector-annotation` | `smg.ai/kv-connector` | The pod annotation that names a worker's KV connector. |
| `--kv-engine-id-annotation` | `smg.ai/kv-engine-id` | The pod annotation that names a worker's engine ids. |
| `--pairing-protocol-label` | `smg.ai/pairing-protocol` | The pod label that names a worker's prefill/decode pairing protocol. |

A prefill and a decode that both name a protocol pair only when the two are equal, under `lenient` and `strict`.

### File, Slurm and Consul

| Flag | Default | Effect |
|---|---|---|
| `--discovery-file` | none | File: the JSON manifest that lists the workers. It may be created after startup. |
| `--slurm-job-name`, `--slurm-encode-job-name`, `--slurm-prefill-job-name`, `--slurm-decode-job-name` | none | Slurm: the job names whose nodes are regular, encode, prefill and decode workers. Each can be given more than once. |
| `--slurm-port` | none | Slurm: the inference port on every node of a discovered job. |
| `--slurm-bootstrap-port` | none | Slurm: the bootstrap port of discovered encode and prefill workers. |
| `--slurm-user` | the user the controller runs as | Slurm: discover this user's jobs. |
| `--slurm-account`, `--slurm-partition` | none | Slurm: discover only the jobs of this account or partition. |
| `--consul-service` | none | Consul: the service whose instances with passing health checks are the workers. |
| `--consul-address` | `CONSUL_HTTP_ADDR`, else 127.0.0.1:8500 | Consul: the HTTP API address. |
| `--consul-datacenter` | the agent's own | Consul: the datacenter. |

`--discovery-file` is required with `file`, and `--consul-service` with `consul`. `slurm` requires `--slurm-port` and at least one job name.

```sh
target/release/smg-grand-central --controller-id controller-0 --spec-file policy.yaml \
  --discovery-provider file --discovery-file /etc/smg/workers.json
```

### Who runs the provider

- The replica that leads runs the provider. A standby runs none.
- The provider stops when the replica loses the lead, when its lease runs out, and at shutdown. The next leader starts its own.
- A provider that cannot start is logged as an error and tried again after the check interval. So is a provider that ends by itself.

### How the settings are checked

- A Slurm or Consul flag, or `--discovery-file`, given with another provider or with none, is refused at the start. The Kubernetes flags are not.
- Under another provider the role selectors select the roles, and the other Kubernetes flags, `--selector` among them, do nothing.
- A selector that is not `key=value` is still refused, and so is a malformed `--model-id-from`.
- The controller checks discovery settings with the gateway's own checks, for the roles its selectors and job names select. It refuses bad settings at the start.
- A refusal of settings that do not fit the roles names `--discovery-provider`. It says that the roles follow the selectors and job names given.
- A Slurm bootstrap port with no encode or prefill job name is refused that way. So is a Slurm job name given for two roles.

### The worker key

- The controller reads a key for discovered workers from `--api-key` or `SMG_API_KEY`, or from a file named by `--api-key-file` or `SMG_API_KEY_FILE`.
- Each flag beats its own variable. A key and a key file together are refused, from any source. The key never prints.
- The key is read only when a provider is selected. A key flag with no provider is refused.
- With a provider a key from any source is refused at the start today, under `--api-key`. The controller keeps no key of a worker it finds, and it drops none in silence.

### What a pass writes

- A registration runs a gateway's registration steps on the worker. It then writes the worker's row as provider `discovery`, under the term the replica leads.
- A removal marks the row draining, waits the worker's drain window and takes the row out.
- A replica that does not lead runs no pass. A write is refused where the replica does not lead when it is made, and what its job wrote before stays: a drain, or a registration's earlier ranks.
- No flag sets a registration's timing yet. A registration takes a gateway's defaults: a pause of 30 seconds between two probes of a starting worker, and a startup timeout of 1800 seconds.
- A found worker's row states the worker crate's default health checks.
- A found gRPC worker's row asks for its engine's cache events where the policy spec in force at its write reads them. The spec is asked for the worker's first model, in the phase of the worker's role. A spec applied later does not write that word again.

### What a pass does not register

- The controller dials a found worker with no client identity and no CA of its own, and no flag sets either.
- A worker that requires mTLS is never registered: every probe of it fails. So does every probe of a worker under a private CA that the host's trust store does not hold.
- A worker whose row another maker holds is not the provider's. A `RegisterWorker` call's row is one such.
- Where a source lists that worker too, a pass can ask its registration again. Such a registration writes nothing: the row stays its maker's.

### Roles

- The controller has no routing mode. A prefill or decode selector or job name selects the prefill and decode pair. An encode one beside them selects all three roles.
- Under Slurm a job's role comes from the job-name list that names it, and the names must fit the selected roles.
- `--router-selector` and `--model-alias` stay with the gateway.

[What is not built yet](../REFERENCE.md#what-is-not-built-yet) lists what discovery does not do yet. [Discovery flags](../src/service/controller/discovery/flags.rs)

## A gateway that calls the service

A gateway you host can call the standalone service. Use the generated client or the crate's `service::client::Client` or `service::client::blocking::Client`. The crate's clients need the `client` feature, which implies `controller`.

**The reference client.**

- It dials and asks `Leader`, opens a session and sends one decision at a time.
- A session opened on a standby is answered `NotLeader`, with the leader's address where the standby knows it.
- It reports `dispatched`, `completed` and `released` when its caller asks.
- Use `connect_with` for a configured TLS endpoint.
- The client exposes `welcome`, `term_floor`, `gateway_id`, `hashes_cut`, `session_end` and `watch_workers`.
- `Watcher` can apply worker events and resume positions to a local store.

**Concurrent requests.**

- Clone a `service::client::Link`: it shares one session and one decide stream.
- An ask takes one credit and waits for its own answer inside its budget. With no credit left an ask does not wait for one: it is refused at once.
- Reports go through a bounded queue, and a full queue ends the link.

**What stays with the host.** The host handles dispatch, fallbacks, retries, lease ownership and reconnection. There is no bundled operator CLI.

[Reference client](../src/service/client.rs) · [Worker watcher](../src/service/watcher.rs)

### The calls

| Scope | Call | What this build serves |
|---|---|---|
| Any gRPC listener | `Leader` | The known leader, its address and its term. |
| Gateway | `Session` | A gateway's session: its hello, then its dispatch, completion, release and veto frames. |
| Gateway | `Decide` | A bidirectional stream of decisions for a present gateway. |
| Gateway | `WatchWorkers` | A snapshot, or a resume from a position, then membership and status changes. For all models or a named set. |
| Control | `RegisterWorker` | `ADD` (default), `REPLACE`, or `UPSERT` by URL. |
| Control | `PatchWorker`, `DeregisterWorker` | Guard by worker ID and revision. |
| Control | `GetSnapshot` | The leader's redacted `SLICE_WORKERS` and `SLICE_SPEC`. |
| Control | `Explain` | Read a sampled decision by gateway and request ID on the leader. |
| Control | `ApplyPolicySpec`, `GetPolicySpec`, `RollbackPolicySpec` | Validate, read or change specs, as [Updating a spec](guarantees.md#updating-a-spec) describes. |
| Control | `StepDown`, `WatchRegime` | Return `UNIMPLEMENTED` after admission. |
| Replication | `Replicate` | A standby follows the leader's term. It replays retained events or restores a snapshot. |

### What each call does

**`Leader`.** One lease covers all models. The call takes no principal.

**`Session`.**

- Start with `Hello` naming the caller and a protocol range containing 1 or 2.
- `Welcome` returns identity and limits. Its spec generation is 0 and it leaves needs and hashing unset.
- A new session replaces the earlier one for that gateway. Hello can restore vetoes.
- Lease and pin restoration via `HelloPage`, `SpecAck` and pushing new specs to gateway sessions are unserved.

**`Decide`.**

- It accepts single decisions, a pair at its prefill, and a decode call that names the pair's lease.
- It accepts an encode request that carries its items (`DecideRequest.encode`).
- It accepts a prefill or a decode request that its plan pairs no side of.
- A decode call its lease refuses gives `Busy` under a reason of its own.
- An encode request's answer names its encoders beside its pick (`DecideAnswer.encode`): one per item, or one for all its items where it sets `one`. No lease is issued for an encoder.
- A request can be answered `Busy` with an unspecified reason, and its gateway then decides it. Among the cases: a frame of the encode phase that does not state `DecideRequest.encode`, and a decode request that names no lease under a pair plan.

**`WatchWorkers`.**

- Resume from a term and sequence. A position outside the retained history can require a new snapshot.
- The service refuses `with_needs`.

**The worker calls.**

- An upsert that changes role, wire, models, demand facts or whether the worker takes images makes a new incarnation.
- `RegisterWorker`'s `wait_ready` requires probes disabled.
- A patch sets and removes labels and can replace overload thresholds.
- A deregistration can mark draining, or remove. A drain needs a later remove.
- A patch or a deregistration of a row that another maker holds is refused with `PERMISSION_DENIED`, whatever revision it names. A row that discovery wrote is one such.
- A decision made behind an acknowledged registration, drain or removal reads the fleet that call left.

**`GetSnapshot`.** Name the slices explicitly. Empty requests ask for all nine and are refused. The spec slice includes retained generations with a store.

**`Replicate`.** Automatic following uses the advertised leader host and the same fixed replication port on each replica. A replica that follows refuses port 0 on its replication listener. It runs in cleartext with `--identity-as-told replication`.

### Health and admin

- The controller exposes the gRPC health service on each gRPC listener.
- The admin listener serves `/liveness` and `/health` even while draining.
- It serves `/readiness`: 200 for a leader and a ready follower, 503 otherwise.
- It serves `/version` and `/metrics`.

[Controller protocol](../contracts/grand_central.proto)

## Holding a decision

A host that decides in process calls `DecisionGuard::decide` with its door. A host with the `client` feature can share a `Link` to a controller and ask for the same kind of handle.

```rust
let now = Instant::now();
let mut handle = DecisionGuard::<&InProcess>::asked(&link, &request, now, budget).await?;
```

**Budgets.**

- Give the controller time to decide in `request.facts.budget_micros`. On the wire, zero means that time is spent.
- A request built for the door states no budget with 0, so set one before you ask. The handle answers a zero budget `DecisionUnavailable::Late` before anything is sent.
- A request that states a hard demand (`Constraints::demands`) states it on its frame, over a session on protocol 2.
- Behind the budget's test, a request that states one over a session on protocol 1 is answered `DecisionUnavailable::Unsent` before it is sent.
- The separate `budget` is how long the link waits for the answer.
- The handle counts the time to dispatch and the lease's dispatch span from `now`.

**Demands.**

- A request states its hard demands in `Constraints::demands`, and a worker's registration states what it offers (`WorkerFacts::demand_facts`).
- The door keeps the workers that meet them before any stage runs.
- A registration that changes what a worker states starts another life of the worker, as a change of its wires does.
- The stock controller's `Decide` request states a request's demands on every frame (`DecideRequest.demands`).
- A registration states whether its worker takes images (`WorkerSpec.vision`), so a demand for vision crosses too.

**Reports.**

- Use `take` to choose a target, then say `dispatched` and `completed`, or `released` if the request is given up.
- For a pair, use `dispatched_decode` for the decode and name each side in its completion.
- Dropping a handle gives up any sides still owed.
- A linked handle reports on the session. On a session that stands, the controller ends the lease's side with the outcome, where that side still stands.
- The controller's session rule covers leases when the link ends.

**What a linked answer holds.**

- Linked targets carry worker ids and ranks.
- Candidate positions stay with the controller.
- A shed keeps its stage where that is one a demand sheds under: `wires`, `vision`, `references`, `pipeline`, `context-window` or `decoding`. Any other stage reads `controller`.
- An answer the handle cannot read returns `DecisionUnavailable::Unsent` and the link releases its leases as late.
- The caller handles an unanswered request.

## Following a controller's workers

With the `client` feature, a host can keep a copy of the controller's worker rows in a `MemoryStore` through a `Watcher`.

- `Watcher::new(store, objects)` takes the store and the host's `Objects`, and the watcher writes each row through them. So a change's worker object stands when the event that told it returns, and a snapshot's rows stand when its last page does.
- Use `Watcher::open_on(&link)` to take the snapshot and the changes that follow it over the link's channel.
- The watch opens within the same five-second bound as `Client::watch_workers`.
- Once open, it has its own stream, so the link can answer asks and carry reports while the watch waits.
- Use one link for the gateway id. A second session with that id takes the first one's place.

A watch can outlive the link. A host that follows only while the link stands can wait for either to end.

```rust
let mut following = watcher.open_on(&link).await?;
let end = loop {
    tokio::select! {
        biased;
        () = link.closed() => break None,
        event = following.next() => match event {
            Ok(true) => {}
            result => break Some(result),
        },
    }
};
```

`None` means the link ended. `Some` gives the watch's own end. `Link::ended()` gives the cause once the link ends, until its next session opens.

- A closed send queue can refuse an ask before its stream reader sees the end.
- A report with no session to take it waits for the link's next session. The line holds 32,768 reports, and a full one drops its oldest.
- A full report queue ends the link at once.

**Opening the watch again.**

- Drop the stream before opening the watch again.
- The watcher asks for changes after the last event it took, or for a snapshot if it has no place.
- When a link ends, the host opens its next session with `Link::redial`, which sends the reports that waited first. A completion that waited the grace period or longer goes as an expired release. A new `Link` holds none of them.
- An ended link lets its channel go, and `Link::watch_workers` on it answers `Error::LinkEnded`.

**A watch that starts from a snapshot.**

- Its last page puts the rows into use through `MemoryStore::watch_from`, and their worker objects with them.
- A worker still present, with the same facts and revision, keeps its object.
- The watch carries worker rows and their status.

For example, the watch may start again while a request is running. Its lease stays on file.

- If its worker is still present, the request still counts toward that worker's in-flight total. The worker keeps its gateway vetoes.
- If the worker has left, the lease ends when the host reports the request's end.
- A worker still present keeps its session pins through the snapshot.

## Worker credentials

A host built with `worker-host` can hand its worker objects a keeper with `Objects::with_credentials`. The keeper has one entry per address an object dials, and each entry is bound to the rows that host's registrations wrote.

For example, two rank rows may name `http://worker:8000` as their engine. The keeper then holds their credential once at `http://worker:8000`, bound to both rows.

### Around a registration

- The host calls `keep` before it writes a registration. It gives the facts it writes, and `None` where the registration has no credential.
- `keep` returns a token.
- The host calls the token's `accepted` with the row the door answered, whether or not the write moved the row. For a registration of several ranks it calls `accepted_rows` with each row the write left standing.
- It calls the token's `put_back` where the write was refused.
- It answers at the door's answer, with no await between the two. A token dropped with no answer is put back.
- Where either answer returns `true`, the host calls `Objects::lay_again` with the same facts. The holder then builds again each object at the address that holds another credential than its row takes now.

One registration is in flight at an address. A `keep` at an address whose last `keep` has no answer yet takes nothing and returns `RegistrationInFlight`. The host waits, or tells its caller to ask again.

### Which object takes a credential

- A worker object takes its address's credential only where a registration of this host bound its row.
- A worker that another replica registered is built here with no key.
- So is a worker that takes an address after its last worker left: it is another row.
- While a registration is in flight, an object built at its address takes what it stated, and the answer settles it.

Registering the same worker again as an upsert, with a new key, moves no row that is not draining: the door answers `MembershipWriteOutcome::Unmoved` for equal facts. The host calls `accepted` all the same. It returns `true`, and `lay_again` builds the worker's objects with the new key.

### What the keeper forgets

The holder forgets by itself.

- A snapshot laid whole drops the rows it finds gone from the keeper, whoever wrote the removal. An entry goes with its last row.
- A snapshot laid whole lays its own rows, then reads the status the table holds for each.
- A write the store refuses leaves the keeper as it was.
- A write call of the holder that the store takes drops from the keeper the row it takes out. A removal a standby applies from its watch leaves the keeper's row for a snapshot or `lay_again`.
- `lay_again` drops each bound row that has no object.

The keeper lives in the host's process and starts empty at restart. The keeper's credentials go into no membership row and no watch. The stock controller builds its worker objects without a keeper.

### The registration line

A host that keeps credentials builds its registration line with `worker_host::registration_into_store_keeping_credentials`.

- The line keeps in the keeper of the holder it is handed (`Objects::with_credentials`) and takes the store its door writes.
- The door must write through the holder's write calls: `Objects::upsert`, `Objects::patch` and `Objects::remove`. So a row's object stands when the door answers, and the line follows no store.
- It returns `worker_host::NoCredentials` for a holder built with no keeper.
- The write keeps what each worker states ahead of its row. It answers the keeper with the door's answer, and has the address laid again where the keeper says so.
- The write runs in the holder's turn. A replacement by id is refused as moved where another object stands under its URL by then.
- That is because a registration of another credential alone can lay another object and move no row.
- `registration_into_store` keeps nothing.
