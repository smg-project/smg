# Worker rows and jobs

The controller keeps one row a worker. This page covers a row's bounds, the door a host writes rows through, what a host reads back and the controller's probes. It also covers the worker jobs of a host that keeps the store in its own process.

[Run the service](service.md#worker-discovery) covers worker discovery, and [Embed the engine](embedding.md#worker-objects) the worker objects a host keeps beside its rows.

## A worker's row

A registration states the row's facts. One that states a fact past its bound below is refused, and a refusal for a bound names the fact.

| Fact | Bound |
|---|---|
| The URL | 2,048 bytes |
| The runtime, the zone, a model id | 256 bytes each |
| The labels | 128 of them, a key of 128 bytes, a value of 4,096 |
| The row as a whole | 16,384 bytes |

- A URL, a runtime and a model id hold no blank and no control character. A model id is not empty.
- No count bounds a row's model ids, and no bound of its own holds an encode offer's connector: the row's size does.
- An encode offer's connector is held to no rule of its words. A control call's is read past the blanks around it, and a blank one states none.
- A control call's row is weighed as the wire frames it.
- A row that a job's step writes at the controller's door is weighed by the store's own measure (`WorkerFacts::weight`). Its numbers weigh 160 bytes, and each text it counts weighs its bytes and 5 more.
- Such a row weighs 22 bytes more where its worker states a fact for a request's hard demands, and 3 more where it takes images. It weighs 11 more where its encode offer states anything. A priority other than zero weighs 7 more, and a drain window 7.
- Three texts are weighed twice: the URL, the runtime under a scrape interval, and the value of a `policy` label. The engine's address is a text of its own beside the URL.
- A provider's row also weighs its maker's name and key.
- That measure is never under the wire's weight, so a row within the bound fits a frame.

## The membership door

A host's registration and removal steps write rows at a membership door, `worker_host::MembershipWriter`.

- The door lays a found worker's facts over the row at its URL, marks a row draining and takes a row away.
- Only a row's maker lays facts over it, save for a refresh of its labels.
- Facts that name another URL than the call's are refused.
- A removal names the row its asker saw, by id and revision. [Removal](#removal) says who may remove whose rows.
- The crate holds one door, the controller's own, for a replica that may lead. It holds none over a store alone.
- `worker_host::facts` reads a worker object back to a row's facts.
- The steps of [Running worker jobs over a local store](#running-worker-jobs-over-a-local-store) call the door. A controller built with the `discovery` feature runs them for the workers its discovery finds.

**A refresh of labels.**

- A health loop's refresh of a worker has an entry of its own, `MembershipWriter::refreshed`.
- The entry asks no maker, and the row keeps the maker it has.
- The refreshed worker may differ in its labels and its pairing descriptor from its row, or from the object `worker_host::object` builds of the row. One that states any other fact is refused, and nothing is written.
- That object holds a part of a second as a whole one, a threshold of zero as one, and a runtime's name as the worker crate reads it. The row keeps its own word of every fact that object holds narrower.
- The pair rule and the encode classes read a row's runtime name as that object does. A name that is not empty and that the worker crate does not know is the exception. The object holds no runtime for it, and the two rules compare it as written.
- A wait wider than a row holds reads as the widest number a row holds. Where the row's object waits past that number, a worker that waits longer still passes as that object.
- The controller's door reads no descriptor of the worker. The row keeps the one it has, so a refresh there moves labels alone.
- Labels that change are bounded as a registration's are. Equal labels with an equal descriptor write nothing.
- A row that is draining takes no refresh.

**What the controller's door also refuses.**

- A fact that its log does not carry to a standby. Those are a pairing descriptor, an empty zone, an encode offer's empty connector, and a zero where the row means none.
- A worker of the host-local ZMQ wire whose row names no gateway, as it refuses a control call's registration of that wire.
- One gateway holds that socket, and such a row would stand in every gateway's pool.

**What a host reads back.**

- A host that keeps worker objects reads a row off their holder, `worker_host::Objects`: each call builds the row again, a copy of its facts.
- `Objects::read_row` reads a row by its id, and `Objects::read_row_under` by its exact registration key.
- `Objects::checkpoint` cuts the rows that the holder and the store both hold, in one turn of the holder. It states the log's place, or the default place for a store that keeps no log. It takes the store that the holder's write calls tell.
- Each read takes the holder's turn, so a writer of a row waits for it and no decision does.
- Code that runs inside that turn calls none of these reads: the read never returns there. The store calls its journal inside the turn of a holder's write of a row, `Journal::released` too, and of its cut (`Journal::head`).
- A row read so carries the models its worker object serves now. A row read alone carries the status that object states, and a row of the cut carries the status the store holds.
- The cut reads the place with the store's word in one hold of the store's lock: the rows that stand, their statuses and the epoch.
- Models that a host sets on worker objects through the registry are read row by row. A cut may hold one object's new list beside another's old one, and a list the log holds no word of yet.

## Probes

The controller probes its workers while it leads.

- Its binary is built with the `controller` feature, which holds the prober. A build without that feature holds no prober.
- A row's health policy sets the check interval, the timeout and the two thresholds.
- The controller counts the interval and the timeout in whole seconds, one at least. It counts each threshold as one at least too.
- Where a pending row is probed, its first probe is due at once. A probe that starts sets the row's next one due an interval later. At most 128 probes run at once, one a row.
- A row that is taken out or failed and passes a probe short of its threshold has its next probe brought forward, to one timeout later where that is sooner.
- A probe that moves a row's status starts both counts in a row again, of passes and of failures.

| Probes | What follows |
|---|---|
| `success_threshold` passed in a row | The row is ready. A failed row comes back the same way. |
| `failure_threshold` failed in a row | A ready row is taken out. |
| Three times as many more failed in a row | That row fails. |
| Ten times `failure_threshold`, for a pending row that never became ready | The row fails. |

- A replica that does not lead moves no row by a probe: its rows keep the status its leader's log tells.
- A row whose policy turns probes off is not probed. A registration that writes such a row sends it a connect signal, which leaves it ready.
- The signal can be lost, as where the call's term ends first or a drain comes ahead of it. An upsert of equal facts sends it again to a row that still waits for it.
- A row that one gateway alone reaches is not probed either, whatever its policy says.
- A probe removes no row.

## Running worker jobs over a local store

A host that keeps the membership store in its own process can use `registration_into_store` and `removal_from_store`, with the `worker-host` feature.

- The steps write rows through the membership door, which the host hands them.
- The door must write through the holder's write calls, `Objects::upsert`, `Objects::patch` and `Objects::remove` among them. The object holder writes the registry with each row.
- The host supplies the job queue and runs the workflows.

### Registration

`write_workers_at_door` waits for creation or the host's last checks. It takes the three job modes. The table and the points below say what this step's write does. A job's earlier steps have refusals of their own.

| Mode | What it does |
|---|---|
| `Upsert` | Registers an absent URL. A standing row under a stated URL is left, changed in place or replaced, as its facts say. Others of the engine leave: see below. |
| `CreateOnly` | Adds rows under free URLs. A row under a stated URL, or any other row of the engine, fails it, whoever made it. One exception stands below. |
| `ReplaceById` | Is laid over the row behind the registry's worker of that id, at the revision the job names. |

- The step lays a job's workers, the ranks of one engine, in one turn of the door. For an `Upsert` or a `CreateOnly` job the door checks each row and the group ahead of the first write, and a refusal there writes nothing.
- A door whose lead can move asks its term ahead of each row's write. A refusal at a row's write, a lead found lost or a row moved by another writer, leaves the rows written before it standing.
- These rules are of the rows the door's object holder has laid, and a row is an engine's where its own facts name that engine. [Embed the engine](embedding.md#worker-objects) says which rows stand apart from the holder.
- A `CreateOnly` job that a standing row fails answers "already exists", unless a row's own check fails first. That row may stand under a URL the job does not state: an add takes no row of its engine away.
- The exception is the write step's own. Standing rows do not refuse its write where each row under a stated URL is the add's maker's, states equal facts and does not drain. The engine must have no other row, and a row must be left to write.
- So a later attempt of the step, under a term that holds, can lay the ranks left unwritten.
- An `Upsert` write takes away the engine's standing rows that the job's workers do not state. Another maker's row among them refuses the write, and nothing is written.
- A `ReplaceById` job takes one built worker. A job that builds several is refused before any is written, as the gateway's own replacement is.
- A `ReplaceById` job fails where that worker left or moved. It also fails while that worker drains: a removal wins over a replacement that lands inside its drain window.
- A replacement that changes labels, checks or thresholds is made in place, and the worker keeps its id.
- One that changes the role, the wire or the models is the URL's next life, and the worker takes a new id.
- `registration_into_store` takes the object holder for that read.
- It is a step of the local kind: it skips a registration of another kind, whose workers the host's own steps write.

**Who writes as whom.**

- API and startup jobs write as control calls.
- A discovery job writes as provider `discovery`, using the spec's address as its key.
- For example, a job for `http://worker:8000` uses that one key even if the job builds two ranks at different registered URLs.
- Health jobs cannot register workers.

### Removal

A removal selects workers and marks their rows with `drain_workers_at_door`. It waits for them to settle, then takes their rows with `remove_workers_at_door`.

- The settle waits the drain window the worker was registered with, one day at most. A longer window ends the removal at the day, with the worker still draining.
- One job waits the longest window among the workers it drains.
- The worker's row carries a window that a registration states, cut at 4,294,967,295 seconds, and the object built from the row takes it.
- A found worker's row carries one where it is not the default. At the default it keeps the number where its standing row states it, and a replacement by id states none.
- Each write checks the row id and revision, and each removal write has one attempt.
- A worker that an `Upsert` job finds again while its removal settles is registered anew. Its row takes the URL's next life, out of the drain, and the pending removal leaves it.

**Who may remove whose rows.**

| Asker | The rows it can remove |
|---|---|
| API and startup jobs | Control rows. |
| Discovery | Rows of provider `discovery`, under any key. |
| Health | Rows within its scope, by the row's maker. |

- Health has three scopes. `Disabled` permits no rows, `DiscoveryOwned` permits provider rows, and `AllLocal` also permits control rows.
- The controller's door includes the asker in its removal audit lines.
- Where the door refuses a removal for its asker and the job drained nothing, the removal fails at the drain. It answers the door's sentence as a rule. Such a row is another maker's, or outside the health scope.
- A host step behind the removal reads in the job's selection the workers the removal took or found gone, and an object no row stands for.
- A step that succeeds drops from that selection a worker it read as left standing. Such a worker moved, stands under an id the removal does not name, or was refused by the door.

### Load and health

- The leader's load monitor takes the load records on a worker's stream, and a standby's drops them, as each monitor reads its replica's lead. A record at a change of lead can go either way.
- The `worker-host` feature has no load sink over the store, only the health checker's.
- `worker_host::HealthAdapter` hands probe outcomes to the worker objects, and tells the store's rows the status each object takes.
- The controller's own service runs the health checker over `HealthAdapter` while it leads.
