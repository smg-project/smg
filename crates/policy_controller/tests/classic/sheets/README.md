# The recorded policy sheets

What the gateway's own routing policy code picks, recorded row by row. The gateway chooses a worker for each request
with a load-balancing policy, and this crate builds those policies a second time. The test target
`tests/classic_rows.rs` replays every row of each sheet that has a module there through the new code, and fails on a
row it decides differently. A sheet with no module yet is read and owed by name. The
reference is the gateway's policy code at the commit a sheet names, run, never read.

One sheet records no pick: `pair_index.json` holds which prefill and decode workers the gateway pairs, the lists a
pair's placement then picks from.

One sheet records one rule under every policy: `all_vetoed.json` holds what the gateway answers when every usable
worker of a pool is over its overload thresholds or doubted by the liveness tracker, and what a routing key gets
under consistent hashing while its own worker is over them or doubted. It also holds the gateway's answer over open
circuit breakers: a pool behind them on every worker, and one worker a soft veto stands on among them.

One sheet records a request that carries encode items: `encode_pair.json` holds the pair the gateway picks for it,
the encoders it offers the request's items and the encoder each item takes, or the name of its refusal.

One sheet records the placement of a pair: `pair_select.json` holds which prefill and which decode the gateway
places a request on, over two pools its pair index has paired.

One sheet records that placement behind the prefill admission alone: `pair_bound.json` holds what the gateway
answers where a prefill that holds the admission's bound has an open partner, and where it has none.

One sheet records that placement on the wire that carries one runtime's request: `pair_runtime.json` holds which
pair the gateway places where the workers of one known runtime stand beside workers of another or of none.

One sheet records no pick and no policy: `overload.json` holds whether the gateway reads one worker as overloaded
after each thing its load monitor is handed about the worker.

One sheet records one rule over the policies that read no routing key themselves: `sticky_override.json` holds
where the gateway's routing-key override places a keyed request, and what it does with a key that has two
requests in flight on its worker.

Two sheets record a rank of one worker and no worker: `minimum_tokens.json` holds the rank the gateway's
minimum-tokens policy gives each request after each thing its rank cache is handed about the worker, and
`sticky_rank.json` holds the rank the gateway pins each request of one grouped worker to by its sticky key.

Four sheets record no policy: each holds one of the gateway's hard demands, what a request must have of the worker
that takes it, kept before any policy runs. `retry_wire.json` holds the runtime and the transport a retry stays on,
`media_refs.json` where a request's forwarded media references and its pixel media may go, `context_window.json`
the two bounds a worker's context window puts on a request, and `constraint_decider.json` the workers a retry may
take once the request's constraint was decided on its first worker. Each also holds the gateway's refusal where a
demand leaves the pool nobody.

```
crates/policy_controller/tests/classic/sheets/
  power_of_two.json        one sheet per policy: its recorded rows
  manual.json
  prefix_hash.json
  pair_index.json
  all_vetoed.json
  encode_pair.json
  pair_select.json
  pair_bound.json
  pair_runtime.json
  overload.json
  bucket.json
  minimum_tokens.json
  sticky_rank.json
  cache_aware.json
  sticky_override.json
  retry_wire.json          one sheet per hard demand of a request
  media_refs.json
  context_window.json
  constraint_decider.json
model_gateway/src/         the gateway's recorders, each a test module beside the code it runs
  policies/classic_rows.rs                                                  power_of_two.json
  policies/classic_manual_rows.rs                                           manual.json
  policies/classic_prefix_hash_rows.rs                                      prefix_hash.json
  policies/classic_all_vetoed_rows.rs                                       all_vetoed.json
  routers/common/pair_index/classic_pair_rows.rs                            pair_index.json
  routers/grpc/common/stages/worker_selection/classic_encode_pair_rows.rs   encode_pair.json
  routers/common/classic_pair_select_rows.rs                                pair_select.json
  routers/common/classic_pair_select_rows/classic_pair_bound_rows.rs        pair_bound.json
  routers/common/classic_pair_select_rows/classic_pair_runtime_rows.rs      pair_runtime.json
  policies/bucket/classic_bucket_rows.rs                                    bucket.json
  routers/grpc/common/stages/request_execution/classic_sticky_rank_rows.rs  sticky_rank.json
  routers/grpc/common/stages/worker_selection/classic_hard_demand_rows.rs   retry_wire.json
  routers/grpc/common/stages/worker_selection/classic_hard_demand_rows.rs   media_refs.json
  routers/grpc/common/stages/worker_selection/classic_hard_demand_rows.rs   context_window.json
  routers/grpc/common/stages/worker_selection/classic_hard_demand_rows.rs   constraint_decider.json
  policies/classic_cache_aware_rows.rs                                      cache_aware.json
  policies/classic_sticky_override_rows.rs                                  sticky_override.json
crates/worker/src/         the worker crate's recorders, test modules beside the load monitor
  load/monitor/classic_overload_rows.rs                                     overload.json
  load/monitor/classic_minimum_tokens_rows.rs                               minimum_tokens.json
```

## A sheet

A sheet is one JSON object: `recorded_from` (the commit its recorder ran in), `policy`, and `rows`. A row is a
pool of workers, what a router knows of each one (the requests it holds on it, and the worker's load report or its
routing keys), one request shape and what the policy picked for it. There are two kinds of row:

- **sequence row:** a list of steps (a pick, or new facts about every worker) and the worker each pick chose, in
  order. The recorder runs the row twenty times, each time from a fresh policy, and stops if the runs disagree, so a
  sequence row is the same in every recording of one commit.
- **drawn row:** the policy draws, so the row records how often each worker was chosen over 6,000 single decisions,
  each from a fresh policy. The draws are the gateway's own and cannot be seeded, so a drawn row is one sample: a
  second recording of the same commit gives other counts within the same spread. Compare shares, never counts.
  The replay takes two rows as one distribution when a worker never picked on one side is never picked on the
  other and every other share lies within four standard deviations of the difference of two samples of that size.

The rows of `all_vetoed.json` are sequence rows and drawn rows of one shape. A row names the policy its pool runs
under and whether shedding is on. Each worker has the requests the router holds on it and a state: `free` (ready,
under its thresholds), `over` (ready, over them), `stalled` (ready, doubted by the liveness tracker),
`stalled_over` (ready, doubted and over them), `breaker` (ready, under its thresholds, behind an open circuit
breaker) or `down` (not ready). A step is a pick (with the routing key the request carries, if any, and the workers
an earlier attempt tried), new loads for every worker, or new states. A pick is recorded as the worker's place in
the row's list. A request placed nowhere is a number past the last worker: the count of the workers when the gateway
sheds it as an overload shed, the count plus one when it finds no worker to use.

A ring row carries `ring_key`: its request is placed by that routing key on the hash ring. Two rings put one key on
different workers, so a ring row lists its workers in the key's ring order: the worker the key lands on first, then
the one it passes to when that worker is down, and so on. The recorder finds that order on the gateway's ring by
placing the key with every worker free and then with the members found so far down, and a reader finds it the same
way on its own ring. A ring row's loads, states, tried lists and answers all count in that order, so an answer of 0
is the key's own worker and 1 the next member of the ring, on whatever ring replays the row. A row under consistent
hashing with no `ring_key` records what a request without a routing key gets.

The rows of `bucket.json` are sequence rows and one drawn row. A row names the policy's configuration in the
gateway's own spelling: the two parts of its balance rule and the interval of its adjustment in seconds. Its pool is
prefill workers the policy lists (`prefill`), or regular workers nobody listed (`regular`), which the policy draws
from. A step is a pick, which workers are ready from here on (`ready`), a prefill worker that registers (`join`: it
takes the next number, and the policy is handed every listed worker again, as the registration does) or a worker
that leaves (`leave`). A request carries the row's clock at its placement in milliseconds (`at_ms`) and its size as
the policy reads it: the count of its token ids (`tokens`), else the characters of its text (`text`), else zero,
with the workers an earlier attempt `tried`. An answer is the worker's number: its place in the row's first listing,
or the number it joined under. The policy reads two clocks, and the recorder stands for both. The window dates each
booking by the system clock, so before a step the recorder moves every queued booking back by the time the row's
clock advanced. The adjustment runs on a thread once an interval, and the recorder runs the same adjustment at every
whole interval of the row's clock, counted from the row's first request, before the request that reaches or passes
it. No row places a request where a booking's age stands within a second under the window's length, so the system
clock's own microseconds between two steps decide nothing.

The rows of `cache_aware.json` are sequence rows and two drawn rows. A row names the matcher its requests reach
(`matcher`): the tree of text (`text`), the tree of token ids (`tokens`) or the index of the blocks the workers
report (`events`). It names the policy's configuration in the gateway's own spelling (`config`): the match
threshold, the two parts of the count gate, the block size and the two KV gates. Each worker has the requests the
router holds on it, a state (`free`, `over` or `down`, as on the all-vetoed sheet) and its load report where it has
one, as on the power-of-two sheet. A step is a pick (its `prompt`, and the workers an earlier attempt `tried`), the
end of an earlier pick's request (`complete`, by the pick's number among the row's picks), new loads, new states,
new reports, or, under the index, the blocks a worker stored (`stored`) or removed (`removed`: the blocks of a
prompt past its first `kept_blocks`). A prompt is written as runs of token ids, as on the prefix hash sheet; under
the text matcher it is carried as text, each id as four hex digits. Every prompt and every shared prefix is whole
blocks: sixteen token ids, or sixteen characters of a text. A picked worker holds one more request. A tree is
written at every pick: the picked worker holds the prompt from then on. The index is written by the `stored` and
`removed` steps alone, and a row under it stores a block before its first pick, since over an empty index the
policy reads its tree of token ids. The replay hands the engine what each worker holds as the affinity column a
host lends: under the index every worker's own depth, and under a tree the depth of the deepest holders alone,
since the gateway's tree names nobody else. A row under the index is replayed by the replay's own count of
the blocks each worker stored. The replay builds a row's profile from the crate's `cache_aware` preset.
It sets on the preset each number of the row's configuration that differs from the gateway's command line
default, and a tree row names its source. So a row recorded under those defaults asks the preset as it is named.
The sheet records none of these yet: the hash index, a cache namespace, a pool over 32 workers, the trees'
eviction, a prompt that ends inside a block, a worker that leaves, and the warm-up slice with its diversion (every
worker of a row is as young as the row, and no index holds a thousand blocks).

The rows of `encode_pair.json` are sequence rows of their own shape. A row is a fleet: its encoders, its prefill
workers and its decode workers, each named by its place in its list, with what the gateway reads of it from its
registration and the requests the router holds on it (`in_flight`). An encoder has its `runtime`, the `handoff` its
runtime hands an encoded item to a prefill worker in (`push`, `rendezvous`, or none), whether it states the
producer role of a `connector` (`ec_producer`), whether it takes `images`, whether it processes `references` itself
and whether it is ready (`open`). A prefill worker has its consumer role and connector and its `partners`, the
decode workers the gateway pairs it with. A decode worker has whether its registration says it takes images; the
gateway holds the decode workers of one runtime to that and no other's. A row names the policy each side of its
pair is picked by and the policy an item's encoder is picked by. No row draws: under `power_of_two:least_requests`
a row leaves the policy at most two workers, which hold different numbers of requests, and it takes the one that
holds fewer. A request has the key of each of its encode items and what it asks: whether the items are references,
whether they carry images, whether the decode side must take images, and whether each item is encoded on its own
(`fanout`). The answer to a request is `placed` (the prefill and the decode worker, the encoders the items were
`offered`, the `keys` the items were placed by and the encoder each key took) or `refused` (the refusal's `name`; a
refused placement of the pair adds the `role` it names and `why`). Without `fanout` a pushing runtime places a
request's items as one key, a hash of them, so such an answer has one key however many items the request has. Under
`consistent_hashing` a key's encoder is the owner the gateway's ring gives it among the encoders offered; another
ring gives the same key another of them.

The rows of `pair_select.json` are sequence rows and drawn rows whose pick is a pair. A row names the policy each
pool runs under (`power_of_two` carries its load metric after a colon), whether shedding is on and, when the row
runs behind the prefill admission, the most requests that admission lets one prefill hold (`prefill_bound`). Each
worker has the requests the router holds on it, a state as in the all-vetoed sheet and the KV transport its
registration names (`a`, `b` or none): two workers pair when their transports are equal or one of them names none.
A decode may report a running window, with the rooms of it the decode admission has claimed (`rooms`). A row places
its requests one after another, and no request changes what the router holds. An answer is the pair, the prefill's
place in its list and then the decode's, or no pair with the role the gateway named and why: `overloaded` (an
overload shed), `unavailable`, `at_capacity` (the chosen prefill holds its bound), `no_candidates` (the role has no
worker at all) or `declined`, which no row answers. A drawn row counts how often each pair took its one request; a
pair never placed is not listed.

Two choices keep those rows replayable. No row runs under `round_robin`: the gateway starts the rotation of a
candidate set it has not seen at a shared count, so which pick comes first is not the policy's rule. Where a row
must replay pick for pick, a pool runs under `power_of_two:least_requests`, which takes the fewest requests in
flight with no draw and no state. Where a row must show whether a side's policy was asked at all, that side runs
under `random`: where the gateway places by its own rule, one pair takes every draw.

The rows of `pair_bound.json` are sequence rows of that same shape, each behind the prefill admission
(`prefill_bound`) and under `power_of_two:least_requests` on both sides, so nothing in them draws. The gateway asks
the bound of the prefills that still have a partner to go to, and of no other. A row holds one state of that rule:
no ready prefill with an open partner, where the gateway answers no pair ahead of the bound; a prefill at its bound
with the one open partner beside a prefill with room and none, where it answers `at_capacity` and the route waits;
a decode pool over its thresholds or doubted on every worker, with shedding on and off; and a prefill with room and
an open partner, which takes the request.

The rows of `pair_runtime.json` are sequence rows of that same shape too, each placed on the wire that carries one
runtime's request (`wire`, `grpc`) and under `power_of_two:least_requests` on both sides, so nothing in them draws.
A worker that is not on the sheet's one runtime names its own (`runtime`): `other`, a second known runtime, or
`unknown`. The gateway's pair index keeps two known runtimes apart on that wire and pairs a worker of no known
runtime with any, and the gateway places a pair on one known runtime ahead of every other. A row holds one state of
that order: the pair on the known runtime past two workers that hold fewer requests; its prefill at the prefill
admission's bound, where the route waits; that prefill not ready or over its thresholds, and its decode not ready,
where every pair stands; a decode pool over its thresholds on every worker, with shedding on and off; a prefill pool
over its thresholds on every worker, which the least-loaded default pairs with any partner; two known runtimes,
which pair apart; and a prefill of a known runtime with no partner on it.

The rows of `prefix_hash.json` are sequence rows and two kinds of counted row. A row names the policy's
configuration in the gateway's own spelling: how many tokens of a prompt it hashes, the two margins of its load
bound, and the boundaries a prompt is keyed at (`cache_boundaries`) where it names any. A prompt is written as runs
of consecutive token ids, `[first, count]` each. A request carries one as `tokens`, or as `text` (each id spelled as
four hex digits, so the policy's four characters a token read the same span), a routing `key`, or nothing to hash,
with the workers an earlier attempt `tried`. A step is a pick, new loads for every worker (`loads`) or which workers
are ready (`ready`). A ring row names a request (`ring`) and lists its workers in that request's ring order, found
as a ring key's order is: the worker the request lands on with every worker ready and idle, then with the members
found so far down. Under boundaries a prompt has a key at each one it reaches, and a level row (`levels`) says how
its request's two keys stand to each other: beside the request's own ring order, which is its deepest boundary's,
the row lists the same workers in the ring order of the request's head at the shallower boundary. Two rings agree
on no pair of keys, so a replay places a prompt of its own that stands so, the first among the prompts of 300 ids
from `3000000 + 1000 * i` on. A pairs row (`prompts` with `differ_at`) places pairs of prompts on an idle pool and
counts the pairs whose two prompts took one worker, then the pairs that took two: prompt `i` of the row is `tokens`
ids from `1000000 + 1000 * i` on, and its twin holds `2000000000 + i` at the one place `differ_at`. Where the row's
prompts are `text`, each is carried as text: prompt `i` is ids from `1000 * i` on, and its twin holds `65000 + i`.
A spread row (`prompts` without `differ_at`) places `count` such prompts once each and counts what each worker
took, the largest count first, since two rings give one worker different shares. Both are written as counts over
`draws` and compared as a drawn row's shares are. Nothing in them draws: the recorder runs each twice and stops if
the runs differ. No row ties the fewest requests: between equal loads the gateway takes the first worker of its
pool, which is the pool's order and no rule of the policy.

`pair_index.json` holds a third kind, which picks nothing:

- **pair row:** a prefill pool and a decode pool, with what each worker offers a partner as the gateway derives it
  from the worker's registration (its runtime, an operator's protocol, the KV transport, the engine version, the KV
  layout facts) and the pairing key the gateway names the worker's group by. For each of the gateway's two pair
  wires and each of its three pairing modes the row records the verdict on every prefill and decode (the component
  the pair is refused on, or none), each prefill's partners by index in the decode pool, and the refusal the gateway
  reports when no pair is left. Nothing in a row draws: the recorder builds each row twice and stops if the two
  builds differ, so a pair row is the same in every recording of one commit and replays whole or not at all.

`overload.json` holds a fourth kind, which picks nothing either:

- **verdict row:** one worker with its data-parallel ranks and two pairs of overload thresholds: the pair the
  gateway holds for every worker (`gateway`) and the pair the worker names for itself (`worker`). A pair is the
  queued requests summed over the worker's ranks and the mean KV usage over them, each at or above which the worker
  is vetoed; `null` leaves a signal off, and a value the worker names stands over the gateway's. A step is what the
  gateway's load monitor is handed: a poll's answer (`poll`: an entry a rank in rank order, `null` where the answer
  omits the rank), a poll that brought no report (`no_report`), a load record one rank pushed on its event stream
  (`record`), or the worker leaving the ready set (`evict`). An entry and a record carry the rank's queued requests
  (`waiting`), its KV usage (`usage`) and, where the rank reports one, the share of that usage its running requests
  hold (`active`), which the verdict then reads in the usage's place. The row records whether the worker reads as
  overloaded after each step. The monitor judges a poll by its answer alone, and an answer with no entry as no
  report. It judges a pushed record by the worker's pushed report: the ranks that have pushed, each at its last
  record or at a later poll's figures, less the ranks an answer with no entry took off it. Nothing in a row draws:
  the recorder runs each row twice, each time from a fresh monitor and a fresh worker, and stops if the two runs
  differ.

`minimum_tokens.json` and `sticky_rank.json` hold a fifth kind, whose picks are ranks of one worker:

- **rank row of `minimum_tokens.json`:** one worker with its data-parallel ranks (`ranks`), which stands for every
  rank or for the one rank `pinned` names. A step is what the gateway's rank cache is handed: a poll's answer
  (`poll`: an entry a rank in rank order, `null` where the answer omits the rank), a poll that failed (`failed`),
  the worker's loads dropped as it leaves the ready set (`evict`), or a request of `cost` estimated tokens
  (`pick`), which asks the cache's select for a rank and books its cost on the rank it takes. The gateway's
  policy forwards a request to that select and adds nothing. An entry carries the tokens
  the rank's cache uses (`used`), which is the count the gateway reads, the rank's token capacity (`capacity`)
  and, where the rank reports them, the context tokens of the requests it holds (`resident`). The row records the
  rank each request took, `null` where it found none. Nothing in a row draws: the recorder runs each row twice,
  each time from a fresh monitor and a fresh worker, and stops if the two runs differ.
- **rank row of `sticky_rank.json`:** one grouped worker, several engines behind one connection: the engines the
  worker names (`ranks`, `null` where it names no count) and the ranks its connection places a pin on (`pinnable`,
  `null` where the connection does not say). A step is a request with its sticky key (`pick`: `key`, `null` for a
  request with none). The row records the rank the gateway pins each request to under hash placement, `null`
  where it sets none and the connection places the request on any engine. A counted row asks `keys.count` keys,
  `key-0` upward: a pairs row (`pairs` true) asks each twice and counts the keys whose two requests took one rank,
  then the others, and a spread row counts what each rank took, the largest count first. Nothing in a row draws:
  a key's rank is its hash, and the recorder runs each row twice and stops if the two runs differ.
- **demand row of the four demand sheets:** a pool of regular workers under `round_robin`, whether a pool over on
  every worker is shed (`shed`), and a list of requests. A worker has its `state` (`free`, `over` or `down`), its
  `runtime` and its `transport` (`grpc` or `zmq`). Each sheet adds the
  registration's own words for its demand and, beside them, what the gateway reads of those words. On
  `media_refs.json`: the `mm_processor` label and a `supports_vision` label of `false`, then
  `processes_references`, `gateway_pipeline` and `vision`. On `context_window.json`: the `window` the model card
  states, absent where it states none. On `constraint_decider.json`: the three decoding labels, then `policy`,
  `whole_turn` and `vocabulary`. A request states its demands, each absent where it has none: the `wire` of a
  retry (a runtime and a transport), `references` with the pipeline pin of its plan (`any`, `only` the gateway's
  own pipeline, `never` it), `pixels`, `input_tokens` with `scoring_position` and `output_tokens`, and
  `decided_on`, the place of the worker the first attempt's constraint was decided on. The recorder places a
  request sixteen times the pool's size,
  so the rotation walks every worker the demands leave it. An answer is `placed`, the places of those workers in
  the row's list, or `refused`: the response's `status`, its `code`, whether it carries `retry_after`, and its
  `message`. Nothing draws: the recorder runs each row twenty times, each time from a fresh pool and a fresh
  policy, and stops if two runs differ.

The rows of `sticky_override.json` are sequence rows and four drawn rows in the manual sheet's shape, taken through
the routing-key override: the policy registry's own selection with the override on. A row names the policy under
the override (`policy`) and the sticky map's assignment mode (`assignment`). Each worker has the requests the router
holds on it and the routing keys with a request in flight on it. A step is a pick (the request's routing key, if it
carries one, and its `model`), a worker that stops being routable (`down`) or is routable again (`up`), new loads
for every worker (`loads`), or the requests of one key in flight on every worker from here on (`held`): the count
the override's cap reads. No row states the cap: it is the gateway's own, two requests of the key on its worker,
and the rows at one request and at two hold it. No pick moves a count: the counts move by the row's steps alone.
Under `delegate` a sequence row runs power of two over two workers with no load report, so the policy compares
both by their requests in flight and draws nothing.

## The sheets

Each sheet is recorded on its own, so the commit is a column of the sheet's row and never one commit for the
directory. The sha256 is of the file as committed.

| sheet | policy | recorded from (commit) | rows | sha256 | recorder |
|---|---|---|---|---|---|
| `power_of_two.json` | power of two, under each of its four load metrics | `1b141d03b5ca9097f54f0ebc4b95d72e810a62c3` | 44: 24 sequence, 20 drawn | `602e1fe790cc5368ad05de2297b6e482b7a8893704ee448699ec467df9e0d87c` | `classic_rows.rs` |
| `manual.json` | manual, under each of its assignment modes | `1b141d03b5ca9097f54f0ebc4b95d72e810a62c3` | 14: 9 sequence, 5 drawn | `d34189025a50a7a31f925c632ce1964a09dac0baab89370e6e87c5c842db9a5a` | `classic_manual_rows.rs` |
| `prefix_hash.json` | prefix hash: a key from a prompt's prefix, from a routing key and from nothing; the load bound and where a passed key goes; how deep the key reads; the boundaries a prompt is keyed at and the hand-down between their workers | `d92647556e91a3601ef647f12ae5784df2080dc3` | 47: 31 sequence, 16 drawn | `7f64bde829f86aed7615d46a82332eaa4cd55b0cdbf1fff59ef1163b59679296` | `classic_prefix_hash_rows.rs` |
| `all_vetoed.json` | every policy, over a pool whose every usable worker is over its overload thresholds or doubted by the liveness tracker; a routing key under consistent hashing whose own worker is over them or doubted; a pool behind open circuit breakers | `342e02185362a5f7fc3b1c2faff76e15233b3ae2` | 104: 98 sequence, 6 drawn | `c0d252e295e893e49490172a4a8db218d3800c9c4e684481653261670e015e9f` | `classic_all_vetoed_rows.rs` |
| `pair_index.json` | the prefill and decode pair index, on both pair wires under each pairing mode | `64ff5a437b5007af5079f0492b46f4e5a712b66f` | 23 pair rows | `383783a8a28bdc3d0753111fbb84df4be2713fe332b448938b9cb75f44dec694` | `classic_pair_rows.rs` |
| `encode_pair.json` | a request with encode items: the pair, the encoders offered and each item's encoder | `4ba2a6431357be5411a2383cdc4916b41471d722` | 52: 52 sequence, 0 drawn | `988bf253da9310aeedd08ec2704d8e01ecc4a1210453353723ba0faa9001de33` | `classic_encode_pair_rows.rs` |
| `pair_select.json` | the placement of a prefill and decode pair: each side under its policy, a side over its overload thresholds or doubted whole, a decode's rooms, the prefill admission's bound, the least-loaded default's own rules, a role with no worker | `4ba2a6431357be5411a2383cdc4916b41471d722` | 40: 23 sequence, 17 drawn | `bd8426316ae7337ede82b5a931c1d00cd9dee1aca22c528278f32b2671846eb8` | `classic_pair_select_rows.rs` |
| `pair_bound.json` | the placement of a pair behind the prefill admission's bound: every prefill at its bound with no decode ready, the prefill at its bound with the only open partner, a decode pool over its thresholds or doubted on every worker, with shedding on and off, a prefill with room and an open partner | `7526c6664e4916620fad715f204d35968568e3ee` | 8: 8 sequence, 0 drawn | `a0af115d810052da2c48dc44fdb4bb5961a6c8e24dfb189f5cbdf0714d7ed426` | `classic_pair_bound_rows.rs` |
| `pair_runtime.json` | the placement of a pair on the wire that carries one runtime's request: a pair on one known runtime beside workers of none, its prefill at the admission's bound, not ready or over its thresholds, its decode not ready, a decode pool and a prefill pool over their thresholds on every worker, two known runtimes, a prefill with no partner on its runtime | `adf621bcfd7c1a7921435a008a58fc297eff6747` | 10: 10 sequence, 0 drawn | `4519479db78be320591b87ff32e7bf8b161c3a23aae6b1d17155641ea0b42e9b` | `classic_pair_runtime_rows.rs` |
| `overload.json` | the overload verdict of one worker: the gateway's thresholds and the worker's own, a poll's answer, a poll with no report, a pushed record, an eviction | `436f0c8c5725b0a408b990003d76eaddc72c4be4` | 25 verdict rows | `29c10ee8876e53d23a824a3b7bd0481efd118e8ac5a0d2822477729f6e21283a` | `classic_overload_rows.rs` |
| `bucket.json` | bucket: a size's range and its worker, the hand-down when that worker is not offered, the two parts of the balance rule, the window a booking leaves, the adjustment's cut, a worker that joins or leaves, a pool nobody listed | `d92647556e91a3601ef647f12ae5784df2080dc3` | 19: 18 sequence, 1 drawn | `71a533017809566486335bb9f212ea49712a833bbe5ebcbde7c5d6319ee764e4` | `classic_bucket_rows.rs` |
| `minimum_tokens.json` | the minimum-tokens rank of one worker: a poll's answer, a rank it omits, an answer with no token capacity or no entry, a failed poll, an eviction, a worker that stands for one rank, used tokens beside context tokens | `615542c4c69bb28fb3cb3e9e4ee29a8e2be0a7bc` | 18: 18 sequence, 0 drawn | `1ab41e023fa4f9b507860655a3d917aee68686479d09eaa3b1e9cf8e71d56139` | `classic_minimum_tokens_rows.rs` |
| `sticky_rank.json` | the rank of a sticky key on one grouped worker under hash placement: a key's rank, a request with no key, a worker of one engine or of no count, the ranks a connection places a pin on, how the keys keep and spread | `615542c4c69bb28fb3cb3e9e4ee29a8e2be0a7bc` | 8: 6 sequence, 2 drawn | `d8663087d143cb6d8f46777b2abf6b38e16153fb738c33240c8c7473c86805af` | `classic_sticky_rank_rows.rs` |
| `retry_wire.json` | the wire of a retry: the runtime and the transport its retained plan was built for, a wire whose workers are down or over their thresholds, a wire no worker is on | `a4935a5acd15151641b1988e234016a95fd172da` | 8 demand rows | `5ab9e99211e10b2fd8e0d3fd8400fe207cfc9adc8e7b1603dd13c70177d7128c` | `classic_hard_demand_rows.rs` |
| `media_refs.json` | media by reference and pixel media: the workers that process references, the pipeline a plan pins, the vision encoder, and each refusal where a demand leaves nobody | `a4935a5acd15151641b1988e234016a95fd172da` | 10 demand rows | `5c624c90b8b3195fa18b73390fb07e822b94876134c088aaccf359da6bdc7990` | `classic_hard_demand_rows.rs` |
| `context_window.json` | the context window: the input and the completion budget each against the window, a scoring position, an unknown window, the widest window a refusal names | `a4935a5acd15151641b1988e234016a95fd172da` | 8 demand rows | `bf49bf479afab2621fa118ea07c35ff21cbbd0b06dcb499fa1d1318bd60e0f62` | `classic_hard_demand_rows.rs` |
| `constraint_decider.json` | the constraint's decider on a retry: the workers that decide a constraint as the first attempt's worker did, and the pool's answer where none is left | `a4935a5acd15151641b1988e234016a95fd172da` | 5 demand rows | `5008922d28db1276f4a202de44b24935b8f10fe8d3581a0e9483d597358493c0` | `classic_hard_demand_rows.rs` |
| `cache_aware.json` | cache aware in single placement, under each of its three matchers: a holder kept, the count gate and its spill, its factor under the command line's defaults, a margin under its floor, a miss, a match at the threshold, a holder tried or not usable, the two KV gates, a request that ends, the blocks a worker stores and removes | `bfe189b332349c44cf7e3b6e1102a9b5a2699d9d` | 28: 26 sequence, 2 drawn | `f13a642dc9252e0328f8092b3ff322a03f270d918311c3b5c5dd29979ab8f085` | `classic_cache_aware_rows.rs` |
| `sticky_override.json` | the routing-key override over a policy that reads no key: a pin under the cap and under its model, a key seen first under the map's modes and under `delegate`, each pin order after a failover, and at the cap a pin that follows, a pick on the pinned worker again, a pick on a worker at the cap, every worker at the cap, no other worker routable | `26f90083091e4dd791420febd88beda3bef3faac` | 18: 14 sequence, 4 drawn | `d74b5e3c8a31b4f1c8cd55cef5629acad0659658c830d937f94b056e9580562b` | `classic_sticky_override_rows.rs` |

## Recording

A recorder is an ignored test of the crate that holds the code it runs: a test module beside that code, so it reads
the code by the code's own functions. Run one with the output directory and the commit it runs at, from the
repository's root:

```bash
CLASSIC_ROWS_OUT=$PWD/crates/policy_controller/tests/classic/sheets CLASSIC_ROWS_COMMIT=<commit> \
  cargo test -p smg --lib policies::classic_rows -- --ignored --nocapture
```

The overload and minimum-tokens sheets' recorders are tests of the worker crate: their commands name
`-p smg-worker --all-features` where the others name `-p smg`.

| sheet | test path |
|---|---|
| `power_of_two.json` | `policies::classic_rows` |
| `manual.json` | `policies::classic_manual_rows` |
| `prefix_hash.json` | `policies::classic_prefix_hash_rows` |
| `all_vetoed.json` | `policies::classic_all_vetoed_rows` |
| `pair_index.json` | `routers::common::pair_index::classic_pair_rows` |
| `encode_pair.json` | `worker_selection::classic_encode_pair_rows` |
| `pair_select.json` | `routers::common::classic_pair_select_rows` |
| `pair_bound.json` | `classic_pair_select_rows::classic_pair_bound_rows` |
| `pair_runtime.json` | `classic_pair_select_rows::classic_pair_runtime_rows` |
| `overload.json` | `load::monitor::classic_overload_rows` |
| `bucket.json` | `policies::bucket::classic_bucket_rows` |
| `minimum_tokens.json` | `load::monitor::classic_minimum_tokens_rows` |
| `sticky_rank.json` | `request_execution::classic_sticky_rank_rows` |
| `retry_wire.json` | `worker_selection::classic_hard_demand_rows` |
| `media_refs.json` | `worker_selection::classic_hard_demand_rows` |
| `context_window.json` | `worker_selection::classic_hard_demand_rows` |
| `constraint_decider.json` | `worker_selection::classic_hard_demand_rows` |
| `cache_aware.json` | `policies::classic_cache_aware_rows` |
| `sticky_override.json` | `policies::classic_sticky_override_rows` |

The pair sheet's recorder is a child of the pair index and reads a pairing descriptor by its readers, so it records the
descriptor the gateway derived and never a second reading of the registration. The encode sheet's recorder is a
child of the gateway's worker selection stage: it runs the stage's own selection, and it wraps the encode policy to
write down each call (the encoders offered and the key) and passes every call on unchanged. The pair placement
sheet's recorder sits beside the placement it calls and builds its pools, policies and pair index with the gateway's
own types. The bound sheet's recorder and the runtime sheet's are children of that recorder and build their rows on
the same bed; the pair placement sheet's filter runs them too, and each writes its own sheet, whose rows draw
nothing. A runtime row's bed dials its workers on the wire the row names and gives each the runtime it names. The
overload sheet's recorder is a child of the load monitor's module, so each step enters the monitor
through the monitor's own functions and the verdict is read off the worker the monitor wrote it on. The bucket
sheet's recorder is a child of the policy's own module and stands in that policy's directory: the adjustment and
the window are the policy's private state, and a module beside the policy's could reach neither.
The minimum-tokens sheet's recorder is a child of the load monitor too: each step reaches the rank cache by the
recorder's own repeat of the hand-over the monitor's poll loop makes, and each request asks the cache's select,
which the policy forwards to. The sticky rank sheet's recorder is a child of the stage that sends a request to its
worker, and calls the stage's own pin. The four demand sheets of the gRPC pipeline share one recorder, a child of the
worker selection stage: it calls the two functions a first attempt and a retry both call for a single worker, the
selection and the refusal where it names nobody, and writes one sheet a demand.

A sheet is never edited by hand. A new recording replaces the whole sheet and its row in the table above, in one
pull request that names the commit and says which rows moved: a sequence row moves only when the policy's
behavior did, and the drawn rows are a new sample.

## Reading

The crate's replay reads this directory, every run of its suite with the `in-process` and `worker-host`
features:

```bash
cargo test -p smg-policy-controller --features in-process,worker-host --test classic_rows
```

A few rows of each sheet a module replays are also copied into the crate's test source, and the replay holds each
copy to its sheet.

`retry_wire.json`, `media_refs.json`, `context_window.json`, `sticky_rank.json` and
`encode_pair.json` replay over worker objects, which the `in-process` feature brings. The encode sheet's fleets are
decided by the engine's entry for an encode request over three lists of worker objects.

`overload.json` replays through the worker crate's load monitor, built as a host of the engine over worker lists
builds it, which the `worker-host` feature brings.

`minimum_tokens.json` replays through the rank cache of that load monitor, behind the same feature.

## Limits left

A limit is a recorded row the replay does not decide as recorded, with the reason. The replay names each one and
fails once the reason no longer holds.

- `power_of_two.json`: none. All 44 rows replay.
- `all_vetoed.json`: none. All 104 rows replay.
- `pair_index.json`: none. All 23 rows replay.
- `encode_pair.json`: all 52 rows replay. In the nine that place items by consistent hashing a key's encoder is
  not held to the recorded one: two rings place no key alike. The replay holds each key to the encoders it was
  offered and equal keys to one encoder.
- `pair_select.json`: one row is decided apart, named by the replay with the answer the pair's decision gives. It
  is a refusal's reason: where the only ready prefill's partners are all over their thresholds beside a free decode
  of another transport, the gateway answers unavailable and the pair's decision names the overload veto. The other
  39 rows replay, the four of a prefill policy that weighs load against cache affinity itself (`cache_aware`)
  among them: the gateway lifts the overload veto for such a policy and lets it place, and a prefill profile that
  reads affinity runs with that veto lifted under a binding that steers.
- `pair_bound.json`: none. All 8 rows replay.
- `pair_runtime.json`: none. All 10 rows replay.
- `overload.json`: none where the worker is one object for all its ranks, as the recording gateway held it. All 25
  rows replay there, verdict for verdict, through the worker crate's load monitor. The replay builds the holder of
  worker objects with the calls a controller builds its own with, and writes the row's worker into the store the
  holder follows. It hands the monitor each step: a poll through the monitor's poll of the worker's group, a pushed
  record through the record's own entry, and an eviction as the worker's row leaving the ready set. It reads the
  verdict off the worker object, where the engine reads it: the engine judges no load.
  Eleven of the 14 rows of several ranks answer apart where each rank is a worker object of its own, as a
  registration of several ranks lays them. The replay runs those 14 rows a second time in that layout and names
  the eleven, each with the verdict on each rank's object. One rule holds all eleven: there the monitor judges a
  rank's object by its own rank's figures, not by the queue summed and the usage averaged over the engine's ranks.
  Two ranks that queue 4 requests each under a threshold of 8 are vetoed as one worker and not as two
  (`the-queue-vetoes-at-its-count-and-above-summed-over-two-ranks`). Under a usage threshold of 0.5 a rank at 0.5
  is vetoed alone where the mean over both ranks is 0.375 and the one worker is not vetoed
  (`the-usage-vetoes-at-its-mean-and-above-over-two-ranks`). A rank's object states the address, the engine's
  address, the rank and the width that the worker crate's builder gives a rank worker, which a gateway that
  registers each data-parallel rank as a worker builds its workers with.
- `manual.json`: all 14 rows replay. One row asks a pair a role at a time,
  `manual-one-key-pins-apart-under-the-prefill-and-the-decode-role`, and the crate decides a pair once: the replay
  joins a prefill step and the decode step behind it into one decision, whose prefill side reads the pool as it
  stood at the prefill step.
- `prefix_hash.json`: two rows are decided apart, each by the key this crate reads, and the replay names both with
  the outcome the crate gives. The gateway hashes a prompt's own tokens. A key here is the hash of a request's last
  whole block of the view's block size, made by the host that feeds the request. So the tokens past that block are
  outside the key (`the-last-token-of-a-prompt-under-256-tokens-is-in-the-key`), and a request under one block has
  no key and is placed as one with nothing to hash, on the fewest requests
  (`a-ten-token-prompt-keeps-its-key-s-worker-over-the-fewest-requests`). The other 45 rows replay.
- `bucket.json`: none. All 19 rows replay.
- `cache_aware.json`: none. All 28 rows replay.
- `sticky_override.json`: none. All 18 rows replay.
- `minimum_tokens.json`: none through the load monitor's rank cache. All 18 rows replay there, rank for rank.
  A row records the rank the load monitor's rank cache gives a request
  on a worker that stands for several ranks: an answer of the host's own code, asked after the pick. The crate's
  store listed a row a rank and decided these rows by rules of its own, eight as recorded and ten apart, until the
  replay stopped reading that store.
  The replay builds the monitor as the overload sheet's replay does and hands it each poll, failed poll and
  eviction. It asks the monitor's rank cache for each request's rank, with the worker as one object for all its
  ranks, or as the object of its one rank in the three rows that say so.
  It does not replay the engine's pick among rank objects, and these rows are not that pick's answers. The engine
  serves no rank option: `refuse_unsupported` in `src/document/routing_settings.rs` refuses the gateway's, and
  where each rank is a worker object the engine picks among them by its profile.
  Three of the ten rows answer apart under the engine's pick among rank objects,
  by rules of its scorer and picker. There a rank is scored by the
  context tokens of its requests, not by the tokens its cache uses
  (`the-used-tokens-decide-where-a-rank-holds-fewer-context-tokens`,
  `a-rank-that-reports-no-context-tokens-is-a-rank-by-its-used-tokens`), and equal ranks are drawn, where the
  gateway takes the lowest (`equal-ranks-go-to-the-lowest-rank`).
- `sticky_rank.json`: four rows are decided apart. The gateway sets no rank on a request with no sticky key, on a
  worker of one engine or of no count it knows, and on a key whose hash rank its connection places no pin on: the
  connection then places the request on any engine. The replay lists a worker object a rank, and the engine over
  worker lists gives every request one of them, a
  key's owner on the ring of the ranks its host lists and the picker's draw for a request with no key
  (`a-request-with-no-key-takes-no-rank`, `a-worker-of-one-engine-takes-no-rank`,
  `a-worker-that-names-no-count-takes-no-rank-where-its-connection-says-none`,
  `a-key-whose-rank-the-connection-places-no-pin-on-takes-none`). Two hashes place no key alike, so a key's rank
  here is not its rank there: a row of steps replays when the requests of one key share one rank and a request
  with no rank there has none here. The other four rows replay so, the pairs row count for count and the spread
  row share for share.
- `retry_wire.json`: every row replays. A worker states its runtime as a number of its registering host's, and
  a retry demands the number of its plan's runtime, so the replay numbers a row's runtimes itself, one a
  spelling on each transport: a retry stays on its wire by that number alone, and its pool is one list
  whatever transports its workers dial.
  A refusal replays by its `status` and its `code`. The crate sheds under a reason and a stage's
  name, and the replay keeps the gateway's words for each: a model no worker serves and a wire no worker is on
  are no model, a pool over its thresholds on every worker of the wire is overloaded where shedding is on, and
  any other pool left nobody is unavailable. The `message` and `retry_after` are no part of a decision.
- `media_refs.json`, `context_window.json`: every row replays.
- `constraint_decider.json`: owed whole. No module replays this sheet yet, and the replay names it as owed
  until the change that brings its module.
