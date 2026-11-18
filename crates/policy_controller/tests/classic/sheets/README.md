# The recorded policy sheets

What the gateway's own routing policy code picks, recorded row by row. The gateway chooses a worker for each request
with a load-balancing policy, and this crate builds those policies a second time. The test target
`tests/classic_rows.rs` replays every row of each sheet that has a module there through the new code, and fails on a
row it decides differently. A sheet with no module yet is read and owed by name. The
reference is the gateway's policy code at the commit a sheet names, run, never read.

One sheet records no pick: `pair_index.json` holds which prefill and decode workers the gateway pairs, the lists a
pair's placement then picks from.

One sheet records one rule under every policy: `all_vetoed.json` holds what the gateway answers when every usable
worker of a pool is over its overload thresholds, and what a routing key gets under consistent hashing while its
own worker is over them.

One sheet records a request that carries encode items: `encode_pair.json` holds the pair the gateway picks for it,
the encoders it offers the request's items and the encoder each item takes, or the name of its refusal.

One sheet records the placement of a pair: `pair_select.json` holds which prefill and which decode the gateway
places a request on, over two pools its pair index has paired.

One sheet records no pick and no policy: `overload.json` holds whether the gateway reads one worker as overloaded
after each thing its load monitor is handed about the worker.

```
crates/policy_controller/tests/classic/sheets/
  power_of_two.json        one sheet per policy: its recorded rows
  manual.json
  prefix_hash.json
  pair_index.json
  all_vetoed.json
  encode_pair.json
  pair_select.json
  overload.json
  bucket.json
model_gateway/src/         each recorder, a test module beside the code it runs
  policies/classic_rows.rs                                                  power_of_two.json
  policies/classic_manual_rows.rs                                           manual.json
  policies/classic_prefix_hash_rows.rs                                      prefix_hash.json
  policies/classic_all_vetoed_rows.rs                                       all_vetoed.json
  worker/pd/pairing/classic_pair_rows.rs                                    pair_index.json
  routers/grpc/common/stages/worker_selection/classic_encode_pair_rows.rs   encode_pair.json
  routers/common/classic_pair_select_rows.rs                                pair_select.json
  worker/load/monitor/classic_overload_rows.rs                              overload.json
  policies/bucket/classic_bucket_rows.rs                                    bucket.json
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
under its thresholds), `over` (ready, over them), `stalled` (ready, doubted by the liveness tracker) or `down` (not
ready). A step is a pick (with the routing key the request carries, if any, and the workers an earlier attempt
tried), new loads for every worker, or new states. A pick is recorded as the worker's place in the row's list. A
request placed nowhere is a number past the last worker: the count of the workers when the gateway sheds it as an
overload shed, the count plus one when it finds no worker to use.

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

## The sheets

Each sheet is recorded on its own, so the commit is a column of the sheet's row and never one commit for the
directory. The sha256 is of the file as committed.

| sheet | policy | recorded from (commit) | rows | sha256 | recorder |
|---|---|---|---|---|---|
| `power_of_two.json` | power of two, under each of its four load metrics | `1b141d03b5ca9097f54f0ebc4b95d72e810a62c3` | 44: 24 sequence, 20 drawn | `602e1fe790cc5368ad05de2297b6e482b7a8893704ee448699ec467df9e0d87c` | `classic_rows.rs` |
| `manual.json` | manual, under each of its assignment modes | `1b141d03b5ca9097f54f0ebc4b95d72e810a62c3` | 14: 9 sequence, 5 drawn | `d34189025a50a7a31f925c632ce1964a09dac0baab89370e6e87c5c842db9a5a` | `classic_manual_rows.rs` |
| `prefix_hash.json` | prefix hash: a key from a prompt's prefix, from a routing key and from nothing; the load bound and where a passed key goes; how deep the key reads; the boundaries a prompt is keyed at and the hand-down between their workers | `d92647556e91a3601ef647f12ae5784df2080dc3` | 47: 31 sequence, 16 drawn | `7f64bde829f86aed7615d46a82332eaa4cd55b0cdbf1fff59ef1163b59679296` | `classic_prefix_hash_rows.rs` |
| `all_vetoed.json` | every policy, over a pool whose every usable worker is over its overload thresholds; a routing key under consistent hashing whose own worker is over them | `d9f8018881cc54c9b0f6e993849a59f8f7c54ab5` | 42: 38 sequence, 4 drawn | `e5496add4cc24e8438e60849570c6f0113f3537eeb2010f448fb8af5a432527b` | `classic_all_vetoed_rows.rs` |
| `pair_index.json` | the prefill and decode pair index, on both pair wires under each pairing mode | `64ff5a437b5007af5079f0492b46f4e5a712b66f` | 23 pair rows | `383783a8a28bdc3d0753111fbb84df4be2713fe332b448938b9cb75f44dec694` | `classic_pair_rows.rs` |
| `encode_pair.json` | a request with encode items: the pair, the encoders offered and each item's encoder | `4ba2a6431357be5411a2383cdc4916b41471d722` | 52: 52 sequence, 0 drawn | `988bf253da9310aeedd08ec2704d8e01ecc4a1210453353723ba0faa9001de33` | `classic_encode_pair_rows.rs` |
| `pair_select.json` | the placement of a prefill and decode pair: each side under its policy, a side over its overload thresholds or doubted whole, a decode's rooms, the prefill admission's bound, the least-loaded default's own rules, a role with no worker | `4ba2a6431357be5411a2383cdc4916b41471d722` | 40: 23 sequence, 17 drawn | `bd8426316ae7337ede82b5a931c1d00cd9dee1aca22c528278f32b2671846eb8` | `classic_pair_select_rows.rs` |
| `overload.json` | the overload verdict of one worker: the gateway's thresholds and the worker's own, a poll's answer, a poll with no report, a pushed record, an eviction | `436f0c8c5725b0a408b990003d76eaddc72c4be4` | 25 verdict rows | `29c10ee8876e53d23a824a3b7bd0481efd118e8ac5a0d2822477729f6e21283a` | `classic_overload_rows.rs` |
| `bucket.json` | bucket: a size's range and its worker, the hand-down when that worker is not offered, the two parts of the balance rule, the window a booking leaves, the adjustment's cut, a worker that joins or leaves, a pool nobody listed | `d92647556e91a3601ef647f12ae5784df2080dc3` | 19: 18 sequence, 1 drawn | `71a533017809566486335bb9f212ea49712a833bbe5ebcbde7c5d6319ee764e4` | `classic_bucket_rows.rs` |

## Recording

A recorder is an ignored test of the gateway crate: a test module beside the code it runs, so it reads that code by
the code's own functions. Run one with the output directory and the commit it runs at, from the repository's root:

```bash
CLASSIC_ROWS_OUT=$PWD/crates/policy_controller/tests/classic/sheets CLASSIC_ROWS_COMMIT=<commit> \
  cargo test -p smg --lib policies::classic_rows -- --ignored --nocapture
```

| sheet | test path |
|---|---|
| `power_of_two.json` | `policies::classic_rows` |
| `manual.json` | `policies::classic_manual_rows` |
| `prefix_hash.json` | `policies::classic_prefix_hash_rows` |
| `all_vetoed.json` | `policies::classic_all_vetoed_rows` |
| `pair_index.json` | `worker::pd::pairing::classic_pair_rows` |
| `encode_pair.json` | `worker_selection::classic_encode_pair_rows` |
| `pair_select.json` | `routers::common::classic_pair_select_rows` |
| `overload.json` | `worker::load::monitor::classic_overload_rows` |
| `bucket.json` | `policies::bucket::classic_bucket_rows` |

The pair sheet's recorder is a child of the module that derives a worker's pairing descriptor, so it records the
descriptor the gateway derived and never a second reading of the registration. The encode sheet's recorder is a
child of the gateway's worker selection stage: it runs the stage's own selection, and it wraps the encode policy to
write down each call (the encoders offered and the key) and passes every call on unchanged. The pair placement
sheet's recorder sits beside the placement it calls and builds its pools, policies and pair index with the gateway's
own types. The overload sheet's recorder is a child of the gateway's load monitor, so each step enters the monitor
through the monitor's own functions and the verdict is read off the worker the monitor wrote it on. The bucket
sheet's recorder is a child of the policy's own module and stands in that policy's directory: the adjustment and
the window are the policy's private state, and a module beside the policy's could reach neither.

A sheet is never edited by hand. A new recording replaces the whole sheet and its row in the table above, in one
pull request that names the commit and says which rows moved: a sequence row moves only when the policy's
behavior did, and the drawn rows are a new sample.

## Reading

The crate's replay reads this directory, every run of its suite with the `ingest` feature:

```bash
cargo test -p smg-policy-controller --features ingest --test classic_rows
```

A few rows of each sheet are also copied into the crate's test source, and the replay holds each copy to its sheet.

## Limits left

A limit is a recorded row the replay does not decide as recorded, with the reason. The replay names each one and
fails once the reason no longer holds.

- `power_of_two.json`: none. All 44 rows replay.
- `all_vetoed.json`: eight rows are owed, the ring rows: the crate does not keep a routing key on its ring worker
  yet, and their module refuses them by name. The other 34 rows replay, the keyed round robin row and the two rows
  of the `bucket` policy (a pool over on every worker, with shedding off and on) among them.
- `pair_index.json`: none. All 23 rows replay.
- `encode_pair.json`: none is recorded yet. The module that replays it arrives with the crate's encoder lists
  and its rule for encode items; until then the replay owes this sheet.
- `pair_select.json`: none is recorded yet. The module that replays it arrives with the crate's pair
  decision; until then the replay owes this sheet.
- `overload.json`: six rows are decided apart, each by a rule of the crate's store that the replay names with the
  verdicts the store gives. The store holds a usage and its threshold in permille, so a usage within half a permille
  under the threshold reads at it (`a-usage-under-the-threshold-by-less-than-half-a-permille-is-under-it`). The
  gateway stops reading a rank at once. The store judges every write over every sample it holds for the worker that
  lies within three report periods of the worker's newest, and the replay sets one period between two steps. So a
  rank an answer omits is read for three periods more (`a-rank-an-answer-omits-leaves-the-sum-at-once`,
  `a-rank-an-answer-omits-leaves-the-mean-at-once`). A pushed record is judged beside a poll's figures for a rank
  that never pushed until those lie three periods behind
  (`a-record-after-a-poll-is-judged-over-the-ranks-that-have-pushed`,
  `a-record-s-usage-is-judged-over-the-ranks-that-have-pushed`). An answer with no entry changes nothing the store
  holds (`an-answer-that-names-no-rank-takes-the-ranks-that-pushed-before-it-off-the-report`). Each of these five
  rows repeats the step that leaves a rank unheard, and replays as recorded from its fourth repeat on. The other 19
  rows replay.
- `manual.json`: one row is owed, `manual-one-key-pins-apart-under-the-prefill-and-the-decode-role`. The gateway
  keeps a key's pins apart under each role of a prefill and decode pair. The crate refuses a sessions rule under a
  pair plan today, so it cannot build the row; the other 13 rows replay.
- `prefix_hash.json`: two rows are decided apart, each by the key this crate reads, and the replay names both with
  the outcome the crate gives. The gateway hashes a prompt's own tokens. A key here is the hash of a request's last
  whole block of the view's block size, made by the host that feeds the request. So the tokens past that block are
  outside the key (`the-last-token-of-a-prompt-under-256-tokens-is-in-the-key`), and a request under one block has
  no key and is placed as one with nothing to hash, on the fewest requests
  (`a-ten-token-prompt-keeps-its-key-s-worker-over-the-fewest-requests`). The other 45 rows replay.
- `bucket.json`: none. All 19 rows replay.
