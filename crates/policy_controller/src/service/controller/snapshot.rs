//! The operators' snapshot (design 5.1 and 8): the control scope's read of
//! what this replica holds, a slice at a time, under the header every
//! snapshot carries. This build serves two slices: the worker rows, and the
//! spec's status.
//!
//! **A slice asked for and not served is refused by its name, never
//! answered empty.** An empty slice reads as a controller that holds
//! nothing of the kind. A request that names no slice asks for every one,
//! so it is refused too until every slice is served.
//!
//! The answer is redacted ([`snapshot`]). Identifiers in clear are
//! an admin's to ask and this build holds no role, so that request is
//! refused by its name: it is never answered redacted in silence.
//!
//! The rows are bounded twice: by the caller's `limit`, and by the bytes
//! of a watch's page whatever the caller asked, so the answer fits a frame.
//! `truncated` says that a bound left a row out. The rows come in the order
//! of their registration keys, so a cut answer names the same rows at each
//! call while the rows stand.
//!
//! The rows come by one of three roads ([`read`]). A cut kept at the log's
//! last watched write is shared with no copy, as a watch's start shares it
//! ([`Feed::rows_at_head`](super::watch_workers::Feed::rows_at_head)).
//! Where another call is let in or a watch is open, the first read at a
//! place copies every row under the membership lock and keeps the copy,
//! and each read that starts there shares it: calls that come together
//! hold that lock once a write, as watches do. A call with no limit and no
//! models takes that road too, alone or not: it would copy every row
//! anyway, and a call let in while it runs shares the copy. The last call
//! out lets the copy go where no watch is open to share it ([`SerialTurn`],
//! [`Watchers::unshared`](super::watch_workers::Watchers::unshared)): a
//! copy of every row is held only while a reader can share it. Any other
//! call that is alone with no watch open copies the rows it may tell and
//! keeps nothing ([`MemoryStore::rows_serving`]): its limit's worth, or
//! every row its models serve where it states no limit.
//!
//! A decision reads the published world and does not wait on that lock.
//! Whatever takes it waits for a copy, on the thread that brought it.
//! Among the store's writes, these take it each time: a membership write,
//! a probe's outcome, a veto that is set, a gateway's report of a request
//! that completed well, and a generation's apply. One takes it only in a
//! corner: the world's publication where a build is owed and due. Among
//! the reads: a watch's open where it copies and a standby's base.
//! **One snapshot is taken at a time, off the runtime's workers**
//! ([`SerialTurn`]): one operator's copy is made at a time, and the take itself
//! holds no thread a stream of the listener runs on. A watch's open and a
//! standby's base copy by themselves, in no turn.
//!
//! No standby is restored from the answer: redacted, and cut where a bound
//! bit, it names no place of the log.

use std::time::Instant;

use prost::encoding::message;
use tonic::Status;

use super::{generations::Generations, watch_workers, wire};
use crate::{
    document::operator::{
        snapshot::{self, Rows, WorkerRowsQuery},
        SerialTurn,
    },
    service::proto::{self, snapshot_request::Slice},
    store::MemoryStore,
};

/// Every slice of the contract, in its order.
const SLICES: [Slice; 10] = [
    Slice::Workers,
    Slice::Health,
    Slice::Loads,
    Slice::Sessions,
    Slice::Leases,
    Slice::Pairs,
    Slice::Spec,
    Slice::Plugins,
    Slice::KvIndex,
    Slice::Gateways,
];

/// The refusal of identifiers in clear.
const PLAIN: &str =
    "plain_identifiers: identifiers in clear are an admin's to ask, and this build holds no role";

/// The refusal of the spec slice by a controller that was given no spec
/// source, in the spec calls' own words.
const NO_SOURCE: &str =
    "slices: SLICE_SPEC is not served by a controller started with no spec source";

/// How many snapshots may wait their turn behind the one that is taken: a
/// call past them is told to ask again.
pub(crate) const SNAPSHOTS_WAITING: usize = 32;

/// Whether this build serves `slice`.
const fn served(slice: Slice) -> bool {
    matches!(slice, Slice::Workers | Slice::Spec)
}

/// The bit of `slice` in a set of slices: its number in the contract.
fn bit(slice: Slice) -> u32 {
    1 << i32::from(slice)
}

/// The slices `keep` keeps, by the contract's names and in its order.
fn names(keep: impl Fn(Slice) -> bool) -> String {
    SLICES
        .into_iter()
        .filter(|slice| keep(*slice))
        .map(|slice| slice.as_str_name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Takes a snapshot off the runtime's workers, with its turn held until
/// the take ends. `out` runs when the last place goes back, including
/// where a caller leaves while it waits.
pub(super) async fn in_turn<T: Send + 'static>(
    turn: &SerialTurn,
    take: impl FnOnce() -> T + Send + 'static,
    out: impl FnOnce() + Send + 'static,
) -> Result<T, Status> {
    let Some(held) = turn.take_with(out).await else {
        return Err(Status::resource_exhausted(format!(
            "{SNAPSHOTS_WAITING} snapshots wait their turn: ask again later"
        )));
    };
    tokio::task::spawn_blocking(move || {
        let _held = held;
        take()
    })
    .await
    .map_err(|join| Status::internal(format!("the snapshot did not end: {join}")))
}

/// What a snapshot's caller asked for, once the request holds this build's
/// rules ([`asked`]).
#[derive(Debug)]
pub(crate) struct SnapshotQuery {
    workers: bool,
    spec: bool,
    rows: WorkerRowsQuery,
}

impl SnapshotQuery {
    /// The refusal of the spec slice, by its name, where the controller was
    /// given no spec source (`bound` says whether it was). The handler
    /// reads it ahead of the turn, as it reads each rule of the request: a
    /// refused call takes no place.
    pub(crate) fn sourced(&self, bound: bool) -> Result<(), Status> {
        if self.spec && !bound {
            return Err(Status::unimplemented(NO_SOURCE));
        }
        Ok(())
    }
}

/// What `request` asks for, or the refusal of a field by its name, the
/// fields in the contract's order. A number that names no slice is the
/// caller's error. A slice this build does not serve, and identifiers in
/// clear, are this build's: `UNIMPLEMENTED`. The models are held to a
/// watch's bounds: the list is read at each row.
pub(crate) fn asked(request: proto::SnapshotRequest) -> Result<SnapshotQuery, Status> {
    let mut named = 0;
    for &number in &request.slices {
        match Slice::try_from(number) {
            Ok(slice) if slice != Slice::Unspecified => named |= bit(slice),
            _ => {
                return Err(Status::invalid_argument(format!(
                    "slices: {number} names no slice"
                )));
            }
        }
    }
    let every = request.slices.is_empty();
    if every {
        named = SLICES.into_iter().fold(0, |set, slice| set | bit(slice));
    }
    let unserved = names(|slice| named & bit(slice) != 0 && !served(slice));
    if !unserved.is_empty() {
        let none = if every {
            "none named is every slice, and "
        } else {
            ""
        };
        return Err(Status::unimplemented(format!(
            "slices: {none}this build does not serve {unserved}: it serves {}",
            names(served)
        )));
    }
    if let Some(why) = WorkerRowsQuery::refused(&request.models) {
        return Err(Status::invalid_argument(format!("models: {why}")));
    }
    if request.plain_identifiers {
        return Err(Status::unimplemented(PLAIN));
    }
    Ok(SnapshotQuery {
        workers: named & bit(Slice::Workers) != 0,
        spec: named & bit(Slice::Spec) != 0,
        rows: WorkerRowsQuery {
            models: request.models,
            // Zero leaves the bound of a page's bytes alone.
            limit: match request.limit {
                0 => usize::MAX,
                limit => usize::try_from(limit).unwrap_or(usize::MAX),
            },
        },
    })
}

/// The rows of one take, by the cheapest road that keeps calls that come
/// together at one copy a write. A cut kept at the log's head is shared
/// with no copy. Where another call is let in or a watch is open the
/// feed's cut is taken and kept: somebody can share it. A call with no
/// limit and no models takes it too: it would copy every row anyway, and a
/// call let in while it runs shares the copy. Any other call that is alone
/// with no watch open reads the rows it may tell and keeps nothing.
pub(crate) fn read(
    asked: &SnapshotQuery,
    turn: &SerialTurn,
    feed: &watch_workers::Feed,
    watchers: &watch_workers::Watchers,
    store: &MemoryStore,
) -> Rows {
    if let Some(cut) = feed.kept_at_head() {
        return Rows::Shared(cut);
    }
    let every = asked.rows.limit == usize::MAX && asked.rows.models.is_empty();
    if every || turn.let_in() > 1 || watchers.open() {
        return Rows::Shared(feed.rows_at_head(store));
    }
    let (rows, serving) = store.rows_serving(&asked.rows.models, asked.rows.limit);
    Rows::Told(rows, serving)
}

/// The snapshot `asked` for, as this replica holds it at `now` under
/// `term`: the header, then each slice asked. The rows are `read`'s, which
/// is called where the worker rows are asked and never else. The spec slice
/// of a controller with no spec source is refused ahead of the turn
/// ([`SnapshotQuery::sourced`]), and the take keeps its own read of the source,
/// before a row is copied. The header's generation is the spec slice's own
/// where that slice is told: one read. The answer names no place of the
/// log, no id table and no document: it is no base a standby starts from.
pub(crate) fn taken(
    asked: &SnapshotQuery,
    term: u64,
    controller_id: &str,
    read: impl FnOnce() -> Rows,
    generations: &Generations,
    now: Instant,
) -> Result<proto::Snapshot, Status> {
    let spec = if asked.spec {
        let status = generations
            .status()
            .ok_or_else(|| Status::unimplemented(NO_SOURCE))?;
        Some(wire::spec_status(status))
    } else {
        None
    };
    let (workers, truncated) = if asked.workers {
        rows(&asked.rows, read(), now)
    } else {
        (Vec::new(), false)
    };
    Ok(proto::Snapshot {
        term,
        generation: spec
            .as_ref()
            .map_or_else(|| generations.generation(), |spec| spec.generation),
        controller_id: controller_id.to_owned(),
        workers,
        spec,
        truncated,
        ..proto::Snapshot::default()
    })
}

/// A controller page keeps its bound on the encoded rows' bytes.
fn rows(asked: &WorkerRowsQuery, read: Rows, now: Instant) -> (Vec<proto::WorkerRow>, bool) {
    snapshot::worker_rows_page_with(asked, read, now, true, |row| {
        let told = wire::snapshot_row(&row, now);
        let framed = message::encoded_len(1, &told);
        (told, framed)
    })
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            mpsc, Arc,
        },
        thread,
        time::{Duration, Instant},
    };

    use prost::Message as _;
    use tokio::task::JoinSet;
    use tonic::Code;

    use super::{
        asked, in_turn, served, taken, Rows, SerialTurn, SnapshotQuery, WorkerRowsQuery, SLICES,
        SNAPSHOTS_WAITING,
    };
    use crate::{
        address::UNPARSABLE,
        document::operator::snapshot::without_userinfo,
        service::{
            controller::{
                generations::{tests::LEAST_LOAD, Format, Generations},
                watch_workers::PAGE_BYTES,
                worker_wire,
            },
            proto::{self, snapshot_request::Slice},
        },
        stages::registry::StageRegistry,
        store::{MemoryStore, StoreCheckpoint, WorkerFacts, WorkerRow, WorkersStore},
    };

    /// The rows slice alone: every model, no limit.
    const ROWS: SnapshotQuery = SnapshotQuery {
        workers: true,
        spec: false,
        rows: WorkerRowsQuery {
            models: Vec::new(),
            limit: usize::MAX,
        },
    };

    fn rows(
        cut: &[WorkerRow],
        asked: &SnapshotQuery,
        now: Instant,
    ) -> (Vec<proto::WorkerRow>, bool) {
        let cut = Arc::new(StoreCheckpoint {
            rows: cut.to_vec(),
            at: Default::default(),
            epoch: 0,
        });
        super::rows(&asked.rows, Rows::Shared(cut), now)
    }

    /// A request for `slices`.
    fn naming(slices: &[Slice]) -> proto::SnapshotRequest {
        proto::SnapshotRequest {
            slices: slices.iter().map(|slice| i32::from(*slice)).collect(),
            ..proto::SnapshotRequest::default()
        }
    }

    /// A row under `key` that serves `model`: its id.
    fn register(store: &MemoryStore, key: &str, model: &str) -> u64 {
        store
            .upsert(WorkerFacts {
                key: key.to_owned(),
                engine: key.to_owned(),
                incarnation: 1,
                models: vec![model.to_owned()],
                dp_size: 1,
                ..WorkerFacts::default()
            })
            .id()
            .0
    }

    /// The address of row `n`.
    fn url(n: usize) -> String {
        format!("grpc://10.0.0.{n}:8000")
    }

    /// The slices of the contract, by the contract's own text: a slice
    /// added there fails this until the table holds it, and is then refused
    /// by name until a step serves it.
    #[test]
    fn the_table_holds_every_slice_of_the_contract_in_its_order() {
        const CONTRACT: &str = include_str!("../../../contracts/policy_controller.proto");
        let named: Vec<&str> = CONTRACT
            .lines()
            .map(str::trim_start)
            .filter(|line| line.starts_with("SLICE_") && !line.starts_with("SLICE_UNSPECIFIED"))
            .filter_map(|line| line.split(' ').next())
            .collect();
        let table: Vec<&str> = SLICES.iter().map(|slice| slice.as_str_name()).collect();
        assert_eq!(table, named);
        assert_eq!(table.len(), 10);
    }

    /// A slice this build does not serve is refused by its name, alone or
    /// beside a served one. A request that names no slice asks for every
    /// one, and is refused by the names of the eight not served.
    #[test]
    fn a_slice_this_build_does_not_serve_is_refused_by_its_name() {
        let unserved: Vec<Slice> = SLICES.into_iter().filter(|slice| !served(*slice)).collect();
        assert_eq!(unserved.len(), 8);
        for slice in unserved {
            for request in [naming(&[slice]), naming(&[Slice::Workers, slice])] {
                let refused = asked(request).expect_err("refused");
                assert_eq!(refused.code(), Code::Unimplemented, "{slice:?}");
                assert_eq!(
                    refused.message(),
                    format!(
                        "slices: this build does not serve {}: it serves SLICE_WORKERS, SLICE_SPEC",
                        slice.as_str_name()
                    )
                );
            }
        }
        // Two of them, in the contract's order whatever the request's.
        let two = asked(naming(&[Slice::Loads, Slice::Spec, Slice::Health])).expect_err("refused");
        assert_eq!(
            two.message(),
            "slices: this build does not serve SLICE_HEALTH, SLICE_LOADS: it serves SLICE_WORKERS, SLICE_SPEC"
        );
        let every = asked(proto::SnapshotRequest::default()).expect_err("refused");
        assert_eq!(every.code(), Code::Unimplemented);
        assert_eq!(
            every.message(),
            "slices: none named is every slice, and this build does not serve SLICE_HEALTH, SLICE_LOADS, SLICE_SESSIONS, SLICE_LEASES, SLICE_PAIRS, SLICE_PLUGINS, SLICE_KV_INDEX, SLICE_GATEWAYS: it serves SLICE_WORKERS, SLICE_SPEC"
        );
    }

    /// A number that names no slice is the caller's error, by the field:
    /// the slice of no kind, a number past the contract's, a negative one.
    #[test]
    fn a_number_that_names_no_slice_is_refused_by_the_field() {
        for number in [0, 11, -1, i32::MAX] {
            let request = proto::SnapshotRequest {
                slices: vec![Slice::Workers.into(), number],
                ..proto::SnapshotRequest::default()
            };
            let refused = asked(request).expect_err("refused");
            assert_eq!(
                (refused.code(), refused.message()),
                (
                    Code::InvalidArgument,
                    format!("slices: {number} names no slice").as_str()
                )
            );
        }
    }

    /// The two served slices are asked alone or together, in any order, a
    /// slice named twice as once. A limit of 0 is no limit.
    #[test]
    fn the_served_slices_are_asked_alone_or_together() {
        let both = asked(naming(&[Slice::Spec, Slice::Workers, Slice::Spec])).expect("served");
        assert!(both.workers && both.spec);
        assert_eq!(both.rows.limit, usize::MAX);
        assert!(both.rows.models.is_empty());

        let rows = asked(proto::SnapshotRequest {
            models: vec!["model-a".to_owned()],
            limit: 7,
            ..naming(&[Slice::Workers])
        })
        .expect("served");
        assert!(rows.workers && !rows.spec);
        assert_eq!(
            (rows.rows.models, rows.rows.limit),
            (vec!["model-a".to_owned()], 7)
        );

        let status = asked(naming(&[Slice::Spec])).expect("served");
        assert!(!status.workers && status.spec);
    }

    /// Identifiers in clear are refused by name: this build holds no role,
    /// and the request is never answered redacted in silence.
    #[test]
    fn identifiers_in_clear_are_refused_by_name() {
        let request = proto::SnapshotRequest {
            plain_identifiers: true,
            ..naming(&[Slice::Workers, Slice::Spec])
        };
        let refused = asked(request).expect_err("refused");
        assert_eq!(
            (refused.code(), refused.message()),
            (
                Code::Unimplemented,
                "plain_identifiers: identifiers in clear are an admin's to ask, and this build holds no role"
            )
        );
    }

    /// The models of a snapshot hold a watch's bounds: the count, then each
    /// id's own rules.
    #[test]
    fn a_snapshots_models_are_held_to_a_watchs_bounds() {
        let of = |models: Vec<String>| proto::SnapshotRequest {
            models,
            ..naming(&[Slice::Workers])
        };
        let ids = |count: usize| (0..count).map(|n| format!("model-{n}")).collect();
        let unnamed = "models: a model id is not empty and holds no blank and no control character";
        let cases: [(Vec<String>, &str); 4] = [
            (
                ids(worker_wire::MODELS + 1),
                "models: a snapshot names at most 128 model ids",
            ),
            (
                vec!["m".repeat(257)],
                "models: a model id is at most 256 bytes",
            ),
            (vec!["model a".to_owned()], unnamed),
            (vec![String::new()], unnamed),
        ];
        for (models, why) in cases {
            let refused = asked(of(models)).expect_err("refused");
            assert_eq!(
                (refused.code(), refused.message()),
                (Code::InvalidArgument, why)
            );
        }
        let most = of(ids(worker_wire::MODELS));
        assert!(asked(most).is_ok(), "the bound itself is served");
    }

    /// A request's fields are read in the contract's order: the slices, the
    /// models, then the identifiers.
    #[test]
    fn a_requests_fields_are_read_in_the_contracts_order() {
        let mut request = proto::SnapshotRequest {
            slices: vec![11],
            models: vec![String::new()],
            plain_identifiers: true,
            limit: 0,
        };
        let refused = asked(request.clone()).expect_err("refused");
        assert_eq!(refused.message(), "slices: 11 names no slice");
        request.slices = vec![Slice::Health.into()];
        let refused = asked(request.clone()).expect_err("refused");
        assert!(refused.message().starts_with("slices: "), "{refused:?}");
        request.slices = vec![Slice::Workers.into()];
        let refused = asked(request.clone()).expect_err("refused");
        assert!(refused.message().starts_with("models: "), "{refused:?}");
        request.models.clear();
        let refused = asked(request.clone()).expect_err("refused");
        assert!(
            refused.message().starts_with("plain_identifiers: "),
            "{refused:?}"
        );
        request.plain_identifiers = false;
        assert!(asked(request).is_ok());
    }

    /// A row tells its address in three fields, each without the userinfo,
    /// and no byte of the row an operator reads holds the userinfo: a field
    /// that carries the key later fails this until it is redacted too. The
    /// store's own row keeps its key. The row tells its age at the read.
    #[test]
    fn a_row_tells_its_address_without_userinfo_in_every_field() {
        let store = MemoryStore::new();
        let key = "grpc://operator-name:s3cret-word@10.0.0.5:8000";
        let id = register(&store, key, "model-a");
        let cut = store.checkpoint();
        let later = cut.rows[0].admitted_at + Duration::from_millis(1_500);
        let (told, truncated) = rows(&cut.rows, &ROWS, later);
        let [row] = told.as_slice() else {
            panic!("one row: {told:?}");
        };
        assert!(!truncated);
        let clean = "grpc://10.0.0.5:8000";
        assert_eq!(
            (
                row.url.as_str(),
                row.registration_key.as_str(),
                row.engine.as_str()
            ),
            (clean, clean, clean)
        );
        let bytes = row.encode_to_vec();
        for secret in ["s3cret-word", "operator-name"] {
            assert!(
                !bytes
                    .windows(secret.len())
                    .any(|window| window == secret.as_bytes()),
                "the row tells {secret}"
            );
        }
        // What a patch and a removal name the row by, and the row's age.
        assert_eq!((row.id, row.revision), (id, cut.rows[0].revision));
        assert_eq!(row.admitted_age_ms, 1_500);
        assert_eq!(row.models, ["model-a"]);
        assert_eq!(cut.rows[0].facts.key, key);
    }

    /// A provider's key for a worker is its own text and may hold an
    /// address with a credential: a redacted row tells a key that is a URL
    /// without its userinfo, and any other key that holds an `@` as the
    /// marker. A key with no `@` is told as it stands, and a control
    /// call's row names no maker.
    #[test]
    fn a_makers_key_is_told_without_its_userinfo() {
        use crate::store::Registrant;

        let store = MemoryStore::new();
        register(&store, &url(1), "model-a");
        let mut row = store.checkpoint().rows[0].clone();
        let page = |row: &WorkerRow| {
            let (mut told, _) = rows(std::slice::from_ref(row), &ROWS, Instant::now());
            told.pop().expect("one row")
        };
        let told = |row: &WorkerRow| page(row).maker.map(|maker| maker.key);
        assert_eq!(told(&row), None, "a control call's row names no maker");
        row.maker = Registrant::provider(
            "pool",
            "http://operator-name:s3cret-word@pool.example/pods/a",
        )
        .expect("a whole maker");
        assert_eq!(
            told(&row).as_deref(),
            Some("http://pool.example/pods/a"),
            "the key keeps its userinfo"
        );
        let bytes = page(&row).encode_to_vec();
        for secret in ["s3cret-word", "operator-name"] {
            assert!(
                !bytes
                    .windows(secret.len())
                    .any(|window| window == secret.as_bytes()),
                "the row tells {secret}"
            );
        }
        // A key with no `@` is told whole. One with an `@` and no scheme
        // is no URL to clean: it reads as the marker, never as a part of
        // itself.
        for (key, reads) in [
            ("pods/a", "pods/a"),
            ("model-a@pool/pods/a", UNPARSABLE),
            ("pods/a@node-3", UNPARSABLE),
            ("operator-name:s3cret-word@", UNPARSABLE),
            ("http://pool.example/pods/a@node-3", UNPARSABLE),
        ] {
            row.maker = Registrant::provider("pool", key).expect("a whole maker");
            assert_eq!(told(&row).as_deref(), Some(reads), "the key {key}");
        }
    }

    /// The caller's limit leaves rows out, and the mark says so: the rows
    /// told are the first by their keys. A limit the rows are within sets
    /// no mark.
    #[test]
    fn a_limit_that_leaves_a_row_out_is_said() {
        let store = MemoryStore::new();
        for n in 0..5 {
            register(&store, &url(n), "model-a");
        }
        let cut = store.checkpoint();
        let now = Instant::now();
        let told = |limit| {
            let (page, truncated) = rows(
                &cut.rows,
                &SnapshotQuery {
                    rows: WorkerRowsQuery { limit, ..ROWS.rows },
                    ..ROWS
                },
                now,
            );
            let urls: Vec<String> = page.into_iter().map(|row| row.url).collect();
            (urls, truncated)
        };
        let all: Vec<String> = (0..5).map(url).collect();
        assert_eq!(told(usize::MAX), (all.clone(), false));
        assert_eq!(told(5), (all.clone(), false));
        assert_eq!(told(4), (all[..4].to_vec(), true));
        assert_eq!(told(1), (all[..1].to_vec(), true));
    }

    /// Rows past a page's bytes are left out whatever the limit, and the
    /// mark says so. The first row is told whatever it weighs.
    #[test]
    fn rows_past_a_pages_bytes_are_left_out_and_said() {
        let store = MemoryStore::new();
        // A row tells its address three times: each of these passes a
        // page's bytes alone.
        for n in 0..3 {
            let wide = format!("grpc://{n}.{}:8000", "u".repeat(PAGE_BYTES / 2));
            register(&store, &wide, "model-a");
        }
        let cut = store.checkpoint();
        let (page, truncated) = rows(&cut.rows, &ROWS, Instant::now());
        assert_eq!((page.len(), truncated), (1, true));
        assert!(page[0].url.starts_with("grpc://0."));
        // A lone call's own read is held to the same bytes.
        let asked = WorkerRowsQuery {
            models: Vec::new(),
            limit: 3,
        };
        let (read, serving) = store.rows_serving(&asked.models, asked.limit);
        let (page, truncated) = super::rows(&asked, Rows::Told(read, serving), Instant::now());
        assert_eq!((page.len(), truncated), (1, true));
        assert!(page[0].url.starts_with("grpc://0."));
    }

    /// The count of a watch's page is no bound of a snapshot. A watch pages
    /// at 128 rows: a snapshot tells more while a page's bytes hold them.
    #[test]
    fn a_snapshot_tells_more_rows_than_a_watchs_page_counts() {
        let store = MemoryStore::new();
        for n in 0..200 {
            let key = format!("grpc://10.0.{}.{}:8000", n / 100, n % 100);
            register(&store, &key, "model-a");
        }
        let cut = store.checkpoint();
        let (page, truncated) = rows(&cut.rows, &ROWS, Instant::now());
        assert_eq!((page.len(), truncated), (200, false));
    }

    /// A snapshot of some models tells the rows that serve one of them, and
    /// its mark counts those rows alone.
    #[test]
    fn a_snapshot_of_some_models_tells_the_rows_that_serve_them() {
        let store = MemoryStore::new();
        let models = ["model-a", "model-b", "model-a", "model-b", "model-a"];
        for (n, model) in models.into_iter().enumerate() {
            register(&store, &url(n), model);
        }
        let cut = store.checkpoint();
        let told = |limit| {
            let asked = SnapshotQuery {
                rows: WorkerRowsQuery {
                    models: vec!["model-b".to_owned(), "model-c".to_owned()],
                    limit,
                },
                ..ROWS
            };
            let (page, truncated) = rows(&cut.rows, &asked, Instant::now());
            let urls: Vec<String> = page.into_iter().map(|row| row.url).collect();
            (urls, truncated)
        };
        let serving = vec![url(1), url(3)];
        assert_eq!(told(usize::MAX), (serving.clone(), false));
        assert_eq!(told(2), (serving.clone(), false));
        assert_eq!(told(1), (serving[..1].to_vec(), true));
    }

    /// The rows a lone call reads for itself are the rows a shared cut
    /// tells, with the same word on what was left out: limit by limit, a
    /// limit of zero among them, with a models filter and with none, and
    /// with a list of models out of order. No more rows are copied than
    /// the limit tells.
    #[test]
    fn the_rows_read_for_one_call_are_the_cuts_rows() {
        let store = MemoryStore::new();
        let models = ["model-a", "model-b", "model-a", "model-b", "model-a"];
        for (n, model) in models.into_iter().enumerate() {
            register(&store, &url(n), model);
        }
        let cut = store.checkpoint();
        let filters = [
            Vec::new(),
            vec!["model-b".to_owned()],
            vec!["model-c".to_owned()],
            vec![
                "model-d".to_owned(),
                "model-c".to_owned(),
                "model-b".to_owned(),
            ],
        ];
        for models in filters {
            for limit in [0, 1, 2, 3, 5, 6, usize::MAX] {
                let asked = SnapshotQuery {
                    rows: WorkerRowsQuery {
                        models: models.clone(),
                        limit,
                    },
                    ..ROWS
                };
                let now = Instant::now();
                let (read, serving) = store.rows_serving(&asked.rows.models, asked.rows.limit);
                assert!(read.len() <= limit, "models {models:?}, limit {limit}");
                assert_eq!(
                    super::rows(&asked.rows, Rows::Told(read, serving), now),
                    rows(&cut.rows, &asked, now),
                    "models {models:?}, limit {limit}"
                );
            }
        }
    }

    /// A controller with no spec source serves its rows under the header
    /// and refuses the spec slice by its name, in the spec calls' words.
    /// The answer names no place of the log: no standby starts from it.
    #[test]
    fn a_controller_with_no_spec_source_serves_its_rows_and_refuses_the_spec_slice() {
        let store = Arc::new(MemoryStore::new());
        let id = register(&store, &url(5), "model-a");
        let generations = Generations::new(Arc::clone(&store), Arc::new(StageRegistry::builtin()));
        let take = |asked: &SnapshotQuery| {
            taken(
                asked,
                3,
                "controller-0",
                || Rows::Shared(Arc::new(store.checkpoint())),
                &generations,
                Instant::now(),
            )
        };

        let refused = take(&SnapshotQuery { spec: true, ..ROWS }).expect_err("refused");
        assert_eq!(
            (refused.code(), refused.message()),
            (
                Code::Unimplemented,
                "slices: SLICE_SPEC is not served by a controller started with no spec source"
            )
        );

        // The handler reads the source ahead of the turn, in the same
        // words, and asks nothing of a request for the rows alone.
        let ahead = SnapshotQuery { spec: true, ..ROWS }
            .sourced(false)
            .expect_err("refused");
        assert_eq!(
            (ahead.code(), ahead.message()),
            (refused.code(), refused.message())
        );
        assert!(ROWS.sourced(false).is_ok());
        assert!(SnapshotQuery { spec: true, ..ROWS }.sourced(true).is_ok());

        let answer = take(&ROWS).expect("the rows");
        assert_eq!(
            (
                answer.term,
                answer.generation,
                answer.controller_id.as_str()
            ),
            (3, 0, "controller-0")
        );
        let ids: Vec<u64> = answer.workers.iter().map(|row| row.id).collect();
        assert_eq!(ids, [id]);
        assert_eq!(answer.spec, None);
        assert!(!answer.truncated);
        assert_eq!(
            (
                answer.replication_seq,
                answer.replication_term,
                answer.membership_epoch
            ),
            (0, 0, 0)
        );
        assert!(answer.ids.is_none() && answer.documents.is_empty());
    }

    /// The header names the generation the spec slice tells, and a slice
    /// that was not asked for is absent.
    #[test]
    fn the_header_names_the_generation_the_spec_slice_tells() {
        let store = Arc::new(MemoryStore::new());
        register(&store, &url(5), "model-a");
        let generations = Generations::new(Arc::clone(&store), Arc::new(StageRegistry::builtin()));
        generations
            .apply(LEAST_LOAD.into(), Format::Yaml)
            .expect("a valid document");
        let status = SnapshotQuery {
            workers: false,
            spec: true,
            ..ROWS
        };
        let answer = taken(
            &status,
            1,
            "controller-0",
            || -> Rows { panic!("no row is asked: no row is read") },
            &generations,
            Instant::now(),
        )
        .expect("the status");
        let spec = answer.spec.expect("the spec slice");
        assert_eq!(
            (answer.generation, spec.generation, spec.source.as_str()),
            (1, 1, "file")
        );
        assert!(answer.workers.is_empty() && !answer.truncated);
    }

    /// One snapshot is taken at a time, off the runtime's workers: a second
    /// call waits its turn, and a call past the ones let in is told to ask
    /// again at once. The case's runtime has one thread, and each take
    /// waits for the case's word: a take on that thread could not be told.
    #[tokio::test]
    async fn one_snapshot_is_taken_at_a_time_and_a_call_past_the_wait_is_told_to_ask_again() {
        let turn = Arc::new(SerialTurn::new(SNAPSHOTS_WAITING));
        let runtime = thread::current().id();
        let (release, word) = mpsc::channel::<()>();
        let word = Arc::new(std::sync::Mutex::new(word));
        let inside = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let on_runtime = Arc::new(AtomicBool::new(false));
        let mut calls = JoinSet::new();
        for n in 0..=SNAPSHOTS_WAITING {
            let turn = Arc::clone(&turn);
            let (word, inside, most) = (Arc::clone(&word), Arc::clone(&inside), Arc::clone(&most));
            let on_runtime = Arc::clone(&on_runtime);
            calls.spawn(async move {
                let take = move || {
                    if thread::current().id() == runtime {
                        // No word can come: the case's thread is this one.
                        on_runtime.store(true, Ordering::SeqCst);
                        return n;
                    }
                    let now = inside.fetch_add(1, Ordering::SeqCst) + 1;
                    most.fetch_max(now, Ordering::SeqCst);
                    let told = word
                        .lock()
                        .expect("the word's lock")
                        .recv_timeout(Duration::from_secs(30));
                    inside.fetch_sub(1, Ordering::SeqCst);
                    assert!(told.is_ok(), "the case's word came");
                    n
                };
                in_turn(&turn, take, || ()).await
            });
        }
        // Every call is let in: one is taken, the others wait their turn.
        let began = Instant::now();
        while turn.let_in() <= SNAPSHOTS_WAITING && began.elapsed() < Duration::from_secs(20) {
            tokio::task::yield_now().await;
        }
        assert!(
            !on_runtime.load(Ordering::SeqCst),
            "a take ran on the runtime's thread"
        );
        assert_eq!(turn.let_in(), SNAPSHOTS_WAITING + 1);
        let past = tokio::time::timeout(Duration::from_secs(5), in_turn(&turn, || (), || ()))
            .await
            .expect("an answer at once")
            .expect_err("no place is left");
        assert_eq!(
            (past.code(), past.message()),
            (
                Code::ResourceExhausted,
                "32 snapshots wait their turn: ask again later"
            )
        );
        for _ in 0..=SNAPSHOTS_WAITING {
            release.send(()).expect("a take waits for the word");
        }
        let mut ended = Vec::new();
        while let Some(call) = calls.join_next().await {
            ended.push(call.expect("the call's task").expect("its turn came"));
        }
        ended.sort_unstable();
        assert_eq!(ended, (0..=SNAPSHOTS_WAITING).collect::<Vec<_>>());
        assert_eq!(most.load(Ordering::SeqCst), 1, "two takes ran at once");
        assert_eq!(turn.let_in(), 0, "each place went back");
    }

    /// A caller that leaves frees the turn no earlier than its take ends:
    /// the take holds the turn and the place, not the caller's future. The
    /// first call's task is ended while its take waits for the case's word.
    /// Its place stays taken, and a second call's take does not begin until
    /// the word comes.
    #[tokio::test]
    async fn a_caller_that_leaves_frees_the_turn_no_earlier_than_its_take_ends() {
        let turn = Arc::new(SerialTurn::new(SNAPSHOTS_WAITING));
        let (release, word) = mpsc::channel::<()>();
        let (started, at_work) = mpsc::channel::<()>();
        let mut calls = JoinSet::new();
        let first = calls.spawn({
            let turn = Arc::clone(&turn);
            async move {
                let take = move || {
                    started.send(()).expect("the case listens");
                    word.recv_timeout(Duration::from_secs(30)).is_ok()
                };
                in_turn(&turn, take, || ()).await.map(|_told| ())
            }
        });
        // The first take is at work on the blocking pool.
        let since = Instant::now();
        while at_work.try_recv().is_err() {
            assert!(since.elapsed() < Duration::from_secs(20), "the take began");
            tokio::task::yield_now().await;
        }
        // Its caller leaves.
        first.abort();
        let left = calls.join_next().await.expect("the first call");
        assert!(
            left.is_err_and(|ended| ended.is_cancelled()),
            "the caller left"
        );
        assert_eq!(turn.let_in(), 1, "the place is the take's until it ends");

        // A second call is let in and waits: its take does not begin.
        let ran = Arc::new(AtomicBool::new(false));
        calls.spawn({
            let (turn, ran) = (Arc::clone(&turn), Arc::clone(&ran));
            async move { in_turn(&turn, move || ran.store(true, Ordering::SeqCst), || ()).await }
        });
        let since = Instant::now();
        while turn.let_in() < 2 {
            assert!(
                since.elapsed() < Duration::from_secs(20),
                "the second call took no place"
            );
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !ran.load(Ordering::SeqCst),
            "a second take ran beside the first"
        );

        // The word comes: the first take ends, and the second has its turn.
        release.send(()).expect("the take waits for the word");
        calls
            .join_next()
            .await
            .expect("the second call")
            .expect("its task")
            .expect("its turn came");
        assert!(ran.load(Ordering::SeqCst), "the second take ran");
        let since = Instant::now();
        while turn.let_in() > 0 {
            assert!(
                since.elapsed() < Duration::from_secs(20),
                "a place did not go back"
            );
            tokio::task::yield_now().await;
        }
    }

    /// The last call out is the one whose place goes back while no other
    /// call is let in, whether its take ended or its caller left while it
    /// waited. A call that leaves another let in is not the last.
    #[tokio::test]
    async fn the_last_call_out_is_the_one_that_leaves_no_call_let_in() {
        let turn = Arc::new(SerialTurn::new(SNAPSHOTS_WAITING));
        let outs = Arc::new(AtomicUsize::new(0));
        let counted = || {
            let outs = Arc::clone(&outs);
            move || {
                outs.fetch_add(1, Ordering::SeqCst);
            }
        };
        // A call alone is the last out.
        in_turn(&turn, || (), counted()).await.expect("its turn");
        assert_eq!(outs.load(Ordering::SeqCst), 1);

        // Two calls are let in: the first out leaves the second let in.
        let held = turn.hold().await;
        let mut calls = JoinSet::new();
        for _ in 0..2 {
            let (turn, out) = (Arc::clone(&turn), counted());
            calls.spawn(async move { in_turn(&turn, || (), out).await });
        }
        let since = Instant::now();
        while turn.let_in() < 2 {
            assert!(
                since.elapsed() < Duration::from_secs(20),
                "the calls took no place"
            );
            tokio::task::yield_now().await;
        }
        drop(held);
        while let Some(call) = calls.join_next().await {
            call.expect("the call's task").expect("its turn came");
        }
        assert_eq!(outs.load(Ordering::SeqCst), 2, "one last call of the two");

        // A caller that leaves while it waits is the last out too.
        let held = turn.hold().await;
        let waiting = calls.spawn({
            let (turn, out) = (Arc::clone(&turn), counted());
            async move { in_turn(&turn, || (), out).await }
        });
        let since = Instant::now();
        while turn.let_in() < 1 {
            assert!(
                since.elapsed() < Duration::from_secs(20),
                "the call took no place"
            );
            tokio::task::yield_now().await;
        }
        waiting.abort();
        let left = calls.join_next().await.expect("the call");
        assert!(
            left.is_err_and(|ended| ended.is_cancelled()),
            "the caller left"
        );
        assert_eq!(turn.let_in(), 0, "its place went back");
        assert_eq!(outs.load(Ordering::SeqCst), 3, "the caller that left");
        drop(held);
    }

    #[test]
    fn plain_snapshot_rows_map_to_the_previous_wire_rows() {
        use crate::document::operator::snapshot::{
            worker_rows_page, Rows as ReadRows, WorkerRowsQuery,
        };

        let store = MemoryStore::new();
        for (key, model) in [
            ("grpc://user:secret@host-a:8000", "a"),
            ("grpc://host-b:8000", "b"),
            ("grpc://user:se/cret@host-c:8000", "a"),
            ("grpc://host-d:8000", "b"),
        ] {
            register(&store, key, model);
        }
        let cut = Arc::new(store.checkpoint());
        let now = Instant::now() + Duration::from_secs(1);
        for models in [Vec::new(), vec!["a".to_owned()], vec!["none".to_owned()]] {
            for limit in [0, 1, 2, 3, usize::MAX] {
                let asked = WorkerRowsQuery {
                    models: models.clone(),
                    limit,
                };
                let read = || store.rows_serving(&models, limit);
                let (rows, serving) = read();
                let expected = legacy_rows(&cut.rows, &models, limit, now);
                for read in [
                    ReadRows::Shared(Arc::clone(&cut)),
                    ReadRows::Told(rows, serving),
                ] {
                    let found = worker_rows_page(&asked, read, now);
                    let mapped: Vec<_> = found
                        .rows
                        .iter()
                        .map(|row| super::wire::snapshot_row(row, now))
                        .collect();
                    assert_eq!((mapped, found.truncated), expected, "{models:?}, {limit}");
                }
            }
        }
    }

    // The wire page that preceded the plain operator rows.
    fn legacy_rows(
        cut: &[WorkerRow],
        models: &[String],
        limit: usize,
        now: Instant,
    ) -> (Vec<proto::WorkerRow>, bool) {
        let serving = || {
            cut.iter().filter(|row| {
                models.is_empty() || row.facts.models.iter().any(|model| models.contains(model))
            })
        };
        let rows = super::watch_workers::fill_within(limit, serving(), |row| {
            let mut told = worker_wire::row(row);
            told.admitted_age_ms = super::super::replication::age_ms(now, row.admitted_at);
            let maker = told.maker.as_mut().map(|maker| &mut maker.key);
            for address in [&mut told.url, &mut told.registration_key, &mut told.engine]
                .into_iter()
                .chain(maker)
            {
                if let Some(clean) = without_userinfo(address) {
                    *address = clean;
                }
            }
            told
        });
        let truncated = rows.len() < serving().count();
        (rows, truncated)
    }
}
