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
//! The answer is redacted. A worker's address is told without its userinfo
//! in each field that carries it ([`redacted`]). Identifiers in clear are
//! an admin's to ask and this build holds no role, so that request is
//! refused by its name: it is never answered redacted in silence.
//!
//! The rows are bounded twice: by the caller's `limit`, and by the bytes
//! of a watch's page whatever the caller asked, so the answer fits a frame.
//! `truncated` says that a bound left a row out. The rows come in the order
//! of their registration keys, so a cut answer names the same rows at each
//! call while the rows stand.
//!
//! The rows are the store's cut of the watched kinds, as a watch's open
//! reads them: every row is copied under the membership lock, whatever the
//! limit. A decision reads the published world and does not wait on it.
//! What takes that lock waits for the copy, on the thread that brought it:
//! a membership write, the report of a completed request on a gateway's
//! session, a load report, a probe's outcome, a veto, and the world's
//! publication.
//! **One snapshot is taken at a time, off the runtime's workers**
//! ([`Turn`]): one copy of the rows stands at a time, each of those waits
//! behind one snapshot at most, and the take itself holds no thread a
//! stream of the listener runs on.
//!
//! No standby is restored from the answer: redacted, and cut where a bound
//! bit, it names no place of the log.

use std::{sync::Arc, time::Instant};

use tokio::sync::{Mutex, Semaphore};
use tonic::Status;

use super::{generations::Generations, replication::age_ms, watch_workers, worker_wire};
use crate::{
    ingest::loads::{split, Address, UNPARSABLE},
    proto::{self, snapshot_request::Slice},
    store::{Kinds, MemoryStore, WorkerRow},
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

/// One snapshot at a time, off the runtime's workers. A snapshot copies
/// every worker row under the membership lock and builds its answer from
/// the copy: tens of milliseconds and megabytes at ten thousand rows. A
/// second call waits its turn, in the order the calls came, and a call past
/// the [`SNAPSHOTS_WAITING`] that wait is told to ask again. The take holds
/// the turn, so a caller that leaves frees it no earlier than its take
/// ends.
#[derive(Debug)]
pub(crate) struct Turn {
    /// The calls let in: the one that is taken and the ones that wait.
    places: Arc<Semaphore>,
    turn: Arc<Mutex<()>>,
}

impl Default for Turn {
    fn default() -> Self {
        Self {
            places: Arc::new(Semaphore::new(SNAPSHOTS_WAITING + 1)),
            turn: Arc::new(Mutex::new(())),
        }
    }
}

impl Turn {
    /// What `take` answers, run in its turn on a thread of the runtime's
    /// blocking pool; or the refusal of a call no place is left for, or of
    /// one whose take did not end.
    pub(crate) async fn run<T: Send + 'static>(
        &self,
        take: impl FnOnce() -> T + Send + 'static,
    ) -> Result<T, Status> {
        let Ok(place) = Arc::clone(&self.places).try_acquire_owned() else {
            return Err(Status::resource_exhausted(format!(
                "{SNAPSHOTS_WAITING} snapshots wait their turn: ask again later"
            )));
        };
        let turn = Arc::clone(&self.turn).lock_owned().await;
        let taking = tokio::task::spawn_blocking(move || {
            let _held = (turn, place);
            take()
        });
        taking
            .await
            .map_err(|join| Status::internal(format!("the snapshot did not end: {join}")))
    }

    /// The calls let in at this instant: the one taken and the ones that
    /// wait.
    #[cfg(test)]
    pub(crate) fn let_in(&self) -> usize {
        SNAPSHOTS_WAITING + 1 - self.places.available_permits()
    }

    /// Every place left, taken for a case: the next call is told to ask
    /// again while the case holds them.
    #[cfg(test)]
    pub(crate) fn fill(&self) -> Vec<tokio::sync::OwnedSemaphorePermit> {
        std::iter::from_fn(|| Arc::clone(&self.places).try_acquire_owned().ok()).collect()
    }

    /// The turn, held for a case: a call let in waits behind it.
    #[cfg(test)]
    pub(crate) async fn hold(&self) -> tokio::sync::OwnedMutexGuard<()> {
        Arc::clone(&self.turn).lock_owned().await
    }
}

/// What a snapshot's caller asked for, once the request holds this build's
/// rules ([`asked`]).
#[derive(Debug)]
pub(crate) struct Asked {
    workers: bool,
    spec: bool,
    /// The models whose rows are told; none is every model.
    models: Vec<String>,
    /// The rows a slice tells at most.
    limit: usize,
}

impl Asked {
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
pub(crate) fn asked(request: proto::SnapshotRequest) -> Result<Asked, Status> {
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
    if request.models.len() > worker_wire::MODELS {
        return Err(Status::invalid_argument(format!(
            "models: a snapshot names at most {} model ids",
            worker_wire::MODELS
        )));
    }
    worker_wire::watched(&request.models).map_err(|refused| {
        Status::invalid_argument(format!("{}: {}", refused.path, refused.why))
    })?;
    if request.plain_identifiers {
        return Err(Status::unimplemented(PLAIN));
    }
    Ok(Asked {
        workers: named & bit(Slice::Workers) != 0,
        spec: named & bit(Slice::Spec) != 0,
        models: request.models,
        // No limit of the caller's leaves the bound of a page's bytes alone.
        limit: match request.limit {
            0 => usize::MAX,
            limit => usize::try_from(limit).unwrap_or(usize::MAX),
        },
    })
}

/// The snapshot `asked` for, as this replica holds it at `now` under
/// `term`: the header, then each slice asked. The spec slice of a
/// controller with no spec source is refused ahead of the turn
/// ([`Asked::sourced`]), and the take keeps its own read of the source,
/// before a row is copied. The header's generation is the spec slice's own
/// where that slice is told: one read. The answer names no place of the
/// log, no id table and no document: it is no base a standby starts from.
pub(crate) fn taken(
    asked: &Asked,
    term: u64,
    controller_id: &str,
    store: &MemoryStore,
    generations: &Generations,
    now: Instant,
) -> Result<proto::Snapshot, Status> {
    let spec = if asked.spec {
        let status = generations
            .status()
            .ok_or_else(|| Status::unimplemented(NO_SOURCE))?;
        Some(status)
    } else {
        None
    };
    let (workers, truncated) = if asked.workers {
        rows(&store.cut(Kinds::Watched).rows, asked, now)
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

/// The rows of `cut` that serve an asked model, each as an operator reads
/// it at `now`, and whether a bound left one of them out: the caller's
/// limit, or a page's bytes ([`watch_workers::fill_within`]).
fn rows(cut: &[WorkerRow], asked: &Asked, now: Instant) -> (Vec<proto::WorkerRow>, bool) {
    let serving = || {
        cut.iter()
            .filter(|row| watch_workers::serves(&asked.models, row))
    };
    let told = watch_workers::fill_within(asked.limit, serving(), |row| redacted(row, now));
    let truncated = told.len() < serving().count();
    (told, truncated)
}

/// A store row as an operator reads it at `now`: the wire's row
/// ([`worker_wire::row`]) with its age, and its address without userinfo in
/// each field that carries it. A row tells its address three times: the
/// URL, the registration key, which is the URL, and the engine, which is
/// the key for a single rank. An operator loses no handle by it: a patch
/// and a removal name a row by its id and its revision.
fn redacted(row: &WorkerRow, now: Instant) -> proto::WorkerRow {
    let mut told = worker_wire::row(row);
    told.admitted_age_ms = age_ms(now, row.admitted_at);
    for address in [&mut told.url, &mut told.registration_key, &mut told.engine] {
        if let Some(clean) = without_userinfo(address) {
            *address = clean;
        }
    }
    told
}

/// `address` without its userinfo, or `None` where it carries none: the
/// userinfo goes, and every other byte stays as registered. The crate's one
/// split tells the userinfo from the address ([`split`]): an address it
/// cannot split is told as [`UNPARSABLE`], the marker a log line writes for
/// it. The scheme is told beside the host, and the split reads one by its
/// form alone: any other text before a `://` is the address's own, read
/// for its userinfo with the rest.
///
/// The userinfo is the one credential a URL carries by its form. A secret a
/// registrant wrote into a path or a query is not known as one here.
fn without_userinfo(address: &str) -> Option<String> {
    let Some(Address {
        scheme,
        userinfo,
        host,
        tail,
    }) = split(address)
    else {
        return Some(UNPARSABLE.to_owned());
    };
    userinfo.then(|| format!("{scheme}{host}{tail}"))
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
        asked, rows, served, taken, without_userinfo, Asked, Turn, SLICES, SNAPSHOTS_WAITING,
        UNPARSABLE,
    };
    use crate::{
        catalog::Catalog,
        controller::{
            generations::{tests::LEAST_LOAD, Format, Generations},
            watch_workers::PAGE_BYTES,
            worker_wire,
        },
        proto::{self, snapshot_request::Slice},
        store::{Kinds, MemoryStore, WorkerFacts, WorkersStore},
    };

    /// The rows slice alone: every model, no limit.
    const ROWS: Asked = Asked {
        workers: true,
        spec: false,
        models: Vec::new(),
        limit: usize::MAX,
    };

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
        const CONTRACT: &str = include_str!("../../contracts/policy_controller.proto");
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
        assert_eq!(both.limit, usize::MAX);
        assert!(both.models.is_empty());

        let rows = asked(proto::SnapshotRequest {
            models: vec!["model-a".to_owned()],
            limit: 7,
            ..naming(&[Slice::Workers])
        })
        .expect("served");
        assert!(rows.workers && !rows.spec);
        assert_eq!((rows.models, rows.limit), (vec!["model-a".to_owned()], 7));

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

    /// An address is told without its userinfo and with every other byte as
    /// registered. One with no userinfo is told as it is. One whose `@`
    /// stands past the authority's end is told by the marker: no part of
    /// what may be a userinfo is told.
    #[test]
    fn an_address_is_told_without_its_userinfo() {
        for (address, told) in [
            (
                "grpc://user:secret@10.0.0.5:8000",
                Some("grpc://10.0.0.5:8000"),
            ),
            (
                "https://user@host.example:8443/v1?x=1#top",
                Some("https://host.example:8443/v1?x=1#top"),
            ),
            ("user:secret@10.0.0.5:8000", Some("10.0.0.5:8000")),
            // An `@` inside the userinfo: the authority's last one ends it.
            (
                "grpc://us@er:se@cret@10.0.0.5:8000",
                Some("grpc://10.0.0.5:8000"),
            ),
            (
                "grpc+tls://user@10.0.0.5:8000",
                Some("grpc+tls://10.0.0.5:8000"),
            ),
            // A `://` behind the authority is no scheme's: the authority
            // starts at the address's first byte, and its userinfo goes.
            (
                "user:secret@10.0.0.5:8000/next/http://other",
                Some("10.0.0.5:8000/next/http://other"),
            ),
            (
                "user:secret@10.0.0.5:8000?next=http://other",
                Some("10.0.0.5:8000?next=http://other"),
            ),
            (
                "user:secret@10.0.0.5:8000#http://other",
                Some("10.0.0.5:8000#http://other"),
            ),
            (
                "grpc://user:secret@10.0.0.5:8000?next=http://other",
                Some("grpc://10.0.0.5:8000?next=http://other"),
            ),
            // No userinfo: every byte stays.
            ("grpc://10.0.0.5:8000", None),
            ("10.0.0.5:8000?next=http://other", None),
            ("http://[::1]:8000/v1?x=1", None),
            ("10.0.0.5:8000", None),
            ("1http://10.0.0.5:8000", None),
            ("", None),
            // A userinfo that holds a raw `/`, `?` or `#` reads like these,
            // and so does a path that holds an `@`.
            ("http://user:se/cret@10.0.0.5:8000", Some(UNPARSABLE)),
            ("http://user:se?cret@10.0.0.5:8000", Some(UNPARSABLE)),
            ("http://user:se#cret@10.0.0.5:8000", Some(UNPARSABLE)),
            ("http://10.0.0.5:8000/a@b", Some(UNPARSABLE)),
            // No scheme by its form stands before `://`: its `/` ends the
            // authority, and the `@` behind it stands past that end.
            ("user:se://cret@10.0.0.5:8000", Some(UNPARSABLE)),
            ("1http://user:secret@10.0.0.5:8000", Some(UNPARSABLE)),
        ] {
            assert_eq!(without_userinfo(address).as_deref(), told, "{address}");
        }
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
        let cut = store.cut(Kinds::Watched);
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

    /// The caller's limit leaves rows out, and the mark says so: the rows
    /// told are the first by their keys. A limit the rows are within sets
    /// no mark.
    #[test]
    fn a_limit_that_leaves_a_row_out_is_said() {
        let store = MemoryStore::new();
        for n in 0..5 {
            register(&store, &url(n), "model-a");
        }
        let cut = store.cut(Kinds::Watched);
        let now = Instant::now();
        let told = |limit| {
            let (page, truncated) = rows(&cut.rows, &Asked { limit, ..ROWS }, now);
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
        let cut = store.cut(Kinds::Watched);
        let (page, truncated) = rows(&cut.rows, &ROWS, Instant::now());
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
        let cut = store.cut(Kinds::Watched);
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
        let cut = store.cut(Kinds::Watched);
        let told = |limit| {
            let asked = Asked {
                models: vec!["model-b".to_owned(), "model-c".to_owned()],
                limit,
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

    /// A controller with no spec source serves its rows under the header
    /// and refuses the spec slice by its name, in the spec calls' words.
    /// The answer names no place of the log: no standby starts from it.
    #[test]
    fn a_controller_with_no_spec_source_serves_its_rows_and_refuses_the_spec_slice() {
        let store = Arc::new(MemoryStore::new());
        let id = register(&store, &url(5), "model-a");
        let generations = Generations::new(Arc::clone(&store), Arc::new(Catalog::builtin()));
        let take = |asked: &Asked| {
            taken(
                asked,
                3,
                "controller-0",
                &store,
                &generations,
                Instant::now(),
            )
        };

        let refused = take(&Asked { spec: true, ..ROWS }).expect_err("refused");
        assert_eq!(
            (refused.code(), refused.message()),
            (
                Code::Unimplemented,
                "slices: SLICE_SPEC is not served by a controller started with no spec source"
            )
        );

        // The handler reads the source ahead of the turn, in the same
        // words, and asks nothing of a request for the rows alone.
        let ahead = Asked { spec: true, ..ROWS }
            .sourced(false)
            .expect_err("refused");
        assert_eq!(
            (ahead.code(), ahead.message()),
            (refused.code(), refused.message())
        );
        assert!(ROWS.sourced(false).is_ok());
        assert!(Asked { spec: true, ..ROWS }.sourced(true).is_ok());

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
        let generations = Generations::new(Arc::clone(&store), Arc::new(Catalog::builtin()));
        generations
            .apply(LEAST_LOAD.into(), Format::Yaml)
            .expect("a valid document");
        let status = Asked {
            workers: false,
            spec: true,
            ..ROWS
        };
        let answer = taken(
            &status,
            1,
            "controller-0",
            &store,
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
        let turn = Arc::new(Turn::default());
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
                turn.run(take).await
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
        let past = tokio::time::timeout(Duration::from_secs(5), turn.run(|| ()))
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
}
