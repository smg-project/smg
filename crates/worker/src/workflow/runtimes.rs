//! What a registration knows of the runtimes it tells apart.

use crate::identity::basic::RuntimeType;

/// What a runtime can serve over a ZMQ link.
pub(super) struct ZmqRuntime {
    runtime: RuntimeType,
    /// Whether the wire also carries disaggregated requests.
    pub(super) supports_disaggregation: bool,
}

/// The runtimes that speak ZMQ. Each can serve a regular worker.
const ZMQ_RUNTIMES: [ZmqRuntime; 3] = [
    ZmqRuntime {
        runtime: RuntimeType::Vllm,
        supports_disaggregation: false,
    },
    ZmqRuntime {
        runtime: RuntimeType::TokenSpeed,
        supports_disaggregation: true,
    },
    ZmqRuntime {
        runtime: RuntimeType::Sglang,
        supports_disaggregation: false,
    },
];

/// The runtime's ZMQ row, if its wire can serve a ZMQ worker.
pub(super) fn zmq_runtime(runtime: RuntimeType) -> Option<&'static ZmqRuntime> {
    ZMQ_RUNTIMES.iter().find(|row| row.runtime == runtime)
}

/// The ZMQ runtime names in the order a refusal lists them.
pub(super) fn zmq_runtime_names() -> String {
    let [first, second, last] = ZMQ_RUNTIMES.map(|row| row.runtime.as_str());
    format!("{first}, {second} and {last}")
}

/// One runtime as a registration tells it apart over HTTP.
pub(super) struct HttpRuntime {
    /// The runtime.
    pub(super) runtime: RuntimeType,
    /// The owner names of a model list that name it.
    pub(super) owners: &'static [&'static str],
    /// The path that tells it apart.
    pub(super) path: &'static str,
    /// How its metadata is read.
    pub(super) metadata: MetadataRead,
    /// Whether its server info reports its data-parallel (DP) size.
    pub(super) reports_dp: bool,
}

/// How an HTTP worker's metadata is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetadataRead {
    /// The info routes: the server's and the model's, each under its older
    /// spelling too, then the model list for what they left open. A runtime
    /// with no row is read so.
    InfoRoutes,
    /// The model list, then the row's own path.
    ModelListAndPath,
}

/// The runtimes told apart over HTTP, the deciding row first. Where the
/// paths of two rows both answer, the worker runs the earlier row's runtime:
/// a later row's path may be served by an earlier row's runtime too.
pub(super) const HTTP_RUNTIMES: [HttpRuntime; 2] = [
    HttpRuntime {
        runtime: RuntimeType::Vllm,
        owners: &["vllm"],
        path: "/version",
        metadata: MetadataRead::ModelListAndPath,
        reports_dp: false,
    },
    HttpRuntime {
        runtime: RuntimeType::Sglang,
        owners: &["sglang", "nvidia"],
        path: "/server_info",
        metadata: MetadataRead::InfoRoutes,
        reports_dp: true,
    },
];

/// The runtimes a gRPC reach asks at once, each on its own health service,
/// in the order its failure names them.
pub(super) const GRPC_RUNTIMES: [RuntimeType; 5] = [
    RuntimeType::Sglang,
    RuntimeType::Vllm,
    RuntimeType::Trtllm,
    RuntimeType::Mlx,
    RuntimeType::TokenSpeed,
];

/// The runtimes a gRPC detection asks one after another, in the order it
/// asks them. The same five as [`GRPC_RUNTIMES`]; the last two stand the
/// other way round, and each list's failure text follows its own order.
pub(super) const GRPC_DETECTION_ORDER: [RuntimeType; 5] = [
    RuntimeType::Sglang,
    RuntimeType::Vllm,
    RuntimeType::Trtllm,
    RuntimeType::TokenSpeed,
    RuntimeType::Mlx,
];

/// The runtime used to read HTTP metadata or create a worker when detection
/// leaves the runtime unspecified.
pub(super) const DEFAULT_RUNTIME: RuntimeType = RuntimeType::Sglang;

/// The row of the runtime that `runtime_type` names by its id, if it has one.
pub(super) fn http_runtime(runtime_type: &str) -> Option<&'static HttpRuntime> {
    HTTP_RUNTIMES
        .iter()
        .find(|row| row.runtime.as_str() == runtime_type)
}

/// Whether the runtime that `runtime_type` names by its id reports its DP
/// size in its server info over HTTP. A runtime with no row reports none.
pub(super) fn reports_dp(runtime_type: &str) -> bool {
    http_runtime(runtime_type).is_some_and(|row| row.reports_dp)
}

/// The runtime a model list's owner names, if it names one.
pub(super) fn runtime_of_owner(owner: &str) -> Option<RuntimeType> {
    HTTP_RUNTIMES
        .iter()
        .find(|row| row.owners.contains(&owner))
        .map(|row| row.runtime)
}

/// The ids of `runtimes`, in their order, as a failure lists them.
pub(super) fn listed(runtimes: &[RuntimeType]) -> String {
    runtimes
        .iter()
        .map(|runtime| runtime.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every owner name of a row names that row's runtime, and a name no row
    /// holds names none.
    #[test]
    fn an_owner_names_the_runtime_of_its_row() {
        for row in &HTTP_RUNTIMES {
            for owner in row.owners {
                assert_eq!(runtime_of_owner(owner), Some(row.runtime), "{owner}");
            }
        }
        assert_eq!(runtime_of_owner("self_hosted"), None);
    }

    /// The two gRPC lists hold the same runtimes, each once.
    #[test]
    fn the_two_grpc_lists_hold_the_same_runtimes() {
        let sorted = |runtimes: &[RuntimeType]| {
            let mut ids: Vec<&str> = runtimes.iter().map(|runtime| runtime.as_str()).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        assert_eq!(sorted(&GRPC_RUNTIMES), sorted(&GRPC_DETECTION_ORDER));
        assert_eq!(sorted(&GRPC_RUNTIMES).len(), GRPC_RUNTIMES.len());
    }

    /// A failure lists the ids in the list's own order.
    #[test]
    fn a_list_is_named_in_its_own_order() {
        let listed_ids = listed(&GRPC_DETECTION_ORDER);
        let ids: Vec<&str> = listed_ids.split(", ").collect();
        assert_eq!(ids.len(), GRPC_DETECTION_ORDER.len());
        for (id, runtime) in ids.iter().zip(GRPC_DETECTION_ORDER) {
            assert_eq!(*id, runtime.as_str());
        }
    }

    /// A runtime is found by its id, written as the id is; a text that is no
    /// row's id finds none, and the default's read is the info routes'.
    #[test]
    fn a_row_is_found_by_its_runtimes_id() {
        for row in &HTTP_RUNTIMES {
            let found = http_runtime(row.runtime.as_str()).map(|found| found.runtime);
            assert_eq!(found, Some(row.runtime));
            assert!(http_runtime(&row.runtime.as_str().to_uppercase()).is_none());
        }
        assert!(http_runtime("a-runtime-with-no-row").is_none());
        let default = http_runtime(DEFAULT_RUNTIME.as_str()).map(|row| row.metadata);
        assert_eq!(default, Some(MetadataRead::InfoRoutes));
    }

    /// A runtime reports its DP size over HTTP where its row says so, and a
    /// runtime with no row reports none. One row at least says so.
    #[test]
    fn a_runtime_reports_its_dp_size_where_its_row_says_so() {
        for row in &HTTP_RUNTIMES {
            assert_eq!(reports_dp(row.runtime.as_str()), row.reports_dp);
        }
        assert!(!reports_dp("a-runtime-with-no-row"));
        assert!(HTTP_RUNTIMES.iter().any(|row| row.reports_dp));
    }
}
