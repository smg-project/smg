//! The registered workers an address names.

use std::sync::Arc;

use crate::{
    identity::{basic::Worker, endpoint::Endpoint},
    registry::{Generation, WorkerRegistry},
};

/// Find the registered workers a query address selects.
///
/// Matching is on [`EndpointKey`](crate::identity::endpoint::EndpointKey), so
/// every spelling of one backend — bare `host:port`, any scheme, either IPv6
/// form — selects the same group. Two narrowings apply on top of the key:
///
/// - A query naming a scheme selects only registrations under that scheme, so
///   `http://h:p` leaves a `grpc://h:p` sibling alone. A scheme-less query, as
///   discovery and the reconciler submit, selects every scheme.
/// - A query carrying a `@rank` suffix selects only that rank. Without one, a
///   base address selects its plain registration and every expanded rank.
///
/// On a network address a trailing `@digits` is always a rank: the URL check
/// refuses the suffix on a posted address. A socket path may end the same way
/// by itself. There a query that writes a suffix names one path and selects
/// only the worker whose URL writes it the same way: `w@0` and `w@00` are two
/// sockets. A query without a suffix selects the worker at that path and the
/// workers that carry the data-parallel rank their URL ends in.
///
/// The group is read from what is registered, never from the host's global
/// DP setting: a rank exists in the registry or it does not, and a removal
/// must find the same set either way.
///
/// An address neither side can parse selects nothing rather than falling back
/// to a string comparison that could match the wrong backend.
pub fn find_workers_by_url<G: Generation>(
    registry: &WorkerRegistry<G>,
    url: &str,
) -> Vec<Arc<dyn Worker>> {
    /// What follows an address's last `@`, as it is written: a rank reads
    /// `@0` and `@00` alike, a socket path does not.
    fn suffix_as_written(url: &str) -> Option<&str> {
        url.trim().rsplit_once('@').map(|(_, suffix)| suffix)
    }

    let Ok((query, query_rank)) = Endpoint::parse_with_rank(url) else {
        return Vec::new();
    };
    let query_key = query.key();
    registry
        .get_all()
        .into_iter()
        .filter(|worker| {
            let Ok((registered, registered_rank)) = Endpoint::parse_with_rank(worker.url()) else {
                return false;
            };
            if registered.key() != query_key {
                return false;
            }
            if query.scheme().is_some() && query.scheme() != registered.scheme() {
                return false;
            }
            if registered.is_ipc() {
                return match query_rank {
                    Some(_) => suffix_as_written(url) == suffix_as_written(worker.url()),
                    None => registered_rank.is_none() || registered_rank == worker.dp_rank(),
                };
            }
            query_rank.is_none() || query_rank == registered_rank
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{BasicWorkerBuilder, WorkerRegistry};

    fn register(registry: &WorkerRegistry, base: &str, rank: Option<usize>) {
        let builder = BasicWorkerBuilder::new(base);
        let builder = match rank {
            Some(rank) => builder.dp_config(rank, 2),
            None => builder,
        };
        registry.register(Arc::new(builder.build())).unwrap();
    }

    fn urls(registry: &WorkerRegistry, url: &str) -> Vec<String> {
        let mut urls: Vec<_> = find_workers_by_url(registry, url)
            .iter()
            .map(|worker| worker.url().to_string())
            .collect();
        urls.sort_unstable();
        urls
    }

    /// Two registrations spelling one IPv6 address differently are one
    /// backend, so either spelling now selects both. Exact string matching
    /// returned only the spelling that was asked for.
    ///
    /// This matters beyond lookup: `DELETE`/`PUT /workers/{id}` resolve an id
    /// to a URL and come back through here, so an id naming one of these
    /// selects both. Resolving those by id instead would keep the caller's
    /// choice intact — see the worker-removal identity follow-up.
    #[test]
    fn one_address_spelled_two_ways_selects_both_registrations() {
        let registry = WorkerRegistry::new();
        register(&registry, "http://[::1]:8080", None);
        register(&registry, "http://[0:0:0:0:0:0:0:1]:8080", None);

        let both = vec!["http://[0:0:0:0:0:0:0:1]:8080", "http://[::1]:8080"];
        assert_eq!(urls(&registry, "http://[::1]:8080"), both);
        assert_eq!(urls(&registry, "http://[0:0:0:0:0:0:0:1]:8080"), both);
    }

    /// `validate_worker_url` accepts a path prefix and `endpoint_url` appends
    /// the route to it, so workers really are registered behind one. Rejecting
    /// those addresses made them unfindable: `DELETE`/`PATCH` failed with
    /// "not found", and health-check removal took the stale-removal branch and
    /// reported success without removing anything.
    #[test]
    fn a_path_prefixed_registration_is_still_found() {
        let registry = WorkerRegistry::new();
        register(&registry, "http://proxy:8080/sglang", None);
        register(&registry, "http://proxy:8080/other", None);
        register(&registry, "http://proxy:8080", None);

        assert_eq!(
            urls(&registry, "http://proxy:8080/sglang"),
            vec!["http://proxy:8080/sglang"]
        );
        // The path is part of the backend, so siblings are left alone.
        assert_eq!(
            urls(&registry, "http://proxy:8080"),
            vec!["http://proxy:8080"]
        );
        // And a scheme-less query still reaches it.
        assert_eq!(
            urls(&registry, "proxy:8080/sglang"),
            vec!["http://proxy:8080/sglang"]
        );
    }

    /// A DP group behind a path prefix resolves as a group.
    #[test]
    fn a_path_prefixed_dp_group_resolves_together() {
        let registry = WorkerRegistry::new();
        let base = "http://proxy:8080/sglang";
        register(&registry, base, Some(0));
        register(&registry, base, Some(1));

        assert_eq!(
            urls(&registry, base),
            vec!["http://proxy:8080/sglang@0", "http://proxy:8080/sglang@1"]
        );
        assert_eq!(
            urls(&registry, "http://proxy:8080/sglang@1"),
            vec!["http://proxy:8080/sglang@1"]
        );
    }

    #[test]
    fn base_url_matches_plain_and_expanded_registrations() {
        for scheme in ["http", "https", "grpc", "grpcs", "ipc", "tcp"] {
            let registry = WorkerRegistry::new();
            let base = format!("{scheme}://worker:3000");
            register(&registry, &base, None);
            register(&registry, &base, Some(0));
            register(&registry, &base, Some(1));
            register(&registry, &format!("{base}0"), None);
            register(&registry, &format!("{base}/other"), Some(0));

            let expected = vec![base.clone(), format!("{base}@0"), format!("{base}@1")];
            assert_eq!(urls(&registry, &base), expected);
            assert_eq!(urls(&registry, "worker:3000"), expected);
        }
    }

    #[test]
    fn rank_url_selects_only_that_registered_rank() {
        let registry = WorkerRegistry::new();
        register(&registry, "http://worker:3000", None);
        register(&registry, "http://worker:3000", Some(0));
        register(&registry, "http://worker:3000", Some(1));
        for query in ["http://worker:3000@1", "worker:3000@1"] {
            assert_eq!(urls(&registry, query), vec!["http://worker:3000@1"]);
        }
        assert!(urls(&registry, "worker:3000@2").is_empty());
    }

    #[test]
    fn explicit_scheme_preserves_other_protocol_registrations() {
        let registry = WorkerRegistry::new();
        register(&registry, "http://worker:3000", None);
        register(&registry, "grpc://worker:3000", Some(0));
        register(&registry, "https://worker:3000", Some(0));
        assert_eq!(
            urls(&registry, "http://worker:3000"),
            vec!["http://worker:3000"]
        );
        assert_eq!(
            urls(&registry, "worker:3000"),
            vec![
                "grpc://worker:3000@0",
                "http://worker:3000",
                "https://worker:3000@0",
            ]
        );
    }

    /// A socket path may itself end in `@` and digits. On a worker that
    /// carries no data-parallel rank those characters are the path's own:
    /// the path without them is another socket, a path that writes the
    /// digits another way is a third, and none of them selects another
    /// worker, a real rank included. A real rank still leaves with its base.
    #[test]
    fn a_socket_path_that_ends_like_a_rank_is_its_own_address() {
        let registry = WorkerRegistry::new();
        register(&registry, "ipc:///tmp/w", None);
        register(&registry, "ipc:///tmp/w@0", None);
        register(&registry, "ipc:///tmp/w@00", None);
        register(&registry, "ipc:///tmp/g", Some(0));
        register(&registry, "ipc:///tmp/g", Some(1));
        register(&registry, "ipc:///tmp/g@00", None);
        // A rank on a network address is selected by its base with or
        // without its data-parallel metadata.
        register(&registry, "grpc://10.0.0.7:8000@1", None);

        for (query, selected) in [
            ("ipc:///tmp/w", vec!["ipc:///tmp/w"]),
            ("ipc:///tmp/w@0", vec!["ipc:///tmp/w@0"]),
            ("ipc:///tmp/w@00", vec!["ipc:///tmp/w@00"]),
            ("ipc:///tmp/g", vec!["ipc:///tmp/g@0", "ipc:///tmp/g@1"]),
            ("ipc:///tmp/g@1", vec!["ipc:///tmp/g@1"]),
            ("ipc:///tmp/g@0", vec!["ipc:///tmp/g@0"]),
            ("ipc:///tmp/g@00", vec!["ipc:///tmp/g@00"]),
            ("grpc://10.0.0.7:8000", vec!["grpc://10.0.0.7:8000@1"]),
        ] {
            assert_eq!(urls(&registry, query), selected, "{query}");
        }
    }

    #[test]
    fn rank_matching_uses_metadata_not_an_at_sign_prefix() {
        let registry = WorkerRegistry::new();
        register(&registry, "http://user@worker:3000", Some(0));
        register(&registry, "ipc:///tmp/worker@socket", None);
        assert!(urls(&registry, "http://user").is_empty());
        assert!(urls(&registry, "ipc:///tmp/worker").is_empty());
        assert_eq!(
            urls(&registry, "http://user@worker:3000"),
            vec!["http://user@worker:3000@0"]
        );
        assert_eq!(
            urls(&registry, "ipc:///tmp/worker@socket"),
            vec!["ipc:///tmp/worker@socket"]
        );
    }
}
