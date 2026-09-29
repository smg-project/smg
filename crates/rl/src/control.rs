//! Where a worker's RL control routes live. For an HTTP worker that is the
//! worker itself; for a gRPC or ZMQ worker it is the URL the engine advertised
//! (label `rl.control_url`), with a wildcard bind host swapped for the host the
//! gateway already reaches the worker on.

/// Resolve an advertised control URL against the worker's own URL.
///
/// An engine that bound `0.0.0.0` (or `::`) advertises what it bound; the
/// only host the gateway knows is reachable is the worker's own, so use it.
pub fn resolve_control_url(advertised: &str, worker_url: &str) -> String {
    let advertised = advertised.trim().trim_end_matches('/');
    let Some((scheme, rest)) = advertised.split_once("://") else {
        return advertised.to_string();
    };
    let (authority, path) = rest.split_once('/').map_or((rest, ""), |(a, p)| (a, p));
    let (host, port) = split_host_port(authority);
    if !is_wildcard_host(host) {
        return advertised.to_string();
    }
    let worker_host = worker_url
        .split_once("://")
        .map_or(worker_url, |(_, r)| r)
        .split('/')
        .next()
        .map(|authority| split_host_port(authority).0)
        .unwrap_or("");
    let worker_host = worker_host.split('@').next().unwrap_or(worker_host);
    let mut out = format!("{scheme}://{worker_host}");
    if let Some(port) = port {
        out.push(':');
        out.push_str(port);
    }
    if !path.is_empty() {
        out.push('/');
        out.push_str(path);
    }
    out
}

fn is_wildcard_host(host: &str) -> bool {
    matches!(host, "" | "0.0.0.0" | "::" | "[::]")
}

/// `host:port` → (`host`, `Some(port)`), with IPv6 brackets kept on the host.
fn split_host_port(authority: &str) -> (&str, Option<&str>) {
    if let Some(end) = authority
        .strip_prefix('[')
        .and_then(|_| authority.find(']'))
    {
        let host = &authority[..=end];
        let port = authority[end + 1..].strip_prefix(':');
        return (host, port);
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') => (host, Some(port)),
        _ => (authority, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_hosts_pass_through() {
        assert_eq!(
            resolve_control_url("http://10.0.0.5:40100", "grpc://10.0.0.5:30000"),
            "http://10.0.0.5:40100"
        );
        assert_eq!(
            resolve_control_url("http://ctl.internal:8001/", "grpc://a:1"),
            "http://ctl.internal:8001"
        );
    }

    #[test]
    fn wildcard_hosts_take_the_worker_host() {
        assert_eq!(
            resolve_control_url("http://0.0.0.0:40100", "grpc://10.0.0.5:30000"),
            "http://10.0.0.5:40100"
        );
        assert_eq!(
            resolve_control_url("http://[::]:40100", "grpc://[fd00::5]:30000"),
            "http://[fd00::5]:40100"
        );
        assert_eq!(
            resolve_control_url("http://:40100", "grpc://10.0.0.5:30000@2"),
            "http://10.0.0.5:40100",
            "a DP-rank suffix on the worker URL is not part of the host"
        );
    }

    #[test]
    fn schemeless_or_odd_input_is_returned_unchanged() {
        assert_eq!(
            resolve_control_url("10.0.0.5:40100", "grpc://a:1"),
            "10.0.0.5:40100"
        );
    }
}
