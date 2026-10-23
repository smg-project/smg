//! A test binary's downloads from the model hub: one at a time per process, a
//! rate-limited answer waited out and sent again, a cached file written whole
//! or not at all. The tokenizer crate's and the gateway's test harnesses carry
//! the same file.

use std::{
    fs,
    path::Path,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

/// How many times a GET the hub rate-limits (HTTP 429) is sent, waiting between
/// tries what the answer says ([`rate_limit_wait`]): a run on a cold cache fetches
/// a few files of several hundred models without a token, more than the hub
/// serves one address in one window.
pub const RATE_LIMITED_TRIES: u32 = 6;
/// The hub's rate-limit window (`ratelimit-policy: "fixed window";...;w=300`):
/// the longest wait between two tries.
pub const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(300);

/// The tests of a binary run in parallel and share the cache directory: one
/// download at a time.
static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());

/// The download lock, taken back from a test that panicked while holding it.
/// The lock guards no state of its own, and a download that did not finish
/// left no file behind ([`cached_download`]), so the next test's download goes
/// ahead instead of failing on the poison with the first test's cause.
pub fn download_lock() -> MutexGuard<'static, ()> {
    DOWNLOAD_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// GET `url`, waiting out the hub's rate limit; the response, whatever its
/// status, after the last try.
pub fn fetch(
    client: &reqwest::blocking::Client,
    url: &str,
) -> Result<reqwest::blocking::Response, String> {
    let mut tries = 0;
    loop {
        let response = client
            .get(url)
            .send()
            .map_err(|e| format!("GET {url}: {e}"))?;
        tries += 1;
        if response.status() != reqwest::StatusCode::TOO_MANY_REQUESTS
            || tries == RATE_LIMITED_TRIES
        {
            return Ok(response);
        }
        std::thread::sleep(rate_limit_wait(response.headers()));
    }
}

/// The wait a rate-limited answer asks for: `Retry-After` in seconds, else the
/// seconds left in the hub's window (`ratelimit: "resolvers";r=0;t=240`) and one
/// more, since `t=` is whole seconds and a try at the boundary is limited again
/// with `t=0`; else ten seconds; a whole window at most.
pub fn rate_limit_wait(headers: &reqwest::header::HeaderMap) -> Duration {
    let seconds = |value: &reqwest::header::HeaderValue| -> Option<u64> {
        value.to_str().ok()?.trim().parse().ok()
    };
    let retry_after = headers.get(reqwest::header::RETRY_AFTER).and_then(seconds);
    let window_left = headers
        .get("ratelimit")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            value
                .split(';')
                .map(str::trim)
                .find_map(|part| part.strip_prefix("t="))
                .and_then(|left| left.parse::<u64>().ok())
                .map(|left| left + 1)
        });
    Duration::from_secs(retry_after.or(window_left).unwrap_or(10)).min(RATE_LIMIT_WINDOW)
}

/// The file at `url` cached at `path`: downloaded under the download lock
/// unless it is there already, written beside it as `<file>.part` and renamed
/// into place, so a download that fails leaves nothing a later test would
/// trust. Panics when the answer past the rate limit is no success, the body
/// is under `min_bytes`, or the cache cannot be written.
#[expect(
    clippy::panic,
    clippy::print_stdout,
    reason = "test helper — diagnostics and loud failures are intentional"
)]
pub fn cached_download(
    client: &reqwest::blocking::Client,
    url: &str,
    path: &Path,
    min_bytes: usize,
) {
    let _guard = download_lock();
    if path.is_file() {
        return;
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)
            .unwrap_or_else(|error| panic!("cannot create {}: {error}", dir.display()));
    }
    println!("Downloading {url}...");
    let response = fetch(client, url).unwrap_or_else(|error| panic!("{error}"));
    let status = response.status();
    assert!(status.is_success(), "GET {url}: HTTP {status}");
    let content = response
        .bytes()
        .unwrap_or_else(|error| panic!("GET {url}: {error}"));
    assert!(
        content.len() >= min_bytes,
        "GET {url}: {} bytes, expected at least {min_bytes}",
        content.len()
    );
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = Path::new(&part);
    fs::write(part, &content)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", part.display()));
    fs::rename(part, path)
        .unwrap_or_else(|error| panic!("cannot rename {} into place: {error}", part.display()));
    println!("Cached {} ({} bytes)", path.display(), content.len());
}
