//! A test binary's downloads from the model hub: one at a time per process, a
//! rate-limited answer waited out and sent again, a GET that broke below HTTP
//! sent again after a short wait, a cached file written whole or not at all.
//! The tokenizer crate's and the gateway's test harnesses carry the same file.

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
/// How many times a GET that broke below HTTP (no connection, a timeout, the
/// request or the body cut short: `error sending request`, `error decoding
/// response body`) is sent again: a run on a cold cache opens thousands of
/// connections, and one of them breaks now and then.
pub const TRANSPORT_ERROR_RETRIES: u32 = 3;
/// The wait before the first of those, doubled before each later one
/// (1, 2, 4 s): seven seconds of waiting at most for a GET that never lands.
pub const TRANSPORT_ERROR_WAIT: Duration = Duration::from_secs(1);

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

/// The hub's answer to a GET, read whole: its status and its body.
#[derive(Debug)]
pub struct Answer {
    pub status: reqwest::StatusCode,
    pub body: Vec<u8>,
}

/// One GET: the headers of a rate-limited answer, else the answer read whole.
enum Attempt {
    Limited(reqwest::header::HeaderMap),
    Answered(Answer),
}

fn attempt(client: &reqwest::blocking::Client, url: &str) -> reqwest::Result<Attempt> {
    let response = client.get(url).send()?;
    let status = response.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Ok(Attempt::Limited(response.headers().clone()));
    }
    let body = response.bytes()?.into();
    Ok(Attempt::Answered(Answer { status, body }))
}

/// Whether a GET failed below HTTP: no connection, a timeout, the request or
/// the body cut short. A malformed URL or a redirect loop is not sent again.
/// Every such error is sent again; [`answered`] tells a caller afterwards
/// whether the hub ever answered.
fn is_transport_error(error: &reqwest::Error) -> bool {
    error.is_connect()
        || error.is_timeout()
        || error.is_request()
        || error.is_body()
        || error.is_decode()
}

/// Whether the hub answered the GET and the body broke under the read (the
/// client's `error decoding response body`), as opposed to no answer at all
/// (no connection, a timeout, the request cut short): a caller that skips
/// when offline keeps failing loudly on a body the hub could not serve whole.
pub fn answered(error: &reqwest::Error) -> bool {
    error.is_decode() || error.is_body()
}

/// GET `url` and read the answer whole, waiting out the hub's rate limit and
/// sending a GET that broke below HTTP again after a short wait (said on
/// stdout). The answer, whatever its status, after the last try: the limit
/// itself, with an empty body, when every try was limited; when every try
/// broke, the first error the hub answered with (a body cut under the read,
/// see [`answered`]) if there was one, else the last try's error, so a caller
/// reads whether the hub ever answered, not how the last try failed.
#[expect(
    clippy::print_stdout,
    reason = "test helper — a GET sent again is said in the log"
)]
pub fn fetch(client: &reqwest::blocking::Client, url: &str) -> reqwest::Result<Answer> {
    let mut limited = 0;
    let mut broken = 0;
    let mut wait = TRANSPORT_ERROR_WAIT;
    let mut answered_error: Option<reqwest::Error> = None;
    loop {
        match attempt(client, url) {
            Ok(Attempt::Answered(answer)) => return Ok(answer),
            Ok(Attempt::Limited(headers)) => {
                limited += 1;
                if limited == RATE_LIMITED_TRIES {
                    return Ok(Answer {
                        status: reqwest::StatusCode::TOO_MANY_REQUESTS,
                        body: Vec::new(),
                    });
                }
                std::thread::sleep(rate_limit_wait(&headers));
            }
            Err(error) if is_transport_error(&error) && broken < TRANSPORT_ERROR_RETRIES => {
                broken += 1;
                println!(
                    "GET {url}: {error}; sent again in {wait:?} ({broken} of {TRANSPORT_ERROR_RETRIES})"
                );
                if answered_error.is_none() && answered(&error) {
                    answered_error = Some(error);
                }
                std::thread::sleep(wait);
                wait *= 2;
            }
            Err(error) => return Err(answered_error.unwrap_or(error)),
        }
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
/// unless it is there already, written beside it as `<file>.<pid>.part` and renamed
/// into place, so a download that fails leaves nothing a later test would
/// trust. Panics when the answer past the rate limit and the retries is no
/// success, the body is under `min_bytes`, or the cache cannot be written.
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
    let answer = fetch(client, url).unwrap_or_else(|error| panic!("GET {url}: {error}"));
    assert!(
        answer.status.is_success(),
        "GET {url}: HTTP {}",
        answer.status
    );
    let content = answer.body;
    assert!(
        content.len() >= min_bytes,
        "GET {url}: {} bytes, expected at least {min_bytes}",
        content.len()
    );
    // Named for this process, as the fixture loaders name theirs: a second
    // process fetching the same file into the same directory renames its own
    // whole file into place.
    let mut part = path.as_os_str().to_owned();
    part.push(format!(".{}.part", std::process::id()));
    let part = Path::new(&part);
    fs::write(part, &content)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", part.display()));
    fs::rename(part, path)
        .unwrap_or_else(|error| panic!("cannot rename {} into place: {error}", part.display()));
    println!("Cached {} ({} bytes)", path.display(), content.len());
}
