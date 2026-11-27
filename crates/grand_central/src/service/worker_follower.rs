//! A gateway's worker follower: one task that follows the worker list of a
//! standalone controller, over that controller's worker watch, into a
//! [`MemoryStore`] and its owner's worker objects. It is no part of the
//! election, where a follower is a replica that does not lead.
//!
//! The task keeps the store as the watch tells it ([`Watcher`]). A watch may
//! reopen or a connection may fail. The same watcher keeps its place, and
//! the store keeps its last fleet until a new snapshot replaces it.
//!
//! The watcher writes through the holder of the owner's worker objects
//! ([`Objects`]), so the object of each row stands when the event that
//! told it returns. The owner follows nothing behind the watch.
//!
//! After each event the store takes at a place, the task calls the
//! function its owner handed it at the start: the task's word that the
//! fleet stands as of a place, for the owner's readiness. The task
//! reports a fleet only after the first such call.
//!
//! The watch and a decision travel apart. A decision the controller
//! answers can name a row whose event the watch has not told yet. The
//! owner passes that target over and asks again.

use std::{io, sync::Arc, time::Duration};

use smg_worker::registry::Generation;
use tokio::{
    sync::{oneshot, watch},
    task::JoinHandle,
    time::sleep,
};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity};
use tracing::warn;

use crate::{
    service::{
        client::{Client, Error, ANSWER_WAIT, KEEPALIVE},
        watcher::{WatchStreamEnd, Watcher},
    },
    store::MemoryStore,
    worker_host::Objects,
};

const FIRST_RETRY: Duration = Duration::from_millis(250);
const MAX_RETRY: Duration = Duration::from_secs(5);

/// A PEM file of the follower's TLS dial.
///
/// The record gains fields: code outside this crate makes one by
/// [`PemFile::of`], and a pattern of it ends with `..`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PemFile {
    /// The name its user set the file by, as a flag or a configuration key.
    /// A read that fails says this name.
    pub setting: &'static str,
    /// The file's path.
    pub path: String,
}

impl PemFile {
    /// The file at `path`, which its user set by `setting`. It takes both
    /// fields: neither has a neutral value. It checks nothing: the dial
    /// reads the file. Code outside this crate starts here: the record
    /// gains fields, so no literal of it builds there.
    #[must_use]
    pub fn of(setting: &'static str, path: String) -> Self {
        Self { setting, path }
    }
}

/// How a follower reaches its controller.
///
/// The record gains fields: code outside this crate makes one by
/// [`WorkerFollowerSettings::of`], and a pattern of it ends with `..`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WorkerFollowerSettings {
    /// The controller's address, as `host:port`.
    pub address: String,
    /// TLS to the controller. Without it the dial is plain.
    pub tls: Option<Tls>,
    /// The name the gateway states to the controller. The controller holds
    /// one session under a name: a newer session replaces the one before
    /// it.
    pub gateway_id: String,
}

/// The TLS files of a follower's dial.
///
/// The record gains fields: code outside this crate makes one by
/// [`Tls::of`], and a pattern of it ends with `..`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Tls {
    /// The certificate authority the controller's certificate is checked
    /// against: the one root the dial trusts.
    pub ca_cert: PemFile,
    /// The gateway's own certificate and its key, sent where they are set.
    pub identity: Option<ClientIdentity>,
}

impl Tls {
    /// The TLS of a dial that trusts `ca_cert` alone and sends `identity`
    /// where one is set. It takes both fields: the root is the guard, and
    /// an identity left out by silence would be a dial that presents no
    /// certificate. It checks nothing: the dial reads the files. Code
    /// outside this crate starts here: the record gains fields, so no
    /// literal of it builds there.
    #[must_use]
    pub fn of(ca_cert: PemFile, identity: Option<ClientIdentity>) -> Self {
        Self { ca_cert, identity }
    }
}

/// A gateway's own certificate and the key of it.
///
/// The record gains fields: code outside this crate makes one by
/// [`ClientIdentity::of`], and a pattern of it ends with `..`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ClientIdentity {
    /// The certificate.
    pub cert: PemFile,
    /// The certificate's key.
    pub key: PemFile,
}

impl ClientIdentity {
    /// The identity of `cert` and its `key`. It takes both fields: a
    /// certificate without its key is no identity. It checks nothing: the
    /// dial reads the files. Code outside this crate starts here: the
    /// record gains fields, so no literal of it builds there.
    #[must_use]
    pub fn of(cert: PemFile, key: PemFile) -> Self {
        Self { cert, key }
    }
}

/// Why a dial of the controller failed.
#[derive(Debug, thiserror::Error)]
enum DialError {
    /// A TLS file could not be read.
    #[error("reading {setting} {path:?}")]
    File {
        setting: &'static str,
        path: String,
        #[source]
        source: io::Error,
    },
    /// The address or the TLS settings make no endpoint.
    #[error(transparent)]
    Endpoint(#[from] tonic::transport::Error),
    /// The client's own error: the dial, the session or the watch.
    #[error(transparent)]
    Client(#[from] Error),
}

/// `error` and each of its sources on one line, a colon between them: the
/// form the retry's warning shows.
fn with_sources(error: &dyn std::error::Error) -> String {
    let mut line = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        line.push_str(": ");
        line.push_str(&cause.to_string());
        source = cause.source();
    }
    line
}

async fn read(file: &PemFile) -> Result<Vec<u8>, DialError> {
    tokio::fs::read(&file.path)
        .await
        .map_err(|source| DialError::File {
            setting: file.setting,
            path: file.path.clone(),
            source,
        })
}

impl WorkerFollowerSettings {
    /// The settings of a follower that dials `address` under `tls` and
    /// states `gateway_id`. It takes all three fields: an address and a
    /// name have no neutral value, and a TLS left out by silence would be
    /// a plain dial. It checks nothing: the dial does. Code outside this
    /// crate starts here: the record gains fields, so no literal of it
    /// builds there.
    #[must_use]
    pub fn of(address: String, tls: Option<Tls>, gateway_id: String) -> Self {
        Self {
            address,
            tls,
            gateway_id,
        }
    }

    async fn connect(&self, address: &str) -> Result<Client, DialError> {
        let Some(files) = &self.tls else {
            return Ok(Client::connect(&format!("http://{address}"), &self.gateway_id).await?);
        };

        let authority = Certificate::from_pem(read(&files.ca_cert).await?);
        let mut tls = ClientTlsConfig::new().ca_certificate(authority);
        if let Some(identity) = &files.identity {
            let cert = read(&identity.cert).await?;
            let key = read(&identity.key).await?;
            tls = tls.identity(Identity::from_pem(cert, key));
        }
        // The watch's next event waits with no bound, so the channel is kept
        // alive as the client's own dial keeps it: a controller gone silent
        // ends the wait.
        let endpoint = Endpoint::from_shared(format!("https://{address}"))?
            .http2_keep_alive_interval(KEEPALIVE)
            .keep_alive_timeout(ANSWER_WAIT)
            .keep_alive_while_idle(true);
        if let Some(host) = endpoint.uri().host() {
            tls = tls.domain_name(host.trim_matches(['[', ']']));
        }
        Ok(Client::connect_with(endpoint.tls_config(tls)?, &self.gateway_id).await?)
    }
}

/// The gateway's side of a controller's watch: one task that keeps a store
/// as the controller tells it.
pub struct WorkerFollower {
    task: JoinHandle<()>,
    stop: oneshot::Sender<()>,
    /// Whether a first snapshot was taken and its owner called after it.
    first_snapshot: watch::Receiver<bool>,
}

impl WorkerFollower {
    /// Starts the task. The watch alone writes `store`'s rows from here,
    /// through `objects`, the holder of the owner's worker objects over
    /// that store: the object of a row stands when its event returns. The
    /// task calls `after_event` after each event the store takes at a
    /// place. The objects stand by then, so the call lays nothing: it is
    /// the word the owner's readiness waits for.
    ///
    /// It spawns the task, so it is called inside a Tokio runtime. Dropping
    /// the follower ends the task too, without the wait [`Self::stop`]
    /// makes.
    pub fn start<G: Generation>(
        settings: WorkerFollowerSettings,
        store: Arc<MemoryStore>,
        objects: Arc<Objects<G>>,
        after_event: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let (stop, stopped) = oneshot::channel();
        let (progress, first_snapshot) = watch::channel(false);
        #[expect(
            clippy::disallowed_methods,
            reason = "shutdown signals and joins the follower task"
        )]
        let task = tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = stopped => {}
                () = run(settings, store, objects, after_event, progress) => {}
            }
        });
        Self {
            task,
            stop,
            first_snapshot,
        }
    }

    /// Whether a first snapshot was taken: the store holds the workers as of
    /// a place, and the owner was called after it. It stays true once it is,
    /// whatever the watch does after.
    pub fn has_first_snapshot(&self) -> bool {
        *self.first_snapshot.borrow()
    }

    /// Ends the task and waits for it.
    pub async fn stop(self) {
        let _ = self.stop.send(());
        if let Err(error) = self.task.await {
            warn!(%error, "Worker follower task ended");
        }
    }
}

fn next_delay(delay: Duration) -> Duration {
    (delay * 2).min(MAX_RETRY)
}

/// The leaders followed at once, one after another, before the watch holds a
/// place again. Two replicas that name each other would else be dialed in a
/// loop with no wait.
const REDIRECTS: u32 = 2;

/// The leader `error` names, where the next dial goes to it at once: a
/// replica that does not lead named one, and fewer than [`REDIRECTS`] were
/// followed since the watch last held a place.
fn named_leader(error: &DialError, followed: u32) -> Option<&str> {
    let DialError::Client(Error::NotLeader(leader)) = error else {
        return None;
    };
    (!leader.is_empty() && followed < REDIRECTS).then_some(leader.as_str())
}

async fn run<G: Generation>(
    settings: WorkerFollowerSettings,
    store: Arc<MemoryStore>,
    objects: Arc<Objects<G>>,
    after_event: impl Fn(),
    progress: watch::Sender<bool>,
) {
    let mut watcher = Watcher::new(store, objects);
    let mut delay = FIRST_RETRY;
    let mut redirect: Option<String> = None;
    // The leaders followed at once since the watch last held a place.
    let mut followed = 0;
    loop {
        let address = redirect.as_deref().unwrap_or(&settings.address);
        let error = match settings.connect(address).await {
            Ok(mut client) => follow(
                &mut client,
                &mut watcher,
                &after_event,
                &progress,
                &mut delay,
                &mut followed,
            )
            .await
            .into(),
            Err(error) => error,
        };
        if let Some(leader) = named_leader(&error, followed) {
            followed += 1;
            redirect = Some(leader.to_owned());
            continue;
        }
        warn!(address, error = %with_sources(&error), ?delay, "Retrying the controller's worker watch");
        sleep(delay).await;
        delay = next_delay(delay);
        redirect = None;
        followed = 0;
    }
}

async fn follow<G: Generation>(
    client: &mut Client,
    watcher: &mut Watcher<G>,
    after_event: &impl Fn(),
    progress: &watch::Sender<bool>,
    delay: &mut Duration,
    followed: &mut u32,
) -> Error {
    loop {
        let mut following = match watcher.open(client).await {
            Ok(following) => following,
            Err(error) => return error,
        };
        // Whether this stream brought the watcher to a place. One that ends
        // before it does is opened again after the wait, so a snapshot that
        // cannot land does not spin.
        let mut placed = false;
        loop {
            match following.next().await {
                Ok(true) => {}
                // The controller ended the stream with no error: the next
                // one goes on after the place.
                Ok(false) => break,
                Err(WatchStreamEnd::Lost(lost)) => {
                    warn!(?lost, "Lost the controller's worker watch position");
                    break;
                }
                Err(WatchStreamEnd::Client(error)) => return error,
            }
            if following.watcher().at().is_some() {
                // The store and the worker objects hold the fleet as of a
                // place. The owner is told, and only then does readiness
                // hear of a fleet.
                after_event();
                placed = true;
                *delay = FIRST_RETRY;
                *followed = 0;
                if !*progress.borrow() {
                    progress.send_replace(true);
                }
            }
        }
        if !placed {
            sleep(*delay).await;
            *delay = next_delay(*delay);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/service/worker_follower/tests.rs"]
mod tests;
