//! The keys reloaded from their files while the gateway runs: the serving
//! key, the tenant keys and the control-plane keys. Each file is read again
//! by the rules the start read it by ([`crate::config::secrets`]) and parsed
//! by the same parsers, the serving set that would be in force passes the
//! start's rules, and each set is put in force as one swap. A file that
//! cannot be used, unreadable, unparsable or breaking those rules, leaves
//! every set as it was and is refused by field and path, never by content.
//! A key the command line or the environment gave is not a file: it stays
//! as the start read it, and a reload rebuilds the serving map from that
//! fixed part and the files'. The admin routes' fallback, in force when no
//! control-plane keys are configured, holds the shared key alone and follows
//! it through a reload of its file.

use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use smg_auth::ApiKeySet;
use smg_service_discovery::fs_watch::{directory_of, notify, settle, DirectoryWatch};
use tokio::{sync::mpsc, task, time};
use tracing::{error, info, warn};

use crate::{
    config::{
        redacted::KeysInForce,
        secrets::{
            parse_control_plane_api_key, parse_tenant_api_key, API_KEY_FILE,
            CONTROL_PLANE_API_KEYS_FILE, TENANT_API_KEYS_FILE,
        },
        validation::ConfigValidator,
        ConfigError, TenantApiKeyEntry,
    },
    middleware::AuthConfig,
    server::posture::{SecretSource, SecretSources},
};

/// How often a lost watch (the key files' directory removed or moved away) is
/// tried again.
const REWATCH: Duration = Duration::from_secs(1);

/// The files the keys were read from at start, where a file was the
/// source: a reload's targets. `None` where the key came from the command
/// line or the environment, or was not given.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SecretFiles {
    /// The shared serving key's file.
    pub api_key: Option<PathBuf>,
    /// The tenant keys' file.
    pub tenant_api_keys: Option<PathBuf>,
    /// The control-plane keys' file.
    pub control_plane_api_keys: Option<PathBuf>,
}

impl SecretFiles {
    /// Whether any key has a file.
    pub fn any(&self) -> bool {
        self.count() > 0
    }

    /// How many keys have a file.
    pub fn count(&self) -> usize {
        self.paths().count()
    }

    /// The names of the settings whose keys have a file, in the order a
    /// reload reads the files.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.by_setting()
            .into_iter()
            .filter_map(|(path, name)| path.as_ref().map(|_| name))
    }

    fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.by_setting()
            .into_iter()
            .filter_map(|(path, _)| path.as_ref())
    }

    /// Each key's file beside the name of its setting, in the order a
    /// reload reads the files.
    fn by_setting(&self) -> [(&Option<PathBuf>, &'static str); 3] {
        [
            (&self.api_key, API_KEY_FILE.field),
            (&self.tenant_api_keys, TENANT_API_KEYS_FILE.field),
            (
                &self.control_plane_api_keys,
                CONTROL_PLANE_API_KEYS_FILE.field,
            ),
        ]
    }

    /// The files by directory: what the watch of `--secrets-auto-reload`
    /// watches, and for which names.
    fn by_directory(&self) -> BTreeMap<PathBuf, Vec<OsString>> {
        let mut dirs: BTreeMap<PathBuf, Vec<OsString>> = BTreeMap::new();
        for path in self.paths() {
            if let Some(name) = path.file_name() {
                dirs.entry(directory_of(path).to_path_buf())
                    .or_default()
                    .push(name.to_os_string());
            }
        }
        dirs
    }
}

/// One file a reload read.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FileRead {
    /// The setting the file serves, by the name its refusals use.
    pub setting: &'static str,
    /// The file's path.
    pub path: String,
    /// The entries the file holds; one for the serving key's.
    pub entries: usize,
}

/// What a reload did: the files read, and whether each set changed.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct ReloadReport {
    /// The files read, in the order of the sets.
    pub files: Vec<FileRead>,
    /// Whether the serving keys in force changed: a key added or removed,
    /// or a key moved to another tenant.
    pub serving_changed: bool,
    /// Whether the control-plane keys in force changed.
    pub control_plane_changed: bool,
}

/// A file a reload could not use; every set stands as it was. The error
/// names the field and the path, never the content.
#[derive(Debug)]
pub struct BadFile(pub ConfigError);

/// The reason of a refusal, for the file's own refusal to carry in front of
/// its field and path: a parser's refusal names its entry's field with the
/// value redacted, a rule its reason. A language binding that reads a key
/// file refuses a bad entry the same way.
pub fn entry_refusal(error: ConfigError) -> String {
    match error {
        ConfigError::InvalidValue { reason, .. } | ConfigError::ValidationFailed { reason } => {
            reason
        }
        other => other.to_string(),
    }
}

/// The key sets the authentication layers read, which a reload swaps: the
/// serving set, the control-plane set when control-plane authentication is
/// on, and the admin routes' fallback when no control-plane keys are
/// configured.
pub struct KeySets {
    /// The serving set: the shared key and the tenant keys.
    pub serving: AuthConfig,
    /// The control-plane set in force, when control-plane authentication
    /// is on.
    pub control_plane: Option<ApiKeySet>,
    /// The admin and worker-management routes' fallback when no
    /// control-plane keys are configured: the shared key alone, never a
    /// tenant key.
    pub admin_fallback: Option<AuthConfig>,
}

impl KeySets {
    /// The serving set alone: no control-plane set, no admin fallback.
    pub fn serving_alone(serving: AuthConfig) -> Self {
        Self {
            serving,
            control_plane: None,
            admin_fallback: None,
        }
    }
}

/// The reload of the keys from their files. It holds the handles the
/// authentication layers read, so a swap reaches every request after it,
/// and the serving keys a file did not give, by hash.
pub struct SecretReloader {
    files: SecretFiles,
    serving: AuthConfig,
    /// The admin and worker-management routes' fallback when no control-plane
    /// keys are configured: the shared key alone, never a tenant key. Held
    /// when the shared key came from a file, so a reload of that file moves
    /// it with the serving set.
    admin_fallback: Option<AuthConfig>,
    /// The shared key when the command line or the environment gave it:
    /// every reload keeps it.
    fixed_api_key: Option<String>,
    /// The tenant entries when no file gave them: every reload keeps them.
    fixed_tenants: Vec<TenantApiKeyEntry>,
    control_plane: Option<ApiKeySet>,
    /// The keys in force as the configuration view names them: the start's
    /// until a reload is accepted, then the last accepted reload's; a refused
    /// reload leaves it as it was.
    in_force: RwLock<KeysInForce>,
    /// One reload at a time: two callers would interleave their reads and
    /// swaps.
    turn: Mutex<()>,
}

impl SecretReloader {
    /// The reloader for a start, `None` when no key came from a file. A
    /// file counts when its source was the file (not a value given beside
    /// it) and, for the control-plane keys, when control-plane
    /// authentication is on (`sets.control_plane` is its set in force).
    /// The serving values are the start's; whichever of them a file did not
    /// give is the fixed part every reload keeps. `sets.admin_fallback` is
    /// the admin routes' set when no control-plane keys are configured; it
    /// is a reload's target when the shared key came from a file.
    pub fn new(
        files: SecretFiles,
        sources: &SecretSources,
        api_key: Option<&str>,
        tenant_api_keys: &[TenantApiKeyEntry],
        sets: KeySets,
    ) -> Option<Self> {
        let KeySets {
            serving,
            control_plane,
            admin_fallback,
        } = sets;
        let from_file = |source: Option<SecretSource>| source == Some(SecretSource::File);
        let files = SecretFiles {
            api_key: files.api_key.filter(|_| from_file(sources.api_key)),
            tenant_api_keys: files
                .tenant_api_keys
                .filter(|_| from_file(sources.tenant_api_keys.map(|(source, _)| source))),
            control_plane_api_keys: files.control_plane_api_keys.filter(|_| {
                control_plane.is_some()
                    && from_file(sources.control_plane_api_keys.map(|(source, _)| source))
            }),
        };
        if !files.any() {
            return None;
        }
        let fixed_api_key = api_key
            .filter(|_| files.api_key.is_none())
            .map(str::to_string);
        let fixed_tenants: &[TenantApiKeyEntry] = if files.tenant_api_keys.is_none() {
            tenant_api_keys
        } else {
            &[]
        };
        let control_plane = control_plane.filter(|_| files.control_plane_api_keys.is_some());
        let admin_fallback = admin_fallback.filter(|_| files.api_key.is_some());
        let in_force = KeysInForce::of(
            api_key,
            tenant_api_keys,
            control_plane.as_ref().map(ApiKeySet::len),
        );
        Some(Self {
            files,
            serving,
            admin_fallback,
            fixed_api_key,
            fixed_tenants: fixed_tenants.to_vec(),
            control_plane,
            in_force: RwLock::new(in_force),
            turn: Mutex::new(()),
        })
    }

    /// The files a reload reads.
    pub fn files(&self) -> &SecretFiles {
        &self.files
    }

    /// The keys in force as the configuration view names them: the start's
    /// until a reload is accepted, then the last accepted reload's; a refused
    /// reload leaves it as it was.
    pub(crate) fn keys_in_force(&self) -> KeysInForce {
        self.in_force.read().clone()
    }

    /// Read every file, check the serving set that would be in force by the
    /// start's rules, then put the sets in force: a file that cannot be used
    /// stops the reload before any swap. A request in flight finishes under
    /// the set it took. The sets are the keys the gateway accepts: the shared
    /// key it sends upstream to its workers is the start's until a restart,
    /// so a secret the gateway and its workers share is rotated with a
    /// restart of the gateway.
    pub fn reload(&self) -> Result<ReloadReport, BadFile> {
        let _turn = self.turn.lock();
        let mut report = ReloadReport::default();
        let api_key = match &self.files.api_key {
            Some(path) => {
                let value = API_KEY_FILE.value(path).map_err(BadFile)?;
                report.files.push(FileRead {
                    setting: API_KEY_FILE.field,
                    path: path.display().to_string(),
                    entries: 1,
                });
                Some(value)
            }
            None => None,
        };
        let tenants = match &self.files.tenant_api_keys {
            Some(path) => {
                let entries = TENANT_API_KEYS_FILE
                    .entries(path)
                    .map_err(BadFile)?
                    .iter()
                    .map(|entry| parse_tenant_api_key(entry))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| {
                        BadFile(TENANT_API_KEYS_FILE.bad_file(path, &entry_refusal(error)))
                    })?;
                report.files.push(FileRead {
                    setting: TENANT_API_KEYS_FILE.field,
                    path: path.display().to_string(),
                    entries: entries.len(),
                });
                Some(entries)
            }
            None => None,
        };
        let control_plane = match (&self.files.control_plane_api_keys, &self.control_plane) {
            (Some(path), Some(set)) => {
                let entries = CONTROL_PLANE_API_KEYS_FILE
                    .entries(path)
                    .map_err(BadFile)?
                    .iter()
                    .map(|entry| parse_control_plane_api_key(entry))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| {
                        BadFile(CONTROL_PLANE_API_KEYS_FILE.bad_file(path, &entry_refusal(error)))
                    })?;
                report.files.push(FileRead {
                    setting: CONTROL_PLANE_API_KEYS_FILE.field,
                    path: path.display().to_string(),
                    entries: entries.len(),
                });
                Some((set, entries))
            }
            _ => None,
        };

        // Every file is read. The serving set that would be in force, the
        // files' part and the fixed part together, passes the start's rules
        // before any swap: a tenant file the start would refuse, or a key a
        // file repeats from the fixed part, puts nothing in force.
        let mut served = None;
        if api_key.is_some() || tenants.is_some() {
            let api_key = api_key.or_else(|| self.fixed_api_key.clone());
            let tenants = tenants.as_deref().unwrap_or(&self.fixed_tenants);
            if let Err(error) = ConfigValidator::tenant_credentials(api_key.as_deref(), tenants) {
                let (file, path) = match (&self.files.tenant_api_keys, &self.files.api_key) {
                    (Some(path), _) => (&TENANT_API_KEYS_FILE, path),
                    (None, Some(path)) => (&API_KEY_FILE, path),
                    (None, None) => return Err(BadFile(error)),
                };
                return Err(BadFile(file.bad_file(path, &entry_refusal(error))));
            }
            served = Some(KeysInForce::of(api_key.as_deref(), tenants, None));
            // The admin routes' fallback follows the shared key alone: a
            // tenant key never reaches it. It takes its copy of the key only
            // when it is held, and swaps after the serving set.
            let fallback = self
                .admin_fallback
                .as_ref()
                .map(|fallback| (fallback, api_key.clone()));
            let keys = AuthConfig::key_map(api_key, tenants);
            let previous = self.serving.replace(keys.clone());
            report.serving_changed = *previous != keys;
            if let Some((fallback, api_key)) = fallback {
                fallback.replace(AuthConfig::key_map(api_key, &[]));
            }
        }
        if let Some((set, entries)) = control_plane {
            report.control_plane_changed = set.load().as_slice() != entries.as_slice();
            set.replace(entries);
        }
        // The view's record follows the sets: the serving half when a
        // serving file was read, the control-plane count when its set is held.
        let mut in_force = self.in_force.write();
        if let Some(served) = served {
            in_force.api_key = served.api_key;
            in_force.tenant_api_keys = served.tenant_api_keys;
        }
        if let Some(set) = &self.control_plane {
            in_force.control_plane_entries = Some(set.len());
        }
        Ok(report)
    }
}

/// One reload off the runtime's threads (the files are small, but a mount
/// can stall), its outcome logged under `trigger` (`route` or `watch`): the
/// files read and whether each set changed, or the file that stopped it with
/// every key in force standing. The task that reloads writes that line,
/// under the caller's span: a caller that leaves mid-reload (the route's
/// client gone) drops this future, not the task, and the keys put in force
/// are still said so. `None` when the reload's task did not complete, which
/// is logged as an error.
pub async fn reload_logged(
    reloader: Arc<SecretReloader>,
    trigger: &'static str,
) -> Option<Result<ReloadReport, BadFile>> {
    let span = tracing::Span::current();
    let reloaded = task::spawn_blocking(move || {
        span.in_scope(|| {
            let outcome = reloader.reload();
            match &outcome {
                Ok(report) => info!(
                    trigger,
                    files = report.files.len(),
                    serving_changed = report.serving_changed,
                    control_plane_changed = report.control_plane_changed,
                    "the keys were reloaded from their files"
                ),
                Err(BadFile(error)) => {
                    warn!(trigger, %error, "the keys were not reloaded; the keys in force stand");
                }
            }
            outcome
        })
    })
    .await;
    match reloaded {
        Ok(outcome) => Some(outcome),
        Err(error) => {
            error!(trigger, %error, "the keys' reload did not complete");
            None
        }
    }
}

/// `--secrets-auto-reload`: the directories of the key files the start read
/// are watched (`DirectoryWatch`), and a change runs the reload the route
/// runs, once a burst of changes has settled. The files are read once more
/// as the watch starts: a file rotated between the start's read and the
/// watch raises no event, and would stay out of force until its next
/// rotation. A directory that cannot be watched refuses the start. A
/// directory removed or moved away is said so once and watched again once it
/// exists, with one reload then. The task runs for the lifetime of the
/// server.
pub fn start_auto_reload(reloader: Arc<SecretReloader>) -> notify::Result<task::JoinHandle<()>> {
    let (wake, woken) = mpsc::channel(1);
    let mut watches = Vec::new();
    for (dir, names) in reloader.files.by_directory() {
        let watch = DirectoryWatch::start(&dir, names.clone(), wake.clone())
            .map_err(|error| watch_refused(&dir, error))?;
        watches.push(Watched {
            dir,
            names,
            watch: Some(watch),
        });
    }
    info!(
        files = reloader.files.count(),
        directories = watches.len(),
        "watching the key files; a change reloads the keys"
    );
    // What changed between the start's read of the files and the watch is
    // read now, as after a lost directory returns: no event reports it.
    let _ = wake.try_send(());
    #[expect(
        clippy::disallowed_methods,
        reason = "the key-file watch runs for the lifetime of the server"
    )]
    let handle = task::spawn(watch_loop(reloader, wake, woken, watches));
    Ok(handle)
}

/// The start's refusal for a key directory that cannot be watched: the flag,
/// the directory as an absolute path, the cause, and the inotify limit behind
/// the cause when one is known. The watch crate reports the watch limit as
/// its own kind (its reading of `ENOSPC` from `add_watch`, the directory
/// attached) and the instance limit as the `EMFILE` that refused the inotify
/// handle (nothing attached); neither names the flag or the limit to raise.
fn watch_refused(dir: &Path, error: notify::Error) -> notify::Error {
    let limit = inotify_limit(&error.kind);
    // The kind alone: a path the error carries is the directory itself.
    let cause = notify::Error::new(error.kind);
    let shown = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    notify::Error::generic(&format!(
        "--secrets-auto-reload: cannot watch {}: {cause}{limit}",
        shown.display()
    ))
}

/// The limit to raise for an error kind an inotify limit surfaces as, or
/// nothing: the watch limit is the crate's own kind; the instance limit is
/// `EMFILE` ("too many open files"), which a process at its open-file limit
/// gets as well, so both are named. The errno is Linux's, and only Linux has
/// an inotify limit to raise: another target carries no hint.
fn inotify_limit(kind: &notify::ErrorKind) -> &'static str {
    #[cfg(target_os = "linux")]
    {
        const EMFILE: i32 = 24;
        match kind {
            notify::ErrorKind::MaxFilesWatch => " (raise fs.inotify.max_user_watches)",
            notify::ErrorKind::Io(error) if error.raw_os_error() == Some(EMFILE) => {
                " (raise fs.inotify.max_user_instances, or the process's open-file limit)"
            }
            _ => "",
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = kind;
        ""
    }
}

/// One watched directory of key files; `watch` is `None` while the directory
/// is gone and the watch waits to be started again.
struct Watched {
    dir: PathBuf,
    names: Vec<OsString>,
    watch: Option<DirectoryWatch>,
}

async fn watch_loop(
    reloader: Arc<SecretReloader>,
    wake: mpsc::Sender<()>,
    mut woken: mpsc::Receiver<()>,
    mut watches: Vec<Watched>,
) {
    let mut rewatch = time::interval(REWATCH);
    rewatch.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            Some(()) = woken.recv() => {
                settle(&mut woken).await;
                for watched in &mut watches {
                    if watched.watch.as_ref().is_some_and(DirectoryWatch::is_lost) {
                        warn!(
                            directory = %watched.dir.display(),
                            "the key files' directory was removed or moved; watching it again once it exists"
                        );
                        watched.watch = None;
                    }
                }
                let _ = reload_logged(Arc::clone(&reloader), "watch").await;
            }
            _ = rewatch.tick(), if watches.iter().any(|watched| watched.watch.is_none()) => {
                for watched in &mut watches {
                    if watched.watch.is_some() {
                        continue;
                    }
                    if let Ok(watch) = DirectoryWatch::start(&watched.dir, watched.names.clone(), wake.clone()) {
                        info!(directory = %watched.dir.display(), "watching the key files' directory again");
                        watched.watch = Some(watch);
                        // What changed while the directory was gone is read now.
                        let _ = wake.try_send(());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/secrets/tests.rs"]
mod tests;
