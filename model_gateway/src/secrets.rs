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
    fs_watch::{directory_of, settle, DirectoryWatch},
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
/// every key in force standing. `None` when the reload's task did not
/// complete, which is logged as an error.
pub async fn reload_logged(
    reloader: Arc<SecretReloader>,
    trigger: &'static str,
) -> Option<Result<ReloadReport, BadFile>> {
    match task::spawn_blocking(move || reloader.reload()).await {
        Ok(Ok(report)) => {
            info!(
                trigger,
                files = report.files.len(),
                serving_changed = report.serving_changed,
                control_plane_changed = report.control_plane_changed,
                "the keys were reloaded from their files"
            );
            Some(Ok(report))
        }
        Ok(Err(BadFile(error))) => {
            warn!(trigger, %error, "the keys were not reloaded; the keys in force stand");
            Some(Err(BadFile(error)))
        }
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
mod tests {
    use smg_auth::{ApiKeyEntry, Role};

    use super::*;

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).expect("the file written");
        path
    }

    fn sources(api_key: SecretSource, tenants: Option<SecretSource>) -> SecretSources {
        SecretSources {
            api_key: Some(api_key),
            tenant_api_keys: tenants.map(|source| (source, 1)),
            control_plane_api_keys: Some((SecretSource::File, 1)),
            ..SecretSources::default()
        }
    }

    #[test]
    fn a_reload_puts_the_files_keys_in_force_and_keeps_the_flags_key() {
        let dir = tempfile::tempdir().expect("a directory");
        let tenants = write(dir.path(), "tenants", "acme:t-one\n");
        let serving = AuthConfig::with_tenant_keys(
            Some("flag-key".into()),
            &[TenantApiKeyEntry {
                tenant_id: "acme".into(),
                key: "t-one".into(),
            }],
        );
        let reloader = SecretReloader::new(
            SecretFiles {
                tenant_api_keys: Some(tenants.clone()),
                ..SecretFiles::default()
            },
            &sources(SecretSource::Flag, Some(SecretSource::File)),
            Some("flag-key"),
            &[TenantApiKeyEntry {
                tenant_id: "acme".into(),
                key: "t-one".into(),
            }],
            KeySets::serving_alone(serving.clone()),
        )
        .expect("a file source");

        write(dir.path(), "tenants", "acme:t-two\nbeta:t-three\n");
        let report = reloader.reload().expect("reloaded");
        assert_eq!(report.files.len(), 1);
        assert_eq!(
            (report.files[0].setting, report.files[0].entries),
            ("tenant-api-keys-file", 2)
        );
        assert!(report.serving_changed);
        assert!(serving.contains_token("flag-key"), "the flag's key is kept");
        assert!(serving.contains_token("t-two") && serving.contains_token("t-three"));
        assert!(!serving.contains_token("t-one"), "the rotated key is out");

        let again = reloader.reload().expect("reloaded");
        assert!(!again.serving_changed, "the same file changes nothing");
    }

    #[test]
    fn a_key_moved_to_another_tenant_changes_the_serving_set() {
        let dir = tempfile::tempdir().expect("a directory");
        let tenants = write(dir.path(), "tenants", "acme:t-one\nbeta:t-two\n");
        let start = [
            TenantApiKeyEntry {
                tenant_id: "acme".into(),
                key: "t-one".into(),
            },
            TenantApiKeyEntry {
                tenant_id: "beta".into(),
                key: "t-two".into(),
            },
        ];
        let serving = AuthConfig::with_tenant_keys(Some("flag-key".into()), &start);
        let reloader = SecretReloader::new(
            SecretFiles {
                tenant_api_keys: Some(tenants),
                ..SecretFiles::default()
            },
            &sources(SecretSource::Flag, Some(SecretSource::File)),
            Some("flag-key"),
            &start,
            KeySets::serving_alone(serving.clone()),
        )
        .expect("a file source");
        assert!(
            !reloader.reload().expect("reloaded").serving_changed,
            "the file as the start read it changes nothing"
        );

        // The same two keys, each under the other tenant: the hashes are
        // the start's, the identities behind them are not.
        write(dir.path(), "tenants", "beta:t-one\nacme:t-two\n");
        let report = reloader.reload().expect("reloaded");
        assert!(
            report.serving_changed,
            "a key moved to another tenant is a change"
        );
        assert!(serving.contains_token("t-one") && serving.contains_token("t-two"));
    }

    #[test]
    fn an_api_key_file_rotates_the_shared_key() {
        let dir = tempfile::tempdir().expect("a directory");
        let file = write(dir.path(), "key", "k-one\n");
        let serving = AuthConfig::new(Some("k-one".into()));
        let reloader = SecretReloader::new(
            SecretFiles {
                api_key: Some(file),
                ..SecretFiles::default()
            },
            &sources(SecretSource::File, None),
            Some("k-one"),
            &[],
            KeySets::serving_alone(serving.clone()),
        )
        .expect("a file source");
        write(dir.path(), "key", "k-two\n");
        assert!(reloader.reload().expect("reloaded").serving_changed);
        assert!(serving.contains_token("k-two") && !serving.contains_token("k-one"));
        assert!(serving.is_enabled());
    }

    #[test]
    fn a_bad_file_leaves_every_set_as_it_was() {
        let dir = tempfile::tempdir().expect("a directory");
        let tenants = write(dir.path(), "tenants", "acme:t-one\n");
        let cp = write(dir.path(), "cp", "ops:Ops:admin:cp-one\n");
        let serving = AuthConfig::with_tenant_keys(None, &[]);
        let set = ApiKeySet::new(vec![ApiKeyEntry::new("ops", "Ops", "cp-one", Role::Admin)]);
        let reloader = SecretReloader::new(
            SecretFiles {
                tenant_api_keys: Some(tenants.clone()),
                control_plane_api_keys: Some(cp.clone()),
                ..SecretFiles::default()
            },
            &sources(SecretSource::Flag, Some(SecretSource::File)),
            None,
            &[],
            KeySets {
                control_plane: Some(set.clone()),
                ..KeySets::serving_alone(serving.clone())
            },
        )
        .expect("file sources");
        reloader.reload().expect("the first reload");
        assert!(serving.contains_token("t-one"));

        write(dir.path(), "cp", "ops:Ops:owner:cp-two\n");
        write(dir.path(), "tenants", "acme:t-two\n");
        let BadFile(error) = reloader.reload().expect_err("the role is refused");
        let text = error.to_string();
        assert!(text.contains("control-plane-api-keys"), "{text}");
        assert!(!text.contains("cp-two"), "{text}");
        assert!(
            serving.contains_token("t-one"),
            "the tenant swap did not happen either"
        );
        assert!(set.find("cp-one").is_some() && set.find("cp-two").is_none());

        write(dir.path(), "cp", "# none\n");
        let BadFile(error) = reloader.reload().expect_err("an empty file is refused");
        assert!(error.to_string().contains("holds no entry"), "{error}");
        assert!(set.find("cp-one").is_some());
    }

    #[test]
    fn a_reload_moves_the_keys_the_view_names() {
        let dir = tempfile::tempdir().expect("a directory");
        let file = write(dir.path(), "key", "k-one\n");
        let cp = write(dir.path(), "cp", "ops:Ops:admin:cp-one\n");
        let set = ApiKeySet::new(vec![ApiKeyEntry::new("ops", "Ops", "cp-one", Role::Admin)]);
        let reloader = SecretReloader::new(
            SecretFiles {
                api_key: Some(file),
                control_plane_api_keys: Some(cp),
                ..SecretFiles::default()
            },
            &sources(SecretSource::File, None),
            Some("k-one"),
            &[],
            KeySets {
                control_plane: Some(set),
                ..KeySets::serving_alone(AuthConfig::new(Some("k-one".into())))
            },
        )
        .expect("file sources");
        assert_eq!(
            reloader.keys_in_force(),
            KeysInForce::of(Some("k-one"), &[], Some(1)),
            "the start's record"
        );

        write(dir.path(), "key", "k-two\n");
        write(
            dir.path(),
            "cp",
            "ops:Ops:admin:cp-one\nsre:Sre:admin:cp-two\n",
        );
        reloader.reload().expect("reloaded");
        assert_eq!(
            reloader.keys_in_force(),
            KeysInForce::of(Some("k-two"), &[], Some(2)),
            "the reload's record"
        );
    }

    #[test]
    fn a_reload_moves_the_tenant_keys_the_view_names() {
        let dir = tempfile::tempdir().expect("a directory");
        let tenants = write(dir.path(), "tenants", "acme:t-one\n");
        let start = [TenantApiKeyEntry {
            tenant_id: "acme".into(),
            key: "t-one".into(),
        }];
        let reloader = SecretReloader::new(
            SecretFiles {
                tenant_api_keys: Some(tenants),
                ..SecretFiles::default()
            },
            &sources(SecretSource::Flag, Some(SecretSource::File)),
            Some("flag-key"),
            &start,
            KeySets::serving_alone(AuthConfig::with_tenant_keys(
                Some("flag-key".into()),
                &start,
            )),
        )
        .expect("a file source");
        assert_eq!(
            reloader.keys_in_force(),
            KeysInForce::of(Some("flag-key"), &start, None),
            "the start's record"
        );

        write(dir.path(), "tenants", "beta:t-two\n");
        reloader.reload().expect("reloaded");
        let rotated = [TenantApiKeyEntry {
            tenant_id: "beta".into(),
            key: "t-two".into(),
        }];
        assert_eq!(
            reloader.keys_in_force(),
            KeysInForce::of(Some("flag-key"), &rotated, None),
            "the file's tenant is in the record, the start's is gone, the flag's key is kept"
        );
    }

    #[test]
    fn a_refused_reload_leaves_the_record() {
        let dir = tempfile::tempdir().expect("a directory");
        let tenants = write(dir.path(), "tenants", "acme:t-one\n");
        let start = [TenantApiKeyEntry {
            tenant_id: "acme".into(),
            key: "t-one".into(),
        }];
        let reloader = SecretReloader::new(
            SecretFiles {
                tenant_api_keys: Some(tenants),
                ..SecretFiles::default()
            },
            &sources(SecretSource::Flag, Some(SecretSource::File)),
            Some("flag-key"),
            &start,
            KeySets::serving_alone(AuthConfig::with_tenant_keys(
                Some("flag-key".into()),
                &start,
            )),
        )
        .expect("a file source");

        // The line parses and breaks a start rule: the refusal is the last
        // gate before the swap, so the pin covers every earlier place the
        // record's write could move to.
        write(dir.path(), "tenants", "acme:flag-key\n");
        reloader
            .reload()
            .expect_err("the flag's shared key repeated by a tenant is refused");
        assert_eq!(
            reloader.keys_in_force(),
            KeysInForce::of(Some("flag-key"), &start, None),
            "a refused reload leaves the record as it was"
        );
    }

    #[test]
    fn the_control_plane_keys_follow_their_file() {
        let dir = tempfile::tempdir().expect("a directory");
        let cp = write(dir.path(), "cp", "ops:Ops:admin:cp-one\n");
        let set = ApiKeySet::new(vec![ApiKeyEntry::new("ops", "Ops", "cp-one", Role::Admin)]);
        let reloader = SecretReloader::new(
            SecretFiles {
                control_plane_api_keys: Some(cp),
                ..SecretFiles::default()
            },
            &sources(SecretSource::Flag, None),
            None,
            &[],
            KeySets {
                control_plane: Some(set.clone()),
                ..KeySets::serving_alone(AuthConfig::new(None))
            },
        )
        .expect("a file source");
        let same = reloader.reload().expect("reloaded");
        assert!(!same.control_plane_changed);
        write(
            dir.path(),
            "cp",
            "ops:Ops:admin:cp-two\nro:Readers:user:cp-three\n",
        );
        let report = reloader.reload().expect("reloaded");
        assert!(report.control_plane_changed);
        assert_eq!(report.files[0].entries, 2);
        assert!(set.find("cp-one").is_none());
        assert_eq!(
            set.find("cp-three").map(|entry| entry.role),
            Some(Role::User)
        );
    }

    /// A tenant file the start would refuse is refused by the reload too,
    /// by the file's field, and every key stands: a key repeated across two
    /// tenants, the flag's shared key repeated by a tenant, an empty key.
    #[test]
    fn a_tenant_file_breaking_the_starts_rules_leaves_every_key_standing() {
        let dir = tempfile::tempdir().expect("a directory");
        let tenants = write(dir.path(), "tenants", "acme:t-one\n");
        let entries = [TenantApiKeyEntry {
            tenant_id: "acme".into(),
            key: "t-one".into(),
        }];
        let serving = AuthConfig::with_tenant_keys(Some("flag-key".into()), &entries);
        let reloader = SecretReloader::new(
            SecretFiles {
                tenant_api_keys: Some(tenants),
                ..SecretFiles::default()
            },
            &sources(SecretSource::Flag, Some(SecretSource::File)),
            Some("flag-key"),
            &entries,
            KeySets::serving_alone(serving.clone()),
        )
        .expect("a file source");

        for (content, rule) in [
            (
                "acme:t-two\nbeta:t-two\n",
                "a key repeated across two tenants",
            ),
            (
                "acme:flag-key\n",
                "the flag's shared key repeated by a tenant",
            ),
            ("acme:\n", "an empty key"),
        ] {
            write(dir.path(), "tenants", content);
            let BadFile(error) = reloader.reload().expect_err(rule);
            let text = error.to_string();
            assert!(text.contains("tenant-api-keys-file"), "{rule}: {text}");
            assert!(
                !text.contains("flag-key") && !text.contains("t-two"),
                "{rule}: a key reached the refusal: {text}"
            );
            assert!(
                serving.contains_token("t-one") && serving.contains_token("flag-key"),
                "{rule}: every key stands"
            );
        }
    }

    /// The watch: a rotation as a secret manager does it, a staged file
    /// renamed over the key file, is in force without a call.
    #[tokio::test]
    async fn the_watch_puts_a_rotated_file_in_force_without_a_call() {
        let dir = tempfile::tempdir().expect("a directory");
        let file = write(dir.path(), "key", "k-one\n");
        let serving = AuthConfig::new(Some("k-one".into()));
        let reloader = Arc::new(
            SecretReloader::new(
                SecretFiles {
                    api_key: Some(file.clone()),
                    ..SecretFiles::default()
                },
                &sources(SecretSource::File, None),
                Some("k-one"),
                &[],
                KeySets::serving_alone(serving.clone()),
            )
            .expect("a file source"),
        );
        let handle = start_auto_reload(Arc::clone(&reloader)).expect("the directory is watched");

        let staged = write(dir.path(), "key.staged", "k-two\n");
        std::fs::rename(&staged, &file).expect("renamed over the key file");
        let deadline = time::Instant::now() + Duration::from_secs(10);
        while !serving.contains_token("k-two") {
            assert!(
                time::Instant::now() < deadline,
                "the rotation was not reloaded"
            );
            time::sleep(Duration::from_millis(50)).await;
        }
        assert!(!serving.contains_token("k-one"), "the old key is out");
        handle.abort();
    }

    /// A key rotated between the start's read of its file and the watch's
    /// start raises no event: the watch reads the files once as it starts,
    /// and the rotated key is in force without a further change.
    #[tokio::test]
    async fn a_rotation_before_the_watch_starts_is_in_force_once_it_starts() {
        let dir = tempfile::tempdir().expect("a directory");
        let file = write(dir.path(), "key", "k-one\n");
        let serving = AuthConfig::new(Some("k-one".into()));
        let reloader = Arc::new(
            SecretReloader::new(
                SecretFiles {
                    api_key: Some(file.clone()),
                    ..SecretFiles::default()
                },
                &sources(SecretSource::File, None),
                Some("k-one"),
                &[],
                KeySets::serving_alone(serving.clone()),
            )
            .expect("a file source"),
        );

        // Rotated before the watch exists: no event will ever report it.
        let staged = write(dir.path(), "key.staged", "k-two\n");
        std::fs::rename(&staged, &file).expect("renamed over the key file");
        let handle = start_auto_reload(Arc::clone(&reloader)).expect("the directory is watched");
        let deadline = time::Instant::now() + Duration::from_secs(10);
        while !serving.contains_token("k-two") {
            assert!(
                time::Instant::now() < deadline,
                "the rotation before the watch was not read"
            );
            time::sleep(Duration::from_millis(50)).await;
        }
        assert!(!serving.contains_token("k-one"), "the old key is out");
        handle.abort();
    }

    /// The key files' directory removed and recreated with a rotated key:
    /// every key stands while it is gone, and the rotated key is in force
    /// within the re-watch's second without a call. The key is written in
    /// the same instant as the directory, before the re-watch's next tick,
    /// so the re-watch's own wake is what reads it.
    #[tokio::test]
    async fn a_lost_directory_is_watched_again_and_its_rotated_key_put_in_force() {
        let root = tempfile::tempdir().expect("a directory");
        let dir = root.path().join("keys");
        std::fs::create_dir(&dir).expect("the key directory");
        let file = write(&dir, "key", "k-one\n");
        let serving = AuthConfig::new(Some("k-one".into()));
        let reloader = Arc::new(
            SecretReloader::new(
                SecretFiles {
                    api_key: Some(file),
                    ..SecretFiles::default()
                },
                &sources(SecretSource::File, None),
                Some("k-one"),
                &[],
                KeySets::serving_alone(serving.clone()),
            )
            .expect("a file source"),
        );
        let handle = start_auto_reload(Arc::clone(&reloader)).expect("the directory is watched");

        std::fs::remove_dir_all(&dir).expect("the directory removed");
        time::sleep(Duration::from_millis(300)).await;
        assert!(
            serving.contains_token("k-one"),
            "every key stands while the directory is gone"
        );

        std::fs::create_dir(&dir).expect("the directory again");
        write(&dir, "key", "k-two\n");
        let deadline = time::Instant::now() + Duration::from_secs(10);
        while !serving.contains_token("k-two") {
            assert!(
                time::Instant::now() < deadline,
                "the rotated key was not reloaded after the re-watch"
            );
            time::sleep(Duration::from_millis(50)).await;
        }
        assert!(!serving.contains_token("k-one"), "the old key is out");
        handle.abort();
    }

    /// A mounted secret in the kubelet's shape: the files in a timestamped
    /// directory, `..data` pointing at it, each file a symlink through
    /// `..data`. The kubelet rotates it by swapping the `..data` symlink,
    /// which the watch sees through the file it reads.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_mounted_secret_swap_is_seen_through_the_data_symlink() {
        use std::os::unix::fs::symlink;

        let mount = tempfile::tempdir().expect("a directory");
        let first = mount.path().join("..2026_11_08_v1");
        std::fs::create_dir(&first).expect("the first version");
        write(&first, "key", "k-one\n");
        symlink("..2026_11_08_v1", mount.path().join("..data")).expect("..data");
        symlink("..data/key", mount.path().join("key")).expect("the file's link");
        let file = mount.path().join("key");

        let serving = AuthConfig::new(Some("k-one".into()));
        let reloader = Arc::new(
            SecretReloader::new(
                SecretFiles {
                    api_key: Some(file),
                    ..SecretFiles::default()
                },
                &sources(SecretSource::File, None),
                Some("k-one"),
                &[],
                KeySets::serving_alone(serving.clone()),
            )
            .expect("a file source"),
        );
        let handle = start_auto_reload(Arc::clone(&reloader)).expect("the mount is watched");

        let second = mount.path().join("..2026_11_08_v2");
        std::fs::create_dir(&second).expect("the second version");
        write(&second, "key", "k-two\n");
        symlink("..2026_11_08_v2", mount.path().join("..data_tmp")).expect("the new link");
        std::fs::rename(mount.path().join("..data_tmp"), mount.path().join("..data"))
            .expect("the swap");
        let deadline = time::Instant::now() + Duration::from_secs(10);
        while !serving.contains_token("k-two") {
            assert!(time::Instant::now() < deadline, "the swap was not reloaded");
            time::sleep(Duration::from_millis(50)).await;
        }
        handle.abort();
    }

    /// A key directory that cannot be watched refuses the start by the
    /// flag and the directory, named once, never by a bare OS error.
    #[tokio::test]
    async fn a_directory_that_cannot_be_watched_is_refused_by_flag_and_directory() {
        let dir = tempfile::tempdir().expect("a directory");
        // A regular file where the key's directory should be.
        let blocker = write(dir.path(), "a-file", "");
        let reloader = Arc::new(
            SecretReloader::new(
                SecretFiles {
                    api_key: Some(blocker.join("key")),
                    ..SecretFiles::default()
                },
                &sources(SecretSource::File, None),
                Some("k-one"),
                &[],
                KeySets::serving_alone(AuthConfig::new(Some("k-one".into()))),
            )
            .expect("a file source"),
        );
        let error = start_auto_reload(reloader).expect_err("a file is no directory to watch");
        let text = error.to_string();
        let directory = blocker.display().to_string();
        assert!(
            text.starts_with("--secrets-auto-reload: cannot watch "),
            "{text}"
        );
        assert_eq!(
            text.matches(&directory).count(),
            1,
            "the directory once: {text}"
        );
        assert!(text.ends_with("not a directory"), "{text}");
    }

    /// The refusal names the directory as an absolute path and, for the two
    /// kinds the inotify limits surface as, the limit to raise: `EMFILE` names
    /// the instance limit and the open-file limit, the crate's watch-limit
    /// kind the watch limit. Any other cause, `ENOSPC` as an io error among
    /// them (the crate never passes it through), carries no hint.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_watch_refusal_names_the_absolute_directory_and_the_inotify_limit() {
        let instances = notify::Error::io(std::io::Error::from_raw_os_error(24));
        let text = watch_refused(Path::new("keys"), instances).to_string();
        let absolute = std::path::absolute("keys").expect("an absolute path");
        let prefix = format!(
            "--secrets-auto-reload: cannot watch {}: ",
            absolute.display()
        );
        assert!(text.starts_with(&prefix), "{text}");
        assert!(
            text.ends_with(
                "(raise fs.inotify.max_user_instances, or the process's open-file limit)"
            ),
            "{text}"
        );
        let watches = notify::Error::new(notify::ErrorKind::MaxFilesWatch);
        let text = watch_refused(Path::new("."), watches).to_string();
        assert!(
            text.ends_with("(raise fs.inotify.max_user_watches)"),
            "{text}"
        );
        assert!(!text.contains("cannot watch .:"), "{text}");
        for other in [20, 28] {
            let other = notify::Error::io(std::io::Error::from_raw_os_error(other));
            let text = watch_refused(Path::new("keys"), other).to_string();
            assert!(!text.contains("fs.inotify"), "{text}");
        }
    }

    /// The admin routes' fallback, in force when no control-plane keys are
    /// configured, follows the shared key through a reload of its file: the
    /// retired key is out there too.
    #[test]
    fn a_shared_key_reload_moves_the_admin_fallback_with_the_serving_set() {
        let dir = tempfile::tempdir().expect("a directory");
        let file = write(dir.path(), "key", "k-one\n");
        let serving = AuthConfig::new(Some("k-one".into()));
        let fallback = AuthConfig::new(Some("k-one".into()));
        let reloader = SecretReloader::new(
            SecretFiles {
                api_key: Some(file),
                ..SecretFiles::default()
            },
            &sources(SecretSource::File, None),
            Some("k-one"),
            &[],
            KeySets {
                admin_fallback: Some(fallback.clone()),
                ..KeySets::serving_alone(serving.clone())
            },
        )
        .expect("a file source");
        write(dir.path(), "key", "k-two\n");
        reloader.reload().expect("reloaded");
        assert!(serving.contains_token("k-two") && !serving.contains_token("k-one"));
        assert!(
            fallback.contains_token("k-two") && !fallback.contains_token("k-one"),
            "the fallback follows the shared key"
        );
    }

    /// A tenant key never reaches the fallback, and a shared key the flag
    /// gave leaves the fallback as the start built it through a tenant-file
    /// reload.
    #[test]
    fn the_admin_fallback_takes_the_shared_key_alone() {
        let dir = tempfile::tempdir().expect("a directory");
        let key = write(dir.path(), "key", "k-one\n");
        let tenants = write(dir.path(), "tenants", "acme:t-one\n");
        let start = [TenantApiKeyEntry {
            tenant_id: "acme".into(),
            key: "t-one".into(),
        }];
        let fallback = AuthConfig::new(Some("k-one".into()));
        let reloader = SecretReloader::new(
            SecretFiles {
                api_key: Some(key),
                tenant_api_keys: Some(tenants.clone()),
                ..SecretFiles::default()
            },
            &sources(SecretSource::File, Some(SecretSource::File)),
            Some("k-one"),
            &start,
            KeySets {
                admin_fallback: Some(fallback.clone()),
                ..KeySets::serving_alone(AuthConfig::with_tenant_keys(Some("k-one".into()), &start))
            },
        )
        .expect("file sources");
        write(dir.path(), "key", "k-two\n");
        write(dir.path(), "tenants", "acme:t-two\n");
        reloader.reload().expect("reloaded");
        assert!(fallback.contains_token("k-two"), "the shared key moved");
        assert!(
            !fallback.contains_token("t-two") && !fallback.contains_token("t-one"),
            "no tenant key reaches the fallback"
        );

        // The shared key from the flag: the fallback is not a reload's target.
        let flagged = AuthConfig::new(Some("flag-key".into()));
        let reloader = SecretReloader::new(
            SecretFiles {
                tenant_api_keys: Some(tenants),
                ..SecretFiles::default()
            },
            &sources(SecretSource::Flag, Some(SecretSource::File)),
            Some("flag-key"),
            &start,
            KeySets {
                admin_fallback: Some(flagged.clone()),
                ..KeySets::serving_alone(AuthConfig::with_tenant_keys(
                    Some("flag-key".into()),
                    &start,
                ))
            },
        )
        .expect("a file source");
        write(dir.path(), "tenants", "beta:t-three\n");
        reloader.reload().expect("reloaded");
        assert!(
            flagged.contains_token("flag-key") && !flagged.contains_token("t-three"),
            "the flag's key stands alone in the fallback"
        );
    }

    /// The record of `--secrets-auto-reload` names the settings whose files
    /// are watched, in the order the reload reads them; no file, no name.
    #[test]
    fn the_watched_files_are_named_by_their_settings() {
        let files = SecretFiles {
            api_key: Some(PathBuf::from("/keys/api-key")),
            control_plane_api_keys: Some(PathBuf::from("/keys/control-plane")),
            ..SecretFiles::default()
        };
        assert_eq!(
            files.names().collect::<Vec<_>>(),
            ["api-key-file", "control-plane-api-keys-file"],
            "the settings with a file, in the reload's order"
        );
        assert_eq!(files.count(), 2);
        assert!(SecretFiles::default().names().next().is_none());
    }

    /// Every key's file has its own name in the record, the tenant file's
    /// between the shared key's and the control plane's: one name per
    /// setting, in the order the reload reads the files.
    #[test]
    fn every_watched_file_is_named_once_in_the_reload_order() {
        let files = SecretFiles {
            api_key: Some(PathBuf::from("/keys/api-key")),
            tenant_api_keys: Some(PathBuf::from("/keys/tenants")),
            control_plane_api_keys: Some(PathBuf::from("/keys/control-plane")),
        };
        assert_eq!(
            files.names().collect::<Vec<_>>(),
            [
                "api-key-file",
                "tenant-api-keys-file",
                "control-plane-api-keys-file"
            ],
            "one name per file, in the reload's order"
        );
        assert_eq!(files.count(), 3);
        let tenants_alone = SecretFiles {
            tenant_api_keys: Some(PathBuf::from("/keys/tenants")),
            ..SecretFiles::default()
        };
        assert_eq!(
            tenants_alone.names().collect::<Vec<_>>(),
            ["tenant-api-keys-file"],
            "the tenant file's own name"
        );
    }

    /// The record's order is the reload's: a reloader over all three files
    /// reports them in the order the record names them.
    #[test]
    fn a_reload_of_every_file_reports_them_in_the_reload_order() {
        let dir = tempfile::tempdir().expect("a directory");
        let key = write(dir.path(), "key", "k-one\n");
        let tenants = write(dir.path(), "tenants", "acme:t-one\n");
        let cp = write(dir.path(), "cp", "ops:Ops:admin:cp-one\n");
        let start = [TenantApiKeyEntry {
            tenant_id: "acme".into(),
            key: "t-one".into(),
        }];
        let set = ApiKeySet::new(vec![ApiKeyEntry::new("ops", "Ops", "cp-one", Role::Admin)]);
        let reloader = SecretReloader::new(
            SecretFiles {
                api_key: Some(key),
                tenant_api_keys: Some(tenants),
                control_plane_api_keys: Some(cp),
            },
            &sources(SecretSource::File, Some(SecretSource::File)),
            Some("k-one"),
            &start,
            KeySets {
                control_plane: Some(set),
                ..KeySets::serving_alone(AuthConfig::with_tenant_keys(Some("k-one".into()), &start))
            },
        )
        .expect("file sources");
        let report = reloader.reload().expect("reloaded");
        assert_eq!(
            report
                .files
                .iter()
                .map(|file| file.setting)
                .collect::<Vec<_>>(),
            reloader.files().names().collect::<Vec<_>>(),
            "the reload reports the files in the order the record names them"
        );
        assert_eq!(report.files.len(), 3, "every file read");
    }

    #[test]
    fn no_key_from_a_file_means_no_reloader() {
        let files = SecretFiles {
            api_key: Some(PathBuf::from("/nowhere")),
            control_plane_api_keys: Some(PathBuf::from("/nowhere-either")),
            ..SecretFiles::default()
        };
        // A path with a value from the flag beside it is not a source; a
        // control-plane file without control-plane authentication is not one.
        assert!(SecretReloader::new(
            files,
            &sources(SecretSource::Flag, None),
            Some("k"),
            &[],
            KeySets::serving_alone(AuthConfig::new(Some("k".into()))),
        )
        .is_none());
    }
}
