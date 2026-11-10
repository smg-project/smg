//! The keys reloaded from their files while the gateway runs: the serving
//! key, the tenant keys and the control-plane keys. Each file is read again
//! by the rules the start read it by ([`crate::config::secrets`]) and parsed
//! by the same parsers, the serving set that would be in force passes the
//! start's rules, and each set is put in force as one swap. A file that
//! cannot be used, unreadable, unparsable or breaking those rules, leaves
//! every set as it was and is refused by field and path, never by content.
//! A key the command line or the environment gave is not a file: it stays
//! as the start read it, and a reload rebuilds the serving map from that
//! fixed part and the files'.

use std::{collections::BTreeMap, ffi::OsString, path::PathBuf, sync::Arc, time::Duration};

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
    fs_watch::{settle, DirectoryWatch},
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

    fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        [
            &self.api_key,
            &self.tenant_api_keys,
            &self.control_plane_api_keys,
        ]
        .into_iter()
        .flatten()
    }

    /// The files by directory: what the watch of `--secrets-auto-reload`
    /// watches, and for which names.
    fn by_directory(&self) -> BTreeMap<PathBuf, Vec<OsString>> {
        let mut dirs: BTreeMap<PathBuf, Vec<OsString>> = BTreeMap::new();
        for path in self.paths() {
            let dir = match path.parent() {
                Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
                _ => PathBuf::from("."),
            };
            if let Some(name) = path.file_name() {
                dirs.entry(dir).or_default().push(name.to_os_string());
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
/// value redacted, a rule its reason.
fn entry_refusal(error: ConfigError) -> String {
    match error {
        ConfigError::InvalidValue { reason, .. } | ConfigError::ValidationFailed { reason } => {
            reason
        }
        other => other.to_string(),
    }
}

/// The reload of the keys from their files. It holds the handles the
/// authentication layers read, so a swap reaches every request after it,
/// and the serving keys a file did not give, by hash.
pub struct SecretReloader {
    files: SecretFiles,
    serving: AuthConfig,
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
    /// authentication is on (`control_plane` is its set in force). The
    /// serving values are the start's; whichever of them a file did not
    /// give is the fixed part every reload keeps.
    pub fn new(
        files: SecretFiles,
        sources: &SecretSources,
        api_key: Option<&str>,
        tenant_api_keys: &[TenantApiKeyEntry],
        serving: AuthConfig,
        control_plane: Option<ApiKeySet>,
    ) -> Option<Self> {
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
        let in_force = KeysInForce::of(
            api_key,
            tenant_api_keys,
            control_plane.as_ref().map(ApiKeySet::len),
        );
        Some(Self {
            files,
            serving,
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
    /// the set it took.
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
            let keys = AuthConfig::key_map(api_key, tenants);
            let previous = self.serving.replace(keys.clone());
            report.serving_changed = *previous != keys;
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
/// runs, once a burst of changes has settled. A directory that cannot be
/// watched refuses the start. A directory removed or moved away is said so
/// once and watched again once it exists, with one reload then. The task runs
/// for the lifetime of the server.
pub fn start_auto_reload(reloader: Arc<SecretReloader>) -> notify::Result<task::JoinHandle<()>> {
    let (wake, woken) = mpsc::channel(1);
    let mut watches = Vec::new();
    for (dir, names) in reloader.files.by_directory() {
        let watch = DirectoryWatch::start(&dir, names.clone(), wake.clone())?;
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
    #[expect(
        clippy::disallowed_methods,
        reason = "the key-file watch runs for the lifetime of the server"
    )]
    let handle = task::spawn(watch_loop(reloader, wake, woken, watches));
    Ok(handle)
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
    use std::path::Path;

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
            serving.clone(),
            None,
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
            serving.clone(),
            None,
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
            serving.clone(),
            None,
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
            serving.clone(),
            Some(set.clone()),
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
            AuthConfig::new(Some("k-one".into())),
            Some(set),
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
            AuthConfig::with_tenant_keys(Some("flag-key".into()), &start),
            None,
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
            AuthConfig::with_tenant_keys(Some("flag-key".into()), &start),
            None,
        )
        .expect("a file source");

        write(dir.path(), "tenants", "no-tenant-separator\n");
        reloader
            .reload()
            .expect_err("a tenant line without its separator is refused");
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
            AuthConfig::new(None),
            Some(set.clone()),
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
            serving.clone(),
            None,
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
                serving.clone(),
                None,
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
                serving.clone(),
                None,
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
                serving.clone(),
                None,
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
            AuthConfig::new(Some("k".into())),
            None,
        )
        .is_none());
    }
}
