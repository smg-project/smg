//! What the host says: the warnings a test process logs, for the cases that
//! read a line of the host (a breaker's opening). One subscriber for the
//! process, shared by the plugin kinds' test binaries.

use std::sync::Arc;

/// What the host says: a subscriber that keeps every warning of this crate
/// as its level and its fields.
pub struct Said(Arc<std::sync::Mutex<Vec<String>>>);

impl tracing::Subscriber for Said {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::WARN
            && metadata.target().starts_with("smg_policy_controller")
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(String);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                let _ = write!(self.0, " {}={value:?}", field.name());
            }
        }
        let mut fields = Fields(event.metadata().level().to_string());
        event.record(&mut fields);
        let line = fields.0;
        let trapped = {
            let mut hit = None;
            TRAPS
                .lock()
                .expect("the traps")
                .retain_mut(|(named, text, left)| {
                    if !(line.contains(named.as_str()) && line.contains(text.as_str())) {
                        return true;
                    }
                    hit = Some(named.clone());
                    *left -= 1;
                    *left > 0
                });
            hit
        };
        self.0.lock().expect("the kept lines").push(line);
        assert!(
            trapped.is_none(),
            "the subscriber panics at a line a case armed: {trapped:?}"
        );
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// The lines the subscriber panics at, one entry a plugin: what names the
/// plugin, a text, and how many times more. A case arms and reads its own
/// plugin's entry, so two cases of one binary do not take each other's
/// panics.
static TRAPS: std::sync::Mutex<Vec<(String, String, u32)>> = std::sync::Mutex::new(Vec::new());

/// Arms the subscriber to panic, on the thread that says it, at each of the
/// next `times` lines that name `plugin` and hold `text`: a case's way to
/// make the host's own telling of a failure panic where it is told.
pub fn panic_at(plugin: &str, text: &str, times: u32) {
    assert!(times > 0, "a trap is armed for one panic at least");
    said();
    let named = format!("plugin=\"{plugin}\"");
    let mut traps = TRAPS.lock().expect("the traps");
    traps.retain(|(armed, _, _)| *armed != named);
    traps.push((named, text.to_owned(), times));
}

/// Whether every panic a case armed for `plugin` has been made.
pub fn trap_spent(plugin: &str) -> bool {
    let named = format!("plugin=\"{plugin}\"");
    !TRAPS
        .lock()
        .expect("the traps")
        .iter()
        .any(|(armed, _, _)| *armed == named)
}

/// The warnings this test process logs, for the cases that read a line of
/// the host. The subscriber is the process's, installed on the first call:
/// a thread's own would miss a line whose callsite another case's thread
/// met first, the callsite cache being the process's. Every warning from
/// then on lands in one buffer, and a case reads the lines that name its
/// plugin.
pub fn said() -> Arc<std::sync::Mutex<Vec<String>>> {
    static SAID: std::sync::OnceLock<Arc<std::sync::Mutex<Vec<String>>>> =
        std::sync::OnceLock::new();
    Arc::clone(SAID.get_or_init(|| {
        let lines = Arc::new(std::sync::Mutex::new(Vec::new()));
        // Set once per process; nothing else in this binary sets one.
        tracing::subscriber::set_global_default(Said(Arc::clone(&lines)))
            .expect("no other subscriber is the process's");
        lines
    }))
}

/// The kept lines that name `plugin` and hold `text`.
pub fn lines(plugin: &str, text: &str) -> Vec<String> {
    let named = format!("plugin=\"{plugin}\"");
    said()
        .lock()
        .expect("the kept lines")
        .iter()
        .filter(|line| line.contains(text) && line.contains(&named))
        .cloned()
        .collect()
}

/// The lines of an opening that name `plugin`.
pub fn openings(plugin: &str) -> Vec<String> {
    lines(plugin, "breaker opened")
}
