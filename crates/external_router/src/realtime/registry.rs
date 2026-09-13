//! In-memory session and call registry for Realtime API connections.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

/// Connection state for a realtime session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// WebSocket upgrade accepted but upstream not yet connected.
    Pending,
    /// Bidirectional proxy is active.
    Connected,
    /// Connection has been closed.
    Disconnected,
}

/// A tracked realtime connection (WebSocket session or WebRTC call).
#[derive(Debug, Clone)]
pub struct ConnectionEntry {
    pub id: String,
    pub model: String,
    pub worker_url: String,
    pub state: ConnectionState,
    pub created_at: Instant,
    pub cancel_token: CancellationToken,
}

/// DashMap-backed registry for realtime sessions and WebRTC calls.
///
/// No fixed capacity — DashMap grows dynamically. The reaper handles
/// cleanup of stale entries.
#[derive(Debug)]
pub struct RealtimeRegistry {
    sessions: ConnectionMap,
    calls: ConnectionMap,
}

/// A named DashMap of connection entries with shared CRUD operations.
#[derive(Debug)]
struct ConnectionMap(DashMap<String, ConnectionEntry>);

impl ConnectionMap {
    fn new() -> Self {
        Self(DashMap::new())
    }

    fn register(&self, id: String, model: String, worker_url: String) -> ConnectionEntry {
        let entry = ConnectionEntry {
            id: id.clone(),
            model,
            worker_url,
            state: ConnectionState::Pending,
            created_at: Instant::now(),
            cancel_token: CancellationToken::new(),
        };
        if let Some(old) = self.0.insert(id, entry.clone()) {
            old.cancel_token.cancel();
        }
        entry
    }

    fn set_state(&self, id: &str, state: ConnectionState) {
        if let Some(mut entry) = self.0.get_mut(id) {
            entry.state = state;
        }
    }

    fn get(&self, id: &str) -> Option<ConnectionEntry> {
        self.0.get(id).map(|e| e.clone())
    }

    fn remove(&self, id: &str) -> Option<ConnectionEntry> {
        self.0.remove(id).map(|(_, e)| {
            e.cancel_token.cancel();
            e
        })
    }

    /// Remove stale entries and cancel their tokens. Returns count of reaped entries.
    fn reap_stale(&self, is_stale: impl Fn(ConnectionState, Duration) -> bool) -> usize {
        let now = Instant::now();
        let mut reaped = 0;
        self.0.retain(|_, entry| {
            if is_stale(entry.state, now.duration_since(entry.created_at)) {
                entry.cancel_token.cancel();
                reaped += 1;
                false
            } else {
                true
            }
        });
        reaped
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl RealtimeRegistry {
    pub fn new() -> Self {
        Self {
            sessions: ConnectionMap::new(),
            calls: ConnectionMap::new(),
        }
    }

    // ---- Session methods ----

    pub fn register_session(
        &self,
        session_id: String,
        model: String,
        worker_url: String,
    ) -> ConnectionEntry {
        self.sessions.register(session_id, model, worker_url)
    }

    pub fn set_session_state(&self, session_id: &str, state: ConnectionState) {
        self.sessions.set_state(session_id, state);
    }

    pub fn get_session(&self, session_id: &str) -> Option<ConnectionEntry> {
        self.sessions.get(session_id)
    }

    pub fn remove_session(&self, session_id: &str) -> Option<ConnectionEntry> {
        self.sessions.remove(session_id)
    }

    // ---- Call methods ----

    pub fn register_call(
        &self,
        call_id: String,
        model: String,
        worker_url: String,
    ) -> ConnectionEntry {
        self.calls.register(call_id, model, worker_url)
    }

    pub fn get_call(&self, call_id: &str) -> Option<ConnectionEntry> {
        self.calls.get(call_id)
    }

    pub fn set_call_state(&self, call_id: &str, state: ConnectionState) {
        self.calls.set_state(call_id, state);
    }

    pub fn remove_call(&self, call_id: &str) -> Option<ConnectionEntry> {
        self.calls.remove(call_id)
    }

    // ---- Reaper ----

    /// Start a background task that evicts stale entries.
    ///
    /// `pending_max_age` applies to `Pending` sessions (upgrade not completed).
    /// `max_age` applies to `Disconnected` sessions (connection closed but not
    /// yet removed). Active (`Connected`) sessions are never reaped.
    ///
    /// Returns a `CancellationToken` that stops the reaper when cancelled.
    pub fn start_reaper(
        self: &Arc<Self>,
        max_age: Duration,
        pending_max_age: Duration,
        interval: Duration,
    ) -> CancellationToken {
        let shutdown = CancellationToken::new();
        let token = shutdown.clone();
        let registry = Arc::clone(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "reaper task cancelled via returned token"
        )]
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            loop {
                tokio::select! {
                    _ = tick.tick() => {}
                    () = shutdown.cancelled() => {
                        info!("Realtime registry reaper shutting down");
                        return;
                    }
                }

                let is_stale = |state: ConnectionState, age: Duration| -> bool {
                    match state {
                        ConnectionState::Connected => false,
                        ConnectionState::Pending => age > pending_max_age,
                        ConnectionState::Disconnected => age > max_age,
                    }
                };

                let sessions_reaped = registry.sessions.reap_stale(is_stale);
                let calls_reaped = registry.calls.reap_stale(is_stale);

                if sessions_reaped > 0 || calls_reaped > 0 {
                    debug!(
                        sessions_reaped,
                        calls_reaped, "Realtime registry reaper cycle"
                    );
                }
            }
        });
        info!("Realtime registry reaper started (max_age={max_age:?}, interval={interval:?})");
        token
    }

    /// Stats for observability.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn call_count(&self) -> usize {
        self.calls.len()
    }
}

impl Default for RealtimeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
