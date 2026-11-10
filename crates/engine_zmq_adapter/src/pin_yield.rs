//! A pin's yield to the room, logged once per connection.
//!
//! On a connection that places a disaggregated request by the residue of its
//! rendezvous room (the prefill side of a disaggregated pair, whose engine
//! dispatches by `bootstrap_room % dp_size` and whose KV sender fails a
//! request that runs elsewhere), a `data_parallel_rank` that names another
//! rank yields to the room: the placement is the engine's own rule, never a
//! refusal. A dp-aware Router in front of a prefill side of several ranks
//! pins a rank per request and mints a room whose residue agrees with the pin
//! for one request in N, so there the yield is the common case, not an
//! operator error. The connection's first yield is a WARN line with the
//! request's fields; every later one is a DEBUG line, so the logger's real
//! warnings stay readable.

use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{debug, warn, Level};

/// Whether a pin's yield to the room has been logged at WARN on this
/// connection yet. One per connection, beside the placement rule it reports on.
#[derive(Debug, Default)]
pub(crate) struct PinYieldNote {
    warned: AtomicBool,
}

impl PinYieldNote {
    /// Log that `request_id`'s pin yielded to the residue of its room over
    /// `ranks` ranks: at WARN on the connection's first yield, at DEBUG
    /// after. Returns the level it logged at, which the note's own test reads
    /// without a subscriber.
    pub(crate) fn log(&self, request_id: &str, pin: i32, room: i64, ranks: u32) -> Level {
        // A load first: after the first yield this is the per-request path.
        let first =
            !self.warned.load(Ordering::Relaxed) && !self.warned.swap(true, Ordering::Relaxed);
        if first {
            warn!(
                request_id,
                data_parallel_rank = pin,
                bootstrap_room = room,
                dp_size = ranks,
                "data_parallel_rank conflicts with the bootstrap_room residue; the room stays \
                 authoritative on a disaggregated request, the pin is ignored; later conflicts \
                 on this connection are logged at debug"
            );
            Level::WARN
        } else {
            debug!(
                request_id,
                data_parallel_rank = pin,
                bootstrap_room = room,
                dp_size = ranks,
                "data_parallel_rank conflicts with the bootstrap_room residue; the room stays \
                 authoritative on a disaggregated request, the pin is ignored"
            );
            Level::DEBUG
        }
    }

    /// Whether the connection has logged a yield: the translation tests read
    /// that the placement went through the note.
    #[cfg(test)]
    pub(crate) fn noted(&self) -> bool {
        self.warned.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The connection's first yield is the WARN line, every later one a
    /// DEBUG line; a new connection warns again.
    #[test]
    fn a_connection_warns_on_its_first_yield_and_debugs_the_rest() {
        let connection = PinYieldNote::default();
        assert!(!connection.noted());
        let levels: Vec<Level> = (0..1_000)
            .map(|n| connection.log(&format!("r{n}"), 0, 7, 2))
            .collect();
        assert_eq!(levels[0], Level::WARN);
        assert!(levels[1..].iter().all(|level| *level == Level::DEBUG));
        assert!(connection.noted());
        assert_eq!(PinYieldNote::default().log("fresh", 0, 7, 2), Level::WARN);
    }
}
