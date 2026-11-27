use std::sync::Arc;

use smg_worker::identity::basic::Worker;

use super::{PrefillAdmissionAttempt, PrefillSelectionContext};

/// The attempt a test makes when it wants exactly `worker`: it is selected,
/// with a unit payload, whenever it has capacity.
pub(crate) fn select_worker(
    worker: &Arc<dyn Worker>,
) -> impl for<'a> FnMut(&PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<()> + Send {
    let worker = Arc::clone(worker);
    move |capacity| capacity.select(Arc::clone(&worker), ())
}
