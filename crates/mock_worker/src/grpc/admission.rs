//! Cancellation retained from the first admission wait through engine submission.
use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex},
};

use tokio::sync::watch;
use tonic::Status;

type Waiter = Arc<watch::Sender<bool>>;
type Pending = HashMap<String, Vec<Waiter>>;

#[derive(Clone, Default)]
pub(super) struct Admissions(Arc<Mutex<Pending>>);

impl Admissions {
    pub(super) fn register(&self, id: String) -> Admission {
        let (cancel, cancelled) = watch::channel(false);
        let cancel = Arc::new(cancel);
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(id.clone())
            .or_default()
            .push(Arc::clone(&cancel));
        Admission {
            registry: self.clone(),
            id,
            cancel,
            cancelled,
        }
    }

    /// Keep abort ordered with submission; unknown IDs leave no tombstones.
    pub(super) fn abort(&self, id: &str, abort_engine: impl FnOnce()) {
        let pending = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(waiters) = pending.get(id) {
            for waiter in waiters {
                waiter.send_replace(true);
            }
        }
        abort_engine();
    }
}

pub(super) struct Admission {
    registry: Admissions,
    id: String,
    cancel: Waiter,
    cancelled: watch::Receiver<bool>,
}

impl Admission {
    /// Wake immediately when cancelled, including while a fault delay is active.
    pub(super) async fn wait(&mut self, ready: impl Future<Output = ()>) -> Result<(), Status> {
        tokio::select! {
            biased;
            _ = self.cancelled.wait_for(|cancelled| *cancelled) => Err(Status::cancelled("request aborted before engine submission")),
            () = ready => Ok(()),
        }
    }

    /// The cancellation check and synchronous engine send share the abort lock.
    pub(super) fn submit(&self, submit: impl FnOnce()) -> Result<(), Status> {
        let _pending = self.registry.0.lock().unwrap_or_else(|p| p.into_inner());
        if *self.cancelled.borrow() {
            return Err(Status::cancelled(
                "request aborted before engine submission",
            ));
        }
        submit();
        Ok(())
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        let mut pending = self.registry.0.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(waiters) = pending.get_mut(&self.id) {
            waiters.retain(|waiter| !Arc::ptr_eq(waiter, &self.cancel));
            if waiters.is_empty() {
                pending.remove(&self.id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abort_cannot_be_lost_between_admission_and_submission() {
        let admissions = Admissions::default();
        let request = admissions.register("cancel-first".into());
        admissions.abort("cancel-first", || {});
        assert_eq!(
            request
                .submit(|| panic!("cancelled request submitted"))
                .unwrap_err()
                .code(),
            tonic::Code::Cancelled
        );
        drop(request);

        let request = admissions.register("submit-first".into());
        let mut order = Vec::new();
        request.submit(|| order.push("submit")).unwrap();
        admissions.abort("submit-first", || order.push("abort"));
        assert_eq!(order, ["submit", "abort"]);
        drop(request);
        assert!(admissions.0.lock().unwrap().is_empty());
    }

    #[test]
    fn dropped_and_duplicate_scopes_do_not_leak_or_remove_each_other() {
        let admissions = Admissions::default();
        let first = admissions.register("shared".into());
        let second = admissions.register("shared".into());
        drop(first);
        admissions.abort("shared", || {});
        assert!(second
            .submit(|| panic!("second scope lost its cancellation"))
            .is_err());
        drop(second);
        admissions.abort("unknown", || {});
        let reused = admissions.register("shared".into());
        reused.submit(|| {}).unwrap();
        drop(reused);
        assert!(admissions.0.lock().unwrap().is_empty());
    }
}
