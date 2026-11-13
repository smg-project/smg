//! Coverage-aware local admission deltas over one encoder instance's WatchLoad.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use smg_grpc_client::sglang_managed::proto;
use tonic::Status;

struct Pending {
    work: u64,
    admission_seq: Option<u64>,
}

pub struct LoadTracker {
    instance: proto::Instance,
    snapshot: Option<(proto::LoadSnapshot, Instant)>,
    pending: HashMap<String, Pending>,
    max_pending: usize,
}

impl LoadTracker {
    /// A restarted worker gets a new tracker; old unknown work belongs to its old instance.
    pub fn new(instance: proto::Instance, max_pending: usize) -> Self {
        Self {
            instance,
            snapshot: None,
            pending: HashMap::new(),
            max_pending,
        }
    }

    /// Register in-flight work before sending Encode, including unknown acceptance outcomes.
    pub fn reserve(&mut self, operation_id: String, work: u64) -> Result<(), Status> {
        if self.pending.contains_key(&operation_id) {
            return Err(Status::already_exists("encode reservation exists"));
        }
        if self.pending.len() >= self.max_pending {
            return Err(Status::resource_exhausted("encode reservations full"));
        }
        self.pending.insert(
            operation_id,
            Pending {
                work,
                admission_seq: None,
            },
        );
        Ok(())
    }

    /// Accepted/GetOperation reconcile with an already-arrived covering snapshot too.
    pub fn accepted(
        &mut self,
        operation_id: &str,
        instance: &proto::Instance,
        seq: u64,
    ) -> Result<(), Status> {
        if instance != &self.instance || seq == 0 {
            return Err(Status::failed_precondition(
                "stale admission acknowledgement",
            ));
        }
        if self
            .snapshot
            .as_ref()
            .is_some_and(|(s, _)| seq <= s.covered_admission_seq)
        {
            self.pending.remove(operation_id);
            return Ok(());
        }
        let pending = self
            .pending
            .get_mut(operation_id)
            .ok_or_else(|| Status::not_found("unknown encode reservation"))?;
        if pending.admission_seq.is_some_and(|old| old != seq) {
            return Err(Status::failed_precondition("admission sequence changed"));
        }
        pending.admission_seq = Some(seq);
        Ok(())
    }

    /// Snapshot order and admission coverage cannot regress within an instance.
    pub fn observe(&mut self, snapshot: proto::LoadSnapshot, now: Instant) -> Result<bool, Status> {
        if snapshot.instance.as_ref() != Some(&self.instance) {
            return Err(Status::failed_precondition("stale load instance"));
        }
        if let Some((old, _)) = &self.snapshot {
            if snapshot.snapshot_seq <= old.snapshot_seq {
                return Ok(false);
            }
            if snapshot.covered_admission_seq < old.covered_admission_seq {
                return Err(Status::failed_precondition("load coverage regressed"));
            }
        }
        self.pending.retain(|_, p| {
            p.admission_seq
                .is_none_or(|seq| seq > snapshot.covered_admission_seq)
        });
        self.snapshot = Some((snapshot, now));
        Ok(true)
    }

    /// Unknown acceptance stays charged; cancellation alone must never call this.
    pub fn rejected_or_quiesced(&mut self, operation_id: &str) {
        self.pending.remove(operation_id);
    }

    /// Keep CPU/GPU availability separate; unknown admissions conservatively charge both.
    /// No fresh snapshot means no normal candidate, including a just-restarted instance.
    pub fn stage_work(&self, now: Instant, max_age: Duration) -> Option<(u64, u64, u64)> {
        let (snapshot, observed) = self.snapshot.as_ref()?;
        if now.saturating_duration_since(*observed) > max_age {
            return None;
        }
        let pending = self
            .pending
            .values()
            .try_fold(0u64, |total, p| total.checked_add(p.work))?;
        Some((
            snapshot.cpu_work.checked_add(pending)?,
            snapshot.gpu_work.checked_add(pending)?,
            snapshot.send_bytes,
        ))
    }

    /// Count only local operations not covered by the same instance's snapshot.
    pub fn has_capacity(&self, now: Instant, max_age: Duration) -> bool {
        self.snapshot.as_ref().is_some_and(|(snapshot, observed)| {
            now.saturating_duration_since(*observed) <= max_age
                && snapshot.active.saturating_add(self.pending.len() as u64) < snapshot.capacity
        })
    }
}
