//! Step to update worker properties.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use openai_protocol::worker::{WorkerModels, WorkerSpec, WorkerStatus};
use tracing::{debug, info};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

#[cfg(test)]
use super::create_worker::labels_for_rank;
use super::create_worker::{
    model_card_for_labels, report_for_rank, validate_parser_overrides,
    warn_on_conflicting_parser_overrides, DP_WIDTH_LABELS,
};
use crate::{
    service_discovery::DISCOVERY_PROVIDER_LABEL,
    worker::{
        basic::merge_labels, overload::OverloadThresholds, service::MAX_WORKER_LABELS, BasicWorker,
        BasicWorkerBuilder, RuntimeType, Worker,
    },
    workflow::data::WorkerUpdateWorkflowData,
};

/// Step to update worker properties.
///
/// This step creates new worker instances with updated properties and
/// re-registers them to replace the old workers in the registry. The model
/// card registration built from a worker's labels follows the labels it has
/// after the update (`models_for_labels`).
pub struct UpdateWorkerPropertiesStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdateWorkerPropertiesStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let request = &context.data.config;
        // A label refresh replaces what discovery read before with what the
        // engine reports now (an operator's own labels keep their values) and
        // rebuilds a worker only when that changes something; an API update
        // merges the request's labels into the existing ones.
        let refresh = context.data.refresh_discovered_labels;
        // The engine's KV transfer metadata travels in its report under label
        // names and is never a label (registration takes it out the same way,
        // `create_worker::take_kv_transfer_metadata`): out of the report before
        // any merge, applied to the worker's dedicated fields below.
        let (fresh_engine_labels, fresh_kv) = match context.data.fresh_engine_labels.clone() {
            Some(mut fresh) => {
                let kv = EngineKvMetadata::take(&mut fresh);
                (Some(fresh), kv)
            }
            None => (None, EngineKvMetadata::default()),
        };
        let snapshot = context.data.snapshot.clone();
        if refresh && fresh_engine_labels.is_none() {
            debug!("No engine labels were re-read; nothing to update");
            context.data.updated_workers = Some(Vec::new());
            return Ok(StepResult::Skip);
        }
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();
        let workers_to_update = context
            .data
            .workers_to_update
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_update".to_string()))?
            .clone();

        debug!(
            "Updating properties for {} worker(s)",
            workers_to_update.len()
        );

        let mut updated_workers: Vec<Arc<dyn Worker>> = Vec::with_capacity(workers_to_update.len());

        // The labels each worker ends up with: the request's merged into its
        // own. The admin check held the cap against the labels the worker had
        // when the request arrived; updates queued together each passed it,
        // so it is held again here, over the labels actually merged, and only
        // when the merge adds keys: a worker over the cap through the labels
        // discovered at registration still takes a rewrite of its keys and an
        // update of its other properties. Every rank is checked before any is
        // replaced, so a refusal leaves no partial update behind. The model
        // card follows the labels, and a parser override the labels now name
        // must be one the gateway can read with, as at registration.
        let merged = workers_to_update
            .iter()
            .map(|worker| {
                let existing = &worker.metadata().spec.labels;
                let updated_labels = if let (true, Some(fresh)) =
                    (refresh, fresh_engine_labels.as_ref())
                {
                    let labels = refreshed_labels(
                        existing,
                        &worker.metadata().discovered_labels,
                        &fresh_in_rank_space(worker, fresh),
                    );
                    // An engine that reports a `constrained_decoding` naming no
                    // policy is refused as at registration: the job fails with
                    // the reason and the worker keeps the labels it had.
                    super::create_worker::validate_constrained_decoding_label(
                        &labels,
                        worker.url(),
                        "update_worker_properties",
                    )?;
                    labels
                } else {
                    // A window set here is the worker's only window, as at
                    // registration (see `create_worker::merge_labels`).
                    let updated_labels = match request.labels {
                        Some(ref new_labels) => {
                            // The admin check refused labels for a discovered
                            // worker when the request arrived, but a provider
                            // can register the URL between then and this job
                            // (its upsert queued first): held again against
                            // the worker the update would change.
                            if let Some(provider) = existing.get(DISCOVERY_PROVIDER_LABEL) {
                                return Err(WorkflowError::StepFailed {
                                    step_id: StepId::new("update_worker_properties"),
                                    message: format!(
                                        "worker {} is owned by the '{provider}' discovery provider \
                                         since this update was accepted; its labels come from its \
                                         discovery source",
                                        worker.url()
                                    ),
                                });
                            }
                            merge_labels(existing, new_labels)
                        }
                        None => existing.clone(),
                    };
                    // A `constrained_decoding` label sent in the update must name a
                    // policy, as at registration (`create_worker`): a typo would
                    // otherwise read as `requested` from here on.
                    super::create_worker::validate_constrained_decoding_label(
                        &updated_labels,
                        worker.url(),
                        "update_worker_properties",
                    )?;
                    if updated_labels.len() > existing.len()
                        && updated_labels.len() > MAX_WORKER_LABELS
                    {
                        return Err(WorkflowError::StepFailed {
                            step_id: StepId::new("update_worker_properties"),
                            message: format!(
                                "{} labels on worker {} after this update ({} now, {} new); at most {MAX_WORKER_LABELS} are accepted",
                                updated_labels.len(),
                                worker.url(),
                                existing.len(),
                                updated_labels.len() - existing.len()
                            ),
                        });
                    }
                    updated_labels
                };
                let (models, card_fields) =
                    models_for_labels(worker.as_ref(), existing, &updated_labels);
                // A parser override the labels now name is checked as at
                // registration: it must be one the gateway reads with, and a
                // peer of the model with another one is warned about.
                if card_fields
                    .iter()
                    .any(|field| matches!(*field, "tool_parser" | "reasoning_parser"))
                {
                    for card in models.iter() {
                        validate_parser_overrides(
                            card,
                            worker.url(),
                            app_context.router_config.parser,
                            app_context.tool_parser_factory.as_ref(),
                            app_context.reasoning_parser_factory.as_ref(),
                        )
                        .map_err(|message| WorkflowError::StepFailed {
                            step_id: StepId::new("update_worker_properties"),
                            message,
                        })?;
                        warn_on_conflicting_parser_overrides(
                            card,
                            worker.base_url(),
                            &app_context.worker_registry,
                        );
                    }
                }
                Ok((updated_labels, models, card_fields))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (index, (worker, (updated_labels, models, card_fields))) in
            workers_to_update.iter().zip(merged).enumerate()
        {
            if refresh {
                // The engine's id is the worker's live id from here, rebuilt
                // or not (a rebuilt worker shares the cell): the id a handoff
                // must target is the one the engine reports, as the recovery
                // re-read holds.
                fresh_kv.apply_engine_id(worker);
            }
            if refresh
                && updated_labels == worker.metadata().spec.labels
                && !fresh_kv.fills_a_field_of(&worker.metadata().spec)
            {
                debug!(
                    worker_url = %worker.url(),
                    "Engine labels unchanged; the worker is kept as it is"
                );
                continue;
            }
            if !card_fields.is_empty() {
                info!(
                    worker_url = %worker.url(),
                    fields = ?card_fields,
                    "The worker's model card follows its labels"
                );
            }
            // A key an update sets is the operator's from then on, whatever
            // value it carries: it leaves the discovered set, so a later
            // refresh keeps it even when the engine reports another value.
            let discovered_labels = match (refresh, fresh_engine_labels.as_ref()) {
                // A refresh records the engine's report minus the keys the
                // operator owns (a value that differs from the recorded report,
                // or a key the report never had), so ownership survives the
                // rebuilds between an operator's setting and the engine's next
                // change of that value.
                (true, Some(fresh)) => {
                    let before = &worker.metadata().discovered_labels;
                    let current = &worker.metadata().spec.labels;
                    fresh_in_rank_space(worker, fresh)
                        .into_iter()
                        .filter(|(key, _)| {
                            !current.contains_key(key) || before.get(key) == current.get(key)
                        })
                        .collect()
                }
                (false, _) => {
                    let mut discovered_labels = (*worker.metadata().discovered_labels).clone();
                    if let Some(labels) = &request.labels {
                        for key in labels.keys() {
                            discovered_labels.remove(key);
                        }
                    }
                    discovered_labels
                }
                _ => (*worker.metadata().discovered_labels).clone(),
            };
            // Resolve priority and cost: use update value if specified, otherwise keep existing
            let updated_priority = request.priority.unwrap_or_else(|| worker.priority());
            let updated_cost = request.cost.unwrap_or_else(|| worker.cost());

            // Build updated health config from resolved runtime config
            let existing_health = &worker.metadata().health_config;
            let updated_health_config = match &request.health {
                Some(update) => update.apply_to(existing_health),
                None => existing_health.clone(),
            };

            // Determine API key: use new one if provided, otherwise keep existing
            let updated_api_key = request
                .api_key
                .clone()
                .or_else(|| worker.metadata().spec.api_key.clone());

            // Keep the full spec so a metadata update preserves PD transport
            // and DP configuration.
            //
            // Status: a metadata-only update is not a re-registration — the
            // worker is the same endpoint. Preserve the old status so a
            // healthy worker stays routable through the update. The one
            // exception is when the update flips `disable_health_check` to
            // `true`: the health checker skips disabled workers, so a stale
            // Pending/NotReady status would never recover. Force Ready in
            // that case.
            let next_status = if updated_health_config.disable_health_check {
                WorkerStatus::Ready
            } else {
                worker.status()
            };
            let mut builder = BasicWorkerBuilder::carrying(worker)
                .labels(updated_labels)
                .models(models)
                .discovered_labels(discovered_labels)
                .health_config(updated_health_config.clone())
                .priority(updated_priority)
                .cost(updated_cost)
                // The effective overload thresholds re-resolve against the
                // same gateway defaults registration used.
                .overload_defaults(OverloadThresholds::from_gateway_config(
                    &app_context.router_config,
                ))
                .status(next_status);
            if refresh {
                // A connector or role the operator configured none of follows
                // the engine's report; a configured one stands, as at
                // registration.
                builder = fresh_kv.fill_fields(builder, &worker.metadata().spec);
            }

            // Adopt the old worker's client only if it was materialized: a
            // never-HTTP worker (ZMQ) keeps a deferred slot instead of having
            // a client it will never use built for the replacement.
            if let Some(client) = worker.http_client_handle_if_initialized() {
                builder = builder.http_client(client);
            }

            if let Some(ref api_key) = updated_api_key {
                builder = builder.api_key(api_key.clone());
            }

            // The spec retains the ZMQ address and engine count. The connect
            // signal still has to come from the registry.
            builder =
                builder.connect_signal_tx(app_context.worker_registry.connect_signal_sender());

            let mut new_worker = builder.build();
            if let Some(previous) = worker.as_any().downcast_ref::<BasicWorker>() {
                // A health probe may finish after this update. Share its state
                // so the result reaches the replacement too.
                new_worker.share_kv_engine_state(previous);
            }
            let new_worker: Arc<dyn Worker> = Arc::new(new_worker);

            // Replace the worker in the registry (overwrite-then-diff),
            // against the id and revision the find step captured: a worker
            // updated or replaced meanwhile (a provider registering its
            // address included) fails the guard and is left as it is, and one
            // removed (or removed and registered again at the same address,
            // under a new id) is neither brought back nor touched. A refresh
            // is dropped, and the next transition into service refreshes
            // again; an update fails, rather than overwrite the newer
            // registration with a rebuild of the one it found.
            let worker_id = match snapshot.as_ref().and_then(|entries| entries.get(index)) {
                Some((id, expected))
                    if app_context.worker_registry.replace_claiming_local(
                        id,
                        new_worker.clone(),
                        *expected,
                    ) =>
                {
                    id.clone()
                }
                _ if refresh => {
                    debug!(
                        worker_url = %worker.url(),
                        "Worker changed or left during the label refresh; not rebuilt"
                    );
                    continue;
                }
                _ => {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("update_worker_properties"),
                        message: format!(
                            "worker {} was replaced or removed since this update found it, so \
                             the update was dropped",
                            worker.url()
                        ),
                    });
                }
            };

            // The rebuilt handle stands at the old one's URL: tell the policy
            // registry, as a removal would, so the engine's rows of the
            // retired handle do not outlive it.
            app_context
                .policy_registry
                .on_worker_replaced(worker.as_ref(), new_worker.as_ref());

            // Same-URL replace preserves the existing shared runtime. If the
            // update disables health checks, force the lifecycle to Ready so
            // the worker does not stay stuck in a non-routable pre-update
            // status while the manager correctly skips future probes.
            if updated_health_config.disable_health_check {
                let _ = app_context
                    .worker_registry
                    .transition_status(&worker_id, WorkerStatus::Ready);
            }

            updated_workers.push(new_worker);
        }

        // Log result
        if updated_workers.len() == 1 {
            info!("Updated worker {}", updated_workers[0].url());
        } else {
            info!("Updated {} workers", updated_workers.len());
        }

        // Store updated workers for subsequent steps
        context.data.updated_workers = Some(updated_workers);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// The worker's model cards for the labels it has now. Registration builds a
/// local worker's one card from its labels (`build_model_card`), and that card
/// follows them (`model_card_for_labels`); what the operator set by hand
/// stands. The cards of an external worker come from the provider's model
/// list, and a list of cards from elsewhere too, so those are left as they
/// are. Also the names of the fields that changed.
fn models_for_labels(
    worker: &dyn Worker,
    labels_before: &HashMap<String, String>,
    labels_now: &HashMap<String, String>,
) -> (WorkerModels, Vec<&'static str>) {
    let spec = &worker.metadata().spec;
    match &spec.models {
        WorkerModels::Single(card) if spec.runtime_type != RuntimeType::External => {
            let (card, changed) = model_card_for_labels(card, labels_before, labels_now);
            (WorkerModels::Single(Box::new(card)), changed)
        }
        models => (models.clone(), Vec::new()),
    }
}

/// The engine's KV transfer metadata, which its report carries under label
/// names (`kv_connector`, `kv_role`, `kv_engine_id`) and the worker holds in
/// dedicated fields, as at registration (`create_worker::take_kv_transfer_metadata`).
#[derive(Debug, Default)]
struct EngineKvMetadata {
    connector: Option<String>,
    role: Option<String>,
    engine_id: Option<String>,
}

impl EngineKvMetadata {
    /// Takes the three keys out of the report; an empty value is none.
    fn take(report: &mut HashMap<String, String>) -> Self {
        let mut take = |key: &str| report.remove(key).filter(|value| !value.is_empty());
        Self {
            connector: take("kv_connector"),
            role: take("kv_role"),
            engine_id: take("kv_engine_id"),
        }
    }

    /// Whether a rebuild would set a field the operator left unset.
    fn fills_a_field_of(&self, spec: &WorkerSpec) -> bool {
        (self.connector.is_some() && spec.kv_connector.is_none())
            || (self.role.is_some() && spec.kv_role.is_none())
    }

    /// The fields the operator left unset, from the engine's report.
    fn fill_fields(
        &self,
        mut builder: BasicWorkerBuilder,
        spec: &WorkerSpec,
    ) -> BasicWorkerBuilder {
        if let (Some(connector), None) = (&self.connector, &spec.kv_connector) {
            builder = builder.kv_connector(connector);
        }
        if let (Some(role), None) = (&self.role, &spec.kv_role) {
            builder = builder.kv_role(role);
        }
        builder
    }

    /// A reported engine id is the worker's live id, confirmed by this read.
    fn apply_engine_id(&self, worker: &Arc<dyn Worker>) {
        let Some(engine_id) = &self.engine_id else {
            return;
        };
        let previous = worker.kv_engine_id();
        if worker.refresh_kv_engine_id(Some(engine_id.clone())) {
            info!(
                worker_url = %worker.url(),
                ?previous,
                engine_id,
                "The worker's KV engine id follows its engine's report"
            );
        }
        worker.set_kv_engine_id_confirmed(true);
    }
}

/// The engine's fresh report in the worker's rank space. The width comes
/// from the labels the worker's own share was sized with: a DP label the
/// operator set (its value differs from what discovery recorded, or
/// discovery never read it) over the engine's report, so the record and the
/// worker's labels agree on the share whichever side names the width.
fn fresh_in_rank_space(
    worker: &Arc<dyn Worker>,
    fresh: &HashMap<String, String>,
) -> HashMap<String, String> {
    let current = &worker.metadata().spec.labels;
    let before = &worker.metadata().discovered_labels;
    let mut width_source = fresh.clone();
    for key in DP_WIDTH_LABELS {
        if let Some(value) = current.get(key) {
            if before.get(key) != Some(value) {
                width_source.insert(key.to_string(), value.clone());
            }
        }
    }
    report_for_rank(fresh, worker.dp_rank().zip(worker.dp_size()), &width_source)
}

/// The labels a worker carries once its engine reports `fresh`: every label
/// whose value came from discovery (`discovered_before`, still unchanged
/// since) follows the engine, disappearing when the engine no longer reports
/// it; a label the operator set or overrode (its value differs from what
/// discovery read, or discovery never read it) keeps its value; a key the
/// engine reports for the first time is added unless the operator already
/// set it. The result then passes through the merge registration and a
/// label update make (`merge_labels`, the operator's labels on top): a
/// running window the operator set by hand stays the worker's only window,
/// so the engine's figure under the other window name, which the reader
/// would prefer, is not added beside it.
pub(crate) fn refreshed_labels(
    current: &HashMap<String, String>,
    discovered_before: &HashMap<String, String>,
    fresh: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut labels = current.clone();
    for (key, before) in discovered_before {
        if current.get(key) != Some(before) {
            // The operator overrode (or removed) it: theirs stands.
            continue;
        }
        match fresh.get(key) {
            Some(now) => {
                labels.insert(key.clone(), now.clone());
            }
            None => {
                labels.remove(key);
            }
        }
    }
    for (key, now) in fresh {
        if !discovered_before.contains_key(key) && !current.contains_key(key) {
            labels.insert(key.clone(), now.clone());
        }
    }
    let by_hand: HashMap<String, String> = current
        .iter()
        .filter(|(key, value)| discovered_before.get(*key) != Some(*value))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    merge_labels(&labels, &by_hand)
}

#[cfg(test)]
mod tests;
