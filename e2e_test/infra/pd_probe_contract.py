"""Validate required startup/admission evidence in a bounded prefix, never completeness.

The caller supplies expected_root from its own installed-source guard. No backend
is imported here; CPU contract stubs are not runtime engine evidence.
"""

import re
from pathlib import Path

from .pd_scheduler_compat import SOURCE_SHA256


def validate_prefix(files, workers, identity, expected_root):
    remaining = list(workers)
    if (
        len(files) != 2
        or len(remaining) != 2
        or {w.worker_type.value for w in remaining} != {"prefill", "decode"}
    ):
        raise ValueError("expected one worker per role")
    launches = [w.observer_instance for w in remaining]
    if (
        any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None
            for value in launches
        )
        or len(set(launches)) != 2
    ):
        raise ValueError("missing, invalid or duplicate worker launch identity")
    instances = set()
    for name, (raw, records) in files.items():
        if (
            not records
            or not isinstance(records[0], dict)
            or records[0].get("event") != "startup"
            or sum(r.get("event") == "startup" for r in records if isinstance(r, dict)) != 1
        ):
            raise ValueError("missing startup")
        startup = records[0]
        matching = [w for w in remaining if w.observer_instance == startup.get("worker_instance")]
        if len(matching) != 1:
            raise ValueError("unknown or duplicate worker instance")
        worker = matching[0]
        remaining.remove(worker)
        ctx, instance = worker.scheduler_observation, startup.get("instance")
        if (ctx.run, ctx.attempt) != (identity.run, identity.attempt):
            raise ValueError("cross-fleet request identity")
        if not isinstance(instance, str) or re.fullmatch(r"[0-9a-f]{32}", instance) is None:
            raise ValueError("unknown scheduler instance")
        if instance in instances or name != f"scheduler-{startup.get('pid')}-{instance}.jsonl":
            raise ValueError("duplicate scheduler or unknown filename")
        instances.add(instance)
        expected = dict(
            run=ctx.run,
            attempt=ctx.attempt,
            worker=ctx.worker,
            role=worker.worker_type.value,
            worker_instance=worker.observer_instance,
            instance=instance,
            pid=startup.get("pid"),
            schema=1,
            observation="partial",
        )
        if type(startup.get("pid")) is not int or startup["pid"] <= 0:
            raise ValueError("unknown scheduler PID")
        for seq, record in enumerate(records, 1):
            if (
                not isinstance(record, dict)
                or type(record.get("schema")) is not int
                or type(record.get("pid")) is not int
                or type(record.get("seq")) is not int
                or record["seq"] != seq
                or any(record.get(k) != v for k, v in expected.items())
                or record.get("event")
                not in {
                    "startup",
                    "add_request_return",
                    "first_scheduled",
                    "finish_requests_return",
                    "free_request_return",
                    "free_blocks_return",
                    "free_request_blocks_return",
                    "drain_deferred_return",
                }
            ):
                raise ValueError("unknown trace identity/sequence")
        asynchronous = startup.get("async_scheduling")
        native = "async_scheduler.AsyncScheduler" if asynchronous else "scheduler.Scheduler"
        observed = "ObservedAsyncScheduler" if asynchronous else "ObservedScheduler"
        contract = startup.get("source_contract", {})
        if (
            type(asynchronous) is not bool
            or contract.get("sources") != SOURCE_SHA256
            or not isinstance(contract.get("root"), str)
            or not Path(contract["root"]).is_absolute()
            or contract["root"] != expected_root
            or startup.get("mro")
            != [
                f"infra.pd_vllm_scheduler.{observed}",
                "infra.pd_scheduler_observer.SchedulerObserverMixin",
                f"vllm.v1.core.sched.{native}",
            ]
            + (["vllm.v1.core.sched.scheduler.Scheduler"] if asynchronous else [])
            + ["vllm.v1.core.sched.interface.SchedulerInterface", "abc.ABC", "builtins.object"]
        ):
            raise ValueError("unknown source/native scheduler contract")
        admissions = [
            r
            for r in records
            if r["event"] == "add_request_return"
            and identity.request_id in str(r.get("request", {}).get("request_id", ""))
        ]
        if len(admissions) != 1:
            raise ValueError("missing or repeated target admission")
        admission_seq = admissions[0]["seq"]
        internal = identity.admitted_id(
            admissions[0], worker=ctx.worker, role=worker.worker_type.value, instance=instance
        )
        scheduled = [
            (r["seq"], entry)
            for r in records
            if r["event"] == "first_scheduled"
            for entry in r["requests"]
            if entry["request"]["request_id"] == internal
        ]
        if len(scheduled) != 1 or scheduled[0][0] <= admission_seq:
            raise ValueError("missing or repeated first scheduling")
        scheduled = scheduled[0][1]
        for entry in [admissions[0], scheduled]:
            snapshot = entry["request"]
            if (
                not isinstance(snapshot.get("status"), str)
                or not snapshot["status"]
                or type(snapshot.get("abort_immediately")) is not bool
                or any(
                    type(snapshot.get(k)) is not int or snapshot[k] < 0
                    for k in (
                        "num_computed_tokens",
                        "num_output_tokens",
                        "num_prompt_tokens",
                        "num_output_placeholders",
                        "last_sched_seq",
                        "max_tokens",
                    )
                )
            ):
                raise ValueError("invalid request snapshot")
        if type(scheduled.get("scheduled_tokens")) is not int or scheduled["scheduled_tokens"] <= 0:
            raise ValueError("request was not scheduled")
