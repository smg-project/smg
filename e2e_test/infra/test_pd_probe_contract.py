"""CPU consumption contracts; fabricated traces never represent a backend pass."""

import copy
import json
from types import SimpleNamespace as NS

import pytest
from infra import pd_probe_contract as contract
from infra import pd_probe_prefix as prefix
from infra.constants import WorkerType
from infra.pd_request_identity import RequestIdentity


@pytest.fixture
def sample(tmp_path):
    identity, workers, records = RequestIdentity("run", "fleet", "nonce"), [], {}
    for index, role in enumerate((WorkerType.PREFILL, WorkerType.DECODE), 1):
        instance, launch = str(index) * 32, str(index + 2) * 32
        workers.append(
            NS(
                worker_type=role,
                observer_instance=launch,
                scheduler_observation=NS(run="run", attempt="fleet", worker=role.value),
            )
        )
        ctx = dict(
            run="run",
            attempt="fleet",
            worker=role.value,
            role=role.value,
            instance=instance,
            worker_instance=launch,
            pid=index,
            schema=1,
            observation="partial",
        )
        snapshot = dict(
            request_id=f"chatcmpl-nonce-{index:08x}",
            status="WAITING",
            abort_immediately=False,
            **dict.fromkeys(
                (
                    "num_computed_tokens",
                    "num_output_tokens",
                    "num_prompt_tokens",
                    "num_output_placeholders",
                    "last_sched_seq",
                    "max_tokens",
                ),
                0,
            ),
        )
        records[f"scheduler-{index}-{instance}.jsonl"] = [
            dict(
                ctx,
                seq=1,
                event="startup",
                async_scheduling=False,
                source_contract={"root": "/pinned/vllm", "sources": dict(contract.SOURCE_SHA256)},
                mro=[
                    "infra.pd_vllm_scheduler.ObservedScheduler",
                    "infra.pd_scheduler_observer.SchedulerObserverMixin",
                    "vllm.v1.core.sched.scheduler.Scheduler",
                    "vllm.v1.core.sched.interface.SchedulerInterface",
                    "abc.ABC",
                    "builtins.object",
                ],
            ),
            dict(
                ctx,
                seq=2,
                event="add_request_return",
                registered=True,
                request=copy.deepcopy(snapshot),
            ),
            dict(
                ctx,
                seq=3,
                event="first_scheduled",
                requests=[dict(request=snapshot, scheduled_tokens=1)],
            ),
        ]

    def write():
        for name, rows in records.items():
            (tmp_path / name).write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
        return prefix.read_prefix(tmp_path)

    return NS(directory=tmp_path, workers=workers, records=records, identity=identity, write=write)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_exact_native_prefix(sample, asynchronous):
    for rows in sample.records.values():
        rows[0]["async_scheduling"] = asynchronous
        if asynchronous:
            rows[0]["mro"][0] = "infra.pd_vllm_scheduler.ObservedAsyncScheduler"
            rows[0]["mro"].insert(2, "vllm.v1.core.sched.async_scheduler.AsyncScheduler")
    contract.validate_prefix(sample.write(), sample.workers, sample.identity, "/pinned/vllm")


@pytest.mark.parametrize(
    "fault",
    [
        "startup",
        "admission",
        "scheduled",
        "order",
        "duplicate",
        "retry",
        "worker",
        "worker_instance",
        "instance",
        "run",
        "attempt",
        "role",
        "schema",
        "seq",
        "pid",
        "event",
        "source",
        "mro",
        "async",
        "nonce",
        "registered",
        "tokens",
        "snapshot",
    ],
)
def test_required_contract_rejects_unknown(sample, fault):
    rows = next(iter(sample.records.values()))
    if fault in ("startup", "admission", "scheduled"):
        rows.pop({"startup": 0, "admission": 1, "scheduled": 2}[fault])
    elif fault == "order":
        rows[1], rows[2] = rows[2], rows[1]
        rows[1]["seq"], rows[2]["seq"] = 2, 3
    elif fault in ("duplicate", "retry"):
        rows.append(copy.deepcopy(rows[0 if fault == "duplicate" else 1]))
        rows[-1]["seq"] = 4
    elif fault in ("worker", "worker_instance", "instance", "run", "attempt", "role", "event"):
        rows[1][fault] = "unknown"
    elif fault in ("schema", "seq", "pid"):
        rows[1][fault] = True
    elif fault == "source":
        rows[0]["source_contract"]["sources"].pop(next(iter(contract.SOURCE_SHA256)))
    elif fault in ("mro", "async"):
        rows[0]["mro" if fault == "mro" else "async_scheduling"] = None
    elif fault == "nonce":
        rows[1]["request"]["request_id"] = "chatcmpl-nonce"
    elif fault == "registered":
        rows[1]["registered"] = 1
    elif fault == "tokens":
        rows[2]["requests"][0]["scheduled_tokens"] = 0
    else:
        rows[1]["request"]["abort_immediately"] = 0
    with pytest.raises((ValueError, TypeError)):
        contract.validate_prefix(sample.write(), sample.workers, sample.identity, "/pinned/vllm")


@pytest.mark.parametrize(
    "fault",
    [
        "early_and_late",
        "root",
        "tail_mro",
        "wrong_fleet",
        "seq_gap",
        "duplicate_seq",
        "negative_tokens",
        "bool_tokens",
        "missing_field",
    ],
)
def test_conflicting_or_incomplete_evidence_is_unknown(sample, fault):
    rows = next(iter(sample.records.values()))
    if fault == "early_and_late":
        rows.insert(1, copy.deepcopy(rows[2]))
        for seq, row in enumerate(rows, 1):
            row["seq"] = seq
    elif fault == "root":
        rows[0]["source_contract"]["root"] = "/unverified/vllm"
    elif fault == "tail_mro":
        rows[0]["mro"].append("foreign.Custom")
    elif fault == "wrong_fleet":
        sample.identity = RequestIdentity("run", "other-fleet", "nonce")
    elif fault in ("seq_gap", "duplicate_seq"):
        rows[2]["seq"] = 4 if fault == "seq_gap" else 2
    elif fault in ("negative_tokens", "bool_tokens"):
        rows[2]["requests"][0]["scheduled_tokens"] = -1 if fault == "negative_tokens" else True
    else:
        del rows[2]["requests"][0]["request"]["num_prompt_tokens"]
    with pytest.raises(ValueError):
        contract.validate_prefix(sample.write(), sample.workers, sample.identity, "/pinned/vllm")


@pytest.mark.parametrize("launch", [None, "launch", "", "a" * 31, "A" * 32])
def test_missing_or_noncanonical_worker_launch_is_unknown(sample, launch):
    sample.workers[0].observer_instance = launch
    for record in next(iter(sample.records.values())):
        if launch is None:
            record.pop("worker_instance")
        else:
            record["worker_instance"] = launch
    with pytest.raises(ValueError, match="launch identity"):
        contract.validate_prefix(sample.write(), sample.workers, sample.identity, "/pinned/vllm")
