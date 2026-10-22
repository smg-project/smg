"""Explicit vLLM 0.31 entry points. No fixture enables these prototype classes."""

from .pd_scheduler_compat import require_installed_contract, require_loaded_contract

_SOURCE_CONTRACT = require_installed_contract()

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

require_loaded_contract(_SOURCE_CONTRACT, Scheduler, AsyncScheduler)

from .pd_scheduler_observer import SchedulerObserverMixin


class ObservedScheduler(SchedulerObserverMixin, Scheduler):
    _observer_expected_async = False
    _observer_source_contract = _SOURCE_CONTRACT


class ObservedAsyncScheduler(SchedulerObserverMixin, AsyncScheduler):
    _observer_expected_async = True
    _observer_source_contract = _SOURCE_CONTRACT


def observer_class_for_config(vllm_config):
    """Call only after native full config/platform resolution, before overriding it."""
    config = vllm_config.scheduler_config
    if type(getattr(config, "async_scheduling", None)) is not bool:
        raise ValueError("unresolved native scheduler configuration")
    native = config.get_scheduler_cls()
    if native is Scheduler:
        return ObservedScheduler
    if native is AsyncScheduler:
        return ObservedAsyncScheduler
    raise ValueError("unsupported actual native scheduler class")
