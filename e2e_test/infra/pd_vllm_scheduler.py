"""Explicit vLLM 0.31 entry points. No fixture enables these prototype classes."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from .pd_scheduler_observer import SchedulerObserverMixin


class ObservedScheduler(SchedulerObserverMixin, Scheduler):
    _observer_expected_async = False


class ObservedAsyncScheduler(SchedulerObserverMixin, AsyncScheduler):
    _observer_expected_async = True


def observer_class_for(resolved_async_scheduling):
    """Mirror the fixed SchedulerConfig default only after config is resolved."""
    if type(resolved_async_scheduling) is not bool:
        raise ValueError("async scheduling has not been resolved")
    return ObservedAsyncScheduler if resolved_async_scheduling else ObservedScheduler
