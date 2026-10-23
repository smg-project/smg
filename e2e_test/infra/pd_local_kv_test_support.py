"""CPU owners around the preserved native allocator and scheduler methods."""

from collections import deque
from types import SimpleNamespace as NS

from .pd_local_kv_native_fixture import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    NativeBlockPool,
    NativeCoordinator,
    NativeKVManager,
    NativeScheduler,
    NativeSingleType,
)
from .pd_scheduler_observer import SchedulerObserverMixin
from .pd_scheduler_test_support import Status, request
from .pd_scheduler_trace import SchedulerTrace


def observed(*, deferred=False, shared=False, cached=False):
    rows = []
    trace = SchedulerTrace(rows.append, {"role": "decode", "instance": "cpu"})

    class Native(NativeScheduler):
        def __init__(self, config):
            self.scheduler_config = config
            self.requests = {}
            self.deferred_frees = deque()
            self.sched_step_seq = 4
            self.processed_step_seq = 2
            self.defer_block_free = deferred
            pool = NativeBlockPool()
            pool.num_gpu_blocks = 4
            pool.enable_caching = True
            pool.blocks = [KVCacheBlock(index, pool=pool) for index in range(4)]
            pool.free_block_queue = FreeKVCacheBlockQueue(pool.blocks)
            pool.free_block_queue.popleft().is_null = True
            self.target = pool.free_block_queue.popleft()
            self.target.ref_cnt = 2 if shared else 1
            if cached:
                self.target.set_block_hash(b"cpu-prefix")
            single = NativeSingleType()
            single.block_pool = pool
            single.req_to_blocks = {}
            single.num_cached_block = {}
            single._partial_hit_reqs = {}
            coordinator = NativeCoordinator()
            coordinator.single_type_managers = [single]
            manager = NativeKVManager()
            manager.coordinator = coordinator
            manager.block_pool = pool
            manager.create_kv_cache_blocks = lambda groups: NS(blocks=groups)
            self.kv_cache_manager = manager
            self.owners = single.req_to_blocks

    class Observed(SchedulerObserverMixin, Native):
        _observer_expected_async = False

        def _make_observer_trace(self):
            return trace

    scheduler = Observed(NS(async_scheduling=False))
    req = request()
    req.status = Status.FINISHED_ABORTED
    req.last_sched_seq = 4
    req.max_tokens = 1024
    req.is_finished = lambda: True
    scheduler.requests[req.request_id] = req
    scheduler.owners[req.request_id] = [scheduler.target]
    if shared:
        scheduler.owners["another-request"] = [scheduler.target]
    return scheduler, req, rows
