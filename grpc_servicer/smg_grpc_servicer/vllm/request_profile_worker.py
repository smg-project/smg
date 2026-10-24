"""Worker-local torch profiling for the explicit vLLM 0.31 request-dir path."""

import copy
import os


def profile_worker(worker, action, owner, directory=None):
    """A static collective RPC; never replace a profiler owned by another caller."""
    from vllm.profiler.wrapper import TorchProfilerWrapper, validate_worker_profiler_config
    from vllm.version import __version__

    if __version__.split("+")[0] != "0.31.0":
        raise RuntimeError("Request-directory profiling requires the verified vLLM 0.31.0 API")
    state = getattr(worker, "_smg_request_profile", None)
    if action == "prepare":
        if state is not None or worker.profiler is not None:
            raise RuntimeError("A worker profiler already exists; refusing to replace it")
        if worker.profiler_config.profiler not in (None, "torch"):
            raise RuntimeError("Refusing to replace another configured profiler backend")
        if worker.profiler_config.capture_torch_profiler:
            raise RuntimeError("Request profiling cannot replace graph-capture profiling")
        config = copy.copy(worker.profiler_config)
        config.profiler, config.torch_profiler_dir = "torch", directory
        config.torch_profiler_activities = ["CPU", "CUDA"]
        config.delay_iterations = config.max_iterations = 0
        config.warmup_iterations = config.wait_iterations = 0
        validate_worker_profiler_config(config)

        class StrictTorchProfiler(TorchProfilerWrapper):
            # Upstream's _call_* log and swallow errors. This path must report them.
            def _call_start(self):
                self._start()
                self._running = True

            def _call_stop(self):
                self._stop()
                self._running = False

            def stop(self):
                try:
                    super().stop()
                except BaseException:
                    self._active = True
                    raise

        state = {"owner": owner, "config": worker.profiler_config, "phase": "preparing"}
        worker._smg_request_profile = state
        try:
            profiler = StrictTorchProfiler(
                config, worker_name=f"smg-{owner}-rank{worker.rank}",
                local_rank=worker.local_rank, activities=("CPU", "CUDA"),
            )
            state["profiler"] = profiler
            worker.profiler_config, worker.profiler = config, profiler
            state["phase"] = "prepared"
        except BaseException:
            state["phase"] = "faulted"
            raise
    else:
        if state is None or state["owner"] != owner:
            raise RuntimeError("Profiler owner differs or no request session exists")
        profiler = state.get("profiler")
        if profiler is None or worker.profiler is not profiler:
            raise RuntimeError("Worker profiler identity changed")
        try:
            if action == "started":
                if state["phase"] != "prepared" or not profiler.is_running or not profiler._active:
                    raise RuntimeError("The real worker profiler did not start")
                state["phase"] = "active"
            elif action == "finish":
                if state["phase"] != "active" or profiler.is_running or profiler._active:
                    raise RuntimeError("The real worker profiler did not stop")
                profiler.shutdown()
                if profiler.is_running or profiler._active:
                    raise RuntimeError("Worker profiler shutdown is incomplete")
                worker.profiler_config, worker.profiler = state["config"], None
                del worker._smg_request_profile
                state["phase"] = "finished"
            else:
                raise ValueError("Unknown profiler transition")
        except BaseException:
            state["phase"] = "faulted"
            raise
    return {"rank": worker.rank, "pid": os.getpid(), "owner": owner, "phase": state["phase"]}


class RequestProfileWorkerExtension:
    """Select explicitly with --worker-extension-cls; no callable deserialization."""

    def smg_request_profile(self, action, owner, directory=None):
        return profile_worker(self, action, owner, directory)
