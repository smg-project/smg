"""Silence LiteLLM's per-response "This model isn't mapped yet" ERROR.

tau2 asks litellm for the dollar cost of every completion. The self-hosted model
under test (served via a local vLLM/SMG OpenAI endpoint) has no entry in litellm's
public price map, so each call logs a noisy ERROR and reports cost $0 — thousands
of lines per leg, drowning the arm logs. Registering the model at zero cost makes
the lookup succeed, so the noise disappears with no effect on rewards/pass^k
(cost is already effectively 0 and not part of the A/B gate).

This file is a `sitecustomize` module: Python's `site` imports it automatically at
interpreter startup when its directory is on PYTHONPATH. The nightly puts this dir
on PYTHONPATH so the registration runs inside the tau2 subprocess (its own venv)
where litellm actually computes cost. The model is read from $TAU2_COST_FREE_MODEL
(falling back to $MODEL) so it stays leg-agnostic across the matrix. Registration
is best-effort; opt-in budget enforcement rejects unverified requests.
"""

import contextlib
import os
import sys
from pathlib import Path

guard_mode = os.environ.get("SMG_TAU2_BUDGET_GUARD")
guard_enabled = guard_mode not in {None, "", "0"}
if guard_enabled and guard_mode != "1":
    raise SystemExit("SMG_TAU2_BUDGET_GUARD must be 0 or 1")

try:
    import litellm
except Exception as exc:
    # run_ab.py, launch_arms.sh's port picker, and vLLM/SMG share this PYTHONPATH.
    # Enable the guard only in the tau2 subprocess env, never the workflow step.
    if guard_enabled:
        raise SystemExit("Cannot initialize the tau2 budget guard without LiteLLM") from exc
    if not isinstance(exc, ImportError):
        raise
    litellm = None


def _register_zero_cost() -> None:
    model = os.environ.get("TAU2_COST_FREE_MODEL") or os.environ.get("MODEL")
    if not (model and litellm):
        return
    with contextlib.suppress(Exception):
        litellm.register_model(
            {
                model: {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "litellm_provider": "openai",
                    "mode": "chat",
                }
            }
        )


_register_zero_cost()

if guard_enabled:
    try:
        original_completion = litellm.completion

        def _deny_uninitialized(*args, **kwargs):
            raise RuntimeError("The tau2 budget hook did not initialize")

        litellm.completion = _deny_uninitialized
        original_path = sys.path[:]
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from budget_hook import install

            install(litellm, original=original_completion)
        finally:
            sys.path[:] = original_path
    except Exception as exc:
        # sitecustomize errors normally allow startup to continue; SystemExit does not.
        raise SystemExit("Cannot initialize the tau2 budget hook") from exc
