"""Pinned, budgeted plain-text tau2 entry. No policy/evidence means no paid send."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget import BudgetError, Ledger, Policy, RunPolicy

MODELS = {"user": "gpt-5.2", "nl_judge": "gpt-4.1-2025-04-14"}
VERSIONS = {"tau2": "1.0.1", "litellm": "1.81.11", "openai": "2.20.0", "httpx": "0.28.1"}
TAU2_SHA = {
    "utils/llm_utils.py": "fa7d61e34ef9c668452ec44e8ba9efafec14a349a530481fac1dc19eff275c65",
    "user/user_simulator.py": "bfe1defe73bce25450b2be697c249d6cec2f3ba25a70e32e2452ba75a53a630c",
    "evaluator/evaluator_nl_assertions.py": "cb6b3758cdceb02a726732b4ee1af618affcc08f8ab84b44f060146b8f3ad90a",
    "runner/batch.py": "a75853aa6a982aaceb1faad4f5c96ec21545795636d68df7dcb59ea0902a39bd",
    "cli.py": "13cb80a99483f85638bd565fd6aa5e3795f7cc2fe35e48b0ecb4d794c23d8b5c",
    "config.py": "6a53f8895b3dd4bffa01c73e11d4b8755181c5bfe46aba5351f89b59143d9ae4",
}
SDK_SHA = {
    "litellm": {
        "llms/openai/chat/gpt_5_transformation.py": "4978d1eb5c558ead94776592c49c3df0c6dd026992f8118ac33a0550d56b87dc",
        "llms/openai/openai.py": "168232225255de4a7030b0340ac97c93f0cc09d40cac69a4c52bcb24311be8c7",
        "main.py": "d300782db65bc01513a26cb6cf9750c8feb30fba117c9901cc7d4163d36b8dd8",
        "utils.py": "0ce358c1c5d2f0c79e530f5d67b898ee7b12ad1436d96a3ca1069cd6ff5b8f67",
    },
    "openai": {
        "_base_client.py": "8bea1cea96dcceae9fe08910d1a8e120d322a41de6bc8691872b6789f5f624c3",
        "_client.py": "dfb5fdbe43391eb631b0b183df2a39028b9be1d5896a0c08cb6d9ae5d59ce127",
        "resources/chat/completions/completions.py": "af93382026fe187254b1bae89315ef422ce1fe206cca08915fe921a3f5fbf094",
    },
}

ROLE = ContextVar("tau2_paid_role", default=None)
TASK = ContextVar("tau2_paid_task", default=None)
LOCAL = ContextVar("tau2_allowed_local_request", default=False)


def resolve_approval(template, context, now=None):
    """Materialize only an explicit exact-job or standing per-job allowance.

    Context comes from GitHub's workflow environment, never workflow inputs.
    Does no I/O and never initializes or replenishes a ledger.
    """
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise BudgetError("UTC approval clock required")
    for name in ("run_id", "run_attempt"):
        if not str(context.get(name, "")).isdigit():
            raise BudgetError("Missing trusted GitHub job identity")
    name = context.get("matrix_name")
    if not isinstance(name, str) or not name or "\n" in name:
        raise BudgetError("Missing trusted matrix identity")
    scope = f"{context['run_id']}:{context['run_attempt']}:{name}"
    data = dict(template)
    approval = data.pop("approval", None)
    if approval is None:
        if data.get("scope_id") != scope:
            raise BudgetError("Exact comparison-job approval mismatch")
        return data
    if "scope_id" in data or not isinstance(approval, dict):
        raise BudgetError("Ambiguous paid approval mode")
    required = {
        "mode",
        "budget_unit",
        "workflow_ref",
        "allowed_events",
        "matrix_names",
        "not_before",
        "expires_at",
    }
    if (
        set(approval) != required
        or approval["mode"] != "standing"
        or approval["budget_unit"] != "per_comparison_job"
    ):
        raise BudgetError("Explicit standing per-comparison-job allowance required")
    for field, actual in (("allowed_events", context.get("event_name")), ("matrix_names", name)):
        values = approval[field]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) or not v for v in values)
            or actual not in values
        ):
            raise BudgetError("Job outside approved standing allowance")
    if not approval["workflow_ref"] or context.get("workflow_ref") != approval["workflow_ref"]:
        raise BudgetError("Workflow/ref outside approved standing allowance")

    def utc(value):
        if not isinstance(value, str) or not value.endswith("Z"):
            raise BudgetError("Explicit UTC allowance window required")
        return datetime.fromisoformat(value[:-1] + "+00:00")

    start, end = utc(approval["not_before"]), utc(approval["expires_at"])
    if not start <= now < end:
        raise BudgetError("Outside standing allowance start window")
    data["scope_id"] = scope
    return data


def load_policy(policy_path, contexts_path, contexts_sha256):
    raw = Path(contexts_path).read_bytes()
    if not contexts_sha256 or hashlib.sha256(raw).hexdigest() != contexts_sha256:
        raise BudgetError("Independently reviewed provider context evidence hash required")
    contexts = json.loads(raw)
    data = json.loads(Path(policy_path).read_text())
    if not isinstance(contexts, dict) or not isinstance(data, dict):
        raise BudgetError("Policy/context inputs must be JSON objects")
    if contexts.get("schema_version") != 1 or set(contexts.get("models", {})) != set(
        MODELS.values()
    ):
        raise BudgetError("Both provider context bounds must be independently verified")
    if set(data.get("models", {})) != {"openai/" + name for name in MODELS.values()}:
        raise BudgetError("Both user and required NL judge model policies are mandatory")
    prices = {}
    for model in MODELS.values():
        key = "openai/" + model
        bound = contexts["models"][model]
        # This is an external provider-contract input, independently reviewed and
        # hash-pinned by the operator, NOT a token-counter estimate or input_limit.
        if (
            type(bound.get("context_tokens")) is not int
            or bound["context_tokens"] <= 0
            or not str(bound.get("source_url", "")).startswith(
                ("https://platform.openai.com/", "https://developers.openai.com/")
            )
        ):
            raise BudgetError("Missing complete provider context bound/source")
        if (
            not isinstance(bound.get("response_models"), list)
            or not bound["response_models"]
            or any(not isinstance(name, str) or not name for name in bound["response_models"])
        ):
            raise BudgetError("Reviewed resolved response model IDs required")
        price = data["models"][key]
        if price.get("service_tier") != "default":
            raise BudgetError("Only explicitly priced standard service tier is qualified")
        prices[key] = Policy(
            key,
            price["input_per_million"],
            price["output_per_million"],
            data["cap_nusd"],
            data["calls"],
            data["calls_per_arm"],
            price["input_limit"],
            price["output_limit"],
        )
        if prices[key].input_limit < bound["context_tokens"]:
            raise BudgetError("Reservation cannot cover the complete provider context bound")
    return RunPolicy(
        prices,
        data["cap_nusd"],
        data["calls"],
        data["calls_per_arm"],
        data["scope_id"],
        contexts_sha256,
    ), contexts


def verify_sources():
    if sys.version_info[:2] != (3, 12):
        raise BudgetError("Only the qualified Python3.12 tau2 interpreter is supported")
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise BudgetError("Unsupported SDK version: " + name)
    for package, files in SDK_SHA.items():
        distribution = importlib.metadata.distribution(package)
        for name, digest in files.items():
            if (
                hashlib.sha256(
                    Path(distribution.locate_file(package + "/" + name)).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise BudgetError("Unqualified SDK request source: " + package + "/" + name)
    # uv's editable install keeps metadata in site-packages but imports the
    # package from the checkout. Check the actual import target in both modes.
    spec = importlib.util.find_spec("tau2")
    locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    if spec is None or spec.origin is None or len(locations) != 1:
        raise BudgetError("Unknown tau2 package source")
    root = Path(spec.origin).resolve().parent
    if Path(spec.origin).name != "__init__.py" or root != Path(locations[0]).resolve():
        raise BudgetError("Unknown tau2 package source")
    for name, digest in TAU2_SHA.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise BudgetError("Unqualified tau2 source: " + name)


def read_descriptor(descriptor):
    """Acceptance opens an EXISTING same-policy/scope ledger; never reset it."""
    policy, _ = load_policy(
        descriptor["policy_path"], descriptor["contexts_path"], descriptor["contexts_sha256"]
    )
    if policy.scope_id != descriptor["scope_id"]:
        raise BudgetError("Saved budget scope changed")
    return Ledger(descriptor["ledger_path"], policy, create=False)


class Guard:
    def __init__(
        self,
        ledger,
        contexts,
        *,
        arm,
        domain,
        invocation_id,
        local_model,
        local_url,
        transport_factory=None,
    ):
        if (
            arm not in {"vllm", "smg"}
            or domain not in {"retail", "airline", "telecom"}
            or not invocation_id
        ):
            raise BudgetError("Unknown run attribution")
        from urllib.parse import urlsplit

        url = urlsplit(local_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or not url.port
            or url.username
            or url.password
        ):
            raise BudgetError("Only an exact loopback agent endpoint is uncharged")
        self.ledger, self.contexts = ledger, contexts
        self.arm, self.domain, self.invocation_id = arm, domain, invocation_id
        if not local_model or local_model in MODELS.values():
            raise BudgetError("Local agent must not alias a paid model")
        self.local_model, self.local_url = local_model, local_url.rstrip("/")
        self.transport_factory = transport_factory
        self.clients, self.patches = [], []
        self.failed = False

    def deny(self, reason):
        self.failed = True
        self.ledger.halt()
        raise BudgetError(reason)

    def patch(self, obj, name, value):
        self.patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def close(self):
        for obj, name, original in reversed(self.patches):
            setattr(obj, name, original)
        for client in self.clients:
            client.close()

    def transport(self):
        import httpx

        guard = self
        underlying = (
            self.transport_factory() if self.transport_factory else httpx.HTTPTransport(retries=0)
        )

        class PaidTransport(httpx.BaseTransport):
            def handle_request(self, request):
                role, task = ROLE.get(), TASK.get()
                if (
                    role not in MODELS
                    or task is None
                    or request.method != "POST"
                    or str(request.url) != "https://api.openai.com/v1/chat/completions"
                ):
                    guard.deny("Unattributed/unsupported paid request path")
                body = request.read()
                if len(body) > 1_000_000:
                    guard.deny("Oversized serialized text request")
                try:
                    data = json.loads(body)
                except (ValueError, TypeError):
                    guard.deny("Invalid paid request body")
                if not isinstance(data, dict):
                    guard.deny("Paid request must be a JSON object")
                allowed = {
                    "model",
                    "messages",
                    "tools",
                    "tool_choice",
                    "temperature",
                    "seed",
                    "max_completion_tokens",
                    "reasoning_effort",
                    "response_format",
                    "n",
                    "stream",
                    "service_tier",
                }
                model = MODELS[role]
                price = guard.ledger.policy.model_policy("openai/" + model)
                if (
                    set(data) - allowed
                    or data.get("service_tier") != "default"
                    or data.get("model") != model
                    or data.get("stream", False) is not False
                    or type(data.get("n", 1)) is not int
                    or data.get("n", 1) != 1
                    or type(data.get("max_completion_tokens")) is not int
                    or not 0 < data["max_completion_tokens"] <= price.output_limit
                    or not isinstance(data.get("messages"), list)
                ):
                    guard.deny("Unbounded/transformed/unsupported paid request")
                if not data["messages"]:
                    guard.deny("Empty paid request messages")
                if not isinstance(data.get("tools", []), (list, type(None))):
                    guard.deny("Unknown tool schema container")
                for message in data["messages"]:
                    if (
                        not isinstance(message, dict)
                        or message.get("role") not in {"system", "user", "assistant", "tool"}
                        or not isinstance(message.get("content", ""), (str, type(None)))
                        or set(message) - {"role", "content", "tool_calls", "tool_call_id", "name"}
                    ):
                        guard.deny("Only complete plain text message payloads are qualified")
                if any(
                    not isinstance(tool, dict) or tool.get("type") != "function"
                    for tool in data.get("tools", []) or []
                ):
                    guard.deny("Billable server tools/other modalities are not qualified")
                request_id = uuid.uuid4().hex
                metadata = dict(
                    role=role,
                    domain=guard.domain,
                    invocation_id=guard.invocation_id,
                    task_id=str(task[0]),
                    trial=task[1],
                    provider="openai",
                )
                try:
                    guard.ledger.reserve(
                        request_id,
                        guard.arm,
                        "openai/" + model,
                        guard.contexts["models"][model]["context_tokens"],
                        data["max_completion_tokens"],
                        num_retries=0,
                        output_limit_verified=True,
                        metadata=metadata,
                    )
                except BudgetError:
                    guard.ledger.halt()  # budget refusal must not become a valid zero reward
                    raise
                outcome = {}
                try:
                    response = underlying.handle_request(request)
                    outcome["http_status"] = response.status_code
                    response.read()
                    if response.status_code != 200:
                        raise BudgetError("Unknown paid outcome: HTTP error")
                    decoded = response.json()
                    observed_model = decoded.get("model")
                    if observed_model not in guard.contexts["models"][model]["response_models"]:
                        outcome["response_model"] = "<unapproved>"
                        raise BudgetError("Unknown actual paid response model")
                    outcome["response_model"] = observed_model
                    observed_tier = decoded.get("service_tier")
                    outcome["service_tier"] = (
                        observed_tier
                        if observed_tier in (None, "auto", "default", "flex", "priority", "scale")
                        else "<unrecognized>"
                    )
                    if outcome["service_tier"] != "default":
                        raise BudgetError("Unknown actual paid service tier")
                    usage = decoded.get("usage", {})
                    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
                    total = usage.get("total_tokens")
                    if (
                        type(total) is not int
                        or type(prompt) is not int
                        or type(completion) is not int
                        or total != prompt + completion
                    ):
                        raise BudgetError("Unknown total paid usage")
                except Exception as error:
                    outcome["error_type"] = type(error).__name__
                    guard.ledger.finish(
                        request_id, outcome=outcome
                    )  # retain reservation + halt; no retry send
                    raise
                guard.ledger.finish(request_id, prompt, completion, outcome=outcome)
                return response

            def close(self):
                underlying.close()

        return PaidTransport()

    def install(self):
        verify_sources()
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        os.environ["LITELLM_TELEMETRY"] = "False"
        import httpx

        original_send = httpx.Client.send

        def send(client, request, **kwargs):
            if client in self.clients or (
                LOCAL.get() and str(request.url).startswith(self.local_url + "/")
            ):
                # HTTPX follows redirects internally without calling Client.send
                # again. Neither paid nor uncharged local traffic may escape.
                response = original_send(client, request, **(kwargs | {"follow_redirects": False}))
                if response.is_redirect:
                    response.close()
                    self.deny("Redirect outside the fixed request contract")
                return response
            self.deny("Unknown HTTP client/path cannot send in guarded tau2")

        async def async_send(*args, **kwargs):
            self.deny("Async paid/alternate SDK path is not qualified")

        self.patch(httpx.Client, "send", send)
        self.patch(httpx.AsyncClient, "send", async_send)
        import litellm
        import openai
        from tau2.evaluator import evaluator_nl_assertions
        from tau2.runner import batch
        from tau2.user import user_simulator
        from tau2.utils import llm_utils

        litellm.register_model(
            {
                self.local_model: {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "litellm_provider": "openai",
                    "mode": "chat",
                }
            }
        )
        original_completion = llm_utils.completion
        self.patch(litellm, "num_retries", 0)

        def protected_completion(*args, **kwargs):
            role = ROLE.get()
            model = str(kwargs.get("model", "")).removeprefix("openai/")
            base = str(kwargs.get("api_base", "")).rstrip("/")
            if args or role not in MODELS or model != MODELS[role] or base:
                self.deny("Unknown paid role/model/endpoint")
            forbidden = {
                "client",
                "fallbacks",
                "context_window_fallback_dict",
                "retry_policy",
                "api_base",
                "base_url",
                "extra_body",
                "acompletion",
                "stream",
                "n",
                "max_tokens",
                "max_completion_tokens",
                "custom_llm_provider",
                "mock_response",
                "service_tier",
            }
            if set(kwargs) & forbidden:
                self.deny("Caller cannot override the protected paid request")
            key = os.environ.get("OPENAI_API_KEY")
            if not key or self.ledger.summary()["halted"]:
                self.deny("Missing credential or halted shared budget")
            http_client = httpx.Client(transport=self.transport(), follow_redirects=False)
            self.clients.append(http_client)
            client = None
            try:
                client = openai.OpenAI(
                    api_key=key,
                    max_retries=0,
                    base_url="https://api.openai.com/v1",
                    http_client=http_client,
                )
                price = self.ledger.policy.model_policy("openai/" + model)
                return original_completion(
                    **kwargs
                    | {
                        "num_retries": 0,
                        "max_retries": 0,
                        "max_completion_tokens": price.output_limit,
                        "service_tier": "default",
                        "client": client,
                    }
                )
            finally:
                # The supported call is synchronous/non-streaming and its body
                # has been consumed. Bound pools by inflight calls, not turns.
                try:
                    if client is not None:
                        client.close()
                    else:
                        http_client.close()
                except Exception:
                    self.ledger.halt()
                    raise
                finally:
                    self.clients.remove(http_client)

        def completion(*args, **kwargs):
            model = str(kwargs.get("model", "")).removeprefix("openai/")
            base = str(kwargs.get("api_base", "")).rstrip("/")
            if model == self.local_model and base == self.local_url + "/v1":
                token = LOCAL.set(True)
                try:
                    return original_completion(*args, **kwargs)
                finally:
                    LOCAL.reset(token)
            try:
                return protected_completion(*args, **kwargs)
            except Exception:
                # tau2 can convert an exception into a zero-reward simulation.
                # Keep an independent process-level failure even if DB reporting
                # itself is unavailable, so that CLI0 cannot become budget success.
                self.failed = True
                self.ledger.halt()
                raise

        self.patch(llm_utils, "completion", completion)

        def role_wrapper(fn, role):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                token = ROLE.set(role)
                try:
                    return fn(*args, **kwargs)
                finally:
                    ROLE.reset(token)

            return wrapped

        self.patch(user_simulator, "generate", role_wrapper(user_simulator.generate, "user"))
        self.patch(
            evaluator_nl_assertions,
            "generate",
            role_wrapper(evaluator_nl_assertions.generate, "nl_judge"),
        )
        original_unit = batch.run_unit
        signature = inspect.signature(original_unit)

        @functools.wraps(original_unit)
        def unit(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            token = TASK.set((bound["task"].id, bound["trial"]))
            try:
                return original_unit(*args, **kwargs)
            finally:
                TASK.reset(token)

        self.patch(batch, "run_unit", unit)


def validate_command(command, domain):
    # Exact current run_ab command surface: no arbitrary tau2 subcommands/features.
    if not command or command[0] != "run":
        raise BudgetError("Only the existing plain text run command is qualified")
    allowed = {
        "--domain",
        "--agent-llm",
        "--agent-llm-args",
        "--user-llm",
        "--user-llm-args",
        "--num-trials",
        "--save-to",
        "--agent",
        "--task-split-name",
        "--seed",
        "--task-ids",
        "--num-tasks",
        "--max-concurrency",
    }
    if any(arg.startswith("--") and arg not in allowed for arg in command[1:]):
        raise BudgetError("Unqualified tau2 command surface")
    flags = [arg for arg in command[1:] if arg.startswith("--")]
    if len(flags) != len(set(flags)):
        raise BudgetError("Duplicate options cannot override the reviewed text scope")

    def value(flag):
        return command[command.index(flag) + 1] if flag in command else None

    if (
        value("--domain") != domain
        or value("--agent") != "llm_agent"
        or value("--user-llm") != "gpt-5.2"
        or value("--task-split-name") != "base"
    ):
        raise BudgetError("Unexpected current text workflow configuration")
    return command + ["--max-retries", "0", "--hallucination-retries", "0"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("policy", "contexts", "contexts-sha256", "ledger", "scope"):
        parser.add_argument("--" + flag, required=True)
    setup = parser.add_mutually_exclusive_group()
    setup.add_argument("--check", action="store_true")
    setup.add_argument("--initialize", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--domain")
    parser.add_argument("--invocation")
    parser.add_argument("--local-model")
    parser.add_argument("--local-url")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    policy, contexts = load_policy(args.policy, args.contexts, args.contexts_sha256)
    if args.scope != policy.scope_id:
        raise BudgetError("Explicit approved run/job scope mismatch")
    verify_sources()
    if args.check:
        return 0  # no ledger created/SDK imported/provider call
    if args.initialize:
        fd = os.open(args.ledger, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        Ledger(args.ledger, policy)
        return 0  # explicit one-time init; never reset a prior scope's ledger
    command = validate_command(
        args.command[1:] if args.command[:1] == ["--"] else args.command, args.domain
    )
    ledger = Ledger(args.ledger, policy, create=False)
    if ledger.summary()["halted"]:
        raise BudgetError("Shared paid budget is halted before tau2 main")
    guard = Guard(
        ledger,
        contexts,
        arm=args.arm,
        domain=args.domain,
        invocation_id=args.invocation,
        local_model=args.local_model,
        local_url=args.local_url,
    )
    # Keep the boundary for the child process lifetime: do not restore unguarded
    # aliases while task/SDK threads might still unwind after CLI return/error.
    guard.install()
    from tau2 import cli

    sys.argv = ["tau2", *command]
    cli.main()
    own = ledger.audit(args.invocation)
    return 2 if guard.failed or own["halted"] or own["pending"] or own["unknown"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BudgetError, OSError, ValueError, KeyError, ImportError) as error:
        print(f"Budgeted tau2 refused: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
