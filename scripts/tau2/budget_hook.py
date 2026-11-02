"""Opt-in synchronous LiteLLM interception; no production authorizer is supplied."""

from uuid import uuid4

from budget import BudgetError


def install(litellm, *, original=None):
    send = original if original is not None else litellm.completion

    def guarded(*args, **payload):
        if args or guarded.authorizer is None:
            raise BudgetError("A verified request authorizer is required")
        if litellm.drop_params is not False or payload.get("drop_params", False) is not False:
            raise BudgetError("Silently dropped provider limits are forbidden")
        if payload.get("stream") not in (None, False):
            raise BudgetError("Streaming usage is not supported by this guard")
        ledger, arm, input_bound, output_parameter, verified = guarded.authorizer(payload)
        if output_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise BudgetError("Unknown provider output limit")
        if any(p in payload for p in {"max_tokens", "max_completion_tokens"} - {output_parameter}):
            raise BudgetError("Ambiguous provider output limits")
        request_id = uuid4().hex
        ledger.reserve(
            request_id,
            arm,
            payload.get("model"),
            input_bound,
            payload.get(output_parameter),
            num_retries=payload.get("num_retries"),
            output_limit_verified=verified,
        )
        try:
            response = send(**payload)
            usage = response.get("usage") or {}
            input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        except BaseException:
            ledger.finish(request_id)
            raise
        ledger.finish(request_id, input_tokens, output_tokens)
        return response

    guarded.authorizer = None
    litellm.completion = guarded
    return guarded
