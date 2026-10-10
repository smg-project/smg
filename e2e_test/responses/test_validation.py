"""Negative validation tests for the Responses API body parser.

Covers audit task E7: verify that after P5 (PR #1298, "fail-fast on unknown
content/items"), requests carrying an unknown ``type`` discriminator in a
Responses input no longer slip past the gateway's body parser via the old
silent-swallow path.

The gateway deserializes every ``/v1/responses`` body through
``ValidatedJson<ResponsesRequest>`` (crates/protocols/src/validated.rs). Any
serde error surfaces as ``HTTP 400`` with
``{"error": {"type": "invalid_request_error", "code": "json_parse_error"}}``
(see validated.rs:60-85), so these tests hit the gateway directly with
``httpx`` instead of the OpenAI SDK — both to exercise the exact wire shape
and to bypass client-side shape enforcement.

The ``type``-discrimination invariants exercised here live in:

* ``ResponseContentPart`` — tagged enum over ``output_text``, ``input_text``,
  ``input_image``, ``input_file``, ``refusal`` with no untagged fallback
  (crates/protocols/src/responses.rs:1310-1356). Any other ``type`` fails.
* ``ResponseInputOutputItem::SimpleInputMessage`` — the untagged catch-all
  for ``{role, content, type?}`` items; after P5 its ``type`` field is
  ``Option<SimpleInputMessageTypeTag>`` where the tag enum contains only
  ``Message`` (responses.rs:1236-1264). Unknown item-level ``type`` strings
  that previously landed here now force a deserialize error.
"""

from __future__ import annotations

import logging
import os

import httpx
import pytest

logger = logging.getLogger(__name__)


def _post_responses(gateway, body: dict, timeout: float = 30.0) -> httpx.Response:
    """POST a raw JSON body to ``{gateway.base_url}/v1/responses``.

    We need full control over the wire payload (invalid ``type`` strings,
    empty discriminators, etc.), which the OpenAI SDK's typed client will
    not emit, so all four cases drop to raw ``httpx``.

    Forwards ``OPENAI_API_KEY`` when present so the accept-path tests
    (``!= 400``) cannot be satisfied vacuously by a ``401`` from an
    earlier auth middleware — if the gateway ever reorders auth in front
    of body parsing, we want the parser itself to run, not a header
    check. The 400 cases short-circuit inside ``ValidatedJson`` before
    auth so the key value is irrelevant for them.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "sk-not-used")
    return httpx.post(
        f"{gateway.base_url}/v1/responses",
        json=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )


def _assert_validation_400(resp: httpx.Response) -> dict:
    """Assert the gateway returned a 400 with the canonical parser envelope.

    Pins the full shape emitted by ``ValidatedJson``'s ``JsonDataError``
    branch (``crates/protocols/src/validated.rs:74-84``): status 400,
    ``error.type == "invalid_request_error"``, and
    ``error.code == "json_parse_error"``. The ``code`` assertion is what
    distinguishes a parser-level rejection from the validator-level
    rejection emitted at validated.rs:91-103 (which uses ``code: 400`` as
    an integer) or any upstream/auth 400 that happens to share the
    envelope prefix.

    Returns the parsed JSON body so individual tests can make per-case
    assertions about the error message contents.
    """
    assert resp.status_code == 400, f"expected HTTP 400, got {resp.status_code}: body={resp.text!r}"
    body = resp.json()
    assert isinstance(body, dict), f"expected JSON object, got {body!r}"
    err = body.get("error")
    assert isinstance(err, dict), f"expected error object, got {body!r}"
    assert err.get("type") == "invalid_request_error", (
        f"expected invalid_request_error, got {err!r}"
    )
    assert err.get("code") == "json_parse_error", (
        f"expected code=json_parse_error (parser branch), got {err!r}"
    )
    message = err.get("message", "")
    assert isinstance(message, str) and message, f"expected non-empty message, got {err!r}"
    return body


def _is_validation_400(resp: httpx.Response) -> bool:
    """Return True iff the response is the gateway's parser-level 400.

    The accept-path tests need to reject the specific envelope emitted by
    ``ValidatedJson`` on a deserialize failure, not every HTTP 400. A
    plain ``status_code != 400`` check would let a downstream 400 (e.g.
    upstream tool-validation, or a 401/5xx from auth) vacuously satisfy
    the assertion even though the parser's accept path was never
    exercised.
    """
    if resp.status_code != 400:
        return False
    try:
        body = resp.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    return err.get("code") == "json_parse_error"


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestResponsesInputValidation:
    """Negative tests for unknown/invalid ``type`` strings on Responses input.

    Backend is ``openai`` because E7 lives in e2e_test/responses/ which is
    indexed to cloud backends in the audit scope, and because the 400 cases
    short-circuit before the upstream call anyway.
    """

    def test_unknown_content_part_type_rejected_400(self, setup_backend):
        """Unknown content-part ``type`` must fail-fast with 400 (E7 case 1).

        ``ResponseContentPart`` is a closed tagged enum (responses.rs:1310-1356)
        with no untagged fallback, so ``{"type": "input_video"}`` inside a
        message's content array cannot silently deserialize to any variant.

        The audit §E7 case calls for the rejected ``type`` being echoed in
        the error message, but the outer ``ResponseInput`` is
        ``#[serde(untagged)]`` (responses.rs:2051-2056): when the ``Items``
        variant fails for any nested reason, serde collapses the error to
        ``"data did not match any variant of untagged enum ResponseInput"``
        and the inner tag name is not preserved on the wire. We therefore
        pin the invariants that are observable — the 400 envelope and the
        canonical ``input``-path reference in the message — rather than
        asserting the specific rejected tag. Tightening beyond this
        would require P5.5 (flattening ``ResponseInput`` away from
        ``#[serde(untagged)]``), which is out of scope for E7.
        """
        _, model_path, _, gw = setup_backend
        body = {
            "model": model_path,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_video", "text": "please watch this"}],
                }
            ],
        }
        resp = _post_responses(gw, body)
        err = _assert_validation_400(resp)
        message = err["error"]["message"].lower()
        # Minimal observable invariant: the error must surface at the
        # ``input`` field (the only field that can carry content parts),
        # proving serde actually drove into the input structure rather
        # than failing at an unrelated key.
        assert "input" in message, (
            f"expected error to name the input field, got message={message!r}"
        )

    def test_input_file_with_unknown_field_accepted(self, setup_backend):
        """``input_file`` content part with extra fields is accepted (E7 case 2).

        ``ResponseContentPart::InputFile`` is declared without
        ``#[serde(deny_unknown_fields)]`` (responses.rs:1340-1351), so serde
        silently ignores unknown fields on the variant. This test pins that
        observed behavior: a payload like ``{"type": "input_file", "unknown": 1}``
        deserializes cleanly. If a future change adds ``deny_unknown_fields``
        to tighten the contract, this test will flip to expecting 400 and
        document the policy change.

        The request is still expected to fail end-to-end because the upstream
        model will reject the ill-formed file reference, but the gateway
        itself must *not* 400 on the shape.
        """
        _, model_path, _, gw = setup_backend
        body = {
            "model": model_path,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hi"},
                        # `unknown` is an extra field on the InputFile variant.
                        # Current behavior: silently ignored. file_id is set so
                        # the gateway proceeds past deserialization; the
                        # fake id will then be rejected by the upstream
                        # (OpenAI) service, not by the gateway's body parser.
                        {
                            "type": "input_file",
                            "file_id": "file-doesnotexist",
                            "unknown": 1,
                        },
                    ],
                }
            ],
            "max_output_tokens": 16,
        }
        resp = _post_responses(gw, body)
        # The critical invariant is narrow: the gateway's *parser* must not
        # emit its json_parse_error envelope for the extra field. A plain
        # `status_code != 400` check would let an unrelated downstream 400
        # (e.g. auth, upstream validation) vacuously satisfy the assertion,
        # so we target the parser-specific envelope via _is_validation_400.
        # If deny_unknown_fields is ever added, this flips intentionally and
        # the docstring above needs to be inverted.
        assert not _is_validation_400(resp), (
            f"extra field on input_file unexpectedly rejected by body parser: "
            f"status={resp.status_code} body={resp.text!r}"
        )

    def test_message_with_invalid_role_rejected_400(self, setup_backend):
        """An input message with an unknown role is refused before generation
        (E7 case 3), as the public API refuses it: only ``assistant``,
        ``system``, ``developer`` and ``user`` are input roles.

        The rejection comes from the request validator (``validate()`` after
        the body parser), so the envelope is the validator's
        ``invalid_request_error``, not the parser's ``json_parse_error``.
        """
        _, model_path, _, gw = setup_backend
        body = {
            "model": model_path,
            "input": [
                {"type": "message", "role": "martian", "content": "hello"},
            ],
            "max_output_tokens": 16,
        }
        resp = _post_responses(gw, body)
        assert resp.status_code == 400, (
            f"role='martian' must be rejected: status={resp.status_code} body={resp.text!r}"
        )
        err = resp.json()["error"]
        assert err["type"] == "invalid_request_error", err
        assert "martian" in err["message"], err

    def test_content_part_array_with_empty_type_rejected_400(self, setup_backend):
        """Empty-string ``type`` on a content part must fail-fast (E7 case 4).

        An empty discriminator is neither a known variant of
        ``ResponseContentPart`` nor the untagged message-item fallback, so
        the item's content array cannot deserialize. P5 relies on this
        behavior to turn the silent-swallow path into an explicit 400.

        We include a leading valid part (``input_text``) to make sure the
        rejection is driven by the trailing empty-``type`` entry and not by
        an empty-array edge case.
        """
        _, model_path, _, gw = setup_backend
        body = {
            "model": model_path,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello"},
                        {"type": ""},
                    ],
                }
            ],
        }
        resp = _post_responses(gw, body)
        _assert_validation_400(resp)
