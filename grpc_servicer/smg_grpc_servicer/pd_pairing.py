"""The operator-set PD pairing protocol, read from the engine's environment.

A deployment can inject ``SMG_PAIRING_PROTOCOL`` into the engine container;
each servicer reports the value in its server info, the gateway turns it into
the worker's ``pairing_protocol`` label, and a prefill and a decode carrying
one pair only when the values are equal (nothing derived about transport,
version or KV layout is compared). Unset or blank means "derive it".
"""

import os
from collections.abc import Mapping

PAIRING_PROTOCOL_ENV = "SMG_PAIRING_PROTOCOL"


def pairing_protocol_from_env(environ: Mapping[str, str] | None = None) -> str:
    """The trimmed ``SMG_PAIRING_PROTOCOL`` value, or ``""`` when unset or blank."""
    source = os.environ if environ is None else environ
    return (source.get(PAIRING_PROTOCOL_ENV) or "").strip()
