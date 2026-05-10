"""Tests for the pure-Python attenuation helpers.

Wire-format compatibility is the central concern: bytes the Python SDK
HMACs MUST match what the Go gateway re-marshals during Verify. Two
strategies cover that:

1. Direct byte comparison against canonical samples (the Caveat
   serialization tests).
2. End-to-end round trip simulation: build a parent token whose
   signature is a known HMAC chain, attenuate it in Python, then
   independently recompute the chain in Python (mimicking what the
   gateway does in Go) and confirm the new signature matches.

A future cross-language fixture suite (Go test outputs imported here)
would lock the format more rigorously; for v0.2 the algebraic check is
sufficient because the chain construction is fully deterministic given
the inputs.
"""

from __future__ import annotations

import base64
import hmac
import json
from hashlib import sha256

import pytest

from intentgate.capability import (
    AttenuationError,
    Caveat,
    _canonical_caveat_bytes,
    attenuate,
    decode_token,
)

# --- Caveat canonical bytes -------------------------------------------------


def test_caveat_bytes_tool_allow():
    c = Caveat(type=Caveat.TOOL_ALLOW, tools=["search", "read"])
    # Field order: t, then tools. No whitespace.
    assert _canonical_caveat_bytes(c) == b'{"t":"tool_allow","tools":["search","read"]}'


def test_caveat_bytes_max_calls():
    c = Caveat(type=Caveat.MAX_CALLS, max_calls=10)
    assert _canonical_caveat_bytes(c) == b'{"t":"max_calls","max_calls":10}'


def test_caveat_bytes_expiry():
    c = Caveat(type=Caveat.EXPIRY, expiry=1700000000)
    assert _canonical_caveat_bytes(c) == b'{"t":"exp","exp":1700000000}'


def test_caveat_bytes_omits_empty_fields():
    """Empty tools list / zero ints / empty strings must not appear."""
    c = Caveat(type=Caveat.AGENT_LOCK, agent="agent-x")
    # 'tools', 'exp', 'max_calls' all omitted because zero-valued.
    assert _canonical_caveat_bytes(c) == b'{"t":"agent_lock","agent":"agent-x"}'


def test_caveat_field_order_is_canonical():
    """Fields appear in the Go struct's declaration order regardless of
    keyword-arg order at construction time."""
    c = Caveat(
        type=Caveat.TOOL_ALLOW,
        tools=["a"],
        agent="ignored",  # not normally combined with tool_allow, but
        # if both were set, agent must come AFTER tools.
    )
    out = _canonical_caveat_bytes(c)
    assert out.index(b'"tools"') < out.index(b'"agent"')


# --- Token round-trip -------------------------------------------------------


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _make_fake_parent(jti: str = "root-jti-1", tenant: str = "default") -> str:
    """Produce a token whose signature is a known HMAC chain seeded
    from a fake master key. Used to drive attenuation tests without
    standing up a Go gateway in pytest.

    Mirrors gateway capability v3 (gateway 0.9+): adds the `tenant`
    field, signed in the canonicalPayload.
    """
    master_key = b"x" * 32
    payload = {
        "v": 3,
        "jti": jti,
        "root_jti": jti,
        "iss": "intentgate",
        "tenant": tenant,
        "sub": "agent-x",
        "iat": 1700000000,
    }
    # Reproduce Go's canonicalPayload byte order: v, jti, root_jti, iss,
    # tenant, sub, iat, nbf. We omit nbf because it's zero.
    payload_bytes = json.dumps(
        {k: payload[k] for k in ("v", "jti", "root_jti", "iss", "tenant", "sub", "iat")},
        separators=(",", ":"),
    ).encode("ascii")
    sig = hmac.new(master_key, payload_bytes, sha256).digest()

    # One agent_lock caveat to mirror what Mint always prepends.
    cav = Caveat(type=Caveat.AGENT_LOCK, agent="agent-x")
    cb = _canonical_caveat_bytes(cav)
    sig = hmac.new(sig, cb, sha256).digest()

    token = {
        "v": 3,
        "jti": jti,
        "root_jti": jti,
        "iss": "intentgate",
        "tenant": tenant,
        "sub": "agent-x",
        "iat": 1700000000,
        "cav": [cav.to_dict()],
        "sig": _b64url_encode(sig),
    }
    return _b64url_encode(json.dumps(token).encode("utf-8"))


def test_attenuate_appends_caveat_and_chains_signature():
    parent_str = _make_fake_parent()
    parent = decode_token(parent_str)
    parent_sig = _b64url_decode(parent["sig"])

    child_str = attenuate(parent_str, add_tools=["read"])
    child = decode_token(child_str)

    # One more caveat than parent.
    assert len(child["cav"]) == len(parent["cav"]) + 1
    assert child["cav"][-1] == {"t": "tool_allow", "tools": ["read"]}

    # Child signature must equal HMAC(parent.sig, canonical(new_caveat)).
    new_cav = Caveat(type=Caveat.TOOL_ALLOW, tools=["read"])
    expected = hmac.new(parent_sig, _canonical_caveat_bytes(new_cav), sha256).digest()
    assert _b64url_decode(child["sig"]) == expected


def test_attenuate_preserves_root_jti_and_subject():
    parent_str = _make_fake_parent("root-A")
    child_str = attenuate(parent_str, add_tools=["read"])
    child = decode_token(child_str)
    assert child["root_jti"] == "root-A"
    assert child["sub"] == "agent-x"


def test_attenuate_preserves_tenant():
    """A child token MUST stay in the parent's tenant. Cryptographically
    enforced by the gateway (tenant is signed in canonicalPayload), so
    the SDK just propagates the field unchanged."""
    parent_str = _make_fake_parent("root-A", tenant="acme")
    child_str = attenuate(parent_str, add_tools=["read"])
    child = decode_token(child_str)
    assert child["tenant"] == "acme"


def test_attenuate_rejects_token_without_tenant():
    """Tokens minted by gateway < v0.9 carry no tenant and should be
    rejected by the SDK so the operator sees the deprecation clearly."""
    legacy_v2 = {
        "v": 2,
        "jti": "old-jti",
        "root_jti": "old-jti",
        "iss": "intentgate",
        "sub": "agent-x",
        "iat": 1700000000,
        "cav": [],
        "sig": _b64url_encode(b"x" * 32),
    }
    encoded = _b64url_encode(json.dumps(legacy_v2).encode("utf-8"))
    with pytest.raises(AttenuationError, match="tenant"):
        attenuate(encoded, add_tools=["read"])


def test_attenuate_chained_max_calls_then_expiry():
    """Two attenuation hops in one call — verify the chain walks both."""
    parent_str = _make_fake_parent()
    parent = decode_token(parent_str)
    parent_sig = _b64url_decode(parent["sig"])

    child_str = attenuate(parent_str, max_calls=5, expires_at=1800000000)
    child = decode_token(child_str)

    assert len(child["cav"]) == len(parent["cav"]) + 2
    # Order in the kwargs maps to order in the chain: max_calls, then exp.
    assert child["cav"][-2]["t"] == "max_calls"
    assert child["cav"][-1]["t"] == "exp"

    # Recompute: hop 1 (max_calls), hop 2 (exp).
    sig = parent_sig
    for c in (
        Caveat(type=Caveat.MAX_CALLS, max_calls=5),
        Caveat(type=Caveat.EXPIRY, expiry=1800000000),
    ):
        sig = hmac.new(sig, _canonical_caveat_bytes(c), sha256).digest()
    assert _b64url_decode(child["sig"]) == sig


def test_attenuate_requires_at_least_one_narrowing_arg():
    parent = _make_fake_parent()
    with pytest.raises(AttenuationError, match="requires at least one"):
        attenuate(parent)


def test_attenuate_rejects_negative_max_calls():
    parent = _make_fake_parent()
    with pytest.raises(AttenuationError, match="max_calls"):
        attenuate(parent, max_calls=-1)


def test_attenuate_rejects_token_without_root_jti():
    """Tokens minted by gateway < v0.7 carry no root_jti and should be
    rejected by the SDK so the operator sees the deprecation clearly."""
    legacy = {
        "v": 1,
        "jti": "old-jti",
        "iss": "intentgate",
        "sub": "agent-x",
        "iat": 1700000000,
        "cav": [],
        "sig": _b64url_encode(b"x" * 32),
    }
    encoded = _b64url_encode(json.dumps(legacy).encode("utf-8"))
    with pytest.raises(AttenuationError, match="root_jti"):
        attenuate(encoded, add_tools=["read"])


def test_attenuate_rejects_malformed_token():
    with pytest.raises(AttenuationError):
        attenuate("not-base64-or-json", add_tools=["read"])


def test_decode_token_roundtrip():
    encoded = _make_fake_parent("root-X")
    parsed = decode_token(encoded)
    assert parsed["jti"] == "root-X"
    assert parsed["root_jti"] == "root-X"
    assert parsed["sub"] == "agent-x"
