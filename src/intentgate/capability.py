"""Pure-Python capability-token helpers.

The IntentGate gateway issues capability tokens with a Macaroons-style
chained-HMAC signature: a token holder can derive a strictly more
restrictive child token by appending a caveat and HMAC'ing it under the
parent's signature *without ever touching the master key*. That is the
defining property — and the reason this module exists in the SDK.

Use case. A parent agent receives a token allowing tools ``[a, b, c]``,
spawns a sub-agent for one task, and wants the sub-agent's token to
allow only ``[a]``. The parent calls :func:`attenuate` and hands the
resulting token string to the sub-agent. The gateway, which has the
master key, accepts the attenuated token and rejects any sub-agent
call that would have needed ``b`` or ``c``.

What we don't do here.

* No master key access. By design — that's what makes attenuation
  safe. If you need to *mint* a brand-new root token, that's the
  ``POST /v1/admin/mint`` endpoint, not this module.
* No semantic narrowing check. Adding a "broader" caveat doesn't
  widen the chain because the parent's narrower caveat fires first
  on the gateway side. We intentionally don't try to second-guess the
  caller — policy belongs in the gateway, not the SDK.

Wire format. We mirror the Go gateway's serialization byte-for-byte
on the one place it matters (the new caveat's canonical JSON, which
seeds the HMAC step). Anywhere else, JSON ordering doesn't affect
correctness because the gateway re-marshals from its parsed Go struct
during ``Verify``.
"""

from __future__ import annotations

import base64
import hmac
import json
import time as _time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

__all__ = ["attenuate", "AttenuationError", "Caveat", "decode_token"]


class AttenuationError(ValueError):
    """Raised when the parent token is malformed or attenuation fails."""


# Caveat field order MUST match the Go ``Caveat`` struct in
# gateway/internal/capability/token.go. Go's encoding/json emits
# fields in declaration order; we replicate that order here so the
# bytes we HMAC match what the gateway re-marshals during Verify.
_CAVEAT_FIELD_ORDER = ("t", "tools", "agent", "exp", "max_calls")


@dataclass
class Caveat:
    """A structured restriction recorded in a token's chain.

    Mirrors gateway/internal/capability/token.go ``Caveat``. Only fields
    relevant to the caveat's :attr:`type` need to be set; the others are
    omitted from the JSON output (matching Go's ``omitempty`` tags).
    """

    type: str
    tools: list[str] | None = None
    agent: str = ""
    expiry: int = 0
    max_calls: int = 0

    # Caveat-type identifiers, kept in sync with the Go consts.
    EXPIRY = "exp"
    TOOL_ALLOW = "tool_allow"
    TOOL_DENY = "tool_deny"
    AGENT_LOCK = "agent_lock"
    MAX_CALLS = "max_calls"

    def to_dict(self) -> dict[str, Any]:
        """Build the dict whose JSON encoding matches Go's bytes.

        Insertion order is fixed by ``_CAVEAT_FIELD_ORDER``. Empty
        / zero values are omitted to match Go's ``json:",omitempty"``.
        """
        out: dict[str, Any] = {}
        # Type ("t") is always present (no omitempty in the Go tag).
        out["t"] = self.type
        if self.tools:  # None or [] both skip
            out["tools"] = list(self.tools)
        if self.agent:
            out["agent"] = self.agent
        if self.expiry:
            out["exp"] = int(self.expiry)
        if self.max_calls:
            out["max_calls"] = int(self.max_calls)
        # Reorder to the canonical sequence (defensive — dict above is
        # already insertion-ordered, but explicit beats implicit).
        return {k: out[k] for k in _CAVEAT_FIELD_ORDER if k in out}


def _b64url_decode(s: str) -> bytes:
    """Base64url-decode an unpadded RawURLEncoding string."""
    pad = "=" * (-len(s) % 4)
    try:
        return base64.urlsafe_b64decode(s + pad)
    except Exception as exc:  # pragma: no cover - defensive
        raise AttenuationError(f"invalid base64url: {exc}") from exc


def _b64url_encode(b: bytes) -> str:
    """Base64url-encode bytes without padding (RawURLEncoding)."""
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _canonical_caveat_bytes(c: Caveat) -> bytes:
    """JSON-encode ``c`` with the exact byte sequence Go produces.

    Critical: this is the input to the HMAC step that derives the
    child's signature. Mismatch here = gateway rejects the child.

    Constraints:
      * Field order: see _CAVEAT_FIELD_ORDER.
      * No whitespace between separators (``separators=(',', ':')``).
      * ASCII-escaped (``ensure_ascii=True`` is the default).
      * Empty/zero fields omitted (matches Go ``omitempty``).
    """
    return json.dumps(c.to_dict(), separators=(",", ":")).encode("ascii")


def decode_token(token: str) -> dict[str, Any]:
    """Decode a base64url(JSON) token into a dict.

    Pure-Python; no signature check (the gateway does that). Useful for
    inspecting the chain — e.g. ``decode_token(t)["cav"]`` to see what
    caveats are bound to the token.
    """
    try:
        raw = _b64url_decode(token)
    except AttenuationError:
        raise
    except Exception as exc:
        raise AttenuationError(f"token is not valid base64url: {exc}") from exc
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise AttenuationError(f"token JSON is malformed: {exc}") from exc


def attenuate(
    token: str,
    *,
    add_tools: list[str] | None = None,
    deny_tools: list[str] | None = None,
    max_calls: int | None = None,
    expires_in_seconds: int | None = None,
    expires_at: int | None = None,
    extra: list[Caveat] | None = None,
) -> str:
    """Append narrowing caveats to ``token`` and return a new token string.

    Each keyword groups one common attenuation pattern. Multiple kwargs
    can be combined; each generates one caveat appended in this order:

    1. ``add_tools`` → :class:`Caveat` of type ``tool_allow`` whose
       ``tools`` is the supplied list (intersect: a child whitelist
       narrower than the parent's).
    2. ``deny_tools`` → :class:`Caveat` of type ``tool_deny`` (additive:
       a child blacklist on top of any parent blacklist).
    3. ``max_calls`` → :class:`Caveat` of type ``max_calls``. The
       gateway's budget check enforces the *minimum* across the chain,
       so adding a child cap only narrows.
    4. ``expires_in_seconds`` / ``expires_at`` → :class:`Caveat` of
       type ``exp``. ``expires_at`` is a unix timestamp; the
       ``expires_in_seconds`` form is computed at call time.
    5. ``extra`` → user-supplied caveats appended last (advanced).

    Example::

        from intentgate.capability import attenuate

        # Parent token: agent allowed to call [search, read, email]
        # for the next hour. Child: only [search, read], one call max.
        child = attenuate(
            parent_token,
            add_tools=["search", "read"],
            max_calls=1,
        )

        # Hand `child` to the sub-agent.
    """
    parsed = decode_token(token)
    if not isinstance(parsed, dict) or "sig" not in parsed:
        raise AttenuationError("token is missing 'sig' field")
    if "cav" not in parsed:
        raise AttenuationError("token is missing 'cav' field")
    if not parsed.get("root_jti"):
        raise AttenuationError("token has no root_jti (was it minted by gateway < v0.7?)")

    parent_sig = _b64url_decode(parsed["sig"])
    cavs: list[dict[str, Any]] = list(parsed["cav"])
    sig = parent_sig

    new_caveats: list[Caveat] = []
    if add_tools:
        new_caveats.append(Caveat(type=Caveat.TOOL_ALLOW, tools=list(add_tools)))
    if deny_tools:
        new_caveats.append(Caveat(type=Caveat.TOOL_DENY, tools=list(deny_tools)))
    if max_calls is not None:
        if max_calls < 0:
            raise AttenuationError("max_calls must be >= 0")
        new_caveats.append(Caveat(type=Caveat.MAX_CALLS, max_calls=int(max_calls)))
    if expires_at is not None or expires_in_seconds is not None:
        if expires_at is None:
            expires_at = int(_time.time()) + int(expires_in_seconds or 0)
        new_caveats.append(Caveat(type=Caveat.EXPIRY, expiry=int(expires_at)))
    if extra:
        new_caveats.extend(extra)

    if not new_caveats:
        raise AttenuationError(
            "attenuate() requires at least one narrowing argument "
            "(add_tools, deny_tools, max_calls, expires_in_seconds, "
            "expires_at, or extra)"
        )

    # Walk the new caveats forward, hopping the chain one step per caveat.
    for c in new_caveats:
        cb = _canonical_caveat_bytes(c)
        sig = hmac.new(sig, cb, sha256).digest()
        cavs.append(c.to_dict())

    child = dict(parsed)
    child["cav"] = cavs
    child["sig"] = _b64url_encode(sig)

    # Re-encode. Go decodes JSON order-independently during Verify, so
    # any field ordering here is fine — separators only matter for the
    # caveat-canonical-bytes used inside the HMAC, not the outer token.
    return _b64url_encode(json.dumps(child).encode("utf-8"))
