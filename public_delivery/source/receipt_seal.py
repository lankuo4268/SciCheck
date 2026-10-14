"""Sealed independent verdict receipts.

A receipt is the JSON record of an independent check (reference, tolerance,
candidate, diff, verdict). It is SEALED with an HMAC using a trusted-side-only
key so that:
  - the model side can verify integrity but NEVER receives the key;
  - the public / model-facing view cannot forge or alter a receipt.

The sealing key is generated per-store on the trusted side and is NOT returned
by any model-facing method. verify_run re-seals the recomputed receipt and
compares HMACs to detect tampering.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from typing import Any, Dict, Optional, Tuple


class ReceiptTampered(Exception):
    """The receipt's HMAC no longer matches: it was altered or forged."""


class ReceiptFieldMissing(Exception):
    """The receipt lacks one or more mandatory binding fields (session/task_id/
    task_hash/run_id/action_id/candidate/original tolerance/verdict)."""


class SealerKeyUnavailable(Exception):
    """Raised when the trusted key file is absent but old data exists -- we must
    NOT generate a fresh key to 'cover' it; old receipts cannot be verified."""


REQUIRED_RECEIPT_FIELDS = (
    "session", "task_id", "task_hash", "run_id", "action_id",
    "candidate", "reference_fraction", "allowed_error_fraction", "verdict",
)


def _stable_bytes(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ReceiptSealer:
    """HMAC sealer for receipts. The key is trusted-side-only."""

    def __init__(self, key: Optional[bytes] = None):
        # Generate a fresh random key on the trusted side if none is provided.
        self._key = key if key is not None else os.urandom(32)

    @property
    def key_fingerprint(self) -> str:
        """A NON-reversible fingerprint so callers can log/audit which key was
        used WITHOUT exposing the key itself."""
        return hashlib.sha256(self._key).hexdigest()[:16]

    def seal(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        """Return the receipt with an hmac field attached. The key is NOT in the
        output."""
        out = dict(receipt)
        body = {k: v for k, v in out.items() if k != "hmac"}
        out["hmac"] = hmac.new(self._key, _stable_bytes(body), hashlib.sha256).hexdigest()
        return out

    def verify(self, receipt: Dict[str, Any]) -> bool:
        hmac_stored = receipt.get("hmac")
        if not hmac_stored:
            return False
        body = {k: v for k, v in receipt.items() if k != "hmac"}
        expected = hmac.new(self._key, _stable_bytes(body), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, hmac_stored)

    def assert_integrity(self, receipt: Dict[str, Any]) -> None:
        if not self.verify(receipt):
            raise ReceiptTampered("receipt HMAC mismatch; receipt was altered or forged")


def make_receipt(
    reference_fraction: str,
    allowed_error_fraction: str,
    candidate,
    diff_fraction: str,
    verdict: str,
    run_id: str,
    task_id: str,
    task_hash: str = "",
    session: str = "",
    action_id: str = "",
    reference_lo_fraction: str = "",
    reference_hi_fraction: str = "",
    allowed_error_min_fraction: str = "",
    allowed_error_max_fraction: str = "",
) -> Dict[str, Any]:
    """Build the base (unsealed) receipt for an independent verdict.

    The receipt carries every identity-binding field required by verify_run:
    session, task_id, task_hash, run_id, action_id (e.g. 'quad_plain', NOT the
    solver module name), the candidate, the original tolerance (as a fraction
    string), and the verdict. The reference answer lives ONLY here, on the
    trusted side.

    ODE domains additionally bind the reference INTERVAL (reference_lo_fraction /
    reference_hi_fraction) and the ALLOWED-ERROR interval (allowed_error_min_fraction
    / allowed_error_max_fraction) so that an INCONCLUSIVE verdict is fully
    reproducible from the receipt alone; for non-ODE domains these are empty
    strings (the ODE controller branch is the only one that populates them).
    The candidate may be None (an ODE solver that did not reach t1) -- JSON
    serializes it as null; it is NEVER a NaN/Infinity.
    """
    out = {
        "task_id": task_id,
        "run_id": run_id,
        "task_hash": task_hash,
        "session": session,
        "action_id": action_id,
        "reference_fraction": reference_fraction,   # trusted side only
        "allowed_error_fraction": allowed_error_fraction,
        "candidate": candidate,
        "abs_diff_fraction": diff_fraction,
        "verdict": verdict,
    }
    # ODE interval-binding fields (populated only for ODE runs; empty strings
    # for every other domain so verify_run's required-field check is unchanged).
    if reference_lo_fraction or reference_hi_fraction or allowed_error_min_fraction or allowed_error_max_fraction:
        out["reference_lo_fraction"] = reference_lo_fraction
        out["reference_hi_fraction"] = reference_hi_fraction
        out["allowed_error_min_fraction"] = allowed_error_min_fraction
        out["allowed_error_max_fraction"] = allowed_error_max_fraction
    return out
