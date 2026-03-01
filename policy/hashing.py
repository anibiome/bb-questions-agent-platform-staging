from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence


def stable_hash_hex(obj: Any) -> str:
    """
    Stable SHA256 hash for audit logs.

    - Serializes JSON with sorted keys
    - Ensures deterministic hashing across runs
    """
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_hash_floats(values: Sequence[float], *, quantize: int = 4) -> str:
    """
    Hash a float vector with optional quantization (privacy + reproducibility).
    """
    q = int(max(0, quantize))
    if q > 0:
        normed = [round(float(v), q) for v in values]
    else:
        normed = [float(v) for v in values]
    payload = json.dumps(normed, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

