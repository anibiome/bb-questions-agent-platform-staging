import hmac

from fastapi import Header, HTTPException
from typing import Optional


def require_api_key(
    x_api_key: Optional[str],
    expected: str,
    *,
    authorization: Optional[str] = None,
) -> None:
    presented = (x_api_key or "").strip()
    if not presented and authorization:
        auth = str(authorization).strip()
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")
