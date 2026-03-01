from fastapi import Header, HTTPException
from typing import Optional


def require_api_key(x_api_key: Optional[str], expected: str) -> None:
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

