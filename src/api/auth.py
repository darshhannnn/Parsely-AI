from fastapi import Header, HTTPException, status, Depends
from typing import Optional
import os

# You can set this in an environment variable or .env file for production
API_KEY = os.environ.get("INSURANCE_API_KEY", "testkey123")

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "API Key"},
        )
