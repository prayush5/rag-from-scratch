import hmac
from fastapi import Header, status, HTTPException
from app.core.config import settings

async def verify_admin(x_admin_key: str = Header(...)):
    """Protects mutating document endpoints (upload/delete) with a shared secret.
    Not per-user auth - just a single admin key you know
    """
    if not hmac.compare_digest(x_admin_key, settings.ADMIN_SECRET):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid admin key."
        )
        