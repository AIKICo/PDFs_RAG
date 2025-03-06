from datetime import datetime, timedelta
from typing import Optional

from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    username: Optional[str] = None


def verify_password(plain_password, hashed_password):
    """Verify if a password matches the hashed version."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Generate a hash from a plaintext password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt
