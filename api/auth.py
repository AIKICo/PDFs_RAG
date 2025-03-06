import secrets
import uuid
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from jose import JWTError, jwt
from pydantic import ValidationError

from ..core.config import settings
from ..core.database import Database
from ..core.security import get_password_hash, verify_password
from ..models.schemas import TokenData, User, UserCreate, APIKey

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")
api_key_header = APIKeyHeader(name="X-API-Key")

db = Database(settings.DB_PATH)


def get_user(username: str):
    """Get a user from the database."""
    return db.get_user(username)


def authenticate_user(username: str, password: str):
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_user(user_data: UserCreate):
    """Create a new user."""
    user = get_user(user_data.username)
    if user:
        return None  # User already exists

    hashed_password = get_password_hash(user_data.password)
    db.add_user(user_data.username, user_data.email, hashed_password)

    return db.get_user(user_data.username)


def generate_api_key(username: str, name: str, expires_in_days: int = 30) -> APIKey:
    """Generate a new API key for a user."""
    key_id = str(uuid.uuid4())
    api_key = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

    db.add_api_key(key_id, username, api_key, name, expires_at)

    return APIKey(
        key_id=key_id,
        name=name,
        api_key=api_key,
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except (JWTError, ValidationError):
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception

    if not user["is_active"]:
        raise HTTPException(status_code=400, detail="Inactive user")

    return User(
        username=user["username"],
        email=user["email"],
        is_active=user["is_active"]
    )


async def get_current_user_from_api_key(api_key: str = Depends(api_key_header)) -> User:
    """Get the current user from API key."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "APIKey"},
    )

    user = db.get_user_by_api_key(api_key)
    if user is None:
        raise credentials_exception

    return User(
        username=user["username"],
        email=user["email"],
        is_active=user["is_active"]
    )
