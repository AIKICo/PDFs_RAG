from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, EmailStr, Field

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    hashed_password: str
    is_active: bool
    created_at: datetime

class User(UserBase):
    is_active: bool

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(4, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

class ProcessedFile(BaseModel):
    file_path: str
    file_name: str
    page_count: int
    processed_at: str

class APIKeyCreate(BaseModel):
    name: str
    expires_in_days: int = Field(30, ge=1, le=365)

class APIKey(BaseModel):
    key_id: str
    name: str
    api_key: str
    created_at: datetime
    expires_at: datetime