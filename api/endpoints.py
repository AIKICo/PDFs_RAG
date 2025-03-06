import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm

from . import auth
from ..core.config import settings
from ..models.schemas import (
    Token, User, UserCreate, QueryRequest, QueryResponse,
    ProcessedFile, APIKeyCreate, APIKey
)
from ..pdf.processor import PDFProcessor
from ..utils.helpers import save_uploaded_file

router = APIRouter()


# Authentication endpoints
@router.post("/auth/register", response_model=User)
def register_user(user_data: UserCreate):
    """Register a new user."""
    user = auth.create_user(user_data)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )

    return User(
        username=user["username"],
        email=user["email"],
        is_active=user["is_active"]
    )


@router.post("/auth/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get an access token."""
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password"
        )

    access_token = auth.create_access_token(data={"sub": user["username"]})

    return Token(access_token=access_token, token_type="bearer")


@router.post("/auth/api-keys", response_model=APIKey)
def create_api_key(key_data: APIKeyCreate, current_user: User = Depends(auth.get_current_user)):
    """Create a new API key for the user."""
    return auth.generate_api_key(
        current_user.username,
        key_data.name,
        key_data.expires_in_days
    )


# PDF processing endpoints
@router.post("/pdfs/upload")
async def upload_pdfs(
        files: List[UploadFile] = File(...),
        current_user: User = Depends(auth.get_current_user)
):
    """Upload and process PDF files."""
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    processor = PDFProcessor()
    try:
        # Save uploaded files
        pdf_paths = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue

            file_path = await save_uploaded_file(file, settings.UPLOAD_DIR)
            pdf_paths.append(file_path)

        # Process PDFs
        if pdf_paths:
            result = processor.process_pdfs(pdf_paths)
            return result
        else:
            raise HTTPException(
                status_code=400,
                detail="No valid PDF files were uploaded"
            )
    finally:
        processor.close()


@router.get("/pdfs", response_model=List[ProcessedFile])
def list_pdfs(current_user: User = Depends(auth.get_current_user)):
    """List all processed PDF files."""
    processor = PDFProcessor()
    try:
        return processor.list_processed_files()
    finally:
        processor.close()


@router.post("/query", response_model=QueryResponse)
def query_pdfs(
        query_req: QueryRequest,
        current_user: User = Depends(auth.get_current_user_from_api_key)
):
    """Query the RAG system."""
    processor = PDFProcessor()
    try:
        if not processor.list_processed_files():
            raise HTTPException(
                status_code=400,
                detail="No PDF files have been processed yet"
            )

        answer, sources = processor.query(query_req.question, query_req.top_k)

        return QueryResponse(
            answer=answer,
            sources=sources
        )
    finally:
        processor.close()
