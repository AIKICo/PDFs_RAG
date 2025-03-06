import os
from pathlib import Path

from fastapi import UploadFile


async def save_uploaded_file(uploaded_file: UploadFile, upload_dir: Path) -> str:
    """Save an uploaded file and return the path."""
    file_path = os.path.join(upload_dir, uploaded_file.filename)

    # Create file
    with open(file_path, "wb") as f:
        content = await uploaded_file.read()
        f.write(content)

    return file_path