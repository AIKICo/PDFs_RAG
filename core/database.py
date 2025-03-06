import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional


class Database:
    def __init__(self, db_path: str):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = self._create_connection()
        self._init_tables()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a database connection."""
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        """Initialize the database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Create processed_files table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            page_count INTEGER,
            processed_at TEXT,
            metadata TEXT
        )
        ''')

        # Create users table for authentication
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            hashed_password TEXT,
            is_active BOOLEAN,
            created_at TEXT
        )
        ''')

        # Create API keys table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id TEXT PRIMARY KEY,
            user_id TEXT,
            api_key TEXT UNIQUE,
            name TEXT,
            created_at TEXT,
            expires_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (username)
        )
        ''')

        self.conn.commit()

    def is_file_processed(self, file_hash: str) -> bool:
        """Check if a file has already been processed based on its hash."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_hash FROM processed_files WHERE file_hash = ?", (file_hash,))
        result = cursor.fetchone()
        return result is not None

    def add_processed_file(self, file_hash: str, file_path: str, file_name: str,
                           page_count: int, metadata: Dict):
        """Add a processed file to the database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO processed_files VALUES (?, ?, ?, ?, datetime('now'), ?)",
            (
                file_hash,
                file_path,
                file_name,
                page_count,
                json.dumps(metadata)
            )
        )
        self.conn.commit()

    def get_processed_files(self) -> List[Dict]:
        """Get all processed files."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path, file_name, page_count, processed_at FROM processed_files")
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "file_path": row[0],
                "file_name": row[1],
                "page_count": row[2],
                "processed_at": row[3]
            })

        return result

    def add_user(self, username: str, email: str, hashed_password: str):
        """Add a new user to the database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, datetime('now'))",
            (username, email, hashed_password, True)
        )
        self.conn.commit()

    def get_user(self, username: str) -> Optional[Dict]:
        """Get a user by username."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT username, email, hashed_password, is_active FROM users WHERE username = ?",
                       (username,))
        row = cursor.fetchone()

        if not row:
            return None

        return {
            "username": row[0],
            "email": row[1],
            "hashed_password": row[2],
            "is_active": bool(row[3])
        }

    def add_api_key(self, key_id: str, user_id: str, api_key: str, name: str, expires_at: datetime):
        """Add a new API key."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, datetime('now'), ?)",
            (key_id, user_id, api_key, name, expires_at.isoformat())
        )
        self.conn.commit()

    def get_user_by_api_key(self, api_key: str) -> Optional[Dict]:
        """Get a user by API key."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT u.username, u.email, u.is_active, a.expires_at 
            FROM users u
            JOIN api_keys a ON u.username = a.user_id
            WHERE a.api_key = ? AND u.is_active = 1
        """, (api_key,))
        row = cursor.fetchone()

        if not row:
            return None

        expires_at = datetime.fromisoformat(row[3])
        if expires_at < datetime.utcnow():
            return None

        return {
            "username": row[0],
            "email": row[1],
            "is_active": bool(row[2])
        }

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()