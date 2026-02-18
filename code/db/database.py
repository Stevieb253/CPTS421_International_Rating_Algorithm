"""
database.py
SQLite database for user management and report storage
LOCK-SAFE VERSION
"""

import sqlite3
import hashlib
import secrets
from typing import Optional, List, Dict
from pathlib import Path


class Database:
    def __init__(self):
        db_dir = Path(__file__).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / "student_scoring.db"
        self._initialize_database()

    # Connection Handling (SAFE)

    def _get_connection(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=10,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")  # Better concurrency
        return conn

    def _initialize_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT,
                    email TEXT,
                    role TEXT DEFAULT 'reviewer',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_login TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    final_score REAL,
                    analyzed_by INTEGER,
                    analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    student_id TEXT,
                    reviewed_by INTEGER,
                    reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    pdf_blob BLOB,
                    file_size INTEGER
                )
            """)

            self._create_default_admin(cursor)

    # Password Handling

    def _hash_password(self, password: str, salt: str = None):
        if salt is None:
            salt = secrets.token_hex(32)

        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000
        ).hex()

        return pwd_hash, salt

    # Admin Creation

    def _create_default_admin(self, cursor):
        cursor.execute("SELECT COUNT(*) as count FROM users")
        if cursor.fetchone()["count"] == 0:
            pwd_hash, salt = self._hash_password("admin123")
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, role)
                VALUES (?, ?, ?, ?)
            """, ("admin", pwd_hash, salt, "admin"))
            print("✅ Default admin created (admin / admin123)")

    # User Management

    def create_user(self, username: str, password: str) -> Optional[int]:
        pwd_hash, salt = self._hash_password(password)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, password_hash, salt)
                    VALUES (?, ?, ?)
                """, (username, pwd_hash, salt))
                user_id = cursor.lastrowid
                self._log_activity(cursor, user_id, "user_created")
                return user_id
        except sqlite3.IntegrityError:
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM users
                WHERE username = ? AND is_active = 1
            """, (username,))
            user = cursor.fetchone()

            if not user:
                return None

            pwd_hash, _ = self._hash_password(password, user["salt"])
            if pwd_hash != user["password_hash"]:
                return None

            cursor.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user["id"],))

            self._log_activity(cursor, user["id"], "login")

            return dict(user)

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            return dict(user) if user else None

    # Activity Logging (NO new connection inside write)

    def _log_activity(self, cursor, user_id: int, action: str, details: str = None):
        cursor.execute("""
            INSERT INTO activity_log (user_id, action, details)
            VALUES (?, ?, ?)
        """, (user_id, action, details))

    def get_activity_log(self, limit: int = 100) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM activity_log
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # Analysis

    def save_analysis(self, student_id: str, final_score: float, user_id: int):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_records (student_id, final_score, analyzed_by)
                VALUES (?, ?, ?)
            """, (student_id, final_score, user_id))
            analysis_id = cursor.lastrowid
            self._log_activity(cursor, user_id, "analysis_created")
            return analysis_id

    # Reports

    def save_report(self, analysis_id: int, student_id: str,
                    user_id: int, pdf_blob: bytes):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports (analysis_id, student_id, reviewed_by, pdf_blob, file_size)
                VALUES (?, ?, ?, ?, ?)
            """, (analysis_id, student_id, user_id,
                  pdf_blob, len(pdf_blob)))
            report_id = cursor.lastrowid
            self._log_activity(cursor, user_id, "report_generated")
            return report_id

# Singleton

_db_instance = None

def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
