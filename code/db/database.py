"""
database.py
SQLite database for user management and report storage
LOCK-SAFE VERSION — includes all methods required by app.py
"""

import os
import sqlite3
import hashlib
import secrets
from typing import Optional, List, Dict
from pathlib import Path


class Database:
    def __init__(self):
        db_dir = Path(__file__).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(os.environ.get("DB_PATH", str(db_dir / "student_scoring.db")))
        self._initialize_database()
        self._migrate_database()

    # ─── Connection ───────────────────────────────────────────────────────────

    def _get_connection(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=10,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    # ─── Schema Setup ─────────────────────────────────────────────────────────

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

            # analysis_id is nullable — no NOT NULL constraint
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

    def _migrate_database(self):
        """
        Fix any existing reports table that was created with NOT NULL on analysis_id.
        Runs silently if the table is already correct.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if analysis_id column allows NULL
            cursor.execute("PRAGMA table_info(reports)")
            cols = {row['name']: row for row in cursor.fetchall()}

            analysis_id_col = cols.get('analysis_id')
            if analysis_id_col and analysis_id_col['notnull'] == 1:
                print("🔧 Migrating reports table to allow nullable analysis_id...")
                cursor.execute("PRAGMA foreign_keys=OFF")
                cursor.execute("""
                    CREATE TABLE reports_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id INTEGER,
                        student_id TEXT,
                        reviewed_by INTEGER,
                        reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        pdf_blob BLOB,
                        file_size INTEGER
                    )
                """)
                cursor.execute("""
                    INSERT INTO reports_new
                    SELECT id, analysis_id, student_id, reviewed_by,
                           reviewed_at, pdf_blob, file_size
                    FROM reports
                """)
                cursor.execute("DROP TABLE reports")
                cursor.execute("ALTER TABLE reports_new RENAME TO reports")
                cursor.execute("PRAGMA foreign_keys=ON")
                print("✅ Migration complete.")

    # ─── Password ─────────────────────────────────────────────────────────────

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

    # ─── Admin bootstrap ──────────────────────────────────────────────────────

    def _create_default_admin(self, cursor):
        cursor.execute("SELECT COUNT(*) as count FROM users")
        if cursor.fetchone()["count"] == 0:
            pwd_hash, salt = self._hash_password("admin123")
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            """, ("admin", pwd_hash, salt, "Administrator", "admin"))
            print("✅ Default admin created  →  username: admin  |  password: admin123")
    
    

    # ─── User Management ──────────────────────────────────────────────────────

    def create_user(self, username: str, password: str,
                    full_name: str = None, email: str = None,
                    role: str = 'reviewer') -> Optional[int]:
        pwd_hash, salt = self._hash_password(password)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, password_hash, salt, full_name, email, role)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (username, pwd_hash, salt, full_name, email, role))
                user_id = cursor.lastrowid
                self._log_activity(cursor, user_id, "user_created",
                                   f"New user '{username}' created with role '{role}'")
                return user_id
        except sqlite3.IntegrityError:
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM users WHERE username = ? AND is_active = 1
            """, (username,))
            user = cursor.fetchone()
            if not user:
                return None
            pwd_hash, _ = self._hash_password(password, user["salt"])
            if pwd_hash != user["password_hash"]:
                return None
            cursor.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            """, (user["id"],))
            self._log_activity(cursor, user["id"], "login",
                               f"User '{username}' logged in")
            return dict(user)

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            return dict(user) if user else None

    def get_all_users(self) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, full_name, email, role, is_active,
                       created_at, last_login
                FROM users
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    

    def delete_user(self, user_id):
        """Permanently delete a user."""
        try:
            self.cursor.execute(
                'DELETE FROM users WHERE id = ?',
                (user_id,)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False
    
    
    def get_user_by_id(self, user_id):
        """Get user by ID."""
        try:
            self.cursor.execute(
                'SELECT * FROM users WHERE id = ?',
                (user_id,)
            )
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    # def toggle_user_status(self, user_id, is_active):
    #     """Toggle user active/inactive status."""
    # try:
    #     self.cursor.execute(
    #         '''UPDATE users SET is_active = ? WHERE id = ?''',
    #         (is_active, user_id)
    #     )
    #     self.conn.commit()
    #     return True
    # except Exception as e:
    #     print(f"Error toggling user status: {e}")
    #     return False
    
    # ─── Activity Logging ─────────────────────────────────────────────────────

    def _log_activity(self, cursor, user_id: int, action: str, details: str = None):
        """Internal — must be called with an existing cursor inside a transaction."""
        cursor.execute("""
            INSERT INTO activity_log (user_id, action, details)
            VALUES (?, ?, ?)
        """, (user_id, action, details))

    def log_activity(self, user_id: int, action: str, details: str = None):
        """Public — opens its own connection."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO activity_log (user_id, action, details)
                VALUES (?, ?, ?)
            """, (user_id, action, details))

    def get_activity_log(self, limit: int = 100) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT al.*, u.username
                FROM activity_log al
                LEFT JOIN users u ON al.user_id = u.id
                ORDER BY al.timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # ─── Analysis Records ─────────────────────────────────────────────────────

    def save_analysis(self, student_id: str, final_score: float,
                      user_id: int) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_records (student_id, final_score, analyzed_by)
                VALUES (?, ?, ?)
            """, (student_id, final_score, user_id))
            analysis_id = cursor.lastrowid
            self._log_activity(cursor, user_id, "analysis_created",
                               f"Analysis saved for student '{student_id}'")
            return analysis_id

    def get_student_analyses(self, student_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM analysis_records
                WHERE student_id = ?
                ORDER BY analyzed_at DESC
            """, (student_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ─── Reports ──────────────────────────────────────────────────────────────

    def save_report(self, analysis_id, student_id: str,
                    user_id: int, pdf_blob: bytes) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports
                    (analysis_id, student_id, reviewed_by, pdf_blob, file_size)
                VALUES (?, ?, ?, ?, ?)
            """, (analysis_id, student_id, user_id,
                  pdf_blob, len(pdf_blob)))
            report_id = cursor.lastrowid
            self._log_activity(cursor, user_id, "report_generated",
                               f"Report generated for student '{student_id}'")
            return report_id

    def get_report_by_id(self, report_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM reports WHERE id = ?
            """, (report_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_reports(self, limit: int = 100) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.id, r.analysis_id, r.student_id, r.reviewed_at,
                       r.file_size, u.username AS reviewer
                FROM reports r
                LEFT JOIN users u ON r.reviewed_by = u.id
                ORDER BY r.reviewed_at DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_student_reports(self, student_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.id, r.analysis_id, r.student_id, r.reviewed_at,
                       r.file_size, u.username AS reviewer
                FROM reports r
                LEFT JOIN users u ON r.reviewed_by = u.id
                WHERE r.student_id = ?
                ORDER BY r.reviewed_at DESC
            """, (student_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ─── Dashboard Stats ──────────────────────────────────────────────────────

    def get_dashboard_stats(self) -> Dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM analysis_records")
            total_analyses = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM reports")
            total_reports = cursor.fetchone()["count"]

            cursor.execute("""
                SELECT COUNT(*) as count FROM reports
                WHERE reviewed_at >= datetime('now', '-7 days')
            """)
            reports_this_week = cursor.fetchone()["count"]

            cursor.execute("""
                SELECT AVG(final_score) as avg FROM analysis_records
            """)
            row = cursor.fetchone()
            avg_score = round(float(row["avg"]), 2) if row["avg"] else 0.0

            cursor.execute("SELECT COUNT(*) as count FROM users WHERE is_active = 1")
            active_users = cursor.fetchone()["count"]

            return {
                'totalAnalyses':    total_analyses,
                'totalReports':     total_reports,
                'reportsThisWeek':  reports_this_week,
                'avgFinalScore':    avg_score,
                'activeUsers':      active_users,
            }


# ─── Singleton ────────────────────────────────────────────────────────────────

_db_instance = None

def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance