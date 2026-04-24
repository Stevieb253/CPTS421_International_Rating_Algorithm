"""
SQLite Database Layer for IARA
Handles user management, analysis records, reports, and activity logging
"""
import os
import sqlite3
import hashlib
import secrets
from typing import Optional, List, Dict
from pathlib import Path


class Database:
    """Thread-safe SQLite database with WAL mode for concurrent access"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection and schema"""
        db_dir = Path(__file__).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = Path(db_path) if db_path else db_dir / "student_scoring.db"
        self._initialize_schema()
        self._migrate_schema()

    # ═══════════════════════════════════════════════════════════════════════
    # DATABASE CONNECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def _get_connection(self) -> sqlite3.Connection:
        """Create thread-safe database connection with WAL mode"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=10,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ═══════════════════════════════════════════════════════════════════════
    # SCHEMA INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def _initialize_schema(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
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
            
            # Activity log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Analysis records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    final_score REAL,
                    analyzed_by INTEGER,
                    analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analyzed_by) REFERENCES users(id)
                )
            """)
            
            # Reports table (analysis_id nullable for flexibility)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    student_id TEXT,
                    reviewed_by INTEGER,
                    reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    pdf_blob BLOB,
                    file_size INTEGER,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_records(id),
                    FOREIGN KEY (reviewed_by) REFERENCES users(id)
                )
            """)
            
            conn.commit()
            self._create_default_admin(cursor)

    def _migrate_schema(self):
        """Apply schema migrations if needed"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if reports.analysis_id needs to be nullable
            cursor.execute("PRAGMA table_info(reports)")
            cols = {row['name']: row for row in cursor.fetchall()}
            
            analysis_id_col = cols.get('analysis_id')
            if analysis_id_col and analysis_id_col['notnull'] == 1:
                # print("🔧 Migrating reports table...")
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
                    SELECT id, analysis_id, student_id, reviewed_by, reviewed_at, pdf_blob, file_size
                    FROM reports
                """)
                
                cursor.execute("DROP TABLE reports")
                cursor.execute("ALTER TABLE reports_new RENAME TO reports")
                cursor.execute("PRAGMA foreign_keys=ON")
                conn.commit()
                # print("✅ Migration complete")

    def _create_default_admin(self, cursor):
        """Create default admin user if no users exist"""
        cursor.execute("SELECT COUNT(*) as count FROM users")
        if cursor.fetchone()["count"] == 0:
            pwd_hash, salt = self._hash_password("admin123")
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            """, ("admin", pwd_hash, salt, "Administrator", "admin"))
            # print("✅ Default admin created → username: admin | password: admin123")

    # ═══════════════════════════════════════════════════════════════════════
    # PASSWORD UTILITIES
    # ═══════════════════════════════════════════════════════════════════════
    
    def _hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        """Hash password with PBKDF2 and salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        
        return pwd_hash, salt

    # ═══════════════════════════════════════════════════════════════════════
    # USER MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_user(self, username: str, password: str, full_name: str = None,
                    email: str = None, role: str = 'reviewer') -> Optional[int]:
        """Create new user account"""
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
                conn.commit()
                return user_id
        except sqlite3.IntegrityError:
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and update last login"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,))
            user = cursor.fetchone()
            
            if not user:
                return None
            
            pwd_hash, _ = self._hash_password(password, user["salt"])
            if pwd_hash != user["password_hash"]:
                return None
            
            cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user["id"],))
            self._log_activity(cursor, user["id"], "login", f"User '{username}' logged in")
            conn.commit()
            
            return dict(user)

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Retrieve user by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            return dict(user) if user else None

    def get_all_users(self) -> List[Dict]:
        """Get all users (excluding passwords)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, full_name, email, role, is_active, created_at, last_login
                FROM users
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def toggle_user_status(self, user_id: int, is_active: int) -> bool:
        """Toggle user active/inactive status"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET is_active = ? WHERE id = ?", (is_active, user_id))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error toggling user status: {e}")
            return False

    def delete_user(self, user_id: int) -> bool:
        """Permanently delete a user"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════
    # ACTIVITY LOGGING
    # ═══════════════════════════════════════════════════════════════════════
    
    def _log_activity(self, cursor, user_id: int, action: str, details: str = None):
        """Internal activity logger (requires existing cursor/transaction)"""
        cursor.execute("""
            INSERT INTO activity_log (user_id, action, details)
            VALUES (?, ?, ?)
        """, (user_id, action, details))

    def log_activity(self, user_id: int, action: str, details: str = None):
        """Public activity logger (opens own connection)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            self._log_activity(cursor, user_id, action, details)
            conn.commit()

    def get_activity_log(self, limit: int = 100) -> List[Dict]:
        """Retrieve recent activity log"""
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

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS RECORDS
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_analysis(self, student_id: str, final_score: float, user_id: int) -> int:
        """Save analysis record"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_records (student_id, final_score, analyzed_by)
                VALUES (?, ?, ?)
            """, (student_id, final_score, user_id))
            
            analysis_id = cursor.lastrowid
            self._log_activity(cursor, user_id, "analysis_created",
                             f"Analysis saved for student '{student_id}'")
            conn.commit()
            return analysis_id

    def get_student_analyses(self, student_id: str) -> List[Dict]:
        """Get all analyses for a student"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM analysis_records
                WHERE student_id = ?
                ORDER BY analyzed_at DESC
            """, (student_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ═══════════════════════════════════════════════════════════════════════
    # REPORTS
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_report(self, analysis_id: Optional[int], student_id: str,
                    user_id: int, pdf_blob: bytes) -> int:
        """Save PDF report"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports (analysis_id, student_id, reviewed_by, pdf_blob, file_size)
                VALUES (?, ?, ?, ?, ?)
            """, (analysis_id, student_id, user_id, pdf_blob, len(pdf_blob)))
            
            report_id = cursor.lastrowid
            self._log_activity(cursor, user_id, "report_generated",
                             f"Report generated for student '{student_id}'")
            conn.commit()
            return report_id

    def get_report_by_id(self, report_id: int) -> Optional[Dict]:
        """Get report by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_reports(self, limit: int = 100) -> List[Dict]:
        """Get all reports with reviewer info"""
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
        """Get all reports for a student"""
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

    # ═══════════════════════════════════════════════════════════════════════
    # DASHBOARD STATISTICS
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_dashboard_stats(self) -> Dict:
        """Calculate dashboard statistics"""
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
            
            cursor.execute("SELECT AVG(final_score) as avg FROM analysis_records")
            row = cursor.fetchone()
            avg_score = round(float(row["avg"]), 2) if row["avg"] else 0.0
            
            cursor.execute("SELECT COUNT(*) as count FROM users WHERE is_active = 1")
            active_users = cursor.fetchone()["count"]
            
            return {
                'totalAnalyses': total_analyses,
                'totalReports': total_reports,
                'reportsThisWeek': reports_this_week,
                'avgFinalScore': avg_score,
                'activeUsers': active_users,
            }


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON PATTERN
# ═══════════════════════════════════════════════════════════════════════

_db_instance = None

def get_db() -> Database:
    """Get database singleton instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance