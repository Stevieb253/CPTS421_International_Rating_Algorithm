#  Database-Backed Report System - Complete Deployment Guide

##  What This System Adds

### **Major Improvements**

**User Authentication** - Secure login with password hashing  
**Audit Trail** - Complete history of who reviewed what, when  
**Report History** - View all reports for any student  
**User Management** - Admin panel to create/manage users  
**Activity Logging** - Track all system actions  
**Dashboard** - Statistics and quick access  

---

##  Files Delivered

### **Core System Files**
1. `database.py` - SQLite database management
2. `app_routes_database.py` - Updated Flask routes
3. `report_generator.py` - PDF report generator
4. `index.html` - Updated UI with report features

### **New Template Files**
5.  `reports.html` - View all generated reports
6.  `student_detail.html` - Student history page
7.  `admin_users.html` - User management (admin only)
8.  `dashboard.html` - Stats dashboard

---

## Database Schema

The system creates **4 tables** automatically:

### **1. users**
Stores staff accounts with secure password hashing
```sql
- id (primary key)
- username (unique)
- password_hash (PBKDF2-HMAC-SHA256)
- salt (random per user)
- full_name
- email
- role ('admin' or 'reviewer')
- is_active (1/0)
- created_at, last_login
```

### **2. analysis_records**
Every student analysis saved with full details
```sql
- id (primary key)
- student_id, country, gpa, curriculum, travel_history
- pos_score, neg_score, final_score
- recommendation, essay metrics
- analyzed_by (user_id foreign key)
- analyzed_at (timestamp)
```

### **3. reports**
PDF/text reports stored as BLOBs
```sql
- id (primary key)
- analysis_id (links to analysis)
- student_id
- report_type ('pdf' or 'txt')
- staff_comments (reviewer's notes)
- reviewer_name
- reviewed_by (user_id foreign key)
- pdf_blob (binary data)
- txt_content (text reports)
- file_size (bytes)
- reviewed_at (timestamp)
```

### **4. activity_log**
Complete audit trail
```sql
- id (primary key)
- user_id
- action ('login', 'analysis_created', 'report_generated', etc.)
- details (additional context)
- timestamp
```

---

## User Workflows

### **For Staff Reviewers**

**1. Analyze a Student**
- Go to homepage
- Fill in student form
- Click "Analyze Student"
- Results appear

**2. Add Comments & Generate Report**
- After analysis, scroll to "Staff Comments" section
- Enter your review notes and justification
- Enter your name in "Reviewed by"
- Click "Download PDF Report"
- **Report is automatically saved to database AND downloaded**

**3. View Report History**
- Click "Reports" in nav
- See all reports ever generated
- Click student ID to see their full history
- Download any report again anytime

**4. View Student History**
- Click any student ID from reports page
- See ALL analyses ever done for that student
- See ALL reports generated
- Track who reviewed them and when

### **For Administrators**

**1. Create New Users**
- Go to " Users" page
- Fill in the form
- Choose role: Admin or Reviewer
- Click "Create User"

**2. View Activity**
- Go to "Users" page
- Scroll to "Recent Activity"
- See complete audit log

**3. View Dashboard**
- Go to " Dashboard"
- See total stats
- Quick links to all features

---

##  Key Features Explained

### **1. Automatic Report Storage**
```python
# When a report is generated:
db.save_report(
    analysis_id    = 123,           # Links to the analysis
    student_id     = "STU_2025_001",
    user_id        = 5,             # Who generated it
    staff_comments = "Recommend fast-track...",
    reviewer_name  = "Dr. Jane Smith",
    pdf_blob       = pdf_bytes,     # Stored as BLOB
    report_type    = 'pdf'
)
```

### **2. Complete Audit Trail**
Every action logged:
```
user_id=5, action='login', details='User jsmith logged in'
user_id=5, action='analysis_created', details='Analyzed student STU_2025_001'
user_id=5, action='report_generated', details='PDF report for STU_2025_001'
```

### **3. Secure Authentication**
```python
# Passwords hashed with PBKDF2-HMAC-SHA256
# 100,000 iterations
# Random salt per user
# No plaintext passwords ever stored
```

### **4. Role-Based Access**
- **Reviewers**: Analyze students, generate reports
- **Admins**: All above + user management + activity logs

---

## Database File

### **Location**
`code/student_scoring.db` (SQLite file)

### **Backup**
```bash
# Backup database
cp code/student_scoring.db backups/backup_$(date +%Y%m%d).db

# Automated daily backups (add to cron)
0 2 * * * cp /path/to/code/student_scoring.db /path/to/backups/backup_$(date +\%Y\%m\%d).db
```

### **Size Estimation**
- Average PDF: ~7 KB
- 1000 reports: ~7 MB
- Database stays manageable even with thousands of reports

### **Query the Database**
```bash
sqlite3 code/student_scoring.db

# See all tables
.tables

# Count reports
SELECT COUNT(*) FROM reports;

# See recent analyses
SELECT student_id, final_score, analyzed_at FROM analysis_records ORDER BY analyzed_at DESC LIMIT 10;

# Find a student's reports
SELECT * FROM reports WHERE student_id = 'STU_2025_001';
```

---

## Navigation Map

```
/login              → Login page
  ↓
/ (index)           → Student analysis form
  ├→ /dashboard     → Stats overview
  ├→ /reports       → All generated reports
  │   └→ /student/<id>  → Specific student history
  ├→ /analytics     → Charts & analytics
  ├→ /batch         → CSV batch processing
  ├→ /admin/users   → User management (admin only)
  └→ /logout        → Logout
```

---

##  Common Tasks

### **Create a New User**
```bash
python
>>> from database import get_db
>>> db = get_db()
>>> db.create_user('jsmith', 'SecurePass123', 'John Smith', 'john@wsu.edu', 'reviewer')
```

### **Find All Reports for a Student**
```bash
python
>>> from database import get_db
>>> db = get_db()
>>> reports = db.get_student_reports('STU_2025_001')
>>> for r in reports:
...     print(f"{r['reviewed_at']}: {r['reviewer_name']}")
```

### **Get Activity Log**
```bash
python
>>> from database import get_db
>>> db = get_db()
>>> logs = db.get_activity_log(limit=20)
>>> for log in logs:
...     print(f"{log['timestamp']}: {log['username']} - {log['action']}")
```

---

##  Troubleshooting

### **"Table already exists" error**
Database already initialized. This is fine - continue.

### **"Database is locked" error**
Another process is accessing the database. Close other connections.

### **Can't login with admin/admin123**
Check if database was created:
```bash
ls -la code/student_scoring.db
python -c "from database import get_db; u=get_db().authenticate_user('admin','admin123'); print('OK' if u else 'FAIL')"
```

### **Reports not saving**
Check the analysis returns an `analysisId`:
```javascript
console.log('Analysis ID:', result.analysisId);  // Should be a number
```

### **"Access denied" when accessing admin pages**
Your user role is 'reviewer'. Only 'admin' role can access `/admin/*` routes.

---

##  Security Best Practices

### **1. Change Default Admin Password**
```python
from database import get_db
db = get_db()
db.update_password(1, 'NewSecurePassword456!')
```

### **2. Use Strong Passwords**
- Minimum 8 characters
- Mix of letters, numbers, symbols

### **3. Regular Backups**
- Daily database backups
- Store offsite

### **4. User Management**
- Deactivate users who leave
- Regular user audits

### **5. HTTPS in Production**
- Use HTTPS (not HTTP)
- Set up SSL certificate

---

##  Scaling Considerations

### **Current Capacity**
-  SQLite handles 1000s of reports easily
-  BLOBs up to 1 GB each (we use ~7 KB)
-  Concurrent reads no problem

### **If You Need More Later**
Migrate to PostgreSQL or MySQL:
```python
# Minor changes to database.py
# Same API, different backend
# All queries work the same
```

---

##  What Staff See

### **Before (Without Database)**
1. Analyze student
2. Download PDF manually
3. Save to computer
4. File gets lost in folders
5. No way to find old reports
6. No idea who reviewed what

### **After (With Database)**
1. Analyze student
2. Add comments
3. Click "Generate Report"
4. **Report auto-saved to system**
5. PDF downloads for their records
6. Anyone can search student ID
7. See complete history instantly
8. Full audit trail maintained

---

## Next Steps

### **Immediate**
1. Deploy the system
2. Create user accounts for staff
3. Train staff on workflow
4. Start using it!

### **Optional Enhancements** (Tell me if you want these)
1. Email reports automatically
2. Report approval workflow
3. Student portal (read-only access)
4. Advanced search filters
5. Export entire student file to ZIP
6. Integration with existing student system
7. Scheduled reports (weekly summaries)

---

## Reference

### **Default Login**
Username: `admin`  
Password: `admin123`

### **Database File**
`code/student_scoring.db`

### **Key Routes**
- Analysis: `/`
- Reports: `/reports`
- Dashboard: `/dashboard`
- Admin: `/admin/users`

### **User Roles**
- `admin`: Full access
- `reviewer`: Cannot manage users

---
