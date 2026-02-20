"""
Flask Web Application for International Student Scoring System
Simplified version without MongoDB - Uses in-memory storage and file exports
"""

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from student_analyzer import StudentAnalyzer
import os
from datetime import datetime, timedelta
import io
import csv
import pandas as pd
import json
import re
from financial_fraud_detector import analyze_financial_pdf
from transcript_fraud_detector import analyze_transcript_pdf
from textwrap import shorten
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from typing import Optional, Dict
import uuid
from werkzeug.utils import secure_filename



app = Flask(__name__)
app.config['SECRET_KEY'] = 'wsu-student-scoring-secret-key-2025-change-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize analyzer
analyzer = StudentAnalyzer()

# In-memory storage
IN_MEMORY_USERS = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'email': 'admin@wsu.edu',
        'role': 'admin'
    }
}

IN_MEMORY_STUDENTS = []

# Sample data
SAMPLE_DATA = {
    'high': {
        'studentId': 'STU_2025_001',
        'country': 'South Korea',
        'gpa': 3.9,
        'curriculum': 'IGCSE/IB',
        'travelHistory': 'SEVIS/Multiple US trips',
        'essayText': '''I am deeply passionate about pursuing my educational goals at Washington State University. Throughout my academic career, I have demonstrated strong motivation and dedication to achieving excellence in computer science. My dream is to contribute to the field of artificial intelligence and use my skills to make a positive impact on society. I have consistently aspired to become a leader in technology and innovation, and I believe WSU provides the perfect environment to help me achieve these goals. During my time at an international school, I developed a keen interest in machine learning and its applications. I am inspired by the potential of technology to solve real-world problems and motivated to dedicate my career to this pursuit.''',
        'negFactors': []
    },
    'medium': {
        'studentId': 'STU_2025_002',
        'country': 'India',
        'gpa': 3.2,
        'curriculum': 'Standard Intl Secondary',
        'travelHistory': '1 listed or multiple non-listed',
        'essayText': '''I want to study at Washington State University because it has good programs. I have worked hard in my studies and believe I can do well. My goal is to get a degree and find a good job after graduation. I think studying in the United States will give me better opportunities for my future career.''',
        'negFactors': ['bankDocsPending']
    },
    'low': {
        'studentId': 'STU_2025_003',
        'country': 'Bangladesh',
        'gpa': 2.5,
        'curriculum': 'N/A',
        'travelHistory': 'No travel abroad',
        'essayText': '''I need to study at university. Education is important. I will study hard.''',
        'negFactors': ['reqAppFeeWaiver', 'cannotPayFee', 'reqEnrollmentFeeWaiver', 'bankDocsPending']
    }
}


# ==================== AUTHENTICATION ====================

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        user = IN_MEMORY_USERS.get(username)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = username
            session['username'] = username
            session['role'] = user.get('role', 'staff')
            session.permanent = True
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('login'))


# ==================== MAIN ROUTES ====================

@app.route('/')
@login_required
def index():
    """Render main page"""
    return render_template('index.html', username=session.get('username'))


@app.route('/analytics')
@login_required
def analytics_page():
    """Analytics dashboard"""
    return render_template('analytics.html', username=session.get('username'))


@app.route('/batch')
@login_required
def batch_page():
    """Batch processing page"""
    return render_template('batch.html', username=session.get('username'))

@app.route('/financial')
@login_required
def financial_page():
    """Financial document fraud screening page"""
    return render_template('financial.html', username=session.get('username'))

@app.route('/transcript')
@login_required
def transcript_page():
    return render_template('transcript.html', username=session.get('username'))


# ==================== API ENDPOINTS ====================

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    """API endpoint for student analysis"""
    try:
        data = request.json
        
        # Extract data
        student_id = data.get('studentId', '')
        country = data.get('country', '')
        gpa = float(data.get('gpa', 0))
        curriculum = data.get('curriculum', '')
        travel_history = data.get('travelHistory', '')
        essay_text = data.get('essayText', '')
        neg_factors = data.get('negFactors', [])
        
        # Validate required fields
        if not curriculum or not travel_history:
            return jsonify({
                'error': 'Missing required fields: curriculum and travel history'
            }), 400
        
        # Perform analysis
        result = analyzer.analyze_student(
            gpa=gpa,
            curriculum=curriculum,
            travel_history=travel_history,
            essay_text=essay_text,
            neg_factors=neg_factors
        )
        
        # Save to in-memory storage
        student_record = {
            'studentId': student_id,
            'country': country,
            'gpa': gpa,
            'curriculum': curriculum,
            'travelHistory': travel_history,
            'essayText': essay_text,
            'negFactors': neg_factors,
            'posScore': result.pos_score,
            'negScore': result.neg_score,
            'finalScore': result.final_score,
            'breakdown': result.breakdown,
            'essayAnalysis': {
                'clarityFocus': result.essay_analysis.clarity_focus,
                'developmentOrganization': result.essay_analysis.development_organization,
                'creativityStyle': result.essay_analysis.creativity_style,
                'totalScore': result.essay_analysis.total_score,
                'insights': result.essay_analysis.insights
            },
            'recommendation': result.recommendation,
            'analyzedBy': session.get('username'),
            'analyzedAt': datetime.utcnow().isoformat()
        }
        
        # Update or append
        existing_index = next((i for i, s in enumerate(IN_MEMORY_STUDENTS) if s.get('studentId') == student_id), None)
        if existing_index is not None:
            IN_MEMORY_STUDENTS[existing_index] = student_record
        else:
            IN_MEMORY_STUDENTS.append(student_record)
        
        # Format response
        response = {
            'posScore': result.pos_score,
            'negScore': result.neg_score,
            'finalScore': result.final_score,
            'breakdown': result.breakdown,
            'essayAnalysis': {
                'clarityFocus': result.essay_analysis.clarity_focus,
                'developmentOrganization': result.essay_analysis.development_organization,
                'creativityStyle': result.essay_analysis.creativity_style,
                'totalScore': result.essay_analysis.total_score,
                'insights': result.essay_analysis.insights
            },
            'recommendation': result.recommendation
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-analyze', methods=['POST'])
@login_required
def batch_analyze():
    """Batch process multiple students"""
    try:
        students = request.json.get('students', [])
        results = []
        
        for student_data in students:
            try:
                result = analyzer.analyze_student(
                    gpa=float(student_data.get('gpa', 0)),
                    curriculum=student_data.get('curriculum', ''),
                    travel_history=student_data.get('travelHistory', ''),
                    essay_text=student_data.get('essayText', ''),
                    neg_factors=student_data.get('negFactors', [])
                )
                
                # Save to in-memory storage
                student_record = {
                    'studentId': student_data.get('studentId', ''),
                    'country': student_data.get('country', ''),
                    'gpa': float(student_data.get('gpa', 0)),
                    'curriculum': student_data.get('curriculum', ''),
                    'travelHistory': student_data.get('travelHistory', ''),
                    'essayText': student_data.get('essayText', ''),
                    'negFactors': student_data.get('negFactors', []),
                    'posScore': result.pos_score,
                    'negScore': result.neg_score,
                    'finalScore': result.final_score,
                    'breakdown': result.breakdown,
                    'recommendation': result.recommendation,
                    'analyzedBy': session.get('username'),
                    'analyzedAt': datetime.utcnow().isoformat()
                }
                
                IN_MEMORY_STUDENTS.append(student_record)
                
                results.append({
                    'studentId': student_data.get('studentId'),
                    'success': True,
                    'finalScore': result.final_score,
                    'recommendation': result.recommendation
                })
            except Exception as e:
                results.append({
                    'studentId': student_data.get('studentId', 'Unknown'),
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({'results': results, 'total': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/students', methods=['GET'])
@login_required
def get_students():
    """Get all student records"""
    try:
        return jsonify(IN_MEMORY_STUDENTS)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/student/<student_id>', methods=['GET'])
@login_required
def get_student(student_id):
    """Get specific student record"""
    try:
        student = next((s for s in IN_MEMORY_STUDENTS if s.get('studentId') == student_id), None)
        if student:
            return jsonify(student)
        return jsonify({'error': 'Student not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics/summary', methods=['GET'])
@login_required
def analytics_summary():
    """Get analytics summary"""
    try:
        total_students = len(IN_MEMORY_STUDENTS)
        
        # Score distribution
        high_potential = sum(1 for s in IN_MEMORY_STUDENTS if s.get('finalScore', 0) >= 30)
        medium_risk = sum(1 for s in IN_MEMORY_STUDENTS if 20 <= s.get('finalScore', 0) < 30)
        high_risk = sum(1 for s in IN_MEMORY_STUDENTS if s.get('finalScore', 0) < 20)
        
        # Country distribution
        country_counts = {}
        for student in IN_MEMORY_STUDENTS:
            country = student.get('country', 'Unknown')
            country_counts[country] = country_counts.get(country, 0) + 1
        
        country_stats = [{'_id': k, 'count': v} for k, v in sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        # Average scores
        if total_students > 0:
            avg_gpa = sum(s.get('gpa', 0) for s in IN_MEMORY_STUDENTS) / total_students
            avg_pos = sum(s.get('posScore', 0) for s in IN_MEMORY_STUDENTS) / total_students
            avg_neg = sum(s.get('negScore', 0) for s in IN_MEMORY_STUDENTS) / total_students
            avg_final = sum(s.get('finalScore', 0) for s in IN_MEMORY_STUDENTS) / total_students
        else:
            avg_gpa = avg_pos = avg_neg = avg_final = 0
        
        return jsonify({
            'totalStudents': total_students,
            'distribution': {
                'highPotential': high_potential,
                'mediumRisk': medium_risk,
                'highRisk': high_risk
            },
            'countryStats': country_stats,
            'averages': {
                'avgGPA': avg_gpa,
                'avgPosScore': avg_pos,
                'avgNegScore': avg_neg,
                'avgFinalScore': avg_final
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/csv', methods=['POST'])
@login_required
def export_csv():
    """Export students to CSV"""
    try:
        data = request.json
        student_ids = data.get('studentIds', [])
        
        # Get students
        if student_ids:
            students = [s for s in IN_MEMORY_STUDENTS if s.get('studentId') in student_ids]
        else:
            students = IN_MEMORY_STUDENTS
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            'Student ID', 'Country', 'GPA', 'Curriculum', 'Travel History',
            'POS Score', 'NEG Score', 'Final Score', 'Recommendation',
            'Essay - Clarity & Focus', 'Essay - Development & Organization', 
            'Essay - Creativity & Style', 'Essay Total', 'Analyzed At'
        ])
        
        # Data
        for student in students:
            essay = student.get('essayAnalysis', {})
            writer.writerow([
                student.get('studentId', ''),
                student.get('country', ''),
                student.get('gpa', ''),
                student.get('curriculum', ''),
                student.get('travelHistory', ''),
                student.get('posScore', ''),
                student.get('negScore', ''),
                student.get('finalScore', ''),
                student.get('recommendation', ''),
                essay.get('clarityFocus', ''),
                essay.get('developmentOrganization', ''),
                essay.get('creativityStyle', ''),
                essay.get('totalScore', ''),
                student.get('analyzedAt', '')
            ])
        
        # Create file
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'student_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/excel', methods=['POST'])
@login_required
def export_excel():
    """Export students to Excel"""
    try:
        data = request.json
        student_ids = data.get('studentIds', [])
        
        # Get students
        if student_ids:
            students = [s for s in IN_MEMORY_STUDENTS if s.get('studentId') in student_ids]
        else:
            students = IN_MEMORY_STUDENTS
        
        # Prepare data for DataFrame
        export_data = []
        for student in students:
            essay = student.get('essayAnalysis', {})
            export_data.append({
                'Student ID': student.get('studentId', ''),
                'Country': student.get('country', ''),
                'GPA': student.get('gpa', ''),
                'Curriculum': student.get('curriculum', ''),
                'Travel History': student.get('travelHistory', ''),
                'POS Score': student.get('posScore', ''),
                'NEG Score': student.get('negScore', ''),
                'Final Score': student.get('finalScore', ''),
                'Recommendation': student.get('recommendation', ''),
                'Essay - Clarity & Focus': essay.get('clarityFocus', ''),
                'Essay - Development & Organization': essay.get('developmentOrganization', ''),
                'Essay - Creativity & Style': essay.get('creativityStyle', ''),
                'Essay Total Score': essay.get('totalScore', ''),
                'Analyzed At': student.get('analyzedAt', '')
            })
        
        # Create Excel file
        df = pd.DataFrame(export_data)
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Student Analysis')
        
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'student_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def build_financial_txt_report(
    result: dict,
    reviewer: str,
    comments: str = "",
    meta: Optional[Dict] = None,
    include_header: bool = True
) -> str:

    meta = meta or {}
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    file_path = result.get("file_path", "")
    doc_sev = result.get("doc_severity", "UNKNOWN")
    pages = result.get("pages", []) or []

    def safe(s: str, width: int = 240) -> str:
        return shorten((s or "").strip().replace("\n", " "), width=width, placeholder="...")

    lines = []

    if include_header:
        lines.append("INTERNATIONAL APPLICANT RATING ALGORITHM (IARA)")
        lines.append("Financial Document Fraud Screening Report")
        lines.append("=" * 78)
        lines.append(f"Generated: {now}")
        lines.append(f"Reviewer: {reviewer}")

        if meta.get("studentId"):
            lines.append(f"Student ID: {meta.get('studentId')}")
        if meta.get("applicantName"):
            lines.append(f"Applicant Name: {meta.get('applicantName')}")
        if meta.get("program"):
            lines.append(f"Program/Term: {meta.get('program')}")
        lines.append("-" * 78)


    lines.append(f"Document: {os.path.basename(file_path) if file_path else '(uploaded file)'}")
    lines.append(f"Document Severity: {doc_sev}")
    lines.append(f"Pages Analyzed: {len(pages)}")
    lines.append("-" * 78)

    # Only include fraud-positive signals in the justification section (clean & defensible)
    for p in pages:
        pnum = p.get("page_number")
        sev = p.get("severity", "UNKNOWN")
        conf = float(p.get("confidence", 0.0) or 0.0)

        summary = safe(p.get("ai_summary", ""), width=260)
        lines.append(f"Page {pnum}: Severity={sev}  Confidence={conf:.2f}")
        if summary:
            lines.append(f"  Summary: {summary}")

        sigs = p.get("fraud_signals", []) or []
        fraud_pos = [
            s for s in sigs
            if isinstance(s, dict) and (s.get("polarity") == "fraud_positive")
        ]

        if fraud_pos:
            lines.append("  Fraud-Positive Signals:")
            for s in fraud_pos[:12]:
                signal = safe(s.get("signal", ""), width=260)
                cat = s.get("category", "other")
                src = s.get("source", "llm")
                sc = float(s.get("confidence", 0.0) or 0.0)
                lines.append(f"    - [{cat}] ({src}, {sc:.2f}) {signal}")
        else:
            lines.append("  Fraud-Positive Signals: None detected on this page.")

        lines.append("")

    lines.append("-" * 78)
    lines.append("Reviewer Comments (for applicant file):")
    lines.append((comments or "").strip() if (comments or "").strip() else "(none)")
    lines.append("=" * 78)
    lines.append("Note: This report is a pre-screening aid. Final decisions require human review.")
    return "\n".join(lines)

def build_multi_financial_txt_report(documents: list, reviewer: str) -> str:
    """
    documents: list of dicts like:
      {
        "originalFilename": "...pdf",
        "result": { ... detector JSON ... },
        "comments": "...",
        "meta": { "studentId": "...", "applicantName": "...", "program": "..." }
      }
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append("INTERNATIONAL APPLICANT RATING ALGORITHM (IARA)")
    lines.append("Financial Document Fraud Screening Report (Combined)")
    lines.append("=" * 78)
    lines.append(f"Generated: {now}")
    lines.append(f"Reviewer: {reviewer}")
    lines.append(f"Documents Included: {len(documents)}")
    lines.append("=" * 78)
    lines.append("")

    for idx, doc in enumerate(documents, start=1):
        result = doc.get("result") or {}
        comments = doc.get("comments") or ""
        meta = doc.get("meta") or {}
        original_filename = doc.get("originalFilename") or "(unknown file)"

        # Section header for each PDF
        lines.append("#" * 78)
        lines.append(f"DOCUMENT {idx}: {original_filename}")
        if meta.get("studentId"):
            lines.append(f"Student ID: {meta.get('studentId')}")
        if meta.get("applicantName"):
            lines.append(f"Applicant Name: {meta.get('applicantName')}")
        if meta.get("program"):
            lines.append(f"Program/Term: {meta.get('program')}")
        lines.append("#" * 78)

        # Reuse your single-doc builder (keeps style consistent)
        lines.append(build_financial_txt_report(
            result=result,
            reviewer=reviewer,
            comments=comments,
            meta=meta,
            include_header=False
        ))

        lines.append("")  # spacer

    return "\n".join(lines)


def build_transcript_txt_report(
    result: dict,
    reviewer: str,
    comments: str = "",
    meta: Optional[Dict] = None,
    include_header: bool = True
) -> str:
    meta = meta or {}
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    file_path = result.get("file_path", "")
    doc_sev = result.get("doc_severity", "UNKNOWN")
    pages = result.get("pages", []) or []

    def safe(s: str, width: int = 240) -> str:
        return shorten((s or "").strip().replace("\n", " "), width=width, placeholder="...")

    lines = []

    if include_header:
        lines.append("INTERNATIONAL APPLICANT RATING ALGORITHM (IARA)")
        lines.append("Transcript Fraud Pre-Screening Report")
        lines.append("=" * 78)
        lines.append(f"Generated: {now}")
        lines.append(f"Reviewer: {reviewer}")

        if meta.get("studentId"):
            lines.append(f"Student ID: {meta.get('studentId')}")
        if meta.get("applicantName"):
            lines.append(f"Applicant Name: {meta.get('applicantName')}")
        if meta.get("program"):
            lines.append(f"Program/Term: {meta.get('program')}")
        lines.append("-" * 78)

    lines.append(f"Document: {os.path.basename(file_path) if file_path else '(uploaded file)'}")
    lines.append(f"Document Severity: {doc_sev}")
    lines.append(f"Pages Analyzed: {len(pages)}")
    lines.append("-" * 78)

    # Only include fraud-positive signals in justification section (defensible)
    for p in pages:
        pnum = p.get("page_number")
        sev = p.get("severity", "UNKNOWN")
        conf = float(p.get("confidence", 0.0) or 0.0)

        summary = safe(p.get("ai_summary", ""), width=260)
        lines.append(f"Page {pnum}: Severity={sev}  Confidence={conf:.2f}")
        if summary:
            lines.append(f"  Summary: {summary}")

        sigs = p.get("fraud_signals", []) or []
        fraud_pos = [
            s for s in sigs
            if isinstance(s, dict) and (s.get("polarity") == "fraud_positive")
        ]

        if fraud_pos:
            lines.append("  Fraud-Positive Signals:")
            for s in fraud_pos[:12]:
                signal = safe(s.get("signal", ""), width=260)
                cat = s.get("category", "other")
                src = s.get("source", "llm")
                sc = float(s.get("confidence", 0.0) or 0.0)
                lines.append(f"    - [{cat}] ({src}, {sc:.2f}) {signal}")
        else:
            lines.append("  Fraud-Positive Signals: None detected on this page.")

        lines.append("")

    lines.append("-" * 78)
    lines.append("Reviewer Comments (for applicant file):")
    lines.append((comments or "").strip() if (comments or "").strip() else "(none)")
    lines.append("=" * 78)
    lines.append("Note: This report is a pre-screening aid. Final decisions require human review.")
    return "\n".join(lines)


def build_multi_transcript_txt_report(documents: list, reviewer: str) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append("INTERNATIONAL APPLICANT RATING ALGORITHM (IARA)")
    lines.append("Transcript Fraud Pre-Screening Report (Combined)")
    lines.append("=" * 78)
    lines.append(f"Generated: {now}")
    lines.append(f"Reviewer: {reviewer}")
    lines.append(f"Documents Included: {len(documents)}")
    lines.append("=" * 78)
    lines.append("")

    for idx, doc in enumerate(documents, start=1):
        result = doc.get("result") or {}
        comments = doc.get("comments") or ""
        meta = doc.get("meta") or {}
        original_filename = doc.get("originalFilename") or "(unknown file)"

        lines.append("#" * 78)
        lines.append(f"DOCUMENT {idx}: {original_filename}")
        if meta.get("studentId"):
            lines.append(f"Student ID: {meta.get('studentId')}")
        if meta.get("applicantName"):
            lines.append(f"Applicant Name: {meta.get('applicantName')}")
        if meta.get("program"):
            lines.append(f"Program/Term: {meta.get('program')}")
        lines.append("#" * 78)

        lines.append(build_transcript_txt_report(
            result=result,
            reviewer=reviewer,
            comments=comments,
            meta=meta,
            include_header=False
        ))

        lines.append("")

    return "\n".join(lines)



def build_pdf_bytes_from_text(report_text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    x = 0.75 * inch
    y = height - 0.75 * inch

    c.setFont("Courier", 10)

    for line in report_text.splitlines():
        if y < 0.75 * inch:
            c.showPage()
            c.setFont("Courier", 10)
            y = height - 0.75 * inch
        # prevent running off the page horizontally
        c.drawString(x, y, (line or "")[:120])
        y -= 12

    c.save()
    buf.seek(0)
    return buf.getvalue()


@app.route('/api/sample/<sample_type>')
def get_sample(sample_type):
    """Get sample student data"""
    if sample_type in SAMPLE_DATA:
        return jsonify(SAMPLE_DATA[sample_type])
    return jsonify({'error': 'Invalid sample type'}), 404

@app.route('/api/fraud/financial', methods=['POST'])
@login_required
def analyze_financial_document():
    """
    Upload a financial PDF and run the fraud detector.

    Expects a multipart/form-data request with:
      - field name: 'file'
      - value: the uploaded PDF
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']

        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # For this internal prototype, we can save to a temporary folder
        upload_dir = os.path.join(os.path.dirname(__file__), 'tmp_uploads')
        os.makedirs(upload_dir, exist_ok=True)

        safe_name = secure_filename(file.filename or "uploaded_financial.pdf")
        unique = uuid.uuid4().hex
        temp_path = os.path.join(upload_dir, f"{unique}_{safe_name}")

        file.save(temp_path)

        try:
            # Run your detector (uses OCR + OpenAI + heuristics)
            result = analyze_financial_pdf(temp_path, max_pages=20)

            # Return JSON that the frontend can render
            return jsonify(result)
        finally:
            # Best-effort clean-up of the temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/fraud/financial/report', methods=['POST'])
@login_required
def export_financial_fraud_report():
    """
    Generate an exportable report artifact for the applicant file.

    Expects JSON body:
      {
        "result": <the JSON returned by /api/fraud/financial>,
        "comments": "...",
        "meta": { "studentId": "...", "applicantName": "...", "program": "..." },
        "format": "pdf" | "txt",
        "originalFilename": "optional.pdf"
      }
    """
    try:
        payload = request.json or {}
        result = payload.get("result")

        if not isinstance(result, dict):
            return jsonify({"error": "Missing or invalid 'result' object"}), 400

        comments = payload.get("comments", "") or ""
        meta = payload.get("meta", {}) or {}
        fmt = (payload.get("format", "pdf") or "pdf").lower()
        original_filename = payload.get("originalFilename", "") or ""

        reviewer = session.get("username", "unknown")
        report_text = build_financial_txt_report(
            result=result,
            reviewer=reviewer,
            comments=comments,
            meta=meta
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = meta.get("studentId") or os.path.splitext(original_filename)[0] or "financial_fraud"
        safe_base = re.sub(r"[^A-Za-z0-9_\-]+", "_", base).strip("_") or "financial_fraud"

        if fmt == "txt":
            return send_file(
                io.BytesIO(report_text.encode("utf-8")),
                mimetype="text/plain",
                as_attachment=True,
                download_name=f"{safe_base}_report_{ts}.txt"
            )

        # default PDF
        pdf_bytes = build_pdf_bytes_from_text(report_text)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{safe_base}_report_{ts}.pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/fraud/financial/report/all', methods=['POST'])
@login_required
def export_financial_fraud_report_all():
    """
    Export ONE combined report for multiple analyzed documents.

    Expects JSON body:
      {
        "documents": [
          {
            "originalFilename": "file1.pdf",
            "result": <JSON returned by /api/fraud/financial>,
            "comments": "...",
            "meta": { "studentId": "...", "applicantName": "...", "program": "..." }
          },
          ...
        ],
        "format": "pdf" | "txt",
        "meta": { "studentId": "...", "applicantName": "...", "program": "..." }  // optional global
      }
    """
    try:
        payload = request.json or {}
        documents = payload.get("documents", [])
        fmt = (payload.get("format", "pdf") or "pdf").lower()

        if not isinstance(documents, list) or len(documents) == 0:
            return jsonify({"error": "No documents provided for export."}), 400

        # Validate each doc has a result dict
        cleaned = []
        for d in documents:
            if not isinstance(d, dict):
                continue
            r = d.get("result")
            if not isinstance(r, dict):
                continue
            cleaned.append({
                "originalFilename": d.get("originalFilename", "") or "",
                "result": r,
                "comments": d.get("comments", "") or "",
                "meta": d.get("meta", {}) or {}
            })

        if len(cleaned) == 0:
            return jsonify({"error": "No valid document results found for export."}), 400

        reviewer = session.get("username", "unknown")
        report_text = build_multi_financial_txt_report(cleaned, reviewer=reviewer)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = "financial_fraud_combined"

        if fmt == "txt":
            return send_file(
                io.BytesIO(report_text.encode("utf-8")),
                mimetype="text/plain",
                as_attachment=True,
                download_name=f"{safe_base}_{ts}.txt"
            )

        pdf_bytes = build_pdf_bytes_from_text(report_text)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{safe_base}_{ts}.pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/fraud/transcript', methods=['POST'])
@login_required
def analyze_transcript_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']

        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        upload_dir = os.path.join(os.path.dirname(__file__), 'tmp_uploads')
        os.makedirs(upload_dir, exist_ok=True)

        safe_name = secure_filename(file.filename or "uploaded_transcript.pdf")
        unique = uuid.uuid4().hex
        temp_path = os.path.join(upload_dir, f"{unique}_{safe_name}")
        file.save(temp_path)

        try:
            # Uses your TranscriptFraudDetector under the hood
            result = analyze_transcript_pdf(temp_path, max_pages=20)
            return jsonify(result)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/fraud/transcript/report', methods=['POST'])
@login_required
def export_transcript_fraud_report():
    """
    Generate an exportable report artifact for the applicant file.

    Expects JSON body:
      {
        "result": <the JSON returned by /api/fraud/transcript>,
        "comments": "...",
        "meta": { "studentId": "...", "applicantName": "...", "program": "..." },
        "format": "pdf" | "txt",
        "originalFilename": "optional.pdf"
      }
    """
    try:
        payload = request.json or {}
        result = payload.get("result")

        if not isinstance(result, dict):
            return jsonify({"error": "Missing or invalid 'result' object"}), 400

        comments = payload.get("comments", "") or ""
        meta = payload.get("meta", {}) or {}
        fmt = (payload.get("format", "pdf") or "pdf").lower()
        original_filename = payload.get("originalFilename", "") or ""

        reviewer = session.get("username", "unknown")
        report_text = build_transcript_txt_report(
            result=result,
            reviewer=reviewer,
            comments=comments,
            meta=meta
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = meta.get("studentId") or os.path.splitext(original_filename)[0] or "transcript_fraud"
        safe_base = re.sub(r"[^A-Za-z0-9_\-]+", "_", base).strip("_") or "transcript_fraud"

        if fmt == "txt":
            return send_file(
                io.BytesIO(report_text.encode("utf-8")),
                mimetype="text/plain",
                as_attachment=True,
                download_name=f"{safe_base}_report_{ts}.txt"
            )

        pdf_bytes = build_pdf_bytes_from_text(report_text)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{safe_base}_report_{ts}.pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/fraud/transcript/report/all', methods=['POST'])
@login_required
def export_transcript_fraud_report_all():
    """
    Export ONE combined report for multiple analyzed transcripts.

    Expects JSON body:
      {
        "documents": [
          {
            "originalFilename": "file1.pdf",
            "result": <JSON returned by /api/fraud/transcript>,
            "comments": "...",
            "meta": { "studentId": "...", "applicantName": "...", "program": "..." }
          },
          ...
        ],
        "format": "pdf" | "txt"
      }
    """
    try:
        payload = request.json or {}
        documents = payload.get("documents", [])
        fmt = (payload.get("format", "pdf") or "pdf").lower()

        if not isinstance(documents, list) or len(documents) == 0:
            return jsonify({"error": "No documents provided for export."}), 400

        cleaned = []
        for d in documents:
            if not isinstance(d, dict):
                continue
            r = d.get("result")
            if not isinstance(r, dict):
                continue
            cleaned.append({
                "originalFilename": d.get("originalFilename", "") or "",
                "result": r,
                "comments": d.get("comments", "") or "",
                "meta": d.get("meta", {}) or {}
            })

        if len(cleaned) == 0:
            return jsonify({"error": "No valid document results found for export."}), 400

        reviewer = session.get("username", "unknown")
        report_text = build_multi_transcript_txt_report(cleaned, reviewer=reviewer)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = "transcript_fraud_combined"

        if fmt == "txt":
            return send_file(
                io.BytesIO(report_text.encode("utf-8")),
                mimetype="text/plain",
                as_attachment=True,
                download_name=f"{safe_base}_{ts}.txt"
            )

        pdf_bytes = build_pdf_bytes_from_text(report_text)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{safe_base}_{ts}.pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== STARTUP ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎓 WSU INTERNATIONAL STUDENT SCORING SYSTEM")
    print("="*60)
    print("✓ Server starting...")
    print("✓ Using in-memory storage (data resets on restart)")
    print("="*60)
    print("📍 Access at: http://127.0.0.1:5000")
    print("🔐 Login: username='admin' | password='admin123'")
    print("="*60)
    print("\n✅ ALL FEATURES AVAILABLE:")
    print("  • Single student analysis with essay rubric")
    print("  • Batch processing (CSV upload)")
    print("  • Analytics dashboard")
    print("  • Export to CSV/Excel")
    print("  • Sample data for testing")
    print("\n⚠  Note: Data stored in memory - will reset on app restart")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)