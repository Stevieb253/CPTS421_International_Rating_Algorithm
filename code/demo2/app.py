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


@app.route('/api/sample/<sample_type>')
def get_sample(sample_type):
    """Get sample student data"""
    if sample_type in SAMPLE_DATA:
        return jsonify(SAMPLE_DATA[sample_type])
    return jsonify({'error': 'Invalid sample type'}), 404


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