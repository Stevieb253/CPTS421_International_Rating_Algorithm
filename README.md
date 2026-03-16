# Team 25 – Washington State University

**Project Lead:** Khushi Panchal  
**Team Member:** Steven Bennett  
**Client:** Trevor Kingsley  
**Mentor:** Parteek Kumar  

---

## Project Overview

The International Student Scoring System (IARA) supports WSU's admissions office by scoring international applicants accurately and efficiently.

**Sprint 5** focuses on backend stabilization, a full WSU-branded UI redesign, admin user management, and fraud detection improvements. The system now generates reports end-to-end, displays live analytics, and is being prepared for deployment on Render.

---

## Demo Video

**Sprint 5 Demo** – Direct link: *[Youtube.com](https://youtu.be/Ap_DkOtFV-E)*

---

## Sprint 5 Features

- **Report Generation Fixed** – PDF and plain text reports now generate end-to-end with scores, essay analysis, staff comments, and recommendation.
- **WSU Brand Redesign** – All 11 templates rebuilt using the official WSU design system (Crimson #a60f2d, Montserrat font, WSU CDN bundle).
- **Live Analytics** – Dashboard now updates after every single student analysis, not just batch.
- **Admin User Management** – Working Deactivate and Delete buttons, protected admin account, all actions logged to activity log.
- **File Upload Fix** – Financial and transcript upload zones now work correctly across all browsers.
- **Checkbox Fix** – Risk factor checkboxes now support multiple independent selections.
- **Fraud Detection Loading Bar** – Progress indicator added to both financial and transcript screening pages.
- **Fraud Detection Test Suite** – Automated tests written for both fraud detection systems.
- **Fraud Detection PDF Fix** – Text overflow in exported fraud detection reports resolved.
- **Database Auto-Migration** – Schema migrates automatically on startup, zero manual intervention required.
- **New Admin Routes** – `POST /api/admin/toggle-user` and `POST /api/admin/delete-user` added with role-based access control.

---

## Installation

### Prerequisites

- Python 3.11+
- Git
- pip

### Additional Requirements

- Flask, ReportLab, OpenPyXL
- Transformers, Torch, Sentence-Transformers
- Tesseract OCR, Poppler / PDFPlumber
- Pandas, NumPy, Pillow, OpenCV, PyMuPDF

### Setup
```bash
git clone https://github.com/Stevieb253/CPTS421_International_Rating_Algorithm.git
cd CPTS421_International_Rating_Algorithm
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Set your HuggingFace token (recommended):
```bash
export HF_TOKEN=your_token_here        # Linux/Mac
set HF_TOKEN=your_token_here           # Windows
```

Run the app:
```bash
cd code
python app.py
```

Access at `http://localhost:5000`  
Default login: **admin / admin123**

---

## Usage

### Single Applicant Analysis
- Log in and navigate to the home page
- Fill in student profile or load a sample (High / Medium / High Risk)
- Click **Analyze Student** to generate scores and essay analysis
- Add staff comments and download PDF or plain text report

### Batch Analysis
- Navigate to **Batch Process**
- Upload a CSV file with columns: `studentId, country, gpa, curriculum, travelHistory, essayText, negFactors`
- Download the template from the page if needed
- Review results and export as CSV or Excel

### Fraud Screening
- Navigate to **Financial Docs** or **Transcripts**
- Upload one or more PDFs
- Review per-page risk levels (LOW / MEDIUM / HIGH), OCR preview, and fraud signals
- Add reviewer notes and export a combined PDF or TXT report

### Admin
- Log in as admin and navigate to `/admin/users`
- Create new staff accounts with Reviewer or Admin role
- Deactivate or delete existing accounts

---

## Project Structure
```
code/
├── app.py                  # Main Flask application
├── db/
│   ├── database.py         # SQLite database layer
│   ├── report_generator.py # PDF report generation
│   └── student_scoring.db  # SQLite database file
├── services/
│   ├── student_analyzer.py       # Core scoring engine
│   ├── nlp_service.py            # Sentiment & similarity
│   ├── financial_fraud_detector.py
│   └── transcript_fraud_detector.py
└── templates/              # All 11 HTML templates
```

---

## Contribution
```bash
git checkout -b my-new-feature
git commit -am 'Add feature'
git push origin my-new-feature
```
Submit a pull request.