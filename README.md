# Team 25 – Washington State University

**Project Lead:** Khushi Panchal  
**Team Member:** Steven Bennett  
**Client:** Trevor Kingsley  
**Mentor:** Parteek Kumar  

---

## Project Overview

The International Student Scoring System (IARA) supports WSU's admissions office by scoring international applicants accurately and efficiently.

**Sprint 6** focused on finalizing the entire system, integrating all major components, stabilizing performance, and preparing the application for stakeholder delivery. This sprint emphasized deployment, system reliability, and ensuring that IARA runs consistently in a real hosted environment.

---

## Demo Video

 **Sprint 6 Demo** Direct link: (https://youtu.be/HQUHaxc2waY)

---

## Sprint 6 Features

- **Final IARA Showcase Video** – A polished 3‑minute professional video demonstrating the full system.
- **Final Poster Presentation** – Completed and formatted for stakeholder review.
- **Stakeholder Final Report** – Comprehensive written report summarizing system capabilities and results.
- **Stakeholder Handoff Document** – Instructions and documentation for future continuation of the project.
- **Azure Deployment** – Full application deployed to Azure App Service / Azure Web App with stable performance.
- **System Integration Complete** – Scoring engine, NLP essay analysis, fraud detection, and reporting all connected end‑to‑end.
- **Report Generation Fixes** – Stability improvements for PDF and text report generation.
- **Hosted Environment Optimization** – Environment variables, dependencies, and memory configuration updated for cloud hosting.

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