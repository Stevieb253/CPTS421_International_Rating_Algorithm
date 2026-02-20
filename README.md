# Team 25 – Washington State University

**Project Lead:** Khushi Panchal  
**Team Member:** Steven Bennet  
**Client:** Trevor Kingsley  
**Mentor:** Parteek Kumar  

---

## Project Overview

The International Student Scoring System supports WSU’s admissions office by scoring international applicants accurately and efficiently.  

**Sprint 4** focuses on usability, data handling, and security: file restructuring, database integration, feedback textbox, redesigned analysis dashboard, and role-based login. The system is now more modular, maintainable, and multi-user ready.  

---

## Demo Video

**Sprint 4 Demo** – Direct link:  https://youtu.be/2RWjarZzjTI

---

## Sprint 4 Features

- **File Restructuring** – Organized project into `services/`, `db/`, `templates/`, `static/`.  
- **Database Integration** – MongoDB backend to store analyses, feedback, and logs.  
- **Feedback Textbox** – Users can submit essay feedback, stored in DB.  
- **Redesigned Dashboard** – Improved UI for scores, essay analysis, and recommendations.  
- **Role-Based Login** – Admin, reviewer, and guest roles with proper access control.  
- **API Improvements** – Robust `/api/analyze` and `/api/batch-analyze` handling with safe JSON/dict conversions.  
- **Scalable Prototype** – Backend ready for batch processing and potential integration with Slate.
- **Transcript & Financial Templates** – Added structured transcript and financial evaluation views.
- **Export Formatting** – Implemented clean report layout for administrative review.
- **Application Refactor** – Integrated new features into updated project structure.

---

## Installation

### Prerequisites

- Python 3.11+  
- Git  
- Node.js 18+  
- MongoDB  
- pip  

### Additional Requirements

- Tesseract OCR, Poppler / PDFPlumber  
- Pandas, NumPy, Matplotlib  
- Flask / FastAPI  
- Pillow, OpenCV, PyMuPDF / pdf2image  
- Flask-Login, WTForms (Sprint 4 additions)  

### Setup

```bash
git clone https://github.com/Stevieb253/CPTS421_International_Rating_Algorithm.git
cd CPTS421_International_Rating_Algorithm
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
Start MongoDB and create student_scoring database

Seed with anonymized datasets in data/ folder

Run the app:
```bash
python app.py
```

### Usage
#### Single Applicant

- Place applicant data in data/

- Access /api/analyze via dashboard or Postman

- Review scores, essay analysis, and recommendations

- Leave feedback in the textbox

#### Batch Analysis
- Submit JSON list of students to /api/batch-analyze

- Review structured results and analytics

#### Role-Based Access
- Admin: Full access

- Reviewer: Score students and leave feedback

- Guest: Limited view

### Contribution
```bash

git checkout -b my-new-feature
git commit -am 'Add feature'
git push origin my-new-feature
```
Submit a pull request.
