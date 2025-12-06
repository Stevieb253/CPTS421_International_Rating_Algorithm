# International Applicant Rating Algorithm 

## Project summary


A data-driven system to predict which international applicants are most likely to enroll at Washington State University.

The International Student Scoring Algorithm is designed to assist the Office of International Programs at WSU in evaluating applicants using real data. It generates both category-specific and overall scores for applicants, flags suspicious documents for manual review, and allows automated output of applicant scores in a spreadsheet format. The project also considers regional trends (South Asia, Vietnam, Kenya, and Nigeria) and provides a prototype framework that could be integrated with existing platforms like Slate.

---

## Installation

### Prerequisites
- Python 3.10 or higher  
- Git  
- Node.js 18+ (for any front-end/back-end integration)  
- MongoDB (for database management)  
- pip (Python package manager)  

### Prerequisites New Additions
- Tesseract OCR - required or doument text extraction
- Poppler or PDFPlumber - pdf parsing dependent on OS 

### Add-ons
- **Pandas** – for data manipulation and preprocessing  
- **NumPy** – for numerical computations  
- **Matplotlib** – for basic data visualization  
- **Jupyter Notebook** – for testing and interactive development  
- **Flask or FastAPI** (future integration) – for backend API if moving beyond prototype  
- **MongoDB Compass** – optional GUI for database inspection  

### ADD-ons Sprint 2 Additions
- **Pillow** - for image processing required for ELA and copy-move detection
- **OpenCV** - for fraud patten and image validation
- **PyMuPDF or pdf2image** - for rendering PDFs into analyzable images

### Installation Steps
1. Clone the repository:  
```bash
git clone https://github.com/Stevieb253/CPTS421_International_Rating_Algorithm.git
cd CPTS421_International_Rating_Algorithm
```
2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Install required Python packages:
```bash
pip install -r requirements.txt
```
4. Set up MongoDB (if using for prototype storage):

- Start the MongoDB server locally
- Create a database called student_scoring
- Use sample or client-provided anonymized datasets to seed data

5. Load test or real data:

- Place CSV/JSON datasets in the data/ folder
- Ensure filenames match the configuration in config.py or your data loader script

6. Run the prototype scoring algorithm:
```bash
python main.py
```
## Functionality

- **Applicant Scoring:** Calculates category-specific and overall scores.
- **Document Handling:** Supports multiple document types (transcripts, financials, essays).
- **Manual Review:** Flags documents with potential fraud indicators.
- **Spreadsheet Output:** Generates CSV output for client review.
- **Prototype Integration:** Designed as a standalone system with potential future integration into Slate.

## Usage Walkthrough:

1. Place applicant data in the data/ folder.
2. Run main.py to process applications.
3. Review generated scores and flagged documents.
4. Export results to CSV for reporting or client review.

## Usage Walkthrough (fraud systems)
1. Have applicant data in data folder
2. Run programs
python financial_fraud_detector.py "data/financial/authentic" "data/financial/fraudulent" --pages 10
python transcript_fraud_detector.py "data/transcript/authentic" "data/transcript/fraudulent" --pages 5
3. Review structured results in respective JSON files
4. Evaluate with eval_results.py

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## License

MIT License

Copyright (c) [2025] [International Scoring system]