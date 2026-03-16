from services.transcript_fraud_detector import TranscriptFraudDetector
from unittest.mock import patch, MagicMock
from services.transcript_fraud_detector import TranscriptFraudDetector, analyze_transcript_pdf


# Test that _extract_features correctly identifies normal transcript information
# such as transcript keywords, registrar mention, course codes, GPA, and term words.
def test_extract_features_basic():
    detector = TranscriptFraudDetector()

    ocr_text = """
    Official Transcript
    Office of the Registrar
    Student Name: John Doe

    Fall 2024
    CSE 101 Introduction to Programming   A
    MATH201 Calculus I                    B+

    GPA: 3.75
    """

    features = detector._extract_features(ocr_text)

    assert features["has_word_transcript"] is True
    assert features["has_registrar"] is True
    assert features["has_official"] is True
    assert features["has_unofficial"] is False

    assert features["course_code_count"] >= 2
    assert "3.75" in features["gpa_values"]

    assert features["has_term_words"] is True
    assert features["char_count"] > 0
    assert features["line_count"] > 1

# Test that _extract_features correctly identifies suspicious phrases
# that may indicate a sample, template, or non-official document.
def test_extract_features_suspicious_phrases():
    detector = TranscriptFraudDetector()
    ocr_text = """
    SAMPLE TRANSCRIPT
    This is a template document.
    Not an official document.
    """
    features = detector._extract_features(ocr_text)
    assert "sample" in features["suspicious_phrases_found"]
    assert "template" in features["suspicious_phrases_found"]
    assert "not an official document" in features["suspicious_phrases_found"]

# Test that _looks_like_transcript correctly identifies text that
# contains keywords typical of a real academic transcript.
def test_looks_like_transcript():
    detector = TranscriptFraudDetector()

    # Should return True for transcript-like text
    assert detector._looks_like_transcript("GPA: 3.5 fall semester") is True
    assert detector._looks_like_transcript("official transcript registrar") is True
    assert detector._looks_like_transcript("credit hours course completed") is True

    # Should return False for unrelated text
    assert detector._looks_like_transcript("Invoice Total $500") is False
    assert detector._looks_like_transcript("") is False

# Test that _scrub_text removes file paths from OCR text
# before it gets sent to the AI model.
def test_scrub_text():
    # Windows-style path: the regex removes the path segment but not the filename
    # This is the current behavior of _scrub_text
    windows_text = r"Student name C:\\Users\\admin\\docs\\transcript.pdf found"
    scrubbed = TranscriptFraudDetector._scrub_text(windows_text)
    assert "Users" not in scrubbed  # path is stripped

    # Unix-style path should be removed
    unix_text = "File located at /home/user/uploads/transcript.pdf done"
    scrubbed = TranscriptFraudDetector._scrub_text(unix_text)
    assert "/home/user" not in scrubbed

    # Normal text should be left alone
    normal_text = "Jane Doe GPA 3.9 CSE 101"
    assert TranscriptFraudDetector._scrub_text(normal_text) == normal_text

# Test that _extract_features correctly detects course codes
# like CSE 101 and MATH 201 from transcript text.
def test_extract_features_course_codes():
    detector = TranscriptFraudDetector()
    ocr_text = """
    CSE 101   Introduction to Computing    A   3 cr
    MATH 201  Calculus I                   B+  4 cr
    ENG 350   Technical Writing            A-  3 cr
    """
    features = detector._extract_features(ocr_text)

    assert features["course_code_count"] == 3
    assert "CSE 101" in features["distinct_course_codes"]
    assert "MATH 201" in features["distinct_course_codes"]
    assert "ENG 350" in features["distinct_course_codes"]

# Test that _extract_features correctly detects GPA values
# from transcript text.
def test_extract_features_gpa_detection():
    detector = TranscriptFraudDetector()
    ocr_text = """
    Fall 2021
    GPA: 3.72

    Spring 2022
    GPA: 3.60

    Cumulative GPA: 3.66
    """
    features = detector._extract_features(ocr_text)

    assert len(features["gpa_values"]) >= 2
    assert "3.72" in features["gpa_values"]
    assert "3.60" in features["gpa_values"]
    for gpa in features["gpa_values"]:
        assert 0.0 <= float(gpa) <= 4.0

# Test that analyze_transcript_pdf returns a properly structured
# dictionary with all the expected keys.
def test_analyze_transcript_pdf_structure():
    with patch("services.transcript_fraud_detector.TranscriptFraudDetector") as MockDet:
        fake_result = MagicMock()
        fake_result.page_number = 1
        fake_result.severity = "LOW"
        fake_result.confidence = 0.1
        fake_result.ocr_text = "preview text"
        fake_result.ai_summary = "Looks clean."
        fake_result.fraud_signals = []
        fake_result.extra = {}

        instance = MockDet.return_value
        instance.analyze_document.return_value = [fake_result]

        result = analyze_transcript_pdf("fake.pdf", max_pages=1)

    assert "file_path" in result
    assert "doc_severity" in result
    assert "pages" in result
    assert len(result["pages"]) == 1
    assert result["file_path"] == "fake.pdf"

# Test that the severity bump rule correctly upgrades LOW to MEDIUM
# when the page looks like a real transcript and confidence is high enough.
def test_looks_like_transcript_bumps_low_to_medium():
    detector = TranscriptFraudDetector(max_pages=1, escalate=False)

    real_transcript_ocr = """
    Official Transcript
    Office of the Registrar
    GPA: 3.5
    Fall semester CSE 101
    """

    with patch.object(detector, "_analyze_with_openai", return_value={
        "summary": "Looks clean.",
        "raw_severity": "LOW",
        "severity": "LOW",
        "overall_confidence": 0.7,
        "fraud_signals": []
    }):
        with patch("services.financial_fraud_detector.extract_pdf_images",
                   return_value=[MagicMock()]):
            with patch("services.financial_fraud_detector.extract_ocr_text",
                       return_value=real_transcript_ocr):
                with patch("services.financial_fraud_detector.finalize_label",
                           side_effect=lambda x: {**x, "severity": x.get("raw_severity", "LOW")}):
                    with patch("services.financial_fraud_detector.ela_score", return_value=0.05):
                        with patch("services.financial_fraud_detector.copy_move_score", return_value=0.03):
                            results = detector.analyze_document("fake.pdf")

    assert results[0].severity == "MEDIUM"

# Test that _sum_line_items_vs_total correctly detects when a stated
# total does not match the sum of line items, which is a fraud indicator.
def test_sum_line_items_vs_total():
    from services.financial_fraud_detector import _sum_line_items_vs_total

    # Mismatched total should return fraud_positive
    mismatched = """
    debit  100.00
    debit  200.00
    credit 50.00
    Total: 9999.00
    """
    result = _sum_line_items_vs_total(mismatched)
    assert result is not None
    assert result["polarity"] == "fraud_positive"

    # No total line should return None
    no_total = "Just some random text with no balance or total"
    result = _sum_line_items_vs_total(no_total)
    assert result is None

# Test that _extract_features handles empty string input safely
# without crashing and returns correct default values.
def test_extract_features_empty_input():
    detector = TranscriptFraudDetector()
    features = detector._extract_features("")

    assert features["has_word_transcript"] is False
    assert features["has_registrar"] is False
    assert features["course_code_count"] == 0
    assert features["gpa_values"] == []
    assert features["suspicious_phrases_found"] == []
    assert features["char_count"] == 0

# Test that escalation calls gpt-5 when there are suspicious phrases
# and stays with the higher severity result.
def test_escalation_calls_gpt5_when_suspicious():
    detector = TranscriptFraudDetector(max_pages=1, escalate=True)

    call_log = []

    def fake_analyze(img, ocr, features, model="gpt-5-mini"):
        call_log.append(model)
        if model == "gpt-5-mini":
            return {"summary": "mini", "raw_severity": "LOW", "severity": "LOW",
                    "overall_confidence": 0.2, "fraud_signals": []}
        return {"summary": "escalated", "raw_severity": "MEDIUM", "severity": "MEDIUM",
                "overall_confidence": 0.6, "fraud_signals": []}

    with patch.object(detector, "_analyze_with_openai", side_effect=fake_analyze):
        with patch("services.financial_fraud_detector.extract_pdf_images",
                   return_value=[MagicMock()]):
            with patch("services.financial_fraud_detector.extract_ocr_text",
                       return_value="void template sample"):
                with patch("services.financial_fraud_detector.finalize_label",
                           side_effect=lambda x: {**x, "severity": x.get("raw_severity", "LOW")}):
                    with patch("services.financial_fraud_detector.ela_score", return_value=0.05):
                        with patch("services.financial_fraud_detector.copy_move_score", return_value=0.03):
                            results = detector.analyze_document("fake.pdf")

    assert "gpt-5" in call_log
    assert results[0].severity == "MEDIUM"