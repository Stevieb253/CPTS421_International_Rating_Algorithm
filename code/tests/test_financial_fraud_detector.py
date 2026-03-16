import pytest

from services.financial_fraud_detector import (
    finalize_label,
    doc_severity,
    FraudResult,
    build_programmatic_signals,
    _parse_numbers,
    _sum_line_items_vs_total,
    _date_window_consistency,
    _bank_identity_consistency,
    analyze_financial_pdf,
)
from unittest.mock import patch, MagicMock

# Test that finalize_label returns LOW when there are no fraud_positive signals
def test_finalize_label_no_signals_returns_low():
    ai_json = {
        "summary": "Looks clean.",
        "raw_severity": "LOW",
        "overall_confidence": 0.8,
        "fraud_signals": []
    }
    result = finalize_label(ai_json)
    assert result["severity"] == "LOW"

# Test that finalize_label returns HIGH when there are strong fraud_positive signals
def test_finalize_label_strong_signals_returns_high():
    ai_json = {
        "summary": "Strong evidence of tampering.",
        "raw_severity": "HIGH",
        "overall_confidence": 0.9,
        "fraud_signals": [
            {"signal": "Altered total", "category": "content_conflict",
             "polarity": "fraud_positive", "confidence": 0.9, "source": "programmatic"},
            {"signal": "Cloned region detected", "category": "visual",
             "polarity": "fraud_positive", "confidence": 0.85, "source": "programmatic"},
        ]
    }
    result = finalize_label(ai_json)
    assert result["severity"] == "HIGH"

# Test that doc_severity correctly collapses page-level severities
# to the highest severity found across all pages.
def test_doc_severity():
    results = [
        FraudResult(file_path="test.pdf", page_number=1, ocr_text="", ai_summary="",
                    fraud_signals=[], severity="LOW", confidence=0.1, extra={}),
        FraudResult(file_path="test.pdf", page_number=2, ocr_text="", ai_summary="",
                    fraud_signals=[], severity="HIGH", confidence=0.9, extra={}),
        FraudResult(file_path="test.pdf", page_number=3, ocr_text="", ai_summary="",
                    fraud_signals=[], severity="MEDIUM", confidence=0.5, extra={}),
    ]
    assert doc_severity(results) == "HIGH"

# Test that _parse_numbers correctly extracts numbers from text
# including US-style and EU-style formatted numbers.
def test_parse_numbers():
    # Standard integers and decimals
    result = _parse_numbers("Total: 1234.56 and 789")
    assert 1234.56 in result
    assert 789.0 in result

    # US-style with comma separator
    result = _parse_numbers("Balance: 1,234.56")
    assert 1234.56 in result

    # Empty string returns empty list
    result = _parse_numbers("")
    assert result == []

# Test that _bank_identity_consistency correctly detects multiple
# bank brands on the same page, which is a fraud indicator.
def test_bank_identity_consistency():
    # Two bank brands on the same page should trigger a fraud_positive signal
    text_two_banks = "Account held at Bank of China. Previous transfer from GTBank."
    signals = _bank_identity_consistency(text_two_banks)
    polarities = [s["polarity"] for s in signals]
    assert "fraud_positive" in polarities

    # Single bank brand should not trigger a fraud signal
    text_one_bank = "Account held at Bank of China."
    signals = _bank_identity_consistency(text_one_bank)
    fraud_pos = [s for s in signals if s["polarity"] == "fraud_positive"]
    assert len(fraud_pos) == 0

    # CamScanner watermark should trigger a fraud_positive signal
    text_camscanner = "Scanned by CamScanner"
    signals = _bank_identity_consistency(text_camscanner)
    polarities = [s["polarity"] for s in signals]
    assert "fraud_positive" in polarities

# Test that _date_window_consistency correctly flags pages where
# multiple distinct years appear against a single period header.
def test_date_window_consistency():
    # Multiple distinct years should trigger a fraud_positive signal
    text_multi_year = """
    Statement Period: January 2021
    01/01/2021 Opening Balance 1000.00
    15/03/2022 Transfer 500.00
    20/07/2023 Closing Balance 1500.00
    """
    result = _date_window_consistency(text_multi_year)
    assert result is not None
    assert result["polarity"] == "fraud_positive"

    # No dates at all should return None
    result = _date_window_consistency("No dates here at all.")
    assert result is None

    # Single year should not trigger fraud_positive
    text_single_year = """
    Statement Period: 2022
    01/01/2022 Opening Balance 1000.00
    15/03/2022 Transfer 500.00
    """
    result = _date_window_consistency(text_single_year)
    assert result is not None
    assert result["polarity"] != "fraud_positive"

# Test that analyze_financial_pdf returns a properly structured
# dictionary with all the expected keys.
def test_analyze_financial_pdf_structure():
    with patch("services.financial_fraud_detector.analyze_document") as mock_analyze:
        fake_result = FraudResult(
            file_path="fake.pdf",
            page_number=1,
            ocr_text="preview text",
            ai_summary="Looks clean.",
            fraud_signals=[],
            severity="LOW",
            confidence=0.1,
            extra={"ela": 0.0, "copy_move": 0.0, "escalated": False}
        )
        mock_analyze.return_value = [fake_result]

        result = analyze_financial_pdf("fake.pdf", max_pages=1)

    assert "file_path" in result
    assert "doc_severity" in result
    assert "pages" in result
    assert len(result["pages"]) == 1
    assert result["file_path"] == "fake.pdf"

# Test that build_programmatic_signals returns a list of signals
# and that each signal has the required keys.
def test_build_programmatic_signals_structure():
    # Text with a CamScanner watermark should produce at least one signal
    text = """
    Statement Period: January 2022
    Scanned by CamScanner
    Total: 1000.00
    """
    signals = build_programmatic_signals(text)

    # Should return a list
    assert isinstance(signals, list)

    # Every signal must have the required keys
    for signal in signals:
        assert "signal" in signal
        assert "category" in signal
        assert "polarity" in signal
        assert "confidence" in signal
        assert "source" in signal
        assert signal["source"] == "programmatic"
        assert signal["polarity"] in ("fraud_positive", "benign", "inconclusive")

# Test that is_benign_signal correctly identifies benign scanning
# artifacts that should not be treated as fraud indicators.
def test_is_benign_signal():
    from services.financial_fraud_detector import is_benign_signal

    # These should all be recognized as benign
    assert is_benign_signal("standard redaction applied") is True
    assert is_benign_signal("no clear paste/clone detected") is True
    assert is_benign_signal("scan artifact in corner") is True
    assert is_benign_signal("scanning artifact found") is True

    # These should not be benign
    assert is_benign_signal("grades appear altered") is False
    assert is_benign_signal("total does not match") is False
    assert is_benign_signal("") is False

# Test that finalize_label with a single strong LLM-only signal
# stays at MEDIUM and does not get upgraded to HIGH.
def test_finalize_label_single_llm_signal_stays_medium():
    ai_json = {
        "summary": "One strong LLM signal.",
        "raw_severity": "MEDIUM",
        "overall_confidence": 0.85,
        "fraud_signals": [
            {"signal": "Suspicious region detected", "category": "visual",
             "polarity": "fraud_positive", "confidence": 0.85, "source": "llm"}
        ]
    }
    result = finalize_label(ai_json)
    assert result["severity"] == "MEDIUM"

# Test that with_retries retries on failure and eventually raises
# if all attempts fail.
def test_with_retries_raises_after_max_attempts():
    from services.financial_fraud_detector import with_retries

    call_count = [0]

    def always_fails():
        call_count[0] += 1
        raise ValueError("always fails")

    with patch("time.sleep"):  # skip actual waiting
        with pytest.raises(ValueError):
            with_retries(always_fails, max_tries=3, base_delay=1.5)

    assert call_count[0] == 3