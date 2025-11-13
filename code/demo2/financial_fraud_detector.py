#!/usr/bin/env python3
"""
financial_fraud_detector.py

Batch-capable PDF forensic pre-screen:
- OCR pages
- Run quick image heuristics (ELA, coarse copy-move)
- Ask GPT-5-mini for structured analysis (fraud signals with polarity & confidence)
- Escalate suspicious pages to GPT-5 for second opinion
- Finalize severity using conservative rules (counts only fraud-positive signals)
- Output combined batch_results.json and colorized console summary

Author: (you)
"""

import os
import io
import sys
import json
import time
import argparse
import glob
import re
import base64
from statistics import mean
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from PIL import Image, ImageChops
import numpy as np
import cv2

# PDF -> images
from pdf2image import convert_from_path

# OCR
import pytesseract

# OpenAI client (new-style)
from openai import OpenAI

# initialize OpenAI client from env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------
@dataclass
class FraudResult:
    file_path: str
    page_number: int
    ocr_text: str
    ai_summary: str
    fraud_signals: List[Any]
    severity: str
    confidence: float
    extra: Dict[str, Any] = None

# -----------------------------------------------------------------------------
# Utilities: retries wrapper
# -----------------------------------------------------------------------------
def with_retries(fn, max_tries=3, base_delay=1.5):
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_tries:
                time.sleep(base_delay * attempt)
            else:
                raise last_err

# -----------------------------------------------------------------------------
# Image / PDF helpers
# -----------------------------------------------------------------------------
def extract_pdf_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert PDF pages to PIL images using pdf2image.
    Higher DPI (e.g., 300) improves OCR quality, at the cost of bigger images.
    Requires poppler on PATH.
    """
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        return pages
    except Exception as e:
        raise RuntimeError(f"pdf->image conversion failed: {e}")


def extract_ocr_text(pil_img: Image.Image) -> str:
    """
    Use pytesseract to extract text with some preprocessing to improve quality:
    - convert to grayscale
    - binarize with Otsu threshold
    - light denoising
    """
    try:
        # 1) Grayscale
        gray = pil_img.convert("L")

        # 2) To numpy for OpenCV
        arr = np.array(gray)

        # 3) Otsu threshold -> clean background, enhance text
        _, thresh = cv2.threshold(
            arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4) Light denoise to remove specks
        thresh = cv2.medianBlur(thresh, 3)

        # 5) Back to PIL for pytesseract
        processed = Image.fromarray(thresh)

        # 6) Tesseract config:
        #    --oem 3 : default LSTM + legacy engine
        #    --psm 6 : assume a block of text (good for statements/certificates)
        config = r"--oem 3 --psm 6"

        txt = pytesseract.image_to_string(processed, config=config)
        return txt or ""
    except Exception as e:
        # You can log e if you want to debug OCR failures
        return ""

# -----------------------------------------------------------------------------
# Quick local heuristics (fast, cheap)
# -----------------------------------------------------------------------------
def ela_score(img_pil: Image.Image, quality: int = 90) -> float:
    """
    Error Level Analysis approximation:
    save as JPEG with chosen quality, compute mean difference.
    Return normalized 0..1 (higher = more likely manipulated heterogenous compression).
    """
    try:
        buf = io.BytesIO()
        img_pil.convert("RGB").save(buf, "JPEG", quality=quality)
        buf.seek(0)
        jpg = Image.open(buf).convert("RGB")
        orig = img_pil.convert("RGB")
        diff = ImageChops.difference(orig, jpg)
        stat = np.asarray(diff).astype(np.float32)
        mean = float(np.mean(stat) / 255.0)
        return float(np.clip(mean, 0.0, 1.0))
    except Exception:
        return 0.0

def copy_move_score(img_pil: Image.Image) -> float:
    """
    Coarse copy-move / duplication detector using ORB features.
    Returns 0..1 score (higher => more duplicated local features).
    This is intentionally coarse and only used to escalate to a stronger model, not to finalize.
    """
    try:
        img = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=1200)
        kps, des = orb.detectAndCompute(img, None)
        if des is None or len(des) < 20:
            return 0.0
        # match descriptors to themselves; keep matches that are spatially apart
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des, des)
        spatial_far = 0
        for m in matches:
            if m.queryIdx == m.trainIdx:
                continue
            p1 = kps[m.queryIdx].pt
            p2 = kps[m.trainIdx].pt
            if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) > 30:  # crude distance threshold
                spatial_far += 1
        # normalize
        score = min(1.0, spatial_far / 800.0)
        return float(score)
    except Exception:
        return 0.0

# -----------------------------------------------------------------------------
# Postprocessing helpers: sensible conservative rules (counts only fraud-positive)
# -----------------------------------------------------------------------------
def is_benign_signal(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in [
        "standard redaction", "not treated as tampering",
        "no clear paste/clone", "no repeated pixel blocks",
        "no obvious inconsistent compression", "scan artifact", "scanning artifact"
    ])

def finalize_label(ai_json: dict) -> dict:
    """
    Final severity policy:

      - LOW  = no meaningful fraud_positive evidence.
      - MED  = some fraud_positive evidence (LLM or programmatic).
      - HIGH = strong or multiple fraud_positive signals, especially
               from programmatic checks.

    Authentic docs are allowed to be LOW or MEDIUM. HIGH should only
    occur when evidence is strong enough that an authentic doc is
    unlikely.
    """
    overall = float(ai_json.get("overall_confidence",
                                ai_json.get("confidence", 0.0) or 0.0))
    signals = ai_json.get("fraud_signals", []) or []

    # Collect only fraud_positive signals
    fraud_pos = [
        s for s in signals
        if isinstance(s, dict) and s.get("polarity") == "fraud_positive"
    ]

    # If model is totally unsure or no fraud_positive, it's LOW.
    if not fraud_pos or overall < 0.35:
        ai_json["severity"] = "LOW"
        ai_json["overall_confidence"] = overall
        ai_json["fraud_positive_counts"] = {
            "programmatic": {"strong": 0, "medium": 0},
            "llm": {"strong": 0, "medium": 0},
        }
        return ai_json

    prog_strong = prog_medium = 0
    llm_strong = llm_medium = 0

    fraud_score = 0.0

    for s in fraud_pos:
        conf = float(s.get("confidence", 0.0))
        src = s.get("source", "llm")  # default to LLM if not tagged

        if src == "programmatic":
            # Programmatic checks are more trusted
            if conf >= 0.75:
                prog_strong += 1
                fraud_score += 3.0
            elif conf >= 0.55:
                prog_medium += 1
                fraud_score += 2.0
        else:
            # LLM-only signals contribute, but are weighted a bit lower
            if conf >= 0.80:
                llm_strong += 1
                fraud_score += 2.0
            elif conf >= 0.60:
                llm_medium += 1
                fraud_score += 1.0

    # Baseline: any non-zero fraud_score -> at least MEDIUM
    if fraud_score <= 0:
        final = "LOW"
    else:
        final = "MEDIUM"

    # Consider upgrading to HIGH if evidence is strong enough
    can_be_high = False

    # Strong programmatic evidence alone is enough for HIGH
    if prog_strong >= 1:
        can_be_high = True

    # Programmatic + LLM together (even if medium) is also strong
    elif (prog_medium >= 1) and (llm_strong + llm_medium >= 1):
        can_be_high = True

    # Multiple strong LLM-only signals (no programmatic) count as strong
    elif (prog_strong + prog_medium) == 0 and llm_strong >= 2:
        can_be_high = True

    # But: do NOT allow HIGH if it's just one strong LLM-only hit
    single_llm_only = (
        (prog_strong + prog_medium) == 0 and
        llm_strong == 1 and
        llm_medium == 0
    )

    if can_be_high and not single_llm_only and overall >= 0.55 and fraud_score >= 3.0:
        final = "HIGH"

    ai_json["severity"] = final
    ai_json["overall_confidence"] = overall
    ai_json["fraud_positive_counts"] = {
        "programmatic": {"strong": prog_strong, "medium": prog_medium},
        "llm": {"strong": llm_strong, "medium": llm_medium},
    }
    return ai_json

# -------------------------------------------------------------------------
# Document-level helpers
# -------------------------------------------------------------------------
def doc_severity(page_results: List[FraudResult]) -> str:
    """
    Collapse page-level severities to a single doc-level label.
    HIGH > MEDIUM > LOW.
    """
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    max_val = -1
    final = "LOW"
    for r in page_results:
        v = order.get(r.severity, 0)
        if v > max_val:
            max_val = v
            final = r.severity
    return final

def _parse_numbers(txt: str) -> list[float]:
    # Extract numbers with , or . separators; be liberal (handles 1,234.56 and 1.234,56)
    # Return as floats best-effort.
    nums = []
    for m in re.finditer(r"(?<![A-Za-z])[-+]?\d[\d,.\s]*\d(?![A-Za-z])", txt):
        raw = m.group(0)
        cand = raw.replace(" ", "")
        # Try US-style first: 1,234.56
        try:
            nums.append(float(cand.replace(",", "")))
            continue
        except:
            pass
        # Try EU-style: 1.234,56
        try:
            if cand.count(",") == 1 and cand.rfind(",") > cand.rfind("."):
                nums.append(float(cand.replace(".", "").replace(",", ".")))
        except:
            pass
    return nums

def _sum_line_items_vs_total(txt: str) -> dict | None:
    """
    Heuristic: If a single 'Total' (or similar) appears, compare with sum of nearby amounts.
    Returns a fraud_positive or inconclusive signal dict, or None.
    """
    low = txt.lower()
    # Find candidate total lines
    total_matches = list(re.finditer(r"(total|closing\s+balance|ending\s+balance|available\s+balance).{0,40}?([-+]?\d[\d,.\s]*\d)", low, re.I))
    if not total_matches:
        return None

    # Crude “line items” collection: amounts appearing in lines containing words like debit/credit/amount
    lines = [l for l in txt.splitlines() if any(k in l.lower() for k in ["debit", "credit", "amount", "txn", "transaction", "narration", "balance"])]
    amounts = []
    for l in lines:
        amounts.extend(_parse_numbers(l))
    if len(amounts) < 3:
        # Not enough structure to test; do not assert fraud yet
        return {"signal": "Not enough line items to verify stated total", "category": "content_conflict", "polarity": "inconclusive", "confidence": 0.25}

    # Take a robust subset: ignore extremes (common OCR junk)
    core = sorted([a for a in amounts if a != 0.0])
    if len(core) >= 6:
        core = core[1:-1]

    approx_sum = sum(core)
    # Extract the first explicit total value
    m = re.search(r"(total|closing\s+balance|ending\s+balance|available\s+balance).{0,40}?([-+]?\d[\d,.\s]*\d)", txt, re.I)
    if not m:
        return {"signal": "Total not reliably found; cannot cross-check", "category": "content_conflict", "polarity": "inconclusive", "confidence": 0.2}
    total_val_candidates = _parse_numbers(m.group(0))
    if not total_val_candidates:
        return {"signal": "Total figure unreadable; cannot cross-check", "category": "content_conflict", "polarity": "inconclusive", "confidence": 0.2}
    total_val = total_val_candidates[-1]

    # Compare with tolerance (OCR noise)
    # Tolerance scales with magnitude (1% of max(|sum|, |total|)) + fixed buffer
    tol = 0.01 * max(abs(approx_sum), abs(total_val)) + 10.0
    if abs(approx_sum - total_val) > tol:
        return {
            "signal": f"Stated TOTAL ({total_val:.2f}) does not match sum of line items (~{approx_sum:.2f}) beyond tolerance ({tol:.2f})",
            "category": "content_conflict",
            "polarity": "fraud_positive",
            "confidence": 0.7
        }
    else:
        return {
            "signal": f"Stated TOTAL matches sum of line items within tolerance ({tol:.2f})",
            "category": "content_conflict",
            "polarity": "benign",
            "confidence": 0.75
        }

def _date_window_consistency(txt: str) -> dict | None:
    """
    Check that Statement Period (or From/To) bounds are present and that all dated entries lie within.
    """
    # Grab any YYYY-MM-DD, DD-MMM-YYYY, DD/MM/YYYY variants
    dates = re.findall(r"\b(\d{1,2}[-/ ](?:[A-Za-z]{3}|\d{1,2})[-/ ]\d{2,4}|\d{4}-\d{2}-\d{2})\b", txt)
    if not dates:
        return None

    # Look for header period
    hdr = re.search(r"(statement\s*period|for\s*the\s*period|from\s*date|period:)\s*[:\-–]\s*([^\n]+)", txt, re.I)
    if not hdr:
        return {"signal": "No explicit statement period header; cannot validate date window", "category": "content_conflict", "polarity": "inconclusive", "confidence": 0.25}

    # We won’t parse calendar precisely; only detect obvious out-of-window hints:
    # If there are many distinct months/years on a single-page “period”, flag as suspicious.
    year_hits = re.findall(r"\b(20\d{2}|19\d{2})\b", txt)
    if len(set(year_hits)) >= 3:
        return {"signal": "Multiple distinct years appear on the page vs single period header", "category": "content_conflict", "polarity": "fraud_positive", "confidence": 0.6}
    return {"signal": "Dates appear broadly consistent with a single period header", "category": "content_conflict", "polarity": "benign", "confidence": 0.6}

def _bank_identity_consistency(txt: str) -> list[dict]:
    """
    Simple checks:
      - multiple bank names/brands on same page
      - URL / SWIFT / IFSC patterns vs named bank (very light)
    """
    low = txt.lower()
    candidates = ["bank of china", "bank of baroda", "access bank", "gtbank", "guaranty trust", "axis bank", "fcmb", "first city monument bank"]
    present = [b for b in candidates if b in low]
    sigs = []
    if len(present) >= 2:
        sigs.append({"signal": f"Multiple bank brands detected on page: {present}", "category": "content_conflict", "polarity": "fraud_positive", "confidence": 0.7})
    # CamScanner watermark = medium risk indicator (not proof)
    if "scanned by camscanner" in low or "camscanner" in low:
        sigs.append({"signal": "Contains 'Scanned by CamScanner' watermark (raises fraud risk; not proof)", "category": "metadata_mismatch", "polarity": "fraud_positive", "confidence": 0.55})
    return sigs

def build_programmatic_signals(ocr_text: str) -> list[dict]:
    """Aggregate programmatic checks into signals."""
    signals = []
    try:
        s = _sum_line_items_vs_total(ocr_text)
        if s:
            s["source"] = "programmatic"
            signals.append(s)
    except Exception:
        pass
    try:
        s = _date_window_consistency(ocr_text)
        if s:
            s["source"] = "programmatic"
            signals.append(s)
    except Exception:
        pass
    try:
        for s in _bank_identity_consistency(ocr_text):
            s["source"] = "programmatic"
            signals.append(s)
    except Exception:
        pass
    return signals

# -----------------------------------------------------------------------------
# OpenAI interaction - structured JSON expected
# -----------------------------------------------------------------------------
def analyze_document(pdf_path: str, max_pages: int = 3, escalate=True) -> List[FraudResult]:
    """
    Analyze a PDF document; return a list of FraudResult (one per page analyzed).
    escalate=True allows calling stronger model gpt-5 when heuristics or content checks trigger.
    """
    pages = extract_pdf_images(pdf_path)
    results: List[FraudResult] = []

    for i, page_img in enumerate(pages[:max_pages]):
        print(f"  Processing page {i+1}/{len(pages)}")

        # 1) OCR
        ocr_text = extract_ocr_text(page_img)

        # 2) local pixel heuristics
        ela = ela_score(page_img)
        cm  = copy_move_score(page_img)

        # 3) NEW: programmatic (semantic) fraud signals from OCR text
        prog_signals = build_programmatic_signals(ocr_text)

        # 4) Primary AI pass (gpt-5-mini)
        ai_raw = analyze_with_openai(page_img, text_hint=ocr_text, model="gpt-5-mini")

        # Merge programmatic signals into model signals BEFORE finalize_label
        ai_raw.setdefault("fraud_signals", []).extend(prog_signals)

        # 5) Finalize severity with merged signals
        ai_raw = finalize_label(ai_raw)

         # 5.5) Heuristic: if this page looks like a bank statement and the model is
        # moderately confident, do not leave it at LOW – bump to MEDIUM so it gets
        # human review.
        text_low = (ocr_text or "").lower()
        looks_like_statement = any(
            kw in text_low
            for kw in [
                "statement of account",
                "summary statement",
                "account statement",
                "statement period",
                "for the period of",
                "for the period:",
                "opening balance",
                "closing balance",
                "total withdrawals",
                "total lodgements",
                "available balance",
            ]
        )

        overall_after = float(
            ai_raw.get("overall_confidence", ai_raw.get("confidence", 0.0) or 0.0)
        )

        if (
            ai_raw.get("severity") == "LOW"
            and looks_like_statement
            and overall_after >= 0.5   # don't bump if the model is basically guessing
        ):
            ai_raw["severity"] = "MEDIUM"

        # 6) Decide whether to escalate (mini said LOW but we have strong content hints or spicy pixels)
        has_prog_pos = any(s.get("polarity") == "fraud_positive" and float(s.get("confidence", 0)) >= 0.55
                           for s in prog_signals)
        needs_escalation = (
            ai_raw.get("severity", "LOW") == "LOW"
            and (has_prog_pos or ela >= 0.10 or cm >= 0.12)
        )

        if needs_escalation and escalate:
            try:
                ai_raw_5 = analyze_with_openai(page_img, text_hint=ocr_text, model="gpt-5")
                ai_raw_5.setdefault("fraud_signals", []).extend(prog_signals)
                ai_raw_5 = finalize_label(ai_raw_5)

                order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
                if order.get(ai_raw_5.get("severity", "LOW"), 0) > order.get(ai_raw.get("severity", "LOW"), 0):
                    ai_raw = ai_raw_5
                    ai_raw["escalated"] = True
            except Exception:
                pass  # keep mini result if escalation fails

        # 7) Additional lightweight local flags (kept as “inconclusive” unless really strong)
        local_flags = []
        low_text = (ocr_text or "").lower()
        if "signature" in low_text and "signed" not in low_text:
            local_flags.append({"signal": "Signature placeholder without signature.",
                                "category": "content_conflict", "polarity": "inconclusive", "confidence": 0.3})
        if "altered" in low_text or "void" in low_text:
            local_flags.append({"signal": "Suspicious keyword indicating alteration.",
                                "category": "content_conflict", "polarity": "fraud_positive", "confidence": 0.6})

        # 8) Compute overall confidence if the model didn't provide one
        overall = ai_raw.get("overall_confidence", ai_raw.get("confidence"))
        if overall is None:
            sigs = ai_raw.get("fraud_signals", [])
            vals = []
            for s in sigs:
                if isinstance(s, dict):
                    vals.append(float(s.get("confidence", 0.0)))
            overall = float(sum(vals) / len(vals)) if vals else 0.0

        # 9) Merge model signals + local flags for output
        model_signals = ai_raw.get("fraud_signals", [])
        merged_signals = []
        for s in model_signals:
            if isinstance(s, dict):
                merged_signals.append(s)
            else:
                merged_signals.append({"signal": str(s), "category": "other",
                                       "polarity": "inconclusive", "confidence": 0.0})
        merged_signals.extend(local_flags)

        result = FraudResult(
            file_path=pdf_path,
            page_number=i + 1,
            ocr_text=(ocr_text or "")[:200],
            ai_summary=ai_raw.get("summary", ""),
            fraud_signals=merged_signals,
            severity=ai_raw.get("severity", "LOW"),
            confidence=float(overall),
            extra={"ela": float(ela), "copy_move": float(cm),
                   "escalated": bool(ai_raw.get("escalated", False))}
        )
        results.append(result)

    return results

def analyze_financial_pdf(pdf_path: str, max_pages: int = 20) -> dict:
    """
    Web-friendly wrapper for use in the app.

    Returns a JSON-serializable dict with:
      - doc-level severity
      - page-level details
    """
    page_results = analyze_document(pdf_path, max_pages=max_pages, escalate=True)
    doc_label = doc_severity(page_results)

    return {
        "file_path": pdf_path,
        "doc_severity": doc_label,
        "pages": [
            {
                "page_number": r.page_number,
                "severity": r.severity,
                "confidence": r.confidence,
                "ocr_preview": r.ocr_text,
                "ai_summary": r.ai_summary,
                "fraud_signals": r.fraud_signals,
                "extra": r.extra,
            }
            for r in page_results
        ],
    }

# PIL TO DATA
def pil_to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    """
    Encode a PIL image as a data: URL for the OpenAI vision model.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

# -----------------------------------------------------------------------------
# Color helper for terminal output
# -----------------------------------------------------------------------------
def color(severity: str) -> str:
    return {
        "LOW": "\033[92mLOW\033[0m",
        "MEDIUM": "\033[93mMEDIUM\033[0m",
        "HIGH": "\033[91mHIGH\033[0m",
    }.get(severity.upper(), severity)


def analyze_with_openai(pil_img: Image.Image, text_hint: str = "", model: str = "gpt-5-mini") -> dict:
    """
    Call OpenAI with a strict JSON contract.
    Now sends BOTH:
      - text (user_prompt)
      - the page image (as base64 data URL)
    """

    def _scrub_path(s: str) -> str:
        s = re.sub(r"[A-Za-z]:\\\\[^\\s]+", "", s)
        s = re.sub(r"/[^\\s]+(?:/[^\\s]+)*", "", s)
        return s

    safe_hint = _scrub_path(text_hint or "")

    # quick numeric heuristics to feed the model
    ela = ela_score(pil_img)
    cm  = copy_move_score(pil_img)

    # ----- SYSTEM PROMPT (JSON contract) -----
    system_prompt = (
        "You are a document forensic assistant. Do not use or infer anything from "
        "filenames, paths, or directory names; you will not be given them. "
        "Return STRICT JSON (no extra text) with the following top-level keys:\n"
        "{\n"
        '  \"summary\": string,\n'
        '  \"raw_severity\": \"LOW\"|\"MEDIUM\"|\"HIGH\",\n'
        '  \"overall_confidence\": number,\n'
        '  \"fraud_signals\": [\n'
        '      { \"signal\": string, \"category\": string, \"polarity\": \"fraud_positive\"|\"benign\"|\"inconclusive\", \"confidence\": number }\n'
        '  ]\n'
        "}\n\n"
        "Important: Count normal redactions and typical scanning artifacts as 'benign'. "
        "Only mark a signal 'fraud_positive' if it meaningfully increases the probability of tampering.\n"
        "If ELA/copy-move are low but CONTENT CONFLICTS are detected (e.g., TOTAL ≠ sum of line items, "
        "multiple bank brands, inconsistent date window, mobile-scan watermarks), mark those signals as "
        "'fraud_positive' with appropriate confidence.\n"
        "If you are unsure, use 'inconclusive' with low confidence (≈0.2). "
        "If numeric or date values are hard to read due to OCR noise or garbling, you MUST treat any apparent "
        "mismatch as 'inconclusive' with confidence ≤ 0.4. "
        "Only call a numeric/date mismatch 'fraud_positive' when the values are clearly readable and you are "
        "confident they truly do not reconcile.\n"
        "If you cannot produce valid JSON, return an object with keys 'summary' and 'raw_severity' at minimum."
    )

    # ----- USER PROMPT (text + heuristics) -----
    user_prompt = f"""
Analyze a single scanned page for visual tampering. Use ONLY the image, OCR hint, and heuristics.
Do not infer anything from filenames or directories (you are not given them).

OCR hint (may be noisy):
\"\"\"{safe_hint[:2000]}\"\"\"


Numeric heuristics:
  - ELA_score: {ela:.4f}   (0..1 ; higher can indicate heterogeneous compression)
  - copy_move_score: {cm:.4f} (0..1 ; higher can indicate copied/duplicated regions)

Focus checks on:
  - copy-move / cloned regions, duplicated texture blocks
  - stamp / seal geometry mismatch vs interior text
  - compression/noise discontinuities around pasted regions
  - unnatural hard edges, perfectly rectangular overlays with no haloes
  - content conflicts (e.g., 'computer generated' yet signed, mismatched totals/dates, mixed bank brands)

Return structured JSON exactly as specified by the system message.
"""

    # ----- Encode image as base64 data URL -----
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{img_b64}"

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[
                # system can stay a simple string
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        # ✅ this must be type "text", not "input_text"
                        {"type": "text", "text": user_prompt},
                        # ✅ this must be type "image_url"
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            timeout=60,
        )

    resp = with_retries(_call, max_tries=2, base_delay=1.5)

    # Parse strict-JSON response
    try:
        content = resp.choices[0].message.content
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            return json.loads(content)
    except Exception:
        pass

    # Fallback if parsing fails
    try:
        raw = resp.choices[0].message.content
        return json.loads(raw) if isinstance(raw, str) else {
            "summary": "ERROR parsing model output",
            "raw_severity": "LOW",
            "overall_confidence": 0.0,
            "fraud_signals": []
        }
    except Exception:
        return {
            "summary": "No structured output",
            "raw_severity": "LOW",
            "overall_confidence": 0.0,
            "fraud_signals": []
        }

# -----------------------------------------------------------------------------
# Main CLI: supports multiple paths (files or folders)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="One or more paths to PDF files or folders")
    parser.add_argument("--pages", type=int, default=1, help="Max pages per document (default: 1 for quick tests)")
    parser.add_argument("--no-escalate", action="store_true", help="Disable escalation to gpt-5")
    args = parser.parse_args()

    all_results: List[FraudResult] = []

    for path in args.paths:
        if os.path.isdir(path):
            pdf_files = glob.glob(os.path.join(path, "*.pdf"))
        elif os.path.isfile(path) and path.lower().endswith(".pdf"):
            pdf_files = [path]
        else:
            print(f"⚠️ Invalid path: {path}")
            continue

        for pdf in pdf_files:
            print(f"\n>> Analyzing {pdf} ...")
            try:
                results = analyze_document(pdf, max_pages=args.pages, escalate=(not args.no_escalate))
                all_results.extend(results)

                # print per-page summaries
                for r in results:
                    print(f"→ {os.path.basename(r.file_path)} (page {r.page_number}) : "
                          f"{color(r.severity)} (signals={len(r.fraud_signals)}, conf={r.confidence:.2f}, ela={r.extra['ela']:.3f}, cm={r.extra['copy_move']:.3f})")
                    
                # NEW: doc-level severity for this PDF
                doc_label = doc_severity(results)
                print(f"   DOC-LEVEL: {os.path.basename(pdf)} -> {color(doc_label)}")

            except KeyboardInterrupt:
                print("Interrupted by user. Exiting.")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error analyzing {pdf}: {e}")

    # Save combined output
    out_path = os.path.join(os.getcwd(), "financial_batch_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        # convert dataclasses to dicts
        json.dump([{
            **r.__dict__,
            "fraud_signals": r.fraud_signals,
            "extra": r.extra
        } for r in all_results], f, indent=2, ensure_ascii=False)

    print(f"\n✅ Batch completed. Results saved to {out_path}")