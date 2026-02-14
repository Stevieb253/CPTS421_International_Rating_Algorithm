#!/usr/bin/env python3
"""
transcript_fraud_detector.py

Batch-capable PDF forensic pre-screen for academic transcripts:
- OCR pages (reuses financial_fraud_detector OCR + pdf2image utilities)
- Run quick image heuristics (ELA, coarse copy-move)
- Extract transcript-specific text features (course codes, GPA, registrar, etc.)
- Ask GPT-5-mini for structured analysis (fraud signals with polarity & confidence)
- Optionally escalate suspicious pages to GPT-5 for second opinion
- Finalize severity using same conservative rules as financial_fraud_detector
- Output combined batch_results.json and colorized console summary

Author: (you)
"""

import os
import io
import json
import re
import glob
import argparse
import base64
from typing import List, Dict, Any

from PIL import Image
from openai import OpenAI

# Reuse helpers + dataclass from your financial detector
import financial_fraud_detector as fin

# OpenAI client (new-style)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class TranscriptFraudDetector:
    """
    Encapsulates all transcript-specific fraud analysis.

    Public API:
        analyze_document(pdf_path: str, max_pages: int = 1) -> List[fin.FraudResult]

    Internally it:
      - converts PDF pages to images
      - runs OCR (reusing fin.extract_ocr_text)
      - extracts transcript-focused heuristic features
      - calls an OpenAI vision+text model with a strict JSON contract
      - passes the JSON through fin.finalize_label to get LOW/MEDIUM/HIGH
    """

    def __init__(self, max_pages: int = 1, escalate: bool = True) -> None:
        self.max_pages = max_pages
        self.escalate = escalate

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def analyze_document(self, pdf_path: str) -> List[fin.FraudResult]:
        pages = fin.extract_pdf_images(pdf_path)  # pdf2image via financial module
        results: List[fin.FraudResult] = []

        for i, page_img in enumerate(pages[: self.max_pages]):
            print(f"  [Transcript] Processing page {i + 1}/{len(pages)}")

            # 1) OCR
            ocr_text = fin.extract_ocr_text(page_img)

            # 2) Transcript-specific features
            features = self._extract_features(ocr_text)

            # 3) Image heuristics (for context)
            ela = fin.ela_score(page_img)
            cm = fin.copy_move_score(page_img)

            # 4) Primary AI pass (gpt-5-mini)
            ai_raw = self._analyze_with_openai(
                page_img, ocr_text, features, model="gpt-5-mini"
            )

            # 5) Finalize LOW/MEDIUM/HIGH severity using shared logic
            ai_raw = fin.finalize_label(ai_raw)

            # 5.5) Policy alignment with financial_fraud_detector:
            # For documents that look like real academic transcripts/diplomas and
            # where the model is at least moderately confident, we do not leave
            # them at LOW. We bump LOW → MEDIUM so they always get human review.
            looks_like_transcript = self._looks_like_transcript(ocr_text)
            has_suspicious_text = bool(features.get("suspicious_phrases_found"))
            suspicious_structure = (
                features.get("course_code_count", 0) == 0
                or not features.get("has_word_transcript", False)
            )

            overall_after = float(
                ai_raw.get("overall_confidence", ai_raw.get("confidence", 0.0) or 0.0)
            )

            if (
                ai_raw.get("severity") == "LOW"
                and looks_like_transcript
                and overall_after >= 0.5
            ):
                ai_raw["severity"] = "MEDIUM"

            # 6) Decide whether to escalate to gpt-5 for a second opinion.
            # Now that we've applied the bump rule, only escalate if it is still LOW
            # and there are suspicious signals or spicy pixels.
            needs_escalation = (
                self.escalate
                and ai_raw.get("severity") == "LOW"
                and (
                    has_suspicious_text
                    or suspicious_structure
                    or ela >= 0.10
                    or cm >= 0.12
                )
            )

            if needs_escalation:
                try:
                    ai_raw_5 = self._analyze_with_openai(
                        page_img, ocr_text, features, model="gpt-5"
                    )
                    ai_raw_5 = fin.finalize_label(ai_raw_5)

                    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
                    if (
                        order.get(ai_raw_5.get("severity", "LOW"), 0)
                        > order.get(ai_raw.get("severity", "LOW"), 0)
                    ):
                        ai_raw = ai_raw_5
                        ai_raw["escalated"] = True
                except Exception:
                    # If escalation fails, keep mini result
                    pass

            # 7) Compute overall confidence if missing
            overall = ai_raw.get("overall_confidence", ai_raw.get("confidence"))
            if overall is None:
                sigs = ai_raw.get("fraud_signals", [])
                vals = []
                for s in sigs:
                    if isinstance(s, dict):
                        vals.append(float(s.get("confidence", 0.0)))
                overall = float(sum(vals) / len(vals)) if vals else 0.0

            # 8) Normalize fraud_signals list
            model_signals = ai_raw.get("fraud_signals", [])
            merged_signals = []
            for s in model_signals:
                if isinstance(s, dict):
                    merged_signals.append(s)
                else:
                    merged_signals.append(
                        {
                            "signal": str(s),
                            "category": "other",
                            "polarity": "inconclusive",
                            "confidence": 0.0,
                        }
                    )

            # 9) Build FraudResult
            result = fin.FraudResult(
                file_path=pdf_path,
                page_number=i + 1,
                ocr_text=(ocr_text or "")[:200],
                ai_summary=ai_raw.get("summary", ""),
                fraud_signals=merged_signals,
                severity=ai_raw.get("severity", "LOW"),
                confidence=float(overall),
                extra={
                    "ela": float(ela),
                    "copy_move": float(cm),
                    "doc_type": "transcript",
                    "escalated": bool(ai_raw.get("escalated", False)),
                    "features": features,
                },
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Feature extraction (transcript-specific)
    # ------------------------------------------------------------------
    def _extract_features(self, ocr_text: str) -> Dict[str, Any]:
        """
        Extract simple, explainable transcript heuristics:
          - key words (transcript, registrar, official/unofficial)
          - course code patterns (e.g., CSE 101, MATH201)
          - GPA-like values
          - term/semester words
          - suspicious phrases commonly found in samples/templates
        """
        text = ocr_text or ""
        text_lower = text.lower()

        features: Dict[str, Any] = {}

        # 1) Key transcript words & fields
        features["has_word_transcript"] = "transcript" in text_lower
        features["has_registrar"] = "registrar" in text_lower
        features["has_official"] = "official" in text_lower
        features["has_unofficial"] = "unofficial" in text_lower

        # 2) Course codes (e.g., CSE 101, MATH201, ENG 350)
        course_pattern = re.compile(r"\b([A-Z]{2,4}\s?\d{3})\b")
        courses = course_pattern.findall(text)
        features["course_code_count"] = len(courses)
        features["distinct_course_codes"] = sorted(set(courses))[:50]

        # 3) GPA patterns (very rough)
        gpa_pattern = re.compile(
            r"\bGPA[:\s]+([0-4]\.\d{1,2})\b", flags=re.IGNORECASE
        )
        gpas = gpa_pattern.findall(text)
        features["gpa_values"] = gpas

        # 4) Term / semester words
        term_words = ["fall", "spring", "summer", "winter", "term", "semester"]
        features["has_term_words"] = any(t in text_lower for t in term_words)

        # 5) Suspicious phrases often seen on samples/templates
        suspicious_phrases = [
            "sample",
            "specimen",
            "void",
            "demo",
            "template",
            "for training use only",
            "for practice only",
            "not an official document",
        ]
        found_suspicious = [p for p in suspicious_phrases if p in text_lower]
        features["suspicious_phrases_found"] = found_suspicious

        # 6) Crude size hints
        features["char_count"] = len(text)
        features["line_count"] = text.count("\n") + 1

        return features

    def _looks_like_transcript(self, ocr_text: str) -> bool:
        low = (ocr_text or "").lower()
        keywords = [
            "transcript",
            "academic record",
            "registrar",
            "grade point average",
            "gpa",
            "credit hours",
            "course",
            "semester",
            "term",
        ]
        return any(k in low for k in keywords)
    
    # ------------------------------------------------------------------
    # OpenAI interaction
    # ------------------------------------------------------------------
    def _analyze_with_openai(
        self,
        pil_img: Image.Image,
        ocr_text: str,
        features: Dict[str, Any],
        model: str = "gpt-5-mini",
    ) -> dict:
        """
        Call OpenAI for transcript-specific analysis.

        Returns JSON with the SAME contract as financial_fraud_detector.analyze_with_openai:
            {
                "summary": str,
                "raw_severity": "LOW"|"MEDIUM"|"HIGH",
                "overall_confidence": number,
                "fraud_signals": [
                    { "signal": str, "category": str,
                      "polarity": "fraud_positive"|"benign"|"inconclusive",
                      "confidence": number }
                ]
            }
        """
        safe_hint = self._scrub_text(ocr_text or "")
        features_json = json.dumps(features, ensure_ascii=False)

        ela = fin.ela_score(pil_img)
        cm = fin.copy_move_score(pil_img)

        # ---- SYSTEM PROMPT ----
        system_prompt = (
            "You are a document forensic assistant helping a university admissions office "
            "evaluate academic transcripts. You only detect potential fraud; you must NOT "
            "provide any advice on how to forge or modify documents.\n\n"
            "You will be given a scanned transcript page (image), noisy OCR text, and a set "
            "of heuristic features (course code count, GPA values, key words, suspicious phrases).\n\n"
            "Return STRICT JSON (no extra text) with top-level keys:\n"
            "{\n"
            '  \"summary\": string,\n'
            '  \"raw_severity\": \"LOW\"|\"MEDIUM\"|\"HIGH\",\n'
            '  \"overall_confidence\": number,\n'
            '  \"fraud_signals\": [\n'
            '      { \"signal\": string, \"category\": string, '
            '\"polarity\": \"fraud_positive\"|\"benign\"|\"inconclusive\", '
            '\"confidence\": number }\n'
            "  ]\n"
            "}\n\n"
            "Interpretation guidelines:\n"
            "- 'LOW' means no meaningful fraud-positive evidence.\n"
            "- 'MEDIUM' means some suspicious elements worth manual review.\n"
            "- 'HIGH' means strong evidence the transcript has been altered or fabricated.\n"
            "Mark signals as 'fraud_positive' only when they meaningfully increase the probability "
            "of tampering (e.g., inconsistent course/GPA structure, mismatched institution names, "
            "obvious pasted-over grades, repeated visual blocks, 'sample' watermarks on a claimed "
            "official transcript).\n"
            "Normal scanning noise and common layout quirks are 'benign'. If you are unsure, use "
            "'inconclusive' with low confidence (~0.2–0.4)."
        )

        # ---- USER PROMPT ----
        user_prompt = (
            "You are analyzing a single scanned academic transcript page.\n\n"
            "OCR hint (may be noisy):\n"
            f"{safe_hint[:3000]}\n\n"
            "Transcript heuristic features (JSON representation):\n"
            f"{features_json}\n\n"
            "Numeric/image heuristics:\n"
            f"  - ELA_score: {ela:.4f}   (0..1 ; higher can indicate heterogeneous compression/possible pasting)\n"
            f"  - copy_move_score: {cm:.4f} (0..1 ; higher can indicate duplicated regions)\n\n"
            "Focus on:\n"
            "  - Visual alterations around grades, name, or institution\n"
            "  - Inconsistent institution/registrar wording vs typical transcripts\n"
            "  - Improbable or inconsistent course / GPA structures\n"
            "  - Watermarks or phrases suggesting 'sample' or 'template'\n"
            "  - Any clear evidence of copy-paste tampering or region duplication\n\n"
            "Return STRICT JSON in the format required by the system message.\n"
        )

        # ---- Encode image ----
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_b64}"

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                timeout=60,
            )

        resp = fin.with_retries(_call, max_tries=2, base_delay=1.5)

        # Parse strict JSON response
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
            return (
                json.loads(raw)
                if isinstance(raw, str)
                else {
                    "summary": "ERROR parsing model output",
                    "raw_severity": "LOW",
                    "overall_confidence": 0.0,
                    "fraud_signals": [],
                }
            )
        except Exception:
            return {
                "summary": "No structured output",
                "raw_severity": "LOW",
                "overall_confidence": 0.0,
                "fraud_signals": [],
            }

    @staticmethod
    def _scrub_text(s: str) -> str:
        """
        Remove obvious file paths / local identifiers from OCR text before
        sending to the model.
        """
        s = re.sub(r"[A-Za-z]:\\\\[^\\s]+", "", s)      # Windows-style paths
        s = re.sub(r"/[^\\s]+(?:/[^\\s]+)*", "", s)     # Unix-style paths
        return s
    
def analyze_transcript_pdf(pdf_path: str, max_pages: int = 20) -> Dict[str, Any]:
    """
    Web-friendly wrapper for transcript screening.

    Returns a JSON-serializable dict with:
    - doc-level severity
    - page-level details
    """
    detector = TranscriptFraudDetector(
        max_pages=max_pages,
        escalate=True
    )
    page_results: List[fin.FraudResult] = detector.analyze_document(pdf_path)
    doc_label = fin.doc_severity(page_results)

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

# -------------------------------------------------------------------------
# CLI wrapper — similar to financial_fraud_detector
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more paths to transcript PDF files or folders",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Max pages per document (default: 1 for quick tests)",
    )
    parser.add_argument(
        "--no-escalate",
        action="store_true",
        help="Disable escalation to gpt-5",
    )
    args = parser.parse_args()

    detector = TranscriptFraudDetector(
        max_pages=args.pages,
        escalate=(not args.no_escalate),
    )

    all_results: List[fin.FraudResult] = []

    for path in args.paths:
        if os.path.isdir(path):
            pdf_files = glob.glob(os.path.join(path, "*.pdf"))
        elif os.path.isfile(path) and path.lower().endswith(".pdf"):
            pdf_files = [path]
        else:
            print(f"⚠️ Invalid path: {path}")
            continue

        for pdf in pdf_files:
            print(f"\n>> [Transcript] Analyzing {pdf} ...")
            try:
                results = detector.analyze_document(pdf)
                all_results.extend(results)

                for r in results:
                    print(
                        f"→ {os.path.basename(r.file_path)} (page {r.page_number}) : "
                        f"{fin.color(r.severity)} "
                        f"(signals={len(r.fraud_signals)}, conf={r.confidence:.2f}, "
                        f"ela={r.extra['ela']:.3f}, cm={r.extra['copy_move']:.3f})"
                    )

                doc_label = fin.doc_severity(results)
                print(
                    f"   DOC-LEVEL: {os.path.basename(pdf)} -> {fin.color(doc_label)}"
                )

            except KeyboardInterrupt:
                print("Interrupted by user. Exiting.")
                raise SystemExit(1)
            except Exception as e:
                print(f"❌ Error analyzing {pdf}: {e}")

    out_path = os.path.join(os.getcwd(), "transcript_batch_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **r.__dict__,
                    "fraud_signals": r.fraud_signals,
                    "extra": r.extra,
                }
                for r in all_results
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n✅ Transcript batch completed. Results saved to {out_path}")