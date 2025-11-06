#!/usr/bin/env python3
"""
eval_results.py

Evaluate fraud_detector.py outputs at the document level.
Computes recall, precision, and confusion matrix from batch_results.json
"""

import os
import json
from financial_fraud_detector import doc_severity, FraudResult

# Load the saved JSON from your last run
RESULT_PATH = os.path.join(os.getcwd(), "batch_results.json")

if not os.path.exists(RESULT_PATH):
    raise FileNotFoundError(f"No batch_results.json found at {RESULT_PATH}")

with open(RESULT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Group page-level entries by document
grouped = {}
for entry in data:
    grouped.setdefault(entry["file_path"], []).append(entry)

results = []
for pdf_path, pages in grouped.items():
    # Rebuild FraudResult objects
    fraud_results = [FraudResult(**p) for p in pages]
    doc_label = doc_severity(fraud_results)

    # Infer ground truth from filename or folder
    label = "fraudulent" if "fraudulent" in pdf_path.lower() else "authentic"

    results.append((os.path.basename(pdf_path), label, doc_label))

# Compute metrics
tp = sum(1 for _, t, p in results if t == "fraudulent" and p in ["MEDIUM", "HIGH"])
fn = sum(1 for _, t, p in results if t == "fraudulent" and p == "LOW")
fp = sum(1 for _, t, p in results if t == "authentic" and p in ["MEDIUM", "HIGH"])
tn = sum(1 for _, t, p in results if t == "authentic" and p == "LOW")

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print("\n=== Document-Level Evaluation ===")
print(f"True Positives (fraudulent correctly flagged): {tp}")
print(f"False Negatives (fraudulent missed):          {fn}")
print(f"False Positives (authentic flagged):          {fp}")
print(f"True Negatives (authentic OK):                {tn}")
print(f"\nRecall:    {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Accuracy:  {(tp + tn) / max(1, len(results)):.2f}")
print("\nDetailed Results:")
for name, true, pred in results:
    print(f" - {name:50s} true={true:<12} predicted={pred}")