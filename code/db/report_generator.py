#!/usr/bin/env python3
"""
report_generator.py
Generates a professional PDF admission analysis report for an applicant's file.
"""

import io
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

from services.nlp_service import NLPService

# =============================================================================
# Brand Colors
# =============================================================================
WSU_PURPLE     = colors.HexColor('#5C1784')
ACCENT_PURPLE  = colors.HexColor('#667eea')
LIGHT_PURPLE   = colors.HexColor('#EDE9FE')
SCORE_GREEN    = colors.HexColor('#166534')
SCORE_GREEN_BG = colors.HexColor('#DCFCE7')
SCORE_RED      = colors.HexColor('#991B1B')
SCORE_RED_BG   = colors.HexColor('#FEE2E2')
SCORE_BLUE     = colors.HexColor('#1E3A8A')
SCORE_BLUE_BG  = colors.HexColor('#DBEAFE')
WARN_AMBER     = colors.HexColor('#92400E')
WARN_AMBER_BG  = colors.HexColor('#FEF3C7')
NEUTRAL_GRAY   = colors.HexColor('#374151')
LIGHT_GRAY     = colors.HexColor('#F9FAFB')
MID_GRAY       = colors.HexColor('#E5E7EB')
TEXT_GRAY      = colors.HexColor('#6B7280')


# =============================================================================
# Numpy Safety Converter
# =============================================================================
def _convert_numpy(obj):
    """Recursively convert all numpy types to plain Python types."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =============================================================================
# Key Normalizer
# =============================================================================
def _normalize_essay_analysis(ea: dict) -> dict:
    """
    NLPService.score_essay() returns camelCase keys.
    report_generator and app.py expect snake_case keys.
    This function accepts either format and returns snake_case.
    """
    if not ea:
        return {}

    def get(camel, snake, default=0.0):
        return ea.get(snake, ea.get(camel, default))

    return {
        'clarity_focus':              float(get('clarityFocus',             'clarity_focus',              0.0)),
        'development_organization':   float(get('developmentOrganization',  'development_organization',   0.0)),
        'creativity_style':           float(get('creativityStyle',          'creativity_style',           0.0)),
        'rubric_score':               float(get('essayRubricScore',         'rubric_score',               0.0)),
        'grammar_score':              float(get('grammarScore',             'grammar_score',              0.0)),
        'coherence_score':            float(get('coherenceScore',           'coherence_score',            0.0)),
        'vocabulary_richness':        float(get('vocabularyRichness',       'vocabulary_richness',        0.0)),
        'insights':                   ea.get('insights',   []),
        'strengths':                  ea.get('strengths',  []),
        'weaknesses':                 ea.get('weaknesses', []),
    }


# =============================================================================
# Styles
# =============================================================================
def _styles():
    base = getSampleStyleSheet()
    return {
        'ReportTitle': ParagraphStyle(
            'ReportTitle', parent=base['Normal'], fontSize=20,
            leading=26, textColor=colors.white, fontName='Helvetica-Bold',
            alignment=TA_LEFT, spaceAfter=4
        ),
        'ReportSubtitle': ParagraphStyle(
            'ReportSubtitle', parent=base['Normal'], fontSize=10,
            leading=14, textColor=colors.HexColor('#C4B5FD'),
            fontName='Helvetica', alignment=TA_LEFT,
        ),
        'SectionHeading': ParagraphStyle(
            'SectionHeading', parent=base['Normal'], fontSize=11,
            leading=14, textColor=WSU_PURPLE, fontName='Helvetica-Bold',
            spaceBefore=18, spaceAfter=8
        ),
        'FieldLabel': ParagraphStyle(
            'FieldLabel', parent=base['Normal'], fontSize=8,
            leading=10, textColor=TEXT_GRAY, fontName='Helvetica-Bold', spaceAfter=2
        ),
        'FieldValue': ParagraphStyle(
            'FieldValue', parent=base['Normal'], fontSize=10,
            leading=14, textColor=NEUTRAL_GRAY, fontName='Helvetica',
        ),
        'BodyText': ParagraphStyle(
            'BodyText', parent=base['Normal'], fontSize=9.5,
            leading=14, textColor=NEUTRAL_GRAY, fontName='Helvetica',
            alignment=TA_JUSTIFY,
        ),
        'CommentBox': ParagraphStyle(
            'CommentBox', parent=base['Normal'], fontSize=9.5,
            leading=15, textColor=NEUTRAL_GRAY, fontName='Helvetica',
            alignment=TA_JUSTIFY,
        ),
        'ScoreNumber': ParagraphStyle(
            'ScoreNumber', parent=base['Normal'], fontSize=26,
            leading=30, fontName='Helvetica-Bold', alignment=TA_CENTER,
        ),
        'ScoreLabel': ParagraphStyle(
            'ScoreLabel', parent=base['Normal'], fontSize=7.5,
            leading=10, fontName='Helvetica-Bold', alignment=TA_CENTER,
        ),
        'ScoreSubLabel': ParagraphStyle(
            'ScoreSubLabel', parent=base['Normal'], fontSize=7,
            leading=9, textColor=TEXT_GRAY, fontName='Helvetica',
            alignment=TA_CENTER,
        ),
        'RecommendationText': ParagraphStyle(
            'RecommendationText', parent=base['Normal'], fontSize=12,
            leading=16, fontName='Helvetica-Bold', alignment=TA_CENTER,
        ),
        'FooterText': ParagraphStyle(
            'FooterText', parent=base['Normal'], fontSize=7.5,
            leading=10, textColor=TEXT_GRAY, fontName='Helvetica',
            alignment=TA_CENTER,
        ),
        'ConfidentialBadge': ParagraphStyle(
            'ConfidentialBadge', parent=base['Normal'], fontSize=7,
            leading=9, textColor=WARN_AMBER, fontName='Helvetica-Bold',
            alignment=TA_RIGHT,
        ),
        'InsightBullet': ParagraphStyle(
            'InsightBullet', parent=base['Normal'], fontSize=9,
            leading=13, textColor=NEUTRAL_GRAY, fontName='Helvetica',
            leftIndent=8, spaceAfter=2,
        ),
    }


# =============================================================================
# PDF Block Builders
# =============================================================================
def _header_table(styles, student_id, country, reviewer_name, now_str):
    header_data = [[
        Paragraph(f"WSU Office of International Programs", styles['ReportTitle']),
        Paragraph("CONFIDENTIAL", styles['ConfidentialBadge']),
    ]]
    sub_data = [[
        Paragraph(
            f"Admission Analysis Report &nbsp;|&nbsp; Student: {student_id} &nbsp;|&nbsp; "
            f"Country: {country} &nbsp;|&nbsp; Reviewer: {reviewer_name} &nbsp;|&nbsp; {now_str}",
            styles['ReportSubtitle']
        ),
        Paragraph("", styles['ReportSubtitle']),
    ]]
    tbl = Table(header_data + sub_data, colWidths=[5.5 * inch, 2 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, -1), WSU_PURPLE),
        ('TOPPADDING',  (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return tbl


def _score_card(label, value, sublabel, text_color, bg_color, styles):
    card_data = [
        [Paragraph(label,          styles['ScoreLabel'])],
        [Paragraph(f"{value:.2f}", styles['ScoreNumber'])],
        [Paragraph(sublabel,       styles['ScoreSubLabel'])],
    ]
    tbl = Table(card_data, colWidths=[2.1 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), bg_color),
        ('TEXTCOLOR',     (0, 1), (0, 1),   text_color),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('ROUNDEDCORNERS', [6]),
    ]))
    return tbl


def _scores_row(pos_score, neg_score, final_score, styles):
    row = [[
        _score_card("POS SCORE",   pos_score,   "positive attributes", SCORE_GREEN, SCORE_GREEN_BG, styles),
        _score_card("NEG SCORE",   neg_score,   "risk deductions",     SCORE_RED,   SCORE_RED_BG,   styles),
        _score_card("FINAL SCORE", final_score, "POS minus NEG",       SCORE_BLUE,  SCORE_BLUE_BG,  styles),
    ]]
    tbl = Table(row, colWidths=[2.1 * inch, 2.1 * inch, 2.1 * inch])
    tbl.setStyle(TableStyle([
        ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',  (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _info_table(rows, styles):
    table_data = [
        [
            Paragraph(label, styles['FieldLabel']),
            Paragraph(str(value), styles['FieldValue']),
        ]
        for label, value in rows
    ]
    tbl = Table(table_data, colWidths=[2 * inch, 5.5 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [LIGHT_GRAY, colors.white]),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('BOX',           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ('LINEBELOW',     (0, 0), (-1, -2), 0.3, MID_GRAY),
    ]))
    return tbl


def _factor_table(breakdown, styles):
    rows = [
        [Paragraph(k, styles['FieldLabel']), Paragraph(f"{float(v):.2f}", styles['FieldValue'])]
        for k, v in breakdown.items()
    ]
    tbl = Table(rows, colWidths=[4 * inch, 3.5 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [LIGHT_GRAY, colors.white]),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('BOX',           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ('LINEBELOW',     (0, 0), (-1, -2), 0.3, MID_GRAY),
    ]))
    return tbl


def _essay_metrics_table(ea, styles):
    """ea is already normalized to snake_case by _normalize_essay_analysis()"""
    metrics = [
        ("Clarity & Focus",          ea.get('clarity_focus', 0)),
        ("Development & Organization", ea.get('development_organization', 0)),
        ("Creativity & Style",        ea.get('creativity_style', 0)),
        ("Essay Rubric Score",        ea.get('rubric_score', 0)),
        ("Grammar Score",             ea.get('grammar_score', 0)),
        ("Coherence Score",           ea.get('coherence_score', 0)),
        ("Vocabulary Richness",       ea.get('vocabulary_richness', 0)),
    ]
    rows = [
        [Paragraph(label, styles['FieldLabel']), Paragraph(f"{float(val):.2f}", styles['FieldValue'])]
        for label, val in metrics
    ]
    tbl = Table(rows, colWidths=[4 * inch, 3.5 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [LIGHT_GRAY, colors.white]),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('BOX',           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ('LINEBELOW',     (0, 0), (-1, -2), 0.3, MID_GRAY),
    ]))
    return tbl


def _recommendation_block(recommendation, final_score, styles):
    # Pick color based on recommendation
    rec_upper = recommendation.upper()
    if 'HIGHLY' in rec_upper:
        bg, fg = SCORE_GREEN_BG, SCORE_GREEN
    elif 'CONDITIONAL' in rec_upper or 'MONITORING' in rec_upper:
        bg, fg = WARN_AMBER_BG, WARN_AMBER
    elif 'RISK' in rec_upper or 'BORDERLINE' in rec_upper:
        bg, fg = SCORE_RED_BG, SCORE_RED
    else:
        bg, fg = SCORE_BLUE_BG, SCORE_BLUE

    data = [[
        Paragraph(recommendation or "N/A", ParagraphStyle(
            'RecText', parent=getSampleStyleSheet()['Normal'],
            fontSize=13, fontName='Helvetica-Bold',
            alignment=TA_CENTER, textColor=fg
        ))
    ]]
    tbl = Table(data, colWidths=[7.5 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), bg),
        ('TOPPADDING',    (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('BOX',           (0, 0), (-1, -1), 1, fg),
    ]))
    return tbl


def _comments_block(staff_comments, reviewer_name, styles):
    comment_text = staff_comments.strip() if staff_comments and staff_comments.strip() \
        else "(No additional comments recorded.)"
    data = [
        [Paragraph(f"Reviewer: {reviewer_name}", styles['FieldLabel'])],
        [Paragraph(comment_text, styles['CommentBox'])],
    ]
    tbl = Table(data, colWidths=[7.5 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), WARN_AMBER_BG),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ('BOX',           (0, 0), (-1, -1), 0.5, WARN_AMBER),
    ]))
    return tbl


def _signature_block(reviewer_name, styles):
    now_str = datetime.now().strftime("%B %d, %Y")
    data = [[
        Paragraph(f"Reviewed by: {reviewer_name}", styles['FieldLabel']),
        Paragraph(f"Date: {now_str}", styles['FieldLabel']),
        Paragraph("Signature: ______________________", styles['FieldLabel']),
    ]]
    tbl = Table(data, colWidths=[2.5 * inch, 2.5 * inch, 2.5 * inch])
    tbl.setStyle(TableStyle([
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
    ]))
    return tbl


# =============================================================================
# Main Public Function
# =============================================================================
def generate_report(
    student_data: dict,
    result_data: dict,
    staff_comments: str = "",
    reviewer_name: str = "Staff"
) -> io.BytesIO:
    """
    Generate a professional PDF admission analysis report.
    Safely handles numpy types and both camelCase / snake_case essay keys.
    """
    # -------------------------------------------------------------------------
    # 0. Convert all numpy types to plain Python (prevents scalar errors)
    # -------------------------------------------------------------------------
    student_data = _convert_numpy(student_data)
    result_data  = _convert_numpy(result_data)

    # -------------------------------------------------------------------------
    # 1. NLP Essay Scoring (only if essay not already scored)
    # -------------------------------------------------------------------------
    essay_text = student_data.get("essayText", "")
    if essay_text and not result_data.get("essayAnalysis"):
        try:
            nlp = NLPService()
            raw_scores = nlp.score_essay(essay_text)
            result_data["essayAnalysis"] = _convert_numpy(raw_scores)
        except Exception as e:
            print(f"NLP scoring skipped: {e}")
            result_data["essayAnalysis"] = {}

    # -------------------------------------------------------------------------
    # 2. Normalize essay analysis keys (camelCase → snake_case)
    # -------------------------------------------------------------------------
    raw_essay = result_data.get("essayAnalysis", {})
    essay_analysis = _normalize_essay_analysis(_convert_numpy(raw_essay))

    # -------------------------------------------------------------------------
    # 3. Extract top-level fields safely
    # -------------------------------------------------------------------------
    student_id     = str(student_data.get('studentId',    'N/A'))
    country        = str(student_data.get('country',      'N/A'))
    gpa            = student_data.get('gpa',              'N/A')
    curriculum     = str(student_data.get('curriculum',   'N/A'))
    travel         = str(student_data.get('travelHistory','N/A'))
    neg_factors    = student_data.get('negFactors',       [])
    essay_len      = len(essay_text)

    pos_score      = float(result_data.get('posScore',    0))
    neg_score      = float(result_data.get('negScore',    0))
    final_score    = float(result_data.get('finalScore',  0))
    recommendation = str(result_data.get('recommendation', ''))
    breakdown      = result_data.get('breakdown',         {})

    # -------------------------------------------------------------------------
    # 4. Build PDF
    # -------------------------------------------------------------------------
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.75 * inch,
    )

    styles  = _styles()
    story   = []
    now_str = datetime.now().strftime("%B %d, %Y  %I:%M %p")

    # Header
    story.append(_header_table(styles, student_id, country, reviewer_name, now_str))
    story.append(Spacer(1, 14))

    # Scores
    story.append(KeepTogether([
        Paragraph("Score Summary", styles['SectionHeading']),
        _scores_row(pos_score, neg_score, final_score, styles),
    ]))
    story.append(Spacer(1, 6))

    # Recommendation
    story.append(KeepTogether([
        Paragraph("Admission Recommendation", styles['SectionHeading']),
        _recommendation_block(recommendation, final_score, styles),
    ]))

    # Student profile
    neg_str = ', '.join(neg_factors) if neg_factors else 'None'
    profile_rows = [
        ("Student ID",        student_id),
        ("Country of Origin", country),
        ("Admissions GPA",    f"{gpa}"),
        ("Curriculum Type",   curriculum),
        ("Travel History",    travel),
        ("Essay Length",      f"~{essay_len} characters"),
        ("Risk Factors",      neg_str),
    ]
    story.append(KeepTogether([
        Paragraph("Student Profile", styles['SectionHeading']),
        _info_table(profile_rows, styles),
    ]))

    # Factor breakdown
    if breakdown:
        story.append(KeepTogether([
            Paragraph("Factor Breakdown", styles['SectionHeading']),
            _factor_table(breakdown, styles),
        ]))

    # Essay analysis metrics
    if essay_analysis:
        story.append(Paragraph("Essay Analysis", styles['SectionHeading']))
        story.append(_essay_metrics_table(essay_analysis, styles))
        story.append(Spacer(1, 6))

        insights  = essay_analysis.get('insights',  [])
        strengths = essay_analysis.get('strengths', [])
        weaknesses= essay_analysis.get('weaknesses',[])

        if insights or strengths or weaknesses:
            insight_items = []
            for item in insights:
                insight_items.append(Paragraph(f"• {item}", styles['InsightBullet']))
            if strengths:
                insight_items.append(Paragraph("<b>Strengths:</b>", styles['FieldLabel']))
                for s in strengths:
                    insight_items.append(Paragraph(f"• {s}", styles['InsightBullet']))
            if weaknesses:
                insight_items.append(Paragraph("<b>Areas for Improvement:</b>", styles['FieldLabel']))
                for w in weaknesses:
                    insight_items.append(Paragraph(f"• {w}", styles['InsightBullet']))

            notes_tbl = Table([[item] for item in insight_items], colWidths=[7 * inch])
            notes_tbl.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GRAY),
                ('TOPPADDING',    (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING',   (0, 0), (-1, -1), 12),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
                ('BOX',           (0, 0), (-1, -1), 0.5, MID_GRAY),
            ]))
            story.append(notes_tbl)

    # Staff comments
    story.append(KeepTogether([
        Paragraph("Staff Comments & Notes", styles['SectionHeading']),
        _comments_block(staff_comments, reviewer_name, styles),
    ]))

    # Signature
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY))
    story.append(Spacer(1, 8))
    story.append(_signature_block(reviewer_name, styles))

    # Footer
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This document is confidential and intended solely for use by authorized "
        "staff of the WSU Office of International Programs. Generated automatically "
        f"by the International Student Scoring System. Report generated on {now_str}.",
        styles['FooterText']
    ))

    doc.build(story)
    buf.seek(0)
    return buf