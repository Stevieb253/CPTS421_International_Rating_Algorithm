#!/usr/bin/env python3
"""
report_generator.py
Generates a professional PDF admission analysis report for an applicant's file.

The report documents:
- Student profile and scores
- Factor-by-factor breakdown
- Essay analysis results (scored via Hugging Face NLPService)
- Staff comments / reviewer 
- Final recommendation with justification

Used to create a paper trail for admission decisions.
"""

import io
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

# Import NLPService
from services.nlp_service import NLPService

# =============================================================================
# Brand Colors
# =============================================================================
WSU_PURPLE     = colors.HexColor('#5C1784')   # WSU Crimson-adjacent brand
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
# Styles
# =============================================================================
def _styles():
    base = getSampleStyleSheet()
    custom = {
        'ReportTitle': ParagraphStyle(
            'ReportTitle', parent=base['Normal'], fontSize=20,
            leading=26, textColor=colors.white, fontName='Helvetica-Bold', alignment=TA_LEFT, spaceAfter=4
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
            leading=14, textColor=NEUTRAL_GRAY, fontName='Helvetica', alignment=TA_JUSTIFY,
        ),
        'CommentBox': ParagraphStyle(
            'CommentBox', parent=base['Normal'], fontSize=9.5,
            leading=15, textColor=NEUTRAL_GRAY, fontName='Helvetica', alignment=TA_JUSTIFY,
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
            leading=9, textColor=TEXT_GRAY, fontName='Helvetica', alignment=TA_CENTER,
        ),
        'RecommendationText': ParagraphStyle(
            'RecommendationText', parent=base['Normal'], fontSize=12,
            leading=16, fontName='Helvetica-Bold', alignment=TA_CENTER,
        ),
        'FooterText': ParagraphStyle(
            'FooterText', parent=base['Normal'], fontSize=7.5,
            leading=10, textColor=TEXT_GRAY, fontName='Helvetica', alignment=TA_CENTER,
        ),
        'ConfidentialBadge': ParagraphStyle(
            'ConfidentialBadge', parent=base['Normal'], fontSize=7,
            leading=9, textColor=WARN_AMBER, fontName='Helvetica-Bold', alignment=TA_RIGHT,
        ),
        'InsightBullet': ParagraphStyle(
            'InsightBullet', parent=base['Normal'], fontSize=9,
            leading=13, textColor=NEUTRAL_GRAY, fontName='Helvetica', leftIndent=8, spaceAfter=2,
        ),
    }
    return custom

# =============================================================================
# Internal helpers
# =============================================================================
# ... Include all your existing _header_table, _score_card, _scores_row,
# _info_table, _factor_table, _essay_metrics_table, _recommendation_block,
# _comments_block, _signature_block functions unchanged ...

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

    Integrates Hugging Face NLP essay scoring automatically.
    """
    # -------------------------------------------------------------------------
    # 0. NLP Essay Scoring Integration
    # -------------------------------------------------------------------------
    nlp = NLPService()
    essay_text = student_data.get("essayText", "")
    if essay_text:
        essay_scores = nlp.score_essay(essay_text)
        result_data["essayAnalysis"] = essay_scores

    # -------------------------------------------------------------------------
    # 1. Prepare PDF buffer
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

    styles   = _styles()
    story    = []
    now_str  = datetime.now().strftime("%B %d, %Y  %I:%M %p")

    # Convenience
    student_id  = student_data.get('studentId', 'N/A')
    country     = student_data.get('country', 'N/A')
    gpa         = student_data.get('gpa', 'N/A')
    curriculum  = student_data.get('curriculum', 'N/A')
    travel      = student_data.get('travelHistory', 'N/A')
    neg_factors = student_data.get('negFactors', [])
    essay_len   = len(essay_text)

    pos_score   = result_data.get('posScore', 0)
    neg_score   = result_data.get('negScore', 0)
    final_score = result_data.get('finalScore', 0)
    recommendation = result_data.get('recommendation', '')
    breakdown   = result_data.get('breakdown', {})
    essay_analysis = result_data.get('essayAnalysis', {})

    # -------------------------------------------------------------------------
    # 2. Build PDF content (header, scores, profile, essay, comments, footer)
    # -------------------------------------------------------------------------
    story.append(_header_table(styles, student_id, country, reviewer_name, now_str))
    story.append(Spacer(1, 14))

    story.append(KeepTogether([
        Paragraph("Score Summary", styles['SectionHeading']),
        _scores_row(pos_score, neg_score, final_score),
    ]))
    story.append(Spacer(1, 6))

    story.append(KeepTogether([
        Paragraph("Admission Recommendation", styles['SectionHeading']),
        _recommendation_block(recommendation, final_score, styles),
    ]))

    # Student profile
    neg_str = ', '.join(neg_factors) if neg_factors else 'None'
    profile_rows = [
        ("Student ID",       student_id),
        ("Country of Origin",country),
        ("Admissions GPA",   f"{gpa}"),
        ("Curriculum Type",  curriculum),
        ("Travel History",   travel),
        ("Essay Word Count", f"~{essay_len} characters"),
        ("Risk Factors",     neg_str),
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

    # Essay analysis
    if essay_analysis:
        story.append(Paragraph("Essay Analysis", styles['SectionHeading']))
        story.append(_essay_metrics_table(essay_analysis, styles))
        story.append(Spacer(1, 6))

        # Insights, strengths, weaknesses
        insights = essay_analysis.get('insights', [])
        strengths = essay_analysis.get('strengths', [])
        weaknesses = essay_analysis.get('weaknesses', [])

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

            essay_notes_tbl = Table([[item] for item in insight_items], colWidths=[7 * inch])
            essay_notes_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('BOX', (0, 0), (-1, -1), 0.5, MID_GRAY),
            ]))
            story.append(essay_notes_tbl)

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
    footer_text = (
        "This document is confidential and intended solely for use by authorized "
        "staff of the WSU Office of International Programs. Generated automatically "
        "by the International Student Scoring System. Report generated on "
        f"{now_str}."
    )
    story.append(Paragraph(footer_text, styles['FooterText']))

    doc.build(story)
    buf.seek(0)
    return buf
