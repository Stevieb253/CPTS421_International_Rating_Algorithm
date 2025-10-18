"""
International Student Applicant Scoring System
Converts applicant data into enrollment prediction scores using AI-powered essay analysis
"""

import requests
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Configuration
HF_API_KEY = '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'
HF_API_URL = 'https://api-inference.huggingface.co/models/'

# Scoring rubrics
CURRICULUM_MAP = {
    'N/A': 1,
    'Standard Intl Secondary': 2,
    'International University/HS English MOI': 3,
    'IGCSE/IB': 4,
    'US HS/University': 5
}

TRAVEL_MAP = {
    'No travel abroad': 1,
    '1 non-listed': 2,
    '1 listed or multiple non-listed': 3,
    'Multiple listed': 4,
    'SEVIS/Multiple US trips': 5
}

NEG_DEDUCTIONS = {
    'reqAppFeeWaiver': 1,
    'cannotPayFee': 1,
    'reqEnrollmentFeeWaiver': 1,
    'bankDocsPending': 1,
    'earlyI20': 1
}


@dataclass
class EssayAnalysis:
    """Container for essay analysis results"""
    clarity_focus: float  # 0-10 scale
    development_organization: float  # 0-10 scale
    creativity_style: float  # 0-10 scale
    total_score: float  # Sum of all three (0-30)
    insights: List[str]


@dataclass
class StudentScore:
    """Container for complete student scoring"""
    pos_score: float
    neg_score: float
    final_score: float
    breakdown: Dict[str, float]
    essay_analysis: EssayAnalysis
    recommendation: str


class HuggingFaceClient:
    """Handles API calls to Hugging Face"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def call_model(self, model: str, payload: dict) -> dict:
        """Make API call to Hugging Face model"""
        url = f"{HF_API_URL}{model}"
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"HF API error: {e}")


class EssayAnalyzer:
    """Analyzes student essays using AI and local methods"""
    
    def __init__(self, hf_client: HuggingFaceClient):
        self.hf_client = hf_client
    
    def analyze_essay_ai(self, text: str) -> EssayAnalysis:
        """Analyze essay using Hugging Face AI models and rubric criteria"""
        if not text or len(text) < 50:
            return EssayAnalysis(
                clarity_focus=0,
                development_organization=0,
                creativity_style=0,
                total_score=0,
                insights=['No essay provided or essay too short (minimum 50 characters)']
            )
        
        try:
            # Limit text length for API
            input_text = text[:2000]
            
            # Sentiment analysis
            sentiment_result = self.hf_client.call_model(
                'cardiffnlp/twitter-roberta-base-sentiment-latest',
                {'inputs': input_text}
            )
            
            # Extract positive score
            positive_score = 0.5  # default
            if isinstance(sentiment_result, list) and sentiment_result:
                for item in sentiment_result[0]:
                    if item.get('label') == 'POSITIVE':
                        positive_score = item.get('score', 0.5)
                        break
            
            # Text analysis metrics
            words = text.strip().split()
            word_count = len(words)
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            sentence_count = len(sentences)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            paragraph_count = len(paragraphs)
            
            # Average words per sentence
            avg_words_per_sentence = word_count / max(sentence_count, 1)
            
            # Vocabulary richness (unique words ratio)
            unique_words = len(set(w.lower() for w in words if len(w) > 3))
            vocab_richness = unique_words / max(word_count, 1)
            
            # Key phrase analysis
            focus_keywords = r'\b(goal|objective|purpose|aim|plan|intend|aspire|pursue|achieve|vision)\b'
            development_keywords = r'\b(because|therefore|however|furthermore|moreover|additionally|specifically|example|demonstrate|evidence)\b'
            creativity_keywords = r'\b(innovative|unique|creative|original|perspective|approach|solution|transform|inspire|passion)\b'
            
            focus_count = len(re.findall(focus_keywords, text, re.IGNORECASE))
            development_count = len(re.findall(development_keywords, text, re.IGNORECASE))
            creativity_count = len(re.findall(creativity_keywords, text, re.IGNORECASE))
            
            # Cliché detection (common overused phrases)
            cliches = r'\b(since i was a child|ever since|always dreamed|passion for|make a difference|give back|reach my full potential)\b'
            cliche_count = len(re.findall(cliches, text, re.IGNORECASE))
            
            # Domain-specific vocabulary (academic/field-specific terms)
            domain_vocab = r'\b(research|analysis|methodology|theory|hypothesis|innovation|implementation|framework|paradigm|discipline)\b'
            domain_count = len(re.findall(domain_vocab, text, re.IGNORECASE))
            
            # === SCORE 1: CLARITY & FOCUS (0-10) ===
            clarity_base = 1.0  # Start at 1
            
            # Length contribution (longer essays generally show more focus)
            if word_count >= 300:
                clarity_base += 2.0
            elif word_count >= 200:
                clarity_base += 1.5
            elif word_count >= 100:
                clarity_base += 1.0
            
            # Focus keywords indicate clear purpose
            clarity_base += min(2.0, focus_count * 0.4)
            
            # Positive sentiment indicates engagement
            clarity_base += positive_score * 2.5
            
            # Paragraph structure shows organization
            if paragraph_count >= 3:
                clarity_base += 1.5
            elif paragraph_count >= 2:
                clarity_base += 1.0
            
            # Self-awareness indicators (first person + reflection)
            self_awareness = len(re.findall(r'\b(I (learned|realized|understand|believe|recognize|grew|developed))\b', text, re.IGNORECASE))
            clarity_base += min(1.0, self_awareness * 0.3)
            
            clarity_focus = min(10.0, clarity_base)
            
            # === SCORE 2: DEVELOPMENT & ORGANIZATION (0-10) ===
            development_base = 1.0
            
            # Sentence variety (ideal range 15-25 words per sentence)
            if 15 <= avg_words_per_sentence <= 25:
                development_base += 2.0
            elif 10 <= avg_words_per_sentence <= 30:
                development_base += 1.5
            else:
                development_base += 1.0
            
            # Transitional phrases and logical connectors
            development_base += min(2.5, development_count * 0.3)
            
            # Sufficient detail (length and vocab richness)
            if word_count >= 250 and vocab_richness > 0.4:
                development_base += 2.0
            elif word_count >= 150:
                development_base += 1.5
            
            # Paragraph count indicates structure
            if paragraph_count >= 4:
                development_base += 1.5
            elif paragraph_count >= 3:
                development_base += 1.0
            
            # Cohesion (longer sentences suggest complex thought)
            if sentence_count >= 8:
                development_base += 1.0
            
            development_organization = min(10.0, development_base)
            
            # === SCORE 3: CREATIVITY & STYLE (0-10) ===
            creativity_base = 1.0
            
            # Vocabulary richness (unique word usage)
            if vocab_richness > 0.5:
                creativity_base += 2.5
            elif vocab_richness > 0.4:
                creativity_base += 2.0
            elif vocab_richness > 0.3:
                creativity_base += 1.5
            
            # Creative keywords indicate innovation
            creativity_base += min(2.0, creativity_count * 0.4)
            
            # Domain-specific vocabulary shows depth
            creativity_base += min(2.0, domain_count * 0.4)
            
            # Sentence variety (more sentences = more varied structure)
            if sentence_count >= 12:
                creativity_base += 1.5
            elif sentence_count >= 8:
                creativity_base += 1.0
            
            # Penalty for clichés
            creativity_base -= min(2.0, cliche_count * 0.5)
            
            # Positive sentiment enhances style
            creativity_base += positive_score * 1.0
            
            creativity_style = max(1.0, min(10.0, creativity_base))
            
            # Total essay score (out of 30)
            total_score = clarity_focus + development_organization + creativity_style
            
            # Generate insights
            sentiment_label = 'Positive' if positive_score > 0.6 else ('Neutral' if positive_score > 0.4 else 'Negative')
            
            insights = [
                f"Essay Length: {word_count} words, {sentence_count} sentences, {paragraph_count} paragraphs",
                f"Sentiment: {sentiment_label} ({int(positive_score*100)}% positive)",
                f"Vocabulary Richness: {int(vocab_richness*100)}% unique words",
                f"Avg Sentence Length: {avg_words_per_sentence:.1f} words",
                f"Focus Keywords: {focus_count} | Development Keywords: {development_count} | Creative Keywords: {creativity_count}",
                f"Clichés Detected: {cliche_count} | Domain Vocabulary: {domain_count}"
            ]
            
            # Add qualitative insights
            if clarity_focus >= 7.5:
                insights.append("✓ Strong clarity and focused central message")
            elif clarity_focus >= 5:
                insights.append("⚠ Adequate focus but could be more engaging")
            else:
                insights.append("⚠ Needs clearer central perspective and focus")
            
            if development_organization >= 7.5:
                insights.append("✓ Well-organized with strong logical flow")
            elif development_organization >= 5:
                insights.append("⚠ Basic organization present but could be enhanced")
            else:
                insights.append("⚠ Needs better structure and supporting details")
            
            if creativity_style >= 7.5:
                insights.append("✓ Creative and engaging with varied style")
            elif creativity_style >= 5:
                insights.append("⚠ Some creativity but contains common phrases")
            else:
                insights.append("⚠ Needs more originality and varied sentence structure")
            
            return EssayAnalysis(
                clarity_focus=round(clarity_focus, 2),
                development_organization=round(development_organization, 2),
                creativity_style=round(creativity_style, 2),
                total_score=round(total_score, 2),
                insights=insights
            )
        
        except Exception as e:
            # Fallback to local analysis
            print(f"AI analysis failed, using local method: {e}")
            return self.analyze_essay_locally(text)
    
    def analyze_essay_locally(self, text: str) -> EssayAnalysis:
        """Fallback local essay analysis without AI"""
        words = text.strip().split()
        word_count = len(words)
        sentences = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        
        # Simple scoring based on length and basic keywords
        clarity = min(10, 1 + (word_count / 50))
        development = min(10, 1 + (sentences / 2))
        creativity = min(10, 1 + (word_count / 60))
        
        total = clarity + development + creativity
        
        return EssayAnalysis(
            clarity_focus=round(clarity, 2),
            development_organization=round(development, 2),
            creativity_style=round(creativity, 2),
            total_score=round(total, 2),
            insights=[f"Local analysis: {word_count} words, {sentences} sentences (AI unavailable)"]
        )


class StudentAnalyzer:
    """Main analyzer for student applications"""
    
    def __init__(self, api_key: str = HF_API_KEY):
        self.hf_client = HuggingFaceClient(api_key)
        self.essay_analyzer = EssayAnalyzer(self.hf_client)
    
    def analyze_student(
        self,
        gpa: float,
        curriculum: str,
        travel_history: str,
        essay_text: str,
        neg_factors: List[str] = None
    ) -> StudentScore:
        """
        Analyze a student application and return comprehensive scoring
        
        Args:
            gpa: Student GPA (0-5 scale typically)
            curriculum: Type of curriculum (must match CURRICULUM_MAP keys)
            travel_history: Travel history category (must match TRAVEL_MAP keys)
            essay_text: Student's essay text
            neg_factors: List of negative factor keys from NEG_DEDUCTIONS
        
        Returns:
            StudentScore object with complete analysis
        """
        # Calculate negative score
        neg_score = 0
        if neg_factors:
            for factor in neg_factors:
                neg_score += NEG_DEDUCTIONS.get(factor, 0)
        
        # Analyze essay
        essay_analysis = self.essay_analyzer.analyze_essay_ai(essay_text)
        
        # Calculate POS score components
        gpa_score = min(5, gpa)
        curriculum_score = CURRICULUM_MAP.get(curriculum, 0)
        travel_score = TRAVEL_MAP.get(travel_history, 0)
        
        # Essay score is now out of 30, normalize to fit POS scale
        # We'll use the total essay score (0-30) directly as it represents comprehensive evaluation
        essay_total = essay_analysis.total_score
        
        # Total POS score (GPA:5 + Curriculum:5 + Travel:5 + Essay:30 = max 45)
        pos_score = (
            gpa_score +
            curriculum_score +
            travel_score +
            essay_total
        )
        
        # Final composite score
        final_score = max(0, pos_score - neg_score)
        
        # Determine recommendation (adjusted thresholds for new scale)
        if final_score >= 30:  # Excellent (high POS, low/no NEG)
            recommendation = "HIGHLY RECOMMENDED - Strong enrollment potential."
        elif final_score >= 20:  # Good
            recommendation = "RECOMMENDED WITH MONITORING - Moderate enrollment potential."
        elif final_score >= 15:  # Fair
            recommendation = "CONDITIONAL - Requires additional review and monitoring."
        else:  # Below threshold
            recommendation = "HIGH RISK - Requires extensive manual review."
        
        # Build breakdown
        breakdown = {
            'GPA Score': gpa_score,
            'Curriculum Score': curriculum_score,
            'Travel Score': travel_score,
            'Essay - Clarity & Focus': essay_analysis.clarity_focus,
            'Essay - Development & Organization': essay_analysis.development_organization,
            'Essay - Creativity & Style': essay_analysis.creativity_style,
            'Essay Total Score': essay_analysis.total_score,
            'NEG Deductions': neg_score
        }
        
        return StudentScore(
            pos_score=round(pos_score, 2),
            neg_score=round(neg_score, 2),
            final_score=round(final_score, 2),
            breakdown=breakdown,
            essay_analysis=essay_analysis,
            recommendation=recommendation
        )
    
    def print_analysis(self, score: StudentScore) -> None:
        """Pretty print analysis results"""
        print("\n" + "="*60)
        print("STUDENT APPLICATION ANALYSIS")
        print("="*60)
        print(f"\nPOS Score: {score.pos_score}")
        print(f"NEG Score: {score.neg_score}")
        print(f"Final Score: {score.final_score}")
        
        print("\n--- Factor Breakdown ---")
        for factor, value in score.breakdown.items():
            print(f"  {factor}: {value}")
        
        print("\n--- AI Essay Insights ---")
        for insight in score.essay_analysis.insights:
            print(f"  • {insight}")
        
        print(f"\n--- Recommendation ---")
        print(f"  {score.recommendation}")
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    analyzer = StudentAnalyzer()
    
    # Example student data
    result = analyzer.analyze_student(
        gpa=3.8,
        curriculum='IGCSE/IB',
        travel_history='Multiple listed',
        essay_text="""
        I am deeply passionate about pursuing my educational goals at Washington State University.
        Throughout my academic career, I have demonstrated strong motivation and dedication to achieving 
        excellence in my field. My dream is to contribute to the field of computer science and use my 
        skills to make a positive impact on society. I aspire to become a leader in technology and 
        innovation, and I believe WSU provides the perfect environment to help me achieve these goals.
        """,
        neg_factors=['bankDocsPending']
    )
    
    analyzer.print_analysis(result)