"""
Enhanced International Student Applicant Scoring System
Replaces your existing student_analyzer.py with advanced AI models

Models Used:
- facebook/bart-large-mnli: Zero-shot essay quality classification
- textattack/roberta-base-CoLA: Grammar quality assessment  
- distilbert-base-uncased-finetuned-sst-2-english: Sentiment analysis
"""
import requests
import re
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Configuration
HF_API_KEY = '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'
HF_API_URL = 'https://api-inference.huggingface.co/models/'

# Scoring rubrics - ALIGNED WITH WSU RUBRIC
GPA_RANGES = [
    (2.0, 2.3, 0.25),
    (2.3, 2.5, 0.5),
    (2.5, 3.0, 0.75),
    (3.0, 3.6, 1.0),
    (3.6, 5.0, 1.25)
]

CURRICULUM_SCORES = {
    'N/A': 0.25,
    'Standard Intl Secondary': 0.5,
    'International University/HS English MOI': 0.75,
    'IGCSE/IB': 1.0,
    'US HS/University': 1.25
}

TRAVEL_SCORES = {
    'No travel abroad': 0.25,
    '1 non-listed': 0.5,
    '1 listed or multiple non-listed': 0.75,
    'Multiple listed': 1.0,
    'SEVIS/Multiple US trips': 1.25
}

ESSAY_RANGES = [
    (0, 11, 0.25),
    (12, 14, 0.5),
    (15, 17, 0.75),
    (18, 20, 1.0),
    (21, 24, 1.25)
]

NEG_DEDUCTIONS = {
    'reqAppFeeWaiver': 1,
    'cannotPayFee': 1,
    'reqEnrollmentFeeWaiver': 1,
    'bankDocsPending': 1,
    'earlyI20': 1
}

@dataclass
class EssayAnalysis:
    """Enhanced essay analysis with detailed metrics"""
    # Main rubric scores (out of 10 each)
    clarity_focus: float
    development_organization: float
    creativity_style: float
    
    # Derived scores
    total_score: float  # Out of 30
    rubric_score: float  # Mapped to 24-point scale
    weighted_score: float  # Final weighted (0.25-1.25)
    
    # Detailed sub-metrics
    grammar_score: float  # 0-100
    coherence_score: float  # 0-100
    authenticity_score: float  # 0-100 (placeholder for AI detection)
    vocabulary_richness: float  # 0-100
    
    # Qualitative insights
    insights: List[str]
    strengths: List[str]
    weaknesses: List[str]
    
    # AI confidence
    analysis_confidence: float  # 0-1

@dataclass
class StudentScore:
    """Complete student evaluation"""
    pos_score: float
    neg_score: float
    final_score: float
    breakdown: Dict[str, float]
    essay_analysis: EssayAnalysis
    recommendation: str
    rank_estimate: int
    overall_confidence: float

class HuggingFaceClient:
    """Handles API calls to Hugging Face with retry logic"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.timeout = 30
    
    def call_model(self, model: str, payload: dict, max_retries: int = 2) -> dict:
        """Make API call with retry logic"""
        url = f"{HF_API_URL}{model}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    print(f"⚠️ Model {model} timed out after {max_retries} attempts")
                    return {}
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"⚠️ HF API error for {model}: {e}")
                    return {}
        
        return {}

class EssayAnalyzer:
    """
    Enhanced multi-model essay analyzer
    Uses state-of-the-art NLP models for comprehensive evaluation
    """
    
    def __init__(self, hf_client: HuggingFaceClient):
        self.hf_client = hf_client
        
        # Model names
        self.zero_shot_model = 'facebook/bart-large-mnli'
        self.sentiment_model = 'distilbert-base-uncased-finetuned-sst-2-english'
        self.grammar_model = 'textattack/roberta-base-CoLA'
    
    def extract_text_features(self, text: str) -> Dict:
        """Extract comprehensive text features"""
        words = text.strip().split()
        word_count = len(words)
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        unique_words = set(w.lower() for w in words if len(w) > 3)
        vocab_richness = len(unique_words) / max(word_count, 1)
        
        word_lengths = [len(w) for w in words]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        # Flesch readability score (approximation)
        syllables_per_word = avg_word_length / 3
        flesch_score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * syllables_per_word
        flesch_score = max(0, min(100, flesch_score))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_words_per_sentence': avg_words_per_sentence,
            'vocab_richness': vocab_richness,
            'avg_word_length': avg_word_length,
            'flesch_score': flesch_score,
            'unique_word_count': len(unique_words)
        }
    
    def analyze_with_zero_shot(self, text: str) -> Dict[str, float]:
        """
        Use BART zero-shot classification for essay quality dimensions
        """
        try:
            labels = [
                "well-organized and clear",
                "poorly structured and unclear",
                "creative and original",
                "generic and cliché",
                "focused and purposeful",
                "unfocused and rambling",
                "academically sophisticated",
                "simplistic and basic",
                "emotionally engaging",
                "emotionally flat"
            ]
            
            input_text = text[:1000]
            
            result = self.hf_client.call_model(
                self.zero_shot_model,
                {
                    'inputs': input_text,
                    'parameters': {'candidate_labels': labels}
                }
            )
            
            scores = {}
            if isinstance(result, dict) and 'labels' in result:
                for label, score in zip(result['labels'], result['scores']):
                    scores[label] = score
            
            # Calculate dimension scores
            clarity_score = scores.get("well-organized and clear", 0.5) - scores.get("poorly structured and unclear", 0)
            creativity_score = scores.get("creative and original", 0.5) - scores.get("generic and cliché", 0)
            focus_score = scores.get("focused and purposeful", 0.5) - scores.get("unfocused and rambling", 0)
            sophistication_score = scores.get("academically sophisticated", 0.5) - scores.get("simplistic and basic", 0)
            engagement_score = scores.get("emotionally engaging", 0.5) - scores.get("emotionally flat", 0)
            
            # Normalize to 0-1
            clarity_score = max(0, min(1, (clarity_score + 1) / 2))
            creativity_score = max(0, min(1, (creativity_score + 1) / 2))
            focus_score = max(0, min(1, (focus_score + 1) / 2))
            sophistication_score = max(0, min(1, (sophistication_score + 1) / 2))
            engagement_score = max(0, min(1, (engagement_score + 1) / 2))
            
            return {
                'clarity': clarity_score,
                'creativity': creativity_score,
                'focus': focus_score,
                'sophistication': sophistication_score,
                'engagement': engagement_score
            }
        
        except Exception as e:
            print(f"Zero-shot analysis failed: {e}")
            return {
                'clarity': 0.5,
                'creativity': 0.5,
                'focus': 0.5,
                'sophistication': 0.5,
                'engagement': 0.5
            }
    
    def analyze_grammar(self, text: str) -> Tuple[float, List[str]]:
        """Assess grammatical quality using CoLA model"""
        try:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            if len(sentences) > 10:
                step = len(sentences) // 10
                sentences = sentences[::step]
            
            grammar_scores = []
            issues = []
            
            for sent in sentences[:10]:
                result = self.hf_client.call_model(
                    self.grammar_model,
                    {'inputs': sent}
                )
                
                if isinstance(result, list) and result:
                    for item in result[0]:
                        if item.get('label') == 'acceptable':
                            score = item.get('score', 0.5)
                            grammar_scores.append(score)
                            if score < 0.6:
                                issues.append(f"Sentence may have issues: '{sent[:50]}...'")
            
            avg_grammar = np.mean(grammar_scores) if grammar_scores else 0.7
            return avg_grammar * 100, issues
        
        except Exception as e:
            print(f"Grammar analysis failed: {e}")
            return 70.0, []
    
    def analyze_sentiment_depth(self, text: str) -> Dict[str, float]:
        """Analyze sentiment for engagement assessment"""
        try:
            input_text = text[:512]
            
            result = self.hf_client.call_model(
                self.sentiment_model,
                {'inputs': input_text}
            )
            
            positive_score = 0.5
            if isinstance(result, list) and result:
                for item in result[0]:
                    if item.get('label') == 'POSITIVE':
                        positive_score = item.get('score', 0.5)
            
            return {
                'positive_score': positive_score,
                'engagement_indicator': positive_score
            }
        
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            return {'positive_score': 0.5, 'engagement_indicator': 0.5}
    
    def detect_keyword_patterns(self, text: str) -> Dict[str, int]:
        """Detect important keyword patterns"""
        patterns = {
            'focus_keywords': r'\b(goal|objective|purpose|aim|plan|intend|aspire|pursue|achieve|vision|mission|dream)\b',
            'development_keywords': r'\b(because|therefore|however|furthermore|moreover|additionally|specifically|example|demonstrate|evidence|consequently|thus)\b',
            'creativity_keywords': r'\b(innovative|unique|creative|original|perspective|approach|solution|transform|inspire|passion|reimagine|pioneer)\b',
            'academic_keywords': r'\b(research|analysis|methodology|theory|hypothesis|innovation|implementation|framework|paradigm|discipline|critical|evaluate)\b',
            'leadership_keywords': r'\b(lead|leadership|initiative|organize|coordinate|mentor|collaborate|teamwork|responsibility)\b',
            'reflection_keywords': r'\b(learned|realized|understand|recognize|grew|developed|discovered|reflected|appreciated|transformed)\b',
            'cliches': r'\b(since i was a child|ever since|always dreamed|make a difference|give back|reach my full potential|follow my passion)\b',
            'transition_words': r'\b(first|second|third|finally|meanwhile|subsequently|in addition|on the other hand|in conclusion)\b'
        }
        
        counts = {}
        for key, pattern in patterns.items():
            counts[key] = len(re.findall(pattern, text, re.IGNORECASE))
        
        return counts
    
    def calculate_coherence_score(self, text: str) -> float:
        """Calculate paragraph-to-paragraph coherence using lexical overlap"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 50.0
        
        coherence_scores = []
        
        for i in range(len(paragraphs) - 1):
            words1 = set(paragraphs[i].lower().split())
            words2 = set(paragraphs[i + 1].lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                coherence_scores.append(overlap)
        
        return np.mean(coherence_scores) * 100 if coherence_scores else 50.0
    
    def map_essay_score_to_rubric(self, raw_score: float) -> Tuple[float, float]:
        """Map 30-point score to 24-point rubric and weighted score"""
        rubric_score = (raw_score / 30.0) * 24.0
        
        for min_score, max_score, weight in ESSAY_RANGES:
            if min_score <= rubric_score <= max_score:
                return round(rubric_score, 2), weight
        
        if rubric_score < 12:
            return round(rubric_score, 2), 0.25
        return 24.0, 1.25
    
    def analyze_essay_ai(self, text: str) -> EssayAnalysis:
        """
        Comprehensive essay analysis using multiple AI models
        """
        if not text or len(text) < 50:
            return EssayAnalysis(
                clarity_focus=1.0,
                development_organization=1.0,
                creativity_style=1.0,
                total_score=3.0,
                rubric_score=2.4,
                weighted_score=0.25,
                grammar_score=0.0,
                coherence_score=0.0,
                authenticity_score=100.0,
                vocabulary_richness=0.0,
                insights=['Essay too short (minimum 50 characters)'],
                strengths=[],
                weaknesses=['Essay length insufficient'],
                analysis_confidence=1.0
            )
        
        # Extract basic features
        features = self.extract_text_features(text)
        
        # Run AI models
        zero_shot_results = self.analyze_with_zero_shot(text)
        grammar_score, grammar_issues = self.analyze_grammar(text)
        sentiment_results = self.analyze_sentiment_depth(text)
        keyword_counts = self.detect_keyword_patterns(text)
        coherence_score = self.calculate_coherence_score(text)
        
        # === SCORE 1: CLARITY & FOCUS (0-10) ===
        clarity_base = 1.0
        
        clarity_base += zero_shot_results['clarity'] * 3.0
        clarity_base += zero_shot_results['focus'] * 2.0
        
        if features['word_count'] >= 300:
            clarity_base += 1.5
        elif features['word_count'] >= 200:
            clarity_base += 1.0
        elif features['word_count'] >= 100:
            clarity_base += 0.5
        
        if features['paragraph_count'] >= 4:
            clarity_base += 1.0
        elif features['paragraph_count'] >= 3:
            clarity_base += 0.7
        elif features['paragraph_count'] >= 2:
            clarity_base += 0.4
        
        clarity_base += min(1.0, keyword_counts['focus_keywords'] * 0.15)
        
        clarity_focus = min(10.0, clarity_base)
        
        # === SCORE 2: DEVELOPMENT & ORGANIZATION (0-10) ===
        development_base = 1.0
        
        development_base += zero_shot_results['sophistication'] * 2.5
        development_base += zero_shot_results['clarity'] * 1.5
        
        if 15 <= features['avg_words_per_sentence'] <= 25:
            development_base += 1.5
        elif 10 <= features['avg_words_per_sentence'] <= 30:
            development_base += 1.0
        else:
            development_base += 0.5
        
        development_base += min(1.5, keyword_counts['development_keywords'] * 0.12)
        development_base += min(0.5, keyword_counts['transition_words'] * 0.1)
        development_base += (coherence_score / 100) * 1.0
        
        development_organization = min(10.0, development_base)
        
        # === SCORE 3: CREATIVITY & STYLE (0-10) ===
        creativity_base = 1.0
        
        creativity_base += zero_shot_results['creativity'] * 2.0
        creativity_base += zero_shot_results['engagement'] * 2.0
        
        vocab_score = features['vocab_richness']
        if vocab_score > 0.5:
            creativity_base += 2.0
        elif vocab_score > 0.4:
            creativity_base += 1.5
        elif vocab_score > 0.3:
            creativity_base += 1.0
        else:
            creativity_base += 0.5
        
        creativity_base += min(1.2, keyword_counts['creativity_keywords'] * 0.15)
        creativity_base += min(0.8, keyword_counts['academic_keywords'] * 0.1)
        creativity_base -= min(1.5, keyword_counts['cliches'] * 0.5)
        creativity_base += (grammar_score / 100) * 0.5
        
        creativity_style = max(1.0, min(10.0, creativity_base))
        
        # Total and mapped scores
        total_score = clarity_focus + development_organization + creativity_style
        rubric_score, weighted_score = self.map_essay_score_to_rubric(total_score)
        
        # Calculate confidence
        confidence = min(1.0, 0.7 + (features['word_count'] / 500) * 0.3)
        
        # Generate insights
        insights = []
        strengths = []
        weaknesses = []
        
        insights.append(f"📊 Length: {features['word_count']} words, {features['sentence_count']} sentences, {features['paragraph_count']} paragraphs")
        insights.append(f"📈 Vocabulary Richness: {features['vocab_richness']*100:.0f}% | Readability: {features['flesch_score']:.0f}/100")
        insights.append(f"✍️ Avg Sentence Length: {features['avg_words_per_sentence']:.1f} words | Grammar: {grammar_score:.0f}/100")
        insights.append(f"🎯 Rubric Score: {rubric_score:.1f}/24 → Weighted: {weighted_score}")
        insights.append(f"🤖 AI Assessment: Clarity {zero_shot_results['clarity']*100:.0f}% | Creativity {zero_shot_results['creativity']*100:.0f}% | Focus {zero_shot_results['focus']*100:.0f}%")
        insights.append(f"🔑 Keywords: Focus={keyword_counts['focus_keywords']} | Development={keyword_counts['development_keywords']} | Creative={keyword_counts['creativity_keywords']} | Clichés={keyword_counts['cliches']}")
        
        # Strengths
        if clarity_focus >= 7.5:
            strengths.append("✓ Excellent clarity and focused central message")
        elif clarity_focus >= 6.0:
            strengths.append("✓ Good clarity with clear purpose")
        
        if development_organization >= 7.5:
            strengths.append("✓ Well-organized with strong logical flow")
        elif development_organization >= 6.0:
            strengths.append("✓ Solid organization and structure")
        
        if creativity_style >= 7.5:
            strengths.append("✓ Creative and engaging with varied style")
        elif creativity_style >= 6.0:
            strengths.append("✓ Good originality and personal voice")
        
        if grammar_score >= 85:
            strengths.append("✓ Strong grammar and linguistic quality")
        
        if features['vocab_richness'] > 0.45:
            strengths.append("✓ Rich and varied vocabulary")
        
        if keyword_counts['reflection_keywords'] >= 3:
            strengths.append("✓ Demonstrates self-awareness and reflection")
        
        # Weaknesses
        if clarity_focus < 5.0:
            weaknesses.append("⚠ Needs clearer central perspective and focus")
        elif clarity_focus < 7.0:
            weaknesses.append("⚠ Could improve clarity and engagement")
        
        if development_organization < 5.0:
            weaknesses.append("⚠ Needs better structure and supporting details")
        elif development_organization < 7.0:
            weaknesses.append("⚠ Organization could be strengthened")
        
        if creativity_style < 5.0:
            weaknesses.append("⚠ Needs more originality and varied sentence structure")
        elif creativity_style < 7.0:
            weaknesses.append("⚠ Could enhance creativity and style")
        
        if grammar_score < 70:
            weaknesses.append("⚠ Grammar and sentence structure need improvement")
        
        if keyword_counts['cliches'] >= 3:
            weaknesses.append("⚠ Contains multiple clichéd phrases")
        
        if features['word_count'] < 150:
            weaknesses.append("⚠ Essay length is below recommended minimum")
        
        if keyword_counts['development_keywords'] < 2:
            weaknesses.append("⚠ Lacks clear logical connectors and transitions")
        
        return EssayAnalysis(
            clarity_focus=round(clarity_focus, 2),
            development_organization=round(development_organization, 2),
            creativity_style=round(creativity_style, 2),
            total_score=round(total_score, 2),
            rubric_score=rubric_score,
            weighted_score=weighted_score,
            grammar_score=round(grammar_score, 2),
            coherence_score=round(coherence_score, 2),
            authenticity_score=100.0,
            vocabulary_richness=round(features['vocab_richness'] * 100, 2),
            insights=insights,
            strengths=strengths,
            weaknesses=weaknesses,
            analysis_confidence=round(confidence, 2)
        )

class StudentAnalyzer:
    """Main analyzer for student applications with enhanced essay evaluation"""
    
    def __init__(self, api_key: str = HF_API_KEY):
        self.hf_client = HuggingFaceClient(api_key)
        self.essay_analyzer = EssayAnalyzer(self.hf_client)
    
    def get_gpa_score(self, gpa: float) -> float:
        """Get GPA score based on ranges"""
        for min_gpa, max_gpa, score in GPA_RANGES:
            if min_gpa <= gpa < max_gpa:
                return score
        if gpa >= 3.6:
            return 1.25
        return 0.25
    
    def analyze_student(
        self,
        gpa: float,
        curriculum: str,
        travel_history: str,
        essay_text: str,
        neg_factors: List[str] = None
    ) -> StudentScore:
        """Comprehensive student evaluation with enhanced essay analysis"""
        
        neg_score = 0
        if neg_factors:
            for factor in neg_factors:
                neg_score += NEG_DEDUCTIONS.get(factor, 0)
        
        essay_analysis = self.essay_analyzer.analyze_essay_ai(essay_text)
        
        gpa_score = self.get_gpa_score(gpa)
        curriculum_score = CURRICULUM_SCORES.get(curriculum, 0.25)
        travel_score = TRAVEL_SCORES.get(travel_history, 0.25)
        essay_score = essay_analysis.weighted_score
        
        base_score = 5.0
        pos_score = base_score + gpa_score + curriculum_score + travel_score + essay_score
        final_score = max(0, pos_score - neg_score)
        overall_confidence = essay_analysis.analysis_confidence
        
        if final_score >= 7.5:
            recommendation = "HIGHLY RECOMMENDED - Strong enrollment potential."
            rank_estimate = 1
        elif final_score >= 7.0:
            recommendation = "RECOMMENDED - Good enrollment potential."
            rank_estimate = 2
        elif final_score >= 6.5:
            recommendation = "RECOMMENDED WITH MONITORING - Moderate enrollment potential."
            rank_estimate = 3
        elif final_score >= 6.0:
            recommendation = "CONDITIONAL - Requires additional review."
            rank_estimate = 4
        elif final_score >= 5.5:
            recommendation = "BORDERLINE - Requires careful evaluation."
            rank_estimate = 5
        else:
            recommendation = "HIGH RISK - Requires extensive manual review."
            rank_estimate = 6
        
        breakdown = {
            'Base Score': base_score,
            'GPA Score': gpa_score,
            'Curriculum Score': curriculum_score,
            'Travel Score': travel_score,
            'Essay Weighted Score': essay_score,
            'Essay - Clarity & Focus': essay_analysis.clarity_focus,
            'Essay - Development & Organization': essay_analysis.development_organization,
            'Essay - Creativity & Style': essay_analysis.creativity_style,
            'Essay Raw Score (out of 30)': essay_analysis.total_score,
            'Essay Rubric Score (out of 24)': essay_analysis.rubric_score,
            'Grammar Score': essay_analysis.grammar_score,
            'Coherence Score': essay_analysis.coherence_score,
            'Vocabulary Richness': essay_analysis.vocabulary_richness,
            'NEG Deductions': neg_score
        }
        
        return StudentScore(
            pos_score=round(pos_score, 2),
            neg_score=round(neg_score, 2),
            final_score=round(final_score, 2),
            breakdown=breakdown,
            essay_analysis=essay_analysis,
            recommendation=recommendation,
            rank_estimate=rank_estimate,
            overall_confidence=overall_confidence
        )
    
    def print_analysis(self, score: StudentScore) -> None:
        """Enhanced pretty print with detailed insights"""
        print("\n" + "="*70)
        print("ENHANCED STUDENT APPLICATION ANALYSIS")
        print("="*70)
        print(f"\n🎯 POS Score: {score.pos_score} | NEG Score: -{score.neg_score} | Final: {score.final_score}")
        print(f"📊 Rank Estimate: #{score.rank_estimate} | Confidence: {score.overall_confidence*100:.0f}%")
        
        print("\n--- Factor Breakdown ---")
        for factor, value in score.breakdown.items():
            print(f"  {factor}: {value if isinstance(value, str) else f'{value:.2f}'}")
        
        print("\n--- Essay Analysis ---")
        print(f"  Clarity & Focus: {score.essay_analysis.clarity_focus}/10")
        print(f"  Development & Organization: {score.essay_analysis.development_organization}/10")
        print(f"  Creativity & Style: {score.essay_analysis.creativity_style}/10")
        print(f"  Grammar Score: {score.essay_analysis.grammar_score}/100")
        print(f"  Coherence: {score.essay_analysis.coherence_score}/100")
        print(f"  Vocabulary Richness: {score.essay_analysis.vocabulary_richness}%")
        
        print("\n--- AI Insights ---")
        for insight in score.essay_analysis.insights:
            print(f"  • {insight}")
        
        if score.essay_analysis.strengths:
            print("\n--- Strengths ---")
            for strength in score.essay_analysis.strengths:
                print(f"  {strength}")
        
        if score.essay_analysis.weaknesses:
            print("\n--- Areas for Improvement ---")
            for weakness in score.essay_analysis.weaknesses:
                print(f"  {weakness}")
        
        print(f"\n--- Recommendation ---")
        print(f"  {score.recommendation}")
        print("="*70 + "\n")

# Example usage
if __name__ == "__main__":
    analyzer = StudentAnalyzer()
    
    # Test case
    result = analyzer.analyze_student(
        gpa=3.5,
        curriculum='IGCSE/IB',
        travel_history='Multiple listed',
        essay_text="""
        Throughout my academic journey, I have developed a profound passion for computer science 
        and artificial intelligence. My experiences in leading the robotics team at my high school 
        taught me the importance of collaboration, innovation, and perseverance. I successfully 
        organized three international STEM workshops that brought together students from diverse 
        backgrounds to explore cutting-edge technology.
        
        I am particularly drawn to Washington State University because of its renowned research 
        programs in AI and its commitment to fostering a collaborative learning environment. I aspire
        to contribute to groundbreaking research that addresses real-world challenges, particularly
        in healthcare and environmental sustainability. My long-term goal is to pioneer innovative
        solutions that leverage AI to improve quality of life globally.
        """,
        neg_factors=['reqAppFeeWaiver']
    )
    analyzer.print_analysis(result)
