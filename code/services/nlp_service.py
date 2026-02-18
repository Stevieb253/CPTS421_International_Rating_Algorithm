from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import numpy as np

class NLPService:
    def __init__(self):
      # Existing sentiment and embedding models
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Hugging Face essay scoring model
        model_name = "yjernite/bart_eli5"  # replace with your private repo if different
        HF_TOKEN = "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"

        # Tokenizer
        self.essay_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HF_TOKEN
        )

        # Model
        self.essay_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            token=HF_TOKEN  # <- new recommended argument
        )

    def analyze_sentiment(self, text: str):
        result = self.sentiment_model(text[:512])
        return result[0]

    def compute_similarity(self, text1: str, text2: str):
        emb1 = self.embedding_model.encode([text1])
        emb2 = self.embedding_model.encode([text2])
        score = cosine_similarity(emb1, emb2)[0][0]
        return float(score)

    def score_essay(self, essay_text: str):
        """
        Returns both single holistic score and rubric-based scores:
        - Clarity & Focus
        - Development & Organization
        - Creativity & Style
        """
        # Tokenize and feed into model
        inputs = self.essay_tokenizer(
            essay_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        with torch.no_grad():
            outputs = self.essay_model(**inputs)
            logits = outputs.logits.squeeze(0).numpy()  # shape: (num_classes,)
        
        # Convert logits to scores between 0-100
        scores = 100 * (1 / (1 + np.exp(-logits)))  # simple sigmoid normalization

        # Example mapping for separate rubric scores (adjust based on model config)
        essay_analysis = {
            "clarityFocus": float(scores[0]),
            "developmentOrganization": float(scores[1]),
            "creativityStyle": float(scores[2]),
            "essayRubricScore": float(np.mean(scores)),  
            "grammarScore": 0,       
            "coherenceScore": 0,     
            "vocabularyRichness": 0, 
            "insights": [],
            "strengths": [],
            "weaknesses": [],
        }
        return essay_analysis
