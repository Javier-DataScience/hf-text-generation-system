# ---------------------------------------------------
# MINI NLP SYSTEM (Hugging Face Lab)
# ---------------------------------------------------
# Combines:
# - Sentiment Analysis
# - Sentence Embeddings
# - Simple NLP Orchestrator
# ---------------------------------------------------

from transformers import pipeline
from src.embedding_service import EmbeddingService


class MiniNLPSystem:

    def __init__(self):

        # Sentiment model (Hugging Face pipeline)
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        # Embedding service (Step 5)
        self.embedding_service = EmbeddingService()

    def analyze_sentiment(self, text):
        """
        Returns sentiment label + confidence score
        """
        result = self.sentiment_model(text)[0]

        return {
            "label": result["label"],
            "score": float(result["score"])
        }

    def similarity(self, text1, text2):
        """
        Semantic similarity using embeddings
        """
        return self.embedding_service.similarity(text1, text2)

    def analyze(self, text, compare_text=None):
        """
        Full NLP analysis pipeline
        """

        result = {
            "input_text": text,
            "sentiment": self.analyze_sentiment(text)
        }

        # Optional similarity check
        if compare_text:
            result["comparison_text"] = compare_text
            result["similarity"] = self.similarity(text, compare_text)

        return result