# ---------------------------------------------------
# Sentence Transformers (Embeddings)
# ---------------------------------------------------
# Goal:
# Convert text into vectors and compare meaning
# ---------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        """
        Convert text into embedding vector
        """
        return self.model.encode(text)

    def similarity(self, text1, text2):
        """
        Compute cosine similarity between two texts
        """

        emb1 = self.encode(text1)
        emb2 = self.encode(text2)

        cosine_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )

        return float(cosine_sim)