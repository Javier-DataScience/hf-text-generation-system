# ---------------------------------------------------
# EMBEDDING DEMO
# ---------------------------------------------------

from src.embedding_service import EmbeddingService


def main():

    service = EmbeddingService()

    sentences = [
        "I love artificial intelligence",
        "I enjoy studying machine learning",
        "The weather is very cold today"
    ]

    print("\n=== EMBEDDING SIMILARITY DEMO ===\n")

    print("Sentence 1 vs 2:")
    print(service.similarity(sentences[0], sentences[1]))

    print("\nSentence 1 vs 3:")
    print(service.similarity(sentences[0], sentences[2]))


if __name__ == "__main__":
    main()