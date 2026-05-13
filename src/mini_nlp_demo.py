# ---------------------------------------------------
# MINI NLP SYSTEM DEMO
# ---------------------------------------------------

from src.mini_nlp_system import MiniNLPSystem


def main():

    system = MiniNLPSystem()

    text = "I really enjoy learning about artificial intelligence"

    compare_text = "Machine learning and AI are very interesting fields"

    print("\n=== MINI NLP SYSTEM ===\n")

    result = system.analyze(text, compare_text)

    print("INPUT:", result["input_text"])

    print("\nSENTIMENT:")
    print(result["sentiment"])

    print("\nSIMILARITY:")
    print(result["similarity"])


if __name__ == "__main__":
    main()