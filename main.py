# ---------------------------------------------------
# MAIN ENTRY POINT
# Text Generation Lab
# ---------------------------------------------------

from src.text_generator import TextGenerator


if __name__ == "__main__":

    engine = TextGenerator()

    prompt = "Artificial intelligence will change the world because"

    print("\n=== TEXT GENERATION LAB ===\n")
    print("PROMPT:", prompt)

    print("\n--- LOW CREATIVITY ---")
    print(engine.generate(prompt, temperature=0.2))

    print("\n--- BALANCED ---")
    print(engine.generate(prompt, temperature=0.7))

    print("\n--- HIGH CREATIVITY ---")
    print(engine.generate(prompt, temperature=1.2))