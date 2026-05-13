# ---------------------------------------------------
# MAIN ENTRY POINT
# Text Generation Lab (Upgraded)
# ---------------------------------------------------

from src.text_generator import TextGenerator


if __name__ == "__main__":

    engine = TextGenerator()

    prompt = "Artificial intelligence will transform the world because"

    print("\n==============================")
    print("LLM TEXT GENERATION EXPERIMENT")
    print("==============================\n")

    print("PROMPT:")
    print(prompt)

    print("\nRunning experiment...\n")

    experiment = engine.run_experiment(prompt)

    print("MODEL:", experiment["model"])

    for i, result in enumerate(experiment["results"]):

        print("\n------------------------------")
        print(f"EXPERIMENT {i+1}")
        print("Temperature:", result["temperature"])
        print("\nOutput:")
        print(result["output"])