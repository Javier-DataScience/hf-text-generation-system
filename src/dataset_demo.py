# ---------------------------------------------------
# DATASETS HUB DEMO (Hugging Face)
# ---------------------------------------------------
# Goal:
# Learn how to load and inspect real datasets
# ---------------------------------------------------

from datasets import load_dataset


def main():

    print("\n=== HUGGING FACE DATASETS DEMO ===\n")

    # Load a real dataset (IMDb movie reviews)
    dataset = load_dataset("imdb")

    print("Dataset loaded successfully!")

    # Show dataset structure
    print("\nAvailable splits:", dataset.keys())

    # Access training data
    train_data = dataset["train"]

    print("\nExample sample:")
    print(train_data[0])

    print("\nLabel meaning:")
    print("0 = negative review")
    print("1 = positive review")


if __name__ == "__main__":
    main()