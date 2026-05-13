# ---------------------------------------------------
# Text Generation Engine (Hugging Face)
# ---------------------------------------------------
# This module is responsible for generating text
# using pretrained transformer models.
# ---------------------------------------------------

from transformers import pipeline


class TextGenerator:
    def __init__(self, model_name="distilgpt2"):
        # Load model once when class is created
        self.generator = pipeline("text-generation", model=model_name)

    def generate(self, prompt, temperature=0.7, max_length=80):
        """
        Generates text from a prompt.

        Parameters:
        - prompt: input text
        - temperature: randomness control
        - max_length: max output length
        """

        result = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1
        )

        return result[0]["generated_text"]