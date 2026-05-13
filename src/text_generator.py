# ---------------------------------------------------
# Text Generation Engine (Upgraded)
# Hugging Face Transformers Lab
# ---------------------------------------------------
# This module is responsible for:
# - Loading a pretrained LLM from Hugging Face Hub
# - Generating text from prompts
# - Running controlled experiments with different temperatures
# ---------------------------------------------------

from transformers import pipeline


class TextGenerator:
    def __init__(self, model_name="distilgpt2"):
        """
        Initialize the text generation model.

        Args:
            model_name (str): Hugging Face model ID
        """
        self.model_name = model_name

        # Load the pipeline once (important for efficiency)
        self.generator = pipeline("text-generation", model=model_name)

    def generate(self, prompt, temperature=0.7, max_length=80):
        """
        Generate text from a prompt.

        Args:
            prompt (str): Input text
            temperature (float): randomness control (0 = deterministic, 1+ = creative)
            max_length (int): maximum number of tokens to generate

        Returns:
            str: generated text
        """

        result = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.2,
            truncation=True
        )

        return result[0]["generated_text"]

    def run_experiment(self, prompt, temperatures=None):
        """
        Run multiple generation experiments with different temperatures.

        Args:
            prompt (str): input text
            temperatures (list): list of temperature values

        Returns:
            dict: structured experiment results
        """

        if temperatures is None:
            temperatures = [0.2, 0.7, 1.2]

        results = []

        for temp in temperatures:
            output = self.generate(prompt, temperature=temp)

            results.append({
                "temperature": temp,
                "output": output
            })

        return {
            "model": self.model_name,
            "prompt": prompt,
            "results": results
        }