from .llm import GenerativeModel
from transformers import pipeline

class HuggingFaceGenerativeModel(GenerativeModel):
    def __init__(self, model_name: str):
        self.pipeline = pipeline("text2text-generation", model=model_name)

    def generate(self, prompt: str):
        result = self.pipeline(prompt)
        return result[0]['generated_text']
