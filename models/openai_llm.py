import os
from openai import OpenAI
from .llm import GenerativeModel

class OpenAIGenerativeModel(GenerativeModel):
    def __init__(self, model_name: str):
        # It's best practice to use an environment variable for your API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str):
        """Generates a response using the OpenAI Chat Completions API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred with the OpenAI API: {e}")
            return "Sorry, I couldn't generate an answer at this time."
