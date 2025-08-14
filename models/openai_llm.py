import os
from openai import OpenAI
from .llm import GenerativeModel

class OpenAIGenerativeModel(GenerativeModel):
    def __init__(self, model_name: str):
        # It's best practice to use an environment variable for your API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        # Check if we're using Azure
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.environ.get("AZURE_CHAT_DEPLOYMENT")
        
        if azure_endpoint and azure_deployment:
            # Azure OpenAI with specific chat deployment and API version
            base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{azure_deployment}/"
            self.client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                default_query={"api-version": "2024-02-15-preview"}
            )
            # For Azure, use the deployment name as the model name
            self.model_name = azure_deployment
            print(f"🔗 Using Azure OpenAI chat: {base_url} (model: {self.model_name})")
        else:
            # Regular OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
            print("🔗 Using regular OpenAI chat")

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
