from .embedding import EmbeddingModel
import os
from openai import OpenAI

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
            
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._dimension = None

    def create_embedding(self, text):
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        embedding = response.data[0].embedding
        if self._dimension is None:
            self._dimension = len(embedding)
        return embedding

    def get_embedding_dimension(self):
        if self._dimension is None:
            # Create a dummy embedding to determine the dimension
            self.create_embedding("test")
        return self._dimension
