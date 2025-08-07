from .embedding import EmbeddingModel
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text).tolist()

    def get_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()
