from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass

    @abstractmethod
    def get_embedding_dimension(self):
        pass
