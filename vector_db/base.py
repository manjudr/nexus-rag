from abc import ABC, abstractmethod

class BaseVectorDB(ABC):
    @abstractmethod
    def setup(self, dimension: int):
        pass

    @abstractmethod
    def insert(self, data, embeddings):
        pass

    @abstractmethod
    def search(self, query_embedding):
        pass
