from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseVectorDB(ABC):
    @abstractmethod
    def setup(self, dimension: int):
        pass

    @abstractmethod
    def insert(self, data, embeddings):
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int, query_metadata: Optional[Dict] = None):
        pass
