from .base import BaseTool
from vector_db.base import BaseVectorDB
from models.embedding import EmbeddingModel

class VectorDBQueryTool(BaseTool):
    def __init__(self, db: BaseVectorDB, embedding_model: EmbeddingModel):
        self.db = db
        self.embedding_model = embedding_model

    def run(self, query: str):
        """Runs the tool to query the vector DB and retrieve relevant text chunks."""
        print(f"Tool: Creating embedding for query: '{query}'")
        query_embedding = self.embedding_model.create_embedding(query)
        
        print(f"Tool: Querying Vector DB...")
        results = self.db.search(query_embedding)
        return results
