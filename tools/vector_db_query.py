from .base import BaseTool
from vector_db.base import BaseVectorDB
from models.embedding import EmbeddingModel
from rank_bm25 import BM25Okapi
from vector_db.milvus_db import MilvusVectorDB

class VectorDBQueryTool(BaseTool):
    def __init__(self, db: BaseVectorDB, embedding_model: EmbeddingModel, top_k: int = 5):
        self.db = db
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.bm25 = self._initialize_bm25()

    def _initialize_bm25(self):
        if isinstance(self.db, MilvusVectorDB):
            print("Tool: Initializing BM25...")
            documents = self.db.get_all_documents()
            if not documents:
                return None
            tokenized_corpus = [doc.split(" ") for doc in documents]
            return BM25Okapi(tokenized_corpus)
        return None

    def run(self, query: str):
        """Runs the tool to query the vector DB and retrieve relevant text chunks."""
        print(f"Tool: Creating embedding for query: '{query}'")
        query_embedding = self.embedding_model.create_embedding(query)
        
        print(f"Tool: Querying Vector DB...")
        vector_results = self.db.search(query_embedding, top_k=self.top_k)

        if self.bm25:
            print("Tool: Performing BM25 search...")
            tokenized_query = query.split(" ")
            bm25_results = self.bm25.get_top_n(tokenized_query, self.db.get_all_documents(), n=self.top_k)
            
            # Combine and re-rank results (simple union for now)
            combined_results = list(set(vector_results + bm25_results))
            return combined_results

        return vector_results
