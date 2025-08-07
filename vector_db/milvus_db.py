from .base import BaseVectorDB
from pymilvus import MilvusClient

class MilvusVectorDB(BaseVectorDB):
    def __init__(self, db_path: str, collection_name: str, top_k: int):
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.top_k = top_k

    def setup(self, dimension: int):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            auto_id=True,
            primary_field_name="id",
            vector_field_name="vector",
        )

    def insert(self, data, embeddings):
        insert_data = [{"vector": emb, "text": text} for text, emb in zip(data, embeddings)]
        self.client.insert(collection_name=self.collection_name, data=insert_data)

    def search(self, query_embedding):
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=self.top_k,
            output_fields=["text"]
        )
        if results and len(results[0]) > 0:
            return [hit.entity.get("text") for hit in results[0]]
        return []
