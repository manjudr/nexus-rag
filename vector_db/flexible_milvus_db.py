from .base import BaseVectorDB
from pymilvus import MilvusClient
from typing import Dict, List, Any, Optional

class FlexibleMilvusDB(BaseVectorDB):
    """
    A flexible Milvus Vector DB that can handle different schemas based on agent requirements.
    Each collection can have its own metadata structure.
    """
    
    def __init__(self, db_path: str, collection_name: str, top_k: int, metadata_schema: Optional[Dict] = None):
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.top_k = top_k
        self.metadata_schema = metadata_schema or {}
        
    def setup(self, dimension: int):
        """Setup collection with optional metadata schema"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
        )

    def insert(self, data: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict]] = None):
        """Insert data with optional metadata based on schema"""
        if metadata and self.metadata_schema:
            # Store metadata as JSON string in a separate field
            insert_data = []
            for text, emb, meta in zip(data, embeddings, metadata):
                # Filter metadata based on schema
                filtered_meta = {k: meta.get(k) for k in self.metadata_schema.keys() if k in meta}
                
                insert_data.append({
                    "vector": emb, 
                    "text": text,
                    "metadata": str(filtered_meta)  # Store as string for simple schema
                })
        else:
            insert_data = [{"vector": emb, "text": text, "metadata": "{}"} 
                          for text, emb in zip(data, embeddings)]
            
        self.client.insert(collection_name=self.collection_name, data=insert_data)

    def search(self, query_embedding: List[float], top_k: int):
        """Search and return results with metadata"""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "metadata"]
        )
        
        if results and len(results[0]) > 0:
            search_results = []
            for hit in results[0]:
                text = hit.entity.get("text")
                metadata_str = hit.entity.get("metadata", "{}")
                
                # Parse metadata back to dict
                try:
                    import ast
                    metadata = ast.literal_eval(metadata_str) if metadata_str != "{}" else {}
                except:
                    metadata = {}
                
                search_results.append({
                    "text": text,
                    "metadata": metadata,
                    "score": hit.distance
                })
            return search_results
        return []

    def get_all_documents(self):
        """Get all documents with metadata"""
        results = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["text", "metadata"],
            limit=16384
        )
        
        documents = []
        for result in results:
            text = result['text']
            metadata_str = result.get('metadata', '{}')
            
            try:
                import ast
                metadata = ast.literal_eval(metadata_str) if metadata_str != "{}" else {}
            except:
                metadata = {}
                
            documents.append({
                "text": text,
                "metadata": metadata
            })
        return documents
