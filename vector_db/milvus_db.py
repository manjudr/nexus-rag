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
        )

    def insert(self, data, embeddings, metadata=None):
        if metadata:
            # Embed metadata in the text for storage
            insert_data = []
            for text, emb, meta in zip(data, embeddings, metadata):
                enhanced_text = f"[FILE: {meta.get('filename', 'unknown.pdf')}] [PAGE: {meta.get('page', 1)}]\n{text}"
                insert_data.append({"vector": emb, "text": enhanced_text})
        else:
            insert_data = [{"vector": emb, "text": text} for text, emb in zip(data, embeddings)]
        self.client.insert(collection_name=self.collection_name, data=insert_data)

    def search(self, query_embedding, top_k: int):
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text"]
        )
        if results and len(results[0]) > 0:
            enhanced_results = []
            for hit in results[0]:
                text = hit.entity.get("text")
                # Extract metadata from enhanced text
                filename, page, clean_text = self._extract_metadata_from_text(text)
                enhanced_results.append((clean_text, filename, page))
            return enhanced_results
        return []
    
    def _extract_metadata_from_text(self, text):
        """Extract filename and page from enhanced text format"""
        import re
        # Look for [FILE: filename] [PAGE: number] pattern
        file_match = re.search(r'\[FILE: ([^\]]+)\]', text)
        page_match = re.search(r'\[PAGE: (\d+)\]', text)
        
        filename = file_match.group(1) if file_match else "unknown.pdf"
        page = int(page_match.group(1)) if page_match else 1
        
        # Remove metadata tags from text
        clean_text = re.sub(r'\[FILE: [^\]]+\]\s*\[PAGE: \d+\]\s*\n?', '', text)
        
        return filename, page, clean_text

    def get_all_documents(self):
        results = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["text"],
            limit=16384  # Max limit for a query
        )
        enhanced_results = []
        for result in results:
            text = result['text']
            filename, page, clean_text = self._extract_metadata_from_text(text)
            enhanced_results.append((clean_text, filename, page))
        return enhanced_results
