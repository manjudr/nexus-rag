from .base import BaseVectorDB
from pymilvus import MilvusClient
from typing import List, Dict, Optional

class ContentDiscoveryVectorDB(BaseVectorDB):
    """
    Specialized Vector DB for Content Discovery Agent
    Handles filename, page, and content metadata
    """
    
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
            primary_field_name="id",  # Add explicit primary field
            vector_field_name="vector",  # Add explicit vector field  
            auto_id=True  # Enable auto ID generation
        )

    def insert(self, data: List[str], embeddings: List[List[float]], metadata: List[Dict] = None):
        """Insert content with filename and page metadata"""
        insert_data = []
        for i, (text, emb) in enumerate(zip(data, embeddings)):
            if metadata and i < len(metadata):
                meta = metadata[i]
                # Store filename and page as separate searchable text
                enhanced_text = f"FILENAME:{meta.get('filename', 'unknown.pdf')} PAGE:{meta.get('page', 1)} CONTENT:{text}"
            else:
                enhanced_text = f"FILENAME:unknown.pdf PAGE:1 CONTENT:{text}"
            
            # Don't include 'id' field - let Milvus auto-generate it
            insert_data.append({"vector": emb, "text": enhanced_text})
            
        self.client.insert(collection_name=self.collection_name, data=insert_data)

    def search(self, query_embedding: List[float], top_k: int, query_metadata: Optional[Dict] = None):
        """Search and return structured content discovery results with metadata filtering"""
        
        # Build filters based on query metadata
        filters = []
        if query_metadata:
            
            if query_metadata.get('board'):
                filters.append(f'board == "{query_metadata.get('board')}"')
            if query_metadata.get('medium'):
                filters.append(f'medium == "{query_metadata.get('medium')}"')
            if query_metadata.get('subject'):
                filters.append(f'subject == "{query_metadata.get('subject')}"')
            if query_metadata.get('grade'):
                filters.append(f'grade == "{query_metadata.get('grade')}"')
        
        # Create filter expression
        expr = " and ".join(filters) if filters else ""

        # Run search with conditional filtering
        if filters:  # Only apply filters if they exist
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text"],  # Only text field exists in basic schema
                filter=expr  # Apply metadata filtering using 'filter' parameter
            )
        else:
            # Run search without filtering
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text"]  # Only text field exists in basic schema
            )

        print("===== results === ", results)
        
        if results and len(results[0]) > 0:
            structured_results = []
            for hit in results[0]:
                text = hit.entity.get("text")
                filename, page, content = self._parse_enhanced_text(text)
                
                # Basic schema only has text field, extract metadata from text
                structured_results.append({
                    "content": content,
                    "filename": filename,
                    "page": page,
                    "board": "",  # Not available in basic schema
                    "medium": "",  # Not available in basic schema
                    "subject": "",  # Not available in basic schema
                    "grade": "",  # Not available in basic schema
                    "score": hit.distance
                })
            return structured_results
        return []
    
    def _parse_enhanced_text(self, text: str):
        """Parse enhanced text to extract filename, page, and content"""
        import re
        
        # Extract filename
        filename_match = re.search(r'FILENAME:([^\s]+)', text)
        filename = filename_match.group(1) if filename_match else "unknown.pdf"
        
        # Extract page
        page_match = re.search(r'PAGE:(\d+)', text)
        page = int(page_match.group(1)) if page_match else 1
        
        # Extract content
        content_match = re.search(r'CONTENT:(.*)', text, re.DOTALL)
        content = content_match.group(1).strip() if content_match else text
        
        return filename, page, content

    def get_all_documents(self):
        """Get all documents with parsed metadata"""
        results = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["text"],  # Only text field exists in basic schema
            limit=16384
        )
        
        documents = []
        for result in results:
            text = result['text']
            filename, page, content = self._parse_enhanced_text(text)
            
            # Basic schema only has text field
            documents.append({
                "content": content,
                "filename": filename,
                "page": page,
                "board": "",  # Not available in basic schema
                "medium": "",  # Not available in basic schema
                "subject": "",  # Not available in basic schema
                "grade": ""  # Not available in basic schema
            })
        return documents
