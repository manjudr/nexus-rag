from agents.base import BaseAgent
from tools.content_discovery_tool import ContentDiscoveryTool
from vector_db.content_discovery_db import ContentDiscoveryVectorDB
from models.embedding import EmbeddingModel
from models.llm import GenerativeModel

class GeneralQueryAgent(BaseAgent):
    """
    General purpose agent for handling non-specialized queries
    Uses general content database for basic question answering
    """
    
    def __init__(self, db_path: str, embedding_model: EmbeddingModel, llm: GenerativeModel, top_k: int = 5):
        self.embedding_model = embedding_model
        self.llm = llm
        self.top_k = top_k
        
        # Create general content database
        self.vector_db = ContentDiscoveryVectorDB(
            db_path=db_path,
            collection_name="general_content",  # General collection for misc content
            top_k=top_k
        )
        
        # Create general content discovery tool
        self.content_tool = ContentDiscoveryTool(
            db=self.vector_db,
            embedding_model=embedding_model,
            llm=llm,
            name="General Content Discovery",
            description="Handles general questions and basic content discovery across all content types",
            top_k=top_k,
            return_json=False
        )
        
        print(f"✅ General Query Agent initialized with collection: general_content")

    def run(self, query: str):
        """Process general queries"""
        print(f"🔍 General Query Agent: Processing general query...")
        print(f"🔍 Query: {query}")
        
        # Use the general content discovery tool
        result = self.content_tool.run(query)
        
        print(f"✅ General Query Agent: Retrieved general content")
        return result

    def get_collection_stats(self):
        """Get statistics about the general content collection"""
        try:
            if hasattr(self.vector_db, 'get_collection_stats'):
                return self.vector_db.get_collection_stats()
            return {"status": "General content collection ready"}
        except Exception as e:
            return {"error": f"Could not get general content collection stats: {e}"}
