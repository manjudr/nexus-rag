from agents.base import BaseAgent
from tools.content_discovery_tool import ContentDiscoveryTool
from vector_db.content_discovery_db import ContentDiscoveryVectorDB
from vector_db.enhanced_content_discovery_db import EnhancedContentDiscoveryVectorDB
from models.embedding import EmbeddingModel
from models.llm import GenerativeModel
import main_config as config

class PDFContentAgent(BaseAgent):
    """
    Specialized agent for handling PDF educational content queries
    Uses dedicated PDF vector database and educational enhancement tools
    """
    
    def __init__(self, db_path: str, embedding_model: EmbeddingModel, llm: GenerativeModel, top_k: int = 5):
        self.embedding_model = embedding_model
        self.llm = llm
        self.top_k = top_k
        
        # Get configuration - centralized in MODELS section
        current_models = config.MODELS[config.MODEL_PROVIDER]
        langextract_enabled = current_models.get("langextract_enabled", False)
        
        # Create appropriate database based on LangExtract setting
        if langextract_enabled:
            # Use enhanced database with LangExtract
            self.vector_db = EnhancedContentDiscoveryVectorDB(
                db_path=db_path,
                collection_name="pdf_educational_content",  # Dedicated collection for PDFs
                top_k=top_k,
                use_langextract=langextract_enabled
            )
            print(f"✅ PDF Agent using ENHANCED database with LangExtract: {langextract_enabled}")
        else:
            # Use basic database for fast loading
            self.vector_db = ContentDiscoveryVectorDB(
                db_path=db_path,
                collection_name="pdf_educational_content",  # Dedicated collection for PDFs
                top_k=top_k
            )
            print(f"✅ PDF Agent using BASIC database (fast mode)")
        
        # Create specialized tool for PDF content discovery
        self.content_tool = ContentDiscoveryTool(
            db=self.vector_db,
            embedding_model=embedding_model,
            llm=llm,
            name="PDF Educational Content Discovery",
            description="Discovers and analyzes educational content from PDF documents with enhanced metadata",
            top_k=top_k,
            return_json=False
        )
        
        print(f"✅ PDF Content Agent initialized with collection: pdf_educational_content")

    def run(self, query: str):
        """Process educational PDF content queries"""
        print(f"📚 PDF Content Agent: Processing educational query...")
        
        # Use the specialized content discovery tool
        result = self.content_tool.run(query)
        
        return result

    def get_collection_stats(self):
        """Get statistics about the PDF collection"""
        try:
            if hasattr(self.vector_db, 'get_collection_stats'):
                return self.vector_db.get_collection_stats()
            return {"status": "PDF collection ready"}
        except Exception as e:
            return {"error": f"Could not get PDF collection stats: {e}"}
