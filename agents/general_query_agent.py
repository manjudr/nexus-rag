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
    
    def __init__(self, db_path: str, embedding_model: EmbeddingModel, llm: GenerativeModel, top_k: int = 5, return_json: bool = False):
        self.embedding_model = embedding_model
        self.llm = llm
        self.top_k = top_k
        self.return_json = return_json
        
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
            return_json=self.return_json
        )
        
        print(f"✅ General Query Agent initialized with collection: general_content")

    def run(self, query: str):
        """Process general queries with fallback to LLM general knowledge"""
        print(f"🔍 General Query Agent: Processing general query...")
        
        # First try to find relevant content in the database
        try:
            result = self.content_tool.run(query)
            
            # Check if we got a meaningful result
            if ("No relevant content found" in str(result) or 
                "low_relevance" in str(result) or
                "error" in str(result).lower()):
                
                print("🔄 Using LLM general knowledge (no relevant content found)")
                return self._generate_general_knowledge_response(query)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Database search failed, using LLM general knowledge")
            return self._generate_general_knowledge_response(query)
    
    def _generate_general_knowledge_response(self, query: str) -> str:
        """Generate a response using LLM's general knowledge when no relevant content exists"""
        prompt = f"""You are a helpful AI assistant. The user is asking about a topic that isn't covered in our specific document database. Please provide a helpful, accurate, and informative answer using your general knowledge.

User Question: {query}

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. Is educational and informative
3. Includes relevant examples where appropriate
4. Mentions that this information comes from general knowledge since specific educational content wasn't found in the database

Answer:"""
        
        try:
            response = self.llm.generate(prompt)
            
            # Add a note about the source
            footer = "\n\n---\n📝 *Note: This response is based on general AI knowledge since specific educational content for this topic wasn't found in our document database. For more detailed or specialized information, please consult relevant educational resources.*"
            
            return response + footer
            
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response for your query '{query}'. Please try rephrasing your question or check if relevant educational content has been loaded into the system. Error: {str(e)}"

    def get_collection_stats(self):
        """Get statistics about the general content collection"""
        try:
            if hasattr(self.vector_db, 'get_collection_stats'):
                return self.vector_db.get_collection_stats()
            return {"status": "General content collection ready"}
        except Exception as e:
            return {"error": f"Could not get general content collection stats: {e}"}
