from agents.base import BaseAgent
from tools.content_discovery_tool import ContentDiscoveryTool
from vector_db.content_discovery_db import ContentDiscoveryVectorDB
from models.embedding import EmbeddingModel
from models.llm import GenerativeModel

class VideoTranscriptAgent(BaseAgent):
    """
    Specialized agent for handling video transcript queries
    Uses dedicated video transcript vector database and video-specific tools
    This is prepared for future implementation
    """
    
    def __init__(self, db_path: str, embedding_model: EmbeddingModel, llm: GenerativeModel, top_k: int = 5, return_json: bool = False):
        self.embedding_model = embedding_model
        self.llm = llm
        self.top_k = top_k
        self.return_json = return_json
        
        # Create dedicated video transcript database
        self.vector_db = ContentDiscoveryVectorDB(
            db_path=db_path,
            collection_name="video_transcripts",  # Separate collection for video transcripts
            top_k=top_k
        )
        
        # Create specialized tool for video transcript discovery
        self.transcript_tool = ContentDiscoveryTool(
            db=self.vector_db,
            embedding_model=embedding_model,
            llm=llm,
            name="Video Transcript Discovery",
            description="Discovers and analyzes content from video transcripts with speaker detection and temporal context",
            top_k=top_k,
            return_json=self.return_json
        )
        
        print(f"✅ Video Transcript Agent initialized with collection: video_transcripts")

    def run(self, query: str):
        """Process video transcript queries"""
        print(f"🎥 Video Transcript Agent: Processing video/speech query...")
        print(f"🔍 Query: {query}")
        
        # Check if we have any video transcripts loaded
        try:
            stats = self.get_collection_stats()
            if stats.get("count", 0) == 0:
                return {
                    "response": "No video transcripts are currently loaded in the system. Please load video transcript data first.",
                    "source": "Video Transcript Agent",
                    "collection": "video_transcripts",
                    "status": "empty_collection"
                }
        except Exception:
            pass
        
        # Use the specialized transcript discovery tool
        result = self.transcript_tool.run(query)
        
        print(f"✅ Video Transcript Agent: Retrieved transcript content")
        return result

    def get_collection_stats(self):
        """Get statistics about the video transcript collection"""
        try:
            if hasattr(self.vector_db, 'get_collection_stats'):
                return self.vector_db.get_collection_stats()
            return {"status": "Video transcript collection ready", "count": 0}
        except Exception as e:
            return {"error": f"Could not get video transcript collection stats: {e}"}
