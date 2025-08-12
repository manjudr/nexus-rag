from .base import BaseVectorDB
from .milvus_db import MilvusVectorDB
from .flexible_milvus_db import FlexibleMilvusDB
from .content_discovery_db import ContentDiscoveryVectorDB
from typing import Dict, Any

class VectorDBFactory:
    """
    Factory class to create appropriate vector database instances
    based on agent requirements and schemas
    """
    
    # Schema definitions for different agents
    AGENT_SCHEMAS = {
        "content_discovery_agent": {
            "filename": str,
            "page": int,
            "title": str,
            "author": str
        },
        "general_query_agent": {},  # No special metadata
        "image_analysis_agent": {
            "image_path": str,
            "image_type": str,
            "dimensions": str
        }
    }
    
    @classmethod
    def create_db(cls, agent_type: str, db_path: str, collection_name: str, top_k: int = 5) -> BaseVectorDB:
        """
        Create appropriate vector database instance based on agent type
        
        Args:
            agent_type: Type of agent (content_discovery_agent, general_query_agent, etc.)
            db_path: Path to the database file
            collection_name: Name of the collection
            top_k: Number of top results to return
            
        Returns:
            BaseVectorDB: Appropriate database instance
        """
        
        if agent_type in ["content_discovery_agent", "educational_content_agent"]:
            # Use specialized content discovery database
            return ContentDiscoveryVectorDB(db_path, collection_name, top_k)
        
        elif agent_type in cls.AGENT_SCHEMAS:
            # Use flexible database with schema
            schema = cls.AGENT_SCHEMAS[agent_type]
            return FlexibleMilvusDB(db_path, collection_name, top_k, metadata_schema=schema)
        
        else:
            # Use basic database for unknown agent types
            return MilvusVectorDB(db_path, collection_name, top_k)
    
    @classmethod
    def register_agent_schema(cls, agent_type: str, schema: Dict[str, Any]):
        """Register a new agent schema"""
        cls.AGENT_SCHEMAS[agent_type] = schema
    
    @classmethod
    def get_supported_agents(cls):
        """Get list of supported agent types"""
        return list(cls.AGENT_SCHEMAS.keys())
