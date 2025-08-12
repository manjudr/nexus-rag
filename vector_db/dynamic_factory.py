"""
Enhanced Vector Database Factory with Full Dynamic Configuration
"""
from .base import BaseVectorDB
from .milvus_db import MilvusVectorDB
from .flexible_milvus_db import FlexibleMilvusDB
from .content_discovery_db import ContentDiscoveryVectorDB
from typing import Dict, Any, Type
import importlib
import os

class DynamicVectorDBFactory:
    """
    Enhanced factory class that can dynamically create vector databases
    based on configuration files, environment variables, or auto-discovery
    """
    
    # Registry of available database types
    DB_REGISTRY: Dict[str, Type[BaseVectorDB]] = {
        "basic": MilvusVectorDB,
        "flexible": FlexibleMilvusDB,
        "content_discovery": ContentDiscoveryVectorDB,
        "milvus": MilvusVectorDB
    }
    
    @classmethod
    def register_db_type(cls, name: str, db_class: Type[BaseVectorDB]):
        """Register a new database type dynamically"""
        cls.DB_REGISTRY[name] = db_class
    
    @classmethod
    def load_db_plugins(cls, plugin_directory: str = "vector_db/plugins"):
        """Dynamically load database plugins from directory"""
        if not os.path.exists(plugin_directory):
            return
            
        for file in os.listdir(plugin_directory):
            if file.endswith('_db.py') and not file.startswith('__'):
                module_name = file[:-3]  # Remove .py
                try:
                    module = importlib.import_module(f"vector_db.plugins.{module_name}")
                    # Look for database classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseVectorDB) and 
                            attr != BaseVectorDB):
                            cls.register_db_type(module_name, attr)
                except ImportError as e:
                    print(f"Warning: Could not load plugin {module_name}: {e}")
    
    @classmethod
    def create_db_from_config(cls, config: Dict[str, Any]) -> BaseVectorDB:
        """
        Create database instance from configuration dictionary
        
        Args:
            config: Configuration dictionary with keys:
                - db_type: Type of database
                - db_path: Path to database file
                - collection_name: Name of collection
                - top_k: Number of results to return
                - schema: Optional schema for flexible databases
                - custom_params: Optional custom parameters
        """
        db_type = config.get("db_type", "basic")
        db_path = config.get("db_path", "milvus_demo.db")
        collection_name = config.get("collection_name", "default_collection")
        top_k = config.get("top_k", 5)
        schema = config.get("schema", {})
        custom_params = config.get("custom_params", {})
        
        if db_type not in cls.DB_REGISTRY:
            raise ValueError(f"Unknown database type: {db_type}. Available types: {list(cls.DB_REGISTRY.keys())}")
        
        db_class = cls.DB_REGISTRY[db_type]
        
        # Handle different database initialization patterns
        if db_type == "flexible" and schema:
            return db_class(db_path, collection_name, top_k, schema=schema, **custom_params)
        else:
            return db_class(db_path, collection_name, top_k, **custom_params)
    
    @classmethod
    def create_db_for_agent(cls, agent_config: Dict[str, Any], global_config: Dict[str, Any] = None) -> BaseVectorDB:
        """
        Create database for specific agent configuration
        
        Args:
            agent_config: Agent-specific configuration
            global_config: Global configuration settings
        """
        # Merge agent config with global defaults
        if global_config:
            db_config = {
                "db_path": global_config.get("db_path", "milvus_demo.db"),
                "top_k": global_config.get("top_k", 5),
                **agent_config.get("db_config", {})
            }
        else:
            db_config = agent_config.get("db_config", {})
        
        db_config["collection_name"] = agent_config.get("collection_name", "default")
        
        return cls.create_db_from_config(db_config)
    
    @classmethod
    def auto_detect_db_type(cls, data_path: str, agent_type: str = None) -> str:
        """
        Automatically detect appropriate database type based on data and agent
        
        Args:
            data_path: Path to data source
            agent_type: Type of agent (optional)
        """
        # Check if it's a directory with metadata (educational content)
        if os.path.isdir(data_path):
            metadata_file = os.path.join(data_path, "metadata.json")
            if os.path.exists(metadata_file):
                return "content_discovery"
        
        # Check agent type hints
        if agent_type:
            if "content" in agent_type.lower():
                return "content_discovery"
            elif "image" in agent_type.lower():
                return "flexible"
        
        # Default to basic
        return "basic"
    
    @classmethod
    def create_db_auto(cls, agent_type: str, data_path: str, collection_name: str, 
                      global_config: Dict[str, Any] = None) -> BaseVectorDB:
        """
        Automatically create appropriate database based on agent type and data
        
        Args:
            agent_type: Type of agent
            data_path: Path to data source  
            collection_name: Name of collection
            global_config: Global configuration settings
        """
        # Auto-detect database type
        db_type = cls.auto_detect_db_type(data_path, agent_type)
        
        # Create configuration
        config = {
            "db_type": db_type,
            "collection_name": collection_name,
            "db_path": global_config.get("db_path", "milvus_demo.db") if global_config else "milvus_demo.db",
            "top_k": global_config.get("top_k", 5) if global_config else 5
        }
        
        # Add schema for flexible databases
        if db_type == "flexible":
            config["schema"] = cls._infer_schema_from_agent_type(agent_type)
        
        return cls.create_db_from_config(config)
    
    @classmethod
    def _infer_schema_from_agent_type(cls, agent_type: str) -> Dict[str, type]:
        """Infer database schema from agent type"""
        schema_mapping = {
            "content_discovery": {"filename": str, "page": int, "title": str, "author": str},
            "image_analysis": {"image_path": str, "image_type": str, "dimensions": str},
            "general_query": {},
            "educational_content": {
                "filename": str, "page": int, "title": str, "author": str, 
                "course": str, "grade": str, "subject": str
            }
        }
        
        # Find matching schema
        for key, schema in schema_mapping.items():
            if key in agent_type.lower():
                return schema
        
        return {}  # Default empty schema

# Backward compatibility wrapper
class VectorDBFactory(DynamicVectorDBFactory):
    """Backward compatible factory class"""
    
    @classmethod
    def create_db(cls, agent_type: str, db_path: str, collection_name: str, top_k: int = 5) -> BaseVectorDB:
        """Create database using the old interface for backward compatibility"""
        global_config = {"db_path": db_path, "top_k": top_k}
        return cls.create_db_auto(agent_type, "", collection_name, global_config)
