"""
Enhanced Configuration System for Full Dynamic Configuration
"""
import os
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatabaseSchema:
    """Dynamic database schema definition"""
    fields: Dict[str, type]
    required_fields: List[str]
    optional_fields: List[str] = None

@dataclass
class AgentConfig:
    """Dynamic agent configuration"""
    name: str
    collection_name: str
    data_sources: List[str]  # Support multiple files/directories
    parser_type: str
    description: str
    db_schema: DatabaseSchema
    db_type: str

class DynamicConfig:
    """Fully dynamic configuration system"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "config.json"
        self.base_data_dir = os.environ.get('NEXUS_DATA_DIR', 'data')
        self.base_db_path = os.environ.get('NEXUS_DB_PATH', 'milvus_demo.db')
        
    def get_content_directories(self) -> List[str]:
        """Dynamically discover content directories"""
        content_dirs = []
        data_path = Path(self.base_data_dir)
        
        for item in data_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                content_dirs.append(str(item))
                
        return content_dirs
    
    def load_agent_configs(self) -> Dict[str, AgentConfig]:
        """Load agent configurations from file or auto-discover"""
        # Try to load from config file first
        if os.path.exists(self.config_file):
            return self._load_from_file()
        else:
            return self._auto_discover_agents()
    
    def _auto_discover_agents(self) -> Dict[str, AgentConfig]:
        """Auto-discover agents based on directory structure"""
        agents = {}
        
        # Discover content directories
        for content_dir in self.get_content_directories():
            dir_name = Path(content_dir).name
            
            # Create agent config based on directory
            agent_key = f"{dir_name}_agent"
            agents[agent_key] = AgentConfig(
                name=f"{dir_name.title()} Agent",
                collection_name=f"{dir_name}_collection",
                data_sources=[content_dir],
                parser_type=self._detect_parser_type(content_dir),
                description=f"Agent for {dir_name} content discovery",
                db_schema=self._infer_schema(content_dir),
                db_type="content_discovery" if "content" in dir_name else "flexible"
            )
            
        return agents
    
    def _detect_parser_type(self, directory: str) -> str:
        """Detect appropriate parser based on file types in directory"""
        dir_path = Path(directory)
        
        if any(f.suffix == '.pdf' for f in dir_path.glob('*')):
            return "pdf"
        elif any(f.suffix == '.txt' for f in dir_path.glob('*')):
            if (dir_path / 'metadata.json').exists():
                return "pdf_with_metadata"
            return "txt"
        else:
            return "auto"
    
    def _infer_schema(self, directory: str) -> DatabaseSchema:
        """Infer database schema based on content type"""
        if "educational" in directory.lower():
            return DatabaseSchema(
                fields={
                    "filename": str,
                    "page": int,
                    "title": str,
                    "author": str,
                    "course": str,
                    "grade": str
                },
                required_fields=["filename", "title"],
                optional_fields=["page", "author", "course", "grade"]
            )
        else:
            return DatabaseSchema(
                fields={"filename": str, "page": int},
                required_fields=["filename"]
            )

# Environment-based configuration
DYNAMIC_CONFIG = {
    "data_directories": {
        "base": os.environ.get('NEXUS_DATA_DIR', 'data'),
        "content": os.environ.get('NEXUS_CONTENT_DIR', 'data/educational_content'),
        "uploads": os.environ.get('NEXUS_UPLOADS_DIR', 'data/uploads')
    },
    "database": {
        "path": os.environ.get('NEXUS_DB_PATH', 'milvus_demo.db'),
        "type": os.environ.get('NEXUS_DB_TYPE', 'milvus')
    },
    "models": {
        "provider": os.environ.get('NEXUS_MODEL_PROVIDER', 'local'),
        "embedding_model": os.environ.get('NEXUS_EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
        "llm_model": os.environ.get('NEXUS_LLM_MODEL', 'google/flan-t5-base')
    },
    "processing": {
        "chunk_size": int(os.environ.get('NEXUS_CHUNK_SIZE', '1024')),
        "chunk_overlap": int(os.environ.get('NEXUS_CHUNK_OVERLAP', '128')),
        "top_k": int(os.environ.get('NEXUS_TOP_K', '5'))
    }
}

# Dynamic stop words loading
def load_stop_words(language: str = "english") -> set:
    """Load stop words dynamically based on language"""
    stop_words_file = f"config/stop_words_{language}.txt"
    
    if os.path.exists(stop_words_file):
        with open(stop_words_file, 'r') as f:
            return set(word.strip().lower() for word in f.readlines())
    
    # Fallback to default English stop words
    return {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was', 'were'
    }
