# Database Configuration with Agent-Specific Settings

# Global Settings
DB_PATH = "milvus_demo.db"
TOP_K = 5
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# Model Configuration
MODEL_CONFIG = {
    "provider": "local",  # "local" or "openai"
    "local": {
        "embedding_model": "all-MiniLM-L6-v2",
        "llm": "google/flan-t5-base"
    },
    "openai": {
        "embedding_model": "text-embedding-3-small",
        "llm": "gpt-3.5-turbo"
    }
}

# Educational Content Configuration (for multi-PDF loading)
EDUCATIONAL_CONTENT_CONFIG = {
    "content_directory": "data/educational_content/",
    "metadata_file": "data/educational_content/metadata.json"
}

DATABASE_CONFIG = {
    "content_discovery_agent": {
        "db_type": "content_discovery",  # Uses ContentDiscoveryVectorDB
        "metadata_required": True,
        "schema": {
            "filename": str,
            "page": int,
            "title": str,
            "author": str
        }
    },
    "general_query_agent": {
        "db_type": "basic",  # Uses basic MilvusVectorDB
        "metadata_required": False,
        "schema": {}
    },
    "image_analysis_agent": {
        "db_type": "flexible",  # Uses FlexibleMilvusDB
        "metadata_required": True,
        "schema": {
            "image_path": str,
            "image_type": str,
            "dimensions": str
        }
    }
}

# Updated Agent Configuration
AGENT_CONFIGS_V2 = {
    "educational_content_agent": {
        "name": "Content Discovery Agent",
        "collection_name": "educational_content",
        "path": "data/educational_content/",
        "parser": "pdf_with_metadata",
        "description": "Discovers educational content from multiple subject files with rich metadata",
        "db_config": DATABASE_CONFIG["content_discovery_agent"]
    },
    "content_discovery_agent": {
        "name": "Basic Content Discovery Agent",
        "collection_name": "content_discovery",
        "path": "data/obsrv.pdf",
        "parser": "pdf",
        "description": "Discovers and recommends content from basic PDF files",
        "db_config": DATABASE_CONFIG["content_discovery_agent"]
    },
    "general_query_agent": {
        "name": "General Query Agent", 
        "collection_name": "general_content",
        "path": "data/obsrv.pdf",
        "parser": "pdf",
        "description": "Answers general questions about the uploaded content",
        "db_config": DATABASE_CONFIG["general_query_agent"]
    }
}
