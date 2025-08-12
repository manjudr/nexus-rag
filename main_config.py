# Streamlined Configuration for Educational Content RAG
# Keeps Milvus, LangExtract, and multiple agents but removes complexity

# Basic Settings
DB_PATH = "milvus_demo.db"
TOP_K = 5
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# Model Configuration - Simple choice
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

# Educational Content Configuration
EDUCATIONAL_CONTENT_CONFIG = {
    "content_directory": "data/educational_content/pdfs/",
    "metadata_file": "data/educational_content/metadata.json"
}

# LangExtract Configuration - Simplified
LANGEXTRACT_CONFIG = {
    "enabled": True,  # Enable for richer educational metadata
    "model_provider": "gemini",  # "gemini", "openai", or "local"
    "gemini_model": "gemini-2.0-flash-exp",
    "openai_model": "gpt-4o",
    "local_model": "llama3.2"
}

# Agent Configurations - Focus on Educational Content
AGENT_CONFIGS = {
    "educational_content_agent": {
        "name": "Educational Content Discovery Agent",
        "collection_name": "educational_content",
        "description": "Discovers educational content with rich metadata and learning context",
        "top_k": TOP_K,
        "enhanced_parsing": True
    },
    "general_query_agent": {
        "name": "General Query Agent",
        "collection_name": "general_content", 
        "description": "Answers general questions about educational content",
        "top_k": TOP_K,
        "enhanced_parsing": False
    }
}
