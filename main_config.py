"""
Clean Configuration for Educational Content RAG System
Unified model configuration, simplified structure
"""

# =============================================================================
# DATABASE & CONTENT SETTINGS
# =============================================================================
DB_PATH = "milvus_demo.db"
TOP_K = 5
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# Educational content paths
CONTENT_DIRECTORY = "data/educational_content/pdfs/"
METADATA_FILE = "data/educational_content/metadata.json"

# =============================================================================
# MODEL CONFIGURATION (Unified for all components)
# =============================================================================
# Choose one provider: "local" (free, offline) or "azure" (Azure OpenAI with OpenAI-compatible API)
MODEL_PROVIDER = "local"

# Model definitions for each provider
MODELS = {
    "local": {
        "embedding": "all-MiniLM-L6-v2",
        "llm": "google/flan-t5-base",
        "langextract_enabled": False  # LangExtract requires API keys
    },
    "azure": {
        "embedding": "text-embedding-3-small",  # Your embedding deployment name
        "llm": "gpt-35-turbo",  # Your chat deployment name  
        "langextract_enabled": False,  # Disabled for comparison testing
        "langextract_model": "gemini-2.0-flash-exp"
        # Azure setup uses OpenAI-compatible format with these env vars:
        # OPENAI_API_KEY = your Azure API key
        # OPENAI_BASE_URL = https://your-resource.openai.azure.com/openai/deployments/your-deployment/
    }
}

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================
AGENTS = {
    "educational_content": {
        "name": "Educational Content Discovery Agent",
        "collection": "educational_content",
        "description": "Discovers educational content with enhanced metadata",
        "enhanced_parsing": True
    },
    "general_query": {
        "name": "General Query Agent",
        "collection": "general_content",
        "description": "Answers general questions about content",
        "enhanced_parsing": False
    }
}
