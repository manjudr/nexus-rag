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

# Content paths - can be easily changed for different projects
CONTENT_DIRECTORY = "data/educational_content/pdfs"  # Change this for your content
METADATA_FILE = "data/educational_content/metadata.json"  # Change this for your metadata

# =============================================================================
# MODEL CONFIGURATION (Simple and clear)
# =============================================================================
# Choose one provider: "local" (free, offline) or "azure" (Azure OpenAI)
MODEL_PROVIDER = "azure"

# Model definitions for each provider - easily customizable
MODELS = {
    "local": {
        "embedding": "all-MiniLM-L6-v2",
        "llm": "google/flan-t5-base",
        "langextract_enabled": False  # LangExtract requires API keys
    },
    "azure": {
        "embedding": "text-embedding-3-small",  # Your embedding deployment name
        "llm": "gpt-35-turbo",  # Your chat deployment name  
        "langextract_enabled": True,  # ✅ ENABLED for testing with LangExtract
        "langextract_model": "gpt-35-turbo"  # Use Azure OpenAI for LangExtract too
        # Azure setup uses OpenAI-compatible format with these env vars:
        # OPENAI_API_KEY = your Azure API key
        # AZURE_OPENAI_ENDPOINT = https://your-resource.openai.azure.com/
        # AZURE_CHAT_DEPLOYMENT = gpt-35-turbo
        # AZURE_EMBEDDING_DEPLOYMENT = text-embedding-3-small
    }
}

# =============================================================================
# AGENT CONFIGURATION - Simple Agent-Based Architecture
# =============================================================================
AGENTS = {
    "pdf_content": {
        "name": "PDF Content Agent",
        "class": "PDFContentAgent",
        "collection": "pdf_educational_content",  # Simple, clear collection name
        "description": "Specialized agent for educational PDF content with enhanced metadata extraction"
    },
    "video_transcript": {
        "name": "Video Transcript Agent", 
        "class": "VideoTranscriptAgent",
        "collection": "video_transcripts",  # Simple, clear collection name
        "description": "Specialized agent for video transcript analysis with speaker detection"
    },
    "general_query": {
        "name": "General Query Agent",
        "class": "GeneralQueryAgent", 
        "collection": "general_content",  # Simple, clear collection name
        "description": "General purpose agent for non-specialized queries"
    }
}
