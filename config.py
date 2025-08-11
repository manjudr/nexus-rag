"""
Multi-Collection RAG Configuration
Supports multiple data sources with intelligent routing
"""

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

# Data Source Configurations (Each acts like a separate "table")
AGENT_CONFIGS = {
    "video_agent": {
        "name": "Video Transcripts", 
        "path": "data/video_transcript.txt",
        "collection_name": "video_transcripts",
        "parser": "txt",
        "description": "Use this for questions about video content, transcripts, spoken content, interviews, and audio-visual material.",
        "keywords": ["video", "transcript", "interview", "spoken", "audio", "visual", "recording"]
    },
    "content_discovery_agent": {
        "name": "Content Discovery",
        "path": "data/educational_content/",
        "collection_name": "educational_content",
        "parser": "pdf_with_metadata",
        "description": "Use this to find and recommend educational content across multiple PDFs. Can search by topic, subject, course, board, etc. and provide page numbers and summaries.",
        "keywords": ["learn", "study", "course", "subject", "education", "find content", "recommend", "page", "which pdf", "document", "pdf", "file"]
    }
    # Easy to add more: website_agent, email_agent, etc.
}

# Educational Content Configuration
EDUCATIONAL_CONTENT_CONFIG = {
    "content_directory": "data/educational_content/",
    "metadata_file": "data/educational_content/metadata.json",
    "supported_formats": [".pdf"],
    "extract_page_numbers": True,
    "generate_summaries": True
}
