# --- PDF Agent Config ---
PDF_AGENT_CONFIG = {
    "PDF_PATH": "data/obsrv.pdf",
    "VECTOR_DB_PROVIDER": "milvus",
    "VECTOR_DB_CONFIG": {
        "milvus": {
            "db_path": "milvus_demo.db",
            "collection_name": "pdf_chunks"
        }
    },
    "MODEL_PROVIDER": "local", # "local" or "openai"
    "LOCAL_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    "LOCAL_QA_MODEL": "google/flan-t5-base",
    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
    "OPENAI_QA_MODEL": "gpt-3.5-turbo",
    "MAX_CHUNK_SIZE": 500,
    "TOP_K": 1
}

# --- Image Agent Config (placeholder) ---
IMAGE_AGENT_CONFIG = {
    # Add any specific configs for this agent here
}
