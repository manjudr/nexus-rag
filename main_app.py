"""
Streamlined NexusRAG Application
Uses Milvus, LangExtract, and multiple agents but with clean organization
Focuses on educational content processing
"""

import argparse
import main_config as config
from typing import List, Dict

# Core imports
from data_processing.pdf_parser import PDFParser

# Vector DB imports
from vector_db.content_discovery_db import ContentDiscoveryVectorDB
from vector_db.enhanced_content_discovery_db import EnhancedContentDiscoveryVectorDB

# Tools
from tools.content_discovery_tool import ContentDiscoveryTool

# Models
from models.llm import GenerativeModel
from models.embedding import EmbeddingModel

# Agents - Proper agent-based architecture
from agents.orchestrator_agent import OrchestratorAgent
from agents.pdf_content_agent import PDFContentAgent
from agents.video_transcript_agent import VideoTranscriptAgent
from agents.general_query_agent import GeneralQueryAgent

def setup_llm() -> GenerativeModel:
    """Initialize the language model based on configuration"""
    provider = config.MODEL_PROVIDER
    model_config = config.MODELS[provider]
    
    if provider == "local":
        # Local model setup
        from models.huggingface_llm import HuggingFaceLLM
        return HuggingFaceLLM(model_config["llm"])
    
    elif provider == "azure":
        # Azure OpenAI setup
        from models.openai_llm import OpenAIGenerativeModel
        return OpenAIGenerativeModel(model_config["llm"])
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

def setup_embedding() -> EmbeddingModel:
    """Initialize the embedding model based on configuration"""
    provider = config.MODEL_PROVIDER
    model_config = config.MODELS[provider]
    
    if provider == "local":
        # Local embedding model
        from models.sentence_transformer import SentenceTransformerEmbedding
        return SentenceTransformerEmbedding(model_config["embedding"])
    
    elif provider == "azure":
        # Azure OpenAI embedding
        from models.openai_embedding import OpenAIEmbeddingModel
        return OpenAIEmbeddingModel(model_config["embedding"])
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

def create_specialized_agents(llm: GenerativeModel, embedding_model: EmbeddingModel) -> list:
    """Create specialized agents for different content types"""
    agents = []
    
    for agent_key, agent_config in config.AGENTS.items():
        print(f"🤖 Setting up {agent_config['name']}...")
        
        agent_class = agent_config["class"]
        
        # Create the appropriate specialized agent
        if agent_class == "PDFContentAgent":
            agent = PDFContentAgent(
                db_path=config.DB_PATH,
                embedding_model=embedding_model,
                llm=llm,
                top_k=config.TOP_K
            )
        elif agent_class == "VideoTranscriptAgent":
            agent = VideoTranscriptAgent(
                db_path=config.DB_PATH,
                embedding_model=embedding_model,
                llm=llm,
                top_k=config.TOP_K
            )
        elif agent_class == "GeneralQueryAgent":
            agent = GeneralQueryAgent(
                db_path=config.DB_PATH,
                embedding_model=embedding_model,
                llm=llm,
                top_k=config.TOP_K
            )
        else:
            print(f"⚠️  Unknown agent class: {agent_class}")
            continue
            
        agents.append(agent)
    
    return agents

def load_educational_content(embedding_model: EmbeddingModel, limit_chunks: int = None):
    """Load educational content into PDF agent's dedicated collection"""
    print("📚 Loading educational content into PDF agent's collection...")
    if limit_chunks:
        print(f"📊 Loading limited to {limit_chunks} chunks for testing")
    
    # Get configuration - centralized in MODELS section
    current_models = config.MODELS[config.MODEL_PROVIDER]
    langextract_enabled = current_models.get("langextract_enabled", False)
    
    # Create appropriate database based on LangExtract setting
    if langextract_enabled:
        # Use enhanced database with LangExtract
        vector_db = EnhancedContentDiscoveryVectorDB(
            db_path=config.DB_PATH,
            collection_name="pdf_educational_content",  # PDF agent's collection
            top_k=config.TOP_K,
            use_langextract=langextract_enabled
        )
        print(f"✅ Using ENHANCED database with LangExtract: {langextract_enabled}")
    else:
        # Use basic database for fast loading
        vector_db = ContentDiscoveryVectorDB(
            db_path=config.DB_PATH,
            collection_name="pdf_educational_content",  # PDF agent's collection
            top_k=config.TOP_K
        )
        print(f"✅ Using BASIC database (fast mode)")

    # Parse PDF content 
    print("📖 Parsing PDF content...")
    pdf_parser = PDFParser(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    # Get all PDF files and parse them
    import os
    all_pdf_files = []
    
    # Search recursively for PDF files
    for root, dirs, files in os.walk(config.CONTENT_DIRECTORY):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                all_pdf_files.append((pdf_path, file))
    
    if not all_pdf_files:
        print("❌ No PDF files found in content directory")
        return
    
    print(f"📁 Found {len(all_pdf_files)} PDF files")
    
    all_chunks = []
    for pdf_path, pdf_file in all_pdf_files:
        print(f"📄 Processing: {pdf_file}")
        chunks, metadata = pdf_parser.parse_pdf(pdf_path)
        
        # Convert to the expected format
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "content": chunk,
                "filename": pdf_file,
                "page": metadata[i]["page"] if i < len(metadata) else 1,
                "chunk_id": i
            }
            all_chunks.append(chunk_data)
    
    chunks = all_chunks
    
    if not chunks:
        print("❌ No content found to load")
        return
    
    # Limit chunks if specified
    if limit_chunks:
        chunks = chunks[:limit_chunks]
    
    print(f"📄 Found {len(chunks)} content chunks")
    
    # Create embeddings
    print("🔗 Creating embeddings...")
    embeddings = embedding_model.create_embedding([chunk["content"] for chunk in chunks])
    
    # Prepare metadata
    metadata = []
    for chunk in chunks:
        meta = {
            "filename": chunk["filename"],
            "page": chunk.get("page", 1),
            "chunk_id": chunk.get("chunk_id", 0),
            "content_type": "educational_pdf",
            "source": "pdf_content_agent"
        }
        metadata.append(meta)
    
    # Insert into PDF agent's database
    print("💾 Inserting into PDF agent's vector database...")
    
    # First, set up the collection with the embedding dimension
    if embeddings:
        dimension = len(embeddings[0])
        vector_db.setup(dimension)
    
    # Extract just the content strings for the insert method
    content_strings = [chunk["content"] for chunk in chunks]
    
    # Use enhanced insert if available, otherwise regular insert
    if hasattr(vector_db, 'insert_enhanced') and langextract_enabled:
        print("🚀 Using ENHANCED INSERT with LangExtract metadata extraction...")
        vector_db.insert_enhanced(data=content_strings, embeddings=embeddings, metadata=metadata, enhance_content=True)
    else:
        print("📦 Using basic insert (no LangExtract enhancement)")
        vector_db.insert(data=content_strings, embeddings=embeddings, metadata=metadata)
    
    print(f"✅ Successfully loaded {len(chunks)} chunks into PDF agent collection")

def query_system(query: str, llm: GenerativeModel, embedding_model: EmbeddingModel, json_output: bool = False):
    """Query the system with orchestrator and specialized agents"""
    print(f"🔍 Processing query: '{query}'...")
    
    # Create specialized agents
    agents = create_specialized_agents(llm, embedding_model)
    
    # Create orchestrator with agents
    orchestrator = OrchestratorAgent(llm=llm, agents=agents)
    
    # Process query
    result = orchestrator.run(query)
    return result

def main():
    """Main application"""
    parser = argparse.ArgumentParser(description="Streamlined Educational Content RAG")
    parser.add_argument("--load", help="Load educational content", action="store_true")
    parser.add_argument("--load-small", help="Load small subset for testing (50 chunks)", action="store_true")
    parser.add_argument("--query", help="Query the system", type=str)
    parser.add_argument("--json", help="Return JSON response", action="store_true")
    parser.add_argument("--status", help="Show system status", action="store_true")
    
    args = parser.parse_args()
    
    if args.status:
        print("🏗️ **Clean NexusRAG Status**")
        print(f"📊 **Configuration:**")
        current_models = config.MODELS[config.MODEL_PROVIDER]
        langextract_status = "Enabled" if current_models.get("langextract_enabled", False) else "Disabled"
        print(f"   • Model Provider: {config.MODEL_PROVIDER}")
        print(f"   • LangExtract: {langextract_status}")
        print(f"   • Agents: {', '.join(config.AGENTS.keys())}")
        print(f"   • Content Directory: {config.CONTENT_DIRECTORY}")
        return
    
    # Setup models
    print("🔧 Setting up models...")
    llm = setup_llm()
    embedding_model = setup_embedding()
    
    if args.load:
        load_educational_content(embedding_model)
        
    elif args.load_small:
        load_educational_content(embedding_model, limit_chunks=50)
        
    elif args.query:
        result = query_system(args.query, llm, embedding_model, args.json)
        print(result)
        
    else:
        print("Please specify --load, --load-small, --query, or --status")

if __name__ == "__main__":
    main()
