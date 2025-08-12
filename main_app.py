"""
Streamlined NexusRAG Application
Uses Milvus, LangExtract, and multiple agents but with clean organization
Focuses on educational content processing
"""

import argparse
import os
import main_config as config

# Core imports - keep the powerful stuff
from models.llm import GenerativeModel
from models.embedding import EmbeddingModel
from models.huggingface_llm import HuggingFaceGenerativeModel
from models.openai_llm import OpenAIGenerativeModel
from models.sentence_transformer import SentenceTransformerEmbeddingModel
from models.openai_embedding import OpenAIEmbeddingModel

# Vector DB - Keep Milvus
from vector_db.enhanced_content_discovery_db import EnhancedContentDiscoveryVectorDB

# Agents - Keep the orchestrator and content discovery
from agents.orchestrator_agent import OrchestratorAgent
from tools.content_discovery_tool import ContentDiscoveryTool
from tools.vector_db_query import VectorDBQueryTool

# Educational content parser - Streamlined
from data_processing.enhanced_educational_parser import EnhancedEducationalContentParser
from data_processing.pdf_parser import PDFParser

def setup_llm():
    """Initialize the language model based on configuration"""
    current_models = config.MODELS[config.MODEL_PROVIDER]
    
    if config.MODEL_PROVIDER == "cloud":
        return OpenAIGenerativeModel(model_name=current_models["llm"])
    else:
        return HuggingFaceGenerativeModel(model_name=current_models["llm"])

def setup_embedding():
    """Initialize the embedding model based on configuration"""
    current_models = config.MODELS[config.MODEL_PROVIDER]
    
    if config.MODEL_PROVIDER == "cloud":
        return OpenAIEmbeddingModel(model_name=current_models["embedding"])
    else:
        return SentenceTransformerEmbeddingModel(model_name=current_models["embedding"])

def create_agent_tools(llm: GenerativeModel, embedding_model: EmbeddingModel, json_output: bool = False) -> list:
    """Create tools for each agent"""
    tools = []
    current_models = config.MODELS[config.MODEL_PROVIDER]
    langextract_enabled = current_models.get("langextract_enabled", False)
    
    for agent_key, agent_config in config.AGENTS.items():
        print(f"🔧 Setting up {agent_config['name']}...")
        
        # Create enhanced vector DB for educational content
        if agent_config.get("enhanced_parsing", False):
            vector_db = EnhancedContentDiscoveryVectorDB(
                db_path=config.DB_PATH,
                collection_name=agent_config["collection"],
                top_k=config.TOP_K,
                use_langextract=langextract_enabled
            )
            
            # Use content discovery tool for enhanced features
            tool = ContentDiscoveryTool(
                db=vector_db,
                embedding_model=embedding_model,
                llm=llm,
                name=agent_config["name"],
                description=agent_config["description"],
                top_k=config.TOP_K,
                return_json=json_output
            )
        else:
            # Use basic vector DB for general queries
            from vector_db.milvus_db import MilvusVectorDB
            vector_db = MilvusVectorDB(
                db_path=config.DB_PATH,
                collection_name=agent_config["collection"],
                top_k=config.TOP_K
            )
            
            tool = VectorDBQueryTool(
                db=vector_db,
                embedding_model=embedding_model,
                llm=llm,
                name=agent_config["name"],
                description=agent_config["description"],
                top_k=config.TOP_K
            )
        
        tools.append(tool)
    
    return tools

def load_educational_content(embedding_model: EmbeddingModel):
    """Load educational content with enhanced parsing"""
    print("📚 Loading educational content...")
    
    current_models = config.MODELS[config.MODEL_PROVIDER]
    langextract_enabled = current_models.get("langextract_enabled", False)
    
    # Setup enhanced parser
    parser = EnhancedEducationalContentParser(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        use_langextract=langextract_enabled
    )
    
    # Setup enhanced vector DB
    vector_db = EnhancedContentDiscoveryVectorDB(
        db_path=config.DB_PATH,
        collection_name="educational_content",
        top_k=config.TOP_K,
        use_langextract=langextract_enabled
    )
    
    vector_db.setup(dimension=embedding_model.get_embedding_dimension())
    
    # Check if we have PDF files to process directly
    pdf_directory = config.CONTENT_DIRECTORY
    if os.path.exists(pdf_directory):
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if pdf_files:
            print(f"📄 Found {len(pdf_files)} PDF files to process...")
            all_chunks = []
            all_metadata = []
            
            # Use PDFParser for parsing actual PDF files
            pdf_parser = PDFParser(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_directory, pdf_file)
                print(f"📖 Processing: {pdf_file}")
                
                try:
                    # Use the PDF parser to extract content
                    chunks, metadata = pdf_parser.parse_pdf(pdf_path)
                    
                    # Add enhanced metadata
                    for i, chunk in enumerate(chunks):
                        enhanced_metadata = metadata[i] if i < len(metadata) else {}
                        enhanced_metadata.update({
                            'source_file': pdf_file,
                            'file_type': 'pdf',
                            'enhanced': True
                        })
                        all_metadata.append(enhanced_metadata)
                        all_chunks.append(chunk)
                    
                    print(f"✅ Processed {pdf_file}: {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"❌ Error processing {pdf_file}: {str(e)}")
            
            if all_chunks:
                print(f"📄 Processing {len(all_chunks)} total chunks...")
                
                # Create embeddings
                print("🔄 Creating embeddings...")
                embeddings = embedding_model.create_embedding(all_chunks)
                
                # Insert into database
                print("💾 Inserting into vector database...")
                vector_db.insert(data=all_chunks, embeddings=embeddings, metadata=all_metadata)
                
                print(f"✅ Successfully loaded {len(all_chunks)} chunks from {len(pdf_files)} PDF files")
                return
    
    # Fallback to original method if no PDFs found
    # Parse content
    chunks, metadata = parser.parse_educational_content_enhanced(
        config.CONTENT_DIRECTORY,
        config.METADATA_FILE
    )
    
    print(f"📄 Processing {len(chunks)} enhanced chunks...")
    
    # Create embeddings
    print("🔄 Creating embeddings...")
    embeddings = embedding_model.create_embedding(chunks)
    
    # Insert into database
    print("💾 Inserting into vector database...")
    vector_db.insert(data=chunks, embeddings=embeddings, metadata=metadata)
    
    print(f"✅ Successfully loaded {len(chunks)} enhanced chunks")

def query_system(query: str, llm: GenerativeModel, embedding_model: EmbeddingModel, json_output: bool = False):
    """Query the system with orchestrator"""
    print(f"🔍 Processing query: '{query}'...")
    
    # Create agent tools
    tools = create_agent_tools(llm, embedding_model, json_output)
    
    # Create orchestrator
    orchestrator = OrchestratorAgent(llm=llm, tools=tools)
    
    # Process query
    result = orchestrator.run(query)
    return result

def main():
    """Main application"""
    parser = argparse.ArgumentParser(description="Streamlined Educational Content RAG")
    parser.add_argument("--load", help="Load educational content", action="store_true")
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
        
    elif args.query:
        result = query_system(args.query, llm, embedding_model, args.json)
        print(result)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
