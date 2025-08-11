"""
Multi-Collection RAG System
Supports intelligent routing between different data sources
"""
import config
import argparse
from models.llm import GenerativeModel
from models.embedding import EmbeddingModel
from models.huggingface_llm import HuggingFaceGenerativeModel
from models.openai_llm import OpenAIGenerativeModel
from models.sentence_transformer import SentenceTransformerEmbeddingModel
from models.openai_embedding import OpenAIEmbeddingModel
from vector_db.milvus_db import MilvusVectorDB
from agents.orchestrator_agent import OrchestratorAgent
from tools.vector_db_query import VectorDBQueryTool
from tools.content_discovery_tool import ContentDiscoveryTool
from data_processing.pdf_parser import PDFParser
from data_processing.text_parser import TextParser
from data_processing.educational_content_parser import EducationalContentParser

def setup_llm() -> GenerativeModel:
    """Setup the language model based on config"""
    if config.MODEL_CONFIG["provider"] == "openai":
        return OpenAIGenerativeModel(model_name=config.MODEL_CONFIG["openai"]["llm"])
    else:
        return HuggingFaceGenerativeModel(model_name=config.MODEL_CONFIG["local"]["llm"])

def setup_embedding_model() -> EmbeddingModel:
    """Setup the embedding model based on config"""
    if config.MODEL_CONFIG["provider"] == "openai":
        return OpenAIEmbeddingModel(model_name=config.MODEL_CONFIG["openai"]["embedding_model"])
    else:
        return SentenceTransformerEmbeddingModel(model_name=config.MODEL_CONFIG["local"]["embedding_model"])

def create_query_tools(llm: GenerativeModel, embedding_model: EmbeddingModel, json_output: bool = False) -> list:
    """Create vector DB query tools for each data source"""
    tools = []
    
    for agent_name, agent_config in config.AGENT_CONFIGS.items():
        # Create separate vector DB for each collection
        vector_db = MilvusVectorDB(
            db_path=config.DB_PATH,
            collection_name=agent_config["collection_name"], 
            top_k=config.TOP_K
        )
        
        # Create appropriate tool based on agent type
        if agent_name == "content_discovery_agent":
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
            # Regular VectorDBQueryTool for other agents
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

def load_data_to_collection(agent_name: str, embedding_model: EmbeddingModel):
    """Load data into a specific collection"""
    if agent_name not in config.AGENT_CONFIGS:
        print(f"❌ Error: Agent '{agent_name}' not found in configuration")
        print(f"📚 Available agents: {', '.join(config.AGENT_CONFIGS.keys())}")
        return
        
    agent_config = config.AGENT_CONFIGS[agent_name]
    
    print(f"📥 Loading data for '{agent_config['name']}'...")
    
    # Create vector DB for this collection
    vector_db = MilvusVectorDB(
        db_path=config.DB_PATH,
        collection_name=agent_config["collection_name"],
        top_k=config.TOP_K
    )
    
    # Setup the collection (this will recreate if exists)
    vector_db.setup(dimension=embedding_model.get_embedding_dimension())
    
    # Parse the data based on file type
    if agent_config["parser"] == "pdf":
        parser = PDFParser(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = parser.parse_pdf(agent_config["path"])
    elif agent_config["parser"] == "txt":
        parser = TextParser(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = parser.parse_txt(agent_config["path"])
    elif agent_config["parser"] == "pdf_with_metadata":
        parser = EducationalContentParser(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks, metadata = parser.parse_educational_content(
            config.EDUCATIONAL_CONTENT_CONFIG["content_directory"],
            config.EDUCATIONAL_CONTENT_CONFIG["metadata_file"]
        )
        print(f"📚 Loaded {len(chunks)} chunks from {len(set(m['filename'] for m in metadata))} educational PDFs")
    else:
        print(f"❌ Error: Unknown parser type '{agent_config['parser']}'")
        return
    
    print(f"📄 Processing {len(chunks)} chunks...")
    
    # Create embeddings
    print("🔄 Creating embeddings...")
    embeddings = embedding_model.create_embedding(chunks)
    
    # Insert into vector DB
    print("💾 Inserting into vector database...")
    vector_db.insert(data=chunks, embeddings=embeddings)
    
    print(f"✅ Successfully loaded {len(chunks)} chunks into '{agent_config['name']}' collection!")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="🚀 Multi-Collection RAG System")
    parser.add_argument("--load", type=str, help="Load data for specific agent (pdf_agent, video_agent)")
    parser.add_argument("--query", type=str, help="Ask a question to the RAG system") 
    parser.add_argument("--list", action="store_true", help="List available data sources")
    parser.add_argument("--json", action="store_true", help="Return response in JSON format (for Content Discovery agent)")
    
    args = parser.parse_args()
    
    # List available data sources
    if args.list:
        print("📚 Available Data Sources:")
        for agent_name, agent_config in config.AGENT_CONFIGS.items():
            print(f"  • {agent_name}: {agent_config['name']} - {agent_config['description']}")
        return
    
    # Setup models
    print("🔧 Setting up models...")
    llm = setup_llm()
    embedding_model = setup_embedding_model()
    
    # Load data into specific collection
    if args.load:
        load_data_to_collection(args.load, embedding_model)
        return
    
    # Query the system
    if args.query:
        print("🎯 Setting up query tools...")
        query_tools = create_query_tools(llm, embedding_model, json_output=args.json)
        
        print("🤖 Initializing orchestrator...")
        orchestrator = OrchestratorAgent(llm=llm, tools=query_tools)
        
        print(f"❓ Processing query: '{args.query}'")
        result = orchestrator.run(args.query)
        
        if args.json:
            # For JSON output, just print the result directly (it's already JSON)
            print(result)
        else:
            # For regular output, format nicely
            print("\n" + "="*50)
            print("🎉 FINAL ANSWER:")
            print("="*50)
            print(result)
        return
    
    # Show help if no arguments
    print("🔍 Multi-Collection RAG System")
    print("\nUsage:")
    print("  --list                    List available data sources")
    print("  --load pdf_agent          Load PDF documents")
    print("  --load video_agent        Load video transcripts") 
    print("  --query 'your question'   Ask a question")
    print("  --json                    Return JSON response (Content Discovery)")
    print("\nExample:")
    print("  python app.py --load pdf_agent")
    print("  python app.py --query 'What are the key metrics in the report?'")
    print("  python app.py --query 'I want to learn about photosynthesis' --json")

if __name__ == "__main__":
    main()
