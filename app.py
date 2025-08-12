"""
Updated Multi-Collection RAG System with Flexible Database Architecture
Supports agent-specific database schemas and configurations
"""
import config
import argparse
from models.llm import GenerativeModel
from models.embedding import EmbeddingModel
from models.huggingface_llm import HuggingFaceGenerativeModel
from models.openai_llm import OpenAIGenerativeModel
from models.sentence_transformer import SentenceTransformerEmbeddingModel
from models.openai_embedding import OpenAIEmbeddingModel
from vector_db.factory import VectorDBFactory
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
    """Create vector DB query tools using the new factory pattern"""
    tools = []
    
    for agent_name, agent_config in config.AGENT_CONFIGS_V2.items():
        # Create appropriate vector DB using factory
        vector_db = VectorDBFactory.create_db(
            agent_type=agent_name,
            db_path=config.DB_PATH,
            collection_name=agent_config["collection_name"], 
            top_k=config.TOP_K
        )
        
        # Create appropriate tool based on agent type
        if agent_name in ["content_discovery_agent", "educational_content_agent"]:
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
    """Load data into a specific collection using new architecture"""
    if agent_name not in config.AGENT_CONFIGS_V2:
        print(f"❌ Error: Agent '{agent_name}' not found in configuration")
        print(f"📚 Available agents: {', '.join(config.AGENT_CONFIGS_V2.keys())}")
        return
        
    agent_config = config.AGENT_CONFIGS_V2[agent_name]
    
    print(f"📥 Loading data for '{agent_config['name']}'...")
    
    # Create vector DB using factory
    vector_db = VectorDBFactory.create_db(
        agent_type=agent_name,
        db_path=config.DB_PATH,
        collection_name=agent_config["collection_name"],
        top_k=config.TOP_K
    )
    
    # Setup the collection
    vector_db.setup(dimension=embedding_model.get_embedding_dimension())
    
    # Parse the data based on file type
    metadata = None
    if agent_config["parser"] == "pdf":
        parser = PDFParser(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks, metadata = parser.parse_pdf(agent_config["path"])
    elif agent_config["parser"] == "txt":
        parser = TextParser(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = parser.parse_txt(agent_config["path"])
        # Generate basic metadata for txt files
        import os
        filename = os.path.basename(agent_config["path"])
        metadata = [{"filename": filename, "page": i+1} for i in range(len(chunks))]
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
    
    # Insert into vector DB with metadata if required
    print("💾 Inserting into vector database...")
    db_config = agent_config.get("db_config", {})
    if db_config.get("metadata_required", False) and metadata:
        vector_db.insert(data=chunks, embeddings=embeddings, metadata=metadata)
    else:
        vector_db.insert(data=chunks, embeddings=embeddings)
    
    print(f"✅ Successfully loaded {len(chunks)} chunks for '{agent_config['name']}'")

def main():
    parser = argparse.ArgumentParser(description="Multi-Collection RAG System")
    parser.add_argument("--load-data", help="Load data for a specific agent", type=str)
    parser.add_argument("--query", help="Query the system", type=str)
    parser.add_argument("--json", help="Return JSON response", action="store_true")
    parser.add_argument("--architecture", help="Show new architecture info", action="store_true")
    
    args = parser.parse_args()
    
    if args.architecture:
        print("🏗️  **NexusRAG Database Architecture**")
        print("\n📊 **Supported Agent Types:**")
        for agent_type in VectorDBFactory.get_supported_agents():
            schema = VectorDBFactory.AGENT_SCHEMAS.get(agent_type, {})
            print(f"   • {agent_type}: {list(schema.keys()) if schema else 'No metadata'}")
        
        print("\n📁 **Available Configurations:**")
        for agent_name, agent_config in config.AGENT_CONFIGS_V2.items():
            db_type = agent_config["db_config"]["db_type"]
            print(f"   • {agent_name}: Uses {db_type} database")
        return
    
    # Setup models
    llm = setup_llm()
    embedding_model = setup_embedding_model()
    
    if args.load_data:
        load_data_to_collection(args.load_data, embedding_model)
    elif args.query:
        tools = create_query_tools(llm, embedding_model, json_output=args.json)
        
        # Create orchestrator agent
        orchestrator = OrchestratorAgent(
            llm=llm,
            tools=tools
        )
        
        result = orchestrator.run(args.query)
        print(result)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
