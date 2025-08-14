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
from vector_db.content_discovery_db import ContentDiscoveryVectorDB

# Agents - Keep the orchestrator and content discovery
from agents.orchestrator_agent import OrchestratorAgent
from tools.content_discovery_tool import ContentDiscoveryTool

# Educational content parser - Streamlined
from data_processing.enhanced_educational_parser import EnhancedEducationalContentParser
from data_processing.hybrid_enhancer import HybridEducationalEnhancer
from data_processing.pdf_parser import PDFParser

def setup_llm():
    """Initialize the language model based on configuration"""
    current_models = config.MODELS[config.MODEL_PROVIDER]
    
    if config.MODEL_PROVIDER == "azure":
        # Use regular OpenAI client with Azure endpoint
        return OpenAIGenerativeModel(model_name=current_models["llm"])
    elif config.MODEL_PROVIDER == "cloud":
        return OpenAIGenerativeModel(model_name=current_models["llm"])
    else:
        return HuggingFaceGenerativeModel(model_name=current_models["llm"])

def setup_embedding():
    """Initialize the embedding model based on configuration"""
    current_models = config.MODELS[config.MODEL_PROVIDER]
    
    if config.MODEL_PROVIDER == "azure":
        # Use regular OpenAI client with Azure endpoint  
        return OpenAIEmbeddingModel(model_name=current_models["embedding"])
    elif config.MODEL_PROVIDER == "cloud":
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
            # Use basic content discovery DB for general queries  
            vector_db = ContentDiscoveryVectorDB(
                db_path=config.DB_PATH,
                collection_name=agent_config["collection"],
                top_k=config.TOP_K
            )
            
            # Use same ContentDiscoveryTool but with basic database
            tool = ContentDiscoveryTool(
                db=vector_db,
                embedding_model=embedding_model,
                llm=llm,
                name=agent_config["name"],
                description=agent_config["description"],
                top_k=config.TOP_K,
                return_json=json_output
            )
        
        tools.append(tool)
    
    return tools

def load_educational_content(embedding_model: EmbeddingModel, limit_chunks: int = None):
    """Load educational content with enhanced parsing"""
    print("📚 Loading educational content...")
    if limit_chunks:
        print(f"🔬 Development mode: limiting to {limit_chunks} chunks for quick testing")
    
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
                    
                    # Apply enhanced extraction to each chunk if enabled
                    if langextract_enabled:
                        enhancer = HybridEducationalEnhancer(use_langextract=True)
                        
                        # Limit chunks for development/testing
                        chunks_to_process = chunks[:limit_chunks] if limit_chunks else chunks
                        metadata_to_process = metadata[:limit_chunks] if limit_chunks else metadata
                        
                        print(f"🤖 Applying enhanced extraction to {len(chunks_to_process)} chunks from {pdf_file}...")
                        
                        # Process chunks in smaller groups to avoid overwhelming the API
                        for chunk_idx, chunk in enumerate(chunks_to_process):
                            try:
                                # Get the content from chunk (handle both string and dict formats)
                                chunk_content = chunk if isinstance(chunk, str) else chunk.get('content', str(chunk))
                                
                                # Apply enhancement to this chunk
                                if len(chunk_content) > 100:  # Only enhance meaningful chunks
                                    enhancement_result = enhancer.enhance_educational_content(
                                        chunk_content, f"{pdf_file}_chunk_{chunk_idx}"
                                    )
                                    
                                    # Add enhancement results to metadata
                                    enhanced_metadata = metadata_to_process[chunk_idx] if chunk_idx < len(metadata_to_process) else {}
                                    enhanced_metadata.update({
                                        'source_file': pdf_file,
                                        'file_type': 'pdf',
                                        'enhanced': True,
                                        'enhancement_method': enhancement_result.extraction_method,
                                        'learning_objectives': [obj.get('text', '') for obj in enhancement_result.learning_objectives],
                                        'key_concepts': [concept.get('text', '') for concept in enhancement_result.key_concepts],
                                        'difficulty_level': enhancement_result.difficulty_level,
                                        'study_questions': [q.get('text', '') for q in enhancement_result.study_questions],
                                        'prerequisites': [p.get('text', '') for p in enhancement_result.prerequisites],
                                        'examples': [e.get('text', '') for e in enhancement_result.examples]
                                    })
                                    all_metadata.append(enhanced_metadata)
                                else:
                                    # For small chunks, just add basic metadata
                                    enhanced_metadata = metadata_to_process[chunk_idx] if chunk_idx < len(metadata_to_process) else {}
                                    enhanced_metadata.update({
                                        'source_file': pdf_file,
                                        'file_type': 'pdf',
                                        'enhanced': False
                                    })
                                    all_metadata.append(enhanced_metadata)
                                
                                all_chunks.append(chunk_content)
                                
                                # Add a small delay every few chunks to avoid rate limiting
                                if (chunk_idx + 1) % 5 == 0:
                                    import time
                                    time.sleep(1)
                                    
                            except Exception as e:
                                print(f"⚠️ Enhancement failed for chunk {chunk_idx}: {str(e)}")
                                # Add chunk without enhancement
                                chunk_content = chunk if isinstance(chunk, str) else chunk.get('content', str(chunk))
                                enhanced_metadata = metadata_to_process[chunk_idx] if chunk_idx < len(metadata_to_process) else {}
                                enhanced_metadata.update({
                                    'source_file': pdf_file,
                                    'file_type': 'pdf',
                                    'enhanced': False
                                })
                                all_metadata.append(enhanced_metadata)
                                all_chunks.append(chunk_content)
                    else:
                        # Without enhancement, just add basic metadata
                        chunks_to_process = chunks[:limit_chunks] if limit_chunks else chunks
                        metadata_to_process = metadata[:limit_chunks] if limit_chunks else metadata
                        
                        for i, chunk in enumerate(chunks_to_process):
                            enhanced_metadata = metadata_to_process[i] if i < len(metadata_to_process) else {}
                            enhanced_metadata.update({
                                'source_file': pdf_file,
                                'file_type': 'pdf',
                                'enhanced': False
                            })
                            all_metadata.append(enhanced_metadata)
                            chunk_content = chunk if isinstance(chunk, str) else chunk.get('content', str(chunk))
                            all_chunks.append(chunk_content)
                    
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
        parser.print_help()

if __name__ == "__main__":
    main()
