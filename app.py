import config
import os
from data_processing.pdf_parser import extract_text_from_pdf, chunk_text
from vector_db.milvus_db import MilvusVectorDB
from tools.vector_db_query import VectorDBQueryTool
from tools.image_analyzer import ImageAnalyzerTool # Placeholder
from agents.generative_agent import GenerativeAgent
from agents.image_agent import ImageAgent # Placeholder
from agents.orchestrator_agent import OrchestratorAgent

# --- Model Imports ---
from models.sentence_transformer import SentenceTransformerEmbeddingModel
from models.huggingface_llm import HuggingFaceGenerativeModel
from models.openai_embedding import OpenAIEmbeddingModel
from models.openai_llm import OpenAIGenerativeModel

def setup_pdf_agent():
    """Initializes and sets up the PDF processing agent and its components."""
    cfg = config.PDF_AGENT_CONFIG
    
    # --- 0. Pre-flight check ---
    if not os.path.exists(cfg["PDF_PATH"]):
        print(f"Error: PDF file not found at '{cfg['PDF_PATH']}'")
        return None

    # --- 1. Initialize Models ---
    print(f"Initializing PDF Agent models with '{cfg['MODEL_PROVIDER']}' provider...")
    if cfg["MODEL_PROVIDER"] == "openai":
        embedding_model = OpenAIEmbeddingModel(model_name=cfg["OPENAI_EMBEDDING_MODEL"])
        llm = OpenAIGenerativeModel(model_name=cfg["OPENAI_QA_MODEL"])
    else:
        embedding_model = SentenceTransformerEmbeddingModel(model_name=cfg["LOCAL_EMBEDDING_MODEL"])
        llm = HuggingFaceGenerativeModel(model_name=cfg["LOCAL_QA_MODEL"])

    # --- 2. Initialize Vector DB ---
    print(f"Initializing Vector DB with '{cfg['VECTOR_DB_PROVIDER']}' provider...")
    if cfg["VECTOR_DB_PROVIDER"] == "milvus":
        db_config = cfg["VECTOR_DB_CONFIG"]["milvus"]
        vector_db = MilvusVectorDB(
            db_path=db_config["db_path"],
            collection_name=db_config["collection_name"],
            top_k=cfg["TOP_K"]
        )
    else:
        raise ValueError(f"Invalid VECTOR_DB_PROVIDER: {cfg['VECTOR_DB_PROVIDER']}.")

    # --- 3. Data Processing and DB Setup ---
    print("Processing PDF and setting up Vector DB for PDF Agent...")
    text = extract_text_from_pdf(cfg["PDF_PATH"])
    chunks = chunk_text(text, cfg["MAX_CHUNK_SIZE"])
    
    vector_db.setup(dimension=embedding_model.get_embedding_dimension())
    
    print("Creating embeddings for PDF chunks...")
    embeddings = [embedding_model.create_embedding(c) for c in chunks]
    vector_db.insert(chunks, embeddings)

    # --- 4. Initialize Agent and Tools ---
    print("Initializing PDF agent and tools...")
    tool = VectorDBQueryTool(db=vector_db, embedding_model=embedding_model)
    return GenerativeAgent(llm=llm, tool=tool)

def setup_image_agent():
    """Initializes and sets up the Image processing agent."""
    print("Initializing Image Agent (placeholder)...")
    # In the future, this would initialize real image models, tools, etc.
    image_tool = ImageAnalyzerTool()
    return ImageAgent(tool=image_tool)

def main():
    print("--- Starting Multi-Agent System ---")
    
    pdf_agent = setup_pdf_agent()
    if not pdf_agent:
        print("Failed to initialize PDF Agent. Exiting.")
        return

    image_agent = setup_image_agent()

    orchestrator = OrchestratorAgent(pdf_agent=pdf_agent, image_agent=image_agent)

    print("\nSetup complete. Orchestrator is ready.")
    print("You can now ask questions. Try 'pdf: your question' or 'image: describe this'.")
    print("-" * 50)

    # --- Interactive Q&A loop ---
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not question.strip():
            continue

        answer = orchestrator.run(question)
        
        print("\nAnswer:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()
