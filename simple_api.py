"""
Simple NexusRAG API Server
A simplified API that initializes models on first request
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NexusRAG Content Discovery API",
    description="RESTful API for intelligent educational content discovery",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class ContentRecommendation(BaseModel):
    filename: str
    title: str
    course: str
    page_number: int
    keywords: List[str]
    summary: str
    relevance_score: float

class QueryResponse(BaseModel):
    query: str
    status: str
    total_results: int
    processing_time_ms: int
    agent_used: str
    recommendations: List[ContentRecommendation]
    message: str
    timestamp: str

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

# Global orchestrator (lazy loading)
orchestrator = None

def get_orchestrator():
    """Lazy load the orchestrator when needed"""
    global orchestrator
    
    if orchestrator is None:
        try:
            # Import and setup
            import main_config as config
            from models.openai_llm import OpenAIGenerativeModel
            from models.openai_embedding import OpenAIEmbeddingModel
            from agents.orchestrator_agent import OrchestratorAgent
            from agents.pdf_content_agent import PDFContentAgent
            from agents.video_transcript_agent import VideoTranscriptAgent
            from agents.general_query_agent import GeneralQueryAgent
            
            # Initialize models
            provider = config.MODEL_PROVIDER
            model_config = config.MODELS[provider]
            
            if provider == "azure":
                llm_model = OpenAIGenerativeModel(model_config["llm"])
                embedding_model = OpenAIEmbeddingModel(model_config["embedding"])
            else:
                raise ValueError("Only Azure OpenAI is supported in this API version")
            
            # Create agents
            agents = []
            for agent_key, agent_config in config.AGENTS.items():
                agent_class = agent_config["class"]
                
                if agent_class == "PDFContentAgent":
                    agent = PDFContentAgent(
                        db_path=config.DB_PATH,
                        embedding_model=embedding_model,
                        llm=llm_model,
                        top_k=config.TOP_K,
                        return_json=True
                    )
                elif agent_class == "VideoTranscriptAgent":
                    agent = VideoTranscriptAgent(
                        db_path=config.DB_PATH,
                        embedding_model=embedding_model,
                        llm=llm_model,
                        top_k=config.TOP_K,
                        return_json=True
                    )
                elif agent_class == "GeneralQueryAgent":
                    agent = GeneralQueryAgent(
                        db_path=config.DB_PATH,
                        embedding_model=embedding_model,
                        llm=llm_model,
                        top_k=config.TOP_K,
                        return_json=True
                    )
                else:
                    continue
                
                agents.append(agent)
            
            # Create orchestrator
            orchestrator = OrchestratorAgent(
                llm=llm_model,
                agents=agents
            )
            
            logger.info("Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise HTTPException(status_code=503, detail=f"System initialization failed: {str(e)}")
    
    return orchestrator

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NexusRAG Content Discovery API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/status")
async def get_status():
    """System status"""
    return {
        "status": "ready",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query_content(request: QueryRequest):
    """Main query endpoint"""
    start_time = datetime.now()
    
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing query: '{request.query}'")
        
        # Get orchestrator (lazy loading)
        orch = get_orchestrator()
        
        # Process query
        result = orch.run(request.query)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Parse result
        if isinstance(result, str):
            # Try to parse as JSON first
            try:
                result_data = json.loads(result)
            except:
                # If not JSON, create a simple response (fallback)
                result_data = {
                    "query": request.query,
                    "status": "success",
                    "recommendations": [{
                        "filename": "response.txt",
                        "title": "AI Response", 
                        "course": "General Knowledge",
                        "page_number": 1,
                        "keywords": [],
                        "summary": result,
                        "relevance_score": 1.0
                    }]
                }
        else:
            result_data = result
        
        # Extract recommendations
        recommendations = []
        for rec in result_data.get("recommendations", []):
            if isinstance(rec, dict):
                recommendation = ContentRecommendation(
                    filename=rec.get("filename", "unknown"),
                    title=rec.get("title", "Untitled"),
                    course=rec.get("course", "General"),
                    page_number=rec.get("page_number", 1),
                    keywords=rec.get("keywords", [])[:5],  # Limit keywords
                    summary=rec.get("summary", "No summary available"),
                    relevance_score=rec.get("relevance_score", 0.0)
                )
                recommendations.append(recommendation)
        
        # Limit results
        recommendations = recommendations[:request.max_results]
        
        response = QueryResponse(
            query=request.query,
            status="success",
            total_results=len(recommendations),
            processing_time_ms=processing_time_ms,
            agent_used="orchestrator",
            recommendations=recommendations,
            message="Query processed successfully",
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Query processed successfully in {processing_time_ms}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/query")
async def query_get(q: str, max_results: int = 5):
    """GET endpoint for queries"""
    request = QueryRequest(query=q, max_results=max_results)
    return await query_content(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "simple_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
