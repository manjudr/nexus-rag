"""
JSON Schema for Content Discovery API Response
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class ContentRecommendation(BaseModel):
    """Schema for a single content recommendation"""
    filename: str = Field(..., description="Name of the PDF file")
    title: str = Field(..., description="Title of the educational content")
    author: str = Field(..., description="Author of the content")
    course: str = Field(..., description="Course subject (Biology, Physics, Mathematics, etc.)")
    page_number: int = Field(..., description="Page number where content is found")
    section: str = Field(..., description="Section or chapter name")
    summary: str = Field(..., description="Brief summary of the content")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score from 0.0 to 1.0")
    keywords: Optional[List[str]] = Field(default=None, description="Key topics covered in this content")

class ContentDiscoveryResponse(BaseModel):
    """Schema for the complete Content Discovery API response"""
    query: str = Field(..., description="Original user query")
    status: str = Field(..., description="Response status: 'success' or 'error'")
    total_results: int = Field(..., description="Total number of content recommendations found")
    recommendations: List[ContentRecommendation] = Field(..., description="List of content recommendations")
    message: Optional[str] = Field(default=None, description="Additional message or tip for the user")
    processing_time_ms: Optional[float] = Field(default=None, description="Time taken to process the query in milliseconds")

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    query: str = Field(..., description="Original user query")
    status: str = Field(default="error", description="Response status")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    total_results: int = Field(default=0, description="Number of results (0 for errors)")
    recommendations: List[ContentRecommendation] = Field(default=[], description="Empty list for errors")

# Example JSON Response Schema:
EXAMPLE_RESPONSE = {
    "query": "I want to learn about photosynthesis",
    "status": "success",
    "total_results": 3,
    "recommendations": [
        {
            "filename": "biology_plants.pdf",
            "title": "Plant Biology and Growth",
            "author": "Dr. Sarah Johnson",
            "course": "Biology",
            "page_number": 4,
            "section": "Chapter 4: Photosynthesis Process",
            "summary": "Photosynthesis is the process by which plants make their own food using carbon dioxide, water, sunlight, and chlorophyll...",
            "relevance_score": 1.0,
            "keywords": ["photosynthesis", "chlorophyll", "carbon dioxide", "sunlight"]
        },
        {
            "filename": "biology_plants.pdf",
            "title": "Plant Biology and Growth", 
            "author": "Dr. Sarah Johnson",
            "course": "Biology",
            "page_number": 2,
            "section": "Chapter 2: How Plants Grow",
            "summary": "Plant growth occurs through cell division and elongation. Light is essential for photosynthesis...",
            "relevance_score": 0.8,
            "keywords": ["plant growth", "photosynthesis", "light"]
        }
    ],
    "message": "Found content about photosynthesis in biology materials. You can ask specific questions about any of these topics for more details.",
    "processing_time_ms": 245.6
}
