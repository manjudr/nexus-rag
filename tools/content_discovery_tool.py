from tools.base import BaseTool
from vector_db.base import BaseVectorDB
from vector_db.content_discovery_db import ContentDiscoveryVectorDB
from models.embedding import EmbeddingModel
from models.llm import GenerativeModel
from rank_bm25 import BM25Okapi
from typing import List, Dict
import json
import time
import re
from schemas.content_discovery_schema import ContentDiscoveryResponse, ContentRecommendation, ErrorResponse

class ContentDiscoveryTool(BaseTool):
    """
    Updated Content Discovery Tool that works with the new database architecture
    Returns actual filenames from the database instead of generating fictional ones
    """
    
    def __init__(self, db: BaseVectorDB, embedding_model: EmbeddingModel, llm: GenerativeModel, 
                 name: str, description: str, top_k: int = 5, return_json: bool = False):
        self.db = db
        self.embedding_model = embedding_model
        self.llm = llm
        self.name = name
        self.description = description
        self.top_k = top_k
        self.return_json = return_json

    def _initialize_bm25(self):
        if isinstance(self.db, ContentDiscoveryVectorDB):
            print(f"Tool ({self.name}): Initializing BM25...")
            documents_data = self.db.get_all_documents()
            if not documents_data:
                return None, None
            # Extract just the content for BM25
            documents = [doc["content"] for doc in documents_data]
            tokenized_corpus = [doc.split(" ") for doc in documents]
            return BM25Okapi(tokenized_corpus), documents_data
        return None, None

    def _extract_keywords_from_content(self, content: str, count: int = 8) -> List[str]:
        """Extract meaningful keywords from content."""
        import re
        from collections import Counter
        
        # Remove special characters and split into words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequency and return top keywords
        word_counts = Counter(meaningful_words)
        keywords = [word for word, count in word_counts.most_common(count)]
        
        return keywords

    def _get_content_recommendations(self, query: str, results: List[Dict]) -> Dict:
        """Generate content recommendations with actual database metadata."""
        recommendations = []
        seen_files = set()
        
        for i, result in enumerate(results[:self.top_k]):
            content = result["content"]
            filename = result["filename"]
            page = result["page"]
            
            # Extract keywords from content
            keywords = self._extract_keywords_from_content(content)
            
            # Extract keywords that match the query
            query_lower = query.lower()
            query_words = set(query_lower.split())
            matched_keywords = [kw for kw in keywords if any(qw in kw for qw in query_words)]
            
            # If no query-specific keywords found, use general keywords from content
            if not matched_keywords:
                matched_keywords = keywords[:5]  # Top 5 keywords
            
            file_key = f"{filename}_page_{page}"
            if file_key not in seen_files:
                seen_files.add(file_key)
                
                summary = content[:200] + "..." if len(content) > 200 else content.strip()
                
                # Generate title from filename
                title = filename.replace('.pdf', '').replace('_', ' ').title()
                
                # Generate course from keywords
                course = f"{matched_keywords[0].title()} Studies" if matched_keywords else "General Studies"
                
                recommendation = {
                    "filename": filename,  # ACTUAL filename from database
                    "title": title,
                    "author": "Document Author",  # Could be enhanced with metadata extraction
                    "course": course,
                    "page_number": page,
                    "section": f"Page {page}",
                    "keywords": matched_keywords,
                    "summary": summary,
                    "relevance_score": round(1.0 - (i * 0.1), 2)  # Decreasing relevance
                }
                recommendations.append(recommendation)
        
        return {
            "total_recommendations": len(recommendations),
            "query": query,
            "recommendations": recommendations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def _create_text_response(self, recommendations_data: Dict) -> str:
        """Create human-readable text response"""
        query = recommendations_data["query"]
        recommendations = recommendations_data["recommendations"]
        total = recommendations_data["total_recommendations"]
        
        if not recommendations:
            return f"I couldn't find any relevant content for '{query}'. Please try a different search term."
        
        response = f"🔍 **Content Discovery Results for: '{query}'**\n\n"
        response += f"Found {total} relevant educational resources:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            response += f"**{i}. {rec['title']}**\n"
            response += f"   📄 File: {rec['filename']} (Page {rec['page_number']})\n"  # REAL filename!
            response += f"   📚 Course: {rec['course']}\n"
            response += f"   🏷️  Keywords: {', '.join(rec['keywords'][:5])}\n"
            response += f"   📝 Summary: {rec['summary']}\n"
            response += f"   ⭐ Relevance: {rec['relevance_score']}\n\n"
        
        return response

    def run(self, query: str):
        """Run content discovery with actual database filenames"""
        try:
            bm25, documents_data = self._initialize_bm25()

            print(f"Tool ({self.name}): Creating embedding for query: '{query}'")
            query_embedding = self.embedding_model.create_embedding(query)
            
            print(f"Tool ({self.name}): Searching content database...")
            vector_results = self.db.search(query_embedding, top_k=self.top_k)

            # Use vector results directly (they already have the right format)
            final_results = vector_results
            
            # Optionally enhance with BM25 if available
            if bm25 and documents_data:
                print(f"Tool ({self.name}): Enhancing with BM25 search...")
                tokenized_query = query.split(" ")
                bm25_scores = bm25.get_scores(tokenized_query)
                
                # Get top BM25 results
                top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.top_k]
                bm25_results = [documents_data[i] for i in top_bm25_indices]
                
                # Combine results (prioritize vector search)
                combined_results = vector_results.copy()
                for bm25_result in bm25_results:
                    if bm25_result not in combined_results:
                        combined_results.append(bm25_result)
                
                final_results = combined_results[:self.top_k]

            if not final_results:
                error_msg = f"No relevant content found for '{query}'"
                if self.return_json:
                    error_response = ErrorResponse(
                        query=query,
                        error_code="no_results",
                        error_message=error_msg
                    )
                    return json.dumps(error_response.dict(), indent=2)
                return error_msg

            # Generate recommendations with ACTUAL filenames
            recommendations_data = self._get_content_recommendations(query, final_results)

            if self.return_json:
                # Return structured JSON response
                content_recommendations = [
                    ContentRecommendation(**rec) for rec in recommendations_data["recommendations"]
                ]
                
                response = ContentDiscoveryResponse(
                    query=recommendations_data["query"],
                    status="success",
                    total_results=recommendations_data["total_recommendations"],
                    recommendations=content_recommendations,
                    message="Content discovery completed successfully"
                )
                
                return json.dumps(response.dict(), indent=2)
            else:
                # Return human-readable text
                return self._create_text_response(recommendations_data)

        except Exception as e:
            error_msg = f"Error in content discovery: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.return_json:
                error_response = ErrorResponse(
                    query=query,
                    error_code="processing_error",
                    error_message=error_msg
                )
                return json.dumps(error_response.dict(), indent=2)
            
            return error_msg
