from tools.base import BaseTool
from vector_db.base import BaseVectorDB
from models.embedding import EmbeddingModel
from models.llm import GenerativeModel
from rank_bm25 import BM25Okapi
from vector_db.milvus_db import MilvusVectorDB
from typing import List, Dict
import json
import time
import re
from schemas.content_discovery_schema import ContentDiscoveryResponse, ContentRecommendation, ErrorResponse

class ContentDiscoveryTool(BaseTool):
    def __init__(self, db: BaseVectorDB, embedding_model: EmbeddingModel, llm: GenerativeModel, 
                 name: str, description: str, top_k: int = 5, return_json: bool = False, metadata_store: Dict = None):
        self.db = db
        self.embedding_model = embedding_model
        self.llm = llm
        self.name = name
        self.description = description
        self.top_k = top_k
        self.return_json = return_json
        self.metadata_store = metadata_store or {}  # Store for chunk metadata

    def _initialize_bm25(self):
        if isinstance(self.db, MilvusVectorDB):
            print(f"Tool ({self.name}): Initializing BM25...")
            documents = self.db.get_all_documents()
            if not documents:
                return None, None
            tokenized_corpus = [doc.split(" ") for doc in documents]
            return BM25Okapi(tokenized_corpus), documents
        return None, None

    def _extract_metadata_from_chunk(self, chunk: str, chunk_index: int) -> Dict:
        """Extract metadata from chunk content using simple, universal patterns."""
        import re
        from collections import Counter
        
        # Initialize with defaults
        title = "Educational Content"
        author = "Unknown Author"
        course = "General Studies"
        section = "Content"
        page_number = 1
        
        # Try to extract title from chunk content
        lines = chunk.strip().split('\n')
        first_line = lines[0] if lines else ""
        
        # Extract title from first line if it looks like a title
        if first_line and len(first_line) < 100:  # Reasonable title length
            if first_line.isupper() or first_line.istitle():
                title = first_line
            elif any(first_line.startswith(prefix) for prefix in ["Chapter", "Section", "Unit", "Lesson"]):
                section = first_line
                # Look for title in next line
                if len(lines) > 1 and len(lines[1]) > 5:
                    title = lines[1]
        
        # Extract chapter/page information
        chapter_match = re.search(r'chapter\s+(\d+)', chunk.lower())
        if chapter_match:
            page_number = int(chapter_match.group(1))
            section = f"Chapter {page_number}"
        
        # Extract keywords without any predefined categories
        # Just find the most meaningful words in the content
        words = re.findall(r'\b[A-Za-z]{3,}\b', chunk)
        
        # Simple stopword removal - basic English stopwords
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'they', 'them', 'their', 'there', 'then',
            'than', 'when', 'where', 'why', 'how', 'what', 'who', 'which', 'whom', 'whose',
            'all', 'any', 'some', 'many', 'much', 'more', 'most', 'other', 'another',
            'such', 'only', 'own', 'same', 'so', 'very', 'just', 'now', 'here', 'way',
            'get', 'got', 'make', 'made', 'take', 'took', 'come', 'came', 'go', 'went',
            'see', 'saw', 'know', 'knew', 'think', 'say', 'said', 'tell', 'told', 'ask',
            'work', 'use', 'used', 'find', 'give', 'gave', 'turn', 'put', 'end', 'why',
            'try', 'call', 'move', 'live', 'seem', 'feel', 'leave', 'hand', 'high',
            'every', 'right', 'still', 'old', 'great', 'last', 'long', 'good', 'new',
            'first', 'little', 'own', 'other', 'many', 'where', 'much', 'before', 'here',
            'through', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'than', 'too', 'very'
        }
        
        # Filter meaningful words
        meaningful_words = [
            word.lower() for word in words 
            if (word.lower() not in stopwords and 
                len(word) > 2 and 
                word.isalpha() and
                not word.isdigit())
        ]
        
        # Get top keywords by frequency
        word_counts = Counter(meaningful_words)
        keywords = [word for word, count in word_counts.most_common(8)]
        
        # Generate course name from content context (no hardcoded subjects)
        if keywords:
            # Use most frequent keyword as basis for course name
            primary_topic = keywords[0].title()
            course = f"{primary_topic} Studies"
        
        # Generate filename from primary keyword
        primary_word = keywords[0] if keywords else f"content_{chunk_index + 1}"
        filename = f"{primary_word}_{chunk_index + 1}.pdf"
        
        return {
            "filename": filename,
            "title": title,
            "author": author,
            "course": course,
            "page_number": page_number,
            "section": section,
            "keywords": keywords
        }

    def _get_content_recommendations(self, query: str, results: List[str]) -> Dict:
        """Generate content recommendations with metadata in JSON format."""
        recommendations = []
        seen_files = set()
        
        for i, chunk in enumerate(results[:self.top_k]):
            # Dynamically extract metadata from chunk content
            metadata = self._extract_metadata_from_chunk(chunk, i)
            
            # Extract keywords that match the query
            query_lower = query.lower()
            query_words = set(query_lower.split())
            matched_keywords = [kw for kw in metadata["keywords"] if any(qw in kw for qw in query_words)]
            
            # If no query-specific keywords found, use general keywords from content
            if not matched_keywords:
                matched_keywords = metadata["keywords"][:5]  # Top 5 keywords
            
            file_key = f"{metadata['filename']}_page_{metadata['page_number']}"
            if file_key not in seen_files:
                seen_files.add(file_key)
                
                summary = chunk[:200] + "..." if len(chunk) > 200 else chunk.strip()
                
                recommendation = {
                    "filename": metadata["filename"],
                    "title": metadata["title"],
                    "author": metadata["author"],
                    "course": metadata["course"],
                    "page_number": metadata["page_number"],
                    "section": metadata["section"],
                    "summary": summary,
                    "relevance_score": round((self.top_k - i) / self.top_k, 2),
                    "keywords": matched_keywords if matched_keywords else None
                }
                recommendations.append(recommendation)
        
        return {
            "recommendations": recommendations,
            "total_results": len(recommendations)
        }

    def _format_recommendations(self, query: str, recommendations: List[Dict]) -> str:
        """Format recommendations into a readable response."""
        if not recommendations:
            return f"I couldn't find any educational content related to '{query}'. Try searching with different keywords."
        
        response = f"📚 **Content Recommendations for: '{query}'**\n\n"
        response += f"Found {len(recommendations)} relevant PDF(s) with content matching your search:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            response += f"**{i}. {rec['title']}**\n"
            response += f"   📄 File: {rec['filename']}\n"
            response += f"   👨‍🏫 Author: {rec['author']}\n"
            response += f"   📖 Course: {rec['course']}\n"
            response += f"   📄 Page: {rec['page_number']}\n"
            response += f"   📑 Section: {rec['section']}\n"
            response += f"   ⭐ Relevance: {rec['relevance_score']}\n"
            response += f"   📝 Summary: {rec['summary']}\n\n"
        
        response += "💡 **Tip**: You can ask specific questions about any of these PDFs for more detailed information."
        
        return response

    def _format_recommendations_text(self, response: Dict) -> str:
        """Format JSON response into readable text for backward compatibility."""
        if response["status"] != "success" or response["total_results"] == 0:
            return response.get("error_message", "No content found.")
        
        text_response = f"📚 **Content Recommendations for: '{response['query']}'**\n\n"
        text_response += f"Found {response['total_results']} relevant PDF(s) with content matching your search:\n\n"
        
        for i, rec in enumerate(response["recommendations"], 1):
            text_response += f"**{i}. {rec['title']}**\n"
            text_response += f"   📄 File: {rec['filename']}\n"
            text_response += f"   👨‍🏫 Author: {rec['author']}\n"
            text_response += f"   📖 Course: {rec['course']}\n"
            text_response += f"   📄 Page: {rec['page_number']}\n"
            text_response += f"   📑 Section: {rec['section']}\n"
            text_response += f"   ⭐ Relevance: {rec['relevance_score']}\n"
            if rec.get('keywords'):
                text_response += f"   🏷️ Keywords: {', '.join(rec['keywords'])}\n"
            text_response += f"   📝 Summary: {rec['summary']}\n\n"
        
        text_response += f"💡 **Tip**: {response['message']}\n"
        text_response += f"⏱️ Processing time: {response['processing_time_ms']}ms"
        
        return text_response

    def run(self, query: str):
        """Find and recommend educational content based on the query."""
        start_time = time.time()
        
        try:
            bm25, documents = self._initialize_bm25()

            print(f"Tool ({self.name}): Creating embedding for query: '{query}'")
            query_embedding = self.embedding_model.create_embedding(query)
            
            print(f"Tool ({self.name}): Searching educational content...")
            vector_results = self.db.search(query_embedding, top_k=self.top_k)

            # Get relevant context chunks
            context_chunks = vector_results
            if bm25 and documents:
                print(f"Tool ({self.name}): Performing keyword search...")
                tokenized_query = query.split(" ")
                bm25_results = bm25.get_top_n(tokenized_query, documents, n=self.top_k)
                
                # Combine and deduplicate results
                combined_results = list(set(vector_results + bm25_results))
                context_chunks = combined_results

            if not context_chunks:
                error_response = {
                    "query": query,
                    "status": "error",
                    "error_code": "NO_CONTENT_FOUND",
                    "error_message": f"No educational content found related to '{query}'. Try searching with different keywords.",
                    "total_results": 0,
                    "recommendations": []
                }
                if self.return_json:
                    return json.dumps(error_response, indent=2)
                else:
                    return error_response["error_message"]

            # Generate content recommendations
            print(f"Tool ({self.name}): Generating content recommendations...")
            recommendations_data = self._get_content_recommendations(query, context_chunks)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Create structured response
            response = {
                "query": query,
                "status": "success",
                "total_results": recommendations_data["total_results"],
                "recommendations": recommendations_data["recommendations"],
                "message": f"Found {recommendations_data['total_results']} relevant content(s) matching your search. You can ask specific questions about any of these topics for more details.",
                "processing_time_ms": round(processing_time, 2)
            }
            
            if self.return_json:
                return json.dumps(response, indent=2)
            else:
                # Return formatted text for backward compatibility
                return self._format_recommendations_text(response)
                
        except Exception as e:
            error_response = {
                "query": query,
                "status": "error",
                "error_code": "PROCESSING_ERROR",
                "error_message": f"Error processing query: {str(e)}",
                "total_results": 0,
                "recommendations": []
            }
            if self.return_json:
                return json.dumps(error_response, indent=2)
            else:
                return error_response["error_message"]
