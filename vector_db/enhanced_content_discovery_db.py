"""
Enhanced Content Discovery Vector Database
Integrates hybrid educational enhancement with vector storage
"""

from .content_discovery_db import ContentDiscoveryVectorDB
from data_processing.hybrid_enhancer import HybridEducationalEnhancer
from typing import List, Dict, Any
import json

class EnhancedContentDiscoveryVectorDB(ContentDiscoveryVectorDB):
    """
    Enhanced vector database that stores rich educational metadata
    extracted using the hybrid enhancement system
    """
    
    def __init__(self, db_path: str, collection_name: str, top_k: int, use_langextract: bool = False):
        """
        Initialize enhanced content discovery database
        
        Args:
            db_path: Path to the database file
            collection_name: Name of the collection
            top_k: Number of top results to return
            use_langextract: Whether to use LangExtract (requires API key)
        """
        super().__init__(db_path, collection_name, top_k)
        self.enhancer = HybridEducationalEnhancer(use_langextract=use_langextract)
        
    def insert_enhanced(self, data: List[str], embeddings: List[List[float]], 
                       metadata: List[Dict] = None, enhance_content: bool = True):
        """
        Insert content with enhanced educational metadata
        
        Args:
            data: List of text content
            embeddings: List of embeddings
            metadata: List of metadata dictionaries
            enhance_content: Whether to apply educational enhancement
        """
        enhanced_data = []
        enhanced_metadata = []
        
        for i, text_content in enumerate(data):
            current_metadata = metadata[i] if metadata and i < len(metadata) else {}
            
            if enhance_content:
                # Apply educational enhancement
                try:
                    filename = current_metadata.get('filename', f'content_{i}')
                    enhancement_result = self.enhancer.enhance_educational_content(
                        text_content, filename
                    )
                    
                    # Create enhanced text with metadata
                    enhanced_text = self._create_enhanced_text_format(
                        text_content, current_metadata, enhancement_result
                    )
                    
                    # Update metadata with enhancements
                    enhanced_meta = self._merge_metadata(current_metadata, enhancement_result)
                    
                except Exception as e:
                    print(f"⚠️ Enhancement failed for item {i}: {str(e)}")
                    enhanced_text = self._create_basic_enhanced_text(text_content, current_metadata)
                    enhanced_meta = current_metadata
            else:
                enhanced_text = self._create_basic_enhanced_text(text_content, current_metadata)
                enhanced_meta = current_metadata
            
            enhanced_data.append(enhanced_text)
            enhanced_metadata.append(enhanced_meta)
        
        # Insert enhanced data using parent method
        super().insert(enhanced_data, embeddings, enhanced_metadata)
        
        print(f"✅ Inserted {len(enhanced_data)} enhanced content items")
    
    def _create_enhanced_text_format(self, content: str, metadata: Dict, 
                                   enhancement: Any) -> str:
        """
        Create enhanced text format with educational metadata
        
        Args:
            content: Original text content
            metadata: Basic metadata
            enhancement: Educational enhancement result
            
        Returns:
            Enhanced text with embedded metadata
        """
        filename = metadata.get('filename', 'unknown.pdf')
        page = metadata.get('page', 1)
        title = metadata.get('title', 'Unknown Title')
        
        # Extract enhanced information
        learning_objectives = [obj.get('text', '') for obj in enhancement.learning_objectives]
        key_concepts = [concept.get('text', '') for concept in enhancement.key_concepts]
        difficulty = enhancement.difficulty_level
        
        # Create enhanced text format
        enhanced_parts = [
            f"FILENAME:{filename}",
            f"PAGE:{page}",
            f"TITLE:{title}",
            f"DIFFICULTY:{difficulty}"
        ]
        
        if learning_objectives:
            enhanced_parts.append(f"LEARNING_OBJECTIVES:{', '.join(learning_objectives[:3])}")
        
        if key_concepts:
            enhanced_parts.append(f"KEY_CONCEPTS:{', '.join(key_concepts[:5])}")
        
        enhanced_parts.append(f"CONTENT:{content}")
        
        return " ".join(enhanced_parts)
    
    def _create_basic_enhanced_text(self, content: str, metadata: Dict) -> str:
        """Create basic enhanced text format when enhancement fails"""
        filename = metadata.get('filename', 'unknown.pdf')
        page = metadata.get('page', 1)
        
        return f"FILENAME:{filename} PAGE:{page} CONTENT:{content}"
    
    def _merge_metadata(self, original_metadata: Dict, enhancement: Any) -> Dict:
        """
        Merge original metadata with educational enhancements
        
        Args:
            original_metadata: Original metadata dictionary
            enhancement: Educational enhancement result
            
        Returns:
            Merged metadata dictionary
        """
        merged = original_metadata.copy()
        
        # Add educational enhancements
        merged.update({
            'enhanced': True,
            'learning_objectives': [obj.get('text', '') for obj in enhancement.learning_objectives],
            'key_concepts': [concept.get('text', '') for concept in enhancement.key_concepts],
            'difficulty_level': enhancement.difficulty_level,
            'study_questions': [q.get('text', '') for q in enhancement.study_questions],
            'prerequisites': [p.get('text', '') for p in enhancement.prerequisites],
            'examples': [e.get('text', '') for e in enhancement.examples],
            'extraction_method': enhancement.extraction_method,
            'content_sections': len(enhancement.content_sections)
        })
        
        return merged
    
    def search_enhanced(self, query_embedding: List[float], top_k: int = None,
                       difficulty_filter: str = None, concept_filter: str = None) -> List[Dict]:
        """
        Enhanced search with educational filtering options
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            difficulty_filter: Filter by difficulty level (basic, intermediate, advanced)
            concept_filter: Filter by key concepts
            
        Returns:
            List of enhanced search results
        """
        # Get basic search results
        results = self.search(query_embedding, top_k or self.top_k)
        
        enhanced_results = []
        for result in results:
            enhanced_result = result.copy()
            
            # Parse enhanced text to extract educational metadata
            text = result.get('content', '')
            enhanced_metadata = self._parse_enhanced_metadata(text)
            enhanced_result.update(enhanced_metadata)
            
            # Apply filters if specified
            if difficulty_filter and enhanced_metadata.get('difficulty') != difficulty_filter:
                continue
                
            if concept_filter:
                concepts = enhanced_metadata.get('key_concepts', [])
                if not any(concept_filter.lower() in concept.lower() for concept in concepts):
                    continue
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _parse_enhanced_metadata(self, enhanced_text: str) -> Dict:
        """
        Parse enhanced text to extract educational metadata
        
        Args:
            enhanced_text: Enhanced text with embedded metadata
            
        Returns:
            Dictionary with parsed educational metadata
        """
        metadata = {}
        
        # Extract basic information
        if 'FILENAME:' in enhanced_text:
            filename_match = enhanced_text.split('FILENAME:')[1].split()[0]
            metadata['filename'] = filename_match
        
        if 'PAGE:' in enhanced_text:
            try:
                page_match = enhanced_text.split('PAGE:')[1].split()[0]
                metadata['page'] = int(page_match)
            except (IndexError, ValueError):
                metadata['page'] = 1
        
        if 'TITLE:' in enhanced_text:
            try:
                title_part = enhanced_text.split('TITLE:')[1].split('DIFFICULTY:')[0].strip()
                metadata['title'] = title_part
            except IndexError:
                metadata['title'] = 'Unknown'
        
        if 'DIFFICULTY:' in enhanced_text:
            try:
                difficulty_part = enhanced_text.split('DIFFICULTY:')[1].split()[0]
                metadata['difficulty'] = difficulty_part
            except IndexError:
                metadata['difficulty'] = 'unknown'
        
        if 'LEARNING_OBJECTIVES:' in enhanced_text:
            try:
                objectives_part = enhanced_text.split('LEARNING_OBJECTIVES:')[1].split('KEY_CONCEPTS:')[0]
                metadata['learning_objectives'] = [obj.strip() for obj in objectives_part.split(',')]
            except IndexError:
                metadata['learning_objectives'] = []
        
        if 'KEY_CONCEPTS:' in enhanced_text:
            try:
                concepts_part = enhanced_text.split('KEY_CONCEPTS:')[1].split('CONTENT:')[0]
                metadata['key_concepts'] = [concept.strip() for concept in concepts_part.split(',')]
            except IndexError:
                metadata['key_concepts'] = []
        
        if 'CONTENT:' in enhanced_text:
            try:
                content_part = enhanced_text.split('CONTENT:')[1]
                metadata['clean_content'] = content_part.strip()
            except IndexError:
                metadata['clean_content'] = enhanced_text
        
        return metadata

# Factory integration for enhanced database
def create_enhanced_content_discovery_db(db_path: str, collection_name: str, 
                                        top_k: int = 5, use_langextract: bool = False):
    """
    Factory function to create enhanced content discovery database
    
    Args:
        db_path: Path to database file
        collection_name: Name of collection
        top_k: Number of top results
        use_langextract: Whether to use LangExtract (requires API key)
        
    Returns:
        EnhancedContentDiscoveryVectorDB instance
    """
    return EnhancedContentDiscoveryVectorDB(db_path, collection_name, top_k, use_langextract)
