"""
Enhanced Educational Content Parser
Integrates hybrid enhancement with existing educational content parsing
"""

import json
import os
from typing import List, Dict, Tuple
from data_processing.educational_content_parser import EducationalContentParser
from data_processing.hybrid_enhancer import HybridEducationalEnhancer

class EnhancedEducationalContentParser(EducationalContentParser):
    """
    Enhanced educational content parser that applies educational enhancement
    to extracted content using the hybrid enhancement system
    """
    
    def __init__(self, chunk_size: int, chunk_overlap: int, use_langextract: bool = False):
        """
        Initialize enhanced educational content parser
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_langextract: Whether to use LangExtract (requires API key)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.enhancer = HybridEducationalEnhancer(use_langextract=use_langextract)
        
    def parse_educational_content_enhanced(self, content_directory: str, 
                                         metadata_file: str) -> Tuple[List[str], List[Dict]]:
        """
        Parse educational content with enhanced metadata extraction
        
        Args:
            content_directory: Directory containing educational content files
            metadata_file: Path to metadata JSON file
            
        Returns:
            Tuple of (enhanced_chunks, enhanced_metadata)
        """
        print("🔍 Enhanced Educational Content Parsing...")
        
        # Load basic metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        all_chunks = []
        all_metadata = []
        
        for filename, file_metadata in metadata.items():
            print(f"📚 Processing {filename}...")
            
            # For demo, we'll use .txt files instead of .pdf
            txt_filename = filename.replace('.pdf', '.txt')
            file_path = os.path.join(content_directory, txt_filename)
            
            if os.path.exists(file_path):
                # Parse file with enhancement
                chunks, chunk_metadata = self._parse_file_with_enhancement(
                    file_path, file_metadata, filename
                )
                all_chunks.extend(chunks)
                all_metadata.extend(chunk_metadata)
                
                print(f"✅ Processed {filename}: {len(chunks)} enhanced chunks")
            else:
                print(f"⚠️ File not found: {file_path}")
        
        print(f"🎉 Enhanced parsing complete: {len(all_chunks)} total chunks from {len(metadata)} files")
        return all_chunks, all_metadata
    
    def _parse_file_with_enhancement(self, file_path: str, file_metadata: Dict, 
                                   original_filename: str) -> Tuple[List[str], List[Dict]]:
        """
        Parse a single file with educational enhancement
        
        Args:
            file_path: Path to the content file
            file_metadata: Metadata for the file
            original_filename: Original filename (e.g., biology_plants.pdf)
            
        Returns:
            Tuple of (enhanced_chunks, enhanced_metadata)
        """
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply educational enhancement to the full content
        try:
            enhancement_result = self.enhancer.enhance_educational_content(
                content, original_filename
            )
            print(f"📈 Enhanced {original_filename} with {enhancement_result.extraction_method}")
        except Exception as e:
            print(f"⚠️ Enhancement failed for {original_filename}: {str(e)}")
            enhancement_result = None
        
        # Split into sections/chapters for page simulation
        sections = content.split('\n\nChapter')
        chunks = []
        chunk_metadata = []
        
        for i, section in enumerate(sections):
            if i > 0:  # Add "Chapter" back except for first section
                section = "Chapter" + section
            
            # Further chunk the section if it's too long
            section_chunks = self._chunk_text(section)
            
            for chunk_idx, chunk in enumerate(section_chunks):
                if chunk.strip():
                    # Create enhanced chunk metadata
                    enhanced_metadata = self._create_enhanced_chunk_metadata(
                        file_metadata, original_filename, i + 1, 
                        chunk_idx, enhancement_result
                    )
                    
                    # Create enhanced chunk text
                    enhanced_chunk = self._create_enhanced_chunk_text(
                        chunk, enhanced_metadata, enhancement_result
                    )
                    
                    chunks.append(enhanced_chunk)
                    chunk_metadata.append(enhanced_metadata)
        
        return chunks, chunk_metadata
    
    def _create_enhanced_chunk_metadata(self, file_metadata: Dict, filename: str, 
                                      page_num: int, chunk_idx: int, 
                                      enhancement_result) -> Dict:
        """
        Create enhanced metadata for a content chunk
        
        Args:
            file_metadata: Basic file metadata
            filename: Original filename
            page_num: Page number (simulated)
            chunk_idx: Chunk index within page
            enhancement_result: Educational enhancement result
            
        Returns:
            Enhanced metadata dictionary
        """
        # Generate a better title if not provided
        def generate_title_from_filename(filename: str) -> str:
            """Generate a readable title from filename"""
            # Remove extension and replace underscores/hyphens with spaces
            title = os.path.splitext(filename)[0]
            title = title.replace('_', ' ').replace('-', ' ')
            # Convert to title case
            title = title.title()
            return title
        
        # Start with basic metadata
        enhanced_metadata = {
            "filename": filename,
            "page": page_num,
            "chunk_index": chunk_idx,
            "title": file_metadata.get("title") or generate_title_from_filename(filename),
            "author": file_metadata.get("author", "Unknown Author"),
            "course": file_metadata.get("course", "Unknown Course"),
            "subject": file_metadata.get("subject", "Unknown Subject"),
            "grade": file_metadata.get("grade", "Unknown Grade"),
            "medium": file_metadata.get("medium", "English"),
            "board": file_metadata.get("board", "Unknown Board"),
            "topics": file_metadata.get("topics", []),
            "description": file_metadata.get("description", "")
        }
        
        # Add educational enhancements if available
        if enhancement_result:
            enhanced_metadata.update({
                "enhanced": True,
                "enhancement_method": enhancement_result.extraction_method,
                "learning_objectives": [obj.get('text', '') for obj in enhancement_result.learning_objectives],
                "key_concepts": [concept.get('text', '') for concept in enhancement_result.key_concepts],
                "difficulty_level": enhancement_result.difficulty_level,
                "study_questions": [q.get('text', '') for q in enhancement_result.study_questions],
                "prerequisites": [p.get('text', '') for p in enhancement_result.prerequisites],
                "examples": [e.get('text', '') for e in enhancement_result.examples],
                "content_sections_count": len(enhancement_result.content_sections)
            })
        else:
            enhanced_metadata.update({
                "enhanced": False,
                "enhancement_method": "none",
                "learning_objectives": [],
                "key_concepts": [],
                "difficulty_level": "unknown",
                "study_questions": [],
                "prerequisites": [],
                "examples": [],
                "content_sections_count": 0
            })
        
        return enhanced_metadata
    
    def _create_enhanced_chunk_text(self, chunk: str, metadata: Dict, 
                                  enhancement_result) -> str:
        """
        Create enhanced chunk text with embedded educational metadata
        
        Args:
            chunk: Original text chunk
            metadata: Enhanced metadata
            enhancement_result: Educational enhancement result
            
        Returns:
            Enhanced chunk text with embedded metadata
        """
        filename = metadata.get('filename', 'unknown.pdf')
        page = metadata.get('page', 1)
        title = metadata.get('title', os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title())
        course = metadata.get('course', 'Unknown Course')
        difficulty = metadata.get('difficulty_level', 'unknown')
        
        # Build enhanced text format
        enhanced_parts = [
            f"FILENAME:{filename}",
            f"PAGE:{page}",
            f"TITLE:{title}",
            f"COURSE:{course}",
            f"DIFFICULTY:{difficulty}"
        ]
        
        # Add learning objectives if available
        objectives = metadata.get('learning_objectives', [])
        if objectives:
            objectives_text = ', '.join(objectives[:2])  # Limit to 2 objectives
            enhanced_parts.append(f"OBJECTIVES:{objectives_text}")
        
        # Add key concepts if available
        concepts = metadata.get('key_concepts', [])
        if concepts:
            concepts_text = ', '.join(concepts[:3])  # Limit to 3 concepts
            enhanced_parts.append(f"CONCEPTS:{concepts_text}")
        
        # Add the actual content
        enhanced_parts.append(f"CONTENT:{chunk}")
        
        return " ".join(enhanced_parts)

def test_enhanced_parser():
    """Test the enhanced educational content parser"""
    
    print("🧪 Testing Enhanced Educational Content Parser...")
    
    # Use existing educational content
    content_dir = "data/educational_content"
    metadata_file = "data/educational_content/metadata.json"
    
    if not os.path.exists(content_dir) or not os.path.exists(metadata_file):
        print("❌ Educational content directory or metadata file not found")
        print("   Please ensure data/educational_content/ exists with metadata.json")
        return None
    
    try:
        # Create enhanced parser (without LangExtract for testing)
        parser = EnhancedEducationalContentParser(
            chunk_size=1024, 
            chunk_overlap=128, 
            use_langextract=False
        )
        
        # Parse content with enhancements
        chunks, metadata_list = parser.parse_educational_content_enhanced(
            content_dir, metadata_file
        )
        
        print(f"📊 Results:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total metadata entries: {len(metadata_list)}")
        
        if chunks:
            print(f"\\n📝 Sample enhanced chunk:")
            print(f"  Length: {len(chunks[0])}")
            print(f"  Preview: {chunks[0][:200]}...")
            
            print(f"\\n📋 Sample enhanced metadata:")
            sample_meta = metadata_list[0]
            print(f"  Filename: {sample_meta.get('filename')}")
            print(f"  Enhanced: {sample_meta.get('enhanced')}")
            print(f"  Difficulty: {sample_meta.get('difficulty_level')}")
            print(f"  Learning Objectives: {len(sample_meta.get('learning_objectives', []))}")
            print(f"  Key Concepts: {len(sample_meta.get('key_concepts', []))}")
        
        return chunks, metadata_list
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None

if __name__ == "__main__":
    test_enhanced_parser()
