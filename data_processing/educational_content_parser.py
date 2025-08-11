import json
import os
from typing import List, Dict, Tuple

class EducationalContentParser:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_educational_content(self, content_directory: str, metadata_file: str) -> Tuple[List[str], List[Dict]]:
        """Parse educational content with metadata and page numbers."""
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        all_chunks = []
        all_metadata = []
        
        for filename, file_metadata in metadata.items():
            # For demo, we'll use .txt files instead of .pdf
            txt_filename = filename.replace('.pdf', '.txt')
            file_path = os.path.join(content_directory, txt_filename)
            
            if os.path.exists(file_path):
                chunks, chunk_metadata = self._parse_file_with_metadata(file_path, file_metadata)
                all_chunks.extend(chunks)
                all_metadata.extend(chunk_metadata)
        
        return all_chunks, all_metadata

    def _parse_file_with_metadata(self, file_path: str, file_metadata: Dict) -> Tuple[List[str], List[Dict]]:
        """Parse a single file and attach metadata to each chunk."""
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
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
                    chunks.append(chunk)
                    
                    # Create metadata for this chunk
                    chunk_meta = file_metadata.copy()
                    chunk_meta.update({
                        'filename': os.path.basename(file_path).replace('.txt', '.pdf'),
                        'page_number': i + 1,  # Simulate page numbers based on chapters
                        'section': f"Chapter {i + 1}" if i > 0 else "Introduction",
                        'chunk_index': chunk_idx
                    })
                    chunk_metadata.append(chunk_meta)
        
        return chunks, chunk_metadata

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += self.chunk_size - self.chunk_overlap
            
        return chunks
