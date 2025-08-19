from pypdf import PdfReader
import os
import json

class PDFParser:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.course_metadata = self._load_course_metadata()

    def _load_course_metadata(self) -> dict:
        """Load course metadata from metadata.json"""
        try:
            # Get the path relative to the project root
            current_dir = os.path.dirname(os.path.dirname(__file__))
            metadata_path = os.path.join(current_dir, 'data', 'educational_content', 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"⚠️ Course metadata file not found at {metadata_path}")
                return {}
        except Exception as e:
            print(f"⚠️ Failed to load course metadata: {e}")
            return {}

    def _get_course_metadata_for_file(self, filename: str) -> dict:
        """Get course metadata for a specific file"""
        # Try exact match first
        if filename in self.course_metadata:
            return self.course_metadata[filename]
        
        # Try case-insensitive match
        for key, value in self.course_metadata.items():
            if key.lower() == filename.lower():
                return value
            
        # Try partial match (without extension)
        base_filename = filename.replace('.pdf', '')
        for key, value in self.course_metadata.items():
            if key.replace('.pdf', '').lower() == base_filename.lower():
                return value
        
        # Try contains match (for truncated filenames)
        for key, value in self.course_metadata.items():
            if filename.lower() in key.lower() or key.lower() in filename.lower():
                return value
                
        return {}

    def parse_pdf(self, pdf_path: str) -> tuple[list[str], list[dict]]:
        """Parse PDF and return list of text chunks with proper page metadata and course information."""
        filename = os.path.basename(pdf_path)
        chunks = []
        metadata = []
        
        # Get course metadata for this file
        course_meta = self._get_course_metadata_for_file(filename)
        
        # Create enriched content prefix for better searchability
        enriched_prefix = ""
        if course_meta:
            course_info = []
            if course_meta.get('course_title'):
                course_info.append(f"Course: {course_meta['course_title']}")
            if course_meta.get('subject'):
                course_info.append(f"Subject: {course_meta['subject']}")
            if course_meta.get('author'):
                course_info.append(f"Author: {course_meta['author']}")
            if course_meta.get('grade'):
                course_info.append(f"Grade: {course_meta['grade']}")
            if course_meta.get('topics'):
                topics = ', '.join(course_meta['topics'][:3])  # Include top 3 topics
                course_info.append(f"Topics: {topics}")
            
            if course_info:
                enriched_prefix = " | ".join(course_info) + " | Content: "
        
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            print(f"   📖 PDF has {total_pages} pages")
            if course_meta:
                print(f"   📚 Course: {course_meta.get('course_title', 'Unknown')}")
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    # Extract text from current page
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        # Clean up the text
                        cleaned_text = self._clean_text(page_text)
                        
                        if len(cleaned_text) > 50:  # Only process pages with substantial content
                            # Split page into chunks if it's too long
                            page_chunks = self._chunk_page_text(cleaned_text)
                            
                            for chunk_idx, chunk in enumerate(page_chunks):
                                # Enrich chunk with course metadata for better searchability
                                enriched_chunk = enriched_prefix + chunk
                                chunks.append(enriched_chunk)
                                
                                # Include course metadata in chunk metadata
                                chunk_metadata = {
                                    "filename": filename,
                                    "page": page_num,
                                    "chunk_index": chunk_idx,
                                    "total_pages": total_pages,
                                    "source_path": pdf_path,
                                    "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                                }
                                
                                # Add course metadata if available
                                if course_meta:
                                    chunk_metadata.update({
                                        "course_id": course_meta.get('course_id'),
                                        "course_title": course_meta.get('course_title'),
                                        "content_title": course_meta.get('content_title'),
                                        "author": course_meta.get('author'),
                                        "subject": course_meta.get('subject'),
                                        "grade": course_meta.get('grade'),
                                        "board": course_meta.get('board'),
                                        "topics": course_meta.get('topics', [])
                                    })
                                
                                metadata.append(chunk_metadata)
                        
                        # Progress indicator
                        if page_num % 50 == 0:
                            print(f"   📄 Processed {page_num}/{total_pages} pages...")
                            
                except Exception as e:
                    print(f"   ⚠️ Error processing page {page_num}: {str(e)}")
                    continue
            
            print(f"   ✅ Extracted {len(chunks)} chunks from {total_pages} pages")
            return chunks, metadata
            
        except Exception as e:
            print(f"   ❌ Error reading PDF {pdf_path}: {str(e)}")
            return [], []

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\&\%\$\#\@\+\=\<\>\~\`\|\\]', ' ', text)
        
        # Fix common OCR issues
        text = text.replace(' . ', '. ')
        text = text.replace(' , ', ', ')
        text = text.replace(' ? ', '? ')
        text = text.replace(' ! ', '! ')
        
        # Remove lines that are mostly numbers (page numbers, etc.)
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^\d+\s*$', line) and len(line) > 3:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()

    def _chunk_page_text(self, page_text: str) -> list[str]:
        """Split page text into appropriate chunks."""
        # If page is small enough, return as single chunk
        if len(page_text) <= self.chunk_size:
            return [page_text]
        
        chunks = []
        
        # Try to split by paragraphs first
        paragraphs = page_text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If we couldn't split by paragraphs effectively, fall back to character splitting
        if not chunks or (len(chunks) == 1 and len(chunks[0]) > self.chunk_size * 1.5):
            return self._chunk_text_by_chars(page_text)
        
        return chunks

    def _chunk_text_by_chars(self, text: str) -> list[str]:
        """Split text by character count with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > start + self.chunk_size * 0.8:  # If we find a space in the last 20%
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - self.chunk_overlap
            
        return chunks

# Keep the original functions for backward compatibility
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n\n"
    return full_text

def chunk_text(text, max_length):
    paras = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paras:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
